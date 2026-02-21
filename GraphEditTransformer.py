#Graph edit transformer: autoregressive model for DSL program generation (SemanticGraph -> DSL tokens)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
from GraphDSL import DSLTokenizer, TransformProgram, Operation, OperationType, Selector
from GraphSemanticNetwork import SemanticGraph, RelationType


class GraphEncoder(nn.Module):
    # Encodes SemanticGraph into node embeddings
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Node embeddings: color + bbox features -> hidden_dim
        self.node_color_embed = nn.Embedding(256, hidden_dim // 2)  # 256 possible colors
        self.node_pos_embed = nn.Linear(4, hidden_dim // 2)  # 4D bbox
        
        # Relation embeddings
        self.relation_embed = nn.Embedding(len(RelationType), hidden_dim)
        
        # Projection to combine color + position
        self.node_combine = nn.Linear(hidden_dim, hidden_dim)
        
        # GNN layers for message passing
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Final projection
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # print(f"GraphEncoder initialized with hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    def forward(self, graph: SemanticGraph) -> Dict[str, Any]:
        # Encode graph into node embeddings
        # Returns: node_embeddings (num_nodes, hidden_dim), node_ids list
        
        if not graph.nodes:
            # Return empty tensors if no nodes
            return {"embeddings": torch.zeros(0, self.hidden_dim), "node_ids": []}
        
        # Create node list ordered by ID for consistency
        node_ids = sorted(graph.nodes.keys())
        num_nodes = len(node_ids)
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        # Initialize node embeddings
        embeddings = []
        for node_id in node_ids:
            node = graph.nodes[node_id]
            
            # Color embedding
            color_emb = self.node_color_embed(torch.tensor(node.color, dtype=torch.long))
            
            # Position embedding (normalized bbox)
            min_r, min_c, max_r, max_c = node.bbox
            bbox_features = torch.tensor([min_r, min_c, max_r, max_c], dtype=torch.float32)
            pos_emb = self.node_pos_embed(bbox_features)
            
            # Combine embeddings
            node_emb = torch.cat([color_emb, pos_emb], dim=-1)  # (hidden_dim,)
            node_emb = self.node_combine(node_emb.unsqueeze(0)).squeeze(0)  # Project to hidden_dim
            embeddings.append(node_emb)
        
        node_embs = torch.stack(embeddings)  # (num_nodes, hidden_dim)
        
        # Message passing with relations
        # print(f"  Encoding {num_nodes} nodes with {len(graph.relations)} relations")
        for layer in self.gnn_layers:
            # Simple aggregation: for each node, aggregate messages from neighbors
            messages = torch.zeros_like(node_embs)
            
            for rel in graph.relations:
                src_idx = node_id_to_idx[rel.source_id]
                tgt_idx = node_id_to_idx[rel.target_id]
                
                # Relation embedding
                rel_emb = self.relation_embed(torch.tensor(rel.type.value.__hash__() % len(RelationType), dtype=torch.long))
                
                # Message: source embedding + relation info
                message = node_embs[src_idx] + rel_emb
                messages[tgt_idx] = messages[tgt_idx] + message
            
            # Update node embeddings
            node_embs = F.relu(layer(node_embs + messages))
        
        # Final projection
        node_embs = self.node_proj(node_embs)
        
        return {
            "embeddings": node_embs,
            "node_ids": node_ids,
            "id_to_idx": node_id_to_idx
        }


class TransformerDecoder(nn.Module):
    # Simplified transformer decoder for token prediction
    
    def __init__(self, vocab_size: int = 102, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Aggregate graph info into a context vector
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Sequence decoder (encodes sequence of tokens + context)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(50, hidden_dim)  # Max sequence length 50
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=min(num_heads, 4),
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # print(f"TransformerDecoder initialized with vocab_size={vocab_size}, hidden_dim={hidden_dim}")
    
    def forward(self, graph_embs: torch.Tensor, seq_length: int = 20) -> torch.Tensor:
        # Predict tokens for a sequence
        # graph_embs: (num_nodes, hidden_dim)
        # Returns: (1, seq_length, vocab_size) logits
        
        device = graph_embs.device
        
        # Aggregate graph embeddings into context
        graph_context = torch.mean(graph_embs, dim=0, keepdim=True)  # (1, hidden_dim)
        graph_context = self.context_proj(graph_context)
        
        # Start with a learned sequence of tokens
        # Initialize with START token and padding
        start_token = torch.full((1, seq_length), 100, dtype=torch.long, device=device)  # END token as placeholder
        
        # Embed tokens
        token_embs = self.token_embed(start_token)  # (1, seq_length, hidden_dim)
        
        # Add positional embeddings
        positions = torch.arange(seq_length, device=device).unsqueeze(0)  # (1, seq_length)
        pos_embs = self.pos_embed(positions)
        token_embs = token_embs + pos_embs
        
        # Create memory from graph context (repeated for sequence length)
        memory = graph_context.expand(1, 1, -1)  # (1, 1, hidden_dim)
        
        # Decode
        decoder_out = self.decoder(token_embs, memory)  # (1, seq_length, hidden_dim)
        
        # Project to vocabulary
        logits = self.output_proj(decoder_out)  # (1, seq_length, vocab_size)
        
        return logits


class GraphEditTransformer(nn.Module):
    """Input: SemanticGraph -> Output: program tokens -> TransformProgram"""
    
    def __init__(self, vocab_size: int = 102, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.tokenizer = DSLTokenizer()
        
        # Components
        self.graph_encoder = GraphEncoder(hidden_dim=hidden_dim, num_layers=num_layers)
        self.decoder = TransformerDecoder(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers)
        
        # print(f"GraphEditTransformer initialized with vocab_size={vocab_size}, hidden_dim={hidden_dim}")
    
    def forward(self, graph: SemanticGraph) -> torch.Tensor:
        # Encode graph
        encoded = self.graph_encoder(graph)
        graph_embs = encoded["embeddings"]  # (num_nodes, hidden_dim)
        
        if graph_embs.shape[0] == 0:
            # Empty graph - return dummy output
            return torch.zeros(1, 1, self.vocab_size)
        
        # Generate tokens
        token_ids = self.decoder(graph_embs)
        
        return token_ids
    
    def tokens_to_program(self, token_ids: List[int]) -> TransformProgram:
        # Convert token sequence back to program
        # Placeholder implementation - simplified token decoding
        # print(f"Converting {len(token_ids)} tokens to program")
        
        operations = []
        i = 0
        while i < len(token_ids):
            token = token_ids[i]
            
            # Check if it's an operation token (0-5)
            if token < 6:
                op_type = list(OperationType)[token]
                
                # Next token should be selector
                if i + 1 < len(token_ids):
                    sel_token = token_ids[i + 1]
                    if 10 <= sel_token < 16:
                        selector = list(Selector)[sel_token - 10]
                        
                        # Collect parameters
                        params = {}
                        j = i + 2
                        while j < len(token_ids) and j - (i + 2) < 4:  # Max 4 params per operation
                            param_token = token_ids[j]
                            if param_token == 100:  # END token
                                break
                            params[f"param_{j}"] = param_token
                            j += 1
                        
                        operations.append(Operation(type=op_type, selector=selector, params=params))
                        i = j
                        continue
            
            i += 1
        
        return TransformProgram(operations=operations)


class GraphHead(nn.Module):
    """Complete graph head: graph -> program."""
    
    def __init__(self, vocab_size: int = 102, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.transformer = GraphEditTransformer(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers)
        self.vocab_size = vocab_size
    
    def forward(self, graph: SemanticGraph) -> TransformProgram:
        # Generate token IDs
        logits = self.transformer(graph)  # (1, seq_len, vocab_size)
        
        # Get predicted tokens by argmax
        if logits.dim() == 3:
            token_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()  # (seq_len,)
        else:
            token_ids = logits.tolist()
        
        program = self.transformer.tokens_to_program(token_ids)
        return program
    
    def get_ttt_parameters(self):
        # Return parameters for test-time training (LoRA adapters)
        # For now, return all transformer parameters
        return list(self.transformer.parameters())
