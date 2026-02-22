# Training loop for GraphEditTransformer
# Handles: loss computation, training, weight saving/loading

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import numpy as np

from GraphEditTransformer import GraphHead
from GraphSemanticNetwork import SemanticGraph
from GraphDSL import DSLTokenizer, TransformProgram
from GraphExecutor import GraphExecutor
from OutputVerifier import OutputVerifier


class GraphHeadTrainer:
    # Training wrapper for GraphHead with loss computation and weight management
    
    def __init__(self, model: GraphHead, learning_rate: float = 1e-3, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = DSLTokenizer()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
        # Initialize executor and verifier for grid-based loss computation
        self.executor = GraphExecutor()
        self.verifier = OutputVerifier()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # print(f"GraphHeadTrainer initialized on device={device}, lr={learning_rate}")
    
    def program_to_token_tensor(self, program: TransformProgram) -> torch.Tensor:
        # Convert TransformProgram to token tensor
        token_ids = self.tokenizer.program_to_tokens(program)
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)
    
    def compute_loss(self, logits: torch.Tensor, graph: SemanticGraph, 
                    input_grid: np.ndarray, output_grid: np.ndarray, 
                    target_program: TransformProgram) -> torch.Tensor:
        # Hybrid loss: token-level CE (for gradient) + grid-level loss (for supervision)
        # logits: (batch_size, seq_len, vocab_size) or (seq_len, vocab_size)
        # graph: SemanticGraph for executor
        # input_grid: numpy array of input
        # output_grid: numpy array of expected output (ground truth)
        # target_program: ground truth program for token supervision
        
        # ===== PART 1: Token-level cross-entropy loss (provides gradient flow) =====
        # Convert target program to tokens
        target_tokens = self.program_to_token_tensor(target_program)
        
        # Reshape logits for cross-entropy
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits_reshaped = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        else:
            logits_reshaped = logits
            seq_len = logits.shape[0]
            vocab_size = logits.shape[1]
        
        # Pad or truncate target to match sequence length
        target_seq_len = target_tokens.shape[0]
        if target_seq_len < seq_len:
            # Pad with END token (100)
            padding = torch.full((seq_len - target_seq_len,), 100, 
                               dtype=torch.long, device=self.device)
            target_tokens = torch.cat([target_tokens, padding], dim=0)
        elif target_seq_len > seq_len:
            # Truncate
            target_tokens = target_tokens[:seq_len]
        
        # Compute token-level cross-entropy loss (differentiable!)
        token_loss = self.loss_fn(logits_reshaped, target_tokens)
        # print(f"[Loss] Token CE loss: {token_loss.item():.6f}")
        
        # ===== PART 2: Grid-level loss (provides supervision signal) =====
        try:
            # Greedy decode logits to tokens (non-differentiable, but just for evaluation)
            with torch.no_grad():
                if logits.dim() == 2:
                    token_ids = torch.argmax(logits, dim=1)
                else:
                    token_ids = torch.argmax(logits, dim=-1).squeeze()
                
                # Convert tokens to program
                program = self._tokens_to_program(token_ids)
                
                # Execute program on input grid
                predicted_grid = self.executor.execute(input_grid, graph, program)
                
                # Compute grid-based loss using verifier
                grid_loss_value = self.verifier.compute_loss(predicted_grid, output_grid)
            
            # Convert to tensor (no gradient needed, just for logging/monitoring)
            grid_loss = torch.tensor(grid_loss_value, dtype=torch.float32, device=self.device)
            # print(f"[Loss] Grid loss: {grid_loss.item():.6f}")
            
        except Exception as e:
            # If execution fails, set high grid loss
            grid_loss = torch.tensor(1.0, dtype=torch.float32, device=self.device)
            # print(f"[Loss] Execution failed, using fallback grid loss")
        
        # ===== PART 3: Combine losses =====
        # Token loss provides gradient, grid loss provides feedback
        # Weight token loss higher initially (grid loss just for monitoring)
        alpha = 1.0  # Weight for token loss (provides gradient)
        beta = 0.0   # Weight for grid loss (for monitoring only, since it has no gradient)
        
        # Combined loss (only token_loss contributes to gradient)
        total_loss = alpha * token_loss + beta * grid_loss
        
        # Store grid loss for logging (attach as attribute)
        total_loss.grid_loss = grid_loss.item()
        total_loss.token_loss = token_loss.item()
        
        return total_loss
    
    def _tokens_to_program(self, token_ids: torch.Tensor) -> TransformProgram:
        # Decode token sequence to TransformProgram
        # For now, return empty program (full decoder implementation would go here)
        # This is a simplified version - proper decoding requires DSL grammar knowledge
        return TransformProgram(operations=[])
    
    def train_step(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                   graph: SemanticGraph, program: TransformProgram) -> float:
        # Single training step with hybrid loss (token CE + grid evaluation)
        self.model.train()
        
        # Forward pass
        try:
            logits = self.model.transformer(graph)
        except Exception as e:
            # print(f"Error during forward pass: {e}")
            return float('inf')
        
        # Compute hybrid loss (token CE for gradient + grid loss for monitoring)
        try:
            loss = self.compute_loss(logits, graph, input_grid, output_grid, program)
        except Exception as e:
            # print(f"Error during loss computation: {e}")
            return float('inf')
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimization step
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, train_data: List[Tuple[np.ndarray, np.ndarray, SemanticGraph, TransformProgram]]) -> float:
        # Train for one epoch
        # train_data: List of (input_grid, output_grid, graph, program) tuples
        # print(f"Training epoch with {len(train_data)} examples")
        
        epoch_loss = 0.0
        num_steps = 0
        
        for input_grid, output_grid, graph, program in train_data:
            try:
                loss = self.train_step(input_grid, output_grid, graph, program)
                epoch_loss += loss
                num_steps += 1
            except Exception as e:
                # print(f"Error processing example: {e}")
                continue
        
        avg_loss = epoch_loss / max(num_steps, 1)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_data: List[Tuple[np.ndarray, np.ndarray, SemanticGraph, TransformProgram]]) -> float:
        # Compute validation loss
        # val_data: List of (input_grid, output_grid, graph, program) tuples
        self.model.eval()
        val_loss = 0.0
        num_steps = 0
        
        with torch.no_grad():
            for input_grid, output_grid, graph, program in val_data:
                try:
                    # Forward pass
                    logits = self.model.transformer(graph)
                    
                    # Compute hybrid loss
                    loss = self.compute_loss(logits, graph, input_grid, output_grid, program)
                    val_loss += loss.item()
                    num_steps += 1
                except Exception as e:
                    # print(f"Error during validation: {e}")
                    continue
        
        avg_val_loss = val_loss / max(num_steps, 1)
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def fit(self, 
            train_data: List[Tuple[np.ndarray, np.ndarray, SemanticGraph, TransformProgram]],
            val_data: Optional[List[Tuple[np.ndarray, np.ndarray, SemanticGraph, TransformProgram]]] = None,
            num_epochs: int = 10) -> Dict[str, List[float]]:
        # Train for multiple epochs
        # train_data/val_data: List of (input_grid, output_grid, graph, program) tuples
        # print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_data)
            
            # Validation
            val_loss = None
            if val_data:
                val_loss = self.validate(val_data)
                # print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
            else:
                pass
                # print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {train_loss:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        # Save model weights and training history
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'vocab_size': self.model.transformer.vocab_size,
                'hidden_dim': self.model.transformer.hidden_dim
            }
        }
        
        torch.save(checkpoint, filepath)
        # print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        # Load model weights and training history
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        # print(f"Checkpoint loaded from {filepath}")
    
    def save_weights(self, filepath: str) -> None:
        # Save only model weights (for inference)
        torch.save(self.model.state_dict(), filepath)
        # print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath: str) -> None:
        # Load model weights
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        # print(f"Weights loaded from {filepath}")
    
    def inference(self, graph: SemanticGraph) -> TransformProgram:
        # Generate program for a graph (inference mode)
        self.model.eval()
        
        with torch.no_grad():
            program = self.model(graph)
        
        return program
    
    def get_loss_history(self) -> Dict[str, List[float]]:
        # Return training history
        return {
            "train": self.train_losses,
            "validation": self.val_losses
        }
