# Training loop for GraphEditTransformer
# Handles: loss computation, training, weight saving/loading

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json

from GraphEditTransformer import GraphHead
from GraphSemanticNetwork import SemanticGraph
from GraphDSL import DSLTokenizer, TransformProgram


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
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # print(f"GraphHeadTrainer initialized on device={device}, lr={learning_rate}")
    
    def program_to_token_tensor(self, program: TransformProgram) -> torch.Tensor:
        # Convert TransformProgram to token tensor
        token_ids = self.tokenizer.program_to_tokens(program)
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)
    
    def compute_loss(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        # TODO: REPLACE THIS ENTIRE METHOD once GraphExecutor and OutputVerifier are implemented
        # Current approach: Cross-entropy loss on token predictions (temporary workaround)
        # Correct approach: 
        #   1. Convert logits → program tokens via greedy decoding
        #   2. Use GraphExecutor to apply program to input grid
        #   3. Use OutputVerifier to compare predicted output grid with ground truth output grid
        #   4. Compute loss from grid difference (e.g., Hamming distance, structural similarity, etc.)
        # This will give us actual supervision signal based on visual correctness, not token sequence matching
        
        # Compute cross-entropy loss between predicted logits and target tokens
        # logits: (batch_size or 1, seq_len, vocab_size)
        # target_tokens: (seq_len,) or shorter
        
        print(f"[DEBUG] logits shape: {logits.shape}, target_tokens shape: {target_tokens.shape}")
        
        # Reshape for cross-entropy
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits_reshaped = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        else:
            logits_reshaped = logits
            seq_len = logits.shape[0]
        
        # Ensure target_tokens is 1D
        if target_tokens.dim() > 1:
            target_tokens = target_tokens.view(-1)
        
        # Pad target tokens to match logits sequence length
        target_seq_len = target_tokens.shape[0]
        if target_seq_len < seq_len:
            # Pad with END token (100) to match logits length
            padding = torch.full((seq_len - target_seq_len,), 100, 
                               dtype=torch.long, device=self.device)
            target_tokens = torch.cat([target_tokens, padding], dim=0)
            print(f"[DEBUG] Padded target from {target_seq_len} to {seq_len}")
        elif target_seq_len > seq_len:
            # Truncate target tokens to match logits length
            target_tokens = target_tokens[:seq_len]
            print(f"[DEBUG] Truncated target from {target_seq_len} to {seq_len}")
        
        print(f"[DEBUG] Final shapes - logits: {logits_reshaped.shape}, target: {target_tokens.shape}")
        print(f"[DEBUG] Target token values (first 10): {target_tokens[:10]}")
        
        # Compute loss
        loss = self.loss_fn(logits_reshaped, target_tokens)
        
        print(f"[DEBUG] Computed loss: {loss.item():.6f}")
        
        return loss
    
    def train_step(self, graph: SemanticGraph, program: TransformProgram) -> float:
        # Single training step
        self.model.train()
        
        # Forward pass
        try:
            logits = self.model.transformer(graph)
        except Exception as e:
            # print(f"Error during forward pass: {e}")
            return float('inf')
        
        # Convert program to tokens
        target_tokens = self.program_to_token_tensor(program)
        
        # Compute loss
        loss = self.compute_loss(logits, target_tokens)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimization step
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, train_data: List[Tuple[SemanticGraph, TransformProgram]]) -> float:
        # Train for one epoch
        # print(f"Training epoch with {len(train_data)} examples")
        
        epoch_loss = 0.0
        num_steps = 0
        
        for graph, program in train_data:
            try:
                loss = self.train_step(graph, program)
                epoch_loss += loss
                num_steps += 1
            except Exception as e:
                # print(f"Error processing example: {e}")
                continue
        
        avg_loss = epoch_loss / max(num_steps, 1)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_data: List[Tuple[SemanticGraph, TransformProgram]]) -> float:
        # Compute validation loss
        self.model.eval()
        val_loss = 0.0
        num_steps = 0
        
        with torch.no_grad():
            for graph, program in val_data:
                try:
                    # Forward pass
                    logits = self.model.transformer(graph)
                    
                    # Convert program to tokens
                    target_tokens = self.program_to_token_tensor(program)
                    
                    # Compute loss
                    loss = self.compute_loss(logits, target_tokens)
                    val_loss += loss.item()
                    num_steps += 1
                except Exception as e:
                    # print(f"Error during validation: {e}")
                    continue
        
        avg_val_loss = val_loss / max(num_steps, 1)
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def fit(self, 
            train_data: List[Tuple[SemanticGraph, TransformProgram]],
            val_data: Optional[List[Tuple[SemanticGraph, TransformProgram]]] = None,
            num_epochs: int = 10) -> Dict[str, List[float]]:
        # Train for multiple epochs
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
