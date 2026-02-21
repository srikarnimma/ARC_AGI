"""
Test-time training: fine-tune models on task-specific examples.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class GraphHeadTTT:
    """Fine-tune graph head on training pairs."""
    
    def __init__(self, graph_head: nn.Module, learning_rate: float = 1e-3, num_steps: int = 20):
        self.model = graph_head
        self.lr = learning_rate
        self.steps = num_steps
    
    def train(self, train_pairs: List[Tuple], graphs: List):
        # TODO: adapt graph_head on training pairs
        return None


class GridHeadTTT:
    """Fine-tune grid head on training pairs."""
    
    def __init__(self, grid_head: nn.Module, learning_rate: float = 5e-4, num_steps: int = 10):
        self.model = grid_head
        self.lr = learning_rate
        self.steps = num_steps
    
    def train(self, train_pairs: List[Tuple]):
        # TODO: adapt grid_head on training pairs
        pass
