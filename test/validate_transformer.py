# Test GraphEditTransformer end-to-end

import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from GraphObjectExtractor import ObjectExtractor
from GraphSemanticNetwork import GraphBuilder
from GraphEditTransformer import GraphEditTransformer, GraphHead


def print_grid(grid):
    # Helper to print grid nicely
    grid_str = "\n       ".join(str(list(row)) for row in grid)
    return f"\n       {grid_str}"


def test_transformer_forward():
    print("\n1. Testing GraphEditTransformer forward pass...")
    try:
        # Create grid with objects
        grid = np.array([
            [0, 1, 1, 0, 2],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 0, 0],
            [3, 3, 0, 0, 0]
        ], dtype=np.uint8)
        print(f"   Grid:{print_grid(grid)}")
        
        # Extract objects and build graph
        extractor = ObjectExtractor()
        objects = extractor.extract(grid)
        builder = GraphBuilder()
        graph = builder.build(objects)
        
        print(f"   [OK] Built semantic graph with {len(graph.nodes)} nodes")
        
        # Create transformer
        transformer = GraphEditTransformer(vocab_size=102, hidden_dim=128, num_layers=2)
        
        # Forward pass
        token_ids = transformer(graph)
        
        print(f"   [OK] Generated {token_ids.shape if isinstance(token_ids, torch.Tensor) else len(token_ids)} tokens")
        print(f"   [OK] Transformer output type: {type(token_ids)}")
        
        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_head():
    print("\n2. Testing GraphHead with program generation...")
    try:
        # Create grid
        grid = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 3, 3, 3],
            [0, 0, 3, 3, 3]
        ], dtype=np.uint8)
        print(f"   Grid:{print_grid(grid)}")
        
        # Extract and build graph
        extractor = ObjectExtractor()
        objects = extractor.extract(grid)
        builder = GraphBuilder()
        graph = builder.build(objects)
        
        # Create graph head
        head = GraphHead(vocab_size=102, hidden_dim=128, num_layers=2)
        
        # Generate program
        program = head(graph)
        
        print(f"   [OK] Generated program with {len(program.operations)} operations")
        for i, op in enumerate(program.operations):
            print(f"       Operation {i}: {op.type.value} + {op.selector.value}")
        
        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_head_parameters():
    print("\n3. Testing GraphHead test-time training parameters...")
    try:
        # Create head
        head = GraphHead(vocab_size=102, hidden_dim=128, num_layers=2)
        
        # Get TTT parameters
        ttt_params = head.get_ttt_parameters()
        
        print(f"   [OK] Found {len(list(ttt_params))} parameter groups for test-time training")
        
        # Count total parameters
        total_params = sum(p.numel() for p in ttt_params)
        print(f"   [OK] Total trainable parameters: {total_params:,}")
        
        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    results = []
    results.append(test_transformer_forward())
    results.append(test_graph_head())
    results.append(test_graph_head_parameters())
    
    print("\n" + "="*50)
    if all(results):
        print("SUCCESS: All transformer tests passed!")
        sys.exit(0)
    else:
        print("FAIL: Some tests failed")
        sys.exit(1)
