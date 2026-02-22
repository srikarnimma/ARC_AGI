#!/usr/bin/env python3
"""
Validate a single example problem: run A* search and show results.
Usage: python3 validate_example.py {problem_id} [--verbose]
Example: python3 validate_example.py 6430c8c4
Example: python3 validate_example.py 6430c8c4 --verbose
"""

import sys
import json
import numpy as np
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GraphObjectExtractor import ObjectExtractor
from GraphSemanticNetwork import GraphBuilder
from AStarProgramSearch import AStarProgramSearch
from GraphExecutor import GraphExecutor
from OutputVerifier import OutputVerifier

parser = argparse.ArgumentParser(description='Validate a single ARC problem')
parser.add_argument('problem_id', help='Problem ID (e.g., 6430c8c4)')
parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose debug output during search')
args = parser.parse_args()

problem_id = args.problem_id
verbose = args.verbose

# Construct path to Milestones directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
problem_path = os.path.join(project_root, 'Milestones', 'B', f'{problem_id}.json')

try:
    with open(problem_path) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Problem {problem_id} not found at {problem_path}")
    sys.exit(1)

# Build training pairs
train_pairs = []
for i, pair in enumerate(data['train']):
    inp = np.array(pair['input'])
    out = np.array(pair['output'])
    
    extractor = ObjectExtractor()
    objects = extractor.extract(inp)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    train_pairs.append((inp, out, graph))

test_input = np.array(data['test'][0]['input'])
test_output = np.array(data['test'][0]['output'])

# Run A* search
print(f"\n{'='*60}")
print(f"Problem: {problem_id}")
print('='*60)
print(f"\nTraining data: {len(train_pairs)} pairs")
for i, (inp, out, _) in enumerate(train_pairs):
    print(f"  Pair {i}: {inp.shape} -> {out.shape}")

executor = GraphExecutor()
verifier = OutputVerifier()

search = AStarProgramSearch(
    executor=executor,
    verifier=verifier,
    max_depth=3,
    max_expansions=500,
    weight=1.5,
    offsets=[-1, 1],
    allow_copy=False,
    debug=verbose,
    debug_every=10
)

result = search.search(train_pairs)

if result is None:
    print(f"\nA* Search Result: No solution found")
else:
    print(f"\nA* Search Result:")
    print(f"  Loss: {result.loss:.4f}")
    print(f"  Expansions: {result.expansions}")
    print(f"  Program length: {len(result.program.operations)}")

    if result.program.operations:
        print(f"\n  Selected Operations:")
        for i, op in enumerate(result.program.operations):
            params_str = ", ".join(f"{k}={v}" for k, v in sorted(op.params.items()))
            print(f"    {i+1}. {op.type.name}({params_str})")
    else:
        print(f"\n  (No operations - using input as output)")

    # Test on test set
    test_extractor = ObjectExtractor()
    test_objects = test_extractor.extract(test_input)
    test_builder = GraphBuilder()
    test_graph = test_builder.build(test_objects)

    try:
        predicted = executor.execute(test_input, test_graph, result.program)
        test_loss = verifier.compute_loss(predicted, test_output)
        
        print(f"\nTest Set Results:")
        print(f"  Predicted shape: {predicted.shape}")
        print(f"  Expected shape: {test_output.shape}")
        print(f"  Test loss: {test_loss:.4f}")
        
        if test_loss == 0.0:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
            if predicted.shape == test_output.shape:
                # Show diff
                diff = np.abs(predicted.astype(int) - test_output.astype(int))
                mismatches = np.sum(diff > 0)
                print(f"  Mismatched pixels: {mismatches}/{predicted.size}")
    except Exception as e:
        print(f"\nTest Execution Error: {e}")
        import traceback
        traceback.print_exc()
