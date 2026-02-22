#!/usr/bin/env python3
"""
Debug action space generation for f25ffba3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from GraphObjectExtractor import ObjectExtractor
from GraphSemanticNetwork import GraphBuilder
from GraphExecutor import GraphExecutor
from OutputVerifier import OutputVerifier
from AStarProgramSearch import AStarProgramSearch

# Load problem
with open('../Milestones/B/f25ffba3.json') as f:
    data = json.load(f)

# Extract training pairs
train_pairs = []
for pair in data['train']:
    inp = np.array(pair['input'])
    out = np.array(pair['output'])
    extractor = ObjectExtractor()
    objects = extractor.extract(inp)
    builder = GraphBuilder()
    graph = builder.build(objects)
    train_pairs.append((inp, out, graph))

print("Training pairs:")
for i, (inp, out, graph) in enumerate(train_pairs):
    print(f"  Pair {i}: input shape {inp.shape}, output shape {out.shape}")
    print(f"    Objects extracted: {len(graph.nodes)}")

# Build action space
print("\nBuilding action space...")
executor = GraphExecutor()
verifier = OutputVerifier()
searcher = AStarProgramSearch(executor, verifier)
actions = searcher._build_action_space(train_pairs)

print(f"\nTotal actions: {len(actions)}")

# Group actions by type
from GraphDSL import OperationType
action_types = {}
for action in actions:
    op_type = action.type.name
    if op_type not in action_types:
        action_types[op_type] = 0
    action_types[op_type] += 1

print("\nActions by type:")
for op_type, count in sorted(action_types.items()):
    print(f"  {op_type}: {count}")

# Check if mirror operations are present
mirror_actions = [a for a in actions if 'MIRROR' in a.type.name]
print(f"\nMirror operations found: {len(mirror_actions)}")
for action in mirror_actions:
    print(f"  {action.type.name}: {action.params}")
