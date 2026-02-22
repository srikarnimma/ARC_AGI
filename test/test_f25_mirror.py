#!/usr/bin/env python3
"""
Test if MIRROR_HORIZONTAL works for f25ffba3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from GraphObjectExtractor import ObjectExtractor
from GraphSemanticNetwork import GraphBuilder
from GraphDSL import TransformProgram, Operation, OperationType, Selector
from GraphExecutor import GraphExecutor
from OutputVerifier import OutputVerifier

# Load problem
with open('../Milestones/B/f25ffba3.json') as f:
    data = json.load(f)

inp = np.array(data['train'][0]['input'])
expected_out = np.array(data['train'][0]['output'])

print('Input (10x4):')
print(inp)
print()
print('Expected output:')
print(expected_out)
print()

# Extract and build graph
extractor = ObjectExtractor()
objects = extractor.extract(inp)
builder = GraphBuilder()
graph = builder.build(objects)

# Try MIRROR_HORIZONTAL
program = TransformProgram(operations=[
    Operation(
        type=OperationType.MIRROR_HORIZONTAL,
        selector=Selector.ALL,
        params={}
    )
])

executor = GraphExecutor()
predicted = executor.execute(inp, graph, program)

print('Predicted after MIRROR_HORIZONTAL:')
print(predicted)
print()

# Check loss
verifier = OutputVerifier()
loss = verifier.compute_loss(predicted, expected_out)
print(f'Loss: {loss}')
print(f'Match: {np.array_equal(predicted, expected_out)}')
