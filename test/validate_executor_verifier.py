#!/usr/bin/env python3
"""
Test suite for GraphExecutor and OutputVerifier
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from GraphObjectExtractor import ObjectExtractor
from GraphSemanticNetwork import GraphBuilder
from GraphDSL import TransformProgram, Operation, OperationType, Selector
from GraphExecutor import GraphExecutor
from OutputVerifier import OutputVerifier


def test_executor_recolor():
    """Test executor: recolor operation"""
    print("\n[TEST 1] GraphExecutor - Recolor Operation")
    
    # Create input: 5x5 grid with 3x3 blue square in center
    input_grid = np.zeros((5, 5), dtype=np.uint8)
    input_grid[1:4, 1:4] = 1  # Blue (1)
    
    print(f"  Input grid:\n{input_grid}")
    
    # Extract objects and build graph
    extractor = ObjectExtractor()
    objects = extractor.extract(input_grid)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    print(f"  Extracted {len(graph.nodes)} objects")
    
    # Create program: recolor blue to red
    program = TransformProgram(operations=[
        Operation(
            type=OperationType.RECOLOR,
            selector=Selector.BY_COLOR,
            params={'color': 1, 'new_color': 2}  # Blue -> Red
        )
    ])
    
    # Execute
    executor = GraphExecutor()
    output_grid = executor.execute(input_grid, graph, program)
    
    print(f"  Output grid:\n{output_grid}")
    
    # Verify: center should be red (2)
    assert output_grid[2, 2] == 2, f"Expected color 2 at center, got {output_grid[2, 2]}"
    assert np.sum(output_grid == 2) == 9, "Expected 9 red pixels"
    
    print(f"  ✓ Recolor successful: 9 pixels changed from blue to red")
    return True


def test_executor_translate():
    """Test executor: translate operation"""
    print("\n[TEST 2] GraphExecutor - Translate Operation")
    
    # Create input: 2x2 object at (0:2, 0:2)
    input_grid = np.zeros((5, 5), dtype=np.uint8)
    input_grid[0:2, 0:2] = 3  # Green
    
    print(f"  Input grid:\n{input_grid}")
    
    # Extract and build graph
    extractor = ObjectExtractor()
    objects = extractor.extract(input_grid)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    # Create program: translate by (2, 2)
    program = TransformProgram(operations=[
        Operation(
            type=OperationType.TRANSLATE,
            selector=Selector.ALL,
            params={'offset_r': 2, 'offset_c': 2}
        )
    ])
    
    # Execute
    executor = GraphExecutor()
    output_grid = executor.execute(input_grid, graph, program)
    
    print(f"  Output grid:\n{output_grid}")
    
    # Verify: object should be at (2:4, 2:4)
    assert output_grid[2, 2] == 3, "Object not at expected position"
    assert np.sum(output_grid == 3) == 4, "Expected 4 pixels"
    
    print(f"  ✓ Translate successful: object moved from (0:2,0:2) to (2:4,2:4)")
    return True


def test_executor_copy():
    """Test executor: copy operation"""
    print("\n[TEST 3] GraphExecutor - Copy Operation")
    
    # Create input: 2x2 object
    input_grid = np.zeros((7, 7), dtype=np.uint8)
    input_grid[0:2, 0:2] = 4  # Orange
    
    print(f"  Input grid:\n{input_grid}")
    
    # Extract and build graph
    extractor = ObjectExtractor()
    objects = extractor.extract(input_grid)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    # Create program: copy all objects, offset by (3, 3)
    program = TransformProgram(operations=[
        Operation(
            type=OperationType.COPY,
            selector=Selector.ALL,
            params={'offset_r': 3, 'offset_c': 3}
        )
    ])
    
    # Execute
    executor = GraphExecutor()
    output_grid = executor.execute(input_grid, graph, program)
    
    print(f"  Output grid:\n{output_grid}")
    
    # Verify: original at (0:2, 0:2) and copy at (3:5, 3:5)
    assert np.sum(output_grid == 4) == 8, "Expected 8 orange pixels (2 copies of 2x2)"
    assert output_grid[0, 0] == 4, "Original not preserved"
    assert output_grid[3, 3] == 4, "Copy not created"
    
    print(f"  ✓ Copy successful: created duplicate at offset")
    return True


def test_verifier_loss():
    """Test OutputVerifier: loss computation"""
    print("\n[TEST 4] OutputVerifier - Loss Computation")
    
    grid_perfect = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=np.uint8)
    
    grid_diff = np.array([[1, 1, 0, 0, 0],
                         [1, 1, 0, 0, 0],
                         [2, 2, 0, 0, 0],
                         [2, 2, 0, 0, 0],
                         [0, 0, 0, 0, 0]], dtype=np.uint8)
    
    verifier = OutputVerifier()
    
    # Perfect match
    loss_perfect = verifier.compute_loss(grid_perfect, grid_perfect)
    print(f"  Loss (perfect match): {loss_perfect:.4f}")
    assert loss_perfect == 0.0, "Perfect match should have 0 loss"
    
    # Partial difference
    loss_partial = verifier.compute_loss(grid_perfect, grid_diff)
    print(f"  Loss (4 pixels differ): {loss_partial:.4f}")
    assert 0 < loss_partial < 1, "Partial difference should have loss between 0 and 1"
    
    # Pixel accuracy
    acc_perfect = verifier.pixel_accuracy(grid_perfect, grid_perfect)
    print(f"  Pixel accuracy (perfect): {acc_perfect:.4f}")
    assert acc_perfect == 1.0, "Perfect match should have accuracy 1.0"
    
    acc_partial = verifier.pixel_accuracy(grid_perfect, grid_diff)
    print(f"  Pixel accuracy (partial): {acc_partial:.4f}")
    assert 0 < acc_partial < 1, "Partial match should have accuracy between 0 and 1"
    
    print(f"  ✓ Loss and accuracy metrics working correctly")
    return True


def test_end_to_end():
    """Test end-to-end pipeline"""
    print("\n[TEST 5] End-to-End: Extract -> Execute -> Verify")
    
    # Input: blue square
    input_grid = np.zeros((5, 5), dtype=np.uint8)
    input_grid[1:4, 1:4] = 1
    
    # Ground truth: red square
    ground_truth = np.zeros((5, 5), dtype=np.uint8)
    ground_truth[1:4, 1:4] = 2
    
    print(f"  Input:\n{input_grid}")
    print(f"  Ground truth:\n{ground_truth}")
    
    # Extract graph
    extractor = ObjectExtractor()
    objects = extractor.extract(input_grid)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    # Create correct program
    program = TransformProgram(operations=[
        Operation(
            type=OperationType.RECOLOR,
            selector=Selector.BY_COLOR,
            params={'color': 1, 'new_color': 2}
        )
    ])
    
    # Execute
    executor = GraphExecutor()
    predicted = executor.execute(input_grid, graph, program)
    
    print(f"  Predicted:\n{predicted}")
    
    # Verify
    verifier = OutputVerifier()
    loss = verifier.compute_loss(predicted, ground_truth)
    accuracy = verifier.pixel_accuracy(predicted, ground_truth)
    
    print(f"  Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    assert loss == 0.0, "Correct program should produce 0 loss"
    assert accuracy == 1.0, "Correct program should have perfect accuracy"
    
    print(f"  ✓ End-to-end pipeline working correctly")
    return True


if __name__ == '__main__':
    print("="*60)
    print("Testing GraphExecutor and OutputVerifier")
    print("="*60)
    
    tests = [
        ("Executor - Recolor", test_executor_recolor),
        ("Executor - Translate", test_executor_translate),
        ("Executor - Copy", test_executor_copy),
        ("Verifier - Loss", test_verifier_loss),
        ("End-to-End", test_end_to_end),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nResult: {passed}/{total} tests passed\n")
    
    sys.exit(0 if passed == total else 1)
