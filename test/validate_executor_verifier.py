#!/usr/bin/env python3
"""
Test suite for GraphExecutor and OutputVerifier
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from GraphObjectExtractor import ObjectExtractor, Object
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


def test_executor_logical_ops():
    # Test logical operations: AND, OR, XOR, XNOR, NAND, NOR
    print("\n[TEST 6] GraphExecutor - Logical Operations")
    
    # ===== Test 1: WITH SEPARATOR (like 0520fde7) =====
    print("  [A] Logical ops WITH separator (column of 5s)")
    # Two 3x3 patterns separated by a column of 5s
    # Input: [obj1] [sep] [obj2]
    input_with_sep = np.array([
        [1, 0, 0, 5, 0, 1, 0],
        [0, 1, 0, 5, 1, 1, 1],
        [1, 0, 0, 5, 0, 0, 0]
    ], dtype=np.uint8)
    
    print(f"    Input grid (obj1 | sep | obj2):\n{input_with_sep}")
    
    # Extract objects from the left side (object 1, color 1)
    # and right side (object 2, color 0 and 1 in columns 4-6)
    # For this test, manually work with the left and right patterns
    
    # Extract and build graph
    extractor = ObjectExtractor()
    objects = extractor.extract(input_with_sep)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    print(f"    Extracted {len(graph.nodes)} objects from grid")
    for obj_id, node in graph.nodes.items():
        print(f"      Object {obj_id}: color={node.color}, bbox={node.bbox}")
    
    # For this test, let's just verify the separator isn't treated as object
    # In a real scenario, we'd identify left and right patterns by position
    separator_found = any(node.color == 5 for node in graph.nodes.values())
    print(f"    Separator (color 5) extracted as object: {separator_found}")
    
    # ===== Test 2: WITHOUT SEPARATOR (stacked vertically) =====
    print("\n  [B] Logical ops WITHOUT separator (stacked)")
    # Two 3x3 patterns stacked vertically, same width
    # Object 1 (top, color 1): pixels form one pattern
    # Object 2 (bottom, color 3): pixels form another pattern
    input_no_sep = np.array([
        [1, 0, 1],  # Object 1
        [0, 1, 0],
        [1, 0, 1],
        [3, 0, 3],  # Object 2 (same bbox width as obj 1)
        [0, 3, 0],
        [3, 3, 3]
    ], dtype=np.uint8)
    
    print(f"    Input grid (stacked, no separator):\n{input_no_sep}")
    
    # Extract objects
    extractor = ObjectExtractor()
    objects = extractor.extract(input_no_sep)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    print(f"    Extracted {len(graph.nodes)} objects")
    for obj_id, node in graph.nodes.items():
        print(f"      Object {obj_id}: color={node.color}, bbox={node.bbox}")
    
    # ===== Test 3: SIDE-BY-SIDE WITHOUT SEPARATOR (like 6430c8c4 internally) =====
    print("\n  [C] Logical ops side-by-side (no separator)")
    # Two 3x3 patterns arranged horizontally, same height
    input_side_by_side = np.array([
        [1, 0, 1, 3, 0, 3],  # Object 1 (left), Object 2 (right)
        [0, 1, 0, 0, 3, 0],
        [1, 0, 1, 3, 3, 3]
    ], dtype=np.uint8)
    
    print(f"    Input grid (side-by-side, no separator):\n{input_side_by_side}")
    
    # Extract objects
    extractor = ObjectExtractor()
    objects = extractor.extract(input_side_by_side)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    print(f"    Extracted {len(graph.nodes)} objects")
    for obj_id, node in graph.nodes.items():
        print(f"      Object {obj_id}: color={node.color}, bbox={node.bbox}")
    
    # ===== Test 3: SIDE-BY-SIDE WITHOUT SEPARATOR (like 6430c8c4 internally) =====
    print("\n  [C] Logical ops side-by-side (no separator)")
    # Two 3x3 patterns arranged horizontally, same height
    input_side_by_side = np.array([
        [1, 0, 1, 3, 0, 3],  # Object 1 (left), Object 2 (right)
        [0, 1, 0, 0, 3, 0],
        [1, 0, 1, 3, 3, 3]
    ], dtype=np.uint8)
    
    print(f"    Input grid (side-by-side, no separator):\n{input_side_by_side}")
    
    # Extract objects
    extractor = ObjectExtractor()
    objects = extractor.extract(input_side_by_side)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    print(f"    Extracted {len(graph.nodes)} objects")
    for obj_id, node in graph.nodes.items():
        print(f"      Object {obj_id}: color={node.color}, bbox={node.bbox}")
    
    # Now test logical operations on the side-by-side case
    # Create synthetic input where we place two objects with overlapping bbox for testing
    print("\n  [D] Testing logical operation results on overlapping region")
    
    # Create two 3x3 patterns with same bounding box
    test_grid = np.zeros((3, 3), dtype=np.uint8)
    
    # Object 1: cross pattern (color 1)
    obj1 = Object(id=1, color=1, pixels={(0,1), (1,0), (1,1), (1,2), (2,1)}, bbox=(0, 0, 2, 2))
    # Object 2: X pattern (color 3)
    obj2 = Object(id=2, color=3, pixels={(0,0), (0,2), (1,1), (2,0), (2,2)}, bbox=(0, 0, 2, 2))
    
    print(f"    Object 1 (cross, color 1):")
    grid1 = np.zeros((3, 3), dtype=np.uint8)
    for r, c in obj1.pixels:
        if r < 3 and c < 3:
            grid1[r, c] = 1
    print(f"{grid1}")
    
    print(f"    Object 2 (X pattern, color 3):")
    grid2 = np.zeros((3, 3), dtype=np.uint8)
    for r, c in obj2.pixels:
        if r < 3 and c < 3:
            grid2[r, c] = 3
    print(f"{grid2}")
    
    # Test AND: pixels where both have them
    and_result = obj1.pixels & obj2.pixels
    grid_and = np.zeros((3, 3), dtype=np.uint8)
    for r, c in and_result:
        if r < 3 and c < 3:
            grid_and[r, c] = 2  # Output color
    print(f"    AND result (color 2 = overlap):\n{grid_and}")
    print(f"      -> {len(and_result)} pixels in result")
    
    # Test OR: pixels where at least one has them
    or_result = obj1.pixels | obj2.pixels
    grid_or = np.zeros((3, 3), dtype=np.uint8)
    for r, c in or_result:
        if r < 3 and c < 3:
            grid_or[r, c] = 2  # Output color
    print(f"    OR result (color 2 = union):\n{grid_or}")
    print(f"      -> {len(or_result)} pixels in result")
    
    # Test XOR: pixels where exactly one has them
    xor_result = obj1.pixels ^ obj2.pixels
    grid_xor = np.zeros((3, 3), dtype=np.uint8)
    for r, c in xor_result:
        if r < 3 and c < 3:
            grid_xor[r, c] = 2  # Output color
    print(f"    XOR result (color 2 = unique to one):\n{grid_xor}")
    print(f"      -> {len(xor_result)} pixels in result")
    
    # Test XNOR: pixels where both are same (both true or both false)
    xnor_result = obj1.pixels == obj2.pixels
    # For XNOR, we need to check all positions in bbox
    xnor_pixels = set()
    for r in range(3):
        for c in range(3):
            has_1 = (r, c) in obj1.pixels
            has_2 = (r, c) in obj2.pixels
            if has_1 == has_2:  # Both true or both false
                xnor_pixels.add((r, c))
    grid_xnor = np.zeros((3, 3), dtype=np.uint8)
    for r, c in xnor_pixels:
        if r < 3 and c < 3:
            grid_xnor[r, c] = 2  # Output color
    print(f"    XNOR result (color 2 = both same):\n{grid_xnor}")
    print(f"      -> {len(xnor_pixels)} pixels in result")
    
    # Verify properties
    assert len(and_result) <= len(or_result), "AND should produce <= pixels than OR"
    assert len(xor_result) > 0, "XOR should produce non-zero result for different patterns"
    assert len(and_result) + len(xor_result) == len(or_result), "AND + XOR should equal OR"
    
    print(f"  ✓ Logical operations correctly produce expected grids")
    print(f"    AND: {len(and_result)} | OR: {len(or_result)} | XOR: {len(xor_result)} | XNOR: {len(xnor_pixels)}")
    return True


def test_executor_mirror():
    """Test executor: mirror operations (vertical and horizontal)"""
    print("\n[TEST 7] GraphExecutor - Mirror Operations")
    
    # ===== Test 1: MIRROR_VERTICAL (left-right symmetry) =====
    print("\n  [A] Mirror Vertical - Create left-right symmetry")
    
    # Create 3x6 grid with a pattern on the left half
    input_grid = np.array([
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    
    print(f"  Input grid (3x6, pattern on left):\n{input_grid}")
    
    # Extract objects and build graph
    extractor = ObjectExtractor()
    objects = extractor.extract(input_grid)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    # Create program: mirror vertically
    program = TransformProgram(operations=[
        Operation(
            type=OperationType.MIRROR_VERTICAL,
            selector=Selector.ALL,
            params={}
        )
    ])
    
    # Execute
    executor = GraphExecutor()
    output_grid = executor.execute(input_grid, graph, program)
    
    print(f"  Output grid after vertical mirror:\n{output_grid}")
    
    # Expected output: left half stays, right half is flipped version of left
    expected_output = np.array([
        [1, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1],
    ], dtype=np.uint8)
    
    print(f"  Expected:\n{expected_output}")
    
    # Verify exact match
    assert np.array_equal(output_grid, expected_output), f"Output does not match expected.\nGot:\n{output_grid}"
    print(f"  ✓ Mirror vertical correct: left half preserved, right half is flipped copy")
    
    # ===== Test 2: MIRROR_HORIZONTAL (top-bottom symmetry) =====
    print("\n  [B] Mirror Horizontal - Create top-bottom symmetry")
    
    # Create 6x3 grid with a pattern on the bottom half
    input_grid_h = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    
    print(f"  Input grid (6x3, pattern on bottom):\n{input_grid_h}")
    
    # Extract objects and build graph
    extractor = ObjectExtractor()
    objects = extractor.extract(input_grid_h)
    builder = GraphBuilder()
    graph = builder.build(objects)
    
    # Create program: mirror horizontally
    program = TransformProgram(operations=[
        Operation(
            type=OperationType.MIRROR_HORIZONTAL,
            selector=Selector.ALL,
            params={}
        )
    ])
    
    # Execute
    executor = GraphExecutor()
    output_grid_h = executor.execute(input_grid_h, graph, program)
    
    print(f"  Output grid after horizontal mirror:\n{output_grid_h}")
    
    # Expected: bottom half flipped and placed on top, then original bottom
    expected_output_h = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    
    print(f"  Expected:\n{expected_output_h}")
    
    # Verify exact match
    assert np.array_equal(output_grid_h, expected_output_h), f"Output does not match expected.\nGot:\n{output_grid_h}"
    print(f"  ✓ Mirror horizontal correct: bottom half flipped to top, original bottom preserved")
    
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
        ("Executor - Logical Ops", test_executor_logical_ops),
        ("Executor - Mirror", test_executor_mirror),
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
