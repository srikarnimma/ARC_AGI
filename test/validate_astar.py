#!/usr/bin/env python3
"""Tests for A* program search on basic grids."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json

from GraphObjectExtractor import ObjectExtractor
from GraphSemanticNetwork import GraphBuilder
from GraphExecutor import GraphExecutor
from OutputVerifier import OutputVerifier
from AStarProgramSearch import AStarProgramSearch


def _build_pair(input_grid: np.ndarray, output_grid: np.ndarray):
    extractor = ObjectExtractor()
    builder = GraphBuilder()
    objects = extractor.extract(input_grid)
    graph = builder.build(objects)
    return (input_grid, output_grid, graph)


def test_astar_recolor():
    print("\n[TEST 1] A* - Recolor")

    input_grid_1 = np.zeros((4, 4), dtype=np.uint8)
    input_grid_1[1:3, 1:3] = 1
    output_grid_1 = np.zeros((4, 4), dtype=np.uint8)
    output_grid_1[1:3, 1:3] = 2

    input_grid_2 = np.zeros((4, 4), dtype=np.uint8)
    input_grid_2[0:2, 0:2] = 1
    output_grid_2 = np.zeros((4, 4), dtype=np.uint8)
    output_grid_2[0:2, 0:2] = 2

    input_grid_3 = np.zeros((4, 4), dtype=np.uint8)
    input_grid_3[2:4, 2:4] = 1
    output_grid_3 = np.zeros((4, 4), dtype=np.uint8)
    output_grid_3[2:4, 2:4] = 2

    training_pairs = [
        _build_pair(input_grid_1, output_grid_1),
        _build_pair(input_grid_2, output_grid_2),
        _build_pair(input_grid_3, output_grid_3),
    ]

    search = AStarProgramSearch(
        GraphExecutor(),
        OutputVerifier(),
        max_depth=1,
        max_expansions=200,
        debug=True,
        debug_every=1,
    )
    result = search.search(training_pairs)

    assert result is not None, "Expected a result"
    print(f"  Result program: {result.program}")
    print(f"  Result loss: {result.loss:.4f}")
    assert result.loss == 0.0, f"Expected perfect loss, got {result.loss:.4f}"

    predicted = GraphExecutor().execute(input_grid_1, training_pairs[0][2], result.program)
    assert np.array_equal(predicted, output_grid_1), "Recolor output mismatch"

    test_input = np.zeros((4, 4), dtype=np.uint8)
    test_input[0:2, 2:4] = 1
    test_expected = np.zeros((4, 4), dtype=np.uint8)
    test_expected[0:2, 2:4] = 2
    test_objects = ObjectExtractor().extract(test_input)
    test_graph = GraphBuilder().build(test_objects)
    test_predicted = GraphExecutor().execute(test_input, test_graph, result.program)
    print(f"  Test input:\n{test_input}")
    print(f"  Test expected:\n{test_expected}")
    print(f"  Test predicted:\n{test_predicted}")
    assert np.array_equal(test_predicted, test_expected), "Recolor test output mismatch"

    print("  ✓ Recolor solved with A*")
    return True


def test_astar_delete_inner():
    print("\n[TEST 2] A* - Hollow Inner Color")

    input_grid_1 = np.zeros((5, 5), dtype=np.uint8)
    input_grid_1[1:4, 1:4] = 1
    output_grid_1 = np.zeros((5, 5), dtype=np.uint8)
    output_grid_1[1:4, 1:4] = 1
    output_grid_1[2, 2] = 0

    input_grid_2 = np.zeros((5, 5), dtype=np.uint8)
    input_grid_2[0:3, 0:3] = 1
    output_grid_2 = np.zeros((5, 5), dtype=np.uint8)
    output_grid_2[0:3, 0:3] = 1
    output_grid_2[1, 1] = 0

    input_grid_3 = np.zeros((5, 5), dtype=np.uint8)
    input_grid_3[2:5, 2:5] = 1
    output_grid_3 = np.zeros((5, 5), dtype=np.uint8)
    output_grid_3[2:5, 2:5] = 1
    output_grid_3[3, 3] = 0

    training_pairs = [
        _build_pair(input_grid_1, output_grid_1),
        _build_pair(input_grid_2, output_grid_2),
        _build_pair(input_grid_3, output_grid_3),
    ]

    search = AStarProgramSearch(
        GraphExecutor(),
        OutputVerifier(),
        max_depth=1,
        max_expansions=300,
        debug=True,
        debug_every=1,
    )
    result = search.search(training_pairs)

    assert result is not None, "Expected a result"
    print(f"  Result program: {result.program}")
    print(f"  Result loss: {result.loss:.4f}")
    assert result.loss == 0.0, f"Expected perfect loss, got {result.loss:.4f}"

    predicted = GraphExecutor().execute(input_grid_1, training_pairs[0][2], result.program)
    assert np.array_equal(predicted, output_grid_1), "Delete output mismatch"

    test_input = np.zeros((5, 5), dtype=np.uint8)
    test_input[0:3, 2:5] = 1
    test_expected = np.zeros((5, 5), dtype=np.uint8)
    test_expected[0:3, 2:5] = 1
    test_expected[1, 3] = 0
    test_objects = ObjectExtractor().extract(test_input)
    test_graph = GraphBuilder().build(test_objects)
    test_predicted = GraphExecutor().execute(test_input, test_graph, result.program)
    print(f"  Test input:\n{test_input}")
    print(f"  Test expected:\n{test_expected}")
    print(f"  Test predicted:\n{test_predicted}")
    assert np.array_equal(test_predicted, test_expected), "Hollow test output mismatch"

    print("  ✓ Hollow inner color solved with A*")
    return True


def test_astar_logical_and_split():
    print("\n[TEST 3] A* - Logical AND Split (0520fde7 example)")

    input_grid_1 = np.array([
        [1, 0, 0, 5, 0, 1, 0],
        [0, 1, 0, 5, 1, 1, 1],
        [1, 0, 0, 5, 0, 0, 0],
    ], dtype=np.uint8)

    output_grid_1 = np.array([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0],
    ], dtype=np.uint8)

    input_grid_2 = np.array([
        [1, 0, 1, 5, 1, 0, 1],
        [0, 1, 0, 5, 1, 0, 1],
        [1, 0, 1, 5, 0, 1, 0],
    ], dtype=np.uint8)

    output_grid_2 = np.array([
        [2, 0, 2],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.uint8)

    input_grid_3 = np.array([
        [0, 1, 0, 5, 0, 1, 0],
        [0, 1, 0, 5, 0, 0, 0],
        [0, 0, 0, 5, 0, 0, 0],
    ], dtype=np.uint8)

    output_grid_3 = np.array([
        [0, 2, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.uint8)

    training_pairs = [
        _build_pair(input_grid_1, output_grid_1),
        _build_pair(input_grid_2, output_grid_2),
        _build_pair(input_grid_3, output_grid_3),
    ]

    print("  Input/Output Pairs:")
    print(f"  Pair 1 input:\n{input_grid_1}")
    print(f"  Pair 1 output:\n{output_grid_1}")
    print(f"  Pair 2 input:\n{input_grid_2}")
    print(f"  Pair 2 output:\n{output_grid_2}")
    print(f"  Pair 3 input:\n{input_grid_3}")
    print(f"  Pair 3 output:\n{output_grid_3}")

    search = AStarProgramSearch(
        GraphExecutor(),
        OutputVerifier(),
        max_depth=1,
        max_expansions=300,
        debug=True,
        debug_every=1,
    )
    result = search.search(training_pairs)

    assert result is not None, "Expected a result"
    print(f"  Result program: {result.program}")
    print(f"  Result loss: {result.loss:.4f}")
    assert result.loss == 0.0, f"Expected perfect loss, got {result.loss:.4f}"

    predicted = GraphExecutor().execute(input_grid_1, training_pairs[0][2], result.program)
    print(f"  Predicted output (pair 1):\n{predicted}")
    assert np.array_equal(predicted, output_grid_1), "AND split output mismatch"

    test_input = np.array([
        [1, 0, 0, 5, 0, 1, 0],
        [0, 1, 0, 5, 0, 1, 0],
        [1, 0, 0, 5, 0, 0, 0],
    ], dtype=np.uint8)
    test_expected = np.array([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0],
    ], dtype=np.uint8)
    test_objects = ObjectExtractor().extract(test_input)
    test_graph = GraphBuilder().build(test_objects)
    test_predicted = GraphExecutor().execute(test_input, test_graph, result.program)
    print(f"  Test input:\n{test_input}")
    print(f"  Test expected:\n{test_expected}")
    print(f"  Test predicted:\n{test_predicted}")
    assert np.array_equal(test_predicted, test_expected), "AND split test output mismatch"

    print("  ✓ AND split solved with A*")
    return True


def test_astar_spiral_fill_28e73c20():
    print("\n[TEST 4] A* - Spiral Fill (28e73c20)")

    project_root = os.path.join(os.path.dirname(__file__), '..')
    problem_path = os.path.join(project_root, 'Milestones', 'B', '28e73c20.json')

    with open(problem_path, 'r', encoding='utf-8') as handle:
        problem = json.load(handle)

    training_pairs = []
    for example in problem['train']:
        input_grid = np.array(example['input'], dtype=np.uint8)
        output_grid = np.array(example['output'], dtype=np.uint8)
        training_pairs.append(_build_pair(input_grid, output_grid))

    search = AStarProgramSearch(
        GraphExecutor(),
        OutputVerifier(),
        max_depth=1,
        max_expansions=400,
        debug=True,
        debug_every=1,
    )
    result = search.search(training_pairs)

    assert result is not None, "Expected a result"
    print(f"  Result program: {result.program}")
    print(f"  Result loss: {result.loss:.4f}")
    assert result.loss == 0.0, f"Expected perfect loss, got {result.loss:.4f}"

    test_input = np.array(problem['test'][0]['input'], dtype=np.uint8)
    test_expected = np.array(problem['test'][0]['output'], dtype=np.uint8)
    test_objects = ObjectExtractor().extract(test_input)
    test_graph = GraphBuilder().build(test_objects)
    test_predicted = GraphExecutor().execute(test_input, test_graph, result.program)

    print(f"  Test predicted shape: {test_predicted.shape}")
    assert np.array_equal(test_predicted, test_expected), "Spiral fill test output mismatch"

    print("  ✓ Spiral fill solved with A*")
    return True


def main():
    results = []
    results.append(test_astar_recolor())
    results.append(test_astar_delete_inner())
    results.append(test_astar_logical_and_split())
    results.append(test_astar_spiral_fill_28e73c20())

    print("\n" + "=" * 50)
    if all(results):
        print("SUCCESS: A* tests passed!")
        sys.exit(0)

    print("FAIL: Some A* tests failed")
    sys.exit(1)


if __name__ == "__main__":
    main()
