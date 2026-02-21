# Test GraphDSL and GraphSemanticNetwork

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from GraphObjectExtractor import ObjectExtractor
from GraphSemanticNetwork import GraphBuilder, RelationType
from GraphDSL import DSLTokenizer, Operation, OperationType, Selector, TransformProgram


def print_grid(grid):
    # Helper to print grid nicely
    grid_str = "\n       ".join(str(list(row)) for row in grid)
    return f"\n       {grid_str}"

def format_relations(relations):
    # Group relations by source object and format as a frame
    relations_by_source = {}
    for rel in relations:
        if rel.source_id not in relations_by_source:
            relations_by_source[rel.source_id] = []
        relations_by_source[rel.source_id].append(rel)
    
    # Sort by source object ID
    lines = ["   Relations:"]
    for source_id in sorted(relations_by_source.keys()):
        lines.append(f"       Object {source_id}:")
        for rel in relations_by_source[source_id]:
            lines.append(f"           - {rel.type.value} Object {rel.target_id}")
    
    return "\n".join(lines)

def test_semantic_graph():
    # Test spatial relationship detection
    print("\n1. Testing GraphSemanticNetwork...")
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
        
        # Extract objects
        extractor = ObjectExtractor()
        objects = extractor.extract(grid)
        print(f"   [OK] Extracted {len(objects)} objects (including grid boundary)")
        
        # Build semantic graph
        builder = GraphBuilder()
        graph = builder.build(objects)
        
        print(f"   [OK] Built graph with {len(graph.nodes)} nodes, {len(graph.relations)} relations")
        
        # Print relations organized by object
        print(format_relations(graph.relations))
        
        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dsl_tokenizer():
    # Test DSL tokenization
    print("\n2. Testing GraphDSL tokenization...")
    try:
        # Create a simple program
        program = TransformProgram(operations=[
            Operation(
                type=OperationType.RECOLOR,
                selector=Selector.BY_COLOR,
                params={"color": 2, "new_color": 5}
            ),
            Operation(
                type=OperationType.TRANSLATE,
                selector=Selector.BY_SHAPE,
                params={"offset_r": 1, "offset_c": 2}
            ),
            Operation(
                type=OperationType.ROTATE,
                selector=Selector.ALL,
                params={"angle": 90}
            )
        ])
        
        print(f"   [OK] Created program with {len(program.operations)} operations")
        
        # Tokenize
        tokenizer = DSLTokenizer()
        tokens = tokenizer.program_to_tokens(program)
        
        print(f"   [OK] Tokenized to {len(tokens)} tokens: {tokens[:10]}...")
        
        # Check vocab
        print(f"   [OK] Vocabulary size: {len(tokenizer.vocab)}")
        
        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    # Test end-to-end integration
    print("\n3. Testing integration...")
    try:
        # Create grid with two objects
        grid = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 3, 3, 3],
            [0, 0, 3, 3, 3]
        ], dtype=np.uint8)
        print(f"   Grid:{print_grid(grid)}")
        
        # Extract objects
        extractor = ObjectExtractor()
        objects = extractor.extract(grid)
        
        # Build graph
        builder = GraphBuilder()
        graph = builder.build(objects)
        
        # Create transformation program
        program = TransformProgram(operations=[
            Operation(
                type=OperationType.RECOLOR,
                selector=Selector.BY_COLOR,
                params={"color": 1, "new_color": 4}
            )
        ])
        
        # Tokenize program
        tokenizer = DSLTokenizer()
        tokens = tokenizer.program_to_tokens(program)
        
        print(f"   [OK] Extracted {len(objects)} objects")
        print(f"   [OK] Built graph with {len(graph.nodes)} nodes and {len(graph.relations)} relations")
        print(f"   [OK] Created and tokenized program ({len(tokens)} tokens)")
        
        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_containment():
    # Test containment detection (INSIDE/CONTAINS relations)
    print("\n4. Testing containment detection (INSIDE/CONTAINS)...")
    try:
        # Grid with one object inside another (hollow rectangle)
        grid = np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        print(f"   Grid:{print_grid(grid)}")
        
        # Extract objects
        extractor = ObjectExtractor()
        objects = extractor.extract(grid)
        
        # Build semantic graph
        builder = GraphBuilder()
        graph = builder.build(objects)
        
        # Check for containment relations
        inside_relations = [rel for rel in graph.relations if rel.type == RelationType.INSIDE]
        contains_relations = [rel for rel in graph.relations if rel.type == RelationType.CONTAINS]
        
        if inside_relations and contains_relations:
            print(f"   [OK] Found {len(inside_relations)} INSIDE and {len(contains_relations)} CONTAINS relations")
            all_containment = inside_relations + contains_relations
            print(format_relations(all_containment))
            return True
        else:
            print(f"   [FAIL] Expected INSIDE and CONTAINS relations, got {len(inside_relations)} and {len(contains_relations)}")
            return False
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    results = []
    results.append(test_semantic_graph())
    results.append(test_dsl_tokenizer())
    results.append(test_integration())
    results.append(test_containment())
    
    print("\n" + "="*50)
    if all(results):
        print("SUCCESS: All validation tests passed!")
        sys.exit(0)
    else:
        print("FAIL: Some tests failed")
        sys.exit(1)
