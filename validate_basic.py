"""
Basic validation: check that all components connect properly
"""

import numpy as np
import sys

def test_imports():
    """Test all imports work"""
    print("\n1. Testing imports...")
    try:
        from GraphObjectExtractor import ObjectExtractor
        from GraphSemanticNetwork import GraphBuilder
        from GraphDSL import DSLTokenizer
        from GraphExecutor import GraphExecutor
        from GraphEditTransformer import GraphHead
        from GraphTTT import GraphHeadTTT
        from OutputVerifier import OutputVerifier, CandidateRanker
        from ArcAgent import ArcAgent
        print("   [OK] All imports successful")
        return True
    except Exception as e:
        print(f"   [FAIL] Import error: {e}")
        return False


def test_components():
    """Test each component individually"""
    print("\n2. Testing component instantiation...")
    try:
        from GraphObjectExtractor import ObjectExtractor
        from GraphSemanticNetwork import GraphBuilder
        from GraphDSL import DSLTokenizer
        from GraphExecutor import GraphExecutor
        from GraphEditTransformer import GraphHead
        from GraphTTT import GraphHeadTTT
        from OutputVerifier import OutputVerifier, CandidateRanker
        
        extractor = ObjectExtractor()
        print("   [OK] ObjectExtractor")
        
        builder = GraphBuilder()
        print("   [OK] GraphBuilder")
        
        tokenizer = DSLTokenizer()
        print("   [OK] DSLTokenizer")
        
        executor = GraphExecutor()
        print("   [OK] GraphExecutor")
        
        head = GraphHead()
        print("   [OK] GraphHead")
        
        ttt = GraphHeadTTT(head)
        print("   [OK] GraphHeadTTT")
        
        verifier = OutputVerifier()
        print("   [OK] OutputVerifier")
        
        ranker = CandidateRanker(verifier)
        print("   [OK] CandidateRanker")
        
        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline():
    """Test end-to-end pipeline with dummy data"""
    print("\n3. Testing pipeline with dummy data...")
    try:
        from GraphObjectExtractor import ObjectExtractor
        from GraphSemanticNetwork import GraphBuilder
        from GraphDSL import DSLTokenizer
        from GraphExecutor import GraphExecutor
        from GraphEditTransformer import GraphHead
        
        # Create dummy grid
        grid = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [2, 2, 0, 0]
        ], dtype=np.uint8)
        
        print("   [OK] Created dummy grid (4x4)")
        
        # Extract objects
        extractor = ObjectExtractor()
        objects = extractor.extract(grid)
        print(f"   [OK] ObjectExtractor.extract() -> {type(objects)}")
        assert isinstance(objects, dict), f"Expected dict, got {type(objects)}"
        
        # Build graph
        builder = GraphBuilder()
        graph = builder.build(objects)
        print(f"   [OK] GraphBuilder.build() -> {type(graph).__name__}")
        
        # Generate program
        head = GraphHead()
        program = head(graph)
        print(f"   [OK] GraphHead.forward() -> {type(program).__name__}")
        
        # Execute program
        executor = GraphExecutor()
        output = executor.execute(graph, program)
        print(f"   [OK] GraphExecutor.execute() -> {type(output).__name__} shape {output.shape}")
        assert isinstance(output, np.ndarray), f"Expected ndarray, got {type(output)}"
        
        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent():
    """Test ArcAgent initialization"""
    print("\n4. Testing ArcAgent instantiation...")
    try:
        from ArcAgent import ArcAgent
        agent = ArcAgent()
        print("   [OK] ArcAgent created")
        
        # Check all components exist
        assert hasattr(agent, 'extractor'), "Missing extractor"
        assert hasattr(agent, 'graph_builder'), "Missing graph_builder"
        assert hasattr(agent, 'executor'), "Missing executor"
        assert hasattr(agent, 'graph_head'), "Missing graph_head"
        assert hasattr(agent, 'graph_ttt'), "Missing graph_ttt"
        assert hasattr(agent, 'verifier'), "Missing verifier"
        assert hasattr(agent, 'ranker'), "Missing ranker"
        print("   [OK] All components present")
        
        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    results = []
    results.append(test_imports())
    results.append(test_components())
    results.append(test_pipeline())
    results.append(test_agent())
    
    print("\n" + "="*50)
    if all(results):
        print("SUCCESS: All validation tests passed!")
        sys.exit(0)
    else:
        print("FAIL: Some tests failed")
        sys.exit(1)
