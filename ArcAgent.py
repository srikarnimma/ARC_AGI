import numpy as np
import torch
import time
from typing import List

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet

from GraphObjectExtractor import ObjectExtractor
from GraphSemanticNetwork import GraphBuilder
from GraphDSL import DSLTokenizer
from GraphExecutor import GraphExecutor
from GraphEditTransformer import GraphHead
from GraphTTT import GraphHeadTTT
from OutputVerifier import OutputVerifier, CandidateRanker
from AStarProgramSearch import AStarProgramSearch


class ArcAgent:
    # Main agent that orchestrates the solve pipeline
    # 1. Extract objs from grid
    # 2. Build semantic graph from objs
    # 3. Test-time training on examples
    # 4. Generate program w/ graph head
    # 5. Execute program to produce output
    # 6. Verify against training outputs
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.max_solve_seconds = 20.0
        
        # Procedural components (no neural nets)
        self.extractor = ObjectExtractor()
        self.graph_builder = GraphBuilder()
        self.executor = GraphExecutor()
        self.tokenizer = DSLTokenizer()
        self.astar_search = AStarProgramSearch(
            executor=self.executor,
            verifier=OutputVerifier(),
            max_depth=3,
            max_expansions=100,
            weight=1.5,
            offsets=[-1, 1],
            allow_translate=True,
            allow_copy=False,
        )
        
        # Neural stuff
        self.graph_head = GraphHead(vocab_size=100, hidden_dim=256).to(self.device)
        
        # Test-time training adapter
        self.graph_ttt = GraphHeadTTT(self.graph_head)
        
        # Output verification & ranking
        self.verifier = OutputVerifier()
        self.ranker = CandidateRanker(self.verifier)

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        start_time = time.perf_counter()
        deadline = start_time + self.max_solve_seconds

        def timed_out() -> bool:
            return time.perf_counter() >= deadline

        predictions: list[np.ndarray] = []
        
        # Grab the training & test data
        training_data = arc_problem.training_set()
        training_inputs = [data.get_input_data().data() for data in training_data]
        training_outputs = [data.get_output_data().data() for data in training_data]
        
        test_data = arc_problem.test_set()
        test_input = test_data.get_input_data().data()

        if timed_out():
            return [test_input]
        
        # Pull out objs from test input
        test_objects = self.extractor.extract(test_input)
        # print(f"Test objects: {len(test_objects)}")
        
        # Build graph from the objs
        test_graph = self.graph_builder.build(test_objects)
        # print(f"Test graph built")
        
        # Extract objs from training examples for test-time training
        try:
            if timed_out():
                return [test_input]
            train_objects_list = [self.extractor.extract(inp) for inp in training_inputs]
            # print(f"Extracted objects from {len(train_objects_list)} training examples")
            
            train_graphs = [self.graph_builder.build(objs) for objs in train_objects_list]
            # Fine-tune on training examples (pair each graph w/ expected output)
            train_pairs = list(zip(train_graphs, training_outputs))
            if not timed_out():
                self.graph_ttt.train(train_pairs, train_graphs)
            # print(f"Test-time training done")
        except Exception as e:
            # print(f"Test-time training failed: {e}")
            pass
        
        # A* program search using training pairs
        candidates = []
        try:
            if timed_out():
                return [test_input]
            training_pairs = []
            for inp, out in zip(training_inputs, training_outputs):
                objects = self.extractor.extract(inp)
                graph = self.graph_builder.build(objects)
                training_pairs.append((inp, out, graph))

            remaining_time = max(0.0, deadline - time.perf_counter())
            result = self.astar_search.search(training_pairs, max_time_seconds=remaining_time)
            if result is not None and not timed_out():
                program = result.program
                output = self.executor.execute(test_input, test_graph, program)
                initial_score = 1.0 - min(1.0, result.loss)
                candidates.append((output, initial_score))
        except Exception:
            pass

        # Generate from graph head
        try:
            if timed_out():
                return [test_input]
            program = self.graph_head(test_graph)
            # print(f"Generated program: {program}")
            output = self.executor.execute(test_input, test_graph, program)
            candidates.append((output, 0.8))
            # print(f"Executed program, generated output shape: {output.shape}")
        except Exception as e:
            # print(f"Graph head generation failed: {e}")
            pass
        
        # Fallback: crop to bbox if no good prediction
        if not candidates:
            if timed_out():
                return [test_input]
            nonzero = np.where(test_input != 0)
            if len(nonzero[0]) > 0:
                r_min, r_max = np.min(nonzero[0]), np.max(nonzero[0])
                c_min, c_max = np.min(nonzero[1]), np.max(nonzero[1])
                candidates.append((test_input[r_min:r_max+1, c_min:c_max+1], 0.5))
                # print(f"Using fallback crop")
        
        # Rank candidates against first training output
        if candidates and training_outputs:
            ranked = self.ranker.rank(candidates, training_outputs[0])
            # rank() returns (output_grid, initial_score, similarity_score)
            predictions = [out for out, _, _ in ranked[:3]]
        elif candidates:
            # No training outputs to rank against, just use candidates
            predictions = [out for out, _ in candidates[:3]]
        
        return predictions[:3] if predictions else [test_input]
