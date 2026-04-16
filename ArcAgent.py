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

    def _debug_print_grid(self, label: str, grid: np.ndarray) -> None:
        print(label)
        print(grid)

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        start_time = time.perf_counter()
        deadline = start_time + self.max_solve_seconds

        def timed_out() -> bool:
            return time.perf_counter() >= deadline

        predictions: list[np.ndarray] = []
        candidates: list[tuple[np.ndarray, float]] = []
        astar_program = None

        def format_program_ops(program) -> str:
            if program is None or not getattr(program, "operations", None):
                return "<none>"
            ops = []
            for op in program.operations:
                params_items = ", ".join(f"{k}={v}" for k, v in sorted(op.params.items()))
                if params_items:
                    ops.append(f"{op.type.name}({op.selector.name}; {params_items})")
                else:
                    ops.append(f"{op.type.name}({op.selector.name})")
            return " -> ".join(ops)

        def finalize_predictions(timeout_triggered: bool = False) -> list[np.ndarray]:
            if candidates and training_outputs:
                ranked = self.ranker.rank(candidates, training_outputs[0])
                final_predictions = [out for out, _, _ in ranked[:3]]
                for index, (out, initial_score, similarity_score) in enumerate(ranked[:3], start=1):
                    self._debug_print_grid(
                        f"[ArcAgent] Final prediction {index} (initial={initial_score:.4f}, similarity={similarity_score:.4f}):",
                        out,
                    )
                if timeout_triggered:
                    print(f"[ArcAgent] Timeout reached. Selected A* ops: {format_program_ops(astar_program)}")
                    self._debug_print_grid("[ArcAgent] Timeout final output:", final_predictions[0])
                return final_predictions
            if candidates:
                final_predictions = [out for out, _ in candidates[:3]]
                for index, out in enumerate(final_predictions, start=1):
                    self._debug_print_grid(f"[ArcAgent] Final prediction {index}:", out)
                if timeout_triggered:
                    print(f"[ArcAgent] Timeout reached. Selected A* ops: {format_program_ops(astar_program)}")
                    self._debug_print_grid("[ArcAgent] Timeout final output:", final_predictions[0])
                return final_predictions
            if timeout_triggered:
                print(f"[ArcAgent] Timeout reached. Selected A* ops: {format_program_ops(astar_program)}")
                self._debug_print_grid("[ArcAgent] Timeout final output:", test_input)
            return [test_input]
        
        # Grab the training & test data
        training_data = arc_problem.training_set()
        training_inputs = [data.get_input_data().data() for data in training_data]
        training_outputs = [data.get_output_data().data() for data in training_data]
        
        test_data = arc_problem.test_set()
        test_input = test_data.get_input_data().data()

        print("------")
        print(f"[ArcAgent] Problem: {arc_problem.problem_name()}")
        self._debug_print_grid("[ArcAgent] Test input grid:", test_input)

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
                return finalize_predictions(timeout_triggered=True)
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
        try:
            if timed_out():
                return finalize_predictions(timeout_triggered=True)
            training_pairs = []
            for inp, out in zip(training_inputs, training_outputs):
                objects = self.extractor.extract(inp)
                graph = self.graph_builder.build(objects)
                training_pairs.append((inp, out, graph))

            remaining_time = max(0.0, deadline - time.perf_counter())
            result = self.astar_search.search(training_pairs, max_time_seconds=remaining_time)
            if result is not None:
                program = result.program
                astar_program = program
                output = self.executor.execute(test_input, test_graph, program)
                initial_score = 1.0 - min(1.0, result.loss)
                candidates.append((output, initial_score))
                self._debug_print_grid(
                    f"[ArcAgent] Proposed output (A* | loss={result.loss:.4f} | score={initial_score:.4f}):",
                    output,
                )
        except Exception:
            pass

        if timed_out():
            return finalize_predictions(timeout_triggered=True)

        # Generate from graph head
        try:
            if timed_out():
                return finalize_predictions(timeout_triggered=True)
            program = self.graph_head(test_graph)
            # print(f"Generated program: {program}")
            output = self.executor.execute(test_input, test_graph, program)
            candidates.append((output, 0.8))
            self._debug_print_grid("[ArcAgent] Proposed output (GraphHead | score=0.8000):", output)
            # print(f"Executed program, generated output shape: {output.shape}")
        except Exception as e:
            # print(f"Graph head generation failed: {e}")
            pass
        
        # Fallback: crop to bbox if no good prediction
        if not candidates:
            if timed_out():
                return finalize_predictions(timeout_triggered=True)
            nonzero = np.where(test_input != 0)
            if len(nonzero[0]) > 0:
                r_min, r_max = np.min(nonzero[0]), np.max(nonzero[0])
                c_min, c_max = np.min(nonzero[1]), np.max(nonzero[1])
                candidates.append((test_input[r_min:r_max+1, c_min:c_max+1], 0.5))
                self._debug_print_grid("[ArcAgent] Proposed output (Fallback crop | score=0.5000):", candidates[-1][0])
                # print(f"Using fallback crop")

        predictions = finalize_predictions()
        return predictions[:3] if predictions else [test_input]
