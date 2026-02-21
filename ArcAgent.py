import numpy as np
import torch
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


class ArcAgent:
    def __init__(self):
        self.device = torch.device('cpu')
        
        # Utilities
        self.extractor = ObjectExtractor()
        self.graph_builder = GraphBuilder()
        self.executor = GraphExecutor()
        self.tokenizer = DSLTokenizer()
        
        # Neural models
        self.graph_head = GraphHead(vocab_size=100, hidden_dim=256).to(self.device)
        
        # Test time training
        self.graph_ttt = GraphHeadTTT(self.graph_head)
        
        # Verification
        self.verifier = OutputVerifier()
        self.ranker = CandidateRanker(self.verifier)

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        predictions: list[np.ndarray] = []
        
        # Get data
        training_data = arc_problem.training_set()
        training_inputs = [data.get_input_data().data() for data in training_data]
        training_outputs = [data.get_output_data().data() for data in training_data]
        
        test_data = arc_problem.test_set()
        test_input = test_data.get_input_data().data()
        
        # Extract objects and build graph
        try:
            objects = self.extractor.extract(test_input)
            graph = self.graph_builder.build(objects)
        except:
            graph = None
        
        # Test-time training
        try:
            train_pairs = list(zip(training_inputs, training_outputs))
            train_graphs = [self.graph_builder.build(self.extractor.extract(inp)) for inp in training_inputs]
            self.graph_ttt.train(train_pairs, train_graphs)
        except:
            pass
        
        # Generate from graph head
        candidates = []
        try:
            if graph is not None:
                program = self.graph_head(graph)
                output = self.executor.execute(graph, program)
                candidates.append((output, 0.8))
        except:
            pass
        
        # Fallback
        if not candidates:
            nonzero = np.where(test_input != 0)
            if len(nonzero[0]) > 0:
                r_min, r_max = np.min(nonzero[0]), np.max(nonzero[0])
                c_min, c_max = np.min(nonzero[1]), np.max(nonzero[1])
                candidates.append((test_input[r_min:r_max+1, c_min:c_max+1], 0.5))
        
        # Rank
        if candidates:
            ranked = self.ranker.rank(candidates, training_outputs)
            predictions = [out for out, score in ranked[:3]]
        
        return predictions[:3] if predictions else [test_input]
