"""
Graph executor: applies transformation programs to semantic graphs.
Input: SemanticGraph + TransformProgram -> Output: numpy grid
"""

import numpy as np
from GraphSemanticNetwork import SemanticGraph
from GraphDSL import TransformProgram


class GraphExecutor:
    """Input: (graph, program) -> Output: output_grid (numpy array)"""
    
    def execute(self, graph: SemanticGraph, program: TransformProgram) -> np.ndarray:
        # TODO
        return np.zeros((5, 5), dtype=np.uint8)
