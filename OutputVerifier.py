"""
Output verification and ranking.
"""

import numpy as np
from typing import List, Tuple


class OutputVerifier:
    """Verify outputs against training constraints."""
    
    def verify(self, output: np.ndarray, train_outputs: List[np.ndarray]) -> float:
        # TODO: return consistency score [0, 1]
        return 0.5


class CandidateRanker:
    """Rank candidate outputs."""
    
    def __init__(self, verifier: OutputVerifier):
        self.verifier = verifier
    
    def rank(self, candidates: List[Tuple[np.ndarray, float]], 
            train_outputs: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        # TODO: score and sort candidates
        return candidates
