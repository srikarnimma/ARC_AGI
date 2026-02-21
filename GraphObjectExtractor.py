"""
Utilities for object extraction.
Input: grid (numpy array)
Output: dict of color -> list of Objects
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class Object:
    """Represents an object/component in a grid."""
    id: int
    color: int
    pixels: Set[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)


class ObjectExtractor:
    """Input: grid -> Output: Dict[color, List[Object]]"""
    
    def __init__(self, connectivity: str = "8"):
        self.connectivity = connectivity
    
    def extract(self, grid: np.ndarray) -> Dict[int, List[Object]]:
        # TODO
        return {}
