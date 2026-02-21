"""
Semantic graph data structure.
Input: objects (from ObjectExtractor) -> Output: SemanticGraph
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class RelationType(Enum):
    ADJACENT = "adjacent"
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class GraphNode:
    """Object node in semantic graph."""
    id: int
    color: int
    bbox: tuple


@dataclass
class Relation:
    """Edge in semantic graph."""
    type: RelationType
    source_id: int
    target_id: int


class SemanticGraph:
    """Input: objects -> Output: graph with nodes and relations."""
    
    def __init__(self):
        self.nodes: Dict[int, GraphNode] = {}
        self.relations: List[Relation] = []


class GraphBuilder:
    """Input: objects -> Output: SemanticGraph"""
    
    def build(self, objects_by_color: Dict) -> SemanticGraph:
        # TODO
        return SemanticGraph()
