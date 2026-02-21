# Semantic graph for object relationships
# Builds spatial graph from extracted objects

from dataclasses import dataclass
from typing import List
from enum import Enum

# Import Object type for type hints
from GraphObjectExtractor import Object


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
    # Semantic graph with object nodes and relationships
    
    def __init__(self):
        self.nodes: dict[int, GraphNode] = {}
        self.relations: List[Relation] = []


class GraphBuilder:
    # Input: list of objects -> Output: SemanticGraph
    
    def build(self, objects: List[Object]) -> SemanticGraph:
        # Build semantic graph from objects
        # TODO: infer spatial relationships (adjacency, above/below, left/right)
        graph = SemanticGraph()
        
        # Add object nodes to graph
        for obj in objects:
            # Skip grid boundary (id=-1)
            if obj.id != -1:
                node = GraphNode(id=obj.id, color=obj.color, bbox=obj.bbox)
                graph.nodes[obj.id] = node
        
        # TODO: Compute spatial relations between objects
        
        return graph
