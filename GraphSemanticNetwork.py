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
    LEFT = "left" #note Obj1 left: Obj2 means object 1 is left of object 2
    RIGHT = "right"
    INSIDE = "inside"
    CONTAINS = "contains"


@dataclass
class GraphNode:
    #Object node in semantic graph
    id: int
    color: int
    bbox: tuple


@dataclass
class Relation:
    #Edge in semantic graph
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
        # Extract nodes and infer spatial relationships
        graph = SemanticGraph()
        
        # Add object nodes to graph (skip grid boundary)
        colored_objects = [obj for obj in objects if obj.id != -1]
        # print(f"Building graph from {len(colored_objects)} objects")
        
        for obj in colored_objects:
            node = GraphNode(id=obj.id, color=obj.color, bbox=obj.bbox)
            graph.nodes[obj.id] = node
        # print(f"Created {len(graph.nodes)} graph nodes")
        
        # Compute spatial relations between all object pairs
        for i, obj1 in enumerate(colored_objects):
            for obj2 in colored_objects[i+1:]:
                # Check spatial relationships
                relations = self._find_relations(obj1, obj2)
                
                for rel_type in relations:
                    # Add relation both directions for ADJACENT, one direction for others
                    graph.relations.append(Relation(rel_type, obj1.id, obj2.id))
                    if rel_type == RelationType.ADJACENT or rel_type == RelationType.ABOVE:
                        # Reverse relation
                        reverse_type = RelationType.BELOW if rel_type == RelationType.ABOVE else rel_type
                        graph.relations.append(Relation(reverse_type, obj2.id, obj1.id))
                    elif rel_type in [RelationType.LEFT, RelationType.RIGHT]:
                        # Add reverse direction
                        reverse_type = RelationType.RIGHT if rel_type == RelationType.LEFT else RelationType.LEFT
                        graph.relations.append(Relation(reverse_type, obj2.id, obj1.id))
                    elif rel_type == RelationType.INSIDE:
                        # Add reverse containment relation
                        graph.relations.append(Relation(RelationType.CONTAINS, obj2.id, obj1.id))
        
        # print(f"Computed {len(graph.relations)} total relations")
        return graph
    
    def _find_relations(self, obj1: Object, obj2: Object) -> List[RelationType]:
        # Detect spatial relationships between two objects
        # Bboxes are (min_r, min_c, max_r, max_c)
        min_r1, min_c1, max_r1, max_c1 = obj1.bbox
        min_r2, min_c2, max_r2, max_c2 = obj2.bbox
        
        relations = []
        # print(f"Checking relations between object {obj1.id} and {obj2.id}")
        
        # Check adjacency (touching or very close)
        h_margin = 1
        v_margin = 1
        
        # Horizontal adjacency
        if (max_c1 + h_margin >= min_c2 and max_c1 <= min_c2) or \
           (max_c2 + h_margin >= min_c1 and max_c2 <= min_c1):
            relations.append(RelationType.ADJACENT)
        
        # Vertical adjacency
        if (max_r1 + v_margin >= min_r2 and max_r1 <= min_r2) or \
           (max_r2 + v_margin >= min_r1 and max_r2 <= min_r1):
            relations.append(RelationType.ADJACENT)
        
        # Above/Below (based on row centers)
        center_r1 = (min_r1 + max_r1) / 2
        center_r2 = (min_r2 + max_r2) / 2
        
        if center_r1 < center_r2 and max_r1 < max_r2:
            relations.append(RelationType.ABOVE)
        elif center_r2 < center_r1 and max_r2 < max_r1:
            relations.append(RelationType.BELOW)
        
        # Left/Right (based on column centers)
        center_c1 = (min_c1 + max_c1) / 2
        center_c2 = (min_c2 + max_c2) / 2
        
        if center_c1 < center_c2 and max_c1 < max_c2:
            relations.append(RelationType.LEFT)
        elif center_c2 < center_c1 and max_c2 < max_c1:
            relations.append(RelationType.RIGHT)
        
        # Inside/Contains (containment)
        # obj1 is INSIDE obj2 if obj1's bbox is fully within obj2's bbox
        if min_r1 >= min_r2 and max_r1 <= max_r2 and min_c1 >= min_c2 and max_c1 <= max_c2:
            # Avoid self-containment (if bboxes are identical)
            if not (min_r1 == min_r2 and max_r1 == max_r2 and min_c1 == min_c2 and max_c1 == max_c2):
                # print(f"  Object {obj1.id} is INSIDE {obj2.id}")
                relations.append(RelationType.INSIDE)
        # obj2 is INSIDE obj1 if obj2's bbox is fully within obj1's bbox
        elif min_r2 >= min_r1 and max_r2 <= max_r1 and min_c2 >= min_c1 and max_c2 <= max_c1:
            # Avoid self-containment (if bboxes are identical)
            if not (min_r1 == min_r2 and max_r1 == max_r2 and min_c1 == min_c2 and max_c1 == max_c2):
                # print(f"  Object {obj2.id} is INSIDE {obj1.id}")
                relations.append(RelationType.INSIDE)
        
        return relations
