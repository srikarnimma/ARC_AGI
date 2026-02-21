"""
DSL for transformation programs.
Defines operations and tokenizes programs for neural model.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any


class OperationType(Enum):
    RECOLOR = "recolor"
    TRANSLATE = "translate"
    ROTATE = "rotate"
    FLIP = "flip"
    COPY = "copy"
    DELETE = "delete"


class Selector(Enum):
    ALL = "all"
    BY_COLOR = "by_color"
    BY_SIZE = "by_size"
    BY_SHAPE = "by_shape"
    BY_POSITION = "by_position"


@dataclass
class Operation:
    """Single transformation operation."""
    type: OperationType
    selector: Selector
    params: Dict[str, Any]


@dataclass
class TransformProgram:
    """Sequence of operations."""
    operations: List[Operation]


class DSLTokenizer:
    """Input: TransformProgram -> Output: List[int] tokens (and vice versa)"""
    
    def __init__(self):
        self.vocab = {}
    
    def program_to_tokens(self, program: TransformProgram) -> List[int]:
        # TODO
        return []
    
    def tokens_to_program(self, tokens: List[int]) -> TransformProgram:
        # TODO
        return TransformProgram(operations=[])
