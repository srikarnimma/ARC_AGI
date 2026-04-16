# DSL for transformation programs
# Defines operations, selectors, and tokenization for programs

from enum import Enum
from typing import List, Dict, Any


class OperationType(Enum):
    RECOLOR = "recolor"
    TRANSLATE = "translate"
    ROTATE = "rotate"
    ROTATE_GRID = "rotate_grid"
    FLIP = "flip"
    FLIP_GRID = "flip_grid"
    MIRROR_VERTICAL = "mirror_vertical"
    MIRROR_HORIZONTAL = "mirror_horizontal"
    COPY = "copy"
    DELETE = "delete"
    HOLLOW = "hollow"
    CROP_NONZERO_BBOX = "crop_nonzero_bbox"
    AND_SPLIT = "and_split"
    AND = "and"
    OR = "or"
    XOR = "xor"
    XNOR = "xnor"
    NAND = "nand"
    NOR = "nor"
    SWAP_COLORS = "swap_colors"
    FILL_ENCLOSED_ZEROS = "fill_enclosed_zeros"
    REMOVE_SINGLE_PIXEL_OBJECTS = "remove_single_pixel_objects"
    CROP_RECOLOR_BY_CORNER_MARKERS = "crop_recolor_by_corner_markers"
    SPAN_MATCHING_COLOR_ENDPOINTS = "span_matching_color_endpoints"
    RECOLOR_MAIN_BY_EXTERNAL_PAIRS = "recolor_main_by_external_pairs"
    CONTEXTUAL_SYMMETRY_FILL = "contextual_symmetry_fill"


class Selector(Enum):
    ALL = "all"
    BY_COLOR = "by_color"
    BY_SIZE = "by_size"
    BY_SHAPE = "by_shape"
    BY_POSITION = "by_position"
    BY_DIRECTION = "by_direction"


class Operation:
    # A transformation operation
    def __init__(self, type: OperationType, selector: Selector, params: Dict[str, Any]):
        self.type = type
        self.selector = selector
        self.params = params


class TransformProgram:
    # Sequence of operations
    def __init__(self, operations: List[Operation]):
        self.operations = operations


class DSLTokenizer:
    # Converts between TransformProgram and token sequences
    # Tokens: [OP_TYPE, SELECTOR, PARAM1, PARAM2, ...]
    
    def __init__(self):
        # Build vocabulary
        self.vocab = {}
        self.reverse_vocab = {}
        token_id = 0
        
        # Operation types (0-9)
        for op in OperationType:
            self.vocab[f"OP_{op.name}"] = token_id
            self.reverse_vocab[token_id] = f"OP_{op.name}"
            token_id += 1
        # print(f"Added {len(OperationType)} operation token types")
        
        # Selectors (10-19)
        for sel in Selector:
            self.vocab[f"SEL_{sel.name}"] = token_id
            self.reverse_vocab[token_id] = f"SEL_{sel.name}"
            token_id += 1
        # print(f"Added {len(Selector)} selector token types")
        
        # Parameters (20+)
        # Common param values
        param_names = [
            "color", "direction", "angle", "offset_r", "offset_c",
            "value_0", "value_1", "value_2", "value_3", "value_4",
            "value_5", "value_6", "value_7", "value_8", "value_9"
        ]
        
        for param in param_names:
            self.vocab[f"PARAM_{param}"] = token_id
            self.reverse_vocab[token_id] = f"PARAM_{param}"
            token_id += 1
        
        # Special tokens (100+)
        self.vocab["END"] = 100
        self.reverse_vocab[100] = "END"
        self.vocab["PAD"] = 101
        self.reverse_vocab[101] = "PAD"
        # print(f"Total vocabulary size: {len(self.vocab)}")
    
    def program_to_tokens(self, program: TransformProgram) -> List[int]:
        # Convert program to token sequence
        # Format: [OP, SELECTOR, PARAM_KEY, PARAM_VALUE, OP, SELECTOR, ...., END]
        tokens = []
        
        # print(f"Tokenizing {len(program.operations)} operations")
        
        for op in program.operations:
            # Operation type
            op_key = f"OP_{op.type.name}"
            tokens.append(self.vocab.get(op_key, 0))
            
            # Selector
            sel_key = f"SEL_{op.selector.name}"
            tokens.append(self.vocab.get(sel_key, 0))
            
            # Parameters
            for param_name, param_value in op.params.items():
                # Encode parameter name
                param_key = f"PARAM_{param_name}"
                # print(f"  Encoding param: {param_name} = {param_value}")
                if param_key in self.vocab:
                    tokens.append(self.vocab[param_key])
                
                # Encode parameter value (map to vocab range to avoid out-of-bounds)
                # Map integer values to range [20-99] to fit within vocab
                if isinstance(param_value, int):
                    # Map 0-9 colors to tokens 20-29, keep small values in range
                    token_val = min(20 + param_value, 99)  # Clamp to [20, 99]
                    tokens.append(token_val)
                elif isinstance(param_value, str):
                    # Hash strings and map to range [30-99]
                    token_val = 30 + (hash(param_value) % 70)  # Map to [30, 99]
                    tokens.append(token_val)
        
        # End token
        tokens.append(self.vocab["END"])
        # print(f"Generated token sequence of length {len(tokens)}")
        
        return tokens
    
    def tokens_to_program(self, tokens: List[int]) -> TransformProgram:
        # Convert token sequence back to program
        # TODO: Implement decoding (reverse of program_to_tokens)
        # For now, return empty program
        return TransformProgram(operations=[])
