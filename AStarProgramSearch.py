import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from GraphDSL import Operation, OperationType, Selector, TransformProgram
from GraphExecutor import GraphExecutor
from GraphSemanticNetwork import SemanticGraph
from OutputVerifier import OutputVerifier


@dataclass(frozen=True)
class SearchResult:
    program: TransformProgram
    loss: float
    expansions: int


def _program_signature(program: TransformProgram) -> Tuple:
    signature = []
    for op in program.operations:
        params_items = tuple(sorted(op.params.items()))
        signature.append((op.type.value, op.selector.value, params_items))
    return tuple(signature)


class AStarProgramSearch:
    """A* search over DSL programs to fit training input/output pairs."""
    def __init__(
        self,
        executor: GraphExecutor,
        verifier: OutputVerifier,
        max_depth: int = 4,
        max_expansions: int = 1500,
        weight: float = 1.0,
        offsets: Optional[List[int]] = None,
        allow_translate: bool = True,
        allow_copy: bool = False,
        size_mismatch_penalty: float = 0.5,
        debug: bool = False,
        debug_every: int = 50,
    ) -> None:
        self.executor = executor
        self.verifier = verifier
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.weight = weight
        self.offsets = offsets if offsets is not None else [-1, 1]
        self.allow_translate = allow_translate
        self.allow_copy = allow_copy
        self.size_mismatch_penalty = size_mismatch_penalty
        self.debug = debug
        self.debug_every = max(1, debug_every)

    def search(self, training_pairs: List[Tuple[np.ndarray, np.ndarray, SemanticGraph]]) -> Optional[SearchResult]:
        if not training_pairs:
            return None

        action_space = self._build_action_space(training_pairs)
        start_program = TransformProgram([])
        start_loss = self._evaluate_program(start_program, training_pairs)

        best_program = start_program
        best_loss = start_loss

        heap: List[Tuple[float, int, int, TransformProgram, float]] = []
        counter = 0
        heapq.heappush(heap, (start_loss, 0, counter, start_program, start_loss))

        visited_depth: Dict[Tuple, int] = {}
        expansions = 0

        while heap and expansions < self.max_expansions:
            _, depth, _, program, loss = heapq.heappop(heap)
            expansions += 1

            if self.debug and expansions % self.debug_every == 0:
                self._log_status(expansions, depth, loss, len(heap), program)

            signature = _program_signature(program)
            if signature in visited_depth and visited_depth[signature] <= depth:
                continue
            visited_depth[signature] = depth

            if loss < best_loss:
                best_loss = loss
                best_program = program

            if loss == 0.0:
                return SearchResult(program=program, loss=loss, expansions=expansions)

            if depth >= self.max_depth:
                continue

            for op in action_space:
                new_program = TransformProgram(program.operations + [op])
                new_loss = self._evaluate_program(new_program, training_pairs)
                if self.debug:
                    self._log_op(depth + 1, op, new_loss)
                if new_loss == 0.0:
                    return SearchResult(program=new_program, loss=new_loss, expansions=expansions)
                g_cost = depth + 1
                f_cost = g_cost + self.weight * new_loss
                counter += 1
                heapq.heappush(heap, (f_cost, g_cost, counter, new_program, new_loss))

        return SearchResult(program=best_program, loss=best_loss, expansions=expansions)

    def _log_status(self, expansions: int, depth: int, loss: float, heap_size: int, program: TransformProgram) -> None:
        print(
            f"[A*] expansions={expansions} depth={depth} loss={loss:.4f} heap={heap_size} "
            f"program_len={len(program.operations)}"
        )

    def _log_op(self, depth: int, op: Operation, loss: float) -> None:
        params_items = ", ".join(f"{k}={v}" for k, v in sorted(op.params.items()))
        print(f"[A*] try depth={depth} op={op.type.name} sel={op.selector.name} {params_items} -> loss={loss:.4f}")

    def _evaluate_program(
        self,
        program: TransformProgram,
        training_pairs: List[Tuple[np.ndarray, np.ndarray, SemanticGraph]],
    ) -> float:
        losses: List[float] = []
        for input_grid, output_grid, graph in training_pairs:
            try:
                predicted = self.executor.execute(input_grid, graph, program)
            except Exception:
                return 1.0
            loss = self.verifier.compute_loss(predicted, output_grid)
            if predicted.shape != output_grid.shape:
                loss = min(1.0, loss + self.size_mismatch_penalty)
            losses.append(loss)
        return float(np.mean(losses)) if losses else 1.0

    def _build_action_space(
        self,
        training_pairs: List[Tuple[np.ndarray, np.ndarray, SemanticGraph]],
    ) -> List[Operation]:
        colors_in = self._collect_colors([pair[0] for pair in training_pairs])
        colors_out = self._collect_colors([pair[1] for pair in training_pairs])
        palette = sorted(colors_in.union(colors_out))

        actions: List[Operation] = []

        # Recolor operations
        for from_color in colors_in:
            for to_color in palette:
                if to_color == from_color:
                    continue
                actions.append(
                    Operation(
                        type=OperationType.RECOLOR,
                        selector=Selector.BY_COLOR,
                        params={"color": from_color, "new_color": to_color},
                    )
                )

        # Flip operations
        for direction in ["horizontal", "vertical"]:
            actions.append(
                Operation(
                    type=OperationType.FLIP,
                    selector=Selector.ALL,
                    params={"direction": direction},
                )
            )
            actions.append(
                Operation(
                    type=OperationType.FLIP_GRID,
                    selector=Selector.ALL,
                    params={"direction": direction},
                )
            )
        
        # Mirror operations (create symmetry around center axis)
        actions.append(
            Operation(
                type=OperationType.MIRROR_VERTICAL,
                selector=Selector.ALL,
                params={},
            )
        )
        actions.append(
            Operation(
                type=OperationType.MIRROR_HORIZONTAL,
                selector=Selector.ALL,
                params={},
            )
        )

        # Rotate operations
        for angle in [90, 180, 270]:
            actions.append(
                Operation(
                    type=OperationType.ROTATE,
                    selector=Selector.ALL,
                    params={"angle": angle},
                )
            )
            actions.append(
                Operation(
                    type=OperationType.ROTATE_GRID,
                    selector=Selector.ALL,
                    params={"angle": angle},
                )
            )

        # Translate operations (small offsets)
        if self.allow_translate:
            for offset_r in self.offsets:
                for offset_c in self.offsets:
                    actions.append(
                        Operation(
                            type=OperationType.TRANSLATE,
                            selector=Selector.ALL,
                            params={"offset_r": offset_r, "offset_c": offset_c},
                        )
                    )

        # Delete operations by color
        for color in colors_in:
            actions.append(
                Operation(
                    type=OperationType.DELETE,
                    selector=Selector.BY_COLOR,
                    params={"color": color},
                )
            )

        # Hollow operations
        actions.append(
            Operation(
                type=OperationType.HOLLOW,
                selector=Selector.ALL,
                params={},
            )
        )
        for color in colors_in:
            actions.append(
                Operation(
                    type=OperationType.HOLLOW,
                    selector=Selector.BY_COLOR,
                    params={"color": color},
                )
            )

        # Crop operations
        actions.append(
            Operation(
                type=OperationType.CROP_NONZERO_BBOX,
                selector=Selector.ALL,
                params={},
            )
        )

        # Separator-based logical operations between subgrids (auto-detect separator)
        output_colors = [c for c in palette if c != 0]
        logic_ops = ['AND', 'OR', 'XOR', 'XNOR', 'NAND', 'NOR']
        split_directions = ['ROW', 'COL']
        for logic_op in logic_ops:
            for out_color in output_colors or [1]:
                for split_dir in split_directions:
                    actions.append(
                        Operation(
                            type=OperationType.AND_SPLIT,
                            selector=Selector.ALL,
                            params={"output_color": out_color, "logic_op": logic_op, "split_direction": split_dir},
                        )
                    )

        # Copy operations by color with small offsets
        if self.allow_copy:
            for color in colors_in:
                for offset_r in self.offsets:
                    for offset_c in self.offsets:
                        actions.append(
                            Operation(
                                type=OperationType.COPY,
                                selector=Selector.BY_COLOR,
                                params={"color": color, "offset_r": offset_r, "offset_c": offset_c},
                            )
                        )

        # Logical operations only when object IDs are consistent across graphs
        if training_pairs:
            node_id_sets = [set(pair[2].nodes.keys()) for pair in training_pairs]
            if all(node_id_sets[0] == node_ids for node_ids in node_id_sets[1:]):
                object_ids = sorted(node_id_sets[0])
                output_colors = [c for c in palette if c != 0]
                for idx, obj_id_1 in enumerate(object_ids):
                    for obj_id_2 in object_ids[idx + 1:]:
                        for out_color in output_colors or [1]:
                            actions.append(
                                Operation(
                                    type=OperationType.XOR,
                                    selector=Selector.ALL,
                                    params={
                                        "object_id_1": obj_id_1,
                                        "object_id_2": obj_id_2,
                                        "output_color": out_color,
                                    },
                                )
                            )
                
                # Swap colors operation (no output_color needed, just swaps existing colors)
                for idx, obj_id_1 in enumerate(object_ids):
                    for obj_id_2 in object_ids[idx + 1:]:
                        actions.append(
                            Operation(
                                type=OperationType.SWAP_COLORS,
                                selector=Selector.ALL,
                                params={
                                    "object_id_1": obj_id_1,
                                    "object_id_2": obj_id_2,
                                },
                            )
                        )

        return actions

    @staticmethod
    def _collect_colors(grids: Iterable[np.ndarray]) -> set[int]:
        colors: set[int] = set()
        for grid in grids:
            if grid.size == 0:
                continue
            colors.update(int(value) for value in np.unique(grid))
        return colors
