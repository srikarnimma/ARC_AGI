import heapq
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from GraphDSL import Operation, OperationType, Selector, TransformProgram
from GraphExecutor import GraphExecutor
from GraphSemanticNetwork import SemanticGraph
from OutputVerifier import OutputVerifier


class SearchResult:
    # Holds the result of an A* search
    def __init__(self, program: TransformProgram, loss: float, expansions: int):
        self.program = program
        self.loss = loss
        self.expansions = expansions


def _program_signature(program: TransformProgram) -> Tuple:
    signature = []
    for op in program.operations:
        params_items = tuple(sorted(op.params.items()))
        signature.append((op.type.value, op.selector.value, params_items))
    return tuple(signature)


class AStarProgramSearch:
    # A* search for DSL programs that fit training examples
    def __init__(
        self,
        executor: GraphExecutor,
        verifier: OutputVerifier,
        max_depth: int = 3,
        max_expansions: int = 500,
        weight: float = 1.0,
        offsets: Optional[List[int]] = None,
        allow_translate: bool = True,
        allow_copy: bool = False,
        size_mismatch_penalty: float = 0.5,
        debug: bool = False,
        debug_every: int = 50,
        print_closest: bool = True,
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
        self.print_closest = print_closest

    def search(
        self,
        training_pairs: List[Tuple[np.ndarray, np.ndarray, SemanticGraph]],
        max_time_seconds: Optional[float] = None,
    ) -> Optional[SearchResult]:
        if not training_pairs:
            return None

        search_start = time.perf_counter()

        def timed_out() -> bool:
            if max_time_seconds is None:
                return False
            return (time.perf_counter() - search_start) >= max_time_seconds

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
        closest_by_depth: Dict[int, Tuple[float, TransformProgram]] = {}

        while heap and expansions < self.max_expansions:
            if timed_out():
                break

            _, depth, _, program, loss = heapq.heappop(heap)
            expansions += 1

            if self.debug and expansions % self.debug_every == 0:
                self._log_status(expansions, depth, loss, len(heap), program)

            signature = _program_signature(program)
            if signature in visited_depth and visited_depth[signature] <= depth:
                if self.debug:
                    self._log_skip_path(depth, program, loss)
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
                if timed_out():
                    break

                new_program = TransformProgram(program.operations + [op])
                new_loss = self._evaluate_program(new_program, training_pairs)

                if new_loss < best_loss:
                    best_loss = new_loss
                    best_program = new_program

                # child_depth = depth + 1
                # prev_closest = closest_by_depth.get(child_depth)
                # if prev_closest is None or new_loss < prev_closest[0]:
                #     closest_by_depth[child_depth] = (new_loss, new_program)
                #     if self.print_closest:
                #         self._log_closest_path(child_depth, new_program, new_loss)

                if self.debug:
                    self._log_op(depth + 1, new_program, new_loss)
                if new_loss == 0.0:
                    return SearchResult(program=new_program, loss=new_loss, expansions=expansions)
                g_cost = depth + 1
                f_cost = g_cost + self.weight * new_loss
                counter += 1
                heapq.heappush(heap, (f_cost, g_cost, counter, new_program, new_loss))

        return SearchResult(program=best_program, loss=best_loss, expansions=expansions)

    def _log_status(self, expansions: int, depth: int, loss: float, heap_size: int, program: TransformProgram) -> None:
        path = self._format_program_path(program)
        print(
            f"[A*] expansions={expansions} depth={depth} loss={loss:.4f} heap={heap_size} "
            f"program_len={len(program.operations)} path={path}"
        )

    def _log_op(self, depth: int, program: TransformProgram, loss: float) -> None:
        path = self._format_program_path(program)
        print(f"[A*] explore depth={depth} loss={loss:.4f} path={path}")

    def _log_skip_path(self, depth: int, program: TransformProgram, loss: float) -> None:
        path = self._format_program_path(program)
        print(f"[A*] skip depth={depth} loss={loss:.4f} reason=visited path={path}")

    def _log_closest_path(self, depth: int, program: TransformProgram, loss: float) -> None:
        path = self._format_program_path(program)
        print(f"[A*] closest depth={depth} loss={loss:.4f} path={path}")

    def _format_program_path(self, program: TransformProgram) -> str:
        if not program.operations:
            return "<empty>"
        return " -> ".join(self._format_op(op) for op in program.operations)

    def _format_op(self, op: Operation) -> str:
        params_items = ", ".join(f"{k}={v}" for k, v in sorted(op.params.items()))
        if params_items:
            return f"{op.type.name}({op.selector.name}; {params_items})"
        return f"{op.type.name}({op.selector.name})"

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
            if predicted.shape == output_grid.shape:
                loss = self.verifier.compute_loss(predicted, output_grid)
            else:
                loss = self._shape_aware_mismatch_loss(predicted, output_grid)
            losses.append(loss)
        return float(np.mean(losses)) if losses else 1.0

    def _shape_aware_mismatch_loss(self, predicted: np.ndarray, target: np.ndarray) -> float:
        # For shape mismatch, estimate how close we are by finding the best-aligned overlap,
        # then add a softer size penalty. This rewards promising intermediate steps.
        p_h, p_w = predicted.shape
        t_h, t_w = target.shape

        # If one grid can fully slide over the other, compare at best alignment.
        if p_h <= t_h and p_w <= t_w:
            best = self._best_window_match(predicted, target)
        elif t_h <= p_h and t_w <= p_w:
            best = self._best_window_match(target, predicted)
        else:
            # Mixed mismatch (one dim larger, one smaller): fallback to overlap crop.
            min_h = min(p_h, t_h)
            min_w = min(p_w, t_w)
            pred_crop = predicted[:min_h, :min_w]
            target_crop = target[:min_h, :min_w]
            content_loss = self.verifier.compute_loss(pred_crop, target_crop)
            mask_loss = self._mask_loss(pred_crop, target_crop)
            best = 0.7 * content_loss + 0.3 * mask_loss

        size_penalty = self._relative_size_penalty(predicted.shape, target.shape)
        return float(min(1.0, best + self.size_mismatch_penalty * size_penalty))

    def _best_window_match(self, small: np.ndarray, large: np.ndarray) -> float:
        s_h, s_w = small.shape
        l_h, l_w = large.shape

        best_loss = 1.0
        for row in range(l_h - s_h + 1):
            for col in range(l_w - s_w + 1):
                window = large[row:row + s_h, col:col + s_w]
                content_loss = self.verifier.compute_loss(small, window)
                mask_loss = self._mask_loss(small, window)
                combined = 0.7 * content_loss + 0.3 * mask_loss
                if combined < best_loss:
                    best_loss = combined
        return float(best_loss)

    def _mask_loss(self, a: np.ndarray, b: np.ndarray) -> float:
        # Compare non-zero support irrespective of exact color values.
        a_mask = (a != 0)
        b_mask = (b != 0)
        union = np.sum(a_mask | b_mask)
        if union == 0:
            return 0.0
        intersection = np.sum(a_mask & b_mask)
        iou = float(intersection) / float(union)
        return float(1.0 - iou)

    def _relative_size_penalty(self, shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> float:
        a_h, a_w = shape_a
        b_h, b_w = shape_b
        row_gap = abs(a_h - b_h) / float(max(a_h, b_h, 1))
        col_gap = abs(a_w - b_w) / float(max(a_w, b_w, 1))
        return float(0.5 * (row_gap + col_gap))

    def _build_action_space(
        self,
        training_pairs: List[Tuple[np.ndarray, np.ndarray, SemanticGraph]],
    ) -> List[Operation]:
        colors_in = self._collect_colors([pair[0] for pair in training_pairs])
        colors_out = self._collect_colors([pair[1] for pair in training_pairs])
        palette = sorted(colors_in.union(colors_out))

        actions: List[Operation] = []

        # Try recoloring each color combo.
        # Include generated colors (e.g., SPIRAL_FILL seed color 3), not only
        # colors present in the raw inputs.
        recolor_sources = sorted(colors_in.union(colors_out).union({3}))
        for from_color in recolor_sources:
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

        # Shape-driven recolor ops (uses extractor features like is_closed_shape)
        shape_labels = ["circle", "cycle", "triangle", "arrow"]
        for shape in shape_labels:
            for to_color in (sorted(colors_out - {0}) or [8]):
                actions.append(
                    Operation(
                        type=OperationType.RECOLOR,
                        selector=Selector.BY_SHAPE,
                        params={"shape": shape, "new_color": to_color},
                    )
                )
                for from_color in colors_in:
                    if from_color == to_color:
                        continue
                    actions.append(
                        Operation(
                            type=OperationType.RECOLOR,
                            selector=Selector.BY_SHAPE,
                            params={"shape": shape, "color": from_color, "new_color": to_color},
                        )
                    )

        # add flip ops (horizontal & vertical)
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
        
        # Mirror ops to create symmetry
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

        # Rotation options
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

        # Small translations
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

        # Delete by color
        for color in colors_in:
            actions.append(
                Operation(
                    type=OperationType.DELETE,
                    selector=Selector.BY_COLOR,
                    params={"color": color},
                )
            )

        # Hollowing (empty out shapes)
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

        # Cropping to non-zero bbox
        actions.append(
            Operation(
                type=OperationType.CROP_NONZERO_BBOX,
                selector=Selector.ALL,
                params={},
            )
        )

        # Contextual crop/recolor operations
        actions.append(
            Operation(
                type=OperationType.CROP_RECOLOR_BY_CORNER_MARKERS,
                selector=Selector.ALL,
                params={},
            )
        )
        actions.append(
            Operation(
                type=OperationType.RECOLOR_MAIN_BY_EXTERNAL_PAIRS,
                selector=Selector.ALL,
                params={},
            )
        )
        actions.append(
            Operation(
                type=OperationType.CONTEXTUAL_SYMMETRY_FILL,
                selector=Selector.ALL,
                params={},
            )
        )

        # Full-grid spiral generation
        actions.append(
            Operation(
                type=OperationType.SPIRAL_FILL,
                selector=Selector.ALL,
                params={"color": 3},
            )
        )

        # Span matching same-color endpoints along rows/columns
        actions.append(
            Operation(
                type=OperationType.SPAN_MATCHING_COLOR_ENDPOINTS,
                selector=Selector.ALL,
                params={"include_cols": False},
            )
        )
        actions.append(
            Operation(
                type=OperationType.SPAN_MATCHING_COLOR_ENDPOINTS,
                selector=Selector.ALL,
                params={"include_cols": True},
            )
        )

        # Fill enclosed zero-regions (with optional exterior recolor)
        output_colors = [c for c in colors_out if c != 0]
        for enclosed_color in output_colors or [2]:
            actions.append(
                Operation(
                    type=OperationType.FILL_ENCLOSED_ZEROS,
                    selector=Selector.ALL,
                    params={"enclosed_color": enclosed_color, "exterior_color": -1},
                )
            )
            actions.append(
                Operation(
                    type=OperationType.FILL_ENCLOSED_ZEROS,
                    selector=Selector.ALL,
                    params={"enclosed_color": enclosed_color, "exterior_color": -1, "mode": "local_component_mode"},
                )
            )
            for exterior_color in output_colors or [3]:
                actions.append(
                    Operation(
                        type=OperationType.FILL_ENCLOSED_ZEROS,
                        selector=Selector.ALL,
                        params={"enclosed_color": enclosed_color, "exterior_color": exterior_color},
                    )
                )

        # Cleanup: remove singleton non-zero objects
        actions.append(
            Operation(
                type=OperationType.REMOVE_SINGLE_PIXEL_OBJECTS,
                selector=Selector.ALL,
                params={},
            )
        )

        # Logical ops between subgrids (auto-detect the separator)
        output_colors = [c for c in palette if c != 0]
        logic_ops = ['AND', 'OR', 'XOR', 'XNOR', 'NAND', 'NOR', 'FIT_ADD']
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

        # Copy objs with small offsets
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

        # Object-based ops (only when obj IDs are consistent)
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
                
                # Swap colors btw objects
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

    def _collect_colors(self, grids: Iterable[np.ndarray]) -> set[int]:
        colors: set[int] = set()
        for grid in grids:
            if grid.size == 0:
                continue
            colors.update(int(value) for value in np.unique(grid))
        return colors
