# Graph executor: applies transformation programs to input grids using semantic graphs
# Input: (input_grid, semantic_graph, program) -> Output: output_grid (numpy array)

import numpy as np
from copy import deepcopy
from typing import Optional, Set, Tuple
from collections import deque
from GraphSemanticNetwork import SemanticGraph
from GraphDSL import TransformProgram, OperationType, Selector
from GraphObjectExtractor import Object, ObjectExtractor


class GraphExecutor:
    # Executes transform programs on grids using semantic graph info

    def _is_grid_operation(self, operation_type: OperationType) -> bool:
        return operation_type in {
            OperationType.ROTATE_GRID,
            OperationType.FLIP_GRID,
            OperationType.MIRROR_VERTICAL,
            OperationType.MIRROR_HORIZONTAL,
            OperationType.CROP_NONZERO_BBOX,
            OperationType.FILL_ENCLOSED_ZEROS,
            OperationType.REMOVE_SINGLE_PIXEL_OBJECTS,
            OperationType.CROP_RECOLOR_BY_CORNER_MARKERS,
            OperationType.SPAN_MATCHING_COLOR_ENDPOINTS,
            OperationType.AND_SPLIT,
            OperationType.RECOLOR_MAIN_BY_EXTERNAL_PAIRS,
            OperationType.CONTEXTUAL_SYMMETRY_FILL,
            OperationType.SPIRAL_FILL,
        }
    
    def execute(self, input_grid: np.ndarray, graph: SemanticGraph, program: TransformProgram) -> np.ndarray:
        # Run a transformation program on an input grid
        grid_height, grid_width = input_grid.shape
        # print(f"[GraphExecutor] Executing {len(program.operations)} ops on {grid_height}x{grid_width} grid")
        
        # Start with blank output
        output_grid = np.zeros_like(input_grid)
        
        # Create mutable copies of graph objects
        # Map object_id -> modified_object
        objects_map = {}
        # print(f"[GraphExecutor] Reconstructing {len(graph.nodes)} objs from graph")
        for obj_id, graph_node in graph.nodes.items():
            # Reconstruct obj from graph node
            obj = Object(
                id=obj_id,
                color=graph_node.color,
                pixels=set(),  # Will be inferred from grid
                bbox=graph_node.bbox
            )
            obj.is_closed_shape = getattr(graph_node, 'is_closed_shape', False)
            obj.is_triangle = getattr(graph_node, 'is_triangle', False)
            obj.is_arrow = getattr(graph_node, 'is_arrow', False)
            obj.is_cyclic = getattr(graph_node, 'is_cyclic', False)
            # Extract pixels from input grid in this bbox
            min_r, min_c, max_r, max_c = graph_node.bbox
            for r in range(max(0, min_r), min(grid_height, max_r + 1)):
                for c in range(max(0, min_c), min(grid_width, max_c + 1)):
                    if input_grid[r, c] == graph_node.color:
                        obj.pixels.add((r, c))
            objects_map[obj_id] = obj

        # If a grid-level op is used, we switch to grid pipeline mode and
        # apply subsequent grid-level ops sequentially.
        current_grid: Optional[np.ndarray] = None
        object_state_dirty = False
        
        # Apply each op in seq
        for operation in program.operations:
            # print(f"[GraphExecutor] Applying {operation.type.name}")
            if current_grid is not None and not self._is_grid_operation(operation.type):
                # Object-level ops cannot be applied meaningfully after switching
                # to grid-level mode; RECOLOR is a special case handled on grid.
                if operation.type != OperationType.RECOLOR:
                    continue

            if current_grid is None and not self._is_grid_operation(operation.type):
                object_state_dirty = True

            # Find matching objs (used by object-level ops)
            matching_ids = self._select_objects(objects_map, operation.selector, operation.params)
            
            # Apply the op
            if operation.type == OperationType.RECOLOR:
                new_color = operation.params.get('new_color', 0)
                if current_grid is not None:
                    print("------")
                    grid_state = current_grid.copy()
                    print(operation.selector, operation.params.get('color', None), new_color)
                    print(grid_state)
                    if operation.selector == Selector.BY_COLOR:
                        source_color = operation.params.get('color', None)
                        if source_color is not None:
                            grid_state[grid_state == source_color] = new_color
                    elif operation.selector == Selector.ALL:
                        grid_state[grid_state != 0] = new_color
                    current_grid = grid_state
                    print(current_grid)
                    print("~~~~~~~~~")
                else:
                    for obj_id in matching_ids:
                        if obj_id in objects_map:
                            objects_map[obj_id].color = new_color
            
            elif operation.type == OperationType.TRANSLATE:
                offset_r = operation.params.get('offset_r', 0)
                offset_c = operation.params.get('offset_c', 0)
                for obj_id in matching_ids:
                    if obj_id in objects_map:
                        self._translate_object(objects_map[obj_id], offset_r, offset_c, grid_height, grid_width)
            
            elif operation.type == OperationType.ROTATE:
                angle = operation.params.get('angle', 0)
                for obj_id in matching_ids:
                    if obj_id in objects_map:
                        self._rotate_object(objects_map[obj_id], angle)

            elif operation.type == OperationType.ROTATE_GRID:
                angle = operation.params.get('angle', 0)
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                current_grid = self._rotate_grid(grid_state, angle)
            
            elif operation.type == OperationType.FLIP:
                direction = operation.params.get('direction', 'horizontal')
                for obj_id in matching_ids:
                    if obj_id in objects_map:
                        self._flip_object(objects_map[obj_id], direction)

            elif operation.type == OperationType.FLIP_GRID:
                direction = operation.params.get('direction', 'horizontal')
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                current_grid = self._flip_grid(grid_state, direction)
            
            elif operation.type == OperationType.MIRROR_VERTICAL:
                # Mirror around vertical center axis (left-right symmetry)
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                current_grid = self._mirror_vertical(grid_state)
            
            elif operation.type == OperationType.MIRROR_HORIZONTAL:
                # Mirror around horizontal center axis (top-bottom symmetry)
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                current_grid = self._mirror_horizontal(grid_state)
            
            elif operation.type == OperationType.COPY:
                offset_r = operation.params.get('offset_r', 0)
                offset_c = operation.params.get('offset_c', 0)
                new_objects = []
                for obj_id in matching_ids:
                    if obj_id in objects_map:
                        new_obj = deepcopy(objects_map[obj_id])
                        new_obj.id = max(objects_map.keys()) + 1 if objects_map else 1
                        self._translate_object(new_obj, offset_r, offset_c, grid_height, grid_width)
                        new_objects.append(new_obj)
                for new_obj in new_objects:
                    objects_map[new_obj.id] = new_obj
            
            elif operation.type == OperationType.DELETE:
                for obj_id in matching_ids:
                    if obj_id in objects_map:
                        del objects_map[obj_id]

            elif operation.type == OperationType.HOLLOW:
                for obj_id in matching_ids:
                    if obj_id in objects_map:
                        self._hollow_object(objects_map[obj_id])

            elif operation.type == OperationType.SWAP_COLORS:
                obj_id_1 = operation.params.get('object_id_1')
                obj_id_2 = operation.params.get('object_id_2')
                if obj_id_1 in objects_map and obj_id_2 in objects_map:
                    # Swap the colors of two objects
                    objects_map[obj_id_1].color, objects_map[obj_id_2].color = objects_map[obj_id_2].color, objects_map[obj_id_1].color

            elif operation.type == OperationType.CROP_NONZERO_BBOX:
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                current_grid = self._crop_nonzero_bbox(grid_state)

            elif operation.type == OperationType.FILL_ENCLOSED_ZEROS:
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                enclosed_color = operation.params.get('enclosed_color', 2)
                exterior_color = operation.params.get('exterior_color', -1)
                mode = operation.params.get('mode', 'global')
                current_grid = self._fill_enclosed_zeros(grid_state, enclosed_color, exterior_color, mode)

            elif operation.type == OperationType.REMOVE_SINGLE_PIXEL_OBJECTS:
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                current_grid = self._remove_single_pixel_objects(grid_state)

            elif operation.type == OperationType.CROP_RECOLOR_BY_CORNER_MARKERS:
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                current_grid = self._crop_recolor_by_corner_markers(grid_state)

            elif operation.type == OperationType.SPAN_MATCHING_COLOR_ENDPOINTS:
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                include_cols = bool(operation.params.get('include_cols', False))
                current_grid = self._span_matching_color_endpoints(grid_state, include_cols=include_cols)

            elif operation.type == OperationType.RECOLOR_MAIN_BY_EXTERNAL_PAIRS:
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                current_grid = self._recolor_main_by_external_pairs(grid_state)

            elif operation.type == OperationType.CONTEXTUAL_SYMMETRY_FILL:
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                current_grid = self._contextual_symmetry_fill(grid_state)

            elif operation.type == OperationType.SPIRAL_FILL:
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                spiral_color = int(operation.params.get('color', 3))
                current_grid = self._spiral_fill(grid_state, spiral_color)

            elif operation.type == OperationType.AND_SPLIT:
                separator_color = operation.params.get('separator_color')
                output_color = operation.params.get('output_color', 2)
                logic_op = operation.params.get('logic_op', 'AND')
                split_direction = operation.params.get('split_direction', 'AUTO')  # AUTO, ROW, or COL
                if current_grid is not None:
                    grid_state = current_grid
                elif object_state_dirty:
                    grid_state = self._render_objects(objects_map, grid_height, grid_width)
                else:
                    grid_state = input_grid.copy()
                current_grid = self._logical_and_split(grid_state, separator_color, output_color, logic_op, split_direction)
            
            elif operation.type in [OperationType.AND, OperationType.OR, OperationType.XOR, 
                                    OperationType.XNOR, OperationType.NAND, OperationType.NOR]:
                # Logical operations: combine two objects pixel-wise
                obj_id_1 = operation.params.get('object_id_1')
                obj_id_2 = operation.params.get('object_id_2')
                output_color = operation.params.get('output_color', 2)
                result_obj_id = operation.params.get('result_id', max(objects_map.keys()) + 1 if objects_map else 1)
                # print(f"[GraphExecutor] Applying {operation.type.name} on objects {obj_id_1} and {obj_id_2}")
                
                if obj_id_1 in objects_map and obj_id_2 in objects_map:
                    obj1 = objects_map[obj_id_1]
                    obj2 = objects_map[obj_id_2]
                    
                    # Get bbox that encompasses both objects
                    min_r = min(obj1.bbox[0], obj2.bbox[0])
                    min_c = min(obj1.bbox[1], obj2.bbox[1])
                    max_r = max(obj1.bbox[2], obj2.bbox[2])
                    max_c = max(obj1.bbox[3], obj2.bbox[3])
                    
                    # Apply logical operation pixel by pixel
                    result_pixels = set()
                    for r in range(min_r, max_r + 1):
                        for c in range(min_c, max_c + 1):
                            has_pixel_1 = (r, c) in obj1.pixels
                            has_pixel_2 = (r, c) in obj2.pixels
                            
                            # Apply logical operation
                            result = False
                            if operation.type == OperationType.AND:
                                result = has_pixel_1 and has_pixel_2
                            elif operation.type == OperationType.OR:
                                result = has_pixel_1 or has_pixel_2
                            elif operation.type == OperationType.XOR:
                                result = has_pixel_1 != has_pixel_2
                            elif operation.type == OperationType.XNOR:
                                result = has_pixel_1 == has_pixel_2
                            elif operation.type == OperationType.NAND:
                                result = not (has_pixel_1 and has_pixel_2)
                            elif operation.type == OperationType.NOR:
                                result = not (has_pixel_1 or has_pixel_2)
                            
                            if result:
                                result_pixels.add((r, c))
                    
                    # Create result object or replace existing
                    if result_obj_id in objects_map:
                        objects_map[result_obj_id].pixels = result_pixels
                        objects_map[result_obj_id].color = output_color
                    else:
                        result_obj = Object(
                            id=result_obj_id,
                            color=output_color,
                            pixels=result_pixels,
                            bbox=(min_r, min_c, max_r, max_c)
                        )
                        objects_map[result_obj_id] = result_obj
        
        if current_grid is not None:
            return current_grid

        # Render all objects to output grid
        output_grid = self._render_objects(objects_map, grid_height, grid_width)
        
        return output_grid
    
    def _select_objects(self, objects_map: dict, selector: Selector, params: dict) -> Set[int]:
        # Select which objects match the selector
        matching = set()
        # print(f"[GraphExecutor] Selecting objects with {selector.name}")
        
        if selector == Selector.ALL:
            matching = set(objects_map.keys())
        
        elif selector == Selector.BY_COLOR:
            color = params.get('color', 0)
            for obj_id, obj in objects_map.items():
                if obj.color == color:
                    matching.add(obj_id)
        
        elif selector == Selector.BY_SIZE:
            size = params.get('size', 0)
            for obj_id, obj in objects_map.items():
                if len(obj.pixels) == size:
                    matching.add(obj_id)
        
        elif selector == Selector.BY_SHAPE:
            shape = params.get('shape', 'unknown')
            color_filter = params.get('color', None)
            for obj_id, obj in objects_map.items():
                if color_filter is not None and obj.color != color_filter:
                    continue
                if shape == 'arrow' and obj.is_arrow:
                    matching.add(obj_id)
                elif shape == 'triangle' and obj.is_triangle:
                    matching.add(obj_id)
                elif shape == 'circle' and obj.is_closed_shape:
                    matching.add(obj_id)
                elif shape == 'cycle' and obj.is_cyclic:
                    matching.add(obj_id)
        
        elif selector == Selector.BY_POSITION:
            # Select by position (top, bottom, left, right, center)
            position = params.get('position', 'center')
            for obj_id, obj in objects_map.items():
                min_r, min_c, max_r, max_c = obj.bbox
                center_r = (min_r + max_r) // 2
                center_c = (min_c + max_c) // 2
                
                if position == 'top' and min_r < 5:
                    matching.add(obj_id)
                elif position == 'bottom' and max_r > 25:
                    matching.add(obj_id)
                elif position == 'left' and min_c < 5:
                    matching.add(obj_id)
                elif position == 'right' and max_c > 25:
                    matching.add(obj_id)
                elif position == 'center' and 5 <= center_r <= 25 and 5 <= center_c <= 25:
                    matching.add(obj_id)
        
        elif selector == Selector.BY_DIRECTION:
            direction = params.get('direction', 'horizontal')
            for obj_id, obj in objects_map.items():
                min_r, min_c, max_r, max_c = obj.bbox
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                
                if direction == 'horizontal' and width > height:
                    matching.add(obj_id)
                elif direction == 'vertical' and height > width:
                    matching.add(obj_id)
        
        return matching
        # print(f"[GraphExecutor] Selected {len(matching)} objects")
    
    def _translate_object(self, obj: Object, offset_r: int, offset_c: int, 
                         grid_height: int, grid_width: int) -> None:
        # Translate object pixels and update bounding box
        # print(f"[GraphExecutor] Translating object {obj.id} by ({offset_r}, {offset_c})")
        new_pixels = set()
        for r, c in obj.pixels:
            new_r, new_c = r + offset_r, c + offset_c
            if 0 <= new_r < grid_height and 0 <= new_c < grid_width:
                new_pixels.add((new_r, new_c))
        
        obj.pixels = new_pixels
        if new_pixels:
            rows, cols = zip(*new_pixels)
            obj.bbox = (min(rows), min(cols), max(rows), max(cols))
        else:
            obj.bbox = (0, 0, 0, 0)

    def _render_objects(self, objects_map: dict, grid_height: int, grid_width: int) -> np.ndarray:
        # Render object pixels into a grid.
        grid = np.zeros((grid_height, grid_width), dtype=int)
        for obj in objects_map.values():
            for r, c in obj.pixels:
                if 0 <= r < grid_height and 0 <= c < grid_width:
                    grid[r, c] = obj.color
        return grid
    
    def _rotate_object(self, obj: Object, angle: int) -> None:
        # Rotate object pixels 90-degree increments around center
        if not obj.pixels:
            return
        
        angle = angle % 360
        if angle == 0:
            return
        
        # Get center
        rows, cols = zip(*obj.pixels)
        center_r = (min(rows) + max(rows)) / 2.0
        center_c = (min(cols) + max(cols)) / 2.0
        
        new_pixels = set()
        for r, c in obj.pixels:
            # Translate to center origin
            dr = r - center_r
            dc = c - center_c
            
            # Rotate
            if angle == 90:
                new_dr, new_dc = -dc, dr
            elif angle == 180:
                new_dr, new_dc = -dr, -dc
            elif angle == 270:
                new_dr, new_dc = dc, -dr
            else:
                new_dr, new_dc = dr, dc
            
            # Translate back
            new_r = int(round(center_r + new_dr))
            new_c = int(round(center_c + new_dc))
            new_pixels.add((new_r, new_c))
        
        obj.pixels = new_pixels
        if new_pixels:
            rows, cols = zip(*new_pixels)
            obj.bbox = (min(rows), min(cols), max(rows), max(cols))

    def _fill_enclosed_zeros(self, grid: np.ndarray, enclosed_color: int, exterior_color: int = -1, mode: str = 'global') -> np.ndarray:
        # Modes:
        # - global: original behavior
        # - local_component_mode: fill each component's enclosed holes by local mode
        if mode in {'local_component_mode', 'local_mode', 'local'}:
            return self._fill_enclosed_by_local_mode(grid)

        # Fill zero-valued connected components:
        # - Components touching border are "exterior"
        # - Remaining components are "enclosed"
        # Non-zero cells are preserved.
        if grid.size == 0:
            return grid.copy()

        rows, cols = grid.shape
        zero_mask = (grid == 0)
        if not np.any(zero_mask):
            return grid.copy()

        exterior = np.zeros_like(zero_mask, dtype=bool)
        queue = deque()

        # Seed border zeros
        for col in range(cols):
            if zero_mask[0, col] and not exterior[0, col]:
                exterior[0, col] = True
                queue.append((0, col))
            if zero_mask[rows - 1, col] and not exterior[rows - 1, col]:
                exterior[rows - 1, col] = True
                queue.append((rows - 1, col))

        for row in range(rows):
            if zero_mask[row, 0] and not exterior[row, 0]:
                exterior[row, 0] = True
                queue.append((row, 0))
            if zero_mask[row, cols - 1] and not exterior[row, cols - 1]:
                exterior[row, cols - 1] = True
                queue.append((row, cols - 1))

        # Flood-fill all exterior zero cells
        while queue:
            row, col = queue.popleft()
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n_row, n_col = row + d_row, col + d_col
                if 0 <= n_row < rows and 0 <= n_col < cols:
                    if zero_mask[n_row, n_col] and not exterior[n_row, n_col]:
                        exterior[n_row, n_col] = True
                        queue.append((n_row, n_col))

        output = grid.copy()
        if exterior_color >= 0:
            output[zero_mask & exterior] = exterior_color
        if enclosed_color >= 0:
            output[zero_mask & (~exterior)] = enclosed_color
        return output
    
    def _flip_object(self, obj: Object, direction: str) -> None:
        # Flip object horizontally or vertically
        if not obj.pixels:
            return
        
        rows, cols = zip(*obj.pixels)
        center_r = (min(rows) + max(rows)) / 2.0
        center_c = (min(cols) + max(cols)) / 2.0
        
        new_pixels = set()
        for r, c in obj.pixels:
            if direction in ['horizontal', 'h']:
                new_r = r
                new_c = int(round(2 * center_c - c))
            elif direction in ['vertical', 'v']:
                new_r = int(round(2 * center_r - r))
                new_c = c
            else:
                new_r, new_c = r, c
            
            new_pixels.add((new_r, new_c))
        
        obj.pixels = new_pixels
        if new_pixels:
            rows, cols = zip(*new_pixels)
            obj.bbox = (min(rows), min(cols), max(rows), max(cols))

    def _rotate_grid(self, grid: np.ndarray, angle: int) -> np.ndarray:
        # Rotate grid around its center in 90-degree steps (counter-clockwise).
        angle = angle % 360
        if angle == 0:
            return grid.copy()
        if angle == 90:
            return np.rot90(grid, k=1)
        if angle == 180:
            return np.rot90(grid, k=2)
        if angle == 270:
            return np.rot90(grid, k=3)
        return grid.copy()

    def _flip_grid(self, grid: np.ndarray, direction: str) -> np.ndarray:
        # Flip grid horizontally (left-right) or vertically (top-bottom).
        if direction in ['horizontal', 'h']:
            return np.fliplr(grid)
        if direction in ['vertical', 'v']:
            return np.flipud(grid)
        return grid.copy()

    def _mirror_vertical(self, grid: np.ndarray) -> np.ndarray:
        # Create vertical symmetry: take left half and mirror to right
        height, width = grid.shape
        mid_col = width // 2
        # Take left half and mirror it
        left_half = grid[:, :mid_col]
        mirrored = np.fliplr(left_half)
        # Create result: left half + mirrored right half
        if width % 2 == 0:
            result = np.hstack([left_half, mirrored])
        else:
            # If odd width, keep center column and mirror around it
            center_col = grid[:, mid_col:mid_col+1]
            result = np.hstack([left_half, center_col, mirrored])
        return result

    def _mirror_horizontal(self, grid: np.ndarray) -> np.ndarray:
        # Create horizontal symmetry: take bottom half and mirror to top
        height, width = grid.shape
        mid_row = height // 2
        # Take bottom half and mirror it
        bottom_half = grid[mid_row:, :]
        mirrored = np.flipud(bottom_half)
        # Create result: mirrored bottom half + original bottom half
        if height % 2 == 0:
            result = np.vstack([mirrored, bottom_half])
        else:
            # If odd height, keep center row and mirror around it
            center_row = grid[mid_row:mid_row+1, :]
            result = np.vstack([mirrored, center_row, bottom_half])
        return result

    def _crop_nonzero_bbox(self, grid: np.ndarray) -> np.ndarray:
        # Crop to the bounding box of all non-zero pixels.
        # print("cropping ----")
        # print(grid)
        nonzero = np.argwhere(grid != 0)
        if nonzero.size == 0:
            return grid.copy()
        min_r, min_c = nonzero.min(axis=0)
        max_r, max_c = nonzero.max(axis=0)
        # print(grid[min_r:max_r + 1, min_c:max_c + 1])
        # print("done cropping ---")
        return grid[min_r:max_r + 1, min_c:max_c + 1]

    def _hollow_object(self, obj: Object) -> None:
        # Remove interior pixels, keep boundary
        if not obj.pixels:
            return

        min_r, min_c, max_r, max_c = obj.bbox
        new_pixels = set()
        for r, c in obj.pixels:
            if r == min_r or r == max_r or c == min_c or c == max_c:
                new_pixels.add((r, c))
        obj.pixels = new_pixels

    def _crop_recolor_by_corner_markers(self, grid: np.ndarray) -> np.ndarray:
        # Assumes input is already cropped to its non-zero bounding box.
        # If all four corners match, remove one border and recolor non-zero interior cells.
        # print("-cornering----")
        # print(grid)
        if grid.size == 0:
            return grid.copy()

        rows, cols = grid.shape
        if rows < 3 or cols < 3:
            return grid.copy()

        corner_color = int(grid[0, 0])
        if corner_color == 0:
            return grid.copy()

        if not (
            int(grid[0, cols - 1]) == corner_color
            and int(grid[rows - 1, 0]) == corner_color
            and int(grid[rows - 1, cols - 1]) == corner_color
        ):
            return grid.copy()

        interior = grid[1:rows - 1, 1:cols - 1].copy()
        if interior.size == 0:
            return grid.copy()

        interior[interior != 0] = corner_color
        # print(interior)
        # print("---done cornering----")
        return interior

    def _span_matching_color_endpoints(self, grid: np.ndarray, include_cols: bool = False) -> np.ndarray:
        # For each row (and optionally column), if the first and last non-zero
        # cells have the same color and the in-between cells contain only 0 or
        # that color, fill the full span with that color.
        if grid.size == 0:
            return grid.copy()

        output = grid.copy()
        base = grid
        rows, cols = base.shape

        for row in range(rows):
            nz_cols = np.flatnonzero(base[row] != 0)
            if nz_cols.size < 2:
                continue
            left = int(nz_cols[0])
            right = int(nz_cols[-1])
            color = int(base[row, left])
            if color == 0 or int(base[row, right]) != color:
                continue
            segment = base[row, left:right + 1]
            if np.all((segment == 0) | (segment == color)):
                output[row, left:right + 1] = color

        if include_cols:
            base_col = output.copy()
            for col in range(cols):
                nz_rows = np.flatnonzero(base_col[:, col] != 0)
                if nz_rows.size < 2:
                    continue
                top = int(nz_rows[0])
                bottom = int(nz_rows[-1])
                color = int(base_col[top, col])
                if color == 0 or int(base_col[bottom, col]) != color:
                    continue
                segment = base_col[top:bottom + 1, col]
                if np.all((segment == 0) | (segment == color)):
                    output[top:bottom + 1, col] = color

        return output

    def _remove_single_pixel_objects(self, grid: np.ndarray) -> np.ndarray:
        # Reuse the object extractor and drop all objects whose area is 1.
        if grid.size == 0:
            return grid.copy()

        extractor = ObjectExtractor(connectivity="4")
        objects = extractor.extract(grid)

        if not objects:
            return grid.copy()

        output = np.zeros_like(grid)
        for obj in objects:
            if getattr(obj, 'is_grid_boundary', False):
                continue
            if len(obj.pixels) <= 1:
                continue
            for row, col in obj.pixels:
                output[row, col] = obj.color

        return output

    def _logical_and_split(self, grid: np.ndarray, separator_color: int | None, output_color: int, logic_op: str = 'AND', split_direction: str = 'AUTO') -> np.ndarray:
        # Split grid on separator line and apply logical operation to the two halves
        # split_direction: 'AUTO' (detect both), 'ROW' (horizontal split only), 'COL' (vertical split only)
        height, width = grid.shape

        def _apply_logic(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
            if logic_op == 'FIT_ADD':
                conflict = (lhs != 0) & (rhs != 0)
                if np.any(conflict):
                    return lhs.copy()
                output = lhs.copy()
                output[rhs != 0] = rhs[rhs != 0]
                return output

            output = np.zeros_like(lhs)
            for r in range(lhs.shape[0]):
                for c in range(lhs.shape[1]):
                    has_lhs = lhs[r, c] != 0
                    has_rhs = rhs[r, c] != 0

                    result = False
                    if logic_op == 'AND':
                        result = has_lhs and has_rhs
                    elif logic_op == 'OR':
                        result = has_lhs or has_rhs
                    elif logic_op == 'XOR':
                        result = has_lhs != has_rhs
                    elif logic_op == 'XNOR':
                        result = has_lhs == has_rhs
                    elif logic_op == 'NAND':
                        result = not (has_lhs and has_rhs)
                    elif logic_op == 'NOR':
                        result = not (has_lhs or has_rhs)

                    if result:
                        output[r, c] = output_color
            return output
        
        split_rows = []
        split_cols = []
        
        if separator_color is not None:
            # Search for rows/cols matching specific separator color
            if split_direction in ['AUTO', 'ROW']:
                split_rows = [r for r in range(height) if np.all(grid[r, :] == separator_color)]
            if split_direction in ['AUTO', 'COL']:
                split_cols = [c for c in range(width) if np.all(grid[:, c] == separator_color)]
        else:
            # Auto-detect: find rows/cols where all pixels are the same (any color)
            if split_direction in ['AUTO', 'ROW']:
                for r in range(height):
                    if np.all(grid[r, :] == grid[r, 0]):
                        split_rows.append(r)
            if split_direction in ['AUTO', 'COL']:
                for c in range(width):
                    if np.all(grid[:, c] == grid[0, c]):
                        split_cols.append(c)

        # For ROW split, find separator that creates equal halves.
        # If none works and no specific separator color is required, fallback to
        # splitting the grid into two contiguous equal halves (no separator row).
        if split_direction in ['AUTO', 'ROW'] and split_rows:
            for split in split_rows:
                top = grid[:split, :]
                bottom = grid[split + 1:, :]
                if top.shape == bottom.shape:
                    return _apply_logic(top, bottom)

        if split_direction in ['AUTO', 'ROW'] and separator_color is None and height % 2 == 0:
            half = height // 2
            top = grid[:half, :]
            bottom = grid[half:, :]
            if top.shape == bottom.shape:
                return _apply_logic(top, bottom)

        # For COL split, find separator that creates equal halves.
        # If none works and no specific separator color is required, fallback to
        # splitting the grid into two contiguous equal halves (no separator col).
        if split_direction in ['AUTO', 'COL'] and split_cols:
            for split in split_cols:
                left = grid[:, :split]
                right = grid[:, split + 1:]
                if left.shape == right.shape:
                    return _apply_logic(left, right)

        if split_direction in ['AUTO', 'COL'] and separator_color is None and width % 2 == 0:
            half = width // 2
            left = grid[:, :half]
            right = grid[:, half:]
            if left.shape == right.shape:
                return _apply_logic(left, right)

        return grid.copy()

    def _recolor_main_by_external_pairs(self, grid: np.ndarray) -> np.ndarray:
        # Find the largest non-zero connected component (the "main" shape),
        # infer recolor rules from external adjacent horizontal color pairs,
        # recolor cells in the main component, and crop to the main bbox.
        if grid.size == 0:
            return grid.copy()

        rows, cols = grid.shape
        nonzero = (grid != 0)
        if not np.any(nonzero):
            return grid.copy()

        visited = np.zeros_like(nonzero, dtype=bool)
        largest_component: set[Tuple[int, int]] = set()

        for r in range(rows):
            for c in range(cols):
                if not nonzero[r, c] or visited[r, c]:
                    continue
                comp: set[Tuple[int, int]] = set()
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.add((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and nonzero[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                if len(comp) > len(largest_component):
                    largest_component = comp

        if not largest_component:
            return grid.copy()

        main_mask = np.zeros_like(nonzero, dtype=bool)
        for r, c in largest_component:
            main_mask[r, c] = True

        # Build recolor map from external horizontal pairs: [new, old].
        recolor_map: dict[int, int] = {}
        for r in range(rows):
            c = 0
            while c < cols - 1:
                left_val = int(grid[r, c])
                right_val = int(grid[r, c + 1])
                if (
                    left_val != 0
                    and right_val != 0
                    and not main_mask[r, c]
                    and not main_mask[r, c + 1]
                ):
                    recolor_map[right_val] = left_val
                    c += 2
                    continue
                c += 1

        # Apply mapping only on the main component.
        output = grid.copy()
        if recolor_map:
            for r, c in largest_component:
                value = int(output[r, c])
                if value in recolor_map:
                    output[r, c] = recolor_map[value]

        main_positions = np.argwhere(main_mask)
        min_r, min_c = main_positions.min(axis=0)
        max_r, max_c = main_positions.max(axis=0)
        return output[min_r:max_r + 1, min_c:max_c + 1]

    def _reflect_pixels_about_center(
        self,
        pixels: Set[Tuple[int, int]],
        center_r: float,
        center_c: float,
        reflect_rows: bool,
        reflect_cols: bool,
    ) -> Set[Tuple[int, int]]:
        reflected: Set[Tuple[int, int]] = set()
        for r, c in pixels:
            new_r = int(round(2 * center_r - r)) if reflect_rows else r
            new_c = int(round(2 * center_c - c)) if reflect_cols else c
            reflected.add((new_r, new_c))
        return reflected

    def _contextual_symmetry_fill(self, grid: np.ndarray) -> np.ndarray:
        # For each hollow container, mirror enclosed objects across the
        # container center. Horizontal vs vertical reflection is determined by
        # the object's position relative to the container center.
        if grid.size == 0:
            return grid.copy()
        
        # print("----")
        # print(grid)

        extractor = ObjectExtractor(connectivity="8")
        objects = [obj for obj in extractor.extract(grid) if not getattr(obj, 'is_grid_boundary', False)]
        if not objects:
            return grid.copy()

        output = grid.copy()

        def _center_of_pixels(pixels: Set[Tuple[int, int]]) -> Tuple[float, float]:
            rows = [r for r, _ in pixels]
            cols = [c for _, c in pixels]
            return (float(sum(rows)) / float(len(rows)), float(sum(cols)) / float(len(cols)))

        for container in objects:
            if not getattr(container, 'is_hollow', False):
                continue

            min_r, min_c, max_r, max_c = container.bbox
            container_center_r = (min_r + max_r) / 2.0
            container_center_c = (min_c + max_c) / 2.0

            enclosed_objects = []
            for obj in objects:
                if obj.id == container.id:
                    continue
                if obj.color == container.color:
                    continue

                obj_min_r, obj_min_c, obj_max_r, obj_max_c = obj.bbox
                if obj_min_r <= min_r or obj_min_c <= min_c or obj_max_r >= max_r or obj_max_c >= max_c:
                    continue
                enclosed_objects.append(obj)

            for obj in enclosed_objects:
                obj_center_r, obj_center_c = _center_of_pixels(obj.pixels)
                row_offset = abs(obj_center_r - container_center_r)
                col_offset = abs(obj_center_c - container_center_c)

                # Mirror on one axis only. Choose the axis where the seed
                # object is farther from the container center.
                if row_offset >= col_offset:
                    reflect_rows = True
                    reflect_cols = False
                else:
                    reflect_rows = False
                    reflect_cols = True

                mirrored_pixels = self._reflect_pixels_about_center(
                    obj.pixels,
                    container_center_r,
                    container_center_c,
                    reflect_rows,
                    reflect_cols,
                )

                for r, c in mirrored_pixels:
                    if min_r <= r <= max_r and min_c <= c <= max_c and output[r, c] == 0:
                        output[r, c] = obj.color

        # print(output)
        # print("---")
        return output

    def _fill_enclosed_by_local_mode(self, grid: np.ndarray) -> np.ndarray:
        # Reuse extracted objects to find hollow components, then fill enclosed
        # cells using the local mode of colors inside each object's bounds.
        if grid.size == 0:
            return grid.copy()

        def _mode_with_tiebreak(values: list[int]) -> int | None:
            if not values:
                return None
            counts: dict[int, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            max_count = max(counts.values())
            candidates = [value for value, count in counts.items() if count == max_count]
            return min(candidates)

        extractor = ObjectExtractor()
        objects = extractor.extract(grid)
        if not objects:
            return grid.copy()

        output = grid.copy()
        for obj in objects:
            if getattr(obj, 'is_grid_boundary', False):
                continue
            if not getattr(obj, 'is_hollow', False):
                continue

            min_r, min_c, max_r, max_c = obj.bbox
            sub = output[min_r:max_r + 1, min_c:max_c + 1]
            sub_h, sub_w = sub.shape
            if sub_h < 3 or sub_w < 3:
                continue

            comp_mask = np.zeros((sub_h, sub_w), dtype=bool)
            for pr, pc in obj.pixels:
                comp_mask[pr - min_r, pc - min_c] = True

            free_mask = ~comp_mask
            outside = np.zeros_like(free_mask, dtype=bool)
            q2 = deque()

            for sc in range(sub_w):
                if free_mask[0, sc] and not outside[0, sc]:
                    outside[0, sc] = True
                    q2.append((0, sc))
                if free_mask[sub_h - 1, sc] and not outside[sub_h - 1, sc]:
                    outside[sub_h - 1, sc] = True
                    q2.append((sub_h - 1, sc))
            for sr in range(sub_h):
                if free_mask[sr, 0] and not outside[sr, 0]:
                    outside[sr, 0] = True
                    q2.append((sr, 0))
                if free_mask[sr, sub_w - 1] and not outside[sr, sub_w - 1]:
                    outside[sr, sub_w - 1] = True
                    q2.append((sr, sub_w - 1))

            while q2:
                cr, cc = q2.popleft()
                for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + d_row, cc + d_col
                    if 0 <= nr < sub_h and 0 <= nc < sub_w and free_mask[nr, nc] and not outside[nr, nc]:
                        outside[nr, nc] = True
                        q2.append((nr, nc))

            enclosed = free_mask & (~outside)
            if not np.any(enclosed):
                continue

            cue_values = [int(v) for v in grid[min_r:max_r + 1, min_c:max_c + 1][enclosed].ravel().tolist() if int(v) != 0]
            fill_color = _mode_with_tiebreak(cue_values) if cue_values else int(obj.color)

            sub[enclosed] = fill_color
            output[min_r:max_r + 1, min_c:max_c + 1] = sub

        return output

    def _spiral_fill(self, grid: np.ndarray, color: int = 3) -> np.ndarray:
        # Draw a one-cell-thick inward spiral with one-cell spacing.
        if grid.size == 0:
            return grid.copy()

        rows, cols = grid.shape
        output = np.zeros_like(grid)

        # Directions: right, down, left, up
        directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
        direction_idx = 0
        row, col = 0, 0

        def _in_bounds(r: int, c: int) -> bool:
            return 0 <= r < rows and 0 <= c < cols

        while True:
            output[row, col] = color

            d_row, d_col = directions[direction_idx]
            next_r, next_c = row + d_row, col + d_col
            far_r, far_c = row + 2 * d_row, col + 2 * d_col

            must_turn = (
                (not _in_bounds(next_r, next_c))
                or output[next_r, next_c] != 0
                or (_in_bounds(far_r, far_c) and output[far_r, far_c] != 0)
            )

            if must_turn:
                direction_idx = (direction_idx + 1) % 4
                d_row, d_col = directions[direction_idx]
                next_r, next_c = row + d_row, col + d_col

                if not _in_bounds(next_r, next_c) or output[next_r, next_c] != 0:
                    break

            row, col = next_r, next_c

        return output
