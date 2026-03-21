# Graph executor: applies transformation programs to input grids using semantic graphs
# Input: (input_grid, semantic_graph, program) -> Output: output_grid (numpy array)

import numpy as np
from copy import deepcopy
from typing import Set, Tuple
from collections import deque
from GraphSemanticNetwork import SemanticGraph
from GraphDSL import TransformProgram, OperationType, Selector
from GraphObjectExtractor import Object


class GraphExecutor:
    # Executes transform programs on grids using semantic graph info
    
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
            # Extract pixels from input grid in this bbox
            min_r, min_c, max_r, max_c = graph_node.bbox
            for r in range(max(0, min_r), min(grid_height, max_r + 1)):
                for c in range(max(0, min_c), min(grid_width, max_c + 1)):
                    if input_grid[r, c] == graph_node.color:
                        obj.pixels.add((r, c))
            objects_map[obj_id] = obj
        
        # Apply each op in seq
        for operation in program.operations:
            # print(f"[GraphExecutor] Applying {operation.type.name}")
            # Find matching objs
            matching_ids = self._select_objects(objects_map, operation.selector, operation.params)
            
            # Apply the op
            if operation.type == OperationType.RECOLOR:
                new_color = operation.params.get('new_color', 0)
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
                grid_state = self._render_objects(objects_map, grid_height, grid_width)
                return self._rotate_grid(grid_state, angle)
            
            elif operation.type == OperationType.FLIP:
                direction = operation.params.get('direction', 'horizontal')
                for obj_id in matching_ids:
                    if obj_id in objects_map:
                        self._flip_object(objects_map[obj_id], direction)

            elif operation.type == OperationType.FLIP_GRID:
                direction = operation.params.get('direction', 'horizontal')
                grid_state = self._render_objects(objects_map, grid_height, grid_width)
                return self._flip_grid(grid_state, direction)
            
            elif operation.type == OperationType.MIRROR_VERTICAL:
                # Mirror around vertical center axis (left-right symmetry)
                grid_state = self._render_objects(objects_map, grid_height, grid_width)
                return self._mirror_vertical(grid_state)
            
            elif operation.type == OperationType.MIRROR_HORIZONTAL:
                # Mirror around horizontal center axis (top-bottom symmetry)
                grid_state = self._render_objects(objects_map, grid_height, grid_width)
                return self._mirror_horizontal(grid_state)
            
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
                grid_state = self._render_objects(objects_map, grid_height, grid_width)
                return self._crop_nonzero_bbox(grid_state)

            elif operation.type == OperationType.FILL_ENCLOSED_ZEROS:
                grid_state = self._render_objects(objects_map, grid_height, grid_width)
                enclosed_color = operation.params.get('enclosed_color', 2)
                exterior_color = operation.params.get('exterior_color', -1)
                return self._fill_enclosed_zeros(grid_state, enclosed_color, exterior_color)

            elif operation.type == OperationType.AND_SPLIT:
                separator_color = operation.params.get('separator_color')
                output_color = operation.params.get('output_color', 2)
                logic_op = operation.params.get('logic_op', 'AND')
                split_direction = operation.params.get('split_direction', 'AUTO')  # AUTO, ROW, or COL
                return self._logical_and_split(input_grid, separator_color, output_color, logic_op, split_direction)
            
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
            for obj_id, obj in objects_map.items():
                if shape == 'arrow' and obj.is_arrow:
                    matching.add(obj_id)
                elif shape == 'triangle' and obj.is_triangle:
                    matching.add(obj_id)
                elif shape == 'circle' and obj.is_closed_shape:
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

    def _fill_enclosed_zeros(self, grid: np.ndarray, enclosed_color: int, exterior_color: int = -1) -> np.ndarray:
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
        nonzero = np.argwhere(grid != 0)
        if nonzero.size == 0:
            return grid.copy()
        min_r, min_c = nonzero.min(axis=0)
        max_r, max_c = nonzero.max(axis=0)
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

    def _logical_and_split(self, grid: np.ndarray, separator_color: int | None, output_color: int, logic_op: str = 'AND', split_direction: str = 'AUTO') -> np.ndarray:
        # Split grid on separator line and apply logical operation to the two halves
        # split_direction: 'AUTO' (detect both), 'ROW' (horizontal split only), 'COL' (vertical split only)
        height, width = grid.shape
        
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

        # For ROW split, find separator that creates equal halves
        if split_direction in ['AUTO', 'ROW'] and split_rows:
            # Try each separator row, prefer one that creates equal-sized halves
            best_split = None
            for split in split_rows:
                top = grid[:split, :]
                bottom = grid[split + 1:, :]
                if top.shape == bottom.shape:
                    best_split = split
                    break  # Found one with equal halves
            if best_split is None:
                best_split = split_rows[0]  # Fallback to first if no equal halves
            
            split = best_split
            top = grid[:split, :]
            bottom = grid[split + 1:, :]
            if top.shape != bottom.shape:
                return grid.copy()
            
            output = np.zeros_like(top)
            for r in range(top.shape[0]):
                for c in range(top.shape[1]):
                    has_top = top[r, c] != 0
                    has_bottom = bottom[r, c] != 0
                    
                    result = False
                    if logic_op == 'AND':
                        result = has_top and has_bottom
                    elif logic_op == 'OR':
                        result = has_top or has_bottom
                    elif logic_op == 'XOR':
                        result = has_top != has_bottom
                    elif logic_op == 'XNOR':
                        result = has_top == has_bottom
                    elif logic_op == 'NAND':
                        result = not (has_top and has_bottom)
                    elif logic_op == 'NOR':
                        result = not (has_top or has_bottom)
                    
                    if result:
                        output[r, c] = output_color
            return output

        # For COL split, find separator that creates equal halves
        if split_direction in ['AUTO', 'COL'] and split_cols:
            # Try each separator col, prefer one that creates equal-sized halves
            best_split = None
            for split in split_cols:
                left = grid[:, :split]
                right = grid[:, split + 1:]
                if left.shape == right.shape:
                    best_split = split
                    break  # Found one with equal halves
            if best_split is None:
                best_split = split_cols[0]  # Fallback to first if no equal halves
            
            split = best_split
            left = grid[:, :split]
            right = grid[:, split + 1:]
            if left.shape != right.shape:
                return grid.copy()
            
            output = np.zeros_like(left)
            for r in range(left.shape[0]):
                for c in range(left.shape[1]):
                    has_left = left[r, c] != 0
                    has_right = right[r, c] != 0
                    
                    result = False
                    if logic_op == 'AND':
                        result = has_left and has_right
                    elif logic_op == 'OR':
                        result = has_left or has_right
                    elif logic_op == 'XOR':
                        result = has_left != has_right
                    elif logic_op == 'XNOR':
                        result = has_left == has_right
                    elif logic_op == 'NAND':
                        result = not (has_left and has_right)
                    elif logic_op == 'NOR':
                        result = not (has_left or has_right)
                    
                    if result:
                        output[r, c] = output_color
            return output

        return grid.copy()
