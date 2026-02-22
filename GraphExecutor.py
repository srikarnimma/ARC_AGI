# Graph executor: applies transformation programs to input grids using semantic graphs
# Input: (input_grid, semantic_graph, program) -> Output: output_grid (numpy array)

import numpy as np
from copy import deepcopy
from typing import Set, Tuple
from GraphSemanticNetwork import SemanticGraph
from GraphDSL import TransformProgram, OperationType, Selector
from GraphObjectExtractor import Object


class GraphExecutor:
    # Executes transformation programs on grids using semantic graph information
    
    def execute(self, input_grid: np.ndarray, graph: SemanticGraph, program: TransformProgram) -> np.ndarray:
        # Execute a transformation program on an input grid
        grid_height, grid_width = input_grid.shape
        # print(f"[GraphExecutor] Executing program with {len(program.operations)} operations on {grid_height}x{grid_width} grid")
        
        # Start with blank output
        output_grid = np.zeros_like(input_grid)
        
        # Create mutable copies of objects from the graph
        # Map object_id -> modified_object
        objects_map = {}
        # print(f"[GraphExecutor] Reconstructing {len(graph.nodes)} objects from graph")
        for obj_id, graph_node in graph.nodes.items():
            # Reconstruct object from graph node
            obj = Object(
                id=obj_id,
                color=graph_node.color,
                pixels=set(),  # Will be inferred from grid
                bbox=graph_node.bbox
            )
            # Extract pixels from input grid in this bounding box
            min_r, min_c, max_r, max_c = graph_node.bbox
            for r in range(max(0, min_r), min(grid_height, max_r + 1)):
                for c in range(max(0, min_c), min(grid_width, max_c + 1)):
                    if input_grid[r, c] == graph_node.color:
                        obj.pixels.add((r, c))
            objects_map[obj_id] = obj
        
        # Apply each operation in sequence
        for operation in program.operations:
            # print(f"[GraphExecutor] Applying {operation.type.name}")
            # Find matching objects
            matching_ids = self._select_objects(objects_map, operation.selector, operation.params)
            
            # Apply operation
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
            
            elif operation.type == OperationType.FLIP:
                direction = operation.params.get('direction', 'horizontal')
                for obj_id in matching_ids:
                    if obj_id in objects_map:
                        self._flip_object(objects_map[obj_id], direction)
            
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
        for obj in objects_map.values():
            for r, c in obj.pixels:
                if 0 <= r < grid_height and 0 <= c < grid_width:
                    output_grid[r, c] = obj.color
        
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
