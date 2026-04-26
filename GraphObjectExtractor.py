# Object extraction utilities
# Takes a grid (numpy array) and extracts objects w features

import numpy as np

from typing import List, Set, Tuple
from scipy import ndimage
from collections import deque


# Represents a single object/component found in a grid
class Object:
    def __init__(self, id: int, color: int, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int], is_grid_boundary: bool = False, area: int = 0, perimeter: int = 0):
        self.id = id
        self.color = color
        self.pixels = pixels
        self.bbox = bbox  # (min_r, min_c, max_r, max_c)
        
        # Flags
        self.is_grid_boundary = is_grid_boundary
        
        # Features
        self.is_closed_shape = False
        self.is_cyclic = False
        self.num_holes = 0
        self.is_hollow = False
        self.is_arrow = False
        self.is_separator = False
        self.is_spiral = False
        self.is_triangle = False
        self.is_grid = False
        self.orientation = "unknown"  # "horizontal", "vertical", "diagonal", "unknown"
        self.area = area
        self.perimeter = perimeter
    
    def __str__(self):
        return f"<Object color={self.color}, pixels={self.pixels}, bbox={self.bbox}>"

# Main extractor class
# Reads a grid and outputs detected objects
class ObjectExtractor:
    def __init__(self, connectivity: str = "8"):
        self.connectivity = connectivity
    
    def extract(self, grid: np.ndarray) -> List[Object]:
        objects: List[Object] = []
        # visited keeps track of EVERY pixel we've assigned to an object
        visited = np.zeros_like(grid, dtype=bool)
        obj_id = 0
        
        height, width = grid.shape

        # Iterate through every cell in the grid
        for r in range(height):
            for c in range(width):
                color = grid[r, c]
                
                # Skip background (0) and pixels we've already assigned to an object
                if color == 0 or visited[r, c]:
                    continue
                
                # Start a BFS to find all connected pixels of the SAME color
                # The BFS will mark all these pixels as 'visited'
                pixels = self._bfs(grid, visited, r, c, color)
                
                if pixels:
                    # Every component found here is a unique object.
                    # If it's one pixel, it's a single-pixel object.
                    # If it's more, it's a large object. 
                    # BFS ensures they don't overlap.
                    obj = self._create_object(obj_id, color, pixels, (height, width))
                    objects.append(obj)
                    obj_id += 1
        
        # Add grid boundary metadata...
        grid_obj = Object(id=-1, color=0, pixels=set(), bbox=(0,0,height-1,width-1), is_grid_boundary=True)
        objects.append(grid_obj)
        
        return objects
    
    def _bfs(self, grid: np.ndarray, visited: np.ndarray, start_r: int, start_c: int, 
             color: int) -> Set[Tuple[int, int]]:
        # BFS to find all pixels of same color connected to start
        pixels = set()
        queue = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        pixels.add((start_r, start_c))
        
        while queue:
            r, c = queue.popleft()
            
            # Get neighbors based on connectivity
            if self.connectivity == "4":
                neighbor_coords = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            else:  # 8-connectivity
                neighbor_coords = [(r-1, c), (r+1, c), (r, c-1), (r, c+1),
                                   (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            
            for nr, nc in neighbor_coords:
                if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and
                    not visited[nr, nc] and grid[nr, nc] == color):
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                    pixels.add((nr, nc))
        
        return pixels
    
    def _create_object(self, obj_id: int, color: int, pixels: Set[Tuple[int, int]], grid_shape: Tuple[int, int]) -> Object:
        # Create object and detect all features
        grid_height, grid_width = grid_shape
        bbox = self._compute_bbox(pixels)
        area = len(pixels)
        perimeter = self._compute_perimeter(pixels)
        
        obj = Object(
            id=obj_id,
            color=color,
            pixels=pixels,
            bbox=bbox,
            area=area,
            perimeter=perimeter
        )
        
        # Detect features
        obj.is_closed_shape = self._is_closed_shape(pixels, bbox)
        obj.is_cyclic = self._has_cycle(pixels)
        obj.is_hollow = self._is_hollow(pixels, bbox)
        obj.num_holes = self._count_holes(pixels, bbox)
        obj.is_arrow = self._is_arrow(pixels, bbox)
        obj.is_separator = self._is_separator(pixels, bbox, grid_height, grid_width)
        obj.is_spiral = self._is_spiral(pixels, bbox)
        obj.is_triangle = self._is_triangle(pixels, bbox)
        obj.is_grid = self._is_grid(pixels, bbox)
        obj.orientation = self._detect_orientation(pixels, bbox)
        # print(f"    bbox: {obj.bbox}, pixels: {obj.pixels}, Features: hollow={obj.is_hollow}, arrow={obj.is_arrow}, spiral={obj.is_spiral}")
        
        return obj
    
    def _compute_bbox(self, pixels: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        # Get bounding box coordinates
        rows, cols = zip(*pixels)
        return (min(rows), min(cols), max(rows), max(cols))
    
    def _compute_perimeter(self, pixels: Set[Tuple[int, int]]) -> int:
        # Count edges touching background
        perimeter = 0
        for r, c in pixels:
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if (nr, nc) not in pixels:
                    perimeter += 1
        return perimeter
    
    def _is_closed_shape(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        min_r, min_c, max_r, max_c = bbox
        h, w = max_r - min_r + 1, max_c - min_c + 1
        
        if len(pixels) < 4 or h < 3 or w < 3:
            return False
            
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if (r, c) not in pixels:
                    # Check if this specific hole is completely sealed
                    if self._is_actually_enclosed(r, c, pixels, bbox):
                        return True
        return False

    def _is_actually_enclosed(self, start_r, start_c, obj_pixels, bbox):
        min_r, min_c, max_r, max_c = bbox
        
        queue = deque([(start_r, start_c)])
        visited = {(start_r, start_c)}
        
        while queue:
            r, c = queue.popleft()
            
            # If we escape bbox, the hole leaks
            if r < min_r or r > max_r or c < min_c or c > max_c:
                return False
            
            # 4-connectivity flood fill
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = r + dr, c + dc
                
                if (nr, nc) not in obj_pixels and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return True

    def _has_cycle(self, pixels: Set[Tuple[int, int]]) -> bool:
        # Graph-theoretic cycle test on 4-neighbor adjacency.
        # For a connected component, cycle exists iff E >= V.
        num_vertices = len(pixels)
        if num_vertices < 4:
            return False

        num_edges = 0
        for r, c in pixels:
            if (r + 1, c) in pixels:
                num_edges += 1
            if (r, c + 1) in pixels:
                num_edges += 1

        return num_edges >= num_vertices
    
    def _is_hollow(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        # Check if interior is mostly empty
        min_r, min_c, max_r, max_c = bbox
        h, w = max_r - min_r + 1, max_c - min_c + 1
        
        # Need minimum size to be hollow
        if h < 3 or w < 3:
            return False
        
        # Check interior pixels - if they're mostly empty, it's hollow
        interior_empty = 0
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if (r, c) not in pixels:
                    interior_empty += 1
        
        interior_total = (h - 2) * (w - 2)
        # print(f"      Interior: {interior_empty}/{interior_total} empty (ratio: {interior_empty/interior_total:.2f})")
        return interior_total > 0 and interior_empty / interior_total > 0.5
    
    def _count_holes(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> int:
        # Count enclosed empty regions (holes)
        min_r, min_c, max_r, max_c = bbox
        interior = set()
        
        # Collect interior empty pixels
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if (r, c) not in pixels:
                    interior.add((r, c))
        
        if not interior:
            return 0
        
        # Count connected components of holes
        visited = set()
        holes = 0
        
        for r, c in interior:
            if (r, c) not in visited:
                # BFS to mark this hole's connected component
                queue = deque([(r, c)])
                visited.add((r, c))
                
                while queue:
                    cr, cc = queue.popleft()
                    for nr, nc in [(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)]:
                        if ((nr, nc) in interior and (nr, nc) not in visited and
                            min_r < nr <= max_r and min_c < nc <= max_c):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                
                holes += 1
        
        # print(f"      Found {holes} hole(s)")
        return holes
    
    def _is_arrow(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        # Detect arrow shape (pointed tip + shaft)
        min_r, min_c, max_r, max_c = bbox
        h, w = max_r - min_r + 1, max_c - min_c + 1
        
        # Arrow should be elongated
        if min(h, w) < 2 or max(h, w) < 4:
            return False
        
        # Check for point extremities (3+ adjacent empty cells around a pixel)
        for r, c in pixels:
            empty_neighbors = 0
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1),
                          (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]:
                if (nr, nc) not in pixels:
                    empty_neighbors += 1
            
            if empty_neighbors >= 6:  # Likely a sharp point
                return True
        
        return False
    
    def _is_separator(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int], grid_height: int, grid_width: int) -> bool:
        # A separator is a thin line that spans most of the grid horizontally or vertically
        min_r, min_c, max_r, max_c = bbox
        h, w = max_r - min_r + 1, max_c - min_c + 1
        
        # Must be thin in one dimension (width 1 or height 1)
        if h != 1 and w != 1:
            return False
        
        # Horizontal line: must span most of the grid width
        if h == 1:
            return w >= grid_width * 0.7
        
        # Vertical line: must span most of the grid height
        if w == 1:
            return h >= grid_height * 0.7
        
        return False
    
    def _is_spiral(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        # Detect spiral: high perimeter/area, balanced aspect ratio, empty interior
        min_r, min_c, max_r, max_c = bbox
        h, w = max_r - min_r + 1, max_c - min_c + 1
        area = len(pixels)
        perimeter = self._compute_perimeter(pixels)
        
        # Spiral needs reasonable bounding box size
        if h < 3 or w < 3:
            return False
        
        # Spiral should be more square-like (not a long thin line)
        aspect_ratio = max(h, w) / min(h, w)
        # print(f"      Spiral aspect ratio: {aspect_ratio:.2f}")
        if aspect_ratio > 3:  # Too elongated, probably just a line
            return False
        
        # Spiral has high perimeter relative to area (winds around)
        perim_area_ratio = perimeter / area if area > 0 else 0
        # print(f"      Spiral perim/area: {perim_area_ratio:.2f}")
        if perim_area_ratio < 0.5:  # Too low, not winding enough
            return False
        
        # Check if it encloses an interior space
        # Count empty pixels inside bbox that are not on boundary
        interior_empty = 0
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if (r, c) not in pixels:
                    interior_empty += 1
        
        interior_total = (h - 2) * (w - 2)
        if interior_total > 0:
            interior_empty_ratio = interior_empty / interior_total
            # print(f"      Spiral interior empty: {interior_empty_ratio:.2f}")
            # Spiral should have significant empty interior (not completely filled)
            if interior_empty_ratio < 0.3:
                return False
        
        return True
    
    def _is_triangle(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        min_r, min_c, max_r, max_c = bbox
        h, w = max_r - min_r + 1, max_c - min_c + 1
        area = len(pixels)
        
        bbox_area = h * w
        if not (0.3 <= area / bbox_area <= 0.75):
            return False

        # Check every row within the bbox
        single_pixel_rows = 0
        for r in range(min_r, max_r + 1):
            if sum(1 for pr, pc in pixels if pr == r) == 1:
                single_pixel_rows += 1
                
        # Check every column within the bbox
        single_pixel_cols = 0
        for c in range(min_c, max_c + 1):
            if sum(1 for pr, pc in pixels if pc == c) == 1:
                single_pixel_cols += 1

        # Up/Down triangles have 1 narrow row (the tip) and 2 narrow columns (the base corners)
        pointing_vertical = (single_pixel_rows == 1 and single_pixel_cols == 2)
        
        # Left/Right triangles have 1 narrow column (the tip) and 2 narrow rows (the base corners)
        pointing_horizontal = (single_pixel_cols == 1 and single_pixel_rows == 2)

        return pointing_vertical or pointing_horizontal
    
    def _is_grid(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        # Detect regular grid pattern
        min_r, min_c, max_r, max_c = bbox
        h, w = max_r - min_r + 1, max_c - min_c + 1
        
        # Grid should be reasonably sized
        if h < 3 or w < 3:
            return False
        
        # Check for regular row/column spacing
        rows_with_pixels = set()
        cols_with_pixels = set()
        
        for r, c in pixels:
            rows_with_pixels.add(r)
            cols_with_pixels.add(c)
        
        # Grid should have multiple rows and columns with similar density
        if len(rows_with_pixels) < 3 or len(cols_with_pixels) < 3:
            return False
        
        # Check regularity: each row/col should have similar pixel counts
        row_counts = [sum(1 for r, c in pixels if r == row) for row in rows_with_pixels]
        col_counts = [sum(1 for r, c in pixels if c == col) for col in cols_with_pixels]
        
        avg_row = np.mean(row_counts)
        avg_col = np.mean(col_counts)
        
        # Regular grid has fairly uniform row/col counts
        row_variance = np.var(row_counts) / (avg_row ** 2) if avg_row > 0 else 0
        col_variance = np.var(col_counts) / (avg_col ** 2) if avg_col > 0 else 0
        # print(f"      Grid variance - row: {row_variance:.3f}, col: {col_variance:.3f}")
        
        return bool(row_variance < 0.3 and col_variance < 0.3)
    
    def _detect_orientation(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> str:
        min_r, min_c, max_r, max_c = bbox
        
        # Count pixels along each edge of the bounding box
        top_count = sum(1 for r, c in pixels if r == min_r)
        bottom_count = sum(1 for r, c in pixels if r == max_r)
        left_count = sum(1 for r, c in pixels if c == min_c)
        right_count = sum(1 for r, c in pixels if c == max_c)
        
        counts = {
            "up": bottom_count,    # Base at bottom -> points up
            "down": top_count,     # Base at top -> points down
            "left": right_count,   # Base at right -> points left
            "right": left_count    # Base at left -> points right
        }
        
        # Manually find the key with the highest value
        best_direction = "unknown"
        max_val = -1
        
        for direction, val in counts.items():
            if val > max_val:
                max_val = val
                best_direction = direction
                
        return best_direction
