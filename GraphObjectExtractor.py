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

# Main extractor class
# Reads a grid and outputs detected objects
class ObjectExtractor:
    def __init__(self, connectivity: str = "8"):
        self.connectivity = connectivity
    
    def extract(self, grid: np.ndarray) -> List[Object]:
        # Extract objects from grid by color & detect features
        # Returns all objs (colored + grid boundary)
        objects: List[Object] = []
        visited = np.zeros_like(grid, dtype=bool)
        obj_id = 0
        
        # Get unique non-zero colors
        colors = np.unique(grid[grid != 0])
        # print(f"Found colors: {colors}")
        
        for color in colors:
            color_mask = (grid == color) & ~visited
            components: List[Set[Tuple[int, int]]] = []

            # BFS to find connected components for this color
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if color_mask[r, c]:
                        pixels = self._bfs(grid, visited, r, c, color)
                        if pixels:
                            components.append(pixels)

            has_large_component = any(len(pixels) >= 2 for pixels in components)

            for pixels in components:
                # Filter out isolated single-pixel noise only when larger components exist.
                if len(pixels) < 2 and has_large_component:
                    continue
                # print(f"  Color {color}: found component with {len(pixels)} pixels")
                grid_height, grid_width = grid.shape
                obj = self._create_object(obj_id, color, pixels, (grid_height, grid_width))
                objects.append(obj)
                obj_id += 1
        
        # Always add grid boundary object (metadata for entire grid)
        grid_height, grid_width = grid.shape
        grid_obj = Object(
            id=-1,
            color=0,
            pixels=set(),
            bbox=(0, 0, grid_height - 1, grid_width - 1),
            is_grid_boundary=True,
            area=grid_height * grid_width,
            perimeter=2 * (grid_height + grid_width)
        )
        objects.append(grid_obj)
        # print(f"Extracted {len(objects)-1} colored objects + grid boundary")
        
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
        obj.is_closed_shape = self._is_closed_shape(pixels)
        obj.is_cyclic = self._has_cycle(pixels)
        obj.is_hollow = self._is_hollow(pixels, bbox)
        obj.num_holes = self._count_holes(pixels, bbox)
        obj.is_arrow = self._is_arrow(pixels, bbox)
        obj.is_separator = self._is_separator(pixels, bbox, grid_height, grid_width)
        obj.is_spiral = self._is_spiral(pixels, bbox)
        obj.is_triangle = self._is_triangle(pixels, bbox)
        obj.is_grid = self._is_grid(pixels, bbox)
        obj.orientation = self._detect_orientation(pixels, bbox)
        # print(f"    Features: hollow={obj.is_hollow}, arrow={obj.is_arrow}, spiral={obj.is_spiral}")
        
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
    
    def _is_closed_shape(self, pixels: Set[Tuple[int, int]]) -> bool:
        # Check for closed contour using perimeter/area ratio
        if len(pixels) < 4:
            return False
        # Rough heuristic: closed shapes have perimeter roughly sqrt(area)
        perimeter = self._compute_perimeter(pixels)
        area = len(pixels)
        # Closed shapes typically have P^2 ~ 4*pi*A, so P^2/A ~ 12-15 for circle
        return (perimeter * perimeter) / area > 10 and (perimeter * perimeter) / area < 25

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
        # Detect triangle: one point + wide base
        min_r, min_c, max_r, max_c = bbox
        h, w = max_r - min_r + 1, max_c - min_c + 1
        area = len(pixels)
        
        # Triangle-like: compact and roughly triangular area
        bbox_area = h * w
        if area < bbox_area * 0.3 or area > bbox_area * 0.7:
            return False
        
        # Has significant perimeter relative to area (not too circular)
        perimeter = self._compute_perimeter(pixels)
        if (perimeter * perimeter) / area < 10 or (perimeter * perimeter) / area > 20:
            return False
        
        # Check for point at top/bottom
        top_row_pixels = sum(1 for r, c in pixels if r == min_r)
        bot_row_pixels = sum(1 for r, c in pixels if r == max_r)
        mid_row_pixels = sum(1 for r, c in pixels if min_r < r < max_r)
        
        if (top_row_pixels == 1 or bot_row_pixels == 1) and mid_row_pixels > top_row_pixels:
            return True
        
        return False
    
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
        # Detect orientation (horizontal/vertical/diagonal)
        min_r, min_c, max_r, max_c = bbox
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        
        # Check for main diagonal
        diag_count = sum(1 for r, c in pixels if (r - min_r) == (c - min_c))
        anti_diag_count = sum(1 for r, c in pixels if (r - min_r) == (max_c - c))
        
        area = len(pixels)
        
        if max(diag_count, anti_diag_count) > area * 0.6:
            return "diagonal"
        
        if h > w * 1.5:
            return "vertical"
        elif w > h * 1.5:
            return "horizontal"
        else:
            return "unknown"
