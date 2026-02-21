"""
Utilities for object extraction.
Input: grid (numpy array)
Output: dict of color -> list of Objects
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from scipy import ndimage
from collections import deque


@dataclass
class Object:
    """Represents an object/component in a grid."""
    id: int
    color: int
    pixels: Set[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    
    # Flags
    is_grid_boundary: bool = False  # True if this is the grid canvas itself
    
    # Features
    is_closed_shape: bool = False
    num_holes: int = 0
    is_hollow: bool = False
    is_arrow: bool = False
    is_separator: bool = False
    is_spiral: bool = False
    is_triangle: bool = False
    is_grid: bool = False
    orientation: str = "unknown"  # "horizontal", "vertical", "diagonal", "unknown"
    area: int = 0
    perimeter: int = 0


class ObjectExtractor:
    """Input: grid -> Output: Dict[color, List[Object]]"""
    
    def __init__(self, connectivity: str = "8"):
        self.connectivity = connectivity
    
    def extract(self, grid: np.ndarray) -> Dict[int, List[Object]]:
        """Extract objects from grid by color and detect features.
        
        Returns dict with colored objects plus a special 'grid' entry containing
        a grid boundary object that represents the entire canvas dimensions.
        """
        objects_by_color: Dict[int, List[Object]] = {}
        visited = np.zeros_like(grid, dtype=bool)
        obj_id = 0
        
        # Get unique non-zero colors
        colors = np.unique(grid[grid != 0])
        
        for color in colors:
            objects_by_color[color] = []
            color_mask = (grid == color) & ~visited
            
            # BFS to find connected components for this color
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if color_mask[r, c]:
                        pixels = self._bfs(grid, visited, r, c, color)
                        # Only create objects with area >= 2 (filter out single-pixel noise)
                        if pixels and len(pixels) >= 2:
                            obj = self._create_object(obj_id, color, pixels)
                            objects_by_color[color].append(obj)
                            obj_id += 1
        
        # Always add grid boundary object (represents entire grid dimensions)
        grid_height, grid_width = grid.shape
        grid_obj = Object(
            id=-1,  # Special ID for grid boundary
            color=0,  # Grid uses background color
            pixels=set(),  # Empty pixel set for metadata object
            bbox=(0, 0, grid_height - 1, grid_width - 1),
            is_grid_boundary=True,
            area=grid_height * grid_width,
            perimeter=2 * (grid_height + grid_width)
        )
        objects_by_color[0] = [grid_obj]  # Store under color 0
        
        return objects_by_color
    
    def _bfs(self, grid: np.ndarray, visited: np.ndarray, start_r: int, start_c: int, 
             color: int) -> Set[Tuple[int, int]]:
        """BFS to find all pixels of same color connected to start position."""
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
    
    def _create_object(self, obj_id: int, color: int, pixels: Set[Tuple[int, int]]) -> Object:
        """Create object with features detected."""
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
        obj.is_hollow = self._is_hollow(pixels, bbox)
        obj.num_holes = self._count_holes(pixels, bbox)
        obj.is_arrow = self._is_arrow(pixels, bbox)
        obj.is_separator = self._is_separator(pixels, bbox)
        obj.is_spiral = self._is_spiral(pixels, bbox)
        obj.is_triangle = self._is_triangle(pixels, bbox)
        obj.is_grid = self._is_grid(pixels, bbox)
        obj.orientation = self._detect_orientation(pixels, bbox)
        
        return obj
    
    def _compute_bbox(self, pixels: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Compute bounding box."""
        rows, cols = zip(*pixels)
        return (min(rows), min(cols), max(rows), max(cols))
    
    def _compute_perimeter(self, pixels: Set[Tuple[int, int]]) -> int:
        """Rough perimeter estimate: count edges touching background."""
        perimeter = 0
        for r, c in pixels:
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if (nr, nc) not in pixels:
                    perimeter += 1
        return perimeter
    
    def _is_closed_shape(self, pixels: Set[Tuple[int, int]]) -> bool:
        """Check if shape forms a closed contour (perimeter vs area ratio)."""
        if len(pixels) < 4:
            return False
        # Rough heuristic: closed shapes have perimeter roughly sqrt(area)
        perimeter = self._compute_perimeter(pixels)
        area = len(pixels)
        # Closed shapes typically have P^2 ~ 4*pi*A, so P^2/A ~ 12-15 for circle
        return (perimeter * perimeter) / area > 10 and (perimeter * perimeter) / area < 25
    
    def _is_hollow(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Check if shape has an empty interior."""
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
        return interior_total > 0 and interior_empty / interior_total > 0.5
    
    def _count_holes(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> int:
        """Count connected empty regions completely enclosed by object."""
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
        
        return holes
    
    def _is_arrow(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Detect if shape looks like arrow (pointed tip + shaft)."""
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
    
    def _is_separator(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Detect if shape is a line/separator (very thin)."""
        min_r, min_c, max_r, max_c = bbox
        h, w = max_r - min_r + 1, max_c - min_c + 1
        
        # Separator is very thin relative to length
        if h == 1 or w == 1:
            return True
        
        if min(h, w) <= 2 and max(h, w) >= 3:
            return True
        
        return False
    
    def _is_spiral(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Detect if shape forms a spiral pattern.
        
        Spiral heuristics:
        - High perimeter-to-area ratio (winds around)
        - Not a simple separator (has width/height balance)
        - Encloses an interior region
        """
        min_r, min_c, max_r, max_c = bbox
        h, w = max_r - min_r + 1, max_c - min_c + 1
        area = len(pixels)
        perimeter = self._compute_perimeter(pixels)
        
        # Spiral needs reasonable bounding box size
        if h < 3 or w < 3:
            return False
        
        # Spiral should be more square-like (not a long thin line)
        aspect_ratio = max(h, w) / min(h, w)
        if aspect_ratio > 3:  # Too elongated, probably just a line
            return False
        
        # Spiral has high perimeter relative to area (winds around)
        perim_area_ratio = perimeter / area if area > 0 else 0
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
            # Spiral should have significant empty interior (not completely filled)
            if interior_empty_ratio < 0.3:
                return False
        
        return True
    
    def _is_triangle(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> bool:
        """Rough detection: shape with one point and wide base."""
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
        """Detect if shape forms a regular grid pattern."""
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
        
        return bool(row_variance < 0.3 and col_variance < 0.3)
    
    def _detect_orientation(self, pixels: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> str:
        """Detect orientation: horizontal, vertical, or diagonal."""
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
