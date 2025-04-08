import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Deque
from collections import deque  # Import deque for BFS
import numba  # Import Numba
import logging  # Import logging

from config import EnvConfig
from .triangle import Triangle
from .shape import Shape

logger = logging.getLogger(__name__)  # Add logger


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig):
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.triangles: List[List[Triangle]] = self._create(env_config)
        self._link_neighbors()

        # --- Internal NumPy arrays for Numba ---
        # Initialize these arrays based on the created grid
        self._occupied_np = np.array(
            [[tri.is_occupied for tri in row] for row in self.triangles], dtype=np.bool_
        )
        self._death_np = np.array(
            [[tri.is_death for tri in row] for row in self.triangles], dtype=np.bool_
        )
        # --- End Internal NumPy arrays ---

        # Store potential lines as frozensets for easy hashing/set operations
        self.potential_lines: Set[frozenset[Triangle]] = set()
        # Index mapping Triangle object to the set of lines it belongs to
        self._triangle_to_lines_map: Dict[Triangle, Set[frozenset[Triangle]]] = {}
        self._initialize_lines_and_index()  # This needs to run after _create and _link_neighbors

    def _create(self, env_config: EnvConfig) -> List[List[Triangle]]:
        """
        Initializes the grid with playable and death cells.
        The playable area defined by cols_per_row is further reduced by
        making the first and last cell of that range into death cells.
        """
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]
        if len(cols_per_row) != self.rows:
            raise ValueError("cols_per_row length mismatch")
        if max(cols_per_row) > self.cols:
            raise ValueError("cols_per_row exceeds EnvConfig.COLS")

        grid: List[List[Triangle]] = []
        for r in range(self.rows):
            row_tris: List[Triangle] = []
            intended_playable_width = cols_per_row[r]
            total_padding = self.cols - intended_playable_width
            pad_l = total_padding // 2
            pad_r = self.cols - (total_padding - pad_l)
            playable_start_col = pad_l + 1
            playable_end_col = pad_r - 1

            for c in range(self.cols):
                is_playable = (
                    intended_playable_width > 2
                    and playable_start_col <= c < playable_end_col
                )
                is_death = not is_playable
                is_up = (r + c) % 2 == 0
                row_tris.append(Triangle(r, c, is_up=is_up, is_death=is_death))
            grid.append(row_tris)
        return grid

    def _link_neighbors(self) -> None:
        """Sets neighbor references for each triangle."""
        for r in range(self.rows):
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if self.valid(r, c - 1):
                    tri.neighbor_left = self.triangles[r][c - 1]
                if self.valid(r, c + 1):
                    tri.neighbor_right = self.triangles[r][c + 1]
                nr, nc = (r + 1, c) if tri.is_up else (r - 1, c)
                if self.valid(nr, nc):
                    tri.neighbor_vert = self.triangles[nr][nc]

    def _get_line_neighbors(self, tri: Triangle, direction: str) -> List[Triangle]:
        """Helper to get relevant neighbors for line tracing in a specific direction."""
        neighbors = []
        if direction == "horizontal":
            if tri.neighbor_left:
                neighbors.append(tri.neighbor_left)
            if tri.neighbor_right:
                neighbors.append(tri.neighbor_right)
        elif direction == "diag1":
            if tri.is_up:
                if tri.neighbor_left and not tri.neighbor_left.is_up:
                    neighbors.append(tri.neighbor_left)
                if tri.neighbor_vert and not tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
            else:
                if tri.neighbor_right and tri.neighbor_right.is_up:
                    neighbors.append(tri.neighbor_right)
                if tri.neighbor_vert and tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
        elif direction == "diag2":
            if tri.is_up:
                if tri.neighbor_right and not tri.neighbor_right.is_up:
                    neighbors.append(tri.neighbor_right)
                if tri.neighbor_vert and not tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
            else:
                if tri.neighbor_left and tri.neighbor_left.is_up:
                    neighbors.append(tri.neighbor_left)
                if tri.neighbor_vert and tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
        return [n for n in neighbors if not n.is_death]

    def _initialize_lines_and_index(self) -> None:
        """
        Identifies all sets of playable triangles forming potential lines
        by tracing connections along horizontal and diagonal axes using BFS.
        Populates self.potential_lines and self._triangle_to_lines_map.
        """
        self.potential_lines = set()
        self._triangle_to_lines_map = {}
        visited_in_direction: Dict[str, Set[Triangle]] = {
            "horizontal": set(),
            "diag1": set(),
            "diag2": set(),
        }
        min_line_length = 3

        for r in range(self.rows):
            for c in range(self.cols):
                start_node = self.triangles[r][c]
                if start_node.is_death:
                    continue

                for direction in ["horizontal", "diag1", "diag2"]:
                    if start_node not in visited_in_direction[direction]:
                        current_line: Set[Triangle] = set()
                        queue: Deque[Triangle] = deque([start_node])
                        visited_this_bfs: Set[Triangle] = {start_node}

                        while queue:
                            tri = queue.popleft()
                            if not tri.is_death:
                                current_line.add(tri)
                            visited_in_direction[direction].add(tri)

                            neighbors = self._get_line_neighbors(tri, direction)
                            for neighbor in neighbors:
                                if neighbor not in visited_this_bfs:
                                    visited_this_bfs.add(neighbor)
                                    queue.append(neighbor)

                        if len(current_line) >= min_line_length:
                            line_frozenset = frozenset(current_line)
                            self.potential_lines.add(line_frozenset)
                            for tri_in_line in current_line:
                                if tri_in_line not in self._triangle_to_lines_map:
                                    self._triangle_to_lines_map[tri_in_line] = set()
                                self._triangle_to_lines_map[tri_in_line].add(
                                    line_frozenset
                                )

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(self, shape: Shape, r: int, c: int) -> bool:
        """Checks if a shape can be placed at the target location."""
        for dr, dc, is_up_shape in shape.triangles:
            nr, nc = r + dr, c + dc
            if not self.valid(nr, nc):
                return False
            # Use pre-computed numpy arrays for faster checks
            if (
                self._death_np[nr, nc]
                or self._occupied_np[nr, nc]
                or (self.triangles[nr][nc].is_up != is_up_shape)
            ):
                return False
        return True

    def place(self, shape: Shape, r: int, c: int) -> Set[Triangle]:
        """
        Places a shape onto the grid (assumes can_place was checked).
        Updates internal numpy arrays and returns the set of occupied Triangles.
        """
        newly_occupied: Set[Triangle] = set()
        for dr, dc, _ in shape.triangles:
            nr, nc = r + dr, c + dc
            # Check validity again just in case, though can_place should precede
            if self.valid(nr, nc):
                tri = self.triangles[nr][nc]
                # Check using numpy arrays first for speed
                if not self._death_np[nr, nc] and not self._occupied_np[nr, nc]:
                    # Update Triangle object
                    tri.is_occupied = True
                    tri.color = shape.color
                    # Update internal numpy array
                    self._occupied_np[nr, nc] = True
                    newly_occupied.add(tri)
        return newly_occupied

    def clear_lines(
        self, newly_occupied_triangles: Optional[Set[Triangle]] = None
    ) -> Tuple[int, int, List[Tuple[int, int]]]:
        """
        Checks and clears completed lines using the optimized index approach.
        Updates internal numpy arrays.
        Returns lines cleared count, total triangles cleared count, and their coordinates.
        """
        lines_to_check: Set[frozenset[Triangle]] = set()
        if newly_occupied_triangles:
            for tri in newly_occupied_triangles:
                if tri in self._triangle_to_lines_map:
                    lines_to_check.update(self._triangle_to_lines_map[tri])
        else:
            lines_to_check = self.potential_lines

        cleared_tris_total: Set[Triangle] = set()
        lines_cleared_count = 0

        for line_set in lines_to_check:
            if not line_set:
                continue
            # Check occupancy using the numpy array for speed
            if all(self._occupied_np[tri.row, tri.col] for tri in line_set):
                if not line_set.issubset(cleared_tris_total):
                    cleared_tris_total.update(line_set)
                    lines_cleared_count += 1

        tris_cleared_count = 0
        coords: List[Tuple[int, int]] = []
        if not cleared_tris_total:
            return 0, 0, []

        for tri in cleared_tris_total:
            # Check using numpy array first
            if (
                not self._death_np[tri.row, tri.col]
                and self._occupied_np[tri.row, tri.col]
            ):
                tris_cleared_count += 1
                # Update Triangle object
                tri.is_occupied = False
                tri.color = None
                # Update internal numpy array
                self._occupied_np[tri.row, tri.col] = False
                coords.append((tri.row, tri.col))

        return lines_cleared_count, tris_cleared_count, coords

    @staticmethod
    @numba.njit(cache=True)
    def _numba_get_column_heights(
        rows: int, cols: int, is_occupied: np.ndarray, is_death: np.ndarray
    ) -> np.ndarray:
        """Numba-accelerated calculation of column heights."""
        heights = np.zeros(cols, dtype=np.int32)
        for c in range(cols):
            col_max_r = -1
            for r in range(rows):
                if not is_death[r, c] and is_occupied[r, c]:
                    col_max_r = max(col_max_r, r)
            heights[c] = col_max_r + 1
        return heights

    def get_column_heights(self) -> List[int]:
        """Calculates the height of occupied cells in each column using Numba and internal arrays."""
        # Use the internal numpy arrays directly
        heights_np = self._numba_get_column_heights(
            self.rows, self.cols, self._occupied_np, self._death_np
        )
        return heights_np.tolist()

    def get_max_height(self) -> int:
        """Returns the maximum height across all columns."""
        heights = self.get_column_heights()
        return max(heights) if heights else 0

    @staticmethod
    @numba.njit(cache=True)
    def _numba_get_bumpiness(heights: np.ndarray) -> int:
        """Numba-accelerated calculation of bumpiness."""
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def get_bumpiness(self) -> int:
        """Calculates the total absolute difference between adjacent column heights using Numba."""
        heights_np = np.array(self.get_column_heights(), dtype=np.int32)
        return self._numba_get_bumpiness(heights_np)

    @staticmethod
    @numba.njit(cache=True)
    def _numba_count_holes(
        rows: int,
        cols: int,
        heights: np.ndarray,
        is_occupied: np.ndarray,
        is_death: np.ndarray,
    ) -> int:
        """Numba-accelerated calculation of holes."""
        holes = 0
        for c in range(cols):
            height = heights[c]
            if height > 0:
                for r in range(height):  # Iterate up to height-1
                    if not is_death[r, c] and not is_occupied[r, c]:
                        holes += 1
        return holes

    def count_holes(self) -> int:
        """Counts empty, non-death cells below the highest occupied cell in the same column using Numba."""
        heights_np = np.array(self.get_column_heights(), dtype=np.int32)
        # Use the internal numpy arrays directly
        return self._numba_count_holes(
            self.rows, self.cols, heights_np, self._occupied_np, self._death_np
        )

    def get_feature_matrix(self) -> np.ndarray:
        """Returns the grid state as a 2-channel numpy array (Occupancy, Orientation)."""
        # Use internal _occupied_np for the first channel
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        grid_state[0, :, :] = self._occupied_np.astype(np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if not self._death_np[r, c]:
                    grid_state[1, r, c] = 1.0 if self.triangles[r][c].is_up else -1.0
        return grid_state

    def get_color_data(self) -> List[List[Optional[Tuple[int, int, int]]]]:
        """Returns a 2D list of colors for occupied cells."""
        color_data = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                # Check internal numpy array first
                if self._occupied_np[r, c] and not self._death_np[r, c]:
                    color_data[r][c] = self.triangles[r][c].color
        return color_data

    def get_death_data(self) -> np.ndarray:
        """Returns a boolean numpy array indicating death cells (uses internal array)."""
        return self._death_np.copy()  # Return a copy to prevent external modification

    def deepcopy_grid(self) -> "Grid":
        """Creates a deep copy of the grid, including Triangle objects and numpy arrays."""
        new_grid = Grid.__new__(Grid)  # Create instance without calling __init__
        new_grid.rows = self.rows
        new_grid.cols = self.cols

        # Deep copy the list of lists of Triangles
        new_grid.triangles = [[tri.copy() for tri in row] for row in self.triangles]

        # Re-link neighbors within the new grid
        new_grid._link_neighbors()

        # Copy the numpy arrays
        new_grid._occupied_np = self._occupied_np.copy()
        new_grid._death_np = self._death_np.copy()

        # Rebuild the lines index based on the *new* triangle objects
        new_grid._initialize_lines_and_index()

        return new_grid
