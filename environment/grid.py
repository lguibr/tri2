# File: environment/grid.py
import numpy as np
from typing import List, Tuple, Set, Dict, Optional

from config import EnvConfig
from .triangle import Triangle
from .shape import Shape


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig):
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.triangles: List[List[Triangle]] = self._create(env_config)
        self._link_neighbors()
        self.potential_lines: List[Set[Triangle]] = self._identify_playable_lines()

    def _create(self, env_config: EnvConfig) -> List[List[Triangle]]:
        """Initializes the grid with playable and death cells."""
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]
        if len(cols_per_row) != self.rows:
            raise ValueError("cols_per_row length mismatch")
        if max(cols_per_row) > self.cols:
            raise ValueError("cols_per_row exceeds EnvConfig.COLS")

        grid: List[List[Triangle]] = []
        for r in range(self.rows):
            row_tris: List[Triangle] = []
            playable = cols_per_row[r]
            pad_l = (self.cols - playable) // 2 + 1
            pad_r = pad_l + playable - 2
            for c in range(self.cols):
                is_death = (
                    (c < pad_l) or (c >= pad_r and pad_r > pad_l) or (playable <= 2)
                )
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

    def _identify_playable_lines(self) -> List[Set[Triangle]]:
        """Identifies all sets of playable triangles forming potential lines."""
        lines: List[Set[Triangle]] = []
        # 1. Horizontal Lines
        for r in range(self.rows):
            segment: List[Triangle] = []
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if not tri.is_death:
                    segment.append(tri)
                else:
                    if segment:
                        lines.append(set(segment))
                    segment = []
            if segment:
                lines.append(set(segment))

        # 2. Diagonal Lines
        diag_tlbr = self._get_single_diagonal_lines(
            lambda r, c: c - r, lambda k, r: k + r
        )
        diag_trbl = self._get_single_diagonal_lines(
            lambda r, c: r + c, lambda k, r: k - r
        )
        for diag_dict in [diag_tlbr, diag_trbl]:
            k_values = sorted(diag_dict.keys())
            for i in range(len(k_values) - 1):
                k1, k2 = k_values[i], k_values[i + 1]
                if k2 == k1 + 1:
                    combined = diag_dict[k1].union(diag_dict[k2])
                    if combined:
                        lines.append(combined)
        return [line for line in lines if line]

    def _get_single_diagonal_lines(self, k_func, c_func) -> Dict[int, Set[Triangle]]:
        """Helper to find single-thickness diagonal lines."""
        single_lines: Dict[int, Set[Triangle]] = {}
        min_k, max_k = float("inf"), float("-inf")
        for r in range(self.rows):
            for c in range(self.cols):
                if self.valid(r, c) and not self.triangles[r][c].is_death:
                    k = k_func(r, c)
                    min_k, max_k = min(min_k, k), max(max_k, k)
        if min_k > max_k:
            return {}

        for k in range(min_k, max_k + 1):
            segment: List[Triangle] = []
            for r in range(self.rows):
                c = c_func(k, r)
                if self.valid(r, c):
                    tri = self.triangles[r][c]
                    if not tri.is_death:
                        segment.append(tri)
                    else:
                        if segment:
                            single_lines.setdefault(k, set()).update(segment)
                        segment = []
                elif segment:
                    single_lines.setdefault(k, set()).update(segment)
                    segment = []
            if segment:
                single_lines.setdefault(k, set()).update(segment)
        return single_lines

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(self, shape: Shape, r: int, c: int) -> bool:
        """Checks if a shape can be placed at the target location."""
        for dr, dc, is_up_shape in shape.triangles:
            nr, nc = r + dr, c + dc
            if not self.valid(nr, nc):
                return False
            tri = self.triangles[nr][nc]
            if tri.is_death or tri.is_occupied or (tri.is_up != is_up_shape):
                return False
        return True

    def place(self, shape: Shape, r: int, c: int) -> None:
        """Places a shape onto the grid (assumes can_place was checked)."""
        for dr, dc, _ in shape.triangles:
            nr, nc = r + dr, c + dc
            if self.valid(nr, nc):
                tri = self.triangles[nr][nc]
                if not tri.is_death and not tri.is_occupied:
                    tri.is_occupied = True
                    tri.color = shape.color

    def clear_lines(self) -> Tuple[int, int, List[Tuple[int, int]]]:
        """Checks and clears completed lines. Returns lines, tris cleared, coords."""
        cleared_tris: Set[Triangle] = set()
        lines_cleared = 0
        for line_set in self.potential_lines:
            if line_set and all(tri.is_occupied for tri in line_set):
                cleared_tris.update(line_set)
                lines_cleared += 1

        tris_count = 0
        coords: List[Tuple[int, int]] = []
        if not cleared_tris:
            return 0, 0, []
        for tri in cleared_tris:
            if not tri.is_death and tri.is_occupied:
                tris_count += 1
                tri.is_occupied = False
                tri.color = None
                coords.append((tri.row, tri.col))
        return lines_cleared, tris_count, coords

    def get_column_heights(self) -> List[int]:
        """Calculates the height of occupied cells in each column."""
        heights = [0] * self.cols
        for c in range(self.cols):
            for r in range(self.rows - 1, -1, -1):
                if (
                    self.triangles[r][c].is_occupied
                    and not self.triangles[r][c].is_death
                ):
                    heights[c] = r + 1
                    break
        return heights

    def get_max_height(self) -> int:
        """Returns the maximum height across all columns."""
        heights = self.get_column_heights()
        return max(heights) if heights else 0

    def get_bumpiness(self) -> int:
        """Calculates the total absolute difference between adjacent column heights."""
        heights = self.get_column_heights()
        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

    def count_holes(self) -> int:
        """Counts empty, non-death cells below an occupied cell in the same column."""
        holes = 0
        for c in range(self.cols):
            occupied_above = False
            for r in range(self.rows - 1, -1, -1):
                tri = self.triangles[r][c]
                if tri.is_death:
                    occupied_above = False
                    continue
                if tri.is_occupied:
                    occupied_above = True
                elif not tri.is_occupied and occupied_above:
                    holes += 1
        return holes

    def get_feature_matrix(self) -> np.ndarray:
        """Returns the grid state as a 2-channel numpy array (Occupancy, Orientation)."""
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if not tri.is_death:
                    grid_state[0, r, c] = 1.0 if tri.is_occupied else 0.0
                    grid_state[1, r, c] = 1.0 if tri.is_up else 0.0
        return grid_state

    def get_color_data(self) -> List[List[Optional[Tuple[int, int, int]]]]:
        """Returns a 2D list of colors for occupied cells."""
        color_data = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if tri.is_occupied and not tri.is_death:
                    color_data[r][c] = tri.color
        return color_data

    def get_death_data(self) -> np.ndarray:
        """Returns a boolean numpy array indicating death cells."""
        death_data = np.zeros((self.rows, self.cols), dtype=bool)
        for r in range(self.rows):
            for c in range(self.cols):
                death_data[r, c] = self.triangles[r][c].is_death
        return death_data
