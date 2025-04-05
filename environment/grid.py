# File: environment/grid.py
import numpy as np
from typing import List, Tuple, Optional
from .triangle import Triangle
from .shape import Shape
from config import EnvConfig


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig):
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.triangles: List[List[Triangle]] = []
        self._create()

    def _create(self) -> None:
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]

        if len(cols_per_row) != self.rows:
            raise ValueError(
                f"Grid._create error: Length of cols_per_row ({len(cols_per_row)}) must match EnvConfig.ROWS ({self.rows})"
            )
        if max(cols_per_row) > self.cols:
            raise ValueError(
                f"Grid._create error: Max playable columns ({max(cols_per_row)}) exceeds EnvConfig.COLS ({self.cols})"
            )

        self.triangles = []
        for r in range(self.rows):
            rowt = []
            base_playable_cols = cols_per_row[r]

            if base_playable_cols <= 0:
                initial_death_cols_left = self.cols
            elif base_playable_cols >= self.cols:
                initial_death_cols_left = 0
            else:
                initial_death_cols_left = (self.cols - base_playable_cols) // 2
            initial_first_death_col_right = initial_death_cols_left + base_playable_cols

            adjusted_death_cols_left = initial_death_cols_left + 1
            adjusted_first_death_col_right = initial_first_death_col_right - 1

            for c in range(self.cols):
                is_death_cell = (
                    (c < adjusted_death_cols_left)
                    or (
                        c >= adjusted_first_death_col_right
                        and adjusted_first_death_col_right > adjusted_death_cols_left
                    )
                    or (base_playable_cols <= 2)
                )

                is_up_cell = (r + c) % 2 == 0
                tri = Triangle(r, c, is_up=is_up_cell, is_death=is_death_cell)
                rowt.append(tri)
            self.triangles.append(rowt)

    def valid(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(self, shp: Shape, rr: int, cc: int) -> bool:
        for dr, dc, up in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if not self.valid(nr, nc):
                return False
            if not (
                0 <= nr < len(self.triangles) and 0 <= nc < len(self.triangles[nr])
            ):
                return False
            tri = self.triangles[nr][nc]
            if tri.is_death or tri.is_occupied or (tri.is_up != up):
                return False
        return True

    def place(self, shp: Shape, rr: int, cc: int) -> None:
        for dr, dc, _ in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if self.valid(nr, nc):
                if not (
                    0 <= nr < len(self.triangles) and 0 <= nc < len(self.triangles[nr])
                ):
                    continue
                tri = self.triangles[nr][nc]
                if not tri.is_death and not tri.is_occupied:
                    tri.is_occupied = True
                    tri.color = shp.color

    def clear_filled_rows(self) -> Tuple[int, int, List[Tuple[int, int]]]:
        lines_cleared = 0
        triangles_cleared = 0
        rows_to_clear_indices = []
        cleared_triangles_coords: List[Tuple[int, int]] = []

        for r in range(self.rows):
            if not (0 <= r < len(self.triangles)):
                continue
            rowt = self.triangles[r]
            is_row_full = True
            num_placeable_triangles_in_row = 0
            for t in rowt:
                if not t.is_death:
                    num_placeable_triangles_in_row += 1
                    if not t.is_occupied:
                        is_row_full = False
                        break

            if is_row_full and num_placeable_triangles_in_row > 0:
                rows_to_clear_indices.append(r)
                lines_cleared += 1

        for r_idx in rows_to_clear_indices:
            if not (0 <= r_idx < len(self.triangles)):
                continue
            for t in self.triangles[r_idx]:
                if not t.is_death and t.is_occupied:
                    triangles_cleared += 1
                    t.is_occupied = False
                    t.color = None
                    cleared_triangles_coords.append((r_idx, t.col))

        return lines_cleared, triangles_cleared, cleared_triangles_coords

    def get_column_heights(self) -> List[int]:
        heights = [0] * self.cols
        for c in range(self.cols):
            for r in range(self.rows - 1, -1, -1):
                if (
                    0 <= r < len(self.triangles)
                    and 0 <= c < len(self.triangles[r])
                    and self.triangles[r][c].is_occupied
                    and not self.triangles[r][c].is_death
                ):
                    heights[c] = r + 1
                    break
        return heights

    def get_max_height(self) -> int:
        """Calculates the highest occupied row index (0-based) + 1."""
        heights = self.get_column_heights()
        return max(heights) if heights else 0

    def get_bumpiness(self) -> int:
        """Calculates the sum of absolute height differences between adjacent columns."""
        heights = self.get_column_heights()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def count_holes(self) -> int:
        holes = 0
        for c in range(self.cols):
            occupied_above = False
            for r in range(self.rows):
                if not (
                    0 <= r < len(self.triangles) and 0 <= c < len(self.triangles[r])
                ):
                    continue
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
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if not (
                    0 <= r < len(self.triangles) and 0 <= c < len(self.triangles[r])
                ):
                    continue
                t = self.triangles[r][c]
                if not t.is_death:
                    grid_state[0, r, c] = 1.0 if t.is_occupied else 0.0
                    grid_state[1, r, c] = 1.0 if t.is_up else 0.0
        return grid_state
