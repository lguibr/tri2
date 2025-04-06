# File: environment/grid.py
import numpy as np
from typing import List, Tuple, Optional, Any

# --- MOVED IMPORT TO TOP LEVEL ---
from config import EnvConfig

# --- END MOVED IMPORT ---

from .triangle import Triangle
from .shape import Shape


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig):  # Accept EnvConfig instance
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.triangles: List[List[Triangle]] = []
        self._create(env_config)  # Pass config to _create

    def _create(self, env_config: EnvConfig) -> None:
        """Initializes the grid with playable and death cells."""
        # Example pattern for a hexagon-like board within the grid bounds
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]  # Specific to 8 rows

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
            row_triangles: List[Triangle] = []
            base_playable_cols = cols_per_row[r]

            # Calculate padding for death cells
            initial_death_cols_left = (
                (self.cols - base_playable_cols) // 2
                if base_playable_cols < self.cols
                else 0
            )
            initial_first_death_col_right = initial_death_cols_left + base_playable_cols

            # Adjustment for Specific Hex Grid Pattern (makes it slightly narrower)
            adjusted_death_cols_left = initial_death_cols_left + 1
            adjusted_first_death_col_right = initial_first_death_col_right - 1

            for c in range(self.cols):
                is_death_cell = (
                    (c < adjusted_death_cols_left)
                    or (
                        c >= adjusted_first_death_col_right
                        and adjusted_first_death_col_right > adjusted_death_cols_left
                    )
                    or (base_playable_cols <= 2)  # Treat very narrow rows as death
                )
                is_up_cell = (r + c) % 2 == 0  # Checkerboard pattern for orientation
                triangle = Triangle(r, c, is_up=is_up_cell, is_death=is_death_cell)
                row_triangles.append(triangle)
            self.triangles.append(row_triangles)

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(
        self, shape_to_place: Shape, target_row: int, target_col: int
    ) -> bool:
        """Checks if a shape can be placed at the target location."""
        for dr, dc, is_up_shape_tri in shape_to_place.triangles:
            nr, nc = target_row + dr, target_col + dc
            if not self.valid(nr, nc):
                return False  # Out of bounds
            grid_triangle = self.triangles[nr][nc]
            # Cannot place on death cells, occupied cells, or cells with mismatching orientation
            if (
                grid_triangle.is_death
                or grid_triangle.is_occupied
                or (grid_triangle.is_up != is_up_shape_tri)
            ):
                return False
        return True  # All shape triangles can be placed

    def place(self, shape_to_place: Shape, target_row: int, target_col: int) -> None:
        """Places a shape onto the grid (assumes can_place was checked)."""
        for dr, dc, _ in shape_to_place.triangles:
            nr, nc = target_row + dr, target_col + dc
            if self.valid(nr, nc):
                grid_triangle = self.triangles[nr][nc]
                # Only occupy non-death, non-occupied cells
                if not grid_triangle.is_death and not grid_triangle.is_occupied:
                    grid_triangle.is_occupied = True
                    grid_triangle.color = shape_to_place.color

    def clear_filled_rows(self) -> Tuple[int, int, List[Tuple[int, int]]]:
        """Clears fully occupied rows and returns stats."""
        lines_cleared = 0
        triangles_cleared = 0
        rows_to_clear_indices: List[int] = []
        cleared_triangles_coords: List[Tuple[int, int]] = []

        # Identify rows to clear
        for r in range(self.rows):
            row_triangles = self.triangles[r]
            is_row_full = True
            num_placeable_triangles_in_row = 0
            for triangle in row_triangles:
                if not triangle.is_death:
                    num_placeable_triangles_in_row += 1
                    if not triangle.is_occupied:
                        is_row_full = False
                        break  # Row is not full
            # Clear if all placeable triangles are occupied (and there are placeable triangles)
            if is_row_full and num_placeable_triangles_in_row > 0:
                rows_to_clear_indices.append(r)
                lines_cleared += 1

        # Clear the identified rows
        for r_idx in rows_to_clear_indices:
            for triangle in self.triangles[r_idx]:
                if not triangle.is_death and triangle.is_occupied:
                    triangles_cleared += 1
                    triangle.is_occupied = False
                    triangle.color = None
                    cleared_triangles_coords.append((r_idx, triangle.col))

        # Note: This implementation doesn't shift rows down.
        # Depending on the game rules, row shifting might be needed here.

        return lines_cleared, triangles_cleared, cleared_triangles_coords

    def get_column_heights(self) -> List[int]:
        """Calculates the height of occupied cells in each column."""
        heights = [0] * self.cols
        for c in range(self.cols):
            for r in range(self.rows - 1, -1, -1):  # Iterate from top down
                triangle = self.triangles[r][c]
                if triangle.is_occupied and not triangle.is_death:
                    heights[c] = r + 1  # Height is row index + 1
                    break  # Found highest occupied cell in this column
        return heights

    def get_max_height(self) -> int:
        """Returns the maximum height across all columns."""
        heights = self.get_column_heights()
        return max(heights) if heights else 0

    def get_bumpiness(self) -> int:
        """Calculates the total absolute difference between adjacent column heights."""
        heights = self.get_column_heights()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def count_holes(self) -> int:
        """Counts the number of empty, non-death cells below an occupied cell in the same column."""
        holes = 0
        for c in range(self.cols):
            occupied_above_found = False
            for r in range(self.rows):  # Iterate from bottom up
                triangle = self.triangles[r][c]
                if triangle.is_death:
                    continue
                if triangle.is_occupied:
                    occupied_above_found = True
                elif not triangle.is_occupied and occupied_above_found:
                    # Found an empty cell below an occupied one in this column
                    holes += 1
        return holes

    def get_feature_matrix(self) -> np.ndarray:
        """Returns the grid state as a 2-channel numpy array (Occupancy, Orientation)."""
        # Channel 0: Occupancy (1.0 if occupied and not death, 0.0 otherwise)
        # Channel 1: Orientation (1.0 if pointing up and not death, 0.0 otherwise)
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                triangle = self.triangles[r][c]
                if not triangle.is_death:
                    grid_state[0, r, c] = 1.0 if triangle.is_occupied else 0.0
                    grid_state[1, r, c] = 1.0 if triangle.is_up else 0.0
        return grid_state
