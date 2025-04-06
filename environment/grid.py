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
        # Determine playable columns based on row index (example pattern)
        # This specific pattern defines a hexagon-like board within the grid bounds
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]  # Example for 8 rows

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

            # Calculate padding for death cells based on total cols and playable cols
            if base_playable_cols <= 0:
                initial_death_cols_left = self.cols  # All death if 0 playable
            elif base_playable_cols >= self.cols:
                initial_death_cols_left = 0  # No death if playable >= total
            else:
                initial_death_cols_left = (self.cols - base_playable_cols) // 2

            # Calculate the column index where death cells start on the right
            initial_first_death_col_right = initial_death_cols_left + base_playable_cols

            # --- Adjustment for Specific Hex Grid Pattern ---
            # This adjustment slightly shifts the death zones inward for the hex pattern
            # If you want a simple rectangle, remove this adjustment
            adjusted_death_cols_left = initial_death_cols_left + 1
            adjusted_first_death_col_right = initial_first_death_col_right - 1
            # --- End Adjustment ---

            for c in range(self.cols):
                # Determine if the cell is a death cell based on adjusted boundaries
                is_death_cell = (
                    (c < adjusted_death_cols_left)
                    or (
                        c >= adjusted_first_death_col_right
                        and adjusted_first_death_col_right > adjusted_death_cols_left
                    )
                    or (base_playable_cols <= 2)  # Treat very narrow rows as death
                )

                # Determine triangle orientation based on row and column index
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
            # Check bounds for self.triangles access
            if not (
                0 <= nr < len(self.triangles) and 0 <= nc < len(self.triangles[nr])
            ):
                return False  # Should not happen if self.valid passed, but safety check
            tri = self.triangles[nr][nc]
            if tri.is_death or tri.is_occupied or (tri.is_up != up):
                return False
        return True

    def place(self, shp: Shape, rr: int, cc: int) -> None:
        for dr, dc, _ in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if self.valid(nr, nc):
                # Check bounds again before accessing
                if not (
                    0 <= nr < len(self.triangles) and 0 <= nc < len(self.triangles[nr])
                ):
                    continue
                tri = self.triangles[nr][nc]
                # Only place if the target cell is valid (not death, not occupied)
                if not tri.is_death and not tri.is_occupied:
                    tri.is_occupied = True
                    tri.color = shp.color  # Assign shape color

    def clear_filled_rows(self) -> Tuple[int, int, List[Tuple[int, int]]]:
        lines_cleared = 0
        triangles_cleared = 0
        rows_to_clear_indices = []
        cleared_triangles_coords: List[Tuple[int, int]] = []

        # Identify full rows
        for r in range(self.rows):
            if not (0 <= r < len(self.triangles)):
                continue  # Bounds check
            rowt = self.triangles[r]
            is_row_full = True
            num_placeable_triangles_in_row = 0
            for t in rowt:
                if not t.is_death:
                    num_placeable_triangles_in_row += 1
                    if not t.is_occupied:
                        is_row_full = False
                        break  # Can stop checking this row

            # A row is considered full if all non-death cells are occupied
            if is_row_full and num_placeable_triangles_in_row > 0:
                rows_to_clear_indices.append(r)
                lines_cleared += 1

        # Clear the identified rows
        for r_idx in rows_to_clear_indices:
            if not (0 <= r_idx < len(self.triangles)):
                continue  # Bounds check
            for t in self.triangles[r_idx]:
                if not t.is_death and t.is_occupied:
                    triangles_cleared += 1
                    t.is_occupied = False
                    t.color = None
                    cleared_triangles_coords.append(
                        (r_idx, t.col)
                    )  # Store coords for visualization

        # (No gravity/dropping logic is implemented here)

        return lines_cleared, triangles_cleared, cleared_triangles_coords

    def get_column_heights(self) -> List[int]:
        """Calculates the height of occupied cells in each column."""
        heights = [0] * self.cols
        for c in range(self.cols):
            for r in range(self.rows - 1, -1, -1):
                # Check bounds before accessing triangles
                if 0 <= r < len(self.triangles) and 0 <= c < len(self.triangles[r]):
                    tri = self.triangles[r][c]
                    # Check if the cell is occupied and not a death cell
                    if tri.is_occupied and not tri.is_death:
                        heights[c] = r + 1  # Height is row index + 1
                        break  # Found highest occupied cell in this column
        return heights

    def get_max_height(self) -> int:
        """Calculates the maximum height across all columns."""
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
        """Counts the number of empty, non-death cells below an occupied cell in the same column."""
        holes = 0
        for c in range(self.cols):
            occupied_above = False
            for r in range(self.rows):  # Iterate from top to bottom
                # Check bounds before accessing triangles
                if not (
                    0 <= r < len(self.triangles) and 0 <= c < len(self.triangles[r])
                ):
                    continue  # Skip if out of bounds
                tri = self.triangles[r][c]

                # Skip death cells entirely, they don't count as holes or blockers
                if tri.is_death:
                    # If we hit a death cell below the highest block, reset occupied_above?
                    # Or just skip? Let's just skip for simplicity. Holes are non-death cells.
                    continue

                if tri.is_occupied:
                    occupied_above = (
                        True  # Mark that we've seen an occupied cell in this column
                    )
                elif not tri.is_occupied and occupied_above:
                    # This is an empty, non-death cell below an occupied cell in the same column
                    holes += 1
        return holes

    def get_feature_matrix(self) -> np.ndarray:
        """Creates a 2-channel feature matrix: [occupied, is_up]."""
        # Channel 0: Occupied (1.0) or Empty (0.0) - only for non-death cells
        # Channel 1: Orientation (1.0 if Up, 0.0 if Down) - only for non-death cells
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                # Check bounds before accessing triangles
                if not (
                    0 <= r < len(self.triangles) and 0 <= c < len(self.triangles[r])
                ):
                    continue  # Skip if out of bounds
                t = self.triangles[r][c]
                # Only populate features for non-death cells
                if not t.is_death:
                    grid_state[0, r, c] = 1.0 if t.is_occupied else 0.0
                    grid_state[1, r, c] = 1.0 if t.is_up else 0.0
                    # Optionally add more channels here (e.g., cell age, color?)
        return grid_state
