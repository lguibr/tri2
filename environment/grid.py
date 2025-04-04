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
        self._create()  # Call the modified create method

    # --- _create applies adjustment for extra padding ---
    def _create(self) -> None:
        """Initializes the grid with Triangle objects based on playable columns per row."""
        # Define the BASE number of desired playable columns for centering
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]  # For ROWS=8, COLS=15

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
            # Get the target playable width for this row before adjustment
            base_playable_cols = cols_per_row[r]

            # Calculate initial centering based on base_playable_cols
            if base_playable_cols <= 0:
                initial_death_cols_left = self.cols
            elif base_playable_cols >= self.cols:
                initial_death_cols_left = 0
            else:
                initial_death_cols_left = (self.cols - base_playable_cols) // 2
            initial_first_death_col_right = initial_death_cols_left + base_playable_cols

            # --- ADJUSTMENT: Add 1 extra dead cell padding on each side ---
            # Increase left padding by 1, decrease right boundary by 1
            # This effectively reduces the playable width by 2 compared to base_playable_cols
            adjusted_death_cols_left = initial_death_cols_left + 1
            adjusted_first_death_col_right = initial_first_death_col_right - 1
            # --- END ADJUSTMENT ---

            for c in range(self.cols):
                # Use the ADJUSTED boundaries to determine death cells
                # A cell is dead if it's left of the adjusted start or at/right of the adjusted end.
                # Ensure right boundary doesn't cross left boundary if base_playable_cols is small
                is_death_cell = (
                    (c < adjusted_death_cols_left)
                    or (
                        c >= adjusted_first_death_col_right
                        and adjusted_first_death_col_right > adjusted_death_cols_left
                    )
                    or (base_playable_cols <= 2)
                )  # Mark all dead if base playable was 2 or less after adjustment

                is_up_cell = (r + c) % 2 == 0
                tri = Triangle(r, c, is_up=is_up_cell, is_death=is_death_cell)
                rowt.append(tri)
            self.triangles.append(rowt)

    # --- END MODIFIED ---

    def valid(self, r: int, c: int) -> bool:
        """Check if (r, c) is within grid boundaries."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(self, shp: Shape, rr: int, cc: int) -> bool:
        """Checks if a shape can be placed at the target root position (rr, cc)."""
        for dr, dc, up in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if not self.valid(nr, nc):
                return False
            # Check bounds before accessing triangles list
            if not (
                0 <= nr < len(self.triangles) and 0 <= nc < len(self.triangles[nr])
            ):
                return False  # Should not happen if self.valid passed, but safety check
            tri = self.triangles[nr][nc]
            # Cannot place on death cells or occupied cells or if orientation mismatches
            if tri.is_death or tri.is_occupied or (tri.is_up != up):
                return False
        return True

    def place(self, shp: Shape, rr: int, cc: int) -> None:
        """Places a shape onto the grid at the target root position."""
        for dr, dc, _ in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if self.valid(nr, nc):
                # Check bounds before accessing triangles list
                if not (
                    0 <= nr < len(self.triangles) and 0 <= nc < len(self.triangles[nr])
                ):
                    continue  # Skip if out of bounds
                tri = self.triangles[nr][nc]
                # Double-check placement logic redundancy (can_place should prevent this)
                if not tri.is_death and not tri.is_occupied:
                    tri.is_occupied = True
                    tri.color = shp.color

    # --- MODIFIED: clear_filled_rows REMOVES gravity ---
    def clear_filled_rows(self) -> Tuple[int, int]:
        """
        Clears fully occupied rows (ignoring death cells) by marking triangles as
        unoccupied. Does NOT apply gravity.
        Returns: (number of lines cleared, number of triangles in cleared lines).
        """
        lines_cleared = 0
        triangles_cleared = 0
        rows_to_clear_indices = []

        # 1. Identify rows to clear
        for r in range(self.rows):
            # Check bounds before accessing triangles list
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
                        break  # No need to check further in this row

            # Only mark for clearing if the row has placeable triangles and all are full
            if is_row_full and num_placeable_triangles_in_row > 0:
                rows_to_clear_indices.append(r)
                lines_cleared += 1

        # 2. Clear the triangles in marked rows (set unoccupied)
        for r_idx in rows_to_clear_indices:
            # Check bounds before accessing triangles list
            if not (0 <= r_idx < len(self.triangles)):
                continue
            for t in self.triangles[r_idx]:
                if (
                    not t.is_death and t.is_occupied
                ):  # Only reset occupied, non-death cells
                    triangles_cleared += 1
                    t.is_occupied = False
                    t.color = None  # Reset color

        # --- REMOVED Gravity Logic ---
        # The grid structure remains unchanged.

        return lines_cleared, triangles_cleared

    # --- END MODIFIED ---

    def count_holes(self) -> int:
        """Counts empty, non-death cells with an occupied cell somewhere above them."""
        holes = 0
        for c in range(self.cols):
            occupied_above = False
            for r in range(self.rows):
                # Check bounds before accessing triangles list
                if not (
                    0 <= r < len(self.triangles) and 0 <= c < len(self.triangles[r])
                ):
                    continue  # Skip if out of bounds
                tri = self.triangles[r][c]
                if tri.is_death:
                    # If a death cell is encountered, reset occupied_above for that column segment
                    occupied_above = False
                    continue

                if tri.is_occupied:
                    occupied_above = True
                elif not tri.is_occupied and occupied_above:
                    holes += 1
        return holes

    def get_feature_matrix(self) -> np.ndarray:
        """
        Returns grid state as a [Channel, Height, Width] numpy array (float32).
        Channels: 0=Occupied, 1=Is_Up
        """
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                # Check bounds before accessing triangles list
                if not (
                    0 <= r < len(self.triangles) and 0 <= c < len(self.triangles[r])
                ):
                    continue  # Skip if out of bounds
                t = self.triangles[r][c]
                if not t.is_death:  # Only consider non-death cells for features
                    grid_state[0, r, c] = 1.0 if t.is_occupied else 0.0
                    grid_state[1, r, c] = 1.0 if t.is_up else 0.0
                # else: Death cells remain 0.0 in both channels
        return grid_state
