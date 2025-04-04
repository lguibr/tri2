# File: environment/grid.py
# (No structural changes, cleanup comments)
import numpy as np
from typing import List, Tuple
from .triangle import Triangle
from .shape import Shape
from config import EnvConfig


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self):
        self.rows = EnvConfig.ROWS
        self.cols = EnvConfig.COLS
        # Padding defines 'death' cells on the sides
        # Extend padding with 0 if grid is taller than 8 rows
        self.pad = [4, 3, 2, 1, 1, 2, 3, 4] + [0] * max(0, self.rows - 8)
        self.triangles: List[List[Triangle]] = []
        self._create()

    def _create(self) -> None:
        """Initializes the grid with Triangle objects."""
        self.triangles = []
        for r in range(self.rows):
            rowt = []
            p = self.pad[
                r % len(self.pad)
            ]  # Use modulo for robustness if rows > len(pad)
            for c in range(self.cols):
                # Determine if cell is a 'death' cell based on padding
                is_death_cell = c < p or c >= self.cols - p
                # Determine orientation (up/down) based on position
                is_up_cell = (r + c) % 2 == 0
                tri = Triangle(r, c, is_up=is_up_cell, is_death=is_death_cell)
                rowt.append(tri)
            self.triangles.append(rowt)

    def valid(self, r: int, c: int) -> bool:
        """Check if (r, c) is within grid boundaries."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(self, shp: Shape, rr: int, cc: int) -> bool:
        """Checks if a shape can be placed at the target root position (rr, cc)."""
        for dr, dc, up in shp.triangles:
            nr, nc = rr + dr, cc + dc  # Absolute grid coordinates
            if not self.valid(nr, nc):
                return False  # Out of bounds
            tri = self.triangles[nr][nc]
            # Cannot place if cell is death, already occupied, or wrong orientation
            if tri.is_death or tri.is_occupied or (tri.is_up != up):
                return False
        return True

    def place(self, shp: Shape, rr: int, cc: int) -> None:
        """Places a shape onto the grid at the target root position."""
        for dr, dc, _ in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if self.valid(nr, nc):  # Safety check
                tri = self.triangles[nr][nc]
                # Place only if valid (not death, not occupied)
                if not tri.is_death and not tri.is_occupied:
                    tri.is_occupied = True
                    tri.color = shp.color
                # else: Log warning if trying to place on invalid cell? Usually caught by can_place.
            # else: Log warning if trying to place out of bounds? Should be caught by can_place.

    def clear_filled_rows(self) -> Tuple[int, int]:
        """
        Clears fully occupied rows (ignoring death cells).
        Returns: (number of lines cleared, number of triangles in cleared lines).
        """
        lines_cleared = 0
        triangles_cleared = 0
        rows_to_clear_indices = []

        for r in range(self.rows):
            rowt = self.triangles[r]
            is_row_full = True
            num_placeable_triangles_in_row = 0
            for t in rowt:
                if not t.is_death:
                    num_placeable_triangles_in_row += 1
                    if not t.is_occupied:
                        is_row_full = False
                        # break # Optimization: stop checking row if one empty cell found

            # Only clear if the row has placeable triangles and all are occupied
            if is_row_full and num_placeable_triangles_in_row > 0:
                rows_to_clear_indices.append(r)
                lines_cleared += 1

        # Clear the identified rows and count cleared triangles
        for r_idx in rows_to_clear_indices:
            for t in self.triangles[r_idx]:
                if not t.is_death and t.is_occupied:
                    triangles_cleared += 1
                    t.is_occupied = False  # Reset state
                    t.color = None

        # Note: Gravity is not implemented here (complex for this grid)

        return lines_cleared, triangles_cleared

    def count_holes(self) -> int:
        """Counts empty, non-death cells with an occupied cell somewhere above them."""
        holes = 0
        for c in range(self.cols):  # Iterate column by column
            occupied_above = False
            for r in range(self.rows):  # Iterate row by row within column
                tri = self.triangles[r][c]
                if tri.is_death:
                    continue  # Ignore death cells

                if tri.is_occupied:
                    occupied_above = True  # Mark that we've seen an occupied cell above
                elif not tri.is_occupied and occupied_above:
                    # This is an empty cell below an occupied one in the same column
                    holes += 1
        return holes

    def get_feature_matrix(self) -> np.ndarray:
        """
        Returns grid state as a [Channel, Height, Width] numpy array (float32).
        Channels: 0=Occupied, 1=Is_Up, 2=Is_Death
        """
        grid_state = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                t = self.triangles[r][c]
                grid_state[0, r, c] = 1.0 if t.is_occupied else 0.0
                grid_state[1, r, c] = 1.0 if t.is_up else 0.0
                grid_state[2, r, c] = 1.0 if t.is_death else 0.0
        return grid_state
