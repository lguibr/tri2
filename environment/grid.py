# File: environment/grid.py
import numpy as np
from typing import List, Tuple
from .triangle import Triangle
from .shape import Shape
from config import EnvConfig


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig):
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        # Padding defines 'death' cells on the sides
        # Use modulo for robustness if rows > len(pad)
        _default_pad = [4, 3, 2, 1, 1, 2, 3, 4]
        self.pad = _default_pad + [0] * max(0, self.rows - len(_default_pad))
        self.triangles: List[List[Triangle]] = []
        self._create()

    def _create(self) -> None:
        """Initializes the grid with Triangle objects."""
        self.triangles = []
        for r in range(self.rows):
            rowt = []
            p = self.pad[r]  # Access pre-calculated pad value
            for c in range(self.cols):
                is_death_cell = c < p or c >= self.cols - p
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
            nr, nc = rr + dr, cc + dc
            if not self.valid(nr, nc):
                return False
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
                tri = self.triangles[nr][nc]
                # Double-check placement logic redundancy (can_place should prevent this)
                if not tri.is_death and not tri.is_occupied:
                    tri.is_occupied = True
                    tri.color = shp.color

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
                        break

            if is_row_full and num_placeable_triangles_in_row > 0:
                rows_to_clear_indices.append(r)
                lines_cleared += 1

        # Clear the rows marked for clearing
        for r_idx in rows_to_clear_indices:
            for t in self.triangles[r_idx]:
                if not t.is_death and t.is_occupied:
                    triangles_cleared += 1
                    t.is_occupied = False
                    t.color = None

        # Implement gravity (shifting rows down) - Basic version
        if lines_cleared > 0:
            # Create a new empty grid structure (preserving death cells)
            new_triangles: List[List[Optional[Triangle]]] = [
                [None for _ in range(self.cols)] for _ in range(self.rows)
            ]
            current_write_row = self.rows - 1

            # Iterate upwards through the old grid
            for r_old in range(self.rows - 1, -1, -1):
                # Skip rows that were cleared
                if r_old in rows_to_clear_indices:
                    continue

                # Copy non-cleared row to the lowest available position in the new grid
                if current_write_row >= 0:
                    for c in range(self.cols):
                        old_tri = self.triangles[r_old][c]
                        # Update row index of the triangle being moved
                        old_tri.row = current_write_row
                        new_triangles[current_write_row][c] = old_tri
                    current_write_row -= 1

            # Fill remaining top rows with new empty triangles (respecting death cells)
            for r_new in range(current_write_row, -1, -1):
                p = self.pad[r_new]  # Access pre-calculated pad value
                for c in range(self.cols):
                    is_death_cell = c < p or c >= self.cols - p
                    is_up_cell = (r_new + c) % 2 == 0
                    new_triangles[r_new][c] = Triangle(
                        r_new, c, is_up=is_up_cell, is_death=is_death_cell
                    )

            # Ensure no None values remain and assign the new grid
            self.triangles = [
                [tri for tri in row if tri is not None] for row in new_triangles
            ]

        return lines_cleared, triangles_cleared

    def count_holes(self) -> int:
        """Counts empty, non-death cells with an occupied cell somewhere above them."""
        holes = 0
        for c in range(self.cols):
            occupied_above = False
            for r in range(self.rows):
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

    # --- MODIFIED: Return 2 channels ---
    def get_feature_matrix(self) -> np.ndarray:
        """
        Returns grid state as a [Channel, Height, Width] numpy array (float32).
        Channels: 0=Occupied, 1=Is_Up
        """
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                t = self.triangles[r][c]
                if not t.is_death:  # Only consider non-death cells for features
                    grid_state[0, r, c] = 1.0 if t.is_occupied else 0.0
                    grid_state[1, r, c] = 1.0 if t.is_up else 0.0
                # else: Death cells remain 0.0 in both channels
        return grid_state

    # --- END MODIFIED ---
