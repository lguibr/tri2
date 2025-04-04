# File: environment/grid.py
import numpy as np
from typing import List, Tuple
from .triangle import Triangle
from .shape import Shape

# --- MODIFIED: Import EnvConfig directly ---
from config import EnvConfig

# --- END MODIFIED ---


class Grid:
    """Represents the game board composed of Triangles."""

    # --- MODIFIED: Accept EnvConfig in init ---
    def __init__(self, env_config: EnvConfig):
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        # --- END MODIFIED ---
        # Padding defines 'death' cells on the sides
        self.pad = [4, 3, 2, 1, 1, 2, 3, 4] + [0] * max(0, self.rows - 8)
        self.triangles: List[List[Triangle]] = []
        self._create()

    def _create(self) -> None:
        """Initializes the grid with Triangle objects."""
        self.triangles = []
        for r in range(self.rows):
            rowt = []
            # Use modulo for robustness if rows > len(pad)
            p = self.pad[r % len(self.pad)]
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
            if tri.is_death or tri.is_occupied or (tri.is_up != up):
                return False
        return True

    def place(self, shp: Shape, rr: int, cc: int) -> None:
        """Places a shape onto the grid at the target root position."""
        for dr, dc, _ in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if self.valid(nr, nc):
                tri = self.triangles[nr][nc]
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
                        break  # Optimization

            if is_row_full and num_placeable_triangles_in_row > 0:
                rows_to_clear_indices.append(r)
                lines_cleared += 1

        for r_idx in rows_to_clear_indices:
            for t in self.triangles[r_idx]:
                if not t.is_death and t.is_occupied:
                    triangles_cleared += 1
                    t.is_occupied = False
                    t.color = None

        # Note: Gravity is not implemented here

        return lines_cleared, triangles_cleared

    def count_holes(self) -> int:
        """Counts empty, non-death cells with an occupied cell somewhere above them."""
        holes = 0
        for c in range(self.cols):
            occupied_above = False
            for r in range(self.rows):
                tri = self.triangles[r][c]
                if tri.is_death:
                    continue

                if tri.is_occupied:
                    occupied_above = True
                elif not tri.is_occupied and occupied_above:
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
