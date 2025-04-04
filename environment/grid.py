# File: environment/grid.py
"""
grid.py - The board of triangles.
"""
import numpy as np
from typing import List
from .triangle import Triangle
from .shape import Shape

# <<< Import EnvConfig to access ROWS/COLS directly >>>
from config import EnvConfig


class Grid:
    def __init__(self):
        # <<< Use EnvConfig directly >>>
        self.rows = EnvConfig.ROWS
        self.cols = EnvConfig.COLS
        # Padding might need adjustment if ROWS changes significantly
        self.pad = [4, 3, 2, 1, 1, 2, 3, 4] + [0] * (
            self.rows - 8
        )  # Basic extension if rows > 8
        self.triangles: List[List[Triangle]] = []
        self._create()

    def _create(self) -> None:
        self.triangles = []  # Ensure clean slate
        for r in range(self.rows):
            rowt = []
            # Use modulo for padding index robustness
            p = self.pad[r % len(self.pad)]
            for c in range(self.cols):
                d = c < p or c >= self.cols - p
                # Alternate 'up' based on row and column
                tri = Triangle(r, c, (r + c) % 2 == 0, d)
                rowt.append(tri)
            self.triangles.append(rowt)

    def valid(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(self, shp: Shape, rr: int, cc: int) -> bool:
        for dr, dc, up in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if not self.valid(nr, nc):
                return False
            tri = self.triangles[nr][nc]
            # Cannot place if: death cell, already occupied, or triangle orientation mismatch
            if tri.is_death or tri.is_occupied or (tri.is_up != up):
                return False
        return True

    def place(self, shp: Shape, rr: int, cc: int) -> None:
        for dr, dc, _ in shp.triangles:
            # Assumes can_place was checked before calling place
            nr, nc = rr + dr, cc + dc
            if self.valid(nr, nc):  # Double check validity for safety
                tri = self.triangles[nr][nc]
                tri.is_occupied = True
                tri.color = shp.color
            else:
                # This should not happen if can_place is checked
                print(
                    f"Warning: Attempted to place triangle out of bounds at ({nr}, {nc}) during place operation."
                )

    def clear_filled_rows(self) -> int:
        cleared = 0
        rows_to_clear_indices = []
        for r in range(self.rows):
            rowt = self.triangles[r]
            all_full = True
            for t in rowt:
                # A row is full if every non-death triangle is occupied
                if not t.is_death and not t.is_occupied:
                    all_full = False
                    break
            if all_full:
                rows_to_clear_indices.append(r)
                cleared += 1

        # Clear the identified rows
        for r_idx in rows_to_clear_indices:
            for t in self.triangles[r_idx]:
                # Only reset non-death triangles
                if not t.is_death:
                    t.is_occupied = False
                    t.color = None

        # Optional: Implement gravity (shifting rows down) - Complex for hex, skipping for now.
        # If implemented, need to be careful with triangle 'is_up' property after shifting.

        return cleared

    # <<< NEW: Hole Counting >>>
    def count_holes(self) -> int:
        """
        Counts the number of holes in the grid.
        Simple definition: An empty, non-death cell is a hole if there is
        an occupied cell *somewhere above it* in a vaguely columnar sense.
        Approximation: Check directly above for simplicity in hex.
        """
        holes = 0
        for r in range(1, self.rows):  # Start from row 1, row 0 cannot have holes above
            for c in range(self.cols):
                current_tri = self.triangles[r][c]
                # Is it a potential hole? (empty, not death)
                if not current_tri.is_occupied and not current_tri.is_death:
                    # Check if cell(s) 'above' are occupied.
                    # Simple check: Direct vertical adjacency (if same 'up' orientation allows it)
                    # More robust: Check a few cells above.
                    # Simplest: Check if *any* cell is occupied in the column above row r.
                    is_covered = False
                    for r_above in range(r):
                        if self.triangles[r_above][c].is_occupied:
                            is_covered = True
                            break
                    if is_covered:
                        holes += 1
        return holes

    # Optional: Add a method to get grid state specifically for ConvNet if needed
    def get_feature_matrix(self) -> np.ndarray:
        """Returns grid state as [Channel, Height, Width] numpy array."""
        # Channels: 0=Occupied, 1=Is_Up, 2=Is_Death
        grid_state = np.zeros((3, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                t = self.triangles[r][c]
                grid_state[0, r, c] = 1.0 if t.is_occupied else 0.0
                grid_state[1, r, c] = 1.0 if t.is_up else 0.0
                grid_state[2, r, c] = 1.0 if t.is_death else 0.0
        return grid_state
