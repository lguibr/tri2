"""
grid.py - The board of triangles.
"""

from typing import List
from .triangle import Triangle
from .shape import Shape
from config import EnvConfig

ROWS = EnvConfig().ROWS
COLS = EnvConfig().COLS


class Grid:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.pad = [4, 3, 2, 1, 1, 2, 3, 4]
        self.triangles: List[List[Triangle]] = []
        self._create()

    def _create(self) -> None:
        for r in range(self.rows):
            rowt = []
            p = self.pad[min(r, len(self.pad) - 1)]
            for c in range(self.cols):
                d = c < p or c >= self.cols - p
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
            if tri.is_death or tri.is_occupied or (tri.is_up != up):
                return False
        return True

    def place(self, shp: Shape, rr: int, cc: int) -> None:
        for dr, dc, _ in shp.triangles:
            tri = self.triangles[rr + dr][cc + dc]
            tri.is_occupied = True
            tri.color = shp.color

    def clear_filled_rows(self) -> int:
        cleared = 0
        for r in range(self.rows):
            rowt = self.triangles[r]
            all_full = True
            for t in rowt:
                if not t.is_death and not t.is_occupied:
                    all_full = False
                    break
            if all_full:
                for t in rowt:
                    t.is_occupied = False
                    t.color = None
                cleared += 1
        return cleared
