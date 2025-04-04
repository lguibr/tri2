# File: environment/shape.py
# (No changes needed)
import random
from typing import List, Tuple
from config import EnvConfig, VisConfig  # Needs VisConfig only for colors

GOOGLE_COLORS = VisConfig.GOOGLE_COLORS  # Use colors from VisConfig


class Shape:
    """Represents a polyomino-like shape made of triangles."""

    def __init__(self) -> None:
        # List of (relative_row, relative_col, is_up) tuples defining the shape
        self.triangles: List[Tuple[int, int, bool]] = []
        self.color: Tuple[int, int, int] = random.choice(GOOGLE_COLORS)
        self._generate()  # Generate the shape structure

    def _generate(self) -> None:
        """Generates a random shape by adding adjacent triangles."""
        n = random.randint(1, 5)  # Number of triangles in the shape
        first_up = random.choice([True, False])  # Orientation of the root triangle
        self.triangles.append((0, 0, first_up))  # Add the root triangle at (0,0)

        # Add remaining triangles adjacent to existing ones
        for _ in range(n - 1):
            # Find valid neighbors of the *last added* triangle
            lr, lc, lu = self.triangles[-1]
            nbrs = self._find_valid_neighbors(lr, lc, lu)
            if nbrs:
                self.triangles.append(random.choice(nbrs))
            # else: Could break early if no valid neighbors found, shape < n

    def _find_valid_neighbors(
        self, r: int, c: int, up: bool
    ) -> List[Tuple[int, int, bool]]:
        """Finds potential neighbor triangles that are not already part of the shape."""
        if up:  # Neighbors of an UP triangle are DOWN triangles
            ns = [(r, c - 1, False), (r, c + 1, False), (r + 1, c, False)]
        else:  # Neighbors of a DOWN triangle are UP triangles
            ns = [(r, c - 1, True), (r, c + 1, True), (r - 1, c, True)]
        # Return only neighbors that are not already in self.triangles
        return [n for n in ns if n not in self.triangles]

    def bbox(self) -> Tuple[int, int, int, int]:
        """Calculates the bounding box (min_r, min_c, max_r, max_c) of the shape."""
        if not self.triangles:
            return (0, 0, 0, 0)
        rr = [t[0] for t in self.triangles]
        cc = [t[1] for t in self.triangles]
        return (min(rr), min(cc), max(rr), max(cc))
