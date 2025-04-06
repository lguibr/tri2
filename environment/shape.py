import random
from typing import List, Tuple

from config.constants import GOOGLE_COLORS


class Shape:
    """Represents a polyomino-like shape made of triangles."""

    def __init__(self) -> None:
        # List of (relative_row, relative_col, is_up) tuples defining the shape
        self.triangles: List[Tuple[int, int, bool]] = []
        # GOOGLE_COLORS is now imported from constants
        self.color: Tuple[int, int, int] = random.choice(GOOGLE_COLORS)
        self._generate()  # Generate the shape structure

    def _generate(self) -> None:
        """Generates a random shape by adding adjacent triangles."""
        num_triangles_in_shape = random.randint(1, 5)
        first_triangle_is_up = random.choice([True, False])
        # Add the root triangle at relative coordinates (0,0)
        self.triangles.append((0, 0, first_triangle_is_up))

        # Add remaining triangles adjacent to existing ones
        for _ in range(num_triangles_in_shape - 1):
            # Find valid neighbors of the *last added* triangle
            if not self.triangles:
                break  # Should not happen
            last_rel_row, last_rel_col, last_is_up = self.triangles[-1]
            valid_neighbors = self._find_valid_neighbors(
                last_rel_row, last_rel_col, last_is_up
            )
            if valid_neighbors:
                self.triangles.append(random.choice(valid_neighbors))
            # else: Could break early if no valid neighbors found, shape < n

    def _find_valid_neighbors(
        self, r: int, c: int, is_up: bool
    ) -> List[Tuple[int, int, bool]]:
        """Finds potential neighbor triangles that are not already part of the shape."""
        potential_neighbors: List[Tuple[int, int, bool]]
        if is_up:  # Neighbors of an UP triangle are DOWN triangles
            potential_neighbors = [
                (r, c - 1, False),
                (r, c + 1, False),
                (r + 1, c, False),
            ]
        else:  # Neighbors of a DOWN triangle are UP triangles
            potential_neighbors = [(r, c - 1, True), (r, c + 1, True), (r - 1, c, True)]
        # Return only neighbors that are not already in self.triangles
        valid_neighbors = [n for n in potential_neighbors if n not in self.triangles]
        return valid_neighbors

    def bbox(self) -> Tuple[int, int, int, int]:
        """Calculates the bounding box (min_r, min_c, max_r, max_c) of the shape."""
        if not self.triangles:
            return (0, 0, 0, 0)
        rows = [t[0] for t in self.triangles]
        cols = [t[1] for t in self.triangles]
        return (min(rows), min(cols), max(rows), max(cols))
