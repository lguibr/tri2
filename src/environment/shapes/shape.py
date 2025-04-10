# File: src/environment/shapes/shape.py
import random
from typing import List, Tuple, Optional

# Removed: from src.visualization import colors


class Shape:
    """Represents a polyomino-like shape made of triangles."""

    def __init__(
        self, triangles: List[Tuple[int, int, bool]], color: Tuple[int, int, int]
    ):
        # (dr, dc, is_up) relative coords from a reference point (usually 0,0)
        self.triangles: List[Tuple[int, int, bool]] = triangles
        self.color: Tuple[int, int, int] = color
        # Ensure triangles are centered around (0,0) or normalize if needed?
        # For now, assume generator provides relative coords correctly.

    def bbox(self) -> Tuple[int, int, int, int]:
        """Calculates bounding box (min_r, min_c, max_r, max_c) in relative coords."""
        if not self.triangles:
            return (0, 0, 0, 0)
        rows = [t[0] for t in self.triangles]
        cols = [t[1] for t in self.triangles]
        return (min(rows), min(cols), max(rows), max(cols))

    def copy(self) -> "Shape":
        """Creates a shallow copy (triangle list is copied, color is shared)."""
        new_shape = Shape.__new__(Shape)
        new_shape.triangles = list(self.triangles)  # Copy the list of tuples
        new_shape.color = self.color  # Color tuple is immutable, can share
        return new_shape

    def __str__(self) -> str:
        return f"Shape(Color:{self.color}, Tris:{len(self.triangles)})"
