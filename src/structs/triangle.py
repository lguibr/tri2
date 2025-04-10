# File: src/structs/triangle.py
from __future__ import annotations  # For type hinting Triangle within Triangle
from typing import Tuple, Optional, List


class Triangle:
    """Represents a single triangular cell on the grid."""

    def __init__(self, row: int, col: int, is_up: bool, is_death: bool = False):
        self.row = row
        self.col = col
        self.is_up = is_up  # Orientation: True=points up, False=points down
        self.is_death = is_death  # Is this cell part of the non-playable border?
        self.is_occupied = is_death  # Occupied if it's a death cell initially
        self.color: Optional[Tuple[int, int, int]] = (
            None  # Color if occupied by a shape
        )

        # Neighbor references (set by grid_neighbors.link_neighbors)
        self.neighbor_left: Optional["Triangle"] = None
        self.neighbor_right: Optional["Triangle"] = None
        self.neighbor_vert: Optional["Triangle"] = (
            None  # The one sharing the horizontal base/tip
        )

    def get_points(
        self, ox: float, oy: float, cw: float, ch: float
    ) -> List[Tuple[float, float]]:
        """Calculates vertex points for drawing, relative to origin (ox, oy)."""
        # x, y is the top-left corner of the bounding box for this triangle
        x = ox + self.col * (cw * 0.75)  # Effective width is 0.75 * cell width
        y = oy + self.row * ch
        # Vertices depend on orientation
        if self.is_up:
            # Points up: base is at bottom
            return [(x, y + ch), (x + cw, y + ch), (x + cw / 2, y)]
        else:
            # Points down: base is at top
            return [(x, y), (x + cw, y), (x + cw / 2, y + ch)]

    def copy(self) -> "Triangle":
        """Creates a copy of the Triangle object's state (neighbors are not copied)."""
        new_tri = Triangle(self.row, self.col, self.is_up, self.is_death)
        new_tri.is_occupied = self.is_occupied
        new_tri.color = self.color
        # Neighbors must be relinked in the copied grid structure
        return new_tri

    def __repr__(self) -> str:
        state = "D" if self.is_death else ("O" if self.is_occupied else ".")
        orient = "^" if self.is_up else "v"
        return f"T({self.row},{self.col} {orient}{state})"

    # Make hashable based on position for use in sets/dicts if needed
    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return NotImplemented
        return self.row == other.row and self.col == other.col
