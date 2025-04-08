from typing import Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .grid import Grid


class Triangle:
    """Represents a single triangular cell on the grid."""

    def __init__(self, row: int, col: int, is_up: bool, is_death: bool = False):
        self.row = row
        self.col = col
        self.is_up = is_up  # True if pointing up, False if pointing down
        self.is_death = is_death  # True if part of the unplayable border
        self.is_occupied = is_death  # Occupied if it's a death cell initially
        self.color: Optional[Tuple[int, int, int]] = (
            None  # Color if occupied by a shape
        )
        # Neighbors based on shared edges - these will be linked by the Grid
        self.neighbor_left: Optional["Triangle"] = None
        self.neighbor_right: Optional["Triangle"] = None
        self.neighbor_vert: Optional["Triangle"] = None

    def get_points(
        self, ox: int, oy: int, cw: int, ch: int
    ) -> List[Tuple[float, float]]:
        """Calculates the vertex points for drawing the triangle."""
        x = ox + self.col * (
            cw * 0.75
        )  # Horizontal position based on column and overlap
        y = oy + self.row * ch  # Vertical position based on row
        if self.is_up:
            # Points for an upward-pointing triangle
            return [(x, y + ch), (x + cw, y + ch), (x + cw / 2, y)]
        else:
            # Points for a downward-pointing triangle
            return [(x, y), (x + cw, y), (x + cw / 2, y + ch)]

    def get_line_neighbors(
        self,
    ) -> Tuple[Optional["Triangle"], Optional["Triangle"], Optional["Triangle"]]:
        """Returns neighbors relevant for line checking (left, right, vertical)."""
        return self.neighbor_left, self.neighbor_right, self.neighbor_vert

    def copy(self) -> "Triangle":
        """Creates a copy of the Triangle object's state, excluding neighbors."""
        new_tri = Triangle.__new__(Triangle)  # Create instance without calling __init__
        new_tri.row = self.row
        new_tri.col = self.col
        new_tri.is_up = self.is_up
        new_tri.is_death = self.is_death
        new_tri.is_occupied = self.is_occupied
        new_tri.color = self.color  # Copy color reference (tuple is immutable)
        # Neighbors are intentionally set to None, they will be re-linked by Grid.deepcopy_grid
        new_tri.neighbor_left = None
        new_tri.neighbor_right = None
        new_tri.neighbor_vert = None
        return new_tri
