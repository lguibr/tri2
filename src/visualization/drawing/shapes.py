# File: src/visualization/drawing/shapes.py
import pygame
from typing import TYPE_CHECKING, Tuple, Optional

# Use relative imports within visualization package
from ..core import colors

# Import Triangle and Shape from the new structs module
from src.structs import Triangle, Shape

if TYPE_CHECKING:
    pass  # No other type hints needed here currently


def draw_shape(
    surface: pygame.Surface,
    shape: Shape,  # Use Shape from structs
    topleft: Tuple[int, int],
    cell_size: float,
    is_selected: bool = False,  # Example parameter, adjust as needed
    origin_offset: Tuple[int, int] = (0, 0),  # Offset for relative coords
) -> None:
    """Draws a single shape onto a surface."""
    if not shape or not shape.triangles or cell_size <= 0:
        return

    # Use shape's color, maybe adjust if selected?
    shape_color = shape.color
    border_color = colors.GRAY  # Example border color

    # Calculate effective cell width/height for triangles
    cw = cell_size
    ch = cell_size

    for dr, dc, is_up in shape.triangles:
        # Adjust relative coords by origin offset
        adj_r, adj_c = dr + origin_offset[0], dc + origin_offset[1]

        # Calculate top-left corner for this triangle's bounding box
        # relative to the provided topleft
        tri_x = topleft[0] + adj_c * (cw * 0.75)
        tri_y = topleft[1] + adj_r * ch

        # Create a temporary Triangle object just for getting points
        temp_tri = Triangle(0, 0, is_up)  # Use Triangle from structs
        # Get points relative to (0,0) with cell_size, then offset
        pts = [(px + tri_x, py + tri_y) for px, py in temp_tri.get_points(0, 0, cw, ch)]

        pygame.draw.polygon(surface, shape_color, pts)
        pygame.draw.polygon(surface, border_color, pts, 1)  # Draw border
