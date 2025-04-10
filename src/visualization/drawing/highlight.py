# File: src/visualization/drawing/highlight.py
import pygame
from typing import TYPE_CHECKING

# Use relative imports within visualization package
from ..core import colors, coord_mapper

# Import Triangle from the new structs module
from src.structs import Triangle

if TYPE_CHECKING:
    from src.config import EnvConfig


def draw_debug_highlight(
    surface: pygame.Surface, r: int, c: int, config: "EnvConfig"
) -> None:
    """Highlights a specific triangle border for debugging."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    is_up = (r + c) % 2 == 0
    temp_tri = Triangle(
        r, c, is_up
    )  # Temp triangle for geometry (uses Triangle from structs)
    pts = temp_tri.get_points(ox, oy, cw, ch)

    pygame.draw.polygon(surface, colors.DEBUG_TOGGLE_COLOR, pts, 3)  # Thicker border
