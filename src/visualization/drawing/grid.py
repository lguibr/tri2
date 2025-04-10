# File: src/visualization/drawing/grid.py
import pygame
from typing import TYPE_CHECKING

# Use relative imports within visualization package
from ..core import colors, coord_mapper

# Import Triangle from the new structs module
from src.structs import Triangle

if TYPE_CHECKING:
    from src.environment import GridData  # GridData remains in environment
    from src.config import EnvConfig


def draw_grid_background(surface: pygame.Surface, bg_color: tuple) -> None:
    """Fills the grid area surface with a background color."""
    surface.fill(bg_color)


def draw_grid_triangles(
    surface: pygame.Surface, grid_data: "GridData", config: "EnvConfig"
) -> None:
    """Draws all triangles (empty, occupied) on the grid surface."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            tri: Triangle = grid_data.triangles[r][c]  # Use Triangle from structs
            if tri.is_death:
                continue  # Skip death cells

            color = tri.color if tri.is_occupied else colors.TRIANGLE_EMPTY_COLOR
            pts = tri.get_points(
                ox, oy, cw, ch
            )  # Get points relative to grid surface origin

            pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, colors.GRID_LINE_COLOR, pts, 1)  # Border
