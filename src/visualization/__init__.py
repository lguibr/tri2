# File: src/visualization/__init__.py
# File: src/visualization/__init__.py
"""
Visualization module for rendering the game state using Pygame.
"""

# Core components
from .core.visualizer import Visualizer
from .core.game_renderer import GameRenderer  # Keep GameRenderer, it's adapted
from .core.layout import calculate_layout  # Keep layout, though simplified
from .core.fonts import load_fonts
from .core import colors  # Expose colors directly
from .core.coord_mapper import (
    get_grid_coords_from_screen,
    get_preview_index_from_screen,
)

# Drawing functions (imported directly from specific files)
from .drawing.grid import draw_grid_background, draw_grid_triangles
from .drawing.shapes import draw_shape
from .drawing.previews import (
    render_previews,
    draw_placement_preview,
    draw_floating_preview,
)
from .drawing.hud import render_hud
from .drawing.highlight import draw_debug_highlight

# UI Components
from .ui.progress_bar import ProgressBar

# Import the function directly from utils.geometry now
from src.utils.geometry import is_point_in_polygon

# Configuration
from src.config import VisConfig

# Removed re-exports of Triangle and Shape as they are in src.structs

__all__ = [
    # Core Classes & Functions
    "Visualizer",
    "GameRenderer",  # Keep GameRenderer
    "calculate_layout",
    "load_fonts",
    "colors",
    "get_grid_coords_from_screen",
    "get_preview_index_from_screen",
    # Drawing Functions
    "draw_grid_background",
    "draw_grid_triangles",
    "draw_shape",
    "render_previews",
    "draw_placement_preview",
    "draw_floating_preview",
    "render_hud",
    "draw_debug_highlight",
    "is_point_in_polygon",  # Re-exported from utils
    # UI Components
    "ProgressBar",
    # Config
    "VisConfig",
]
