# File: src/interaction/debug_mode_handler.py
import pygame
import logging
from typing import TYPE_CHECKING, Tuple

# Import specific modules/functions needed
from src.visualization import core as vis_core  # Import core visualization elements
from src.environment import core as env_core  # Import core environment elements
from src.environment import grid as env_grid  # Import grid logic

# Import Triangle from the new structs module
from src.structs import Triangle

if TYPE_CHECKING:
    from src.environment.core.game_state import GameState
    from src.visualization.core.visualizer import Visualizer

logger = logging.getLogger(__name__)


def handle_debug_click(
    event: pygame.event.Event,
    mouse_pos: Tuple[int, int],
    game_state: "GameState",
    visualizer: "Visualizer",
) -> None:
    """Handles mouse clicks in debug mode (toggle triangle state)."""
    if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
        return

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect:
        logger.error("Grid layout rectangle not available for debug click.")
        return

    # Use coord_mapper from visualization.core
    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )
    if not grid_coords:
        return

    r, c = grid_coords
    # Use grid_data directly from game_state
    if game_state.grid_data.valid(r, c):
        tri: Triangle = game_state.grid_data.triangles[r][
            c
        ]  # Use Triangle from structs
        if not tri.is_death:
            # Toggle occupancy state
            tri.is_occupied = not tri.is_occupied
            # Update the corresponding numpy array state
            game_state.grid_data._occupied_np[r, c] = tri.is_occupied
            # Update color for visualization using colors from visualization.core
            tri.color = vis_core.colors.DEBUG_TOGGLE_COLOR if tri.is_occupied else None
            logger.info(
                f"DEBUG: Toggled triangle ({r},{c}) -> {'Occupied' if tri.is_occupied else 'Empty'}"
            )

            # Check for line clears AFTER potentially setting to occupied
            if tri.is_occupied:
                # Use grid logic from environment.grid
                # Correct the keyword argument name here:
                lines, tris, coords = env_grid.logic.check_and_clear_lines(
                    game_state.grid_data, newly_occupied_triangles={tri}
                )
                if lines > 0:
                    logger.info(
                        f"DEBUG: Cleared {lines} lines ({tris} tris) at {coords} after toggle."
                    )
        else:
            logger.info(f"Clicked on death cell ({r},{c}). No action.")


def update_debug_hover(
    mouse_pos: Tuple[int, int], game_state: "GameState", visualizer: "Visualizer"
) -> None:
    """Updates the debug highlight position based on mouse hover."""
    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect:
        game_state.debug_highlight_pos = None
        return

    # Use coord_mapper from visualization.core
    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )
    valid_hover = False
    if grid_coords:
        r, c = grid_coords
        # Highlight only if hovering over a valid, non-death cell
        # Use grid_data directly from game_state
        if (
            game_state.grid_data.valid(r, c)
            and not game_state.grid_data.triangles[r][c].is_death
        ):
            game_state.debug_highlight_pos = grid_coords
            valid_hover = True

    if not valid_hover:
        game_state.debug_highlight_pos = None
