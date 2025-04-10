# File: src/interaction/play_mode_handler.py
import pygame
import logging
from typing import TYPE_CHECKING, Tuple, Optional

# Import specific modules/functions needed
from src.visualization import core as vis_core  # Import core visualization elements
from src.environment import core as env_core  # Import core environment elements
from src.environment import grid as env_grid  # Import grid logic
from src.environment import logic as env_logic  # Import game logic

# Import Shape from the new structs module
from src.structs import Shape

if TYPE_CHECKING:
    from src.environment.core.game_state import GameState
    from src.visualization.core.visualizer import Visualizer

logger = logging.getLogger(__name__)


def handle_play_click(
    event: pygame.event.Event,
    mouse_pos: Tuple[int, int],
    game_state: "GameState",
    visualizer: "Visualizer",
) -> None:
    """Handles mouse clicks in play mode (select preview, place shape)."""
    if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
        return  # Only handle left clicks

    # Prevent actions if game is over
    if game_state.is_over():
        logger.info("Game is over, ignoring click.")
        return

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    preview_rects = visualizer.preview_rects  # Get cached preview rects from visualizer

    # 1. Check for clicks on shape previews
    # Use coord_mapper from visualization.core
    preview_idx = vis_core.coord_mapper.get_preview_index_from_screen(
        mouse_pos, preview_rects
    )
    if preview_idx is not None:
        if game_state.demo_selected_shape_idx == preview_idx:
            # Clicked selected shape again: deselect
            game_state.demo_selected_shape_idx = -1
            game_state.demo_snapped_position = None  # Clear snap on deselect
            logger.info("Deselected shape.")
        elif (
            0 <= preview_idx < len(game_state.shapes) and game_state.shapes[preview_idx]
        ):
            # Clicked a valid, available shape: select it
            game_state.demo_selected_shape_idx = preview_idx
            logger.info(f"Selected shape index: {preview_idx}")
            # Immediately update hover based on current mouse pos after selection
            update_play_hover(mouse_pos, game_state, visualizer)
        else:
            # Clicked an empty or invalid slot
            logger.info(f"Clicked empty/invalid preview slot: {preview_idx}")
            # Deselect if clicking an empty slot while another is selected
            if game_state.demo_selected_shape_idx != -1:
                game_state.demo_selected_shape_idx = -1
                game_state.demo_snapped_position = None
        return  # Handled preview click (or lack thereof)

    # 2. Check for clicks on the grid (if a shape is selected)
    selected_idx = game_state.demo_selected_shape_idx
    if selected_idx != -1 and grid_rect and grid_rect.collidepoint(mouse_pos):
        # A shape is selected, and the click is within the grid area.
        # Use coord_mapper from visualization.core
        grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
            mouse_pos, grid_rect, game_state.env_config
        )
        shape_to_place: Optional[Shape] = game_state.shapes[
            selected_idx
        ]  # Use Shape from structs

        # Check if the placement is valid *at the clicked location*
        # Use grid logic from environment.grid
        if (
            grid_coords
            and shape_to_place
            and env_grid.logic.can_place(  # Use function from grid logic
                game_state.grid_data, shape_to_place, grid_coords[0], grid_coords[1]
            )
        ):
            # Valid placement click!
            r, c = grid_coords
            # Use action codec from environment.core
            action = env_core.action_codec.encode_action(
                selected_idx, r, c, game_state.env_config
            )
            # Execute the step using the game state's method (which uses env logic internally)
            _, reward, done = game_state.step(
                action
            )  # GameState.step returns placeholder_val, reward, done
            logger.info(
                f"Placed shape {selected_idx} at {grid_coords}. R={reward:.1f}, Done={done}"
            )
            # Deselect shape after successful placement
            game_state.demo_selected_shape_idx = -1
            game_state.demo_snapped_position = None  # Clear snap state
        else:
            # Clicked grid, shape selected, but not a valid placement spot for the click
            logger.info(f"Clicked grid at {grid_coords}, but placement invalid.")


def update_play_hover(
    mouse_pos: Tuple[int, int], game_state: "GameState", visualizer: "Visualizer"
) -> None:
    """Updates the snapped position based on mouse hover in play mode."""
    game_state.demo_snapped_position = None

    if game_state.is_over() or game_state.demo_selected_shape_idx == -1:
        return

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect:
        return

    shape_idx = game_state.demo_selected_shape_idx
    if not (0 <= shape_idx < len(game_state.shapes)):
        return
    shape: Optional[Shape] = game_state.shapes[shape_idx]  # Use Shape from structs
    if not shape:
        return

    # Use coord_mapper from visualization.core
    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )

    # Use grid logic from environment.grid
    if grid_coords and env_grid.logic.can_place(  # Use function from grid logic
        game_state.grid_data, shape, grid_coords[0], grid_coords[1]
    ):
        game_state.demo_snapped_position = grid_coords
