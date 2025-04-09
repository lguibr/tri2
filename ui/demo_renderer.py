# File: ui/demo_renderer.py
import pygame
import math
import traceback
from typing import Tuple, Dict, Any, Optional

from config import VisConfig, EnvConfig, DemoConfig, RED

# from environment.game_state import GameState # No longer use GameState directly
from .panels.game_area import GameAreaRenderer  # Keep for rendering logic if needed
from .demo_components.grid_renderer import DemoGridRenderer
from .demo_components.preview_renderer import DemoPreviewRenderer
from .demo_components.hud_renderer import DemoHudRenderer


class DemoRenderer:
    """
    Handles rendering for Demo/Debug Mode based on data received from logic process.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        # GameAreaRenderer might not be needed if all logic is self-contained
        self.game_area_renderer = game_area_renderer

        # Initialize sub-renderers (pass screen, configs)
        self.grid_renderer = DemoGridRenderer(
            screen, vis_config, demo_config, game_area_renderer
        )
        self.preview_renderer = DemoPreviewRenderer(
            screen, vis_config, demo_config, game_area_renderer
        )
        self.hud_renderer = DemoHudRenderer(
            screen, vis_config, demo_config, game_area_renderer
        )

        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

    def render(
        self,
        demo_env_data: Dict[str, Any],  # Now accepts a data dictionary
        env_config: Optional[Dict[str, Any]] = None,  # Env config values as dict
        is_debug: bool = False,
    ):
        """Renders the entire demo/debug mode screen using provided data."""
        if not demo_env_data or not env_config:
            print("Error: DemoRenderer called with missing data or env_config")
            # Optionally render an error message
            return

        # Extract necessary info from demo_env_data
        # This replaces accessing demo_env object attributes
        is_over = demo_env_data.get("demo_env_is_over", False)
        score = demo_env_data.get("demo_env_score", 0)
        state_dict = demo_env_data.get("demo_env_state")  # The StateType dict
        dragged_shape_idx = demo_env_data.get("demo_env_dragged_shape_idx")
        snapped_pos = demo_env_data.get("demo_env_snapped_pos")
        selected_shape_idx = demo_env_data.get("demo_env_selected_shape_idx", -1)
        # Get shape data (assuming it's part of the state_dict or stats)
        available_shapes_data = []
        if (
            state_dict and "shapes" in state_dict
        ):  # Placeholder: Need actual shape info passed
            pass  # Need to reconstruct shape info for previews if not passed separately

        # Determine background color based on state flags (passed in demo_env_data)
        # bg_color = self.hud_renderer.determine_background_color(demo_env_data) # Adapt this method
        bg_color = self.demo_config.BACKGROUND_COLOR  # Simplified for now
        self.screen.fill(bg_color)

        screen_width, screen_height = self.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        # Calculate game area using env_config dict
        game_rect, clipped_game_rect = self.grid_renderer.calculate_game_area_rect(
            screen_width, screen_height, padding, hud_height, help_height, env_config
        )

        if clipped_game_rect.width > 10 and clipped_game_rect.height > 10:
            # Pass necessary data down to grid renderer
            self.grid_renderer.render_game_area(
                demo_env_data,  # Pass the data dict
                env_config,
                clipped_game_rect,
                bg_color,
                is_debug,
            )
        else:
            self.hud_renderer.render_too_small_message(
                "Demo Area Too Small", clipped_game_rect
            )

        if not is_debug:
            # Pass necessary data down to preview renderer
            self.shape_preview_rects = self.preview_renderer.render_shape_previews_area(
                demo_env_data,  # Pass the data dict
                screen_width,
                clipped_game_rect,
                padding,
            )
        else:
            self.shape_preview_rects.clear()

        # Pass necessary data down to HUD renderer
        self.hud_renderer.render_hud(
            demo_env_data,  # Pass the data dict
            screen_width,
            game_rect.bottom + 10,
            is_debug,
        )
        self.hud_renderer.render_help_text(screen_width, screen_height, is_debug)

    # Expose calculation methods if needed by InputHandler (unlikely now)
    # ...

    def get_shape_preview_rects(self) -> Dict[int, pygame.Rect]:
        """Returns the dictionary of screen-relative shape preview rects."""
        return self.preview_renderer.get_shape_preview_rects()
