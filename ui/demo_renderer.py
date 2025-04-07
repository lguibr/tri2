# File: ui/demo_renderer.py
import pygame
import math
import traceback
from typing import Tuple, Dict

from config import VisConfig, EnvConfig, DemoConfig, RED
from environment.game_state import GameState
from .panels.game_area import GameAreaRenderer  # Keep for grid rendering logic
from .demo_components.grid_renderer import DemoGridRenderer
from .demo_components.preview_renderer import DemoPreviewRenderer
from .demo_components.hud_renderer import DemoHudRenderer


class DemoRenderer:
    """
    Handles rendering specifically for the interactive Demo/Debug Mode.
    Delegates rendering tasks to sub-components.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,  # Pass GameAreaRenderer for shared logic/fonts
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer  # Keep reference

        # Initialize sub-renderers
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
        self, demo_env: GameState, env_config: EnvConfig, is_debug: bool = False
    ):
        """Renders the entire demo/debug mode screen."""
        if not demo_env:
            print("Error: DemoRenderer called with demo_env=None")
            return

        bg_color = self.hud_renderer.determine_background_color(demo_env)
        self.screen.fill(bg_color)

        screen_width, screen_height = self.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        game_rect, clipped_game_rect = self.grid_renderer.calculate_game_area_rect(
            screen_width, screen_height, padding, hud_height, help_height, env_config
        )

        if clipped_game_rect.width > 10 and clipped_game_rect.height > 10:
            self.grid_renderer.render_game_area(
                demo_env, env_config, clipped_game_rect, bg_color, is_debug
            )
        else:
            self.hud_renderer.render_too_small_message(
                "Demo Area Too Small", clipped_game_rect
            )

        if not is_debug:
            self.shape_preview_rects = self.preview_renderer.render_shape_previews_area(
                demo_env, screen_width, clipped_game_rect, padding
            )
        else:
            self.shape_preview_rects.clear()

        self.hud_renderer.render_hud(
            demo_env, screen_width, game_rect.bottom + 10, is_debug
        )
        self.hud_renderer.render_help_text(screen_width, screen_height, is_debug)

    # Expose calculation methods if needed by InputHandler
    def _calculate_game_area_rect(self, *args, **kwargs):
        return self.grid_renderer.calculate_game_area_rect(*args, **kwargs)

    def _calculate_demo_triangle_size(self, *args, **kwargs):
        return self.grid_renderer.calculate_demo_triangle_size(*args, **kwargs)

    def _calculate_grid_offset(self, *args, **kwargs):
        return self.grid_renderer.calculate_grid_offset(*args, **kwargs)

    def get_shape_preview_rects(self) -> Dict[int, pygame.Rect]:
        """Returns the dictionary of screen-relative shape preview rects."""
        # Get rects from the preview renderer
        return self.preview_renderer.get_shape_preview_rects()
