# File: app_ui_utils.py
import pygame
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from main_pygame import MainApp
    from environment.game_state import GameState
    from ui.renderer import UIRenderer


class AppUIUtils:
    """Utility functions related to mapping screen coordinates to game elements."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def map_screen_to_grid(
        self, screen_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Maps screen coordinates to grid row/column for demo/debug."""
        if (
            self.app.renderer is None
            or self.app.initializer.demo_env is None
            or self.app.renderer.demo_renderer is None
        ):
            return None
        if self.app.app_state not in [
            self.app.app_state.PLAYING,
            self.app.app_state.DEBUG,
        ]:
            return None

        demo_env: "GameState" = self.app.initializer.demo_env
        renderer: "UIRenderer" = self.app.renderer

        screen_width, screen_height = self.app.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        _, clipped_game_rect = renderer.demo_renderer._calculate_game_area_rect(
            screen_width,
            screen_height,
            padding,
            hud_height,
            help_height,
            self.app.env_config,
        )

        if not clipped_game_rect.collidepoint(screen_pos):
            return None

        relative_x = screen_pos[0] - clipped_game_rect.left
        relative_y = screen_pos[1] - clipped_game_rect.top

        tri_cell_w, tri_cell_h = renderer.demo_renderer._calculate_demo_triangle_size(
            clipped_game_rect.width, clipped_game_rect.height, self.app.env_config
        )
        grid_ox, grid_oy = renderer.demo_renderer._calculate_grid_offset(
            clipped_game_rect.width, clipped_game_rect.height, self.app.env_config
        )

        if tri_cell_w <= 0 or tri_cell_h <= 0:
            return None

        grid_relative_x = relative_x - grid_ox
        grid_relative_y = relative_y - grid_oy

        # Approximate calculation (might need refinement based on triangle geometry)
        approx_row = int(grid_relative_y / tri_cell_h)
        approx_col = int(grid_relative_x / (tri_cell_w * 0.75))

        if (
            0 <= approx_row < self.app.env_config.ROWS
            and 0 <= approx_col < self.app.env_config.COLS
        ):
            if (
                demo_env.grid.valid(approx_row, approx_col)
                and not demo_env.grid.triangles[approx_row][approx_col].is_death
            ):
                return approx_row, approx_col
        return None

    def map_screen_to_preview(self, screen_pos: Tuple[int, int]) -> Optional[int]:
        """Maps screen coordinates to a shape preview index."""
        if (
            self.app.renderer is None
            or self.app.initializer.demo_env is None
            or self.app.input_handler is None
        ):
            return None
        if self.app.app_state != self.app.app_state.PLAYING:
            return None

        # Access preview rects directly from the input handler
        if hasattr(self.app.input_handler, "shape_preview_rects"):
            for idx, rect in self.app.input_handler.shape_preview_rects.items():
                if rect.collidepoint(screen_pos):
                    return idx
        return None
