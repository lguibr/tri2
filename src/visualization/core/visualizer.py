# File: src/visualization/core/visualizer.py
import pygame
import logging
from typing import TYPE_CHECKING, Dict, Optional

# Use relative imports within the visualization package
from . import colors, layout, coord_mapper

# Import drawing functions/modules directly from their files
from ..drawing import grid as grid_drawing
from ..drawing import previews as preview_drawing
from ..drawing import hud as hud_drawing
from ..drawing import highlight as highlight_drawing

# Type hinting imports from other top-level packages
if TYPE_CHECKING:
    from src.config import VisConfig, EnvConfig
    from src.environment.core.game_state import GameState

logger = logging.getLogger(__name__)


class Visualizer:
    """Orchestrates rendering of the game state for interactive modes."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: Dict[str, Optional[pygame.font.Font]],
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.layout_rects: Optional[Dict[str, pygame.Rect]] = None
        # Cache for preview area rects (mapping index to screen Rect)
        self.preview_rects: Dict[int, pygame.Rect] = {}
        self.ensure_layout()

    def ensure_layout(self) -> Dict[str, pygame.Rect]:
        """Returns cached layout or calculates it if needed."""
        current_w, current_h = self.screen.get_size()
        layout_w, layout_h = 0, 0
        if self.layout_rects:
            # Estimate expected size based on current layout
            grid_r = self.layout_rects.get("grid")
            preview_r = self.layout_rects.get("preview")
            hud_h = self.vis_config.HUD_HEIGHT
            pad = self.vis_config.PADDING
            if grid_r and preview_r:
                layout_w = grid_r.width + preview_r.width + 3 * pad
                layout_h = max(grid_r.height, preview_r.height) + 2 * pad + hud_h
            elif grid_r:
                layout_w = grid_r.width + 2 * pad
                layout_h = grid_r.height + 2 * pad + hud_h

        # Recalculate if layout is missing or screen size mismatch significantly
        if self.layout_rects is None or layout_w != current_w or layout_h != current_h:
            self.layout_rects = layout.calculate_layout(
                current_w, current_h, self.vis_config
            )
            logger.info(f"Recalculated layout: {self.layout_rects}")
            self.preview_rects = {}  # Clear preview cache on layout change
        return self.layout_rects

    def render(self, game_state: "GameState", mode: str):
        """Renders the entire game visualization for interactive modes."""
        self.screen.fill(colors.GRID_BG_DEFAULT)
        layout_rects = self.ensure_layout()
        grid_rect = layout_rects.get("grid")
        preview_rect = layout_rects.get("preview")

        # --- Render Grid Area ---
        if grid_rect and grid_rect.width > 0 and grid_rect.height > 0:
            try:
                grid_surf = self.screen.subsurface(grid_rect)
                self._render_grid_area(grid_surf, game_state, mode, grid_rect)
            except ValueError as e:
                logger.error(f"Error creating grid subsurface ({grid_rect}): {e}")
                pygame.draw.rect(self.screen, colors.RED, grid_rect, 1)

        # --- Render Preview Area ---
        if preview_rect and preview_rect.width > 0 and preview_rect.height > 0:
            try:
                preview_surf = self.screen.subsurface(preview_rect)
                self._render_preview_area(preview_surf, game_state, mode, preview_rect)
            except ValueError as e:
                logger.error(f"Error creating preview subsurface ({preview_rect}): {e}")
                pygame.draw.rect(self.screen, colors.RED, preview_rect, 1)

        # --- Render HUD (Pass None for display_stats in interactive mode) ---
        hud_drawing.render_hud(
            self.screen, game_state, mode, self.fonts, display_stats=None
        )

    def _render_grid_area(
        self,
        grid_surf: pygame.Surface,
        game_state: "GameState",
        mode: str,
        grid_rect: pygame.Rect,
    ):
        """Renders the main game grid and overlays onto the provided grid_surf."""
        bg_color = (
            colors.GRID_BG_GAME_OVER if game_state.is_over() else colors.GRID_BG_DEFAULT
        )
        grid_drawing.draw_grid_background(grid_surf, bg_color)
        grid_drawing.draw_grid_triangles(
            grid_surf, game_state.grid_data, self.env_config
        )

        if mode == "play" and game_state.demo_selected_shape_idx != -1:
            self._draw_play_previews(grid_surf, game_state, grid_rect)

        if mode == "debug" and game_state.debug_highlight_pos:
            r, c = game_state.debug_highlight_pos
            highlight_drawing.draw_debug_highlight(grid_surf, r, c, self.env_config)

    def _draw_play_previews(
        self, grid_surf: pygame.Surface, game_state: "GameState", grid_rect: pygame.Rect
    ):
        """Draws placement or floating previews in play mode onto grid_surf."""
        shape_idx = game_state.demo_selected_shape_idx
        if not (0 <= shape_idx < len(game_state.shapes)):
            return
        shape = game_state.shapes[shape_idx]
        if not shape:
            return

        snapped_pos = game_state.demo_snapped_position
        if snapped_pos:
            preview_drawing.draw_placement_preview(
                grid_surf,
                shape,
                snapped_pos[0],
                snapped_pos[1],
                is_valid=True,
                config=self.env_config,
            )
        else:
            mouse_screen_pos = pygame.mouse.get_pos()
            if grid_rect.collidepoint(mouse_screen_pos):
                mouse_local_x = mouse_screen_pos[0] - grid_rect.left
                mouse_local_y = mouse_screen_pos[1] - grid_rect.top
                if grid_surf.get_rect().collidepoint(mouse_local_x, mouse_local_y):
                    preview_drawing.draw_floating_preview(
                        grid_surf,
                        shape,
                        (mouse_local_x, mouse_local_y),
                        self.env_config,
                    )

    def _render_preview_area(
        self,
        preview_surf: pygame.Surface,
        game_state: "GameState",
        mode: str,
        preview_rect: pygame.Rect,
    ):
        """Renders the shape preview slots onto preview_surf and caches rects."""
        self.preview_rects = preview_drawing.render_previews(
            preview_surf,
            game_state,
            preview_rect.topleft,
            mode,
            self.env_config,
            self.vis_config,
        )
