# File: ui/panels/left_panel_components/tb_status_renderer.py
import pygame
import os
from typing import Dict, Optional, Tuple
from config import (
    VisConfig,
    TensorBoardConfig,
    GRAY,
    LIGHTG,
    GOOGLE_COLORS,
    WHITE,  # Added WHITE
)


class TBStatusRenderer:
    """Renders the TensorBoard status line."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts

    def _shorten_path(self, path: str, max_chars: int) -> str:
        """Attempts to shorten a path string for display."""
        if len(path) <= max_chars:
            return path
        try:
            rel_path = os.path.relpath(path)
        except ValueError:
            rel_path = path
        if len(rel_path) <= max_chars:
            return rel_path
        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            short_path = os.path.join("...", *parts[-2:])
            if len(short_path) <= max_chars:
                return short_path
        basename = os.path.basename(path)
        return (
            "..." + basename[-(max_chars - 3) :]
            if len(basename) > max_chars - 3
            else basename
        )

    def render(
        self, y_start: int, log_dir: Optional[str], panel_width: int
    ) -> int:  # Removed returning rects
        """Renders the TB status. Returns next_y."""
        # Removed stat_rects initialization
        ui_font, logdir_font = self.fonts.get("ui"), self.fonts.get("logdir")
        if not ui_font or not logdir_font:
            return y_start + 30

        tb_active = (
            TensorBoardConfig.LOG_HISTOGRAMS
            or TensorBoardConfig.LOG_IMAGES
            or TensorBoardConfig.LOG_SHAPE_PLACEMENT_Q_VALUES
        )
        tb_color = GOOGLE_COLORS[0] if tb_active else GRAY
        tb_text = f"TensorBoard: {'Logging Active' if tb_active else 'Logging Minimal'}"

        tb_surf = ui_font.render(tb_text, True, tb_color)
        tb_rect = tb_surf.get_rect(topleft=(10, y_start))
        self.screen.blit(tb_surf, tb_rect)
        # Removed stat_rects update
        last_y = tb_rect.bottom

        if log_dir:
            try:
                panel_char_width = max(
                    10, panel_width // max(1, logdir_font.size("A")[0])
                )
            except Exception:
                panel_char_width = 30  # Fallback
            short_log_dir = self._shorten_path(log_dir, panel_char_width)

            # --- Change color here ---
            dir_surf = logdir_font.render(f"Log Dir: {short_log_dir}", True, WHITE)
            # --- End change color ---
            dir_rect = dir_surf.get_rect(topleft=(10, tb_rect.bottom + 2))
            clip_width = max(0, panel_width - dir_rect.left - 10)
            blit_area = (
                pygame.Rect(0, 0, clip_width, dir_rect.height)
                if dir_rect.width > clip_width
                else None
            )
            self.screen.blit(dir_surf, dir_rect, area=blit_area)

            # Removed combined_tb_rect calculation and stat_rects update
            last_y = dir_rect.bottom

        return last_y  # Return only next_y
