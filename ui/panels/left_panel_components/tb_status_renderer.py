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
            if len(basename) > max_chars
            else basename
        )

    def render(
        self, y_start: int, tb_info: Dict[str, Optional[TensorBoardConfig]]
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the TensorBoard status block."""
        stat_rects: Dict[str, pygame.Rect] = {}
        if not tb_info or not self.fonts.get("tb_status"):
            return y_start, stat_rects

        x_margin = 10
        current_y = y_start
        line_height = self.fonts["tb_status"].get_linesize()

        tb_path = tb_info.get("tb_path", None)
        if tb_path:
            tb_path = self._shorten_path(tb_path, 50)
