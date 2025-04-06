import pygame
import os
from typing import Dict, Optional, Tuple
from config import VisConfig, TensorBoardConfig


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
        except ValueError:  # Handle different drives on Windows
            rel_path = path
        if len(rel_path) <= max_chars:
            return rel_path

        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            short_path = os.path.join("...", *parts[-2:])
            if len(short_path) <= max_chars:
                return short_path
        # Fallback: Ellipsis + end of basename
        basename = os.path.basename(path)
        return (
            "..." + basename[-(max_chars - 3) :]
            if len(basename) > max_chars - 3
            else basename
        )

    def render(
        self, y_start: int, log_dir: Optional[str], panel_width: int
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the TB status. Returns next_y and stat_rects."""
        stat_rects: Dict[str, pygame.Rect] = {}
        ui_font = self.fonts.get("ui")
        logdir_font = self.fonts.get("logdir")
        if not ui_font or not logdir_font:
            return y_start + 30, stat_rects

        tb_active = (
            TensorBoardConfig.LOG_HISTOGRAMS
            or TensorBoardConfig.LOG_IMAGES
            or TensorBoardConfig.LOG_SHAPE_PLACEMENT_Q_VALUES
        )
        tb_color = VisConfig.GOOGLE_COLORS[0] if tb_active else VisConfig.GRAY
        tb_text = f"TensorBoard: {'Logging Active' if tb_active else 'Logging Minimal'}"

        tb_surf = ui_font.render(tb_text, True, tb_color)
        tb_rect = tb_surf.get_rect(topleft=(10, y_start))
        self.screen.blit(tb_surf, tb_rect)
        stat_rects["TensorBoard Status"] = tb_rect
        last_y = tb_rect.bottom

        if log_dir:
            try:
                panel_char_width = max(
                    10, panel_width // max(1, logdir_font.size("A")[0])
                )
                short_log_dir = self._shorten_path(log_dir, panel_char_width)
            except Exception:
                short_log_dir = os.path.basename(log_dir)

            dir_surf = logdir_font.render(
                f"Log Dir: {short_log_dir}", True, VisConfig.LIGHTG
            )
            dir_rect = dir_surf.get_rect(topleft=(10, tb_rect.bottom + 2))

            clip_width = max(0, panel_width - dir_rect.left - 10)
            if dir_rect.width > clip_width:
                self.screen.blit(
                    dir_surf,
                    dir_rect,
                    area=pygame.Rect(0, 0, clip_width, dir_rect.height),
                )
            else:
                self.screen.blit(dir_surf, dir_rect)

            combined_tb_rect = tb_rect.union(dir_rect)
            combined_tb_rect.width = min(
                combined_tb_rect.width, panel_width - 10 - combined_tb_rect.left
            )
            stat_rects["TensorBoard Status"] = combined_tb_rect
            last_y = dir_rect.bottom

        return last_y, stat_rects
