# File: ui/panels/left_panel_components/tb_status_renderer.py
import pygame
import os
from typing import Dict, Optional, Tuple
from config import (
    VisConfig,
    TensorBoardConfig,  # Keep for potential future use, but not needed now
    GRAY,
    LIGHTG,
    GOOGLE_COLORS,
    WHITE,
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
            # Attempt to get relative path first
            rel_path = os.path.relpath(path)
            if len(rel_path) <= max_chars:
                return rel_path
        except ValueError:
            # Fallback if relpath fails (e.g., different drives on Windows)
            pass

        # If relative path is still too long or failed, use ellipsis for parent dirs
        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            # Show ".../last_dir/run_id" or similar
            short_path = os.path.join("...", *parts[-2:])
            if len(short_path) <= max_chars:
                return short_path

        # If even that is too long, show ellipsis and end of the basename
        basename = os.path.basename(path)
        if len(basename) > max_chars:
            return "..." + basename[-(max_chars - 3) :]
        else:
            return (
                basename  # Should not happen if previous checks failed, but as fallback
            )

    def render(
        self,
        y_start: int,
        tb_log_dir: Optional[str],
        panel_width: int,  # Add panel_width
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the TensorBoard status block."""
        stat_rects: Dict[str, pygame.Rect] = {}
        tb_status_font = self.fonts.get("tb_status")  # Use the correct font key
        if not tb_status_font:  # Check font existence
            print("Warning: TB Status font ('tb_status') not found.")
            return y_start, stat_rects  # Return original y_start if font missing

        x_margin = 10
        current_y = y_start
        line_height = tb_status_font.get_linesize()  # Use the correct font

        if tb_log_dir:
            # Use panel_width passed as argument
            max_chars_for_path = max(
                30, int(panel_width * 0.15)
            )  # Adjust multiplier as needed
            tb_path_short = self._shorten_path(tb_log_dir, max_chars_for_path)
            tb_text = f"TB Logs: {tb_path_short}"
            tb_color = WHITE
        else:
            tb_text = "TB Logs: Not Configured"
            tb_color = GRAY

        try:
            tb_surf = tb_status_font.render(tb_text, True, tb_color)
            tb_rect = tb_surf.get_rect(topleft=(x_margin, current_y))

            # Clip rendering if text exceeds panel width
            clip_width = max(0, panel_width - tb_rect.left - x_margin)
            blit_area = None
            if tb_rect.width > clip_width:
                blit_area = pygame.Rect(0, 0, clip_width, tb_rect.height)

            self.screen.blit(tb_surf, tb_rect, area=blit_area)
            stat_rects["TensorBoard Path"] = tb_rect
            current_y += line_height  # Increment y position correctly
        except Exception as e:
            print(f"Error rendering TB status: {e}")
            # Don't increment current_y if rendering failed, return original y_start
            return y_start, stat_rects

        return (
            int(current_y),
            stat_rects,
        )  # Return the potentially incremented y position as int
