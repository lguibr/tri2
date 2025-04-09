# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple
import logging

from config import WHITE, LIGHTG, GRAY, YELLOW, GREEN

# Don't import config.general directly in UI process
# import config.general as config_general

logger = logging.getLogger(__name__)


class InfoTextRenderer:
    """Renders essential non-plotted information text based on provided data."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.detail_font = fonts.get("detail", pygame.font.Font(None, 16))
        # stats_summary_cache is not needed as data is passed in each frame

    def _get_network_description(self) -> str:
        """Builds a description string based on network components."""
        return "AlphaZero Neural Network"  # Keep simple for now

    # _render_key_value_line remains the same
    def _render_key_value_line(
        self,
        y_pos: int,
        panel_width: int,
        key: str,
        value: str,
        key_font: pygame.font.Font,
        value_font: pygame.font.Font,
        key_color=LIGHTG,
        value_color=WHITE,
    ) -> int:
        x_pos_key = 10
        x_pos_val_offset = 5
        try:
            key_surf = key_font.render(f"{key}:", True, key_color)
            key_rect = key_surf.get_rect(topleft=(x_pos_key, y_pos))
            self.screen.blit(key_surf, key_rect)
            value_surf = value_font.render(f"{value}", True, value_color)
            value_rect = value_surf.get_rect(
                topleft=(key_rect.right + x_pos_val_offset, y_pos)
            )
            clip_width = max(0, panel_width - value_rect.left - 10)
            blit_area = (
                pygame.Rect(0, 0, clip_width, value_rect.height)
                if value_rect.width > clip_width
                else None
            )
            self.screen.blit(value_surf, value_rect, area=blit_area)
            return max(key_rect.bottom, value_rect.bottom)
        except Exception as e:
            logger.error(f"Error rendering info line '{key}': {e}")
            return y_pos + key_font.get_linesize()

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],
        panel_width: int,
        agent_param_count: int,
        worker_counts: Dict[str, int],
    ) -> int:
        """Renders the info text block based on provided data. Returns next_y."""
        if not self.ui_font or not self.detail_font:
            logger.warning("Missing fonts for InfoTextRenderer.")
            return y_start

        current_y = y_start

        # --- General Info (Extract from stats_summary or passed args) ---
        device_type_str = stats_summary.get(
            "device", "N/A"
        )  # Expect device in summary now
        network_desc = self._get_network_description()
        param_str = (
            f"{agent_param_count / 1e6:.2f} M" if agent_param_count > 0 else "N/A"
        )
        start_time_unix = stats_summary.get("start_time", 0.0)
        start_time_str = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_unix))
            if start_time_unix > 0
            else "N/A"
        )
        sp_workers = worker_counts.get("SelfPlay", 0)
        tr_workers = worker_counts.get("Training", 0)
        worker_str = f"SP: {sp_workers}, TR: {tr_workers}"
        steps_sec = stats_summary.get("steps_per_second_avg", 0.0)
        steps_sec_str = f"{steps_sec:.1f}"

        general_info_lines = [
            ("Device", device_type_str),
            ("Network", network_desc),
            ("Params", param_str),
            ("Workers", worker_str),
            ("Run Started", start_time_str),
            ("Steps/Sec (Avg)", steps_sec_str),
        ]

        # Render lines using the helper function
        line_spacing = 2
        for key, value_str in general_info_lines:
            current_y = (
                self._render_key_value_line(
                    current_y, panel_width, key, value_str, self.ui_font, self.ui_font
                )
                + line_spacing
            )

        return int(current_y) + 5
