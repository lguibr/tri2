# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple
import logging

from config import WHITE, LIGHTG, GRAY

# Import the module itself, not the variable directly
import config.general as config_general

logger = logging.getLogger(__name__)


class InfoTextRenderer:
    """Renders essential non-plotted information text.
    Refactored for AlphaZero focus and worker counts. Resource usage removed."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.detail_font = fonts.get("detail", pygame.font.Font(None, 16))
        # Resource font removed
        self.stats_summary_cache: Dict[str, Any] = {}

    def _get_network_description(self) -> str:
        """Builds a description string based on network components."""
        return "AlphaZero Neural Network"

    # _get_live_resource_usage method removed

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],
        panel_width: int,
        agent_param_count: int,
        worker_counts: Dict[str, int],
    ) -> int:
        """Renders the info text block. Returns next_y."""
        self.stats_summary_cache = stats_summary

        if not self.ui_font or not self.detail_font:
            logger.warning("Missing fonts for InfoTextRenderer.")
            return y_start

        line_height_ui = self.ui_font.get_linesize()
        # line_height_resource removed

        device_type_str = (
            config_general.DEVICE.type.upper() if config_general.DEVICE else "CPU"
        )

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

        info_lines = [
            ("Device", device_type_str),  # Uses the correctly accessed value now
            ("Network", network_desc),
            ("Params", param_str),
            ("Workers", worker_str),
            ("Run Started", start_time_str),
        ]

        last_y, x_pos_key, x_pos_val_offset, current_y = y_start, 10, 5, y_start

        # Render Key-Value Info Lines
        for idx, (key, value_str) in enumerate(info_lines):
            line_y = current_y + idx * line_height_ui
            try:
                key_surf = self.ui_font.render(f"{key}:", True, LIGHTG)
                key_rect = key_surf.get_rect(topleft=(x_pos_key, line_y))
                self.screen.blit(key_surf, key_rect)

                value_surf = self.ui_font.render(f"{value_str}", True, WHITE)
                value_rect = value_surf.get_rect(
                    topleft=(key_rect.right + x_pos_val_offset, line_y)
                )

                clip_width = max(0, panel_width - value_rect.left - 10)
                blit_area = (
                    pygame.Rect(0, 0, clip_width, value_rect.height)
                    if value_rect.width > clip_width
                    else None
                )
                self.screen.blit(value_surf, value_rect, area=blit_area)

                last_y = key_rect.union(value_rect).bottom
            except Exception as e:
                logger.error(f"Error rendering stat line '{key}': {e}")
                last_y = line_y + line_height_ui

        current_y = last_y + 4

        # Live Resource Usage rendering removed

        return int(last_y)
