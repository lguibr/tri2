# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple
import logging

from config import WHITE, LIGHTG, GRAY, YELLOW, GREEN  # Added YELLOW, GREEN
import config.general as config_general

logger = logging.getLogger(__name__)


class InfoTextRenderer:
    """Renders essential non-plotted information text, including MCTS stats."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.detail_font = fonts.get("detail", pygame.font.Font(None, 16))
        self.mcts_label_font = fonts.get("mcts_stats_label", pygame.font.Font(None, 18))
        self.mcts_value_font = fonts.get("mcts_stats_value", pygame.font.Font(None, 18))
        self.stats_summary_cache: Dict[str, Any] = {}

    def _get_network_description(self) -> str:
        """Builds a description string based on network components."""
        # Placeholder, could be made more dynamic later if needed
        return "AlphaZero Neural Network"

    def _render_key_value_line(
        self,
        y_pos: int,
        panel_width: int,
        key: str,
        value: str,
        key_font,
        value_font,
        key_color=LIGHTG,
        value_color=WHITE,
    ) -> int:
        """Helper to render a single key-value line and return its bottom y."""
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

            # Clip value if it exceeds panel width
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
            return y_pos + key_font.get_linesize()  # Fallback height

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],
        panel_width: int,
        agent_param_count: int,
        worker_counts: Dict[str, int],
    ) -> int:
        """Renders the info text block including MCTS stats. Returns next_y."""
        self.stats_summary_cache = stats_summary

        if (
            not self.ui_font
            or not self.detail_font
            or not self.mcts_label_font
            or not self.mcts_value_font
        ):
            logger.warning("Missing fonts for InfoTextRenderer.")
            return y_start

        current_y = y_start

        # --- General Info ---
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

        general_info_lines = [
            ("Device", device_type_str),
            ("Network", network_desc),
            ("Params", param_str),
            ("Workers", worker_str),
            ("Run Started", start_time_str),
        ]

        for key, value_str in general_info_lines:
            current_y = (
                self._render_key_value_line(
                    current_y, panel_width, key, value_str, self.ui_font, self.ui_font
                )
                + 2
            )  # Add small gap

        current_y += 5  # Extra gap before MCTS stats

        # --- MCTS Stats ---
        mcts_sim_t_avg = stats_summary.get("avg_mcts_sim_time_window", 0.0) * 1000  # ms
        mcts_nn_t_avg = stats_summary.get("avg_mcts_nn_time_window", 0.0) * 1000  # ms
        mcts_nodes_avg = stats_summary.get("avg_mcts_nodes_explored_window", 0)
        mcts_depth_avg = stats_summary.get("avg_mcts_avg_depth_window", 0.0)

        mcts_sim_t_inst = stats_summary.get("mcts_sim_time", 0.0) * 1000  # ms
        mcts_nn_t_inst = stats_summary.get("mcts_nn_time", 0.0) * 1000  # ms
        mcts_nodes_inst = stats_summary.get("mcts_nodes_explored", 0)
        mcts_depth_inst = stats_summary.get("mcts_avg_depth", 0.0)

        avg_win_size = stats_summary.get("summary_avg_window_size", "?")

        mcts_lines = [
            (
                f"MCTS Sim Time (Avg{avg_win_size})",
                f"{mcts_sim_t_avg:.1f} ms",
                f"(Now: {mcts_sim_t_inst:.1f} ms)",
            ),
            (
                f"MCTS NN Time (Avg{avg_win_size})",
                f"{mcts_nn_t_avg:.1f} ms",
                f"(Now: {mcts_nn_t_inst:.1f} ms)",
            ),
            (
                f"MCTS Nodes (Avg{avg_win_size})",
                f"{mcts_nodes_avg:.0f}",
                f"(Now: {mcts_nodes_inst})",
            ),
            (
                f"MCTS Depth (Avg{avg_win_size})",
                f"{mcts_depth_avg:.1f}",
                f"(Now: {mcts_depth_inst:.1f})",
            ),
        ]

        for key, value_avg_str, value_now_str in mcts_lines:
            # Render Avg Value
            line_bottom = self._render_key_value_line(
                current_y,
                panel_width,
                key,
                value_avg_str,
                self.mcts_label_font,
                self.mcts_value_font,
                key_color=YELLOW,
                value_color=WHITE,
            )
            # Render "Now" value next to it if space allows
            try:
                avg_val_surf = self.mcts_value_font.render(value_avg_str, True, WHITE)
                avg_val_rect = avg_val_surf.get_rect(
                    topleft=(
                        self.mcts_label_font.size(f"{key}: ")[0] + 10 + 5,
                        current_y,
                    )
                )

                now_surf = self.detail_font.render(value_now_str, True, GRAY)
                now_rect = now_surf.get_rect(
                    topleft=(avg_val_rect.right + 8, current_y + 2)
                )

                if now_rect.right < panel_width - 10:
                    self.screen.blit(now_surf, now_rect)
            except Exception as e:
                logger.debug(f"Could not render 'now' value for {key}: {e}")

            current_y = line_bottom + 2  # Add small gap

        return int(current_y) + 5  # Add final padding
