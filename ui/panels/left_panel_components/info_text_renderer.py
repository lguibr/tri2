# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
from typing import Dict, Any, Tuple
from config import (
    VisConfig,
    StatsConfig,
    PPOConfig,
    RNNConfig,
    DEVICE,
    TOTAL_TRAINING_STEPS,
)


class InfoTextRenderer:
    """Renders the main block of statistics text."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],
        panel_width: int,
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the info text block. Returns next_y and stat_rects."""
        stat_rects: Dict[str, pygame.Rect] = {}
        ui_font = self.fonts.get("ui")
        if not ui_font:
            return y_start + 100, stat_rects

        line_height = ui_font.get_linesize()
        global_step = stats_summary.get("global_step", 0)

        # --- MODIFIED: Use specific PPO metric keys ---
        info_lines = [
            (
                "Global Steps",
                f"{global_step/1e6:.2f}M / {TOTAL_TRAINING_STEPS/1e6:.1f}M",
            ),
            ("Total Episodes", f"{stats_summary.get('total_episodes', 0)}"),
            (
                "Steps/Sec (Current)",
                f"{stats_summary.get('steps_per_second', 0.0):.1f}",
            ),
            (
                "Policy Loss",
                f"{stats_summary.get('policy_loss', 0.0):.4f}",
            ),  # Use 'policy_loss'
            (
                "Value Loss",
                f"{stats_summary.get('value_loss', 0.0):.4f}",
            ),  # Use 'value_loss'
            ("Entropy", f"{stats_summary.get('entropy', 0.0):.4f}"),  # Use 'entropy'
            ("Learning Rate", f"{stats_summary.get('current_lr', 0.0):.1e}"),
            ("Device", f"{DEVICE.type.upper()}"),
            (
                "Network",
                f"Actor-Critic (CNN+MLP->LSTM:{RNNConfig.USE_RNN})",
            ),
        ]
        # --- END MODIFIED ---

        last_y = y_start
        x_pos_key, x_pos_val_offset = 10, 5

        for idx, (key, value_str) in enumerate(info_lines):
            current_y = y_start + idx * line_height
            try:
                key_surf = ui_font.render(f"{key}:", True, VisConfig.LIGHTG)
                key_rect = key_surf.get_rect(topleft=(x_pos_key, current_y))
                self.screen.blit(key_surf, key_rect)

                value_surf = ui_font.render(f"{value_str}", True, VisConfig.WHITE)
                value_rect = value_surf.get_rect(
                    topleft=(key_rect.right + x_pos_val_offset, current_y)
                )

                clip_width = max(0, panel_width - value_rect.left - 10)
                if value_rect.width > clip_width:
                    self.screen.blit(
                        value_surf,
                        value_rect,
                        area=pygame.Rect(0, 0, clip_width, value_rect.height),
                    )
                else:
                    self.screen.blit(value_surf, value_rect)

                combined_rect = key_rect.union(value_rect)
                combined_rect.width = min(
                    combined_rect.width, panel_width - x_pos_key - 10
                )
                stat_rects[key] = combined_rect
                last_y = combined_rect.bottom
            except Exception as e:
                print(f"Error rendering stat line '{key}': {e}")
                last_y = current_y + line_height

        return last_y, stat_rects
