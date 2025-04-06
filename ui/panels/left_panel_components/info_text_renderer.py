import pygame
from typing import Dict, Any, Tuple

from config import (
    RNNConfig,
    TransformerConfig,
    WHITE,
    LIGHTG,
)
from config.general import DEVICE


class InfoTextRenderer:
    """Renders essential non-plotted information text."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.rnn_config = RNNConfig()
        self.transformer_config = TransformerConfig()

    def _get_network_description(self) -> str:
        """Builds a description string based on network components."""
        parts = ["CNN+MLP Fusion"]
        if self.transformer_config.USE_TRANSFORMER:
            parts.append("Transformer")
        if self.rnn_config.USE_RNN:
            parts.append("LSTM")
        if len(parts) == 1:  # Only Fusion
            return "Actor-Critic (CNN+MLP Fusion)"
        else:
            return f"Actor-Critic ({' -> '.join(parts)})"

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
            return y_start, stat_rects

        line_height = ui_font.get_linesize()

        device_type_str = "UNKNOWN"
        if DEVICE and hasattr(DEVICE, "type"):
            device_type_str = DEVICE.type.upper()

        network_desc = self._get_network_description()
        info_lines = [
            ("Device", device_type_str),
            ("Network", network_desc),
        ]

        last_y = y_start
        x_pos_key, x_pos_val_offset = 10, 5
        current_y = y_start + 5

        for idx, (key, value_str) in enumerate(info_lines):
            line_y = current_y + idx * line_height
            try:
                key_surf = ui_font.render(f"{key}:", True, LIGHTG)
                key_rect = key_surf.get_rect(topleft=(x_pos_key, line_y))
                self.screen.blit(key_surf, key_rect)

                value_surf = ui_font.render(f"{value_str}", True, WHITE)
                value_rect = value_surf.get_rect(
                    topleft=(key_rect.right + x_pos_val_offset, line_y)
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
                last_y = line_y + line_height

        return last_y, stat_rects
