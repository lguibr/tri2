# File: ui/panels/left_panel_components/info_text_renderer.py
# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
import time
import psutil
import torch
from typing import Dict, Any, Tuple

from config import (
    RNNConfig,
    TransformerConfig,
    ModelConfig,
    WHITE,
    LIGHTG,
    GRAY,
)
from config.general import DEVICE


class InfoTextRenderer:
    """Renders essential non-plotted information text, including network config and live resources."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.rnn_config = RNNConfig()
        self.transformer_config = TransformerConfig()
        self.model_config_net = ModelConfig.Network()
        self.resource_font = fonts.get("logdir", pygame.font.Font(None, 16))
        self.stats_summary_cache: Dict[str, Any] = (
            {}
        )  # Cache summary for resource usage

    def _get_network_description(self) -> str:
        """Builds a description string based on network components."""
        parts = ["CNN+MLP Fusion"]
        if self.transformer_config.USE_TRANSFORMER:
            parts.append("Transformer")
        if self.rnn_config.USE_RNN:
            parts.append("LSTM")
        if len(parts) == 1:
            return "Actor-Critic (CNN+MLP Fusion)"
        else:
            return f"Actor-Critic ({' -> '.join(parts)})"

    def _get_network_details(self) -> str:
        """Builds a detailed string of network configuration."""
        details = []
        cnn_str = str(self.model_config_net.CONV_CHANNELS).replace(" ", "")
        details.append(
            f"CNN: {cnn_str} (K={self.model_config_net.CONV_KERNEL_SIZE}, S={self.model_config_net.CONV_STRIDE}, P={self.model_config_net.CONV_PADDING})"
        )
        shape_mlp_str = str(self.model_config_net.SHAPE_FEATURE_MLP_DIMS).replace(
            " ", ""
        )
        details.append(f"Shape MLP: {shape_mlp_str}")
        fusion_mlp_str = str(self.model_config_net.COMBINED_FC_DIMS).replace(" ", "")
        details.append(f"Fusion MLP: {fusion_mlp_str}")
        if self.transformer_config.USE_TRANSFORMER:
            details.append(
                f"Transformer: L={self.transformer_config.TRANSFORMER_NUM_LAYERS}, H={self.transformer_config.TRANSFORMER_NHEAD}, D={self.transformer_config.TRANSFORMER_D_MODEL}"
            )
        if self.rnn_config.USE_RNN:
            details.append(f"LSTM: H={self.rnn_config.LSTM_HIDDEN_SIZE}")
        return " | ".join(details)

    def _get_live_resource_usage(self) -> Dict[str, str]:
        """Fetches live CPU, Memory, and GPU Memory usage from cached summary."""
        usage = {"CPU": "N/A", "Mem": "N/A", "GPU Mem": "N/A"}
        # Use cached summary data instead of calling psutil/torch directly here
        cpu_val = self.stats_summary_cache.get("current_cpu_usage")
        mem_val = self.stats_summary_cache.get("current_memory_usage")
        gpu_val = self.stats_summary_cache.get(
            "current_gpu_memory_usage_percent"
        )  # Use percentage

        if cpu_val is not None:
            usage["CPU"] = f"{cpu_val:.1f}%"
        if mem_val is not None:
            usage["Mem"] = f"{mem_val:.1f}%"
        if gpu_val is not None:
            if DEVICE.type == "cuda":
                usage["GPU Mem"] = f"{gpu_val:.1f}%"  # Display percentage
            else:
                usage["GPU Mem"] = "N/A (CPU)"
        elif DEVICE.type != "cuda":
            usage["GPU Mem"] = "N/A (CPU)"

        return usage

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],
        panel_width: int,
        agent_param_count: int,
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the info text block. Returns next_y and stat_rects."""
        self.stats_summary_cache = stats_summary  # Cache the summary for resource usage
        stat_rects: Dict[str, pygame.Rect] = {}
        ui_font = self.fonts.get("ui")
        detail_font = self.fonts.get("logdir")
        resource_font = self.resource_font
        if not ui_font or not detail_font or not resource_font:
            return y_start, stat_rects

        line_height_ui = ui_font.get_linesize()
        line_height_detail = detail_font.get_linesize()
        line_height_resource = resource_font.get_linesize()

        device_type_str = "UNKNOWN"
        if DEVICE and hasattr(DEVICE, "type"):
            device_type_str = DEVICE.type.upper()

        network_desc = self._get_network_description()
        network_details = self._get_network_details()
        param_str = (
            f"{agent_param_count / 1e6:.2f} M" if agent_param_count > 0 else "N/A"
        )

        start_time_unix = stats_summary.get("start_time", 0.0)
        start_time_str = "N/A"
        if start_time_unix > 0:
            try:
                start_time_str = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(start_time_unix)
                )
            except ValueError:
                start_time_str = "Invalid Date"

        info_lines = [
            ("Device", device_type_str),
            ("Network", network_desc),
            ("Params", param_str),
            ("Run Started", start_time_str),
        ]

        last_y = y_start
        x_pos_key, x_pos_val_offset = 10, 5
        current_y = y_start + 5

        # Render standard info lines
        for idx, (key, value_str) in enumerate(info_lines):
            line_y = current_y + idx * line_height_ui
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
                last_y = line_y + line_height_ui

        # Render Network Details
        current_y = last_y + 2
        try:
            detail_surf = detail_font.render(network_details, True, GRAY)
            detail_rect = detail_surf.get_rect(topleft=(x_pos_key, current_y))
            clip_width_detail = max(0, panel_width - detail_rect.left - 10)
            if detail_rect.width > clip_width_detail:
                self.screen.blit(
                    detail_surf,
                    detail_rect,
                    area=pygame.Rect(0, 0, clip_width_detail, detail_rect.height),
                )
            else:
                self.screen.blit(detail_surf, detail_rect)
            stat_rects["Network Details"] = detail_rect.clip(self.screen.get_rect())
            last_y = detail_rect.bottom
        except Exception as e:
            print(f"Error rendering network details: {e}")
            last_y = current_y + line_height_detail

        # Render Live Resource Usage
        current_y = last_y + 4
        resource_usage = self._get_live_resource_usage()
        # Updated string to show percentage for GPU
        resource_str = f"Live Usage | CPU: {resource_usage['CPU']} | Mem: {resource_usage['Mem']} | GPU Mem: {resource_usage['GPU Mem']}"
        try:
            resource_surf = resource_font.render(resource_str, True, GRAY)
            resource_rect = resource_surf.get_rect(topleft=(x_pos_key, current_y))
            clip_width_resource = max(0, panel_width - resource_rect.left - 10)
            if resource_rect.width > clip_width_resource:
                self.screen.blit(
                    resource_surf,
                    resource_rect,
                    area=pygame.Rect(0, 0, clip_width_resource, resource_rect.height),
                )
            else:
                self.screen.blit(resource_surf, resource_rect)
            stat_rects["Resource Usage"] = resource_rect.clip(self.screen.get_rect())
            last_y = resource_rect.bottom
        except Exception as e:
            print(f"Error rendering resource usage: {e}")
            last_y = current_y + line_height_resource

        return last_y, stat_rects
