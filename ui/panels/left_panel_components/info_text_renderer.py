# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple

from config import RNNConfig, TransformerConfig, ModelConfig, WHITE, LIGHTG, GRAY


class InfoTextRenderer:
    """Renders essential non-plotted information text."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.rnn_config = RNNConfig()
        self.transformer_config = TransformerConfig()
        self.model_config_net = ModelConfig.Network()
        self.resource_font = fonts.get("logdir", pygame.font.Font(None, 16))
        self.stats_summary_cache: Dict[str, Any] = {}

    def _get_network_description(self) -> str:
        """Builds a description string based on network components."""
        return "AlphaZero Neural Network"

    def _get_network_details(self) -> str:
        """Builds a detailed string of network configuration."""
        details = []
        cnn_str = str(self.model_config_net.CONV_CHANNELS).replace(" ", "")
        mlp_str = str(self.model_config_net.COMBINED_FC_DIMS).replace(" ", "")
        shape_mlp_cfg_str = str(self.model_config_net.SHAPE_FEATURE_MLP_DIMS).replace(
            " ", ""
        )
        details.append(f"CNN={cnn_str}, ShapeMLP={shape_mlp_cfg_str}, Fusion={mlp_str}")
        # Add more details if needed, e.g., activation functions, batchnorm use
        return ", ".join(details)

    def _get_live_resource_usage(self) -> Dict[str, str]:
        """Fetches live CPU, Memory, and GPU Memory usage from cached summary."""
        from config.general import DEVICE

        usage = {"CPU": "N/A", "Mem": "N/A", "GPU Mem": "N/A"}
        cpu_val = self.stats_summary_cache.get("current_cpu_usage")
        mem_val = self.stats_summary_cache.get("current_memory_usage")
        gpu_val = self.stats_summary_cache.get("current_gpu_memory_usage_percent")

        if cpu_val is not None:
            usage["CPU"] = f"{cpu_val:.1f}%"
        if mem_val is not None:
            usage["Mem"] = f"{mem_val:.1f}%"

        device_type = DEVICE.type if DEVICE else "cpu"
        if gpu_val is not None:
            usage["GPU Mem"] = (
                f"{gpu_val:.1f}%" if device_type == "cuda" else "N/A (CPU)"
            )
        elif device_type != "cuda":
            usage["GPU Mem"] = "N/A (CPU)"
        return usage

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],
        panel_width: int,
        agent_param_count: int,
        worker_counts: Dict[str, int],  # Keep for potential future worker info
    ) -> int:
        """Renders the info text block. Returns next_y."""
        from config.general import DEVICE

        self.stats_summary_cache = stats_summary
        ui_font, detail_font, resource_font = (
            self.fonts.get("ui"),
            self.fonts.get("logdir"),
            self.resource_font,
        )
        if not ui_font or not detail_font or not resource_font:
            return y_start

        line_height_ui, line_height_detail, line_height_resource = (
            ui_font.get_linesize(),
            detail_font.get_linesize(),
            resource_font.get_linesize(),
        )
        device_type_str = DEVICE.type.upper() if DEVICE else "CPU"
        network_desc, network_details = (
            self._get_network_description(),
            self._get_network_details(),
        )
        param_str = (
            f"{agent_param_count / 1e6:.2f} M" if agent_param_count > 0 else "N/A"
        )
        start_time_unix = stats_summary.get("start_time", 0.0)
        start_time_str = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_unix))
            if start_time_unix > 0
            else "N/A"
        )
        # Worker info placeholder (adapt later based on actual workers)
        worker_str = "Self-Play: ?, Training: ?"  # Placeholder

        info_lines = [
            ("Device", device_type_str),
            ("Network", network_desc),
            ("Params", param_str),
            # ("Workers", worker_str), # Can add back when workers are implemented
            ("Run Started", start_time_str),
        ]
        last_y, x_pos_key, x_pos_val_offset, current_y = y_start, 10, 5, y_start + 5

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
                blit_area = (
                    pygame.Rect(0, 0, clip_width, value_rect.height)
                    if value_rect.width > clip_width
                    else None
                )
                self.screen.blit(value_surf, value_rect, area=blit_area)
                last_y = key_rect.union(value_rect).bottom
            except Exception as e:
                print(f"Error rendering stat line '{key}': {e}")
                last_y = line_y + line_height_ui

        current_y = last_y + 2
        try:
            detail_surf = detail_font.render(network_details, True, WHITE)
            detail_rect = detail_surf.get_rect(topleft=(x_pos_key, current_y))
            clip_width_detail = max(0, panel_width - detail_rect.left - 10)
            blit_area_detail = (
                pygame.Rect(0, 0, clip_width_detail, detail_rect.height)
                if detail_rect.width > clip_width_detail
                else None
            )
            self.screen.blit(detail_surf, detail_rect, area=blit_area_detail)
            last_y = detail_rect.bottom
        except Exception as e:
            print(f"Error rendering network details: {e}")
            last_y = current_y + line_height_detail

        current_y = last_y + 4
        resource_usage = self._get_live_resource_usage()
        resource_str = f"Live Usage | CPU: {resource_usage['CPU']} | Mem: {resource_usage['Mem']} | GPU Mem: {resource_usage['GPU Mem']}"
        try:
            resource_surf = resource_font.render(resource_str, True, WHITE)
            resource_rect = resource_surf.get_rect(topleft=(x_pos_key, current_y))
            clip_width_resource = max(0, panel_width - resource_rect.left - 10)
            blit_area_resource = (
                pygame.Rect(0, 0, clip_width_resource, resource_rect.height)
                if resource_rect.width > clip_width_resource
                else None
            )
            self.screen.blit(resource_surf, resource_rect, area=blit_area_resource)
            last_y = resource_rect.bottom
        except Exception as e:
            print(f"Error rendering resource usage: {e}")
            last_y = current_y + line_height_resource

        return last_y
