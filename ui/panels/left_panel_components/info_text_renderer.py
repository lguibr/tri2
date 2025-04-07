# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple

from config import RNNConfig, TransformerConfig, ModelConfig, WHITE, LIGHTG, GRAY

# Removed DEVICE import from here


class InfoTextRenderer:
    """Renders essential non-plotted information text, including network config and live resources."""

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
        parts = ["CNN+MLP Fusion"]
        if self.transformer_config.USE_TRANSFORMER:
            parts.append("Transformer")
        if self.rnn_config.USE_RNN:
            parts.append("LSTM")
        return (
            f"Actor-Critic ({' -> '.join(parts)})"
            if len(parts) > 1
            else "Actor-Critic (CNN+MLP Fusion)"
        )

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
        # Import DEVICE here to ensure the updated value is used
        from config.general import DEVICE

        usage = {"CPU": "N/A", "Mem": "N/A", "GPU Mem": "N/A"}
        cpu_val = self.stats_summary_cache.get("current_cpu_usage")
        mem_val = self.stats_summary_cache.get("current_memory_usage")
        gpu_val = self.stats_summary_cache.get("current_gpu_memory_usage_percent")

        if cpu_val is not None:
            usage["CPU"] = f"{cpu_val:.1f}%"
        if mem_val is not None:
            # Added missing closing quote
            usage["Mem"] = f"{mem_val:.1f}%"

        # Check if DEVICE is None before accessing its type
        device_type = DEVICE.type if DEVICE else "cpu"  # Default to 'cpu' if None

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
        worker_counts: Dict[str, int],  # Added worker_counts
    ) -> int:
        """Renders the info text block. Returns next_y."""
        # Import DEVICE here as well for the device_type_str
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
        # Check if DEVICE is None before accessing its type
        device_type_str = (
            DEVICE.type.upper() if DEVICE and hasattr(DEVICE, "type") else "CPU"
        )
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
        # --- Get Worker Counts ---
        env_runners = worker_counts.get("env_runners", 0)
        trainers = worker_counts.get("trainers", 0)
        worker_str = f"Env: {env_runners} | Train: {trainers}"
        # --- End Worker Counts ---
        # --- Get Learning Rate ---
        lr_val = stats_summary.get("current_lr", 0.0)
        lr_str = f"{lr_val:.1e}" if lr_val > 0 else "N/A"
        # --- End Learning Rate ---

        info_lines = [
            ("Device", device_type_str),
            ("Network", network_desc),
            ("Params", param_str),
            ("LR", lr_str),  # Added LR
            ("Workers", worker_str),  # Added Workers
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
            # --- Change color here ---
            detail_surf = detail_font.render(network_details, True, WHITE)
            # --- End change color ---
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
            # --- Change color here ---
            resource_surf = resource_font.render(resource_str, True, WHITE)
            # --- End change color ---
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
