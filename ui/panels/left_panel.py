import pygame
from typing import Dict, Any, Optional, Deque

from config import (
    VisConfig,
    RNNConfig,
    TransformerConfig,
    ModelConfig,
    TOTAL_TRAINING_STEPS,
)
from config.general import DEVICE

from ui.plotter import Plotter
from ui.input_handler import InputHandler
from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    TBStatusRenderer,
    PlotAreaRenderer,
)

TOOLTIP_TEXTS_BASE = {
    "Status": "Current application state: Ready, Collecting Experience, Updating Agent, Confirm Cleanup, Cleaning, Error, Playing Demo, Debugging Grid.",
    "Run Button": "Click to Start/Stop the training process. Enabled only in Main Menu.",
    "Cleanup Button": "Click to DELETE agent ckpt for CURRENT run ONLY, then re-init. Enabled only when stopped in Main Menu.",
    "Play Demo Button": "Click to enter interactive play mode. Enabled only when stopped in Main Menu.",
    "Debug Mode Button": "Click to enter grid debug mode (toggle cells, check lines). Enabled only when stopped in Main Menu.",
    "Training Progress": f"Overall training progress towards the current target step count. Shows current steps, percentage, and estimated time remaining (ETA). Visible only during training.",
    "Device": "Computation device detected.",
    "Network": "High-level description of the network architecture.",
    "Params": "Total number of trainable parameters in the network model.",
    "Network Details": "Detailed configuration of network layers (CNN, MLP, Transformer, LSTM).",
    "Run Started": "Timestamp when the current run (or resumed run) was started.",
    # Updated tooltip for resource usage
    "Resource Usage": "Live system resource usage: CPU %, System Memory %, GPU Memory (% allocated by PyTorch).",
    "TensorBoard Status": "Indicates TB logging status and log directory.",
    "Steps Info": "Global Steps / Target Steps for this training session.",
    "Episodes Info": "Total Completed Episodes.",
    "SPS Info": "Steps Per Second (Collection + Update Avg).",
    "Update Epoch Info": "Current PPO Epoch / Total PPO Epochs for this update cycle.",
    "Update Epoch Progress": "Progress through minibatches within the current PPO epoch. Shows percentage and estimated time remaining (ETA) for this epoch.",
    "Update Overall Progress": "Overall progress through all minibatches across all PPO epochs for this update cycle. Shows percentage and estimated time remaining (ETA) for the entire update phase.",
}


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.stat_rects: Dict[str, pygame.Rect] = {}
        self.input_handler: Optional[InputHandler] = None

        self.button_status_renderer = ButtonStatusRenderer(self.screen, self.fonts)
        self.info_text_renderer = InfoTextRenderer(self.screen, self.fonts)
        self.tb_status_renderer = TBStatusRenderer(self.screen, self.fonts)
        self.plot_area_renderer = PlotAreaRenderer(
            self.screen, self.fonts, self.plotter
        )
        self.rnn_config = RNNConfig()
        self.transformer_config = TransformerConfig()
        self.model_config_net = ModelConfig.Network()

    def _init_fonts(self):
        """Initializes fonts used in the left panel."""
        fonts = {}
        font_configs = {
            "ui": 24,
            "status": 28,
            "logdir": 16,
            "plot_placeholder": 20,
            "notification_label": 16,
            "plot_title_values": 8,
            "progress_bar": 14,
            "notification": 18,
        }
        for key, size in font_configs.items():
            try:
                fonts[key] = pygame.font.SysFont(None, size)
            except Exception:
                fonts[key] = pygame.font.Font(None, size)
            if fonts[key] is None:
                print(f"ERROR: Font '{key}' failed to load.")
        return fonts

    def render(
        self,
        panel_width: int,
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        app_state: str,
        update_progress_details: Dict[str, Any],
        agent_param_count: int,
    ):
        """Renders the entire left panel within the given width."""
        current_height = self.screen.get_height()
        lp_width = panel_width
        lp_rect = pygame.Rect(0, 0, lp_width, current_height)

        status_color_map = {
            "Ready": (30, 30, 30),
            "Collecting Experience": (30, 40, 30),
            "Updating Agent": (30, 30, 50),
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Debugging Grid": (40, 30, 40),
            "Initializing": (40, 40, 40),
            "Training Complete": (30, 50, 30),
        }
        base_status = status.split(" (")[0] if "(" in status else status
        bg_color = status_color_map.get(base_status, (30, 30, 30))

        pygame.draw.rect(self.screen, bg_color, lp_rect)
        self.stat_rects.clear()

        current_y = 10

        # Render Buttons, Status, and Update Progress
        next_y, rects_bs = self.button_status_renderer.render(
            y_start=current_y,
            panel_width=lp_width,
            app_state=app_state,
            is_process_running=is_process_running,
            status=status,
            stats_summary=stats_summary,
            update_progress_details=update_progress_details,
        )
        self.stat_rects.update(rects_bs)
        current_y = next_y

        # Render Info Text Block
        next_y, rects_info = self.info_text_renderer.render(
            current_y + 5, stats_summary, lp_width, agent_param_count
        )
        self.stat_rects.update(rects_info)
        current_y = next_y

        # Render TensorBoard Status
        next_y, rects_tb = self.tb_status_renderer.render(
            current_y + 10, tensorboard_log_dir, lp_width
        )
        self.stat_rects.update(rects_tb)
        current_y = next_y

        # Render Plot Area
        self.plot_area_renderer.render(
            y_start=current_y + 5,
            panel_width=lp_width,
            screen_height=current_height,
            plot_data=plot_data,
            status=status,
        )

    def get_stat_rects(self) -> Dict[str, pygame.Rect]:
        """Returns the dictionary of rectangles for tooltip detection."""
        return self.stat_rects.copy()

    def get_tooltip_texts(self) -> Dict[str, str]:
        """Returns the dictionary of tooltip texts, formatting dynamic ones."""
        texts = TOOLTIP_TEXTS_BASE.copy()
        device_type_str = "UNKNOWN"
        if DEVICE and hasattr(DEVICE, "type"):
            device_type_str = DEVICE.type.upper()
        texts["Device"] = f"Computation device detected ({device_type_str})."
        texts["Network"] = self.info_text_renderer._get_network_description()
        texts["Network Details"] = self.info_text_renderer._get_network_details()

        # Update steps info tooltip based on target step from cached summary
        target_step = self.info_text_renderer.stats_summary_cache.get(
            "training_target_step", 0
        )
        if target_step > 0:
            texts["Steps Info"] = (
                f"Global Steps / Target Steps ({target_step/1e6:.1f}M) for this training session."
            )
        else:
            texts["Steps Info"] = "Global Steps accumulated so far."

        return texts
