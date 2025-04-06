# File: ui/panels/left_panel.py
import pygame
import os
import time
from typing import Dict, Any, Optional, Deque, Tuple

from config import (
    VisConfig,
    StatsConfig,
    PPOConfig,
    RNNConfig,
    DEVICE,
    TensorBoardConfig,
)
from config.general import TOTAL_TRAINING_STEPS
from ui.plotter import Plotter

from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    TBStatusRenderer,
    PlotAreaRenderer,
)

TOOLTIP_TEXTS = {
    "Status": "Current application state: Ready, Collecting Experience, Updating Agent, Confirm Cleanup, Cleaning, or Error.",
    "Run Button": "Click to Start/Stop training run (or press 'P').",
    "Cleanup Button": "Click to DELETE agent ckpt for CURRENT run ONLY, then re-init.",
    "Play Demo Button": "Click to enter interactive play mode.",
    "Device": f"Computation device detected ({DEVICE.type.upper()}).",
    "Network": f"Actor-Critic (CNN+MLP Fusion -> Optional LSTM:{RNNConfig.USE_RNN})",
    "TensorBoard Status": "Indicates TB logging status and log directory.",
    # Add tooltips for compact status line elements if desired
    "Steps Info": "Global Steps / Total Planned Steps",
    "Episodes Info": "Total Completed Episodes",
    "SPS Info": "Steps Per Second (Collection + Update Avg)",
    "Update Progress": "Progress of the current agent neural network update cycle.",
}


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.stat_rects: Dict[str, pygame.Rect] = {}

        self.button_status_renderer = ButtonStatusRenderer(self.screen, self.fonts)
        self.info_text_renderer = InfoTextRenderer(self.screen, self.fonts)
        self.tb_status_renderer = TBStatusRenderer(self.screen, self.fonts)
        self.plot_area_renderer = PlotAreaRenderer(
            self.screen, self.fonts, self.plotter
        )

    def _init_fonts(self):
        fonts = {}
        font_configs = {
            "ui": 24,
            "status": 28,
            "logdir": 16,
            "plot_placeholder": 20,
            "notification_label": 16,
            "plot_title_values": 8,
            "progress_bar": 14,  # Font for progress bar percentage
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
        is_training_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        app_state: str,
        update_progress: float,  # Added update_progress
    ):
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(300, self.vis_config.LEFT_PANEL_WIDTH))
        lp_rect = pygame.Rect(0, 0, lp_width, current_height)

        status_color_map = {
            "Ready": (30, 30, 30),
            "Collecting Experience": (30, 40, 30),  # Greenish tint
            "Updating Agent": (30, 30, 50),  # Bluish tint
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Initializing": (40, 40, 40),
        }
        bg_color = status_color_map.get(status, (30, 30, 30))
        pygame.draw.rect(self.screen, bg_color, lp_rect)
        self.stat_rects.clear()

        current_y = 10

        # Render Buttons and Compact Status Block (Pass update_progress)
        next_y, rects_bs = self.button_status_renderer.render(
            y_start=current_y,
            panel_width=lp_width,
            app_state=app_state,
            is_training_running=is_training_running,
            status=status,
            stats_summary=stats_summary,
            update_progress=update_progress,  # Pass progress
        )
        self.stat_rects.update(rects_bs)
        current_y = next_y

        # Render Info Text Block
        next_y, rects_info = self.info_text_renderer.render(
            current_y + 5, stats_summary, lp_width
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
            current_y + 15, lp_width, current_height, plot_data, status
        )

    def get_stat_rects(self) -> Dict[str, pygame.Rect]:
        return self.stat_rects.copy()

    def get_tooltip_texts(self) -> Dict[str, str]:
        return TOOLTIP_TEXTS
