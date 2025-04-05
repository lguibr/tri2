# File: ui/panels/left_panel.py
import pygame
import os
import time
from typing import Dict, Any, Optional, Deque, Tuple

from config import (
    VisConfig,
    StatsConfig,
    PPOConfig,
    RNNConfig,  # Added
    DEVICE,
    TensorBoardConfig,
)
from config.general import TOTAL_TRAINING_STEPS
from ui.plotter import Plotter

from .left_panel_components import (
    ButtonStatusRenderer,
    NotificationRenderer,
    InfoTextRenderer,
    TBStatusRenderer,
    PlotAreaRenderer,
)

TOOLTIP_TEXTS = {
    "Status": "Current state: Ready, Training, Confirm Cleanup, Cleaning, or Error.",
    "Global Steps": "Total environment steps taken / Total planned steps.",
    "Total Episodes": "Total completed episodes across all environments.",
    "Steps/Sec (Current)": "Current avg Steps/Sec (Collection + Update). See plot for history.",
    "Learning Rate": "Current learning rate. See plot for history/schedule.",
    "Run Button": "Click to Start/Stop training run (or press 'P').",  # Renamed
    "Cleanup Button": "Click to DELETE agent ckpt for CURRENT run ONLY, then re-init.",
    "Play Demo Button": "Click to enter interactive play mode.",
    "Device": f"Computation device detected ({DEVICE.type.upper()}).",
    "Network": f"Actor-Critic (CNN+MLP Fusion -> Optional LSTM:{RNNConfig.USE_RNN})",  # Updated
    "TensorBoard Status": "Indicates TB logging status and log directory.",
    "Notification Area": "Displays the latest best achievements (RL Score, Game Score, Value Loss).",
    "Best RL Score Info": "Best RL Score achieved: Current Value (Previous Value) - Steps Ago",
    "Best Game Score Info": "Best Game Score achieved: Current Value (Previous Value) - Steps Ago",
    "Best Loss Info": "Best (Lowest) Value Loss achieved: Current Value (Previous Value) - Steps Ago",
    "Policy Loss": "Average loss for the policy network during the last update.",
    "Value Loss": "Average loss for the value network during the last update.",
    "Entropy": "Average policy entropy during the last update (encourages exploration).",
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
        self.notification_renderer = NotificationRenderer(self.screen, self.fonts)
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
            "notification": 19,
            "notification_label": 16,
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
        is_running: bool,  # Renamed from is_training
        status: str,
        stats_summary: Dict[str, Any],
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        app_state: str,
    ):
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(300, self.vis_config.LEFT_PANEL_WIDTH))
        lp_rect = pygame.Rect(0, 0, lp_width, current_height)

        status_color_map = {
            "Ready": (30, 30, 30),
            "Training": (30, 40, 30),
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
        notification_area_rect = None

        next_y, rects_bs, notification_area_rect = self.button_status_renderer.render(
            current_y, lp_width, app_state, is_running, status
        )
        self.stat_rects.update(rects_bs)
        current_y = next_y

        if notification_area_rect:
            rects_notif = self.notification_renderer.render(
                notification_area_rect, stats_summary
            )
            self.stat_rects.update(rects_notif)

        next_y, rects_info = self.info_text_renderer.render(
            current_y, stats_summary, lp_width
        )
        self.stat_rects.update(rects_info)
        current_y = next_y

        next_y, rects_tb = self.tb_status_renderer.render(
            current_y + 10, tensorboard_log_dir, lp_width
        )
        self.stat_rects.update(rects_tb)
        current_y = next_y

        self.plot_area_renderer.render(
            current_y + 15, lp_width, current_height, plot_data, status
        )

    def get_stat_rects(self) -> Dict[str, pygame.Rect]:
        return self.stat_rects.copy()

    def get_tooltip_texts(self) -> Dict[str, str]:
        return TOOLTIP_TEXTS
