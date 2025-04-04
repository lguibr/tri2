# File: ui/panels/left_panel.py
import pygame
import os
import time
from typing import Dict, Any, Optional, Deque, Tuple

from config import (
    VisConfig,
    BufferConfig,
    StatsConfig,
    DQNConfig,
    DEVICE,
    TensorBoardConfig,
)
from config.general import TOTAL_TRAINING_STEPS
from ui.plotter import Plotter

# Import the new component renderers
from .left_panel_components import (
    ButtonStatusRenderer,
    NotificationRenderer,
    InfoTextRenderer,
    TBStatusRenderer,
    PlotAreaRenderer,
)

# Tooltips specific to this panel
TOOLTIP_TEXTS = {
    "Status": "Current state: Paused, Buffering, Training, Confirm Cleanup, Cleaning, or Error.",
    "Global Steps": "Total environment steps taken / Total planned steps.",
    "Total Episodes": "Total completed episodes across all environments.",
    "Steps/Sec (Current)": f"Current avg Steps/Sec (~{StatsConfig.STATS_AVG_WINDOW} intervals). See plot for history.",
    "Buffer Fill": f"Current replay buffer fill % ({BufferConfig.REPLAY_BUFFER_SIZE / 1e6:.1f}M cap). See plot.",
    "PER Beta": f"Current PER IS exponent ({BufferConfig.PER_BETA_START:.1f}->1.0). See plot.",
    "Learning Rate": "Current learning rate. See plot for history/schedule.",
    "Train Button": "Click to Start/Pause training (or press 'P').",
    "Cleanup Button": "Click to DELETE agent ckpt & buffer for CURRENT run ONLY, then re-init.",
    "Play Demo Button": "Click to enter interactive play mode. Gameplay fills the replay buffer.",
    "Device": f"Computation device detected ({DEVICE.type.upper()}).",
    "Network": f"CNN+MLP Fusion. Dueling={DQNConfig.USE_DUELING}, Noisy={DQNConfig.USE_NOISY_NETS}, C51={DQNConfig.USE_DISTRIBUTIONAL}",
    "TensorBoard Status": "Indicates TB logging status and log directory.",
    "Notification Area": "Displays the latest best achievements (RL Score, Game Score, Loss).",
    "Best RL Score Info": "Best RL Score achieved: Current Value (Previous Value) - Steps Ago",
    "Best Game Score Info": "Best Game Score achieved: Current Value (Previous Value) - Steps Ago",
    "Best Loss Info": "Best (Lowest) Loss achieved: Current Value (Previous Value) - Steps Ago",
}


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.stat_rects: Dict[str, pygame.Rect] = {}

        # Instantiate sub-renderers
        self.button_status_renderer = ButtonStatusRenderer(self.screen, self.fonts)
        self.notification_renderer = NotificationRenderer(self.screen, self.fonts)
        self.info_text_renderer = InfoTextRenderer(self.screen, self.fonts)
        self.tb_status_renderer = TBStatusRenderer(self.screen, self.fonts)
        self.plot_area_renderer = PlotAreaRenderer(
            self.screen, self.fonts, self.plotter
        )

    def _init_fonts(self):
        """Loads necessary fonts using system defaults."""
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
                fonts[key] = pygame.font.Font(None, size)  # Fallback
            if fonts[key] is None:
                print(f"ERROR: Font '{key}' failed to load.")
        return fonts

    def render(
        self,
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        app_state: str,
    ):
        """Renders the entire left panel by calling sub-render methods."""
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(300, self.vis_config.LEFT_PANEL_WIDTH))
        lp_rect = pygame.Rect(0, 0, lp_width, current_height)

        status_color_map = {
            "Paused": (30, 30, 30),
            "Buffering": (30, 40, 30),
            "Training": (40, 30, 30),
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

        # 1. Buttons and Status
        next_y, rects_bs, notification_area_rect = self.button_status_renderer.render(
            current_y, lp_width, app_state, is_training, status
        )
        self.stat_rects.update(rects_bs)
        current_y = next_y

        # 2. Notification Area (if applicable)
        if notification_area_rect:
            rects_notif = self.notification_renderer.render(
                notification_area_rect, stats_summary
            )
            self.stat_rects.update(rects_notif)
            # Adjust current_y only if notifications were below buttons (which they aren't)
            # current_y = max(current_y, notification_area_rect.bottom + 10) # Not needed here

        # 3. Info Text Block
        next_y, rects_info = self.info_text_renderer.render(
            current_y, stats_summary, buffer_capacity, lp_width
        )
        self.stat_rects.update(rects_info)
        current_y = next_y

        # 4. TensorBoard Status
        next_y, rects_tb = self.tb_status_renderer.render(
            current_y + 10, tensorboard_log_dir, lp_width
        )
        self.stat_rects.update(rects_tb)
        current_y = next_y

        # 5. Plot Area
        self.plot_area_renderer.render(
            current_y + 15, lp_width, current_height, plot_data, status
        )

    def get_stat_rects(self) -> Dict[str, pygame.Rect]:
        """Returns the dictionary of rects for tooltip handling."""
        return self.stat_rects.copy()

    def get_tooltip_texts(self) -> Dict[str, str]:
        """Returns the dictionary of tooltip texts."""
        return TOOLTIP_TEXTS
