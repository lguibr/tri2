# File: ui/panels/left_panel.py
import pygame
import os
import time
from typing import Dict, Any, Optional, Deque, Tuple

# --- MODIFIED: Import DEVICE directly ---
from config import (
    VisConfig,
    StatsConfig,
    PPOConfig,
    RNNConfig,
    TensorBoardConfig,
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    LIGHTG,
    BLUE,  # Import colors if needed
)
from config.general import TOTAL_TRAINING_STEPS, DEVICE  # Import DEVICE here

# --- END MODIFIED ---

from ui.plotter import Plotter
from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    TBStatusRenderer,
    PlotAreaRenderer,
)

# --- MODIFIED: Remove direct DEVICE access from TOOLTIP_TEXTS definition ---
TOOLTIP_TEXTS_BASE = {
    "Status": "Current application state: Ready, Collecting Experience, Updating Agent, Confirm Cleanup, Cleaning, or Error.",
    "Run Button": "Click to Start/Stop training run (or press 'P').",
    "Cleanup Button": "Click to DELETE agent ckpt for CURRENT run ONLY, then re-init.",
    "Play Demo Button": "Click to enter interactive play mode.",
    "Device": "Computation device detected.",  # Placeholder text
    "Network": f"Actor-Critic (CNN+MLP Fusion -> Optional LSTM:{RNNConfig.USE_RNN})",
    "TensorBoard Status": "Indicates TB logging status and log directory.",
    "Steps Info": "Global Steps / Total Planned Steps",
    "Episodes Info": "Total Completed Episodes",
    "SPS Info": "Steps Per Second (Collection + Update Avg)",
    "Update Progress": "Progress of the current agent neural network update cycle.",
}
# --- END MODIFIED ---


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
        """Initializes fonts used in the left panel."""
        fonts = {}
        # Define font configurations
        font_configs = {
            "ui": 24,
            "status": 28,
            "logdir": 16,
            "plot_placeholder": 20,
            "notification_label": 16,
            "plot_title_values": 8,
            "progress_bar": 14,
            "notification": 18,  # Added font used in NotificationRenderer
        }
        # Load fonts, falling back to default if SysFont fails
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
        update_progress: float,
    ):
        """Renders the entire left panel."""
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(300, self.vis_config.LEFT_PANEL_WIDTH))
        lp_rect = pygame.Rect(0, 0, lp_width, current_height)

        # Determine background color based on status
        status_color_map = {
            "Ready": (30, 30, 30),
            "Collecting Experience": (30, 40, 30),
            "Updating Agent": (30, 30, 50),
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Initializing": (40, 40, 40),
        }
        bg_color = status_color_map.get(status, (30, 30, 30))
        pygame.draw.rect(self.screen, bg_color, lp_rect)
        self.stat_rects.clear()  # Clear rects for the new frame

        current_y = 10

        # Render Buttons and Compact Status Block
        next_y, rects_bs = self.button_status_renderer.render(
            y_start=current_y,
            panel_width=lp_width,
            app_state=app_state,
            is_training_running=is_training_running,
            status=status,
            stats_summary=stats_summary,
            update_progress=update_progress,
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
        """Returns the dictionary of rectangles for tooltip detection."""
        return self.stat_rects.copy()

    # --- MODIFIED: Dynamically format Device tooltip ---
    def get_tooltip_texts(self) -> Dict[str, str]:
        """Returns the dictionary of tooltip texts, formatting dynamic ones."""
        texts = TOOLTIP_TEXTS_BASE.copy()
        # Ensure DEVICE is not None before accessing .type
        device_type_str = "UNKNOWN"
        if DEVICE and hasattr(DEVICE, "type"):
            device_type_str = DEVICE.type.upper()
        texts["Device"] = f"Computation device detected ({device_type_str})."
        return texts

    # --- END MODIFIED ---
