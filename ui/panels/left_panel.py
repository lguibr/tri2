# File: ui/panels/left_panel.py
import pygame
from typing import Dict, Any, Optional, Deque

from config import (
    VisConfig,
    RNNConfig,
    TransformerConfig,
    ModelConfig,
    # Removed TOTAL_TRAINING_STEPS
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
from app_state import AppState


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
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
        is_process_running: bool,  # Keep for potential future use (MCTS/NN running)
        status: str,
        stats_summary: Dict[str, Any],
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        app_state: str,
        update_progress_details: Dict[str, Any],  # Keep for potential NN progress
        agent_param_count: int,
        worker_counts: Dict[str, int],  # Keep structure, content will change
    ):
        """Renders the entire left panel within the given width."""
        current_height = self.screen.get_height()
        lp_rect = pygame.Rect(0, 0, panel_width, current_height)

        # Simplified status mapping
        status_color_map = {
            "Ready": (30, 30, 30),
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Debugging Grid": (40, 30, 40),
            "Initializing": (40, 40, 40),
            # Add potential future states
            "Running MCTS": (30, 40, 30),
            "Training NN": (30, 30, 50),
        }
        base_status = status.split(" (")[0] if "(" in status else status
        bg_color = status_color_map.get(base_status, (30, 30, 30))

        pygame.draw.rect(self.screen, bg_color, lp_rect)
        current_y = 10

        # Render Buttons and Status (simplified)
        next_y = self.button_status_renderer.render(
            y_start=current_y,
            panel_width=panel_width,
            app_state=app_state,
            is_process_running=is_process_running,  # Pass for potential future use
            status=status,
            stats_summary=stats_summary,
            update_progress_details=update_progress_details,  # Pass for potential NN progress
        )
        current_y = next_y

        # Render Info Text (simplified)
        next_y = self.info_text_renderer.render(
            current_y + 5,
            stats_summary,
            panel_width,
            agent_param_count,
            worker_counts,  # Pass potentially adapted worker counts
        )
        current_y = next_y

        # Render TB Status (unchanged)
        # next_y = self.tb_status_renderer.render(
        #     current_y + 10, tensorboard_log_dir, panel_width
        # )
        # current_y = next_y

        # Render Plots (unchanged condition, but plots themselves are simplified)
        if app_state == AppState.MAIN_MENU.value:
            self.plot_area_renderer.render(
                y_start=current_y + 5,
                panel_width=panel_width,
                screen_height=current_height,
                plot_data=plot_data,
                status=status,  # Pass status for placeholder text
            )
