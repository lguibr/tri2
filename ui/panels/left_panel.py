# File: ui/panels/left_panel.py
import pygame
from typing import Dict, Any, Optional, Deque

from config import (
    VisConfig,
    RNNConfig,
    TransformerConfig,
    ModelConfig,
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
            "tb_status": 16,
        }
        for key, size in font_configs.items():
            try:
                fonts[key] = pygame.font.SysFont(None, size)
            except Exception:
                try:
                    fonts[key] = pygame.font.Font(None, size)
                except Exception as e:
                    print(
                        f"ERROR: Font '{key}' failed to load with SysFont and Font(None): {e}"
                    )
                    fonts[key] = None  # Set to None if both fail
            if fonts[key] is None:
                print(f"ERROR: Font '{key}' could not be loaded.")
        return fonts

    def render(
        self,
        panel_width: int,
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        tensorboard_log_dir: Optional[str],  # This is the string path
        plot_data: Dict[str, Deque],
        app_state: str,
        update_progress_details: Dict[str, Any],
        agent_param_count: int,
        worker_counts: Dict[str, int],
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
            "Running AlphaZero": (30, 50, 30),  # Combined running state
        }
        base_status = status.split(" (")[0] if "(" in status else status
        bg_color = status_color_map.get(base_status, (30, 30, 30))

        pygame.draw.rect(self.screen, bg_color, lp_rect)
        current_y = 10  # Start with a definite integer

        # Render Buttons and Status
        try:
            next_y = self.button_status_renderer.render(  # Should return int
                y_start=current_y,
                panel_width=panel_width,
                app_state=app_state,
                is_process_running=is_process_running,
                status=status,
                stats_summary=stats_summary,
                update_progress_details=update_progress_details,
            )
            current_y = next_y  # Assign the returned int
        except Exception as e:
            print(f"Error in button_status_renderer: {e}")
            current_y += 50  # Fallback increment

        # Render Info Text
        try:
            next_y = self.info_text_renderer.render(  # Should return int
                current_y + 5,  # int + int = int
                stats_summary,
                panel_width,
                agent_param_count,
                worker_counts,
            )
            current_y = next_y  # Assign the returned int
        except Exception as e:
            print(f"Error in info_text_renderer: {e}")
            current_y += 50  # Fallback increment

        # Render TB Status - Pass the string path and panel_width
        try:
            # Pass panel_width here
            next_y_val, _ = self.tb_status_renderer.render(
                current_y + 10, tensorboard_log_dir, panel_width
            )
            current_y = next_y_val  # Assign the returned int
        except Exception as e:
            print(f"Error in tb_status_renderer: {e}")
            current_y += 20  # Fallback increment

        # Render Plots
        if app_state == AppState.MAIN_MENU.value:
            # Ensure current_y is an int before adding 5
            if isinstance(current_y, (int, float)):
                plot_y_start = int(current_y) + 5  # This line should now work
                try:
                    self.plot_area_renderer.render(
                        y_start=plot_y_start,
                        panel_width=panel_width,
                        screen_height=current_height,
                        plot_data=plot_data,
                        status=status,
                    )
                except Exception as e:
                    print(f"Error in plot_area_renderer: {e}")
            else:
                print(
                    f"Error: current_y is not a number before plotting. Type: {type(current_y)}, Value: {current_y}"
                )
                # Optionally render an error message or skip plotting
