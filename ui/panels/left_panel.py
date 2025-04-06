import pygame
from typing import Dict, Any, Optional, Deque

from config import (
    VisConfig,
    RNNConfig,
    TransformerConfig,
)
from config.general import DEVICE

from ui.plotter import Plotter
from ui.input_handler import InputHandler  # Import InputHandler for type hint
from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    TBStatusRenderer,
    PlotAreaRenderer,
    # Removed NotificationRenderer import as best values are now in PlotAreaRenderer
)

TOOLTIP_TEXTS_BASE = {
    "Status": "Current application state: Ready, Collecting Experience, Updating Agent, Confirm Cleanup, Cleaning, Error, Playing Demo, Debugging Grid.",
    "Run Button": "Click to Start/Stop the training process. Disabled during active phases.",
    "Cleanup Button": "Click to DELETE agent ckpt for CURRENT run ONLY, then re-init. Disabled during active phases.",
    "Play Demo Button": "Click to enter interactive play mode. Disabled during active phases.",
    "Debug Mode Button": "Click to enter grid debug mode (toggle cells, check lines). Disabled during active phases.",
    "Device": "Computation device detected.",
    "Network": "Neural network architecture.",
    "TensorBoard Status": "Indicates TB logging status and log directory.",
    "Steps Info": "Global Steps / Total Planned Steps (Regular Training)",
    "Episodes Info": "Total Completed Episodes (Regular Training)",
    "SPS Info": "Steps Per Second (Collection + Update Avg, Regular Training)",
    "Update Epoch Info": "Current PPO Epoch / Total PPO Epochs for this update cycle.",
    "Update Epoch Progress": "Progress through minibatches within the current PPO epoch.",
    "Update Overall Progress": "Overall progress through all minibatches across all PPO epochs for this update cycle.",
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

    def _init_fonts(self):
        """Initializes fonts used in the left panel."""
        fonts = {}
        font_configs = {
            "ui": 24,
            "status": 28,
            "logdir": 16,
            "plot_placeholder": 20,
            "notification_label": 16,  # Keep for plot area best values
            "plot_title_values": 8,
            "progress_bar": 14,
            "notification": 18,  # Keep for plot area best values
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
        panel_width: int,  # Accept panel_width
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        app_state: str,
        update_progress_details: Dict[str, Any],
    ):
        """Renders the entire left panel within the given width."""
        current_height = self.screen.get_height()
        # Use the provided panel_width
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
        }
        bg_color = status_color_map.get(status, (30, 30, 30))
        pygame.draw.rect(self.screen, bg_color, lp_rect)
        self.stat_rects.clear()

        current_y = 10

        # Render Buttons, Status, and Update Progress
        next_y, rects_bs = self.button_status_renderer.render(
            y_start=current_y,
            panel_width=lp_width,  # Pass width
            app_state=app_state,
            is_process_running=is_process_running,
            status=status,
            stats_summary=stats_summary,
            update_progress_details=update_progress_details,
        )

        # Get button rects from InputHandler for tooltips
        if self.input_handler:
            self.stat_rects["Run Button"] = self.input_handler.run_btn_rect
            self.stat_rects["Cleanup Button"] = self.input_handler.cleanup_btn_rect
            self.stat_rects["Play Demo Button"] = self.input_handler.demo_btn_rect
            self.stat_rects["Debug Mode Button"] = self.input_handler.debug_btn_rect
        else:  # Fallback if input_handler not set yet (shouldn't happen after init)
            button_height = 40
            button_y_pos = (
                current_y  # Use current_y from before ButtonStatusRenderer.render
            )
            run_button_width = 100
            cleanup_button_width = 160
            demo_button_width = 120
            debug_button_width = 120
            button_spacing = 10
            run_button_rect = pygame.Rect(
                button_spacing, button_y_pos, run_button_width, button_height
            )
            cleanup_button_rect = pygame.Rect(
                run_button_rect.right + button_spacing,
                button_y_pos,
                cleanup_button_width,
                button_height,
            )
            demo_button_rect = pygame.Rect(
                cleanup_button_rect.right + button_spacing,
                button_y_pos,
                demo_button_width,
                button_height,
            )
            debug_button_rect = pygame.Rect(
                demo_button_rect.right + button_spacing,
                button_y_pos,
                debug_button_width,
                button_height,
            )
            self.stat_rects["Run Button"] = run_button_rect
            self.stat_rects["Cleanup Button"] = cleanup_button_rect
            self.stat_rects["Play Demo Button"] = demo_button_rect
            self.stat_rects["Debug Mode Button"] = debug_button_rect

        self.stat_rects.update(
            rects_bs
        )  # Add status rects etc. from ButtonStatusRenderer
        current_y = next_y

        # Render Info Text Block
        next_y, rects_info = self.info_text_renderer.render(
            current_y + 5, stats_summary, lp_width  # Pass width
        )
        self.stat_rects.update(rects_info)
        current_y = next_y

        # Render TensorBoard Status
        next_y, rects_tb = self.tb_status_renderer.render(
            current_y + 10, tensorboard_log_dir, lp_width  # Pass width
        )
        self.stat_rects.update(rects_tb)
        current_y = next_y

        # Render Plot Area (pass stats_summary for best values)
        self.plot_area_renderer.render(
            y_start=current_y + 5,  # Use current_y directly
            panel_width=lp_width,
            screen_height=current_height,
            plot_data=plot_data,
            # stats_summary=stats_summary, # Removed stats_summary
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
        parts = ["CNN+MLP Fusion"]
        if self.transformer_config.USE_TRANSFORMER:
            parts.append("Transformer")
        if self.rnn_config.USE_RNN:
            parts.append("LSTM")
        if len(parts) == 1:
            texts["Network"] = "Actor-Critic network using CNN and MLP feature fusion."
        else:
            texts["Network"] = f"Actor-Critic network using {' -> '.join(parts)}."
        if "Update Epoch Progress" in texts:
            texts.pop("Update Progress", None)
        return texts
