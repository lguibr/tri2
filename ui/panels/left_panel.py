import pygame
from typing import Optional, Tuple, Dict, Any, List
import logging

from config import VisConfig
from ui.plotter import Plotter
from ui.input_handler import InputHandler  # Keep for type hint if needed
from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    PlotAreaRenderer,
    NotificationRenderer,
)
from app_state import AppState

logger = logging.getLogger(__name__)


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components based on provided data."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.input_handler: Optional[InputHandler] = None  # Reference set by UIRenderer

        # Initialize components
        self.button_status_renderer = ButtonStatusRenderer(self.screen, self.fonts)
        self.info_text_renderer = InfoTextRenderer(self.screen, self.fonts)
        self.notification_renderer = NotificationRenderer(self.screen, self.fonts)
        self.plot_area_renderer = PlotAreaRenderer(
            self.screen, self.fonts, self.plotter
        )

    def _init_fonts(self):
        """Initializes fonts used in the left panel."""
        # ... (font init remains the same)
        fonts = {}
        font_configs = {
            "ui": 24,
            "status": 28,
            "detail": 16,
            "resource": 16,
            "notification_label": 16,
            "notification": 18,
            "plot_placeholder": 20,
            "plot_title_values": 8,
            "mcts_stats_label": 18,
            "mcts_stats_value": 18,
        }
        for key, size in font_configs.items():
            try:
                fonts[key] = pygame.font.SysFont(None, size)
            except Exception:
                try:
                    fonts[key] = pygame.font.Font(None, size)
                except Exception as e:
                    logger.error(f"ERROR: Font '{key}' failed: {e}")
                    fonts[key] = None
        # Fallbacks
        if fonts.get("ui") is None:
            fonts["ui"] = pygame.font.Font(None, 24)
        if fonts.get("status") is None:
            fonts["status"] = pygame.font.Font(None, 28)
        if fonts.get("mcts_stats_label") is None:
            fonts["mcts_stats_label"] = pygame.font.Font(None, 18)
        if fonts.get("mcts_stats_value") is None:
            fonts["mcts_stats_value"] = pygame.font.Font(None, 18)
        return fonts

    def _get_background_color(self, status: str) -> Tuple[int, int, int]:
        """Determines background color based on status."""
        # ... (background color logic remains the same)
        status_color_map = {
            "Ready": (30, 30, 30),
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Debugging Grid": (40, 30, 40),
            "Initializing": (40, 40, 40),
            "Running AlphaZero": (30, 50, 30),
        }
        base_status = status.split(" (")[0] if "(" in status else status
        return status_color_map.get(base_status, (30, 30, 30))

    def render(self, panel_width: int, **render_data: Dict[str, Any]):
        """Renders the entire left panel based on the provided render_data dictionary."""
        current_height = self.screen.get_height()
        lp_rect = pygame.Rect(0, 0, panel_width, current_height)
        status = render_data.get("status", "")
        bg_color = self._get_background_color(status)
        pygame.draw.rect(self.screen, bg_color, lp_rect)

        current_y = 10
        # Define render order and estimated heights, pass data down
        render_order: List[Tuple[callable, int, Dict[str, Any]]] = [
            (
                self.button_status_renderer.render,
                60,
                {
                    k: render_data.get(k)
                    for k in [
                        "app_state",
                        "is_process_running",
                        "status",
                        "stats_summary",
                        "update_progress_details",
                    ]
                },
            ),
            (
                self.info_text_renderer.render,
                120,
                {
                    k: render_data.get(k)
                    for k in ["stats_summary", "agent_param_count", "worker_counts"]
                },
            ),
            (
                self.notification_renderer.render,
                70,
                {k: render_data.get(k) for k in ["stats_summary"]},
            ),
        ]

        # Render static components sequentially
        for render_func, fallback_height, func_kwargs in render_order:
            try:
                # Pass specific arguments required by each component
                if render_func == self.notification_renderer.render:
                    notification_rect = pygame.Rect(
                        10, current_y + 5, panel_width - 20, fallback_height
                    )
                    render_func(notification_rect, func_kwargs.get("stats_summary", {}))
                    next_y = notification_rect.bottom
                else:
                    next_y = render_func(
                        y_start=current_y + 5,
                        panel_width=panel_width,
                        **func_kwargs,  # Pass the specific kwargs for this function
                    )

                current_y = (
                    next_y
                    if isinstance(next_y, (int, float))
                    else current_y + fallback_height + 5
                )
            except Exception as e:
                logger.error(
                    f"Error rendering component {render_func.__name__}: {e}",
                    exc_info=True,
                )
                error_rect = pygame.Rect(
                    10, current_y + 5, panel_width - 20, fallback_height
                )
                pygame.draw.rect(self.screen, VisConfig.RED, error_rect, 1)
                current_y += fallback_height + 5

        # --- Render Plots Area ---
        app_state_str = render_data.get("app_state", AppState.UNKNOWN.value)
        should_render_plots = app_state_str == AppState.MAIN_MENU.value

        plot_y_start = current_y + 5
        try:
            self.plot_area_renderer.render(
                y_start=plot_y_start,
                panel_width=panel_width,
                screen_height=current_height,
                plot_data=render_data.get("plot_data", {}),
                status=status,
                render_enabled=should_render_plots,
            )
        except Exception as e:
            logger.error(f"Error in plot_area_renderer: {e}", exc_info=True)
            plot_area_height = current_height - plot_y_start - 10
            plot_area_width = panel_width - 20
            if plot_area_width > 10 and plot_area_height > 10:
                plot_area_rect = pygame.Rect(
                    10, plot_y_start, plot_area_width, plot_area_height
                )
                pygame.draw.rect(self.screen, (80, 0, 0), plot_area_rect, 2)
