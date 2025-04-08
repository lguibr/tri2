import pygame
from typing import Optional, Tuple
import logging

from config import VisConfig
from ui.plotter import Plotter
from ui.input_handler import InputHandler
from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    PlotAreaRenderer,
    NotificationRenderer,
)
from app_state import AppState

logger = logging.getLogger(__name__)


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.input_handler: Optional[InputHandler] = None

        # Initialize components
        self.button_status_renderer = ButtonStatusRenderer(self.screen, self.fonts)
        self.info_text_renderer = InfoTextRenderer(self.screen, self.fonts)
        self.notification_renderer = NotificationRenderer(self.screen, self.fonts)
        self.plot_area_renderer = PlotAreaRenderer(
            self.screen, self.fonts, self.plotter
        )

    def _init_fonts(self):
        """Initializes fonts used in the left panel."""
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
        # Ensure essential fonts have fallbacks
        if fonts.get("ui") is None:
            fonts["ui"] = pygame.font.Font(None, 24)
        if fonts.get("status") is None:
            fonts["status"] = pygame.font.Font(None, 28)
        return fonts

    def _get_background_color(self, status: str) -> Tuple[int, int, int]:
        """Determines background color based on status."""
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

    def render(self, panel_width: int, **kwargs):  # Use kwargs
        """Renders the entire left panel within the given width."""
        current_height = self.screen.get_height()
        lp_rect = pygame.Rect(0, 0, panel_width, current_height)
        status = kwargs.get("status", "")
        bg_color = self._get_background_color(status)
        pygame.draw.rect(self.screen, bg_color, lp_rect)

        current_y = 10
        render_order = [
            (self.button_status_renderer.render, 60),
            (self.info_text_renderer.render, 80),
            (self.notification_renderer.render, 90),
        ]

        # Render static components
        for render_func, fallback_height in render_order:
            try:
                # --- Pass only the required arguments ---
                if render_func == self.button_status_renderer.render:
                    next_y = render_func(
                        y_start=current_y,
                        panel_width=panel_width,
                        app_state=kwargs.get("app_state", ""),
                        is_process_running=kwargs.get("is_process_running", False),
                        status=status,
                        stats_summary=kwargs.get("stats_summary", {}),
                        update_progress_details=kwargs.get(
                            "update_progress_details", {}
                        ),
                    )
                elif render_func == self.info_text_renderer.render:
                    next_y = render_func(
                        y_start=current_y + 5,
                        stats_summary=kwargs.get("stats_summary", {}),
                        panel_width=panel_width,
                        agent_param_count=kwargs.get("agent_param_count", 0),
                        worker_counts=kwargs.get("worker_counts", {}),
                    )
                elif render_func == self.notification_renderer.render:
                    notification_rect = pygame.Rect(
                        10, current_y + 5, panel_width - 20, fallback_height
                    )
                    render_func(notification_rect, kwargs.get("stats_summary", {}))
                    next_y = notification_rect.bottom
                else:  # Default case if more components added (might need adjustment)
                    next_y = render_func(
                        y_start=current_y + 5, panel_width=panel_width, **kwargs
                    )  # Keep kwargs for unknown future components

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
                current_y += fallback_height + 5  # Fallback increment

        # --- Render Plots Area ---
        # Determine if plots should be rendered based on app state
        app_state_str = kwargs.get("app_state", AppState.UNKNOWN.value)
        # Render plots only when in the main menu
        should_render_plots = app_state_str == AppState.MAIN_MENU.value

        plot_y_start = current_y + 5
        try:
            # Pass the render_enabled flag directly
            self.plot_area_renderer.render(
                y_start=plot_y_start,
                panel_width=panel_width,
                screen_height=current_height,
                plot_data=kwargs.get("plot_data", {}),
                status=status,
                render_enabled=should_render_plots,  # Pass the calculated flag
            )
        except Exception as e:
            logger.error(f"Error in plot_area_renderer: {e}", exc_info=True)
            # Optionally draw an error box in the plot area
            plot_area_height = current_height - plot_y_start - 10
            plot_area_width = panel_width - 20
            if plot_area_width > 10 and plot_area_height > 10:
                plot_area_rect = pygame.Rect(
                    10, plot_y_start, plot_area_width, plot_area_height
                )
                pygame.draw.rect(self.screen, (80, 0, 0), plot_area_rect, 2)

    def _render_plot_placeholder(
        self,
        y_start: int,
        panel_width: int,
        screen_height: int,
        message: str = "Plots Disabled",
    ):
        """Renders a placeholder when plots are disabled."""
        plot_area_height = screen_height - y_start - 10
        plot_area_width = panel_width - 20
        if plot_area_width > 10 and plot_area_height > 10:
            plot_area_rect = pygame.Rect(10, y_start, plot_area_width, plot_area_height)
            pygame.draw.rect(self.screen, (40, 40, 40), plot_area_rect, 1)
            placeholder_font = self.fonts.get("plot_placeholder")
            if placeholder_font:
                placeholder_surf = placeholder_font.render(
                    message, True, (100, 100, 100)
                )
                placeholder_rect = placeholder_surf.get_rect(
                    center=plot_area_rect.center
                )
                self.screen.blit(placeholder_surf, placeholder_rect)
