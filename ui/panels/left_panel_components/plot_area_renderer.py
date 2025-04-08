# File: ui/panels/left_panel_components/plot_area_renderer.py
import pygame
from typing import Dict, Deque, Any, Optional, Tuple
import numpy as np
import logging

from config import (
    VisConfig,
    LIGHTG,
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    GRAY,
)
from ui.plotter import Plotter

logger = logging.getLogger(__name__)


class PlotAreaRenderer:
    """Renders the plot area using a Plotter instance."""

    def __init__(
        self,
        screen: pygame.Surface,
        fonts: Dict[str, pygame.font.Font],
        plotter: Plotter,
    ):
        self.screen = screen
        self.fonts = fonts
        self.plotter = plotter
        self.placeholder_font = fonts.get(
            "plot_placeholder", pygame.font.Font(None, 20)
        )

    def render(
        self,
        y_start: int,
        panel_width: int,
        screen_height: int,
        plot_data: Dict[str, Deque],
        status: str,
        render_enabled: bool = True,  # Add the new argument with a default
    ):
        """Renders the plot area, conditionally based on render_enabled."""
        plot_area_y_start = y_start
        plot_area_height = screen_height - plot_area_y_start - 10
        plot_area_width = panel_width - 20

        if plot_area_width <= 50 or plot_area_height <= 50:
            # Optionally render a "too small" message if needed
            return

        plot_area_rect = pygame.Rect(
            10, plot_area_y_start, plot_area_width, plot_area_height
        )

        if not render_enabled:
            # Render placeholder if plots are explicitly disabled
            self._render_placeholder(plot_area_rect, "Plots Disabled")
            return

        # Attempt to get/create the plot surface only if enabled
        plot_surface = self.plotter.get_cached_or_updated_plot(
            plot_data, plot_area_width, plot_area_height
        )

        # Render the plot or a placeholder if data is missing/error
        if plot_surface:
            self.screen.blit(plot_surface, plot_area_rect.topleft)
        else:
            # Draw border and placeholder text
            placeholder_text = "Waiting for plot data..."
            if status == "Error":
                placeholder_text = "Plotting disabled due to error."
            elif not plot_data or not any(plot_data.values()):
                placeholder_text = "No plot data yet..."
            self._render_placeholder(plot_area_rect, placeholder_text)

    def _render_placeholder(self, plot_area_rect: pygame.Rect, message: str):
        """Renders a placeholder message within the plot area."""
        pygame.draw.rect(self.screen, (40, 40, 40), plot_area_rect, 1)
        if self.placeholder_font:
            placeholder_surf = self.placeholder_font.render(message, True, GRAY)
            placeholder_rect = placeholder_surf.get_rect(center=plot_area_rect.center)
            # Clip rendering to the plot area
            blit_pos = (
                max(plot_area_rect.left, placeholder_rect.left),
                max(plot_area_rect.top, placeholder_rect.top),
            )
            clip_area_rect = plot_area_rect.clip(placeholder_rect)
            blit_area = clip_area_rect.move(
                -placeholder_rect.left, -placeholder_rect.top
            )
            if blit_area.width > 0 and blit_area.height > 0:
                self.screen.blit(placeholder_surf, blit_pos, area=blit_area)
        else:  # Fallback cross if font failed
            pygame.draw.line(
                self.screen,
                GRAY,
                plot_area_rect.topleft,
                plot_area_rect.bottomright,
            )
            pygame.draw.line(
                self.screen,
                GRAY,
                plot_area_rect.topright,
                plot_area_rect.bottomleft,
            )
