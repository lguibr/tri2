# File: ui/panels/left_panel_components/plot_area_renderer.py
import pygame
from typing import Dict, Deque, Any, Optional, Tuple
import numpy as np
from config import (
    VisConfig,
    LIGHTG,
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    GRAY,
)  # Added GRAY
from ui.plotter import Plotter


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

    def render(
        self,
        y_start: int,
        panel_width: int,
        screen_height: int,
        plot_data: Dict[str, Deque],
        status: str,
    ):
        """Renders the plot area."""
        plot_area_y_start = y_start
        plot_area_height = screen_height - plot_area_y_start - 10
        plot_area_width = panel_width - 20

        if plot_area_width <= 50 or plot_area_height <= 50:
            return

        plot_surface = self.plotter.get_cached_or_updated_plot(
            plot_data, plot_area_width, plot_area_height
        )
        plot_area_rect = pygame.Rect(
            10, plot_area_y_start, plot_area_width, plot_area_height
        )

        if plot_surface:
            self.screen.blit(plot_surface, plot_area_rect.topleft)
        else:
            pygame.draw.rect(self.screen, (40, 40, 40), plot_area_rect, 1)
            placeholder_text = "Waiting for data..."
            if status == "Buffering":
                placeholder_text = "Buffering... Waiting for plot data..."
            elif status == "Error":
                placeholder_text = "Plotting disabled due to error."
            elif not plot_data or not any(plot_data.values()):
                placeholder_text = "No plot data yet..."

            placeholder_font = self.fonts.get("plot_placeholder")
            if placeholder_font:
                placeholder_surf = placeholder_font.render(placeholder_text, True, GRAY)
                placeholder_rect = placeholder_surf.get_rect(
                    center=plot_area_rect.center
                )
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
            else:  # Fallback cross
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
