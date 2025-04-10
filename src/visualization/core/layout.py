# File: src/visualization/core/layout.py
import pygame
from typing import Tuple, Dict, Optional
import logging

# Use VisConfig from central config
from ...config import VisConfig

logger = logging.getLogger(__name__)


def calculate_layout(
    screen_width: int,
    screen_height: int,
    vis_config: VisConfig,
    bottom_margin: int = 0,  # Margin needed below plots (for progress bars)
) -> Dict[str, pygame.Rect]:
    """
    Calculates layout rectangles splitting screen between worker grid and stats area.
    Gives more space to the stats/plots area.
    """
    sw, sh = screen_width, screen_height
    pad = vis_config.PADDING
    plot_internal_padding = 15  # Extra padding inside the stats area for the plots

    # Calculate total available height excluding top/bottom padding and HUD
    hud_h = vis_config.HUD_HEIGHT
    total_available_h = max(0, sh - hud_h - 2 * pad)

    # Target height for the top area (worker grid) ~35% (Reduced from 60%)
    top_area_h = int(total_available_h * 0.35)
    top_area_w = sw - 2 * pad

    worker_grid_rect = pygame.Rect(pad, pad, top_area_w, top_area_h)

    # Position stats area (plots + progress bars) below worker grid
    stats_area_y = worker_grid_rect.bottom + pad
    stats_area_w = sw - 2 * pad

    # Calculate stats area height: remaining space
    stats_area_h = max(0, sh - stats_area_y - pad - hud_h)
    stats_area_rect = pygame.Rect(pad, stats_area_y, stats_area_w, stats_area_h)

    # Subdivide stats area for plots and progress bars
    # Progress bars take 'bottom_margin' height from the bottom of the stats area
    # Add extra padding for the plot area itself
    plot_area_x = stats_area_rect.left + plot_internal_padding
    plot_area_y = stats_area_rect.top + plot_internal_padding
    plot_area_w = max(0, stats_area_rect.width - 2 * plot_internal_padding)
    # Calculate height available for plots, considering bottom margin and internal padding
    plot_area_h = max(
        0, stats_area_rect.height - bottom_margin - pad - 2 * plot_internal_padding
    )

    plot_rect = pygame.Rect(plot_area_x, plot_area_y, plot_area_w, plot_area_h)

    # Clip rectangles to screen bounds just in case
    screen_rect = pygame.Rect(0, 0, sw, sh)
    worker_grid_rect = worker_grid_rect.clip(screen_rect)
    stats_area_rect = stats_area_rect.clip(screen_rect)
    plot_rect = plot_rect.clip(screen_rect)  # Clip the padded plot rect

    logger.debug(
        f"Layout calculated: WorkerGrid={worker_grid_rect}, StatsArea={stats_area_rect}, PlotRect={plot_rect}"
    )

    return {
        "worker_grid": worker_grid_rect,
        "stats_area": stats_area_rect,  # The overall area for stats+progress
        "plots": plot_rect,  # The specific, padded area for plots
    }
