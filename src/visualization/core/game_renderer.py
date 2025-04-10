# File: src/visualization/core/game_renderer.py
import pygame
import logging
import math
import ray # Import Ray
from typing import TYPE_CHECKING, Dict, Optional, Any

# Use relative imports within visualization package
from . import colors, layout, coord_mapper
from ..ui import ProgressBar

# Import drawing functions/modules directly from their files
from ..drawing import grid as grid_drawing
from ..drawing import previews as preview_drawing
from ..drawing import hud as hud_drawing

# Import Plotter and StatsCollectorData
from src.stats import Plotter, StatsCollectorData # Keep Plotter import
# Removed: from src.utils.types import StatsCollectorData

# Import GameState for type hinting AND for creating a default instance
from src.environment import GameState

if TYPE_CHECKING:
    from ...config import VisConfig, EnvConfig
    from src.stats import StatsCollectorActor # Import actor type hint

logger = logging.getLogger(__name__)


class GameRenderer:
    """
    Renders multiple GameStates (from workers) in a top grid area,
    and statistics plots/progress bars in a bottom area.
    Fetches plot data from StatsCollectorActor.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: Dict[str, Optional[pygame.font.Font]],
        stats_collector_actor: Optional["StatsCollectorActor"] = None, # Add actor handle
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.stats_collector_actor = stats_collector_actor # Store handle
        self.layout_rects: Optional[Dict[str, pygame.Rect]] = None
        self.worker_sub_rects: Dict[int, Dict[str, pygame.Rect]] = {}
        self.last_worker_grid_size = (0, 0)
        self.last_num_workers = 0

        # Instantiate Plotter locally - it will be used within render
        # Its internal caching will handle performance.
        self.plotter = Plotter(plot_update_interval=0.2) # Faster update interval for plotter cache check

        # Progress bar layout constants
        self.progress_bar_height_per_bar = 45
        self.num_progress_bars = 2
        self.progress_bar_spacing = 5
        self.progress_bars_total_height = (self.progress_bar_height_per_bar + self.progress_bar_spacing) * self.num_progress_bars

    def _calculate_layout(self):
        """Calculates or retrieves the main layout areas."""
        current_w, current_h = self.screen.get_size()
        needs_recalc = True
        if self.layout_rects:
            if self.layout_rects.get("screen_size") == (current_w, current_h):
                needs_recalc = False

        if needs_recalc:
            required_bottom_margin_for_stats = self.progress_bars_total_height + self.vis_config.PADDING
            self.layout_rects = layout.calculate_layout(current_w, current_h, self.vis_config, bottom_margin=required_bottom_margin_for_stats)
            self.layout_rects["screen_size"] = (current_w, current_h)
            logger.debug(f"Recalculated main layout for screen {current_w}x{current_h}: {self.layout_rects}")
            self.last_worker_grid_size = (0, 0)
            self.worker_sub_rects = {}

        return self.layout_rects

    def _calculate_worker_sub_layout(self, worker_grid_area: pygame.Rect, num_workers: int):
        """Calculates the grid layout within the worker_grid_area."""
        area_w, area_h = worker_grid_area.size
        if (area_w, area_h) == self.last_worker_grid_size and num_workers == self.last_num_workers:
            return

        logger.debug(f"Recalculating worker sub-layout for {num_workers} workers in area {area_w}x{area_h}")
        self.last_worker_grid_size = (area_w, area_h)
        self.last_num_workers = num_workers
        self.worker_sub_rects = {}
        pad = 5

        if area_h <= 10 or area_w <= 10 or num_workers <= 0:
            logger.warning(f"Worker grid area too small ({area_w}x{area_h}) or no workers ({num_workers}). Cannot calculate sub-layout.")
            return

        cols = math.ceil(math.sqrt(num_workers))
        rows = math.ceil(num_workers / cols)
        cell_w = (area_w - (cols - 1) * pad) / cols
        cell_h = (area_h - (rows - 1) * pad) / rows
        min_cell_w, min_cell_h = 80, 60
        if cell_w < min_cell_w or cell_h < min_cell_h:
            logger.warning(f"Worker grid cells too small ({cell_w:.1f}x{cell_h:.1f}). May impact rendering.")

        logger.info(f"Calculated worker sub-layout: {rows}x{cols} for {num_workers} workers. Cell: {cell_w:.1f}x{cell_h:.1f}")

        for worker_id in range(num_workers):
            row = worker_id // cols
            col = worker_id % cols
            worker_area_x = worker_grid_area.left + col * (cell_w + pad)
            worker_area_y = worker_grid_area.top + row * (cell_h + pad)
            worker_area_w = cell_w
            worker_area_h = cell_h
            preview_w = max(10, min(worker_area_w * 0.2, 60))
            grid_w = worker_area_w - preview_w - pad
            grid_w = max(0, grid_w)
            preview_w = max(0, preview_w)
            worker_area_h = max(0, worker_area_h)
            grid_rect = pygame.Rect(worker_area_x, worker_area_y, grid_w, worker_area_h)
            preview_rect = pygame.Rect(grid_rect.right + pad, worker_area_y, preview_w, worker_area_h)
            worker_rect = pygame.Rect(worker_area_x, worker_area_y, worker_area_w, worker_area_h)

            self.worker_sub_rects[worker_id] = {
                "area": worker_rect.clip(worker_grid_area),
                "grid": grid_rect.clip(worker_grid_area),
                "preview": preview_rect.clip(worker_grid_area),
            }
            logger.debug(f"  Worker {worker_id} layout: Area={self.worker_sub_rects[worker_id]['area']}, Grid={self.worker_sub_rects[worker_id]['grid']}, Preview={self.worker_sub_rects[worker_id]['preview']}")

    def render(
        self,
        worker_states: Dict[int, "GameState"],
        global_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Renders worker states, plots (fetching data from actor), and progress bars.
        """
        layout_rects = self._calculate_layout()
        if not layout_rects:
            logger.error("Main layout calculation failed.")
            return

        worker_grid_area = layout_rects.get("worker_grid")
        stats_area = layout_rects.get("stats_area")
        plots_rect = layout_rects.get("plots")

        # --- Render Worker Grid Area ---
        if worker_grid_area and worker_grid_area.width > 0 and worker_grid_area.height > 0:
            num_workers_to_display = len(worker_states)
            self._calculate_worker_sub_layout(worker_grid_area, num_workers_to_display)
            pygame.draw.rect(self.screen, colors.DARK_GRAY, worker_grid_area)

            for worker_id, game_state in worker_states.items():
                if worker_id not in self.worker_sub_rects:
                    logger.warning(f"Skipping render for worker {worker_id}: No sub-layout calculated.")
                    continue
                sub_layout = self.worker_sub_rects[worker_id]
                area_rect, grid_rect, preview_rect = sub_layout["area"], sub_layout["grid"], sub_layout["preview"]
                pygame.draw.rect(self.screen, colors.GRAY, area_rect, 1)
                logger.debug(f"Rendering Worker {worker_id} in area {area_rect}")

                # Render Grid
                if grid_rect.width > 0 and grid_rect.height > 0:
                    try:
                        grid_surf = self.screen.subsurface(grid_rect)
                        bg_color = colors.GRID_BG_GAME_OVER if game_state.is_over() else colors.GRID_BG_DEFAULT
                        grid_drawing.draw_grid_background(grid_surf, bg_color)
                        grid_drawing.draw_grid_triangles(grid_surf, game_state.grid_data, self.env_config)
                        id_font = self.fonts.get("help")
                        if id_font:
                            id_surf = id_font.render(f"W{worker_id}", True, colors.LIGHT_GRAY)
                            grid_surf.blit(id_surf, (3, 3))
                    except ValueError as e:
                        if "subsurface rectangle is invalid" not in str(e): logger.error(f"Error creating grid subsurface for W{worker_id} ({grid_rect}): {e}")
                        pygame.draw.rect(self.screen, colors.RED, grid_rect, 1)

                # Render Previews
                if preview_rect.width > 0 and preview_rect.height > 0:
                    try:
                        preview_surf = self.screen.subsurface(preview_rect)
                        _ = preview_drawing.render_previews(preview_surf, game_state, preview_rect.topleft, "training_visual", self.env_config, self.vis_config)
                    except ValueError as e:
                        if "subsurface rectangle is invalid" not in str(e): logger.error(f"Error creating preview subsurface for W{worker_id} ({preview_rect}): {e}")
                        pygame.draw.rect(self.screen, colors.RED, preview_rect, 1)
        else:
            logger.warning("Worker grid area not available or too small.")

        # --- Render Stats Area (Plots and Progress Bars) ---
        if stats_area and global_stats:
            pygame.draw.rect(self.screen, colors.DARK_GRAY, stats_area) # Background

            # --- Render Plots ---
            plot_surface = None
            if plots_rect and plots_rect.width > 0 and plots_rect.height > 0:
                # Fetch data from StatsCollectorActor if available
                stats_data_for_plot: Optional[StatsCollectorData] = global_stats.get("stats_data") # Get data passed via queue

                if stats_data_for_plot is not None:
                    # Use the local plotter instance to get the surface
                    plot_surface = self.plotter.get_plot_surface(
                        stats_data_for_plot,
                        plots_rect.width,
                        plots_rect.height,
                    )

                if plot_surface:
                    self.screen.blit(plot_surface, plots_rect.topleft)
                else:
                    # Draw placeholder if no surface generated (no data or error)
                    pygame.draw.rect(self.screen, colors.DARK_GRAY, plots_rect)
                    plot_font = self.fonts.get("help")
                    if plot_font:
                        wait_surf = plot_font.render("Plot Area (Waiting for data...)", True, colors.LIGHT_GRAY)
                        wait_rect = wait_surf.get_rect(center=plots_rect.center)
                        self.screen.blit(wait_surf, wait_rect)
                    pygame.draw.rect(self.screen, colors.GRAY, plots_rect, 1)

            # --- Render Progress Bars below plots ---
            progress_bar_font = self.fonts.get("help")
            if progress_bar_font and plots_rect:
                bar_y = plots_rect.bottom + self.progress_bar_spacing
                bar_width = stats_area.width
                bar_x = stats_area.left
                bar_height = self.progress_bar_height_per_bar

                if bar_y + bar_height <= stats_area.bottom:
                    train_progress = global_stats.get("train_progress")
                    if isinstance(train_progress, ProgressBar):
                        train_progress.render(self.screen, (bar_x, bar_y), bar_width, bar_height, progress_bar_font, bar_color=colors.GREEN)
                        bar_y += bar_height + self.progress_bar_spacing
                    else: logger.debug("Train progress bar data not available.")

                    if bar_y + bar_height <= stats_area.bottom:
                        buffer_progress = global_stats.get("buffer_progress")
                        if isinstance(buffer_progress, ProgressBar):
                            buffer_progress.render(self.screen, (bar_x, bar_y), bar_width, bar_height, progress_bar_font, bar_color=colors.ORANGE)
                        else: logger.debug("Buffer progress bar data not available.")
                    else: logger.warning("Not enough vertical space in stats_area for the second progress bar.")
                else: logger.warning("Not enough vertical space in stats_area for the first progress bar.")

        # --- Render Global HUD (at the very bottom) ---
        representative_gs = worker_states.get(0) or next(iter(worker_states.values()), GameState(self.env_config))
        hud_drawing.render_hud(self.screen, representative_gs, "training_visual", self.fonts, global_stats)
