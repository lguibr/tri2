# File: src/stats/plotter.py
import pygame
from typing import Dict, Optional, Deque, Tuple, List
from collections import deque
import matplotlib
import time
from io import BytesIO
import logging
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use relative imports within stats module
from .collector import StatsCollectorData
from .plot_utils import render_single_plot, normalize_color_for_matplotlib

# Import colors from visualization module
from src.visualization.core import colors as vis_colors

logger = logging.getLogger(__name__)


class Plotter:
    """Handles creation and caching of the multi-plot Matplotlib surface."""

    def __init__(self, plot_update_interval: float = 0.5):
        self.plot_surface_cache: Optional[pygame.Surface] = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = plot_update_interval
        # Change: Update rolling window sizes
        self.rolling_window_sizes: List[int] = [10, 50, 100, 500, 1000, 5000]
        self.colors = self._init_colors()

        self.fig: Optional[plt.Figure] = None
        self.axes: Optional[np.ndarray] = None
        self.last_target_size: Tuple[int, int] = (0, 0)
        self.last_data_hash: Optional[int] = None

        logger.info(
            f"[Plotter] Initialized with update interval: {self.plot_update_interval}s"
        )

    def _init_colors(self) -> Dict[str, Tuple[float, float, float]]:
        """Initializes plot colors using vis_colors."""
        return {
            "SelfPlay/Episode_Score": normalize_color_for_matplotlib(vis_colors.YELLOW),
            "Loss/Total": normalize_color_for_matplotlib(vis_colors.RED),
            "Loss/Value": normalize_color_for_matplotlib(vis_colors.BLUE),
            "Loss/Policy": normalize_color_for_matplotlib(vis_colors.GREEN),
            "LearningRate": normalize_color_for_matplotlib(vis_colors.CYAN),
            "SelfPlay/Episode_Length": normalize_color_for_matplotlib(
                vis_colors.ORANGE
            ),
            "Buffer/Size": normalize_color_for_matplotlib(vis_colors.PURPLE),
            "MCTS/Avg_Root_Visits": normalize_color_for_matplotlib(
                vis_colors.LIGHT_GRAY
            ),
            "MCTS/Avg_Tree_Depth": normalize_color_for_matplotlib(vis_colors.LIGHTG),
            "placeholder": normalize_color_for_matplotlib(vis_colors.GRAY),
        }

    def _init_figure(self, target_width: int, target_height: int):
        """Initializes the Matplotlib figure and axes."""
        logger.info(
            f"[Plotter] Initializing Matplotlib figure for size {target_width}x{target_height}"
        )
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                logger.warning(f"Error closing previous figure: {e}")

        dpi = 96
        fig_width_in = max(1, target_width / dpi)
        fig_height_in = max(1, target_height / dpi)

        try:
            nrows, ncols = 2, 4
            self.fig, self.axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width_in, fig_height_in),
                dpi=dpi,
                sharex=False,
            )
            self.fig.subplots_adjust(
                hspace=0.4, wspace=0.35, left=0.08, right=0.98, bottom=0.15, top=0.92
            )
            self.last_target_size = (target_width, target_height)
            logger.info(
                f"[Plotter] Matplotlib figure initialized ({nrows}x{ncols} grid)."
            )
        except Exception as e:
            logger.error(f"Error creating Matplotlib figure: {e}", exc_info=True)
            self.fig = None
            self.axes = None
            self.last_target_size = (0, 0)

    def _get_data_hash(self, plot_data: StatsCollectorData) -> int:
        """Generates a simple hash based on data lengths and last elements."""
        hash_val = 0
        for key in sorted(plot_data.keys()):
            dq = plot_data[key]
            if not dq:
                continue
            hash_val ^= hash(key)
            hash_val ^= len(dq)
            try:
                last_step, last_val = dq[-1]
                hash_val ^= hash(last_step)
                hash_val ^= hash(f"{last_val:.6f}")
            except IndexError:
                pass
        return hash_val

    def _update_plot_data(self, plot_data: StatsCollectorData):
        """Updates the data on the existing Matplotlib axes."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()
            plot_defs = [
                ("SelfPlay/Episode_Score", "Ep Score", False),
                ("Loss/Total", "Total Loss", True),
                ("MCTS/Avg_Root_Visits", "Root Visits", False),
                ("LearningRate", "Learn Rate", True),
                ("SelfPlay/Episode_Length", "Ep Length", False),
                ("Loss/Value", "Value Loss", True),
                ("Loss/Policy", "Policy Loss", True),
                ("MCTS/Avg_Tree_Depth", "Tree Depth", False),
            ]

            # Extract steps and values separately
            data_values: Dict[str, List[float]] = {}
            data_steps: Dict[str, List[int]] = {}
            for key, _, _ in plot_defs:
                dq = plot_data.get(key, deque())
                if dq:
                    steps, values = zip(*dq)
                    data_values[key] = list(values)
                    data_steps[key] = list(steps)
                else:
                    data_values[key] = []
                    data_steps[key] = []

            for i, (data_key, label, log_scale) in enumerate(plot_defs):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                ax.clear()

                current_values = data_values.get(data_key, [])
                current_steps = data_steps.get(data_key, [])  # Get corresponding steps
                color_mpl = self.colors.get(data_key, (0.5, 0.5, 0.5))

                render_single_plot(
                    ax,
                    current_steps,  # Pass steps as x_coords
                    current_values,  # Pass values as y_data
                    label,
                    color_mpl,
                    self.rolling_window_sizes,
                    show_placeholder=(not current_values),
                    placeholder_text=label,
                    y_log_scale=log_scale,
                )
                nrows, ncols = self.axes.shape
                if i < (nrows - 1) * ncols:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=0)

            plot_update_duration = time.monotonic() - plot_update_start
            return True

        except Exception as e:
            logger.error(f"Error updating plot data: {e}", exc_info=True)
            try:
                if self.axes is not None:
                    for ax in self.axes.flatten():
                        ax.clear()
            except Exception:
                pass
            return False

    def _render_figure_to_surface(
        self, target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Renders the current Matplotlib figure to a Pygame surface."""
        if self.fig is None:
            logger.warning("[Plotter] Cannot render figure, not initialized.")
            return None

        render_start = time.monotonic()
        try:
            self.fig.canvas.draw()
            buf = BytesIO()
            self.fig.savefig(
                buf,
                format="png",
                transparent=False,
                facecolor=plt.rcParams["figure.facecolor"],
            )
            buf.seek(0)
            plot_img_surface = pygame.image.load(buf, "png").convert()
            buf.close()

            current_size = plot_img_surface.get_size()
            if current_size != (target_width, target_height):
                plot_img_surface = pygame.transform.scale(
                    plot_img_surface, (target_width, target_height)
                )
            render_duration = time.monotonic() - render_start
            return plot_img_surface

        except Exception as e:
            logger.error(f"Error rendering Matplotlib figure: {e}", exc_info=True)
            return None

    def get_plot_surface(
        self, plot_data: StatsCollectorData, target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Returns the cached plot surface or creates/updates one if needed."""
        current_time = time.time()
        has_data = any(plot_data.values())
        target_size = (target_width, target_height)

        needs_reinit = (
            self.fig is None
            or self.axes is None
            or self.last_target_size != target_size
        )
        current_data_hash = self._get_data_hash(plot_data)
        data_changed = self.last_data_hash != current_data_hash
        time_elapsed = (
            current_time - self.last_plot_update_time
        ) > self.plot_update_interval
        needs_update = data_changed or time_elapsed
        can_create_plot = target_width > 50 and target_height > 50

        if not can_create_plot:
            if self.plot_surface_cache is not None:
                logger.info("[Plotter] Target size too small, clearing cache/figure.")
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None

        if not has_data:
            if self.plot_surface_cache is not None:
                logger.info("[Plotter] No plot data, clearing cache/figure.")
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None

        cache_status = "HIT"
        try:
            if needs_reinit:
                cache_status = "MISS (Re-init)"
                self._init_figure(target_width, target_height)
                if self.fig and self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    self.plot_surface_cache = None
            elif needs_update:
                cache_status = (
                    f"MISS (Update - Data: {data_changed}, Time: {time_elapsed})"
                )
                if self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    logger.warning(
                        "[Plotter] Plot update failed, returning stale cache."
                    )
                    cache_status = "ERROR (Update Failed)"
            elif self.plot_surface_cache is None:
                cache_status = "MISS (Cache None)"
                if self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash

        except Exception as e:
            logger.error(f"[Plotter] Error in get_plot_surface: {e}", exc_info=True)
            self.plot_surface_cache = None
            if self.fig:
                plt.close(self.fig)
            self.fig, self.axes = None, None
            self.last_target_size = (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        """Ensure Matplotlib figure is closed."""
        if self.fig:
            try:
                plt.close(self.fig)
                logger.info("[Plotter] Matplotlib figure closed.")
            except Exception as e:
                logger.error(f"[Plotter] Error closing figure in destructor: {e}")
