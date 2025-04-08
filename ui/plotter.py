# File: ui/plotter.py
import pygame
from typing import Dict, Optional, Deque, Tuple
from collections import deque
import matplotlib
import time
from io import BytesIO
import logging
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, StatsConfig
from .plot_utils import render_single_plot, normalize_color_for_matplotlib

logger = logging.getLogger(__name__)


class Plotter:
    """Handles creation and caching of the multi-plot Matplotlib surface."""

    def __init__(self):
        self.plot_surface_cache: Optional[pygame.Surface] = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = 2.5
        self.rolling_window_sizes = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW
        self.colors = self._init_colors()

        self.fig: Optional[plt.Figure] = None
        self.axes: Optional[np.ndarray] = None
        self.last_target_size: Tuple[int, int] = (0, 0)
        self.last_data_hash: Optional[int] = None

        logger.info(
            f"[Plotter] Initialized with update interval: {self.plot_update_interval}s"
        )

    def _init_colors(self) -> Dict[str, Tuple[float, float, float]]:
        """Initializes plot colors."""
        return {
            "game_scores": normalize_color_for_matplotlib(VisConfig.GREEN),
            "policy_losses": normalize_color_for_matplotlib(VisConfig.RED),
            "value_losses": normalize_color_for_matplotlib(VisConfig.ORANGE),
            "episode_lengths": normalize_color_for_matplotlib(VisConfig.BLUE),
            "episode_outcomes": normalize_color_for_matplotlib(VisConfig.YELLOW),
            "episode_triangles_cleared": normalize_color_for_matplotlib(VisConfig.CYAN),
            "lr_values": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[2]),
            "buffer_sizes": normalize_color_for_matplotlib(VisConfig.PURPLE),
            "best_game_score_history": normalize_color_for_matplotlib(VisConfig.GREEN),
            "mcts_simulation_times": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[1]
            ),
            "mcts_nn_prediction_times": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[3]
            ),
            "mcts_nodes_explored": normalize_color_for_matplotlib(VisConfig.LIGHTG),
            "mcts_avg_depths": normalize_color_for_matplotlib(VisConfig.WHITE),
            "steps_per_second": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[0]
            ),
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
        }

    def _init_figure(self, target_width: int, target_height: int):
        """Initializes the Matplotlib figure and axes."""
        logger.debug(
            f"[Plotter] Initializing Matplotlib figure for size {target_width}x{target_height}"
        )
        if self.fig:
            plt.close(self.fig)

        dpi = 90
        fig_width_in = max(1, target_width / dpi)
        fig_height_in = max(1, target_height / dpi)

        try:
            # 5 rows, 3 columns grid
            self.fig, self.axes = plt.subplots(
                5, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
            )
            self.fig.subplots_adjust(
                hspace=0.3, wspace=0.2, left=0.07, right=0.97, bottom=0.05, top=0.97
            )
            self.last_target_size = (target_width, target_height)
            logger.debug("[Plotter] Matplotlib figure initialized (5x3 grid).")
        except Exception as e:
            logger.error(f"Error creating Matplotlib figure: {e}", exc_info=True)
            self.fig = None
            self.axes = None
            self.last_target_size = (0, 0)

    def _get_data_hash(self, plot_data: Dict[str, Deque]) -> int:
        """Generates a simple hash based on data lengths and last elements."""
        hash_val = 0
        # Include all keys used in plot_defs
        keys_to_hash = [
            "game_scores",
            "episode_outcomes",
            "episode_lengths",
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_triangles_cleared",
            "best_game_score_history",
            "buffer_sizes",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "steps_per_second",
            "mcts_nodes_explored",
            "mcts_avg_depths",
        ]
        for key in sorted(keys_to_hash):
            dq = plot_data.get(key)  # Use .get() to handle missing keys gracefully
            if dq is None:
                continue  # Skip if key doesn't exist in plot_data
            hash_val ^= hash(key)
            hash_val ^= len(dq)
            if dq:
                try:
                    last_elem = dq[-1]
                    if isinstance(last_elem, (int, float)):
                        hash_val ^= hash(f"{last_elem:.6f}")
                    else:
                        hash_val ^= hash(str(last_elem))
                except IndexError:
                    pass
        return hash_val

    def _update_plot_data(self, plot_data: Dict[str, Deque]):
        """Updates the data on the existing Matplotlib axes."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()
            # --- Define Plot Order (5x3 Grid) ---
            plot_defs = [
                # Row 1
                ("game_scores", "Game Score", self.colors["game_scores"], False),
                (
                    "episode_outcomes",
                    "Ep Outcome",
                    self.colors["episode_outcomes"],
                    False,
                ),
                ("episode_lengths", "Ep Length", self.colors["episode_lengths"], False),
                # Row 2
                ("policy_losses", "Policy Loss", self.colors["policy_losses"], True),
                ("value_losses", "Value Loss", self.colors["value_losses"], True),
                ("lr_values", "Learning Rate", self.colors["lr_values"], True),
                # Row 3
                (
                    "episode_triangles_cleared",
                    "Tris Cleared/Ep",
                    self.colors["episode_triangles_cleared"],
                    False,
                ),
                (
                    "best_game_score_history",
                    "Best Score Hist",
                    self.colors["best_game_score_history"],
                    False,
                ),
                ("buffer_sizes", "Buffer Size", self.colors["buffer_sizes"], False),
                # Row 4 (MCTS Timings / System)
                (
                    "mcts_simulation_times",
                    "MCTS Sim Time (s)",
                    self.colors["mcts_simulation_times"],
                    False,
                ),
                (
                    "mcts_nn_prediction_times",
                    "MCTS NN Time (s)",
                    self.colors["mcts_nn_prediction_times"],
                    False,
                ),
                (
                    "steps_per_second",
                    "Steps/Sec",
                    self.colors["steps_per_second"],
                    False,
                ),
                # Row 5 (MCTS Structure)
                (
                    "mcts_nodes_explored",
                    "MCTS Nodes Explored",
                    self.colors["mcts_nodes_explored"],
                    False,
                ),
                (
                    "mcts_avg_depths",
                    "MCTS Avg Depth",
                    self.colors["mcts_avg_depths"],
                    False,
                ),
                ("placeholder3", "Future Plot", self.colors["placeholder"], False),
            ]

            data_lists = {
                key: list(plot_data.get(key, deque())) for key, _, _, _ in plot_defs
            }

            for i, (data_key, label, color, log_scale) in enumerate(plot_defs):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                ax.clear()
                render_single_plot(
                    ax,
                    data_lists[data_key],
                    label,
                    color,
                    self.rolling_window_sizes,
                    placeholder_text=label,
                    y_log_scale=log_scale,
                )
                if i < 12:  # Hide x-labels for top 4 rows
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=0)

            plot_update_duration = time.monotonic() - plot_update_start
            logger.debug(f"[Plotter] Plot data updated in {plot_update_duration:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error updating plot data: {e}", exc_info=True)
            try:
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
                scale_diff = abs(current_size[0] - target_width) + abs(
                    current_size[1] - target_height
                )
                if scale_diff > 10:
                    plot_img_surface = pygame.transform.smoothscale(
                        plot_img_surface, (target_width, target_height)
                    )
                else:
                    plot_img_surface = pygame.transform.scale(
                        plot_img_surface, (target_width, target_height)
                    )

            render_duration = time.monotonic() - render_start
            logger.debug(
                f"[Plotter] Figure rendered to surface in {render_duration:.4f}s"
            )
            return plot_img_surface

        except Exception as e:
            logger.error(
                f"Error rendering Matplotlib figure to surface: {e}", exc_info=True
            )
            return None

    def get_cached_or_updated_plot(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Returns the cached plot surface or creates/updates one if needed."""
        current_time = time.time()
        has_data = any(d for d in plot_data.values())
        target_size = (target_width, target_height)

        needs_reinit = (
            self.fig is None
            or self.axes is None
            or self.last_target_size != target_size
        )

        current_data_hash = self._get_data_hash(plot_data)
        data_changed = self.last_data_hash != current_data_hash
        time_elapsed = (
            current_time - self.last_plot_update_time > self.plot_update_interval
        )
        needs_update = data_changed or time_elapsed

        can_create_plot = target_width > 50 and target_height > 50

        if not can_create_plot:
            if self.plot_surface_cache is not None:
                logger.debug("[Plotter] Target size too small, clearing cache.")
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None

        if not has_data:
            if self.plot_surface_cache is not None:
                logger.debug("[Plotter] No data, clearing cache.")
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
                logger.debug(
                    f"[Plotter] {cache_status}. Reason: fig/axes is None or size changed ({self.last_target_size} != {target_size})"
                )
                self._init_figure(target_width, target_height)
                if self.fig:
                    if self._update_plot_data(plot_data):
                        self.plot_surface_cache = self._render_figure_to_surface(
                            target_width, target_height
                        )
                        self.last_plot_update_time = current_time
                        self.last_data_hash = current_data_hash
                    else:
                        self.plot_surface_cache = None
                else:
                    self.plot_surface_cache = None

            elif needs_update:
                cache_status = (
                    f"MISS (Update - Data: {data_changed}, Time: {time_elapsed})"
                )
                logger.debug(f"[Plotter] {cache_status}")
                if self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    logger.warning(
                        "[Plotter] Plot data update failed, returning potentially stale cache."
                    )
                    cache_status = "ERROR (Update Failed)"

            elif self.plot_surface_cache is None:
                cache_status = "MISS (Cache None)"
                logger.debug(f"[Plotter] {cache_status}, attempting re-render.")
                if self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash


        except Exception as e:
            logger.error(
                f"[Plotter] Unexpected error in get_cached_or_updated_plot: {e}",
                exc_info=True,
            )
            self.plot_surface_cache = None
            if self.fig:
                plt.close(self.fig)
            self.fig, self.axes = None, None
            self.last_target_size = (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        if self.fig:
            try:
                plt.close(self.fig)
                logger.debug("[Plotter] Matplotlib figure closed in destructor.")
            except Exception as e:
                logger.error(f"[Plotter] Error closing figure in destructor: {e}")
