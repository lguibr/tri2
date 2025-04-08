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
        self.plot_update_interval: float = (
            2.0  # Faster update for more responsive plots
        )
        self.rolling_window_sizes = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW
        self.colors = self._init_colors()

        self.fig: Optional[plt.Figure] = None
        self.axes: Optional[np.ndarray] = None  # Will be a 2D numpy array of axes
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
            # MCTS Stats Colors
            "mcts_simulation_times": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[1]
            ),
            "mcts_nn_prediction_times": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[3]
            ),
            "mcts_nodes_explored": normalize_color_for_matplotlib(VisConfig.LIGHTG),
            "mcts_avg_depths": normalize_color_for_matplotlib(VisConfig.WHITE),
            # System Stats Colors
            "steps_per_second": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[0]
            ),
            # Placeholder Color
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
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

        dpi = 96  # Standard DPI
        fig_width_in = max(1, target_width / dpi)
        fig_height_in = max(1, target_height / dpi)

        try:
            # 5 rows, 3 columns grid = 15 plots
            self.fig, self.axes = plt.subplots(
                5, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
            )
            self.fig.subplots_adjust(
                hspace=0.4,
                wspace=0.3,  # Increased spacing slightly
                left=0.08,
                right=0.97,
                bottom=0.06,
                top=0.96,  # Adjusted margins
            )
            self.last_target_size = (target_width, target_height)
            logger.info("[Plotter] Matplotlib figure initialized (5x3 grid).")
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
            if dq is None or not dq:  # Check if deque exists and is not empty
                continue  # Skip if key doesn't exist or deque is empty

            hash_val ^= hash(key)
            hash_val ^= len(dq)
            try:
                # Hash based on the last element to detect changes
                last_elem = dq[-1]
                if isinstance(last_elem, (int, float)):
                    hash_val ^= hash(f"{last_elem:.6f}")  # Format floats consistently
                else:
                    hash_val ^= hash(str(last_elem))
            except IndexError:
                pass  # Should not happen if dq is not empty
        return hash_val

    def _update_plot_data(self, plot_data: Dict[str, Deque]):
        """Updates the data on the existing Matplotlib axes."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()  # Flatten the 2D array of axes
            # --- Define Plot Order (5x3 Grid) ---
            plot_defs = [
                # Row 1: Game Performance
                ("game_scores", "Game Score", self.colors["game_scores"], False),
                ("episode_lengths", "Ep Length", self.colors["episode_lengths"], False),
                (
                    "episode_triangles_cleared",
                    "Tris Cleared/Ep",
                    self.colors["episode_triangles_cleared"],
                    False,
                ),
                # Row 2: Losses & LR
                (
                    "policy_losses",
                    "Policy Loss",
                    self.colors["policy_losses"],
                    True,
                ),  # Log scale often helpful for losses
                (
                    "value_losses",
                    "Value Loss",
                    self.colors["value_losses"],
                    True,
                ),  # Log scale often helpful for losses
                (
                    "lr_values",
                    "Learning Rate",
                    self.colors["lr_values"],
                    True,
                ),  # Log scale for LR is common
                # Row 3: Buffer & History
                ("buffer_sizes", "Buffer Size", self.colors["buffer_sizes"], False),
                (
                    "best_game_score_history",
                    "Best Score Hist",
                    self.colors["best_game_score_history"],
                    False,
                ),
                (
                    "episode_outcomes",
                    "Ep Outcome (-1,0,1)",
                    self.colors["episode_outcomes"],
                    False,
                ),  # Added outcome plot
                # Row 4: MCTS Timings / System
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
                # Row 5: MCTS Structure
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
                (
                    "placeholder",
                    "Future Plot",
                    self.colors["placeholder"],
                    False,
                ),  # Keep one placeholder
            ]

            # Convert deques to lists for plotting (avoids issues with some matplotlib versions)
            data_lists = {
                key: list(plot_data.get(key, deque())) for key, _, _, _ in plot_defs
            }

            for i, (data_key, label, color, log_scale) in enumerate(plot_defs):
                if i >= len(axes_flat):
                    break  # Stop if we run out of axes
                ax = axes_flat[i]
                ax.clear()  # Clear previous plot content

                # Check if data exists before plotting
                current_data = data_lists.get(data_key, [])
                show_placeholder = (data_key == "placeholder") or not current_data

                render_single_plot(
                    ax,
                    current_data,
                    label,
                    color,
                    self.rolling_window_sizes,
                    show_placeholder=show_placeholder,
                    placeholder_text=label,  # Use label as placeholder text
                    y_log_scale=log_scale,
                )
                # Hide x-labels for plots not in the last row
                if i < len(axes_flat) - 3:  # Assuming 3 columns
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=0)  # Keep rotation at 0

            plot_update_duration = time.monotonic() - plot_update_start
            logger.info(f"[Plotter] Plot data updated in {plot_update_duration:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error updating plot data: {e}", exc_info=True)
            # Attempt to clear axes on error to avoid showing stale incorrect plots
            try:
                if self.axes is not None:
                    for ax in self.axes.flatten():
                        ax.clear()
            except Exception:
                pass  # Ignore errors during cleanup
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
            # Explicitly draw the canvas
            self.fig.canvas.draw()

            # Use BytesIO buffer to avoid disk I/O
            buf = BytesIO()
            self.fig.savefig(
                buf,
                format="png",
                transparent=False,  # Keep background
                facecolor=plt.rcParams[
                    "figure.facecolor"
                ],  # Use defined background color
            )
            buf.seek(0)

            # Load buffer into Pygame surface
            plot_img_surface = pygame.image.load(
                buf, "png"
            ).convert()  # Use convert() for performance
            buf.close()

            # --- Scaling ---
            # Scale the rendered surface to the exact target size if needed
            current_size = plot_img_surface.get_size()
            if current_size != (target_width, target_height):
                scale_diff = abs(current_size[0] - target_width) + abs(
                    current_size[1] - target_height
                )
                # Use smoothscale for larger differences, faster scale otherwise
                if scale_diff > 10:  # Arbitrary threshold
                    plot_img_surface = pygame.transform.smoothscale(
                        plot_img_surface, (target_width, target_height)
                    )
                else:
                    plot_img_surface = pygame.transform.scale(
                        plot_img_surface, (target_width, target_height)
                    )

            render_duration = time.monotonic() - render_start
            logger.info(
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
        has_data = any(d for d in plot_data.values())  # Check if any data exists
        target_size = (target_width, target_height)

        # Check if figure needs reinitialization (first time, or size changed)
        needs_reinit = (
            self.fig is None
            or self.axes is None
            or self.last_target_size != target_size
        )

        # Check if data has changed or enough time has passed
        current_data_hash = self._get_data_hash(plot_data)
        data_changed = self.last_data_hash != current_data_hash
        time_elapsed = (
            current_time - self.last_plot_update_time
        ) > self.plot_update_interval
        needs_update = data_changed or time_elapsed

        # Check if target size is large enough to render plots meaningfully
        can_create_plot = target_width > 50 and target_height > 50

        # Handle cases where plotting is not possible or not needed
        if not can_create_plot:
            if self.plot_surface_cache is not None:
                logger.info(
                    "[Plotter] Target size too small, clearing cache and figure."
                )
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None  # Return None if area is too small

        if not has_data:
            if self.plot_surface_cache is not None:
                logger.info(
                    "[Plotter] No plot data available, clearing cache and figure."
                )
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None  # Return None if there's no data

        # --- Update or Reinitialize ---
        cache_status = "HIT"
        try:
            if needs_reinit:
                cache_status = "MISS (Re-init)"
                logger.info(
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
                        self.plot_surface_cache = None  # Update failed
                else:
                    self.plot_surface_cache = None  # Init failed

            elif needs_update:
                cache_status = (
                    f"MISS (Update - Data: {data_changed}, Time: {time_elapsed})"
                )
                logger.info(f"[Plotter] {cache_status}")
                if self._update_plot_data(plot_data):  # Update existing figure data
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
                # Cache is None, but figure might exist and data hasn't changed recently
                # This can happen if the first render failed
                cache_status = "MISS (Cache None)"
                logger.info(f"[Plotter] {cache_status}, attempting re-render.")
                if self._update_plot_data(plot_data):  # Try updating data again
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = (
                        current_time  # Update time even on re-render
                    )
                    self.last_data_hash = current_data_hash

            # Log cache status for debugging performance
            # logger.info(f"[Plotter] Cache status: {cache_status}")

        except Exception as e:
            logger.error(
                f"[Plotter] Unexpected error in get_cached_or_updated_plot: {e}",
                exc_info=True,
            )
            self.plot_surface_cache = None
            if self.fig:
                plt.close(self.fig)  # Clean up figure on error
            self.fig, self.axes = None, None
            self.last_target_size = (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        """Ensure Matplotlib figure is closed when the plotter is garbage collected."""
        if self.fig:
            try:
                plt.close(self.fig)
                logger.info("[Plotter] Matplotlib figure closed in destructor.")
            except Exception as e:
                # Log error but don't crash during GC
                logger.error(f"[Plotter] Error closing figure in destructor: {e}")
