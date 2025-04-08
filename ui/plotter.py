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
        # Significantly increase update interval to reduce overhead
        self.plot_update_interval: float = 2.5  # Increased from 2.0
        self.rolling_window_sizes = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW
        self.colors = self._init_colors()

        # --- Matplotlib Caching ---
        self.fig: Optional[plt.Figure] = None
        self.axes: Optional[np.ndarray] = None
        self.last_target_size: Tuple[int, int] = (0, 0)
        self.last_data_hash: Optional[int] = None

        logger.info(
            f"[Plotter] Initialized with update interval: {self.plot_update_interval}s"
        )

    def _init_colors(self) -> Dict[str, Tuple[float, float, float]]:
        """Initializes plot colors."""
        # Use distinct colors
        return {
            "game_score": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[0]
            ),  # Green
            "policy_loss": normalize_color_for_matplotlib(VisConfig.RED),
            "value_loss": normalize_color_for_matplotlib(VisConfig.ORANGE),
            "episode_lengths": normalize_color_for_matplotlib(VisConfig.BLUE),
            "episode_outcomes": normalize_color_for_matplotlib(VisConfig.YELLOW),
            "tris_cleared": normalize_color_for_matplotlib(VisConfig.CYAN),
            "lr": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[2]),  # Blueish
            "buffer": normalize_color_for_matplotlib(VisConfig.PURPLE),
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
        }

    def _init_figure(self, target_width: int, target_height: int):
        """Initializes the Matplotlib figure and axes."""
        logger.debug(
            f"[Plotter] Initializing Matplotlib figure for size {target_width}x{target_height}"
        )
        if self.fig:
            plt.close(self.fig)  # Ensure old figure is closed

        dpi = 90  # Keep DPI reasonable
        fig_width_in = max(1, target_width / dpi)
        fig_height_in = max(1, target_height / dpi)

        try:
            self.fig, self.axes = plt.subplots(
                4, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
            )
            # Adjust subplot parameters for tighter layout
            self.fig.subplots_adjust(
                hspace=0.25,  # Increased vertical space slightly
                wspace=0.15,  # Increased horizontal space slightly
                left=0.06,  # Reduced left margin
                right=0.98,  # Kept right margin
                bottom=0.06,  # Reduced bottom margin
                top=0.96,  # Reduced top margin
            )
            self.last_target_size = (target_width, target_height)
            logger.debug("[Plotter] Matplotlib figure initialized.")
        except Exception as e:
            logger.error(f"Error creating Matplotlib figure: {e}", exc_info=True)
            self.fig = None
            self.axes = None
            self.last_target_size = (0, 0)

    def _get_data_hash(self, plot_data: Dict[str, Deque]) -> int:
        """Generates a simple hash based on data lengths and last elements."""
        hash_val = 0
        # Use sorted keys for consistent hash order
        for key in sorted(plot_data.keys()):
            dq = plot_data[key]
            hash_val ^= hash(key)
            hash_val ^= len(dq)
            if dq:
                try:
                    # Hash the last element for change detection
                    last_elem = dq[-1]
                    if isinstance(last_elem, (int, float)):
                        # Use a stable representation for floats
                        hash_val ^= hash(f"{last_elem:.6f}")
                    else:
                        hash_val ^= hash(str(last_elem))  # Fallback to string hash
                except IndexError:
                    pass  # deque might be empty
        return hash_val

    def _update_plot_data(self, plot_data: Dict[str, Deque]):
        """Updates the data on the existing Matplotlib axes."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()
            # --- Define Plot Order ---
            # Row 1: Core Performance
            # Row 2: Losses
            # Row 3: Game Details
            # Row 4: System/Training Details
            plot_defs = [
                # --- Row 1 ---
                ("game_scores", "Game Score", self.colors["game_score"], False),
                (
                    "episode_outcomes",
                    "Episode Outcome",
                    self.colors["episode_outcomes"],
                    False,
                ),
                ("episode_lengths", "Ep Length", self.colors["episode_lengths"], False),
                # --- Row 2 ---
                (
                    "policy_losses",
                    "Policy Loss",
                    self.colors["policy_loss"],
                    True,
                ),  # Log scale
                (
                    "value_losses",
                    "Value Loss",
                    self.colors["value_loss"],
                    True,
                ),  # Log scale
                (
                    "episode_triangles_cleared",
                    "Tris Cleared / Ep",
                    self.colors["tris_cleared"],
                    False,
                ),
                # --- Row 3 ---
                ("lr_values", "Learning Rate", self.colors["lr"], True),  # Log scale
                ("buffer_sizes", "Buffer Size", self.colors["buffer"], False),
                (
                    "best_game_score_history",
                    "Best Score History",
                    self.colors["game_score"],
                    False,
                ),  # Re-use color
                # --- Row 4 (Placeholders) ---
                ("placeholder1", "Future Plot 1", self.colors["placeholder"], False),
                ("placeholder2", "Future Plot 2", self.colors["placeholder"], False),
                ("placeholder3", "Future Plot 3", self.colors["placeholder"], False),
            ]

            data_lists = {
                key: list(plot_data.get(key, deque())) for key, _, _, _ in plot_defs
            }

            for i, (data_key, label, color, log_scale) in enumerate(plot_defs):
                if i >= len(axes_flat):
                    break  # Avoid index error if grid changes
                ax = axes_flat[i]
                ax.clear()  # Clear previous plot elements efficiently
                render_single_plot(
                    ax,
                    data_lists[data_key],
                    label,
                    color,
                    self.rolling_window_sizes,
                    placeholder_text=label,
                    y_log_scale=log_scale,
                )
                # Configure axes appearance after plotting
                if i < 9:  # Hide x-labels for top rows
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=0)  # Ensure x-ticks aren't rotated

            plot_update_duration = time.monotonic() - plot_update_start
            logger.debug(f"[Plotter] Plot data updated in {plot_update_duration:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error updating plot data: {e}", exc_info=True)
            # Attempt to clear axes to prevent stale display on error
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
            # Draw the canvas without recalculating layout
            self.fig.canvas.draw()

            # Render to buffer
            buf = BytesIO()
            self.fig.savefig(
                buf,
                format="png",
                transparent=False,
                facecolor=plt.rcParams["figure.facecolor"],
            )
            buf.seek(0)
            plot_img_surface = pygame.image.load(
                buf, "png"
            ).convert()  # Use convert for performance
            buf.close()

            # Resize if necessary (only if target size differs from figure size)
            # Note: Figure size might slightly differ due to DPI and subplot adjustments
            current_size = plot_img_surface.get_size()
            if current_size != (target_width, target_height):
                # Use smoothscale for better quality if sizes differ significantly
                scale_diff = abs(current_size[0] - target_width) + abs(
                    current_size[1] - target_height
                )
                if scale_diff > 10:  # Threshold to use smoothscale
                    plot_img_surface = pygame.transform.smoothscale(
                        plot_img_surface, (target_width, target_height)
                    )
                else:  # Use faster scale for minor adjustments
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

        # Conditions for full re-initialization
        needs_reinit = (
            self.fig is None
            or self.axes is None
            or self.last_target_size != target_size
        )

        # Conditions for data update and re-render
        current_data_hash = self._get_data_hash(plot_data)
        data_changed = self.last_data_hash != current_data_hash
        time_elapsed = (
            current_time - self.last_plot_update_time > self.plot_update_interval
        )
        needs_update = data_changed or time_elapsed

        # Check if plotting is feasible
        can_create_plot = target_width > 50 and target_height > 50

        if not can_create_plot:
            if self.plot_surface_cache is not None:
                logger.debug("[Plotter] Target size too small, clearing cache.")
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)  # Close figure if clearing cache
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None  # Cannot plot if area is too small

        if not has_data:
            if self.plot_surface_cache is not None:
                logger.debug("[Plotter] No data, clearing cache.")
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)  # Close figure if clearing cache
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None  # Cannot plot if no data

        # --- Logic ---
        cache_status = "HIT"  # Assume cache hit initially
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
                    else:  # Update failed after re-init
                        self.plot_surface_cache = None
                else:  # Figure init failed
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
                else:  # Update failed, keep old cache but log error
                    logger.warning(
                        "[Plotter] Plot data update failed, returning potentially stale cache."
                    )
                    cache_status = "ERROR (Update Failed)"

            # Else: No need to update, return existing cache
            elif self.plot_surface_cache is None:
                # Edge case: cache is None but shouldn't be (e.g., after error)
                cache_status = "MISS (Cache None)"
                logger.debug(f"[Plotter] {cache_status}, attempting re-render.")
                if self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash

            # Log cache status only if it wasn't a hit
            if cache_status != "HIT":
                logger.info(f"[Plotter] Cache Status: {cache_status}")

        except Exception as e:
            logger.error(
                f"[Plotter] Unexpected error in get_cached_or_updated_plot: {e}",
                exc_info=True,
            )
            self.plot_surface_cache = None  # Clear cache on major error
            if self.fig:
                plt.close(self.fig)
            self.fig, self.axes = None, None
            self.last_target_size = (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        # Ensure Matplotlib figure is closed when Plotter is garbage collected
        if self.fig:
            try:
                plt.close(self.fig)
                logger.debug("[Plotter] Matplotlib figure closed in destructor.")
            except Exception as e:
                logger.error(f"[Plotter] Error closing figure in destructor: {e}")
