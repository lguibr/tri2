# File: ui/plotter.py
import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
from collections import deque
import matplotlib
import time
import warnings
from io import BytesIO

# Ensure Agg backend is used before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, BufferConfig, StatsConfig, DQNConfig
from .plot_utils import (
    render_single_plot,
    normalize_color_for_matplotlib,
)  # Import helper


class Plotter:
    """Handles creation and caching of the multi-plot Matplotlib surface."""

    def __init__(self):
        self.plot_surface: Optional[pygame.Surface] = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = 1.0  # Update plot every second
        self.rolling_window_size = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW
        self.default_line_width = 1.0
        self.avg_line_width = 1.5
        self.avg_line_alpha = 0.7

        # Define colors once
        self.colors = {
            "rl_score": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[0]),
            "game_score": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[1]),
            "loss": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[3]),
            "len": normalize_color_for_matplotlib(VisConfig.BLUE),
            "sps": normalize_color_for_matplotlib(VisConfig.LIGHTG),
            "best_game": normalize_color_for_matplotlib((255, 165, 0)),
            "lr": normalize_color_for_matplotlib((255, 0, 255)),
            "buffer": normalize_color_for_matplotlib(VisConfig.RED),
            "beta": normalize_color_for_matplotlib((100, 100, 255)),
            "avg": normalize_color_for_matplotlib(VisConfig.YELLOW),
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
        }

    def create_plot_surface(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Creates the Matplotlib figure and renders subplots."""
        if target_width <= 10 or target_height <= 10 or not plot_data:
            return None

        data_keys = [
            "episode_scores",
            "game_scores",
            "losses",
            "episode_lengths",
            "sps_values",
            "best_game_score_history",
            "lr_values",
            "buffer_sizes",
            "beta_values",
        ]
        data_lists = {key: list(plot_data.get(key, deque())) for key in data_keys}

        if not any(data_lists.values()):
            return None

        fig = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                dpi = 90
                fig_width_in = target_width / dpi
                fig_height_in = target_height / dpi
                fig, axes = plt.subplots(
                    3, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
                )
                fig.subplots_adjust(
                    hspace=0.55,
                    wspace=0.35,
                    left=0.10,
                    right=0.98,
                    bottom=0.12,
                    top=0.95,
                )
                axes_flat = axes.flatten()

                max_len = max((len(d) for d in data_lists.values() if d), default=0)
                plot_window_label = f"Latest {min(self.plot_data_window, max_len)} Pts"

                # --- Render Subplots using Helper ---
                render_single_plot(
                    axes_flat[0],
                    data_lists["episode_scores"],
                    "RL Score",
                    self.colors["rl_score"],
                    self.colors["avg"],
                    self.rolling_window_size,
                    xlabel=plot_window_label,
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[1],
                    data_lists["game_scores"],
                    "Game Score",
                    self.colors["game_score"],
                    self.colors["avg"],
                    self.rolling_window_size,
                    xlabel=plot_window_label,
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[2],
                    data_lists["losses"],
                    "Loss",
                    self.colors["loss"],
                    self.colors["avg"],
                    self.rolling_window_size,
                    xlabel=plot_window_label,
                    show_placeholder=True,
                    placeholder_text="Loss data after Learn Start",
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[3],
                    data_lists["episode_lengths"],
                    "Ep Length",
                    self.colors["len"],
                    self.colors["avg"],
                    self.rolling_window_size,
                    xlabel=plot_window_label,
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[4],
                    data_lists["best_game_score_history"],
                    "Best Game Score",
                    self.colors["best_game"],
                    None,
                    self.rolling_window_size,
                    xlabel=plot_window_label,
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[5],
                    data_lists["sps_values"],
                    "Steps/Sec",
                    self.colors["sps"],
                    self.colors["avg"],
                    self.rolling_window_size,
                    xlabel=plot_window_label,
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[6],
                    data_lists["lr_values"],
                    "Learning Rate",
                    self.colors["lr"],
                    None,
                    self.rolling_window_size,
                    xlabel=plot_window_label,
                    y_log_scale=True,
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                buffer_fill_percent = [
                    (s / max(1, BufferConfig.REPLAY_BUFFER_SIZE) * 100)
                    for s in data_lists["buffer_sizes"]
                ]
                render_single_plot(
                    axes_flat[7],
                    buffer_fill_percent,
                    "Buffer Fill %",
                    self.colors["buffer"],
                    self.colors["avg"],
                    self.rolling_window_size,
                    xlabel=plot_window_label,
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                if BufferConfig.USE_PER:
                    render_single_plot(
                        axes_flat[8],
                        data_lists["beta_values"],
                        "PER Beta",
                        self.colors["beta"],
                        self.colors["avg"],
                        self.rolling_window_size,
                        xlabel=plot_window_label,
                        default_line_width=self.default_line_width,
                        avg_line_width=self.avg_line_width,
                        avg_line_alpha=self.avg_line_alpha,
                    )
                else:
                    axes_flat[8].text(
                        0.5,
                        0.5,
                        "PER Disabled",
                        ha="center",
                        va="center",
                        transform=axes_flat[8].transAxes,
                        fontsize=8,
                        color=self.colors["placeholder"],
                    )
                    axes_flat[8].set_yticks([])
                    axes_flat[8].set_xticks([])
                # --- End Subplot Rendering ---

                for ax in axes_flat:
                    ax.tick_params(axis="x", rotation=0)

                buf = BytesIO()
                fig.savefig(
                    buf,
                    format="png",
                    transparent=False,
                    facecolor=plt.rcParams["figure.facecolor"],
                )
                buf.seek(0)
                plot_img_surface = pygame.image.load(buf).convert()
                buf.close()

                if (
                    plot_img_surface.get_size() != (target_width, target_height)
                    and target_width > 0
                    and target_height > 0
                ):
                    plot_img_surface = pygame.transform.smoothscale(
                        plot_img_surface, (target_width, target_height)
                    )
                return plot_img_surface

        except Exception as e:
            print(f"Error creating plot surface: {e}")
            import traceback

            traceback.print_exc()
            return None
        finally:
            if fig is not None:
                plt.close(fig)

    def get_cached_or_updated_plot(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Returns the cached plot surface or generates a new one if needed."""
        current_time = time.time()
        has_data = any(plot_data.values())
        needs_update_time = (
            current_time - self.last_plot_update_time > self.plot_update_interval
        )
        size_changed = self.plot_surface and self.plot_surface.get_size() != (
            target_width,
            target_height,
        )
        first_data_received = has_data and self.plot_surface is None
        can_create_plot = target_width > 50 and target_height > 50

        if can_create_plot and (
            needs_update_time or size_changed or first_data_received
        ):
            new_plot_surface = self.create_plot_surface(
                plot_data, target_width, target_height
            )
            if new_plot_surface:
                self.plot_surface = new_plot_surface
                self.last_plot_update_time = current_time

        return self.plot_surface
