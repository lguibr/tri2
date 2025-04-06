import pygame
from typing import Dict, Optional, Deque
from collections import deque
import matplotlib
import time
import warnings
from io import BytesIO
import traceback

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, StatsConfig
from .plot_utils import (
    render_single_plot,
    normalize_color_for_matplotlib,
)


class Plotter:
    """Handles creation and caching of the multi-plot Matplotlib surface."""

    def __init__(self):
        self.plot_surface: Optional[pygame.Surface] = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = 1.0
        self.rolling_window_sizes = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW

        self.colors = {
            "rl_score": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[0]),
            "game_score": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[1]),
            "policy_loss": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[3]),
            "value_loss": normalize_color_for_matplotlib(VisConfig.BLUE),
            "entropy": normalize_color_for_matplotlib((150, 150, 150)),
            "len": normalize_color_for_matplotlib(VisConfig.BLUE),
            "sps": normalize_color_for_matplotlib(VisConfig.LIGHTG),
            "best_game": normalize_color_for_matplotlib((255, 165, 0)),
            "lr": normalize_color_for_matplotlib((255, 0, 255)),
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
        }

    def create_plot_surface(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:

        if target_width <= 10 or target_height <= 10 or not plot_data:
            return None

        data_keys = [
            "game_scores",
            "best_game_score_history",
            "episode_scores",
            "episode_lengths",
            "sps_values",
            "lr_values",
            "value_loss",
            "policy_loss",
            "entropy",
        ]
        data_lists = {key: list(plot_data.get(key, deque())) for key in data_keys}

        has_any_data = any(len(d) > 0 for d in data_lists.values())
        if not has_any_data:
            return None

        fig = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                dpi = 90
                fig_width_in = max(1, target_width / dpi)
                fig_height_in = max(1, target_height / dpi)

                fig, axes = plt.subplots(
                    3, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
                )
                fig.subplots_adjust(
                    hspace=0.30,
                    wspace=0.15,
                    left=0.08,
                    right=0.98,
                    bottom=0.10,
                    top=0.92,
                )
                axes_flat = axes.flatten()

                max_len = max((len(d) for d in data_lists.values() if d), default=0)
                plot_window_label = (
                    f"Latest {min(self.plot_data_window, max_len)} Updates"
                )

                # Row 1
                render_single_plot(
                    axes_flat[0],
                    data_lists["game_scores"],
                    "Game Score",
                    self.colors["game_score"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Game Score",
                )
                render_single_plot(
                    axes_flat[1],
                    data_lists["best_game_score_history"],
                    "Best Game Score",
                    self.colors["best_game"],
                    [],  # No rolling average for best score history
                    xlabel=plot_window_label,
                    placeholder_text="Best Game Score",
                )
                render_single_plot(
                    axes_flat[2],
                    data_lists["episode_scores"],
                    "RL Score",
                    self.colors["rl_score"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="RL Score",
                )

                # Row 2
                render_single_plot(
                    axes_flat[3],
                    data_lists["episode_lengths"],
                    "Ep Length",
                    self.colors["len"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Episode Length",
                )
                render_single_plot(
                    axes_flat[4],
                    data_lists["sps_values"],
                    "Steps/Sec",
                    self.colors["sps"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="SPS",
                )
                render_single_plot(
                    axes_flat[5],
                    data_lists["lr_values"],
                    "Learning Rate",
                    self.colors["lr"],
                    [],  # No rolling average for LR
                    xlabel=plot_window_label,
                    y_log_scale=True,
                    placeholder_text="Learning Rate",
                )

                # Row 3
                render_single_plot(
                    axes_flat[6],
                    data_lists["value_loss"],
                    "Value Loss",
                    self.colors["value_loss"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Value Loss",
                )
                render_single_plot(
                    axes_flat[7],
                    data_lists["policy_loss"],
                    "Policy Loss",
                    self.colors["policy_loss"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Policy Loss",
                )
                render_single_plot(
                    axes_flat[8],
                    data_lists["entropy"],
                    "Entropy",
                    self.colors["entropy"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Entropy",
                )

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

                current_size = plot_img_surface.get_size()
                if current_size != (target_width, target_height):
                    plot_img_surface = pygame.transform.smoothscale(
                        plot_img_surface, (target_width, target_height)
                    )

                return plot_img_surface

        except Exception as e:
            print(f"Error creating plot surface: {e}")
            traceback.print_exc()
            return None
        finally:
            if fig is not None:
                plt.close(fig)

    def get_cached_or_updated_plot(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        current_time = time.time()
        has_data = any(d for d in plot_data.values())
        needs_update_time = (
            current_time - self.last_plot_update_time > self.plot_update_interval
        )
        size_changed = self.plot_surface and self.plot_surface.get_size() != (
            target_width,
            target_height,
        )
        first_plot_needed = has_data and self.plot_surface is None
        can_create_plot = target_width > 50 and target_height > 50

        if can_create_plot and (needs_update_time or size_changed or first_plot_needed):
            if has_data:
                new_plot_surface = self.create_plot_surface(
                    plot_data, target_width, target_height
                )
                # Only update cache if plot creation was successful
                if new_plot_surface:
                    self.plot_surface = new_plot_surface
                # If creation failed (e.g., due to error), keep the old cached plot (if any)
                # else: self.plot_surface = None # Optionally clear cache on failure
                self.last_plot_update_time = current_time
            elif not has_data:  # No data at all, clear cache
                self.plot_surface = None

        return self.plot_surface
