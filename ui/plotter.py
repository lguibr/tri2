# File: ui/plotter.py
import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
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
        self.plot_update_interval: float = 0.5
        self.rolling_window_sizes = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW
        self.default_line_width = 1.0
        self.avg_line_width = 1.5
        self.avg_line_alpha = 0.8

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
            "avg_primary": normalize_color_for_matplotlib(VisConfig.YELLOW),
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
        }
        self.avg_line_colors_secondary = [
            normalize_color_for_matplotlib((0, 255, 255)),
            normalize_color_for_matplotlib((255, 165, 0)),
            normalize_color_for_matplotlib((0, 255, 0)),
            normalize_color_for_matplotlib((255, 0, 255)),
        ]

    def create_plot_surface(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:

        if target_width <= 10 or target_height <= 10 or not plot_data:
            return None

        data_keys = [
            "episode_scores",
            "game_scores",
            "policy_loss",
            "value_loss",
            "entropy",
            "episode_lengths",
            "sps_values",
            "best_game_score_history",
            "lr_values",
        ]
        data_lists = {key: list(plot_data.get(key, deque())) for key in data_keys}

        # --- DEBUG LOGGING ---
        # print(f"[Plotter Debug] Data lengths: "
        #       f"RLScore={len(data_lists['episode_scores'])}, GameScore={len(data_lists['game_scores'])}, EpLen={len(data_lists['episode_lengths'])}, "
        #       f"PLoss={len(data_lists['policy_loss'])}, VLoss={len(data_lists['value_loss'])}, Ent={len(data_lists['entropy'])}")
        # --- END DEBUG LOGGING ---

        if not any(len(d) > 0 for d in data_lists.values()):
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
                    hspace=0.55,
                    wspace=0.35,
                    left=0.10,
                    right=0.98,
                    bottom=0.12,
                    top=0.95,
                )
                axes_flat = axes.flatten()

                max_len = max((len(d) for d in data_lists.values() if d), default=0)
                plot_window_label = (
                    f"Latest {min(self.plot_data_window, max_len)} Updates"
                )

                render_single_plot(
                    axes_flat[0],
                    data_lists["episode_scores"],
                    "RL Score",
                    self.colors["rl_score"],
                    self.colors["avg_primary"],
                    self.avg_line_colors_secondary,
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="RL Score",
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[1],
                    data_lists["game_scores"],
                    "Game Score",
                    self.colors["game_score"],
                    self.colors["avg_primary"],
                    self.avg_line_colors_secondary,
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Game Score",
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[2],
                    data_lists["policy_loss"],
                    "Policy Loss",
                    self.colors["policy_loss"],
                    self.colors["avg_primary"],
                    self.avg_line_colors_secondary,
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Policy Loss",
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[3],
                    data_lists["value_loss"],
                    "Value Loss",
                    self.colors["value_loss"],
                    self.colors["avg_primary"],
                    self.avg_line_colors_secondary,
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Value Loss",
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[4],
                    data_lists["entropy"],
                    "Entropy",
                    self.colors["entropy"],
                    self.colors["avg_primary"],
                    self.avg_line_colors_secondary,
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Entropy",
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[5],
                    data_lists["episode_lengths"],
                    "Ep Length",
                    self.colors["len"],
                    self.colors["avg_primary"],
                    self.avg_line_colors_secondary,
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Episode Length",
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[6],
                    data_lists["sps_values"],
                    "Steps/Sec",
                    self.colors["sps"],
                    self.colors["avg_primary"],
                    self.avg_line_colors_secondary,
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="SPS",
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[7],
                    data_lists["best_game_score_history"],
                    "Best Game Score",
                    self.colors["best_game"],
                    None,
                    [],
                    [],
                    xlabel=plot_window_label,
                    placeholder_text="Best Game Score",
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
                )
                render_single_plot(
                    axes_flat[8],
                    data_lists["lr_values"],
                    "Learning Rate",
                    self.colors["lr"],
                    None,
                    [],
                    [],
                    xlabel=plot_window_label,
                    y_log_scale=True,
                    placeholder_text="Learning Rate",
                    default_line_width=self.default_line_width,
                    avg_line_width=self.avg_line_width,
                    avg_line_alpha=self.avg_line_alpha,
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
        """Returns the cached plot surface or generates a new one if needed."""
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
            new_plot_surface = self.create_plot_surface(
                plot_data, target_width, target_height
            )
            if new_plot_surface:
                self.plot_surface = new_plot_surface
                self.last_plot_update_time = current_time

        return self.plot_surface
