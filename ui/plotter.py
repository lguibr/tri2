# File: ui/plotter.py
import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
from collections import deque
import matplotlib
import time
import warnings
from io import BytesIO

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, BufferConfig, StatsConfig, DQNConfig

# --- Matplotlib Style Configuration ---
try:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "font.size": 8,  # Slightly smaller default font
            "axes.labelsize": 8,
            "axes.titlesize": 9,  # Slightly smaller title
            "xtick.labelsize": 7,  # Smaller ticks
            "ytick.labelsize": 7,
            "legend.fontsize": 6,  # Smaller legend
            "figure.facecolor": "#262626",
            "axes.facecolor": "#303030",
            "axes.edgecolor": "#707070",
            "axes.labelcolor": "#D0D0D0",
            "xtick.color": "#C0C0C0",
            "ytick.color": "#C0C0C0",
            "grid.color": "#505050",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
        }
    )
except Exception as e:
    print(f"Warning: Failed to set Matplotlib style: {e}")
# --- End Style Configuration ---


def normalize_color_for_matplotlib(
    color_tuple_0_255: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        return tuple(c / 255.0 for c in color_tuple_0_255)
    else:
        print(f"Warning: Invalid color tuple {color_tuple_0_255}, returning black.")
        return (0.0, 0.0, 0.0)


class Plotter:
    def __init__(self):
        self.plot_surface: Optional[pygame.Surface] = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = 1.0
        self.rolling_window_size = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW  # Max points to plot
        self.default_line_width = 1.0  # Thinner default line
        self.avg_line_width = 1.5  # Thinner avg line
        self.avg_line_alpha = 0.7

    def create_plot_surface(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        if target_width <= 10 or target_height <= 10 or not plot_data:
            return None

        data_lists = {
            key: list(plot_data.get(key, deque()))
            for key in [
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
        }

        if not any(data_lists.values()):
            return None

        fig = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)

                # --- MODIFIED: Slightly higher DPI ---
                dpi = 90
                fig_width_in = target_width / dpi
                fig_height_in = target_height / dpi

                fig, axes = plt.subplots(
                    3, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
                )
                # --- MODIFIED: Tighter subplot adjustments ---
                fig.subplots_adjust(
                    hspace=0.55,  # Reduced vertical space
                    wspace=0.35,  # Reduced horizontal space
                    left=0.10,  # Reduced left margin
                    right=0.98,  # Reduced right margin
                    bottom=0.12,  # Reduced bottom margin
                    top=0.95,  # Reduced top margin
                )
                # --- END MODIFIED ---
                axes_flat = axes.flatten()

                colors = {
                    "rl_score": normalize_color_for_matplotlib(
                        VisConfig.GOOGLE_COLORS[0]
                    ),
                    "game_score": normalize_color_for_matplotlib(
                        VisConfig.GOOGLE_COLORS[1]
                    ),
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

                # --- MODIFIED: X-axis label reflects plot window ---
                plot_window_label = f"Latest {min(self.plot_data_window, max(len(d) for d in data_lists.values() if d))} Points"
                # --- END MODIFIED ---

                self._plot_data_list(
                    axes_flat[0],
                    data_lists["episode_scores"],
                    "RL Score",
                    colors["rl_score"],
                    colors["avg"],
                    xlabel=plot_window_label,
                )
                self._plot_data_list(
                    axes_flat[1],
                    data_lists["game_scores"],
                    "Game Score",
                    colors["game_score"],
                    colors["avg"],
                    xlabel=plot_window_label,
                )
                self._plot_data_list(
                    axes_flat[2],
                    data_lists["losses"],
                    "Loss",
                    colors["loss"],
                    colors["avg"],
                    xlabel=plot_window_label,
                    show_placeholder=True,
                    placeholder_text="Loss data after Learn Start",
                )
                self._plot_data_list(
                    axes_flat[3],
                    data_lists["episode_lengths"],
                    "Ep Length",
                    colors["len"],
                    colors["avg"],
                    xlabel=plot_window_label,
                )
                self._plot_data_list(
                    axes_flat[4],
                    data_lists["best_game_score_history"],
                    "Best Game Score",
                    colors["best_game"],
                    None,
                    xlabel=plot_window_label,
                )
                self._plot_data_list(
                    axes_flat[5],
                    data_lists["sps_values"],
                    "Steps/Sec",
                    colors["sps"],
                    colors["avg"],
                    xlabel=plot_window_label,
                )
                self._plot_data_list(
                    axes_flat[6],
                    data_lists["lr_values"],
                    "Learning Rate",
                    colors["lr"],
                    None,
                    xlabel=plot_window_label,
                    y_log_scale=True,
                )

                buffer_fill_percent = [
                    (s / max(1, BufferConfig.REPLAY_BUFFER_SIZE) * 100)
                    for s in data_lists["buffer_sizes"]
                ]
                self._plot_data_list(
                    axes_flat[7],
                    buffer_fill_percent,
                    "Buffer Fill %",
                    colors["buffer"],
                    colors["avg"],
                    xlabel=plot_window_label,
                )

                if BufferConfig.USE_PER:
                    self._plot_data_list(
                        axes_flat[8],
                        data_lists["beta_values"],
                        "PER Beta",
                        colors["beta"],
                        colors["avg"],
                        xlabel=plot_window_label,
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
                        color=colors["placeholder"],
                    )
                    axes_flat[8].set_yticks([])
                    axes_flat[8].set_xticks([])

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

    def _plot_data_list(
        self,
        ax,
        data: List[Union[float, int]],
        label: str,
        color,
        avg_color: Optional[Tuple[float, float, float]],
        xlabel: Optional[str] = None,
        show_placeholder: bool = True,
        placeholder_text: Optional[str] = None,
        y_log_scale: bool = False,
    ):
        n_points = len(data)
        latest_val_str = ""
        avg_window = self.rolling_window_size  # Window for rolling average line

        if data:
            current_val = data[-1]
            if n_points >= avg_window and avg_color is not None:
                try:
                    latest_avg = np.mean(data[-avg_window:])
                    latest_val_str = (
                        f" (Now: {current_val:.3g}, Avg{avg_window}: {latest_avg:.3g})"
                    )
                except Exception:
                    latest_val_str = f" (Now: {current_val:.3g})"
            else:
                latest_val_str = f" (Now: {current_val:.3g})"
        ax.set_title(
            f"{label}{latest_val_str}", fontsize=plt.rcParams["axes.titlesize"]
        )

        placeholder_text_color = normalize_color_for_matplotlib(VisConfig.GRAY)
        if n_points == 0:
            if show_placeholder:
                p_text = (
                    placeholder_text if placeholder_text else f"{label}\n(No data yet)"
                )
                ax.text(
                    0.5,
                    0.5,
                    p_text,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                    color=placeholder_text_color,
                )
            ax.set_yticks([])
            ax.set_xticks([])
            return

        try:
            x_coords = np.arange(n_points)
            ax.plot(
                x_coords,
                data,
                color=color,
                linewidth=self.default_line_width,
                label=f"{label} (Raw)",
            )

            if avg_color is not None and n_points >= avg_window:
                weights = np.ones(avg_window) / avg_window
                rolling_avg = np.convolve(data, weights, mode="valid")
                avg_x_coords = np.arange(avg_window - 1, n_points)
                ax.plot(
                    avg_x_coords,
                    rolling_avg,
                    color=avg_color,
                    linewidth=self.avg_line_width,
                    alpha=self.avg_line_alpha,
                    label=f"{label} (Avg {avg_window})",
                )

            ax.tick_params(axis="both", which="major")
            if xlabel:
                ax.set_xlabel(xlabel)
            ax.grid(
                True,
                linestyle=plt.rcParams["grid.linestyle"],
                alpha=plt.rcParams["grid.alpha"],
            )

            min_val = np.min(data)
            max_val = np.max(data)
            padding = (
                (max_val - min_val) * 0.1
                if max_val > min_val
                else max(abs(max_val * 0.1), 1.0)
            )
            padding = max(padding, 1e-6)
            ax.set_ylim(min_val - padding, max_val + padding)

            if y_log_scale and min_val > 1e-9:
                ax.set_yscale("log")
                ax.set_ylim(bottom=max(min_val * 0.9, 1e-9))
            else:
                ax.set_yscale("linear")

            if n_points > 1:
                ax.set_xlim(-0.02 * n_points, n_points - 1 + 0.02 * n_points)
            elif n_points == 1:
                ax.set_xlim(-0.5, 0.5)

            # --- MODIFIED: Adjust x-ticks based on number of points ---
            if n_points > 1000:
                ax.xaxis.set_major_locator(
                    plt.MaxNLocator(integer=True, nbins=4)
                )  # Fewer ticks for many points

                # Optionally format as 'k' or 'M'
                def format_func(value, tick_number):
                    if value >= 1_000_000:
                        return f"{value/1_000_000:.1f}M"
                    if value >= 1_000:
                        return f"{value/1_000:.0f}k"
                    return f"{int(value)}"

                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            elif n_points > 10:
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
            # --- END MODIFIED ---

        except Exception as plot_err:
            print(f"ERROR during _plot_data_list for '{label}': {plot_err}")
            error_text_color = normalize_color_for_matplotlib(VisConfig.RED)
            ax.text(
                0.5,
                0.5,
                f"Plotting Error\n({label})",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color=error_text_color,
            )
            ax.set_yticks([])
            ax.set_xticks([])

    def get_cached_or_updated_plot(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
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
