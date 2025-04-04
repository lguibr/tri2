# File: ui/plotter.py
import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
from collections import deque
import matplotlib
import time
import warnings

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from config import VisConfig, BufferConfig, StatsConfig, DQNConfig

# Configure Matplotlib style
plt.style.use("dark_background")
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
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


def normalize_color_for_matplotlib(
    color_tuple_0_255: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Converts a 0-255 RGB tuple to a 0.0-1.0 RGB tuple."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        return tuple(c / 255.0 for c in color_tuple_0_255)
    else:
        return (0.0, 0.0, 0.0)


class Plotter:
    """Handles creating Pygame surfaces from Matplotlib plots."""

    def __init__(self):
        self.plot_surface: Optional[pygame.Surface] = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = 1.0
        self.rolling_window_size = (
            StatsConfig.STATS_AVG_WINDOW
        )  # Window for the average LINE
        self.default_line_width = 1.5
        self.avg_line_width = 2.0
        self.avg_line_alpha = 0.7

    def create_plot_surface(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Creates a Pygame surface containing Matplotlib plots (3x3 layout)."""
        if target_width <= 10 or target_height <= 10 or not plot_data:
            return None

        scores = list(plot_data.get("episode_scores", deque()))
        game_scores = list(plot_data.get("game_scores", deque()))
        losses = list(plot_data.get("losses", deque()))
        ep_lengths = list(plot_data.get("episode_lengths", deque()))
        sps_values = list(plot_data.get("sps_values", deque()))
        best_game_score_history = list(
            plot_data.get("best_game_score_history", deque())
        )
        lr_values = list(plot_data.get("lr_values", deque()))
        buffer_sizes = list(plot_data.get("buffer_sizes", deque()))
        beta_values = list(plot_data.get("beta_values", deque()))

        has_any_data = any(
            d
            for d in [
                scores,
                game_scores,
                losses,
                ep_lengths,
                sps_values,
                best_game_score_history,
                lr_values,
                buffer_sizes,
                beta_values,
            ]
        )
        if not has_any_data:
            return None

        fig = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)

                dpi = 85
                fig_width_in = target_width / dpi
                fig_height_in = target_height / dpi

                fig, axes = plt.subplots(
                    3, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
                )
                fig.subplots_adjust(
                    hspace=0.65,
                    wspace=0.4,
                    left=0.12,
                    right=0.97,
                    bottom=0.15,
                    top=0.92,
                )
                axes_flat = axes.flatten()

                c_rl_score = normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[0])
                c_game_score = normalize_color_for_matplotlib(
                    VisConfig.GOOGLE_COLORS[1]
                )
                c_loss = normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[3])
                c_len = normalize_color_for_matplotlib(VisConfig.BLUE)
                c_sps = normalize_color_for_matplotlib(VisConfig.LIGHTG)
                c_best_game = normalize_color_for_matplotlib((255, 165, 0))
                c_lr = normalize_color_for_matplotlib((255, 0, 255))
                c_buffer = normalize_color_for_matplotlib(VisConfig.RED)
                c_beta = normalize_color_for_matplotlib((100, 100, 255))
                c_avg = normalize_color_for_matplotlib(VisConfig.YELLOW)

                # --- MODIFIED: X-axis label ---
                # Label reflects the number of points *shown* on the plot,
                # which is controlled by StatsConfig.STATS_AVG_WINDOW
                plot_window_label = f"Plot Window (~{self.rolling_window_size} points)"
                # --- END MODIFIED ---

                self._plot_data_list(
                    axes_flat[0],
                    scores,
                    "RL Score",
                    c_rl_score,
                    c_avg,
                    xlabel=plot_window_label,
                )
                self._plot_data_list(
                    axes_flat[1],
                    game_scores,
                    "Game Score",
                    c_game_score,
                    c_avg,
                    xlabel=plot_window_label,
                )
                self._plot_data_list(
                    axes_flat[2],
                    losses,
                    "Loss",
                    c_loss,
                    c_avg,
                    xlabel=plot_window_label,
                    show_placeholder=True,
                    placeholder_text="Loss data after Learn Start",
                )
                self._plot_data_list(
                    axes_flat[3],
                    ep_lengths,
                    "Ep Length",
                    c_len,
                    c_avg,
                    xlabel=plot_window_label,
                )
                self._plot_data_list(
                    axes_flat[4],
                    best_game_score_history,
                    "Best Game Score",
                    c_best_game,
                    None,
                    xlabel=plot_window_label,
                )
                self._plot_data_list(
                    axes_flat[5],
                    sps_values,
                    "Steps/Sec",
                    c_sps,
                    c_avg,
                    xlabel=plot_window_label,
                )
                self._plot_data_list(
                    axes_flat[6],
                    lr_values,
                    "Learning Rate",
                    c_lr,
                    None,
                    xlabel=plot_window_label,
                    y_log_scale=True,
                )

                buffer_fill_percent = [
                    (s / max(1, BufferConfig.REPLAY_BUFFER_SIZE) * 100)
                    for s in buffer_sizes
                ]
                self._plot_data_list(
                    axes_flat[7],
                    buffer_fill_percent,
                    "Buffer Fill %",
                    c_buffer,
                    c_avg,
                    xlabel=plot_window_label,
                )

                if BufferConfig.USE_PER:
                    self._plot_data_list(
                        axes_flat[8],
                        beta_values,
                        "PER Beta",
                        c_beta,
                        c_avg,
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
                        color=normalize_color_for_matplotlib(VisConfig.GRAY),
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
        xlabel: Optional[str] = None,  # Use the passed xlabel
        show_placeholder: bool = True,
        placeholder_text: Optional[str] = None,
        y_log_scale: bool = False,
    ):
        """Plots the list data and optionally its rolling average."""
        n_points = len(data)
        latest_val_str = ""
        window_size = self.rolling_window_size  # Use for average calculation

        # Calculate latest value/average for title
        if data:
            current_val = data[-1]
            if n_points >= window_size and avg_color is not None:
                try:
                    latest_avg = np.mean(data[-window_size:])
                    latest_val_str = f" (Now: {current_val:.3g}, Avg: {latest_avg:.3g})"
                except Exception:
                    latest_val_str = f" (Now: {current_val:.3g})"
            else:
                latest_val_str = f" (Now: {current_val:.3g})"
        ax.set_title(
            f"{label}{latest_val_str}", fontsize=plt.rcParams["axes.titlesize"]
        )

        # Handle empty data
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
            # Plot raw data
            x_coords = np.arange(n_points)
            ax.plot(
                x_coords,
                data,
                color=color,
                linewidth=self.default_line_width,
                label=f"{label} (Raw)",
            )

            # Plot rolling average
            if avg_color is not None and n_points >= window_size:
                weights = np.ones(window_size) / window_size
                rolling_avg = np.convolve(data, weights, mode="valid")
                avg_x_coords = np.arange(window_size - 1, n_points)
                ax.plot(
                    avg_x_coords,
                    rolling_avg,
                    color=avg_color,
                    linewidth=self.avg_line_width,
                    alpha=self.avg_line_alpha,
                    label=f"{label} (Avg {window_size})",
                )

            # Styling
            ax.tick_params(axis="both", which="major")
            if xlabel:
                ax.set_xlabel(xlabel)  # Use the generated label here
            ax.grid(
                True,
                linestyle=plt.rcParams["grid.linestyle"],
                alpha=plt.rcParams["grid.alpha"],
            )

            # Set Y limits
            min_val = np.min(data)
            max_val = np.max(data)
            padding = (max_val - min_val) * 0.1 if max_val > min_val else 1.0
            padding = max(padding, 1e-6)
            ax.set_ylim(min_val - padding, max_val + padding)

            if y_log_scale and min_val > 0:
                ax.set_yscale("log")
                ax.set_ylim(bottom=max(min_val * 0.9, 1e-9))
            else:
                ax.set_yscale("linear")

            # Set X limits
            if n_points > 1:
                ax.set_xlim(-0.02 * n_points, n_points - 1 + 0.02 * n_points)
            elif n_points == 1:
                ax.set_xlim(-0.5, 0.5)

            if n_points > 10:
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

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
        """Returns the cached plot surface or generates a new one if needed."""
        current_time = time.time()
        has_data = any(plot_data.get(key) for key in plot_data)

        should_update_time = (
            current_time - self.last_plot_update_time > self.plot_update_interval
        )
        size_changed = self.plot_surface and self.plot_surface.get_size() != (
            target_width,
            target_height,
        )
        first_data_received = has_data and self.plot_surface is None
        can_create_plot = target_width > 50 and target_height > 50

        if can_create_plot and (
            should_update_time or size_changed or first_data_received
        ):
            new_plot_surface = self.create_plot_surface(
                plot_data, target_width, target_height
            )
            if new_plot_surface:
                self.plot_surface = new_plot_surface
                self.last_plot_update_time = current_time

        return self.plot_surface
