# File: ui/plotter.py
# (No significant changes needed, this file was already focused)
import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
from collections import deque
import matplotlib
import time
import warnings
from io import BytesIO

# Ensure Matplotlib uses Agg backend for non-interactive plotting
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, BufferConfig, StatsConfig, DQNConfig

# --- Matplotlib Style Configuration ---
try:
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
except Exception as e:
    print(f"Warning: Failed to set Matplotlib style: {e}")
# --- End Style Configuration ---


def normalize_color_for_matplotlib(
    color_tuple_0_255: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Converts a 0-255 RGB tuple to a 0.0-1.0 RGB tuple for Matplotlib."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        return tuple(c / 255.0 for c in color_tuple_0_255)
    else:
        print(f"Warning: Invalid color tuple {color_tuple_0_255}, returning black.")
        return (0.0, 0.0, 0.0)  # Return black on error


class Plotter:
    """Handles creating Pygame surfaces from Matplotlib plots of training data."""

    def __init__(self):
        self.plot_surface: Optional[pygame.Surface] = None  # Cached plot surface
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = 1.0  # Seconds between plot redraws
        self.rolling_window_size = (
            StatsConfig.STATS_AVG_WINDOW
        )  # Window for rolling avg line
        self.default_line_width = 1.5
        self.avg_line_width = 2.0
        self.avg_line_alpha = 0.7

    def create_plot_surface(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Creates a Pygame surface containing Matplotlib plots (3x3 layout)."""
        if target_width <= 10 or target_height <= 10 or not plot_data:
            return None  # Cannot render if area is too small or no data

        # Extract data lists from the dictionary, handle missing keys gracefully
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

        # Check if there's *any* data to plot
        if not any(data_lists.values()):
            return None

        fig = None
        try:
            # Ignore UserWarnings from Matplotlib (e.g., about tight_layout)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)

                dpi = 85  # Adjust DPI based on desired resolution vs performance
                fig_width_in = target_width / dpi
                fig_height_in = target_height / dpi

                fig, axes = plt.subplots(
                    3, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
                )
                # Adjust spacing to prevent labels/titles overlapping
                fig.subplots_adjust(
                    hspace=0.65,
                    wspace=0.4,
                    left=0.12,
                    right=0.97,
                    bottom=0.15,
                    top=0.92,
                )
                axes_flat = axes.flatten()

                # Define colors using VisConfig and normalize for Matplotlib
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
                    "best_game": normalize_color_for_matplotlib(
                        (255, 165, 0)
                    ),  # Orange
                    "lr": normalize_color_for_matplotlib((255, 0, 255)),  # Magenta
                    "buffer": normalize_color_for_matplotlib(VisConfig.RED),
                    "beta": normalize_color_for_matplotlib(
                        (100, 100, 255)
                    ),  # Light Blue
                    "avg": normalize_color_for_matplotlib(VisConfig.YELLOW),
                    "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
                }

                # X-axis label reflects the visible window size
                plot_window_label = f"Plot Window (~{self.rolling_window_size} points)"

                # --- Plot each metric ---
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

                # Convert buffer size to fill percentage
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

                # Plot PER Beta only if PER is enabled
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
                    # Show placeholder if PER is disabled
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

                # General plot styling
                for ax in axes_flat:
                    ax.tick_params(
                        axis="x", rotation=0
                    )  # Ensure x-axis labels are horizontal

                # --- Convert Matplotlib plot to Pygame surface ---
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

                # Scale surface smoothly if generated size doesn't match target (can happen with DPI/figsize)
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
            # Ensure Matplotlib figure is closed to free memory
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
        """Helper function to plot a single list of data onto a Matplotlib axis."""
        n_points = len(data)
        latest_val_str = ""
        window_size = self.rolling_window_size  # Window for rolling average

        # Calculate latest value/average to display in title
        if data:
            current_val = data[-1]
            if n_points >= window_size and avg_color is not None:
                try:  # Handle potential errors if data contains non-numerics briefly
                    latest_avg = np.mean(data[-window_size:])
                    latest_val_str = f" (Now: {current_val:.3g}, Avg: {latest_avg:.3g})"
                except Exception:
                    latest_val_str = f" (Now: {current_val:.3g})"
            else:
                latest_val_str = f" (Now: {current_val:.3g})"
        ax.set_title(
            f"{label}{latest_val_str}", fontsize=plt.rcParams["axes.titlesize"]
        )

        # Handle empty data case
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
            # Plot raw data points
            x_coords = np.arange(n_points)
            ax.plot(
                x_coords,
                data,
                color=color,
                linewidth=self.default_line_width,
                label=f"{label} (Raw)",
            )

            # Plot rolling average if applicable
            if avg_color is not None and n_points >= window_size:
                # Use convolution for efficient rolling average calculation
                weights = np.ones(window_size) / window_size
                rolling_avg = np.convolve(data, weights, mode="valid")
                # Adjust x-coordinates for the rolling average plot
                avg_x_coords = np.arange(window_size - 1, n_points)
                ax.plot(
                    avg_x_coords,
                    rolling_avg,
                    color=avg_color,
                    linewidth=self.avg_line_width,
                    alpha=self.avg_line_alpha,
                    label=f"{label} (Avg {window_size})",
                )

            # --- Styling ---
            ax.tick_params(axis="both", which="major")
            if xlabel:
                ax.set_xlabel(xlabel)
            ax.grid(
                True,
                linestyle=plt.rcParams["grid.linestyle"],
                alpha=plt.rcParams["grid.alpha"],
            )

            # Set Y limits dynamically based on data range
            min_val = np.min(data)
            max_val = np.max(data)
            # Add padding to Y limits, ensuring some padding even if min=max
            padding = (
                (max_val - min_val) * 0.1
                if max_val > min_val
                else max(abs(max_val * 0.1), 1.0)
            )
            padding = max(padding, 1e-6)  # Ensure padding is non-zero
            ax.set_ylim(min_val - padding, max_val + padding)

            # Apply log scale if requested and data is positive
            if y_log_scale and min_val > 1e-9:  # Use small epsilon for > 0 check
                ax.set_yscale("log")
                # Adjust lower y-limit for log scale to avoid issues near zero
                ax.set_ylim(bottom=max(min_val * 0.9, 1e-9))
            else:
                ax.set_yscale("linear")  # Ensure linear scale otherwise

            # Set X limits dynamically
            if n_points > 1:
                ax.set_xlim(-0.02 * n_points, n_points - 1 + 0.02 * n_points)
            elif n_points == 1:
                ax.set_xlim(-0.5, 0.5)  # Handle single point case

            # Adjust number of x-axis ticks for readability
            if n_points > 10:
                ax.xaxis.set_major_locator(
                    plt.MaxNLocator(integer=True, nbins=4)
                )  # Limit tick count

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
        """Returns the cached plot surface or generates a new one if the interval has passed,
        size changed, or data just arrived."""
        current_time = time.time()
        has_data = any(plot_data.values())  # Check if any deque has data

        # Conditions for updating the plot
        needs_update_time = (
            current_time - self.last_plot_update_time > self.plot_update_interval
        )
        size_changed = self.plot_surface and self.plot_surface.get_size() != (
            target_width,
            target_height,
        )
        first_data_received = (
            has_data and self.plot_surface is None
        )  # Update when first data comes in
        can_create_plot = target_width > 50 and target_height > 50  # Basic size check

        if can_create_plot and (
            needs_update_time or size_changed or first_data_received
        ):
            new_plot_surface = self.create_plot_surface(
                plot_data, target_width, target_height
            )
            # Only update cache if plot generation was successful
            if new_plot_surface:
                self.plot_surface = new_plot_surface
                self.last_plot_update_time = current_time

        return self.plot_surface
