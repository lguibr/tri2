# File: ui/plot_utils.py
import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
import matplotlib
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
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "figure.facecolor": "#262626",
            "axes.facecolor": "#303030",
            "axes.edgecolor": "#707070",  # Default edge color
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


# --- NEW: Trend Colors and Settings ---
TREND_INCREASING_COLOR = normalize_color_for_matplotlib((0, 180, 0))  # Darker Green
TREND_DECREASING_COLOR = normalize_color_for_matplotlib((180, 0, 0))  # Darker Red
TREND_STABLE_COLOR = normalize_color_for_matplotlib((70, 70, 70))  # Default gray border
TREND_LINEWIDTH = 1.5  # Make trend border slightly thicker
DEFAULT_LINEWIDTH = 0.8  # Default border thickness
TREND_TOLERANCE = 1e-6  # Tolerance for stability check
# --- END NEW ---


# --- MODIFIED: Function signature and logic (added trend border) ---
def render_single_plot(
    ax,
    data: List[Union[float, int]],
    label: str,
    color: Tuple[float, float, float],
    avg_color_primary: Optional[Tuple[float, float, float]],
    avg_colors_secondary: List[Tuple[float, float, float]],
    rolling_window_sizes: List[int],
    xlabel: Optional[str] = None,
    show_placeholder: bool = True,
    placeholder_text: Optional[str] = None,
    y_log_scale: bool = False,
    default_line_width: float = 1.0,
    avg_line_width: float = 1.5,
    avg_line_alpha: float = 0.7,
):
    """Renders data onto a single Matplotlib Axes object, including multiple rolling averages and trend indicator."""
    n_points = len(data)
    latest_val_str = ""
    primary_avg_window = rolling_window_sizes[0] if rolling_window_sizes else 0

    # --- Reset border style initially ---
    for spine in ax.spines.values():
        spine.set_color(TREND_STABLE_COLOR)
        spine.set_linewidth(DEFAULT_LINEWIDTH)
    # --- End Reset ---

    if data:
        current_val = data[-1]
        if (
            n_points >= primary_avg_window
            and avg_color_primary is not None
            and primary_avg_window > 0
        ):
            try:
                latest_avg = np.mean(data[-primary_avg_window:])
                latest_val_str = f" (Now: {current_val:.3g}, Avg{primary_avg_window}: {latest_avg:.3g})"
            except Exception:
                latest_val_str = f" (Now: {current_val:.3g})"
        else:
            latest_val_str = f" (Now: {current_val:.3g})"
    ax.set_title(f"{label}{latest_val_str}", fontsize=plt.rcParams["axes.titlesize"])

    placeholder_text_color = normalize_color_for_matplotlib(VisConfig.GRAY)
    if n_points == 0:
        if show_placeholder:
            p_text = placeholder_text if placeholder_text else f"{label}\n(No data yet)"
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
            x_coords, data, color=color, linewidth=default_line_width, label=f"{label}"
        )

        avg_linestyles = ["-", "--", ":", "-."]
        plotted_averages = False
        longest_avg_rolling = None
        longest_window = 0

        for i, avg_window in enumerate(rolling_window_sizes):
            if n_points >= avg_window:
                if i == 0 and avg_color_primary:
                    current_avg_color = avg_color_primary
                elif avg_colors_secondary:
                    current_avg_color = avg_colors_secondary[
                        (i - 1) % len(avg_colors_secondary)
                    ]
                else:
                    current_avg_color = color

                weights = np.ones(avg_window) / avg_window
                rolling_avg = np.convolve(data, weights, mode="valid")
                avg_x_coords = np.arange(avg_window - 1, n_points)
                linestyle = avg_linestyles[i % len(avg_linestyles)]
                ax.plot(
                    avg_x_coords,
                    rolling_avg,
                    color=current_avg_color,
                    linewidth=avg_line_width,
                    alpha=avg_line_alpha,
                    linestyle=linestyle,
                    label=f"Avg {avg_window}",
                )
                plotted_averages = True
                # --- Store the longest average calculated ---
                if avg_window >= longest_window:
                    longest_window = avg_window
                    longest_avg_rolling = rolling_avg
            # --- End average plotting loop ---

        # --- Determine and apply trend border based on longest average ---
        if longest_avg_rolling is not None and len(longest_avg_rolling) >= 2:
            latest_avg_long = longest_avg_rolling[-1]
            prev_avg_long = longest_avg_rolling[-2]
            trend_color = TREND_STABLE_COLOR
            trend_lw = DEFAULT_LINEWIDTH

            if latest_avg_long > prev_avg_long + TREND_TOLERANCE:
                trend_color = TREND_INCREASING_COLOR
                trend_lw = TREND_LINEWIDTH
            elif latest_avg_long < prev_avg_long - TREND_TOLERANCE:
                trend_color = TREND_DECREASING_COLOR
                trend_lw = TREND_LINEWIDTH

            for spine in ax.spines.values():
                spine.set_color(trend_color)
                spine.set_linewidth(trend_lw)
        # --- End trend border ---

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
        padding_factor = 0.1
        # --- Prevent y-axis range collapsing to zero ---
        range_val = max_val - min_val
        if range_val < 1e-6:  # Very small range
            padding = max(
                abs(max_val * padding_factor), 0.5
            )  # Use larger default padding
        else:
            padding = range_val * padding_factor

        # Ensure padding is not excessively small
        padding = max(padding, 1e-6)
        # Set y-limits, ensuring lower can be negative if min_val is negative
        ax.set_ylim(min_val - padding, max_val + padding)
        # --- End y-axis range fix ---

        if y_log_scale and min_val > 1e-9:
            # Ensure bottom limit is positive for log scale
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(min_val * 0.9, 1e-9))
        else:
            ax.set_yscale("linear")

        if n_points > 1:
            ax.set_xlim(-0.02 * n_points, n_points - 1 + 0.02 * n_points)
        elif n_points == 1:
            ax.set_xlim(-0.5, 0.5)

        if n_points > 1000:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

            def format_func(value, tick_number):
                if value >= 1_000_000:
                    return f"{value/1_000_000:.1f}M"
                if value >= 1_000:
                    return f"{value/1_000:.0f}k"
                return f"{int(value)}"

            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        elif n_points > 10:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))

        if plotted_averages:
            ax.legend(loc="best", fontsize=plt.rcParams["legend.fontsize"])

    except Exception as plot_err:
        print(f"ERROR during render_single_plot for '{label}': {plot_err}")
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


# --- END MODIFIED ---
