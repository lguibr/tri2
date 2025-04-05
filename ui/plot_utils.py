# File: ui/plot_utils.py
import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
import matplotlib
import warnings
from io import BytesIO
import traceback
import math  # Added for atan

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, StatsConfig


def normalize_color_for_matplotlib(
    color_tuple_0_255: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        return tuple(c / 255.0 for c in color_tuple_0_255)
    else:
        print(f"Warning: Invalid color tuple {color_tuple_0_255}, using black.")
        return (0.0, 0.0, 0.0)


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


# --- Trend Border Configuration ---
TREND_SLOPE_TOLERANCE = 1e-5
TREND_MIN_LINEWIDTH = 1.5
TREND_MAX_LINEWIDTH = 3.5  # Slightly increased max
# MODIFIED: Changed stable color and increased scale factor
TREND_COLOR_STABLE = normalize_color_for_matplotlib(
    VisConfig.YELLOW
)  # Stable/zero slope is Yellow
TREND_COLOR_INCREASING = normalize_color_for_matplotlib((0, 200, 0))  # Max green
TREND_COLOR_DECREASING = normalize_color_for_matplotlib((200, 0, 0))  # Max red
TREND_SLOPE_SCALE_FACTOR = 5.0  # Increased scale factor to amplify smaller slopes


def calculate_trend_slope(data: np.ndarray) -> float:
    """Calculates the slope of the linear regression line for the data."""
    n_points = len(data)
    if n_points < 2:
        return 0.0
    try:
        x_coords = np.arange(n_points)
        finite_mask = np.isfinite(data)
        if np.sum(finite_mask) < 2:
            return 0.0
        coeffs = np.polyfit(x_coords[finite_mask], data[finite_mask], 1)
        slope = coeffs[0]
        return slope if np.isfinite(slope) else 0.0
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def get_trend_color(slope: float) -> Tuple[float, float, float]:
    """Maps a slope to a color (Red -> Yellow(Stable) -> Green)."""
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_COLOR_STABLE

    norm_slope = math.atan(slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0)
    norm_slope = np.clip(norm_slope, -1.0, 1.0)

    if norm_slope > 0:  # Increasing trend (Stable(Yellow) -> Green)
        t = norm_slope
        color = tuple(
            TREND_COLOR_STABLE[i] * (1 - t) + TREND_COLOR_INCREASING[i] * t
            for i in range(3)
        )
    else:  # Decreasing trend (Stable(Yellow) -> Red)
        t = abs(norm_slope)
        color = tuple(
            TREND_COLOR_STABLE[i] * (1 - t) + TREND_COLOR_DECREASING[i] * t
            for i in range(3)
        )
    return tuple(np.clip(c, 0.0, 1.0) for c in color)


def get_trend_linewidth(slope: float) -> float:
    """Maps a slope to a border linewidth."""
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_MIN_LINEWIDTH

    norm_slope_mag = abs(math.atan(slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0))
    norm_slope_mag = np.clip(norm_slope_mag, 0.0, 1.0)

    linewidth = TREND_MIN_LINEWIDTH + norm_slope_mag * (
        TREND_MAX_LINEWIDTH - TREND_MIN_LINEWIDTH
    )
    return linewidth


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
    """Renders data onto a single Matplotlib Axes object with trend-based border."""
    valid_data = [
        d for d in data if isinstance(d, (int, float, np.number)) and np.isfinite(d)
    ]
    n_points = len(valid_data)

    data_to_plot = np.array(valid_data)
    trend_slope = calculate_trend_slope(data_to_plot)
    trend_color = get_trend_color(trend_slope)
    trend_lw = get_trend_linewidth(trend_slope)

    placeholder_text_color = normalize_color_for_matplotlib(VisConfig.GRAY)

    # --- Handle Placeholder or Plot Data ---
    if n_points == 0:
        if show_placeholder:
            p_text = placeholder_text if placeholder_text else f"{label}\n(No data)"
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
        ax.set_title(f"{label} (N/A)", fontsize=plt.rcParams["axes.titlesize"])
        ax.grid(False)
    else:
        # Determine Title
        latest_val_str = ""
        primary_avg_window = rolling_window_sizes[0] if rolling_window_sizes else 0
        current_val = data_to_plot[-1]
        if (
            n_points >= primary_avg_window
            and avg_color_primary is not None
            and primary_avg_window > 0
        ):
            try:
                latest_avg = np.mean(data_to_plot[-primary_avg_window:])
                latest_val_str = f" (Now: {current_val:.3g}, Avg{primary_avg_window}: {latest_avg:.3g})"
            except Exception:
                latest_val_str = f" (Now: {current_val:.3g})"
        else:
            latest_val_str = f" (Now: {current_val:.3g})"
        ax.set_title(
            f"{label}{latest_val_str}", fontsize=plt.rcParams["axes.titlesize"]
        )

        # Plot Data and Averages
        try:
            x_coords = np.arange(n_points)
            ax.plot(
                x_coords,
                data_to_plot,
                color=color,
                linewidth=default_line_width,
                label=f"{label}",
            )

            avg_linestyles = ["-", "--", ":", "-."]
            plotted_averages = False

            for i, avg_window in enumerate(rolling_window_sizes):
                if n_points >= avg_window:
                    current_avg_color = color
                    if i == 0 and avg_color_primary:
                        current_avg_color = avg_color_primary
                    elif i > 0 and avg_colors_secondary:
                        current_avg_color = avg_colors_secondary[
                            (i - 1) % len(avg_colors_secondary)
                        ]

                    weights = np.ones(avg_window) / avg_window
                    rolling_avg = np.convolve(data_to_plot, weights, mode="valid")
                    avg_x_coords = np.arange(avg_window - 1, n_points)
                    linestyle = avg_linestyles[i % len(avg_linestyles)]

                    if len(avg_x_coords) == len(rolling_avg):
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
                    else:
                        print(
                            f"Warning: Mismatch in rolling avg shapes for {label} (window {avg_window})"
                        )

            # Axes Formatting
            ax.tick_params(axis="both", which="major")
            if xlabel:
                ax.set_xlabel(xlabel)
            ax.grid(
                True,
                linestyle=plt.rcParams["grid.linestyle"],
                alpha=plt.rcParams["grid.alpha"],
            )

            min_val = np.min(data_to_plot)
            max_val = np.max(data_to_plot)
            padding_factor = 0.1
            range_val = max_val - min_val
            if abs(range_val) < 1e-6:
                padding = (
                    max(abs(max_val * padding_factor), 0.5) if max_val != 0 else 0.5
                )
            else:
                padding = range_val * padding_factor
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

            if n_points > 1000:
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

                def format_func(value, tick_number):
                    val_int = int(value)
                    if val_int >= 1_000_000:
                        return f"{val_int/1_000_000:.1f}M"
                    if val_int >= 1_000:
                        return f"{val_int/1_000:.0f}k"
                    return f"{val_int}"

                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            elif n_points > 10:
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))

            if plotted_averages:
                ax.legend(loc="best", fontsize=plt.rcParams["legend.fontsize"])

        except Exception as plot_err:
            print(f"ERROR during render_single_plot for '{label}': {plot_err}")
            traceback.print_exc()
            error_text_color = normalize_color_for_matplotlib(VisConfig.RED)
            ax.text(
                0.5,
                0.5,
                f"Plot Error\n({label})",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color=error_text_color,
            )
            ax.set_yticks([])
            ax.set_xticks([])
            ax.grid(False)

    # --- Set Border AFTER all plotting/formatting ---
    for spine in ax.spines.values():
        spine.set_color(trend_color)
        spine.set_linewidth(trend_lw)
