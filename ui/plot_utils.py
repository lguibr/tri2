import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
import matplotlib
import warnings
from io import BytesIO
import traceback
import math

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
            "axes.edgecolor": "#707070",
            "axes.labelcolor": "#D0D0D0",
            "xtick.color": "#C0C0C0",
            "ytick.color": "#C0C0C0",
            "grid.color": "#505050",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "font.small_title_values": 7,
            "axes.titlepad": 12,
        }
    )
except Exception as e:
    print(f"Warning: Failed to set Matplotlib style: {e}")


TREND_SLOPE_TOLERANCE = 1e-5
TREND_MIN_LINEWIDTH = 1.5
TREND_MAX_LINEWIDTH = 3.5
TREND_COLOR_STABLE = normalize_color_for_matplotlib(VisConfig.YELLOW)
TREND_COLOR_INCREASING = normalize_color_for_matplotlib((0, 200, 0))
TREND_COLOR_DECREASING = normalize_color_for_matplotlib((200, 0, 0))
TREND_SLOPE_SCALE_FACTOR = 5.0
TREND_BACKGROUND_ALPHA = 0.15

TREND_LINE_COLOR = (1.0, 1.0, 1.0)
TREND_LINE_STYLE = (0, (5, 10))
TREND_LINE_WIDTH = 0.75
TREND_LINE_ALPHA = 0.7
TREND_LINE_ZORDER = 10

MIN_ALPHA = 0.3
MAX_ALPHA = 1.0
MIN_DATA_AVG_LINEWIDTH = 1
MAX_DATA_AVG_LINEWIDTH = 3


def calculate_trend_line(data: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Calculates the slope and intercept of the linear regression line.
    Returns (slope, intercept) or None if calculation fails.
    """
    n_points = len(data)
    if n_points < 2:
        return None
    try:
        x_coords = np.arange(n_points)
        finite_mask = np.isfinite(data)
        if np.sum(finite_mask) < 2:
            return None
        coeffs = np.polyfit(x_coords[finite_mask], data[finite_mask], 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return None
        return slope, intercept
    except (np.linalg.LinAlgError, ValueError):
        return None


def get_trend_color(slope: float) -> Tuple[float, float, float]:
    """
    Maps a slope to a color (Red -> Yellow(Stable) -> Green).
    Assumes positive slope is "good" and negative is "bad".
    The caller should adjust the slope sign based on metric goal.
    """
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_COLOR_STABLE
    norm_slope = math.atan(slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0)
    norm_slope = np.clip(norm_slope, -1.0, 1.0)
    if norm_slope > 0:
        t = norm_slope
        color = tuple(
            TREND_COLOR_STABLE[i] * (1 - t) + TREND_COLOR_INCREASING[i] * t
            for i in range(3)
        )
    else:
        t = abs(norm_slope)
        color = tuple(
            TREND_COLOR_STABLE[i] * (1 - t) + TREND_COLOR_DECREASING[i] * t
            for i in range(3)
        )
    return tuple(np.clip(c, 0.0, 1.0) for c in color)


def get_trend_linewidth(slope: float) -> float:
    """Maps the *magnitude* of a slope to a border linewidth."""
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_MIN_LINEWIDTH
    norm_slope_mag = abs(math.atan(slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0))
    norm_slope_mag = np.clip(norm_slope_mag, 0.0, 1.0)
    linewidth = TREND_MIN_LINEWIDTH + norm_slope_mag * (
        TREND_MAX_LINEWIDTH - TREND_MIN_LINEWIDTH
    )
    return linewidth


def _interpolate_visual_property(
    rank: int, total_ranks: int, min_val: float, max_val: float
) -> float:
    """
    Linearly interpolates alpha or linewidth based on rank.
    Rank 0 corresponds to max_val (most prominent).
    Rank (total_ranks - 1) corresponds to min_val (least prominent).
    """
    if total_ranks <= 1:
        return max_val
    inverted_rank = (total_ranks - 1) - rank
    fraction = inverted_rank / max(1, total_ranks - 1)
    value = min_val + (max_val - min_val) * fraction
    return np.clip(value, min_val, max_val)


def _format_value(value: float, is_loss: bool) -> str:
    """Formats value based on magnitude and whether it's a loss."""
    if not np.isfinite(value):
        return "N/A"
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.1e}"
    if abs(value) >= 1000:
        return f"{value:.2g}"
    if is_loss:
        return f"{value:.3f}"
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.2f}"


def _format_slope(slope: float) -> str:
    """Formats slope value for display in the legend."""
    if not np.isfinite(slope):
        return "N/A"
    sign = "+" if slope >= 0 else ""
    if abs(slope) < 1e-4:
        return f"{sign}{slope:.1e}"
    elif abs(slope) < 0.1:
        return f"{sign}{slope:.3f}"
    else:
        return f"{sign}{slope:.2f}"


def render_single_plot(
    ax,
    data: List[Union[float, int]],
    label: str,
    color: Tuple[float, float, float],
    rolling_window_sizes: List[int],
    xlabel: Optional[str] = None,
    show_placeholder: bool = True,
    placeholder_text: Optional[str] = None,
    y_log_scale: bool = False,
):
    """
    Renders data with linearly scaled alpha/linewidth. Trend line is thin, white, dashed.
    Title is two lines: Label, then compact value pairs.
    Applies a background tint and border to the entire subplot based on trend desirability.
    Legend now includes current values and trend slope.
    Handles empty data explicitly to show placeholder.
    """
    try:
        data_np = np.array(data, dtype=float)
        finite_mask = np.isfinite(data_np)
        valid_data = data_np[finite_mask]
    except (ValueError, TypeError):
        valid_data = np.array([])

    n_points = len(valid_data)
    placeholder_text_color = normalize_color_for_matplotlib(VisConfig.GRAY)

    # --- Explicitly handle n_points == 0 case ---
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
        ax.patch.set_facecolor(plt.rcParams["axes.facecolor"])
        ax.patch.set_edgecolor(plt.rcParams["axes.edgecolor"])
        ax.patch.set_linewidth(0.5)
        return  # Exit early if no data
    # --- End n_points == 0 handling ---

    # --- Continue with plotting if n_points > 0 ---
    trend_params = calculate_trend_line(valid_data)
    trend_slope = trend_params[0] if trend_params is not None else 0.0

    is_lower_better = "loss" in label.lower() or "entropy" in label.lower()

    effective_slope_for_color = -trend_slope if is_lower_better else trend_slope
    trend_indicator_color = get_trend_color(effective_slope_for_color)
    trend_indicator_lw = get_trend_linewidth(trend_slope)

    plotted_windows = sorted([w for w in rolling_window_sizes if n_points >= w])
    total_ranks = 1 + len(plotted_windows)

    current_val = valid_data[-1]
    best_val = np.min(valid_data) if is_lower_better else np.max(valid_data)

    title_line1 = f"{label}"
    value_pairs = []
    value_pairs.append(
        f"Now:{_format_value(current_val, is_lower_better)}|Best:{_format_value(best_val, is_lower_better)}"
    )

    avg_values = {}
    for avg_window in plotted_windows:
        try:
            avg_values[avg_window] = np.mean(valid_data[-avg_window:])
        except Exception:
            avg_values[avg_window] = np.nan

    avg_windows_sorted = sorted(avg_values.keys())
    for i in range(0, len(avg_windows_sorted), 2):
        win1 = avg_windows_sorted[i]
        avg1 = avg_values.get(win1, np.nan)
        pair_str = f"A{win1}:{_format_value(avg1, is_lower_better)}"

        if i + 1 < len(avg_windows_sorted):
            win2 = avg_windows_sorted[i + 1]
            avg2 = avg_values.get(win2, np.nan)
            pair_str += f"|A{win2}:{_format_value(avg2, is_lower_better)}"
            if np.isfinite(avg1) and np.isfinite(avg2):
                diff = avg1 - avg2
                diff_sign = "+" if diff >= 0 else ""
                pair_str += f"(D:{diff_sign}{_format_value(diff, is_lower_better)})"
        value_pairs.append(pair_str)

    title_line2 = " • ".join(value_pairs)

    ax.set_title(
        title_line1,
        loc="left",
        fontsize=plt.rcParams["axes.titlesize"],
        pad=plt.rcParams.get("axes.titlepad", 6),
    )
    ax.text(
        0.01,
        0.99,
        title_line2,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=plt.rcParams.get("font.small_title_values", 7),
        color=plt.rcParams["axes.labelcolor"],
    )

    try:
        x_coords = np.arange(n_points)
        plotted_legend_items = False

        # Plot Raw Data
        raw_data_rank = total_ranks - 1
        raw_data_alpha = _interpolate_visual_property(
            raw_data_rank, total_ranks, MIN_ALPHA, MAX_ALPHA
        )
        raw_data_lw = _interpolate_visual_property(
            raw_data_rank,
            total_ranks,
            MIN_DATA_AVG_LINEWIDTH,
            MAX_DATA_AVG_LINEWIDTH,
        )
        raw_label = f"Raw: {_format_value(current_val, is_lower_better)}"
        ax.plot(
            x_coords,
            valid_data,
            color=color,
            linewidth=raw_data_lw,
            label=raw_label,
            alpha=raw_data_alpha,
        )
        plotted_legend_items = True

        # Plot Rolling Averages
        for i, avg_window in enumerate(plotted_windows):
            avg_rank = len(plotted_windows) - 1 - i
            current_alpha = _interpolate_visual_property(
                avg_rank, total_ranks, MIN_ALPHA, MAX_ALPHA
            )
            current_avg_lw = _interpolate_visual_property(
                avg_rank,
                total_ranks,
                MIN_DATA_AVG_LINEWIDTH,
                MAX_DATA_AVG_LINEWIDTH,
            )
            weights = np.ones(avg_window) / avg_window
            rolling_avg = np.convolve(valid_data, weights, mode="valid")
            avg_x_coords = np.arange(avg_window - 1, n_points)
            linestyle = "-"
            if len(avg_x_coords) == len(rolling_avg):
                last_avg_val = rolling_avg[-1] if len(rolling_avg) > 0 else np.nan
                avg_label = (
                    f"Avg {avg_window}: {_format_value(last_avg_val, is_lower_better)}"
                )
                ax.plot(
                    avg_x_coords,
                    rolling_avg,
                    color=color,
                    linewidth=current_avg_lw,
                    alpha=current_alpha,
                    linestyle=linestyle,
                    label=avg_label,
                )
                plotted_legend_items = True

        # Plot Trend Line
        if trend_params is not None and n_points >= 2:
            slope, intercept = trend_params
            x_trend = np.array([0, n_points - 1])
            y_trend = slope * x_trend + intercept
            trend_label = f"Trend: {_format_slope(slope)}"
            ax.plot(
                x_trend,
                y_trend,
                color=TREND_LINE_COLOR,
                linestyle=TREND_LINE_STYLE,
                linewidth=TREND_LINE_WIDTH,
                alpha=TREND_LINE_ALPHA,
                label=trend_label,
                zorder=TREND_LINE_ZORDER,
            )
            plotted_legend_items = True

        ax.tick_params(axis="both", which="major")
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(
            True,
            linestyle=plt.rcParams["grid.linestyle"],
            alpha=plt.rcParams["grid.alpha"],
        )

        min_val_plot = np.min(valid_data)
        max_val_plot = np.max(valid_data)
        padding_factor = 0.1
        range_val = max_val_plot - min_val_plot
        if abs(range_val) < 1e-6:
            padding = (
                max(abs(max_val_plot * padding_factor), 0.5)
                if max_val_plot != 0
                else 0.5
            )
        else:
            padding = range_val * padding_factor
        padding = max(padding, 1e-6)
        ax.set_ylim(min_val_plot - padding, max_val_plot + padding)

        if y_log_scale and min_val_plot > 1e-9:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(min_val_plot * 0.9, 1e-9))
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

        if plotted_legend_items:
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

    # Apply background tint and border based on trend
    bg_color_with_alpha = (*trend_indicator_color, TREND_BACKGROUND_ALPHA)
    ax.patch.set_facecolor(bg_color_with_alpha)
    ax.patch.set_edgecolor(trend_indicator_color)
    ax.patch.set_linewidth(trend_indicator_lw)
