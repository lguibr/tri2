# File: ui/plot_utils.py
import numpy as np
from typing import Optional, List, Union, Tuple
import matplotlib
import traceback
import math

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig


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
            "font.size": 9,  # Base font size slightly increased
            "axes.labelsize": 9,  # Increased label size
            "axes.titlesize": 11,  # Increased title size
            "xtick.labelsize": 8,  # Increased tick label size
            "ytick.labelsize": 8,  # Increased tick label size
            "legend.fontsize": 8,  # Increased legend font size
            "figure.facecolor": "#262626",
            "axes.facecolor": "#303030",
            "axes.edgecolor": "#707070",
            "axes.labelcolor": "#D0D0D0",
            "xtick.color": "#C0C0C0",
            "ytick.color": "#C0C0C0",
            "grid.color": "#505050",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "axes.titlepad": 6,  # Reduced title padding
            "legend.frameon": True,  # Add frame to legend
            "legend.framealpha": 0.85,  # Increased legend background alpha
            "legend.facecolor": "#202020",  # Darker legend background
            "legend.title_fontsize": 8,  # Increased legend title font size
        }
    )
except Exception as e:
    print(f"Warning: Failed to set Matplotlib style: {e}")


TREND_SLOPE_TOLERANCE = 1e-5
TREND_MIN_LINEWIDTH = 1
TREND_MAX_LINEWIDTH = 2
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

MIN_ALPHA = 0.4
MAX_ALPHA = 1.0
MIN_DATA_AVG_LINEWIDTH = 1
MAX_DATA_AVG_LINEWIDTH = 2

# Y_PADDING_FACTOR = 0.20 # Removed vertical padding factor


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
        return float(max_val)  # Ensure float
    inverted_rank = (total_ranks - 1) - rank
    fraction = inverted_rank / max(1, total_ranks - 1)
    # --- Explicitly cast to float before subtraction/addition ---
    f_min_val = float(min_val)
    f_max_val = float(max_val)
    value = f_min_val + (f_max_val - f_min_val) * fraction
    # --- End explicit cast ---
    # Clip using original min/max in case casting caused issues, ensure float return
    return float(np.clip(value, min_val, max_val))


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
    show_placeholder: bool = True,
    placeholder_text: Optional[str] = None,
    y_log_scale: bool = False,
):
    """
    Renders data with linearly scaled alpha/linewidth. Trend line is thin, white, dashed.
    Title is just the label. Detailed values moved to legend. Best value shown as legend title.
    Applies a background tint and border to the entire subplot based on trend desirability.
    Legend now includes current values and trend slope, placed at center-left.
    Handles empty data explicitly to show placeholder.
    Removed vertical padding, removed horizontal padding and xlabel.
    Increased title and legend font sizes, increased legend background alpha.
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
    best_val_str = f"Best: {_format_value(best_val, is_lower_better)}"

    # Set only the main title
    ax.set_title(
        label,
        loc="left",
        fontsize=plt.rcParams["axes.titlesize"],
        pad=plt.rcParams.get("axes.titlepad", 6),
    )

    try:
        x_coords = np.arange(n_points)
        plotted_legend_items = False
        min_y_overall = float("inf")
        max_y_overall = float("-inf")

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
        min_y_overall = min(min_y_overall, np.min(valid_data))
        max_y_overall = max(max_y_overall, np.max(valid_data))
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
                if len(rolling_avg) > 0:
                    min_y_overall = min(min_y_overall, np.min(rolling_avg))
                    max_y_overall = max(max_y_overall, np.max(rolling_avg))
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
            # Don't include trend line in min/max calculation for ylim
            plotted_legend_items = True

        ax.tick_params(axis="both", which="major")
        ax.grid(
            True,
            linestyle=plt.rcParams["grid.linestyle"],
            alpha=plt.rcParams["grid.alpha"],
        )

        # --- Adjust Y-axis limits WITHOUT padding ---
        if np.isfinite(min_y_overall) and np.isfinite(max_y_overall):
            if abs(max_y_overall - min_y_overall) < 1e-6:  # Handle constant data
                # Add a tiny epsilon to avoid zero range
                epsilon = max(abs(min_y_overall * 0.01), 1e-6)
                ax.set_ylim(min_y_overall - epsilon, max_y_overall + epsilon)
            else:
                ax.set_ylim(min_y_overall, max_y_overall)
        # else: Keep default limits if min/max calculation failed

        if y_log_scale and min_y_overall > 1e-9:
            ax.set_yscale("log")
            # Adjust log scale limits if needed, ensuring bottom is positive
            current_bottom, current_top = ax.get_ylim()
            new_bottom = max(current_bottom, 1e-9)  # Ensure bottom is positive
            if new_bottom >= current_top:  # Prevent invalid limits
                new_bottom = current_top / 10
            ax.set_ylim(bottom=new_bottom, top=current_top)
        else:
            ax.set_yscale("linear")

        # --- Adjust X-axis limits (remove padding) ---
        if n_points > 1:
            ax.set_xlim(0, n_points - 1)  # Set limits tightly
        elif n_points == 1:
            ax.set_xlim(-0.5, 0.5)  # Keep slight padding for single point

        # --- X-axis Ticks Formatting ---
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

        # --- Legend ---
        if plotted_legend_items:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(0, 0.5),
                title=best_val_str,
                fontsize=plt.rcParams["legend.fontsize"],
            )

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
