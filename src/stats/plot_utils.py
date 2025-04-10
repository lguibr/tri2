# File: src/stats/plot_utils.py
import numpy as np
from typing import Optional, List, Union, Tuple
import matplotlib
import traceback
import math
import logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import colors from visualization module
from src.visualization.core import colors as vis_colors

logger = logging.getLogger(__name__)

# --- Constants ---
TREND_SLOPE_TOLERANCE = 1e-5
TREND_MIN_LINEWIDTH = 1
TREND_MAX_LINEWIDTH = 2
TREND_COLOR_STABLE = (1.0, 1.0, 0.0)  # Yellow
TREND_COLOR_INCREASING = (0.0, 0.8, 0.0)  # Green
TREND_COLOR_DECREASING = (0.8, 0.0, 0.0)  # Red
TREND_SLOPE_SCALE_FACTOR = 5.0
TREND_BACKGROUND_ALPHA = 0.15
TREND_LINE_COLOR = (1.0, 1.0, 1.0)
TREND_LINE_STYLE = (0, (5, 10))
TREND_LINE_WIDTH = 0.75
TREND_LINE_ALPHA = 0.7
TREND_LINE_ZORDER = 10
MIN_ALPHA = 0.4  # Alpha for averages/raw line
MAX_ALPHA = 1.0  # Alpha for averages
MIN_DATA_AVG_LINEWIDTH = 1
MAX_DATA_AVG_LINEWIDTH = 2
RAW_DATA_LINEWIDTH = 0.8  # Thinner line for raw data
RAW_DATA_ALPHA = 0.3  # Lower alpha for raw data
DEFAULT_ROLLING_WINDOWS = [10, 50]  # Default rolling average windows


# --- Helper Functions ---
def normalize_color_for_matplotlib(
    color_tuple_0_255: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        return tuple(c / 255.0 for c in color_tuple_0_255)
    logger.warning(f"Invalid color format for normalization: {color_tuple_0_255}")
    return (0.0, 0.0, 0.0)  # Default black


# --- Matplotlib Style Setup ---
try:
    plt.style.use("dark_background")
    # Remove invalid legend.bbox_to_anchor parameter
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,  # Slightly smaller title
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,  # Smaller legend
            "figure.facecolor": "#262626",
            "axes.facecolor": "#303030",
            "axes.edgecolor": "#707070",
            "axes.labelcolor": "#D0D0D0",
            "xtick.color": "#C0C0C0",
            "ytick.color": "#C0C0C0",
            "grid.color": "#505050",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "axes.titlepad": 4,  # Reduced padding
            "legend.frameon": False,  # No frame for legend
            "legend.loc": "best",  # Use 'best' or specify per plot if needed
            # "legend.bbox_to_anchor": (0, 0.5), # REMOVED - Invalid parameter
            "legend.labelspacing": 0.3,
            "legend.handletextpad": 0.5,
            "legend.handlelength": 1.0,
        }
    )
except Exception as e:
    logger.warning(f"Failed to set Matplotlib style: {e}")


# --- Trend Calculation ---
def calculate_trend_line(
    x_coords: np.ndarray, y_data: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Calculates the slope and intercept of the linear regression line."""
    mask = np.isfinite(y_data)
    if np.sum(mask) < 2:
        return None
    try:
        coeffs = np.polyfit(x_coords[mask], y_data[mask], 1)
        if not all(np.isfinite(c) for c in coeffs):
            return None
        return coeffs[0], coeffs[1]  # slope, intercept
    except (np.linalg.LinAlgError, ValueError):
        return None


def get_trend_color(slope: float, lower_is_better: bool) -> Tuple[float, float, float]:
    """Maps slope to color (Red -> Yellow -> Green)."""
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_COLOR_STABLE
    eff_slope = -slope if lower_is_better else slope
    norm_slope = np.clip(
        math.atan(eff_slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0), -1.0, 1.0
    )
    t = abs(norm_slope)
    base, target = (
        (TREND_COLOR_STABLE, TREND_COLOR_INCREASING)
        if norm_slope > 0
        else (TREND_COLOR_STABLE, TREND_COLOR_DECREASING)
    )
    color = tuple(base[i] * (1 - t) + target[i] * t for i in range(3))
    return tuple(np.clip(c, 0.0, 1.0) for c in color)


def get_trend_linewidth(slope: float) -> float:
    """Maps slope magnitude to border linewidth."""
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_MIN_LINEWIDTH
    norm_mag = np.clip(
        abs(math.atan(slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0)), 0.0, 1.0
    )
    return TREND_MIN_LINEWIDTH + norm_mag * (TREND_MAX_LINEWIDTH - TREND_MIN_LINEWIDTH)


# --- Visual Property Interpolation ---
def _interpolate_visual_property(
    rank: int, total_ranks: int, min_val: float, max_val: float
) -> float:
    """Linearly interpolates alpha/linewidth based on rank."""
    if total_ranks <= 1:
        return float(max_val)
    # Rank 0 is the longest average (most prominent), higher ranks are shorter averages
    fraction = rank / max(1, total_ranks - 1)
    value = float(max_val) - (float(max_val) - float(min_val)) * fraction
    return float(np.clip(value, min_val, max_val))


# --- Value Formatting ---
def _format_value(value: float, is_loss: bool) -> str:
    """Formats value based on magnitude and whether it's a loss."""
    if not np.isfinite(value):
        return "N/A"
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.1e}"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", "_")
    if is_loss:
        return f"{value:.3f}"
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _format_slope(slope: float) -> str:
    """Formats slope value for display in the legend."""
    if not np.isfinite(slope):
        return "N/A"
    sign = "+" if slope >= 0 else ""
    abs_slope = abs(slope)
    if abs_slope < 1e-4:
        return f"{sign}{slope:.1e}"
    if abs_slope < 0.1:
        return f"{sign}{slope:.3f}"
    return f"{sign}{slope:.2f}"


# --- Main Plotting Function ---
def render_single_plot(
    ax,
    x_coords: List[int],  # Added x_coords (steps)
    y_data: List[Union[float, int]],  # Renamed data to y_data
    label: str,
    color: Tuple[float, float, float],
    rolling_window_sizes: List[int] = DEFAULT_ROLLING_WINDOWS,
    show_placeholder: bool = True,
    placeholder_text: Optional[str] = None,
    y_log_scale: bool = False,
):
    """Renders data with rolling averages, trend line, and informative legend."""
    try:
        x_coords_np = np.array(x_coords, dtype=float)
        y_data_np = np.array(y_data, dtype=float)
        # Ensure x and y have same length after potential conversion errors
        min_len = min(len(x_coords_np), len(y_data_np))
        x_coords_np = x_coords_np[:min_len]
        y_data_np = y_data_np[:min_len]

        valid_mask = np.isfinite(y_data_np)
        valid_y = y_data_np[valid_mask]
        valid_x = x_coords_np[valid_mask]

    except (ValueError, TypeError):
        valid_y = np.array([])
        valid_x = np.array([])

    n_points = len(valid_y)
    is_lower_better = "loss" in label.lower()

    if n_points == 0:  # Handle empty data
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
                color=normalize_color_for_matplotlib(vis_colors.GRAY),
            )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"{label} (N/A)", loc="left")
        ax.grid(False)
        ax.patch.set_facecolor(plt.rcParams["axes.facecolor"])
        ax.patch.set_edgecolor(plt.rcParams["axes.edgecolor"])
        ax.patch.set_linewidth(0.5)
        return

    trend_params = calculate_trend_line(valid_x, valid_y)  # Use valid_x for trend calc
    trend_slope = trend_params[0] if trend_params else 0.0
    trend_color = get_trend_color(trend_slope, is_lower_better)
    trend_lw = get_trend_linewidth(trend_slope)
    plotted_windows = sorted([w for w in rolling_window_sizes if n_points >= w])
    total_ranks = 1 + len(plotted_windows)
    current_val = valid_y[-1]
    best_val = np.min(valid_y) if is_lower_better else np.max(valid_y)
    best_val_str = f"Best: {_format_value(best_val, is_lower_better)}"
    ax.set_title(label, loc="left")

    try:
        plotted_legend_items = 0
        min_y, max_y = float("inf"), float("-inf")

        # --- Always plot raw data first (less prominent) ---
        raw_label = f"Val: {_format_value(current_val, is_lower_better)}"
        ax.plot(
            valid_x,  # Use valid_x
            valid_y,
            color=color,
            linewidth=RAW_DATA_LINEWIDTH,
            label=raw_label,
            alpha=RAW_DATA_ALPHA,
            zorder=5,
        )
        min_y = min(min_y, np.min(valid_y))
        max_y = max(max_y, np.max(valid_y))
        plotted_legend_items += 1
        # --- End Plot Raw Data ---

        # --- Plot Rolling Averages (more prominent) ---
        for i, avg_win in enumerate(plotted_windows):
            rank = i
            alpha = _interpolate_visual_property(
                rank, len(plotted_windows), MIN_ALPHA, MAX_ALPHA
            )
            lw = _interpolate_visual_property(
                rank,
                len(plotted_windows),
                MIN_DATA_AVG_LINEWIDTH,
                MAX_DATA_AVG_LINEWIDTH,
            )

            weights = np.ones(avg_win) / avg_win
            # Convolve on valid y data
            rolling_avg = np.convolve(valid_y, weights, mode="valid")
            # Calculate corresponding x coordinates for the rolling average
            avg_x = valid_x[avg_win - 1 :]  # Use valid_x

            if len(avg_x) == len(rolling_avg):
                last_avg = rolling_avg[-1] if len(rolling_avg) > 0 else np.nan
                avg_label = f"Avg {avg_win}: {_format_value(last_avg, is_lower_better)}"
                ax.plot(
                    avg_x,
                    rolling_avg,
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                    linestyle="-",
                    label=avg_label,
                    zorder=6 + i,
                )
                if len(rolling_avg) > 0:
                    min_y = min(min_y, np.min(rolling_avg))
                    max_y = max(max_y, np.max(rolling_avg))
                plotted_legend_items += 1
        # --- End Plot Rolling Averages ---

        # --- Plot Trend Line ---
        if trend_params and n_points >= 2:
            slope, intercept = trend_params
            # Use min/max of valid_x for trend line ends
            x_trend = np.array([valid_x[0], valid_x[-1]])
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
            plotted_legend_items += 1
        # --- End Plot Trend Line ---

        ax.tick_params(axis="both", which="major")
        ax.grid(
            True,
            linestyle=plt.rcParams["grid.linestyle"],
            alpha=plt.rcParams["grid.alpha"],
        )

        if np.isfinite(min_y) and np.isfinite(max_y):
            yrange = max(max_y - min_y, 1e-6)
            pad = yrange * 0.05
            ax.set_ylim(min_y - pad, max_y + pad)

        if y_log_scale and min_y > 1e-9:
            ax.set_yscale("log")
            bottom, top = ax.get_ylim()
            new_bottom = max(bottom, 1e-9)
            if new_bottom >= top:
                new_bottom = top / 10
            ax.set_ylim(bottom=new_bottom, top=top)
        else:
            ax.set_yscale("linear")

        # Set x-limits based on actual step values
        if n_points > 1:
            ax.set_xlim(valid_x[0], valid_x[-1])
        elif n_points == 1:
            ax.set_xlim(valid_x[0] - 0.5, valid_x[0] + 0.5)  # Center single point

        # X-axis formatting for large numbers (based on step values)
        max_step = valid_x[-1] if n_points > 0 else 0
        if max_step > 1000:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

            def fmt_func(v, _):
                val = int(v)
                return (
                    f"{val/1e6:.1f}M"
                    if val >= 1e6
                    else (f"{val/1e3:.0f}k" if val >= 1e3 else f"{val}")
                )

            ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_func))
        elif n_points > 10:  # Use n_points check for density, max_step for scale
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))

        # Add legend if items were plotted
        if plotted_legend_items > 0:
            ax.legend(title=best_val_str)

    except Exception as plot_err:
        logger.error(
            f"ERROR during render_single_plot for '{label}': {plot_err}", exc_info=True
        )
        ax.text(
            0.5,
            0.5,
            f"Plot Error\n({label})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color=normalize_color_for_matplotlib(vis_colors.RED),
        )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)

    # Set background and border based on trend
    ax.patch.set_facecolor((*trend_color, TREND_BACKGROUND_ALPHA))
    ax.patch.set_edgecolor(trend_color)
    ax.patch.set_linewidth(trend_lw)
