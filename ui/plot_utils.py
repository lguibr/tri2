# File: ui/plot_utils.py
import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
import matplotlib
import warnings
from io import BytesIO

# Ensure Agg backend is used before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, BufferConfig, StatsConfig, DQNConfig

# --- Matplotlib Style Configuration ---
# Moved here for better organization
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
        }
    )
except Exception as e:
    print(f"Warning: Failed to set Matplotlib style: {e}")
# --- End Style Configuration ---


def normalize_color_for_matplotlib(
    color_tuple_0_255: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Converts RGB 0-255 tuple to Matplotlib's 0.0-1.0 tuple."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        return tuple(c / 255.0 for c in color_tuple_0_255)
    else:
        print(f"Warning: Invalid color tuple {color_tuple_0_255}, returning black.")
        return (0.0, 0.0, 0.0)


# --- Extracted Subplot Rendering Logic ---
def render_single_plot(
    ax,
    data: List[Union[float, int]],
    label: str,
    color: Tuple[float, float, float],
    avg_color: Optional[Tuple[float, float, float]],
    rolling_window_size: int,
    xlabel: Optional[str] = None,
    show_placeholder: bool = True,
    placeholder_text: Optional[str] = None,
    y_log_scale: bool = False,
    default_line_width: float = 1.0,
    avg_line_width: float = 1.5,
    avg_line_alpha: float = 0.7,
):
    """Renders data onto a single Matplotlib Axes object."""
    n_points = len(data)
    latest_val_str = ""
    avg_window = rolling_window_size

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

        if avg_color is not None and n_points >= avg_window:
            weights = np.ones(avg_window) / avg_window
            rolling_avg = np.convolve(data, weights, mode="valid")
            avg_x_coords = np.arange(avg_window - 1, n_points)
            ax.plot(
                avg_x_coords,
                rolling_avg,
                color=avg_color,
                linewidth=avg_line_width,
                alpha=avg_line_alpha,
                label=f"Avg {avg_window}",
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
