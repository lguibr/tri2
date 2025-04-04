# File: visualization/plotter.py
import pygame
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
from config import VisConfig


class FourStatsPlotter:
    """Renders multiple live training statistics plots."""

    def __init__(self, smooth_window=10):  # Slightly increased default smoothing
        # Deques to store (x, y) points for each metric
        self.loss_pts: Deque[Tuple[int, float]] = deque(
            maxlen=VisConfig.MAX_PLOT_POINTS
        )
        self.grad_pts: Deque[Tuple[int, float]] = deque(
            maxlen=VisConfig.MAX_PLOT_POINTS
        )  # Gradient norm
        self.arwd_pts: Deque[Tuple[int, float]] = deque(
            maxlen=VisConfig.MAX_PLOT_POINTS
        )  # Avg step reward
        self.asco_pts: Deque[Tuple[int, float]] = deque(
            maxlen=VisConfig.MAX_PLOT_POINTS
        )  # Avg episode score
        self.alen_pts: Deque[Tuple[int, float]] = deque(
            maxlen=VisConfig.MAX_PLOT_POINTS
        )  # Avg episode length
        self.eps_pts: Deque[Tuple[int, float]] = deque(
            maxlen=VisConfig.MAX_PLOT_POINTS
        )  # Epsilon
        self.beta_pts: Deque[Tuple[int, float]] = deque(
            maxlen=VisConfig.MAX_PLOT_POINTS
        )  # PER Beta
        self.sps_pts: Deque[Tuple[int, float]] = deque(
            maxlen=VisConfig.MAX_PLOT_POINTS
        )  # Steps per second
        self.buf_pts: Deque[Tuple[int, float]] = deque(
            maxlen=VisConfig.MAX_PLOT_POINTS
        )  # Buffer size
        self.q_pts: Deque[Tuple[int, float]] = deque(
            maxlen=VisConfig.MAX_PLOT_POINTS
        )  # Avg Max Q

        # Keep track of the latest raw values for display
        self.latest_values: Dict[str, Optional[float]] = {
            "loss": None,
            "grad": None,
            "arwd": None,
            "asco": None,
            "alen": None,
            "eps": None,
            "beta": None,
            "sps": None,
            "buf": None,
            "q": None,
        }

        # Smoothing (optional, applied before adding to main deques)
        self.smooth_window = max(1, smooth_window)
        self._smooth_buffers: Dict[str, deque[float]] = {
            k: deque(maxlen=self.smooth_window) for k in self.latest_values.keys()
        }

        # Pygame font setup
        if not pygame.font.get_init():
            pygame.font.init()
        try:
            # Smaller fonts for better fit
            self.font_small = pygame.font.SysFont(None, 15)
            self.font_title = pygame.font.SysFont(None, 16)
            self.font_value = pygame.font.SysFont(None, 16)  # Font for dynamic value
        except Exception as e:
            print(f"Error initializing Pygame font: {e}. Using default.")
            self.font_small = pygame.font.Font(None, 15)
            self.font_title = pygame.font.Font(None, 16)
            self.font_value = pygame.font.Font(None, 16)

    def _add_point(
        self,
        key: str,  # Identifier for the metric (e.g., "loss", "asco")
        point_deque: deque[Tuple[int, float]],
        x: int,  # Global step
        val: Optional[float],  # Raw value from stats_summary
    ):
        """Adds a point, applies smoothing, and stores the raw value."""
        # Store the latest raw value regardless of validity for plotting
        self.latest_values[key] = val

        # Only add valid points to the plot deque
        if (
            val is None
            or not isinstance(val, (int, float))
            or math.isnan(val)
            or math.isinf(val)
        ):
            # Don't add invalid points, but keep latest_value updated
            return

        # Apply smoothing
        smooth_deque = self._smooth_buffers[key]
        smooth_deque.append(val)
        smoothed_val = np.mean(smooth_deque) if smooth_deque else val

        # Add the smoothed value to the plot deque
        point_deque.append((x, smoothed_val))
        # MAX_PLOT_POINTS handled by deque's maxlen

    def update_data(self, global_step: int, stats_summary: Dict[str, Any]):
        """Updates all data points based on the stats summary dictionary."""
        self._add_point(
            "loss", self.loss_pts, global_step, stats_summary.get("avg_loss_100")
        )
        self._add_point(
            "grad", self.grad_pts, global_step, stats_summary.get("avg_grad_100")
        )
        self._add_point(
            "arwd", self.arwd_pts, global_step, stats_summary.get("avg_step_reward_1k")
        )
        self._add_point(
            "asco", self.asco_pts, global_step, stats_summary.get("avg_score_100")
        )
        self._add_point(
            "alen", self.alen_pts, global_step, stats_summary.get("avg_length_100")
        )
        self._add_point("eps", self.eps_pts, global_step, stats_summary.get("epsilon"))
        self._add_point("beta", self.beta_pts, global_step, stats_summary.get("beta"))
        self._add_point(
            "sps", self.sps_pts, global_step, stats_summary.get("steps_per_second")
        )
        self._add_point(
            "buf", self.buf_pts, global_step, stats_summary.get("buffer_size")
        )
        self._add_point(
            "q", self.q_pts, global_step, stats_summary.get("avg_max_q_100")
        )  # Use the averaged Q

    def render(self, surf: pygame.Surface):
        """Renders all the defined plots onto the given surface."""
        surf.fill(VisConfig.BLACK)  # Background for the whole plot area
        w, h = surf.get_size()
        if w < 100 or h < 150:  # Minimum size to be useful
            title_surf = self.font_title.render(
                "Plot Area Too Small", True, VisConfig.WHITE
            )
            surf.blit(title_surf, title_surf.get_rect(center=(w // 2, h // 2)))
            return

        # --- Define Plots ---
        # Arrange plots in a grid
        num_cols = 2
        num_rows = 3  # Adjust if more plots are needed (e.g., 4 rows for 8 plots)
        plot_w = w // num_cols
        plot_h = h // num_rows

        # Dictionary mapping keys to plot configurations
        plot_definitions = {
            "asco": {
                "data": self.asco_pts,
                "color": (220, 220, 80),
                "title": "Avg Score",
                "col": 0,
                "row": 0,
                "format": ".2f",
            },
            "loss": {
                "data": self.loss_pts,
                "color": (220, 80, 80),
                "title": "Avg Loss",
                "col": 1,
                "row": 0,
                "format": ".4f",
            },
            "alen": {
                "data": self.alen_pts,
                "color": (80, 80, 220),
                "title": "Avg Length",
                "col": 0,
                "row": 1,
                "format": ".1f",
            },
            "q": {
                "data": self.q_pts,
                "color": (80, 220, 80),
                "title": "Avg Max Q",
                "col": 1,
                "row": 1,
                "format": ".3f",
            },
            "eps": {
                "data": self.eps_pts,
                "color": (200, 100, 200),
                "title": "Epsilon",
                "col": 0,
                "row": 2,
                "format": ".3f",
            },
            "sps": {
                "data": self.sps_pts,
                "color": (100, 200, 200),
                "title": "Steps/Sec",
                "col": 1,
                "row": 2,
                "format": ".0f",
            },
            # Add more plots here if needed (e.g., beta, buffer size, grad norm) by adjusting rows/cols
            # "beta": {"data": self.beta_pts, "color": (255, 165, 0), "title": "PER Beta", "col": ?, "row": ?, "format": ".3f"},
            # "buf": {"data": self.buf_pts, "color": (160, 160, 160), "title": "Buffer Size", "col": ?, "row": ?, "format": ".0f"},
        }

        # --- Render Each Plot ---
        for key, plot_def in plot_definitions.items():
            # Calculate rectangle for this specific plot
            rect = pygame.Rect(
                plot_def["col"] * plot_w, plot_def["row"] * plot_h, plot_w, plot_h
            )
            # Add small padding/margin around each plot rect for visual separation
            padded_rect = rect.inflate(-6, -6)  # Slightly more padding

            if padded_rect.width > 20 and padded_rect.height > 20:  # Min sensible size
                self._draw_chart(
                    surf,
                    padded_rect,
                    plot_def["data"],
                    plot_def["color"],
                    plot_def["title"],
                    self.latest_values.get(key),  # Pass latest raw value
                    plot_def["format"],  # Pass format string
                )
            else:
                # Draw placeholder if too small
                pygame.draw.rect(surf, (50, 0, 0), rect, 1)
                if (
                    padded_rect.width > 5 and padded_rect.height > 5
                ):  # Render text if minimal space
                    err_surf = self.font_small.render("Too Small", True, VisConfig.GRAY)
                    surf.blit(err_surf, err_surf.get_rect(center=rect.center))

    def _draw_chart(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        pts: deque[Tuple[int, float]],
        color: Tuple[int, int, int],
        title: str,
        latest_val: Optional[float],  # <<< NEW: Latest raw value
        val_format: str,  # <<< NEW: Format string for value display
    ):
        """Draws a single chart with grid lines and dynamic value display."""
        # --- Background and Border ---
        pygame.draw.rect(
            surf, (35, 35, 35), rect, border_radius=3
        )  # Slightly lighter background
        pygame.draw.rect(surf, VisConfig.LIGHTG, rect, 1, border_radius=3)

        # --- Plotting Area Margins ---
        # Adjusted margins for better axis label space
        margin_left, margin_right = 35, 10
        margin_top, margin_bottom = 25, 20  # Increased top margin for title+value
        plot_area_width = rect.width - margin_left - margin_right
        plot_area_height = rect.height - margin_top - margin_bottom
        plot_origin_x = rect.x + margin_left
        plot_origin_y = rect.y + rect.height - margin_bottom  # Bottom-left of plot area

        # Check if plot area is too small to draw
        if plot_area_width <= 10 or plot_area_height <= 10:
            surf.blit(
                self.font_small.render("Area too small", True, VisConfig.WHITE),
                (rect.x + 2, rect.y + 2),
            )
            return

        # --- Title and Dynamic Value ---
        # Display Latest Value
        if latest_val is not None:
            val_str = f"{latest_val:{val_format}}"
        else:
            val_str = "N/A"  # If no data yet
        dynamic_title = f"{title}: {val_str}"
        title_surf = self.font_title.render(dynamic_title, True, VisConfig.WHITE)
        # Position title at the top-left of the padded rect
        title_rect = title_surf.get_rect(topleft=(rect.x + 5, rect.y + 4))
        surf.blit(title_surf, title_rect)

        # --- Handle No Data ---
        if not pts:
            no_data_surf = self.font_small.render("No data", True, VisConfig.LIGHTG)
            surf.blit(no_data_surf, no_data_surf.get_rect(center=rect.center))
            return

        # --- Calculate Data Ranges ---
        try:
            # Deques might be empty initially
            if not pts:
                raise ValueError("No points to plot")
            xvals, yvals = zip(*pts)
            min_x, max_x = min(xvals), (
                max(xvals) if len(xvals) > 1 else (xvals[0], xvals[0])
            )
            min_y_raw, max_y_raw = min(yvals), (
                max(yvals) if len(yvals) > 1 else (yvals[0], yvals[0])
            )
        except ValueError:
            # Handle cases with zero or one point gracefully
            min_x, max_x, min_y_raw, max_y_raw = 0, 1, 0, 1  # Default ranges
            if len(pts) == 1:
                min_x, max_x = pts[0][0], pts[0][0]
                min_y_raw, max_y_raw = pts[0][1], pts[0][1]

        # Add padding to Y range for better visualization
        range_y_raw = max_y_raw - min_y_raw
        padding_y = range_y_raw * 0.1 if range_y_raw > 1e-6 else 0.1
        min_y = min_y_raw - padding_y
        max_y = max_y_raw + padding_y
        range_y = max(1e-6, max_y - min_y)  # Avoid division by zero
        range_x = max(1.0, float(max_x - min_x))  # Avoid division by zero

        # --- Draw Axes and Grid Lines ---
        num_ticks_x = 4  # More ticks for X axis
        num_ticks_y = 4  # Ticks for Y axis

        # Y-Axis Ticks and Horizontal Grid Lines
        for i in range(num_ticks_y + 1):  # +1 to include top line
            val_y = min_y + (range_y * i / num_ticks_y) if num_ticks_y > 0 else min_y
            py = plot_origin_y - np.clip(
                int((val_y - min_y) / range_y * plot_area_height), 0, plot_area_height
            )
            # Draw Grid Line
            pygame.draw.line(
                surf,
                VisConfig.GRAY,
                (plot_origin_x, py),
                (plot_origin_x + plot_area_width, py),
                1,
            )
            # Draw Tick Mark
            pygame.draw.line(
                surf, VisConfig.WHITE, (plot_origin_x, py), (plot_origin_x - 4, py), 1
            )
            # Draw Label
            if abs(val_y) > 1e4 or (0 < abs(val_y) < 1e-2 and val_y != 0):
                label_str = f"{val_y:.1e}"
            elif val_y == 0:
                label_str = "0.0"
            else:
                label_str = f"{val_y:.2f}"
            lbl = self.font_small.render(label_str, True, VisConfig.WHITE)
            lbl_rect = lbl.get_rect(midright=(plot_origin_x - 6, py))
            # Ensure label doesn't overlap plot above/below
            if lbl_rect.top < rect.y + margin_top - 10:
                lbl_rect.top = rect.y + margin_top - 10
            if lbl_rect.bottom > rect.y + rect.height:
                lbl_rect.bottom = rect.y + rect.height
            surf.blit(lbl, lbl_rect)

        # X-Axis Ticks and Vertical Grid Lines
        for i in range(num_ticks_x + 1):  # +1 to include right line
            val_x = min_x + (range_x * i / num_ticks_x) if num_ticks_x > 0 else min_x
            px = plot_origin_x + np.clip(
                int((val_x - min_x) / range_x * plot_area_width), 0, plot_area_width
            )
            # Draw Grid Line
            pygame.draw.line(
                surf,
                VisConfig.GRAY,
                (px, plot_origin_y),
                (px, plot_origin_y - plot_area_height),
                1,
            )
            # Draw Tick Mark
            pygame.draw.line(
                surf, VisConfig.WHITE, (px, plot_origin_y), (px, plot_origin_y + 4), 1
            )
            # Draw Label (formatted: K, M)
            if val_x >= 1_000_000:
                label_str = f"{val_x/1_000_000:.1f}M"
            elif val_x >= 1000:
                label_str = f"{val_x/1000:.0f}k"
            else:
                label_str = f"{int(val_x)}"
            lbl = self.font_small.render(label_str, True, VisConfig.WHITE)
            lbl_rect = lbl.get_rect(midtop=(px, plot_origin_y + 6))
            # Ensure label doesn't overlap plot left/right
            if lbl_rect.left < rect.x:
                lbl_rect.left = rect.x
            if lbl_rect.right > rect.x + rect.width:
                lbl_rect.right = rect.x + rect.width
            surf.blit(lbl, lbl_rect)

        # Re-draw main axes lines on top of grid
        pygame.draw.line(
            surf,
            VisConfig.WHITE,
            (plot_origin_x, plot_origin_y),
            (plot_origin_x + plot_area_width, plot_origin_y),
            1,
        )  # X-axis
        pygame.draw.line(
            surf,
            VisConfig.WHITE,
            (plot_origin_x, plot_origin_y),
            (plot_origin_x, rect.y + margin_top),
            1,
        )  # Y-axis

        # --- Draw Data Line ---
        line_points = []
        for x_val, y_val in pts:
            # Map data coordinates to pixel coordinates
            px = plot_origin_x + np.clip(
                int((x_val - min_x) / range_x * plot_area_width), 0, plot_area_width
            )
            py = plot_origin_y - np.clip(
                int((y_val - min_y) / range_y * plot_area_height), 0, plot_area_height
            )
            line_points.append((px, py))

        if len(line_points) > 1:
            try:
                # Use anti-aliased lines if available and less prone to errors
                pygame.draw.aalines(surf, color, False, line_points)
            except Exception:
                # Fallback to standard lines
                pygame.draw.lines(surf, color, False, line_points, 1)  # Thickness 1
        elif len(line_points) == 1:
            # Draw a small circle if only one point exists
            pygame.draw.circle(surf, color, line_points[0], 2)
