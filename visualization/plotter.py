import pygame
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
from config import VisConfig


class FourStatsPlotter:
    def __init__(self, smooth_window=5):
        self.loss_pts: List[Tuple[int, float]] = []
        self.grad_pts: List[Tuple[int, float]] = []
        self.arwd_pts: List[Tuple[int, float]] = [] 
        self.asco_pts: List[Tuple[int, float]] = [] 
        self.alen_pts: List[Tuple[int, float]] = [] 
        self.max_points = VisConfig.MAX_PLOT_POINTS

        self.smooth_window = max(1, smooth_window)
        self.smooth_loss = deque(maxlen=self.smooth_window)
        self.smooth_grad = deque(maxlen=self.smooth_window)
        self.smooth_arwd = deque(maxlen=self.smooth_window)
        self.smooth_asco = deque(maxlen=self.smooth_window)
        self.smooth_alen = deque(maxlen=self.smooth_window)

        if not pygame.font.get_init():
            pygame.font.init()
        try:
            self.font_small = pygame.font.SysFont(None, 16)
            self.font_title = pygame.font.SysFont(None, 18)
        except Exception as e:
            print(f"Error initializing Pygame font: {e}. Using default.")
            self.font_small = pygame.font.Font(None, 16)
            self.font_title = pygame.font.Font(None, 18)

    def _add_point(
        self,
        point_list: List[Tuple[int, float]],
        smooth_deque: deque[float],
        x: int,
        val: Optional[float],
    ):
        if (
            val is None
            or not isinstance(val, (int, float))
            or math.isnan(val)
            or math.isinf(val)
        ):
            return

        smooth_deque.append(val)
        smoothed_val = np.mean(smooth_deque)

        point_list.append((x, smoothed_val))
        if len(point_list) > self.max_points:
            del point_list[: -self.max_points]

    def update_data(self, global_step: int, stats_summary: Dict[str, Any]):
        loss = stats_summary.get("avg_loss_100")
        grad = stats_summary.get("avg_grad_100")
        avg_rwd = stats_summary.get("avg_step_reward_1k")
        avg_sco = stats_summary.get("avg_score_100")
        avg_len = stats_summary.get("avg_length_100")

        self._add_point(self.loss_pts, self.smooth_loss, global_step, loss)
        self._add_point(self.grad_pts, self.smooth_grad, global_step, grad)
        self._add_point(self.arwd_pts, self.smooth_arwd, global_step, avg_rwd)
        self._add_point(self.asco_pts, self.smooth_asco, global_step, avg_sco)
        self._add_point(self.alen_pts, self.smooth_alen, global_step, avg_len)

    def render(self, surf: pygame.Surface):
        surf.fill(VisConfig.BLACK)
        w, h = surf.get_size()
        if w < 100 or h < 100:
            title_surf = self.font_title.render("Area Too Small", True, VisConfig.WHITE)
            surf.blit(title_surf, title_surf.get_rect(center=(w // 2, h // 2)))
            return

        num_cols = 2
        num_rows = 2
        cw = w // num_cols
        ch = h // num_rows

        rects = [
            pygame.Rect(0, 0, cw, ch),
            pygame.Rect(cw, 0, w - cw, ch),
            pygame.Rect(0, ch, cw, h - ch),
            pygame.Rect(cw, ch, w - cw, h - ch),
        ]

        data = [
            (self.loss_pts, (220, 80, 80), "Avg Loss (Smoothed)"),
            (self.asco_pts, (220, 220, 80), "Avg Ep Score (Smoothed)"),
            (self.alen_pts, (80, 80, 220), "Avg Ep Len (Smoothed)"),
            (self.arwd_pts, (80, 220, 80), "Avg Step Rwd (Smoothed)"),
        ]

        for rect, (pts, color, title) in zip(rects, data):
            self._draw_chart(surf, rect, pts, color, title)

    # _draw_chart method remains unchanged from the previous version
    def _draw_chart(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        pts: List[Tuple[int, float]],
        color: Tuple[int, int, int],
        title: str,
    ):
        pygame.draw.rect(surf, (40, 40, 40), rect, border_radius=3)
        pygame.draw.rect(surf, VisConfig.LIGHTG, rect, 1, border_radius=3)
        margin_left, margin_right, margin_top, margin_bottom = 35, 10, 20, 20
        plot_area_width = rect.width - margin_left - margin_right
        plot_area_height = rect.height - margin_top - margin_bottom
        plot_origin_x = rect.x + margin_left
        plot_origin_y = rect.y + rect.height - margin_bottom

        if plot_area_width <= 10 or plot_area_height <= 10:
            surf.blit(
                self.font_small.render("Area too small", True, VisConfig.WHITE),
                (rect.x + 2, rect.y + 2),
            )
            return

        pygame.draw.line(
            surf,
            VisConfig.WHITE,
            (plot_origin_x, plot_origin_y),
            (plot_origin_x + plot_area_width, plot_origin_y),
            1,
        )
        pygame.draw.line(
            surf,
            VisConfig.WHITE,
            (plot_origin_x, plot_origin_y),
            (plot_origin_x, rect.y + margin_top),
            1,
        )

        title_surf = self.font_title.render(title, True, VisConfig.WHITE)
        title_rect = title_surf.get_rect(topleft=(rect.x + 5, rect.y + 3))
        surf.blit(title_surf, title_rect)

        if not pts:
            no_data_surf = self.font_small.render("No data", True, VisConfig.LIGHTG)
            surf.blit(no_data_surf, no_data_surf.get_rect(center=rect.center))
            return

        try:
            xvals, yvals = zip(*pts)
            min_x, max_x = min(xvals), max(xvals)
            min_y_raw, max_y_raw = min(yvals), max(yvals)
        except ValueError:
            return

        range_x = max(1.0, float(max_x - min_x))
        range_y_raw = max_y_raw - min_y_raw
        padding_y = range_y_raw * 0.1 if range_y_raw > 1e-6 else 0.1
        min_y = min_y_raw - padding_y
        max_y = max_y_raw + padding_y
        range_y = max(1e-6, max_y - min_y)

        num_ticks_x = 3
        num_ticks_y = 4

        # X Ticks
        for i in range(num_ticks_x):
            val_x = (
                min_x + (range_x * i / (num_ticks_x - 1)) if num_ticks_x > 1 else min_x
            )
            px = plot_origin_x + np.clip(
                int((val_x - min_x) / range_x * plot_area_width), 0, plot_area_width
            )
            pygame.draw.line(
                surf, VisConfig.LIGHTG, (px, plot_origin_y), (px, plot_origin_y + 4), 1
            )
            if val_x >= 1_000_000:
                label_str = f"{val_x/1_000_000:.1f}M"
            elif val_x >= 1000:
                label_str = f"{val_x/1000:.0f}k"
            else:
                label_str = f"{int(val_x)}"
            lbl = self.font_small.render(label_str, True, VisConfig.WHITE)
            lbl_rect = lbl.get_rect(midtop=(px, plot_origin_y + 5))
            lbl_rect.clamp_ip(rect)
            surf.blit(lbl, lbl_rect)

        # Y Ticks
        for i in range(num_ticks_y):
            val_y = (
                max_y - (range_y * i / (num_ticks_y - 1)) if num_ticks_y > 1 else min_y
            )
            py = plot_origin_y - np.clip(
                int((val_y - min_y) / range_y * plot_area_height), 0, plot_area_height
            )
            pygame.draw.line(
                surf, VisConfig.LIGHTG, (plot_origin_x, py), (plot_origin_x - 4, py), 1
            )
            if abs(val_y) > 1e4 or (0 < abs(val_y) < 1e-2 and val_y != 0):
                label_str = f"{val_y:.1e}"
            elif val_y == 0:
                label_str = "0.00"
            else:
                label_str = f"{val_y:.2f}"
            lbl = self.font_small.render(label_str, True, VisConfig.WHITE)
            lbl_rect = lbl.get_rect(midright=(plot_origin_x - 5, py))
            lbl_rect.clamp_ip(rect)
            surf.blit(lbl, lbl_rect)

        # Draw Data Line
        line_points = []
        for x_val, y_val in pts:
            px = plot_origin_x + np.clip(
                int((x_val - min_x) / range_x * plot_area_width), 0, plot_area_width
            )
            py = plot_origin_y - np.clip(
                int((y_val - min_y) / range_y * plot_area_height), 0, plot_area_height
            )
            line_points.append((px, py))

        if len(line_points) > 1:
            try:
                pygame.draw.aalines(surf, color, False, line_points)
            except Exception:
                pygame.draw.lines(surf, color, False, line_points, 2)
        elif len(line_points) == 1:
            pygame.draw.circle(surf, color, line_points[0], 3)
