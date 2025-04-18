# File: ui/panels/left_panel_components/notification_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple
from config import VisConfig, StatsConfig
import numpy as np


class NotificationRenderer:
    """Renders the notification area with best scores/loss."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts

    def _format_steps_ago(self, current_step: int, best_step: int) -> str:
        """Formats the difference in steps into a readable string."""
        if best_step <= 0 or current_step <= best_step:
            return "Now"
        diff = current_step - best_step
        if diff < 1000:
            return f"{diff} steps ago"
        elif diff < 1_000_000:
            return f"{diff / 1000:.1f}k steps ago"
        else:
            return f"{diff / 1_000_000:.1f}M steps ago"

    def _render_line(
        self,
        area_rect: pygame.Rect,
        y_pos: int,
        label: str,
        current_val: Any,
        prev_val: Any,
        best_step: int,
        val_format: str,
        current_step: int,
    ) -> pygame.Rect:
        """Renders a single line within the notification area."""
        label_font = self.fonts.get("notification_label")
        value_font = self.fonts.get("notification")
        if not label_font or not value_font:
            return pygame.Rect(0, y_pos, 0, 0)

        padding = 5
        label_color, value_color = VisConfig.LIGHTG, VisConfig.WHITE
        prev_color, time_color = VisConfig.GRAY, (180, 180, 100)

        label_surf = label_font.render(label, True, label_color)
        label_rect = label_surf.get_rect(topleft=(area_rect.left + padding, y_pos))
        self.screen.blit(label_surf, label_rect)
        current_x = label_rect.right + 4

        current_val_str = "N/A"
        # --- Convert to float *before* checking ---
        val_as_float: Optional[float] = None
        if isinstance(
            current_val, (int, float, np.number)
        ):  # Check against np.number too
            try:
                val_as_float = float(current_val)
            except (ValueError, TypeError):
                val_as_float = None  # Conversion failed

        # --- Use the converted float for checks and formatting ---
        if val_as_float is not None and np.isfinite(val_as_float):
            try:
                current_val_str = val_format.format(val_as_float)
            except (ValueError, TypeError) as fmt_err:
                current_val_str = "ErrFmt"

        val_surf = value_font.render(current_val_str, True, value_color)
        val_rect = val_surf.get_rect(topleft=(current_x, y_pos))
        self.screen.blit(val_surf, val_rect)
        current_x = val_rect.right + 4

        prev_val_str = "(N/A)"
        # --- Convert prev_val to float before checking ---
        prev_val_as_float: Optional[float] = None
        if isinstance(prev_val, (int, float, np.number)):
            try:
                prev_val_as_float = float(prev_val)
            except (ValueError, TypeError):
                prev_val_as_float = None

        if prev_val_as_float is not None and np.isfinite(prev_val_as_float):
            try:
                prev_val_str = f"({val_format.format(prev_val_as_float)})"
            except (ValueError, TypeError):
                prev_val_str = "(ErrFmt)"

        prev_surf = label_font.render(prev_val_str, True, prev_color)
        prev_rect = prev_surf.get_rect(topleft=(current_x, y_pos + 1))
        self.screen.blit(prev_surf, prev_rect)
        current_x = prev_rect.right + 6

        steps_ago_str = self._format_steps_ago(current_step, best_step)
        time_surf = label_font.render(steps_ago_str, True, time_color)
        time_rect = time_surf.get_rect(topleft=(current_x, y_pos + 1))

        available_width = area_rect.right - time_rect.left - padding
        clip_rect = pygame.Rect(0, 0, max(0, available_width), time_rect.height)
        if time_rect.width > available_width > 0:
            self.screen.blit(time_surf, time_rect, area=clip_rect)
        elif available_width > 0:
            self.screen.blit(time_surf, time_rect)

        union_rect = label_rect.union(val_rect).union(prev_rect).union(time_rect)
        union_rect.width = min(union_rect.width, area_rect.width - 2 * padding)
        return union_rect

    def render(
        self, area_rect: pygame.Rect, stats_summary: Dict[str, Any]
    ) -> Dict[str, pygame.Rect]:
        """Renders the notification content."""
        stat_rects: Dict[str, pygame.Rect] = {}
        pygame.draw.rect(self.screen, (45, 45, 45), area_rect, border_radius=3)
        pygame.draw.rect(self.screen, VisConfig.LIGHTG, area_rect, 1, border_radius=3)
        stat_rects["Notification Area"] = area_rect

        value_font = self.fonts.get("notification")
        if not value_font:
            return stat_rects

        padding = 5
        line_height = value_font.get_linesize()
        current_step = stats_summary.get("global_step", 0)
        y = area_rect.top + padding

        # --- Pass values from summary, _render_line handles conversion/check ---
        rect_rl = self._render_line(
            area_rect,
            y,
            "RL Score:",
            stats_summary.get("best_score", -float("inf")),
            stats_summary.get("previous_best_score", -float("inf")),
            stats_summary.get("best_score_step", 0),
            "{:.2f}",
            current_step,
        )
        stat_rects["Best RL Score Info"] = rect_rl.clip(area_rect)
        y += line_height

        rect_game = self._render_line(
            area_rect,
            y,
            "Game Score:",
            stats_summary.get("best_game_score", -float("inf")),
            stats_summary.get("previous_best_game_score", -float("inf")),
            stats_summary.get("best_game_score_step", 0),
            "{:.0f}",
            current_step,
        )
        stat_rects["Best Game Score Info"] = rect_game.clip(area_rect)
        y += line_height

        rect_loss = self._render_line(
            area_rect,
            y,
            "Loss:",
            stats_summary.get("best_loss", float("inf")),
            stats_summary.get("previous_best_loss", float("inf")),
            stats_summary.get("best_loss_step", 0),
            "{:.4f}",
            current_step,
        )
        stat_rects["Best Loss Info"] = rect_loss.clip(area_rect)

        return stat_rects
