# File: ui/panels/left_panel_components/notification_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple, Optional
from config import VisConfig, StatsConfig, WHITE, LIGHTG, GRAY, YELLOW, RED, GREEN
import numpy as np


class NotificationRenderer:
    """Renders the notification area with best scores/loss.
    Refactored for AlphaZero focus."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.label_font = fonts.get("notification_label", pygame.font.Font(None, 16))
        self.value_font = fonts.get("notification", pygame.font.Font(None, 18))

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
        lower_is_better: bool = False,  # Added flag for loss vs score
    ) -> pygame.Rect:
        """Renders a single line within the notification area."""
        if not self.label_font or not self.value_font:
            return pygame.Rect(0, y_pos, 0, 0)

        padding = 5
        label_color, value_color = LIGHTG, WHITE
        prev_color, time_color = GRAY, (180, 180, 100)

        # 1. Render Label
        label_surf = self.label_font.render(label, True, label_color)
        label_rect = label_surf.get_rect(topleft=(area_rect.left + padding, y_pos))
        self.screen.blit(label_surf, label_rect)
        current_x = label_rect.right + 4

        # 2. Render Current Best Value
        current_val_str = "N/A"
        val_as_float: Optional[float] = None
        if isinstance(current_val, (int, float, np.number)):
            try:
                val_as_float = float(current_val)
            except (ValueError, TypeError):
                val_as_float = None

        if val_as_float is not None and np.isfinite(val_as_float):
            try:
                current_val_str = val_format.format(val_as_float)
            except (ValueError, TypeError):
                current_val_str = "ErrFmt"

        val_surf = self.value_font.render(current_val_str, True, value_color)
        val_rect = val_surf.get_rect(topleft=(current_x, y_pos))
        self.screen.blit(val_surf, val_rect)
        current_x = val_rect.right + 4

        # 3. Render Previous Best Value (Optional)
        prev_val_str = "(N/A)"
        prev_val_as_float: Optional[float] = None
        if isinstance(prev_val, (int, float, np.number)):
            try:
                prev_val_as_float = float(prev_val)
            except (ValueError, TypeError):
                prev_val_as_float = None

        if prev_val_as_float is not None and np.isfinite(prev_val_as_float):
            # Only show previous if it's different and valid
            if val_as_float is None or not np.isclose(val_as_float, prev_val_as_float):
                try:
                    prev_val_str = f"({val_format.format(prev_val_as_float)})"
                except (ValueError, TypeError):
                    prev_val_str = "(ErrFmt)"
            else:
                prev_val_str = ""  # Don't show if same as current

        prev_rect = pygame.Rect(current_x, y_pos + 1, 0, 0)  # Initialize rect
        if prev_val_str:
            prev_surf = self.label_font.render(prev_val_str, True, prev_color)
            prev_rect = prev_surf.get_rect(topleft=(current_x, y_pos + 1))
            self.screen.blit(prev_surf, prev_rect)
            current_x = prev_rect.right + 6
        else:
            current_x += 6  # Add spacing even if not shown

        # 4. Render Time Since Best
        steps_ago_str = self._format_steps_ago(current_step, best_step)
        time_surf = self.label_font.render(steps_ago_str, True, time_color)
        time_rect = time_surf.get_rect(topleft=(current_x, y_pos + 1))

        # Clip time text if needed
        available_width = area_rect.right - time_rect.left - padding
        clip_rect = pygame.Rect(0, 0, max(0, available_width), time_rect.height)
        if time_rect.width > available_width > 0:
            self.screen.blit(time_surf, time_rect, area=clip_rect)
        elif available_width > 0:
            self.screen.blit(time_surf, time_rect)

        # Return the union rect for hover detection if needed
        union_rect = label_rect.union(val_rect).union(prev_rect).union(time_rect)
        union_rect.width = min(union_rect.width, area_rect.width - 2 * padding)
        return union_rect

    def render(
        self, area_rect: pygame.Rect, stats_summary: Dict[str, Any]
    ) -> Dict[str, pygame.Rect]:
        """Renders the notification content with relevant AlphaZero bests."""
        stat_rects: Dict[str, pygame.Rect] = {}
        pygame.draw.rect(self.screen, (45, 45, 45), area_rect, border_radius=3)
        pygame.draw.rect(self.screen, LIGHTG, area_rect, 1, border_radius=3)
        stat_rects["Notification Area"] = area_rect

        if not self.value_font:
            return stat_rects

        padding = 5
        line_height = self.value_font.get_linesize()
        current_step = stats_summary.get("global_step", 0)
        y = area_rect.top + padding

        # Render Best Game Score
        rect_game = self._render_line(
            area_rect,
            y,
            "Best Game Score:",
            stats_summary.get("best_game_score", -float("inf")),
            stats_summary.get("previous_best_game_score", -float("inf")),
            stats_summary.get("best_game_score_step", 0),
            "{:.0f}",
            current_step,
            lower_is_better=False,
        )
        stat_rects["Best Game Score Info"] = rect_game.clip(area_rect)
        y += line_height

        # Render Best Value Loss
        rect_v_loss = self._render_line(
            area_rect,
            y,
            "Best Value Loss:",
            stats_summary.get("best_value_loss", float("inf")),
            stats_summary.get("previous_best_value_loss", float("inf")),
            stats_summary.get("best_value_loss_step", 0),
            "{:.4f}",
            current_step,
            lower_is_better=True,
        )
        stat_rects["Best Value Loss Info"] = rect_v_loss.clip(area_rect)
        y += line_height

        # Render Best Policy Loss
        rect_p_loss = self._render_line(
            area_rect,
            y,
            "Best Policy Loss:",
            stats_summary.get("best_policy_loss", float("inf")),
            stats_summary.get("previous_best_policy_loss", float("inf")),
            stats_summary.get("best_policy_loss_step", 0),
            "{:.4f}",
            current_step,
            lower_is_better=True,
        )
        stat_rects["Best Policy Loss Info"] = rect_p_loss.clip(area_rect)

        return stat_rects
