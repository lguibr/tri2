# File: ui/panels/left_panel_components/notification_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple, Optional
from config import VisConfig, StatsConfig, WHITE, LIGHTG, GRAY, YELLOW, RED, GREEN
import numpy as np


class NotificationRenderer:
    """Renders the notification area (Simplified)."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.label_font = fonts.get("notification_label", pygame.font.Font(None, 16))
        self.value_font = fonts.get("notification", pygame.font.Font(None, 18))

    def render(
        self, area_rect: pygame.Rect, stats_summary: Dict[str, Any]
    ) -> Dict[str, pygame.Rect]:
        """Renders the simplified notification content (e.g., total episodes)."""
        stat_rects: Dict[str, pygame.Rect] = {}
        pygame.draw.rect(self.screen, (45, 45, 45), area_rect, border_radius=3)
        pygame.draw.rect(self.screen, LIGHTG, area_rect, 1, border_radius=3)
        stat_rects["Notification Area"] = area_rect

        if not self.value_font or not self.label_font:
            return stat_rects

        padding = 5
        line_height = self.value_font.get_linesize() + 2
        y = area_rect.top + padding

        # Display Total Episodes
        total_episodes = stats_summary.get("total_episodes", 0)
        label_surf = self.label_font.render("Total Episodes:", True, LIGHTG)
        value_surf = self.value_font.render(
            f"{total_episodes:,}".replace(",", "_"), True, WHITE
        )

        label_rect = label_surf.get_rect(topleft=(area_rect.left + padding, y))
        value_rect = value_surf.get_rect(topleft=(label_rect.right + 4, y))

        self.screen.blit(label_surf, label_rect)
        self.screen.blit(value_surf, value_rect)

        stat_rects["Total Episodes Info"] = label_rect.union(value_rect)

        # Best value rendering removed

        return stat_rects
