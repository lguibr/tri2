# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
from typing import Dict, Tuple, Optional
from config import VisConfig


class ButtonStatusRenderer:
    """Renders the top buttons and status line in the left panel."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts

    def _draw_button(self, rect: pygame.Rect, text: str, color: Tuple[int, int, int]):
        """Helper to draw a single button."""
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        ui_font = self.fonts.get("ui")
        if ui_font:
            lbl_surf = ui_font.render(text, True, VisConfig.WHITE)
            self.screen.blit(lbl_surf, lbl_surf.get_rect(center=rect.center))
        else:
            pygame.draw.line(
                self.screen, VisConfig.RED, rect.topleft, rect.bottomright, 2
            )
            pygame.draw.line(
                self.screen, VisConfig.RED, rect.topright, rect.bottomleft, 2
            )

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_running: bool,  # Renamed from is_training
        status: str,
    ) -> Tuple[int, Dict[str, pygame.Rect], Optional[pygame.Rect]]:
        """Renders buttons and status. Returns next_y, stat_rects, notification_rect."""
        stat_rects: Dict[str, pygame.Rect] = {}
        notification_rect = None
        next_y = y_start

        if app_state == "MainMenu":
            run_btn_rect = pygame.Rect(10, y_start, 100, 40)  # Renamed
            cleanup_btn_rect = pygame.Rect(run_btn_rect.right + 10, y_start, 160, 40)
            demo_btn_rect = pygame.Rect(cleanup_btn_rect.right + 10, y_start, 120, 40)

            self._draw_button(
                run_btn_rect,  # Use run_btn_rect
                (
                    "Stop" if is_running and status == "Training" else "Run"
                ),  # Updated text
                (70, 70, 70),
            )
            self._draw_button(cleanup_btn_rect, "Cleanup This Run", (100, 40, 40))
            self._draw_button(demo_btn_rect, "Play Demo", (40, 100, 40))

            stat_rects["Run Button"] = run_btn_rect  # Renamed key
            stat_rects["Cleanup Button"] = cleanup_btn_rect
            stat_rects["Play Demo Button"] = demo_btn_rect

            notification_x = demo_btn_rect.right + 15
            notification_w = panel_width - notification_x - 10
            notif_font = self.fonts.get("notification")
            if notification_w > 50 and notif_font:
                line_h = notif_font.get_linesize()
                notification_h = line_h * 3 + 12
                notification_rect = pygame.Rect(
                    notification_x, y_start, notification_w, notification_h
                )

            next_y = run_btn_rect.bottom + 10  # Use run_btn_rect
        # --- Status Text ---
        status_text = f"Status: {status}"
        if app_state == "Playing":
            status_text = "Status: Playing Demo"
        elif app_state != "MainMenu":
            status_text = f"Status: {app_state}"

        status_font = self.fonts.get("status")
        if status_font:
            status_surf = status_font.render(status_text, True, VisConfig.YELLOW)
            status_rect_top = next_y if app_state == "MainMenu" else y_start
            status_rect = status_surf.get_rect(topleft=(10, status_rect_top))
            self.screen.blit(status_surf, status_rect)
            if app_state == "MainMenu":
                stat_rects["Status"] = status_rect
            next_y = status_rect.bottom + 5
        else:
            next_y += 20

        return next_y, stat_rects, notification_rect
