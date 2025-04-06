# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
from typing import Dict, Tuple, Optional, Any
from config import VisConfig, TOTAL_TRAINING_STEPS


class ButtonStatusRenderer:
    """Renders the top buttons, compact status block, and update progress bar."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("status", pygame.font.Font(None, 28))
        self.status_label_font = fonts.get(
            "notification_label", pygame.font.Font(None, 16)
        )
        self.progress_font = fonts.get("progress_bar", pygame.font.Font(None, 14))

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

    def _render_compact_status(
        self, y_start: int, panel_width: int, status: str, stats_summary: Dict[str, Any]
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the compact status block."""
        stat_rects: Dict[str, pygame.Rect] = {}
        x_margin = 10
        line_height = self.status_font.get_linesize()
        label_line_height = self.status_label_font.get_linesize()
        current_y = y_start

        # Line 1: Status
        status_text = f"Status: {status}"
        status_color = VisConfig.YELLOW
        if status == "Error":
            status_color = VisConfig.RED
        elif status == "Collecting Experience":
            status_color = VisConfig.GOOGLE_COLORS[0]  # Green
        elif status == "Updating Agent":
            status_color = VisConfig.GOOGLE_COLORS[2]  # Blue
        elif status == "Ready":
            status_color = VisConfig.WHITE

        status_surf = self.status_font.render(status_text, True, status_color)
        status_rect = status_surf.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(status_surf, status_rect)
        stat_rects["Status"] = status_rect
        current_y += line_height

        # Line 2: Steps | Episodes | SPS
        global_step = stats_summary.get("global_step", 0)
        total_episodes = stats_summary.get("total_episodes", 0)
        sps = stats_summary.get("steps_per_second", 0.0)

        steps_str = f"{global_step/1e6:.2f}M/{TOTAL_TRAINING_STEPS/1e6:.1f}M Steps"
        eps_str = f"{total_episodes} Eps"
        sps_str = f"{sps:.0f} SPS"

        line2_text = f"{steps_str}  |  {eps_str}  |  {sps_str}"
        line2_surf = self.status_label_font.render(line2_text, True, VisConfig.LIGHTG)
        line2_rect = line2_surf.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(line2_surf, line2_rect)

        # Add individual rects for tooltips on line 2 elements
        steps_surf = self.status_label_font.render(steps_str, True, VisConfig.LIGHTG)
        eps_surf = self.status_label_font.render(eps_str, True, VisConfig.LIGHTG)
        sps_surf = self.status_label_font.render(sps_str, True, VisConfig.LIGHTG)

        steps_rect = steps_surf.get_rect(topleft=(x_margin, current_y))
        eps_rect = eps_surf.get_rect(
            midleft=(steps_rect.right + 10, steps_rect.centery)
        )
        sps_rect = sps_surf.get_rect(midleft=(eps_rect.right + 10, eps_rect.centery))

        stat_rects["Steps Info"] = steps_rect
        stat_rects["Episodes Info"] = eps_rect
        stat_rects["SPS Info"] = sps_rect

        current_y += label_line_height + 2

        return current_y, stat_rects

    def _render_progress_bar(
        self, y_start: int, panel_width: int, progress: float
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the agent update progress bar."""
        stat_rects: Dict[str, pygame.Rect] = {}
        x_margin = 10
        bar_height = 18
        bar_width = panel_width - 2 * x_margin
        current_y = y_start

        if bar_width <= 0:
            return current_y, stat_rects

        # Background
        bg_rect = pygame.Rect(x_margin, current_y, bar_width, bar_height)
        pygame.draw.rect(
            self.screen, (60, 60, 80), bg_rect, border_radius=3
        )  # Darker blue bg

        # Foreground (Progress)
        progress_width = int(bar_width * progress)
        if progress_width > 0:
            fg_rect = pygame.Rect(x_margin, current_y, progress_width, bar_height)
            pygame.draw.rect(
                self.screen, VisConfig.GOOGLE_COLORS[2], fg_rect, border_radius=3
            )  # Blue progress

        # Border
        pygame.draw.rect(self.screen, VisConfig.LIGHTG, bg_rect, 1, border_radius=3)

        # Percentage Text
        if self.progress_font:
            progress_text = f"{progress:.0%}"
            text_surf = self.progress_font.render(progress_text, True, VisConfig.WHITE)
            text_rect = text_surf.get_rect(center=bg_rect.center)
            self.screen.blit(text_surf, text_rect)

        stat_rects["Update Progress"] = bg_rect  # Tooltip for the whole bar
        current_y += bar_height + 5  # Add padding below bar

        return current_y, stat_rects

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_training_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        update_progress: float,  # Added update_progress
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders buttons, status, and progress bar. Returns next_y, stat_rects."""
        stat_rects: Dict[str, pygame.Rect] = {}
        next_y = y_start

        # Render Buttons (only in MainMenu)
        if app_state == "MainMenu":
            button_h = 40
            button_y = y_start
            run_btn_w = 100
            cleanup_btn_w = 160
            demo_btn_w = 120
            spacing = 10

            run_btn_rect = pygame.Rect(spacing, button_y, run_btn_w, button_h)
            cleanup_btn_rect = pygame.Rect(
                run_btn_rect.right + spacing, button_y, cleanup_btn_w, button_h
            )
            demo_btn_rect = pygame.Rect(
                cleanup_btn_rect.right + spacing, button_y, demo_btn_w, button_h
            )

            run_button_text = "Run"
            run_button_color = (70, 70, 70)  # Default gray
            if is_training_running:
                run_button_text = "Stop"
                if status == "Collecting Experience":
                    run_button_color = (40, 80, 40)  # Green
                elif status == "Updating Agent":
                    run_button_color = (40, 40, 80)  # Blue
                else:
                    run_button_color = (80, 80, 40)  # Yellowish if other training state
            elif status == "Ready":
                run_button_color = (40, 40, 80)  # Ready blue

            self._draw_button(run_btn_rect, run_button_text, run_button_color)
            self._draw_button(cleanup_btn_rect, "Cleanup This Run", (100, 40, 40))
            self._draw_button(demo_btn_rect, "Play Demo", (40, 100, 40))

            stat_rects["Run Button"] = run_btn_rect
            stat_rects["Cleanup Button"] = cleanup_btn_rect
            stat_rects["Play Demo Button"] = demo_btn_rect

            next_y = run_btn_rect.bottom + 10

        # Render Compact Status Block
        status_y = next_y if app_state == "MainMenu" else y_start
        next_y, status_rects = self._render_compact_status(
            status_y, panel_width, status, stats_summary
        )
        stat_rects.update(status_rects)

        # Render Progress Bar (only if updating)
        if status == "Updating Agent" and update_progress >= 0:
            next_y, progress_rects = self._render_progress_bar(
                next_y, panel_width, update_progress
            )
            stat_rects.update(progress_rects)

        return next_y, stat_rects
