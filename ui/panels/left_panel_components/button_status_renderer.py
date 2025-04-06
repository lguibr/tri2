# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
from typing import Dict, Tuple, Optional, Any

# --- MODIFIED: Import constants ---
from config import (
    VisConfig,
    TOTAL_TRAINING_STEPS,
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    LIGHTG,
    BLUE,
)

# --- END MODIFIED ---


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
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))

    def _draw_button(self, rect: pygame.Rect, text: str, color: Tuple[int, int, int]):
        """Helper to draw a single button."""
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        if self.ui_font:
            label_surface = self.ui_font.render(text, True, WHITE)
            self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        else:  # Fallback if font failed
            pygame.draw.line(self.screen, RED, rect.topleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, RED, rect.topright, rect.bottomleft, 2)

    def _render_compact_status(
        self, y_start: int, panel_width: int, status: str, stats_summary: Dict[str, Any]
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the compact status block below buttons."""
        stat_rects: Dict[str, pygame.Rect] = {}
        x_margin = 10
        line_height_status = self.status_font.get_linesize()
        line_height_label = self.status_label_font.get_linesize()
        current_y = y_start

        # Line 1: Status Text
        status_text = f"Status: {status}"
        status_color = YELLOW  # Default
        if status == "Error":
            status_color = RED
        elif status == "Collecting Experience":
            status_color = GOOGLE_COLORS[0]  # Green
        elif status == "Updating Agent":
            status_color = GOOGLE_COLORS[2]  # Blue
        elif status == "Ready":
            status_color = WHITE

        status_surface = self.status_font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(status_surface, status_rect)
        stat_rects["Status"] = status_rect
        current_y += line_height_status

        # Line 2: Steps | Episodes | SPS
        global_step = stats_summary.get("global_step", 0)
        total_episodes = stats_summary.get("total_episodes", 0)
        sps = stats_summary.get("steps_per_second", 0.0)

        steps_str = f"{global_step/1e6:.2f}M/{TOTAL_TRAINING_STEPS/1e6:.1f}M Steps"
        eps_str = f"{total_episodes} Eps"
        sps_str = f"{sps:.0f} SPS"
        line2_text = f"{steps_str}  |  {eps_str}  |  {sps_str}"

        line2_surface = self.status_label_font.render(line2_text, True, LIGHTG)
        line2_rect = line2_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(line2_surface, line2_rect)

        # Add individual rects for tooltips on line 2 elements
        steps_surface = self.status_label_font.render(steps_str, True, LIGHTG)
        eps_surface = self.status_label_font.render(eps_str, True, LIGHTG)
        sps_surface = self.status_label_font.render(sps_str, True, LIGHTG)
        steps_rect = steps_surface.get_rect(topleft=(x_margin, current_y))
        eps_rect = eps_surface.get_rect(
            midleft=(steps_rect.right + 10, steps_rect.centery)
        )
        sps_rect = sps_surface.get_rect(midleft=(eps_rect.right + 10, eps_rect.centery))
        stat_rects["Steps Info"] = steps_rect
        stat_rects["Episodes Info"] = eps_rect
        stat_rects["SPS Info"] = sps_rect

        current_y += line_height_label + 2  # Add padding below line 2

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
        background_rect = pygame.Rect(x_margin, current_y, bar_width, bar_height)
        pygame.draw.rect(
            self.screen, (60, 60, 80), background_rect, border_radius=3
        )  # Darker blue bg

        # Foreground (Progress)
        progress_width = int(bar_width * progress)
        if progress_width > 0:
            foreground_rect = pygame.Rect(
                x_margin, current_y, progress_width, bar_height
            )
            pygame.draw.rect(
                self.screen, GOOGLE_COLORS[2], foreground_rect, border_radius=3
            )  # Blue progress

        # Border
        pygame.draw.rect(self.screen, LIGHTG, background_rect, 1, border_radius=3)

        # Percentage Text
        if self.progress_font:
            progress_text = f"{progress:.0%}"
            text_surface = self.progress_font.render(progress_text, True, WHITE)
            text_rect = text_surface.get_rect(center=background_rect.center)
            self.screen.blit(text_surface, text_rect)

        stat_rects["Update Progress"] = background_rect  # Tooltip for the whole bar
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
        update_progress: float,
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders buttons, status, and progress bar. Returns next_y, stat_rects."""
        stat_rects: Dict[str, pygame.Rect] = {}
        next_y = y_start

        # Render Buttons (only in MainMenu)
        if app_state == "MainMenu":
            button_height = 40
            button_y_pos = y_start
            run_button_width = 100
            cleanup_button_width = 160
            demo_button_width = 120
            button_spacing = 10

            run_button_rect = pygame.Rect(
                button_spacing, button_y_pos, run_button_width, button_height
            )
            cleanup_button_rect = pygame.Rect(
                run_button_rect.right + button_spacing,
                button_y_pos,
                cleanup_button_width,
                button_height,
            )
            demo_button_rect = pygame.Rect(
                cleanup_button_rect.right + button_spacing,
                button_y_pos,
                demo_button_width,
                button_height,
            )

            # Determine Run button text and color based on state
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

            self._draw_button(run_button_rect, run_button_text, run_button_color)
            self._draw_button(
                cleanup_button_rect, "Cleanup This Run", (100, 40, 40)
            )  # Red
            self._draw_button(demo_button_rect, "Play Demo", (40, 100, 40))  # Green

            stat_rects["Run Button"] = run_button_rect
            stat_rects["Cleanup Button"] = cleanup_button_rect
            stat_rects["Play Demo Button"] = demo_button_rect

            next_y = run_button_rect.bottom + 10  # Position below buttons

        # Render Compact Status Block
        status_block_y = next_y if app_state == "MainMenu" else y_start
        next_y, status_rects = self._render_compact_status(
            status_block_y, panel_width, status, stats_summary
        )
        stat_rects.update(status_rects)

        # Render Progress Bar (only if updating)
        if status == "Updating Agent" and update_progress >= 0:
            next_y, progress_rects = self._render_progress_bar(
                next_y, panel_width, update_progress
            )
            stat_rects.update(progress_rects)

        return next_y, stat_rects
