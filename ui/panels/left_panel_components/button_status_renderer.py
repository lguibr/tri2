import pygame
from typing import Dict, Tuple, Any

from config import (
    TOTAL_TRAINING_STEPS,
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    LIGHTG,
)


class ButtonStatusRenderer:
    """Renders the top buttons, compact status block, and update progress bar(s)."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("status", pygame.font.Font(None, 28))
        self.status_label_font = fonts.get(
            "notification_label", pygame.font.Font(None, 16)
        )
        self.progress_font = fonts.get("progress_bar", pygame.font.Font(None, 14))
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        # Store reference to input handler to get button rects (set externally if needed)
        self.input_handler_ref = None

    def _draw_button(
        self,
        rect: pygame.Rect,
        text: str,
        color: Tuple[int, int, int],
        enabled: bool = True,
    ):
        """Helper to draw a single button, optionally grayed out."""
        if enabled:
            final_color = color
        else:
            if isinstance(color, tuple) and len(color) >= 3:
                final_color = tuple(max(30, c // 2) for c in color[:3])
            else:
                final_color = (50, 50, 50)

        pygame.draw.rect(self.screen, final_color, rect, border_radius=5)
        text_color = WHITE if enabled else (150, 150, 150)
        if self.ui_font:
            label_surface = self.ui_font.render(text, True, text_color)
            self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        else:
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
        if "Error" in status:
            status_color = RED
        elif "Collecting" in status:
            status_color = GOOGLE_COLORS[0]  # Green
        elif "Updating" in status:
            status_color = GOOGLE_COLORS[2]  # Blue
        elif "Ready" in status:
            status_color = WHITE
        elif "Debugging" in status:  # Added Debug color
            status_color = (200, 100, 200)  # Magenta-ish
        elif "Initializing" in status:
            status_color = LIGHTG

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
        # Add individual rects for tooltips
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
        current_y += line_height_label + 2

        return current_y, stat_rects

    def _render_single_progress_bar(
        self,
        y_pos: int,
        panel_width: int,
        progress: float,
        bar_color: Tuple[int, int, int],
        text_prefix: str = "",
    ) -> pygame.Rect:
        """Renders a single progress bar component."""
        x_margin = 10
        bar_height = 16
        bar_width = panel_width - 2 * x_margin
        if bar_width <= 0:
            return pygame.Rect(x_margin, y_pos, 0, 0)

        background_rect = pygame.Rect(x_margin, y_pos, bar_width, bar_height)
        pygame.draw.rect(self.screen, (60, 60, 80), background_rect, border_radius=3)
        clamped_progress = max(0.0, min(1.0, progress))
        progress_width = int(bar_width * clamped_progress)
        if progress_width > 0:
            foreground_rect = pygame.Rect(x_margin, y_pos, progress_width, bar_height)
            pygame.draw.rect(self.screen, bar_color, foreground_rect, border_radius=3)
        pygame.draw.rect(self.screen, LIGHTG, background_rect, 1, border_radius=3)

        if self.progress_font:
            progress_text_str = f"{text_prefix}{clamped_progress:.0%}"
            text_surface = self.progress_font.render(progress_text_str, True, WHITE)
            text_rect = text_surface.get_rect(center=background_rect.center)
            self.screen.blit(text_surface, text_rect)
        return background_rect

    def _render_detailed_progress_bars(
        self,
        y_start: int,
        panel_width: int,
        progress_details: Dict[str, Any],
        bar_color: Tuple[int, int, int],
        tooltip_key_prefix: str = "Update",
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders two progress bars (overall, epoch) and epoch text."""
        stat_rects: Dict[str, pygame.Rect] = {}
        x_margin = 10
        bar_height = 16
        text_height = self.progress_font.get_linesize() if self.progress_font else 14
        bar_spacing = 2
        current_y = y_start

        overall_progress = progress_details.get("overall_progress", 0.0)
        epoch_progress = progress_details.get("epoch_progress", 0.0)
        current_epoch = progress_details.get("current_epoch", 0)
        total_epochs = progress_details.get("total_epochs", 0)

        bar_width = panel_width - 2 * x_margin
        if bar_width <= 0:
            return current_y, stat_rects

        # 1. Epoch Text
        epoch_text = f"Epoch {current_epoch}/{total_epochs}"
        if self.progress_font:
            text_surface = self.progress_font.render(epoch_text, True, WHITE)
            text_rect = text_surface.get_rect(topleft=(x_margin, current_y))
            self.screen.blit(text_surface, text_rect)
            current_y += text_height + 2
            stat_rects[f"{tooltip_key_prefix} Epoch Info"] = text_rect

        # 2. Epoch Progress Bar
        epoch_bar_rect = self._render_single_progress_bar(
            current_y, panel_width, epoch_progress, bar_color, "Epoch: "
        )
        stat_rects[f"{tooltip_key_prefix} Epoch Progress"] = epoch_bar_rect
        current_y += bar_height + bar_spacing

        # 3. Overall Progress Bar
        overall_bar_color = tuple(min(255, max(0, int(c * 0.7))) for c in bar_color[:3])
        overall_bar_rect = self._render_single_progress_bar(
            current_y, panel_width, overall_progress, overall_bar_color, "Overall: "
        )
        stat_rects[f"{tooltip_key_prefix} Overall Progress"] = overall_bar_rect
        current_y += bar_height + 5

        return current_y, stat_rects

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        update_progress_details: Dict[str, Any],
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders buttons, status, and progress bar(s). Returns next_y, stat_rects."""
        stat_rects: Dict[str, pygame.Rect] = {}
        next_y = y_start

        buttons_enabled = app_state == "MainMenu" and not is_process_running
        run_stop_enabled = app_state == "MainMenu"

        button_height = 40
        button_y_pos = y_start
        run_button_width = 100
        cleanup_button_width = 160
        demo_button_width = 120
        debug_button_width = 120  # Added width
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
        debug_button_rect = pygame.Rect(  # Added rect calculation
            demo_button_rect.right + button_spacing,
            button_y_pos,
            debug_button_width,
            button_height,
        )

        run_button_text = "Run"
        run_button_color = (40, 40, 80)
        if is_process_running:
            run_button_text = "Stop"
            if "Collecting" in status:
                run_button_color = (40, 80, 40)
            elif "Updating" in status:
                run_button_color = (40, 40, 80)
            else:
                run_button_color = (80, 40, 40)

        self._draw_button(
            run_button_rect, run_button_text, run_button_color, enabled=run_stop_enabled
        )
        self._draw_button(
            cleanup_button_rect,
            "Cleanup This Run",
            (100, 40, 40),
            enabled=buttons_enabled,
        )
        self._draw_button(
            demo_button_rect, "Play Demo", (40, 100, 40), enabled=buttons_enabled
        )
        # Draw the Debug button
        self._draw_button(
            debug_button_rect, "Debug Mode", (100, 40, 100), enabled=buttons_enabled
        )

        # Store rects for tooltips (these might be overwritten by LeftPanelRenderer if using InputHandler ref)
        stat_rects["Run Button"] = run_button_rect
        stat_rects["Cleanup Button"] = cleanup_button_rect
        stat_rects["Play Demo Button"] = demo_button_rect
        stat_rects["Debug Mode Button"] = debug_button_rect  # Added Debug button rect

        next_y = run_button_rect.bottom + 10

        # Render Compact Status Block
        status_block_y = next_y
        next_y, status_rects = self._render_compact_status(
            status_block_y, panel_width, status, stats_summary
        )
        stat_rects.update(status_rects)

        # Render Progress Bars if Updating
        update_phase = update_progress_details.get("phase")
        if update_phase == "Train Update":
            next_y, progress_rects = self._render_detailed_progress_bars(
                next_y,
                panel_width,
                update_progress_details,
                GOOGLE_COLORS[2],
                tooltip_key_prefix="Update",
            )
            stat_rects.update(progress_rects)

        return next_y, stat_rects
