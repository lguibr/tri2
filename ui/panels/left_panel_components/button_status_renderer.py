# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
import time
import math
from typing import Dict, Tuple, Any, Optional, TYPE_CHECKING

from config import WHITE, YELLOW, RED, GOOGLE_COLORS, LIGHTG, GREEN, DARK_GREEN
from utils.helpers import format_eta
from ui.input_handler import InputHandler

if TYPE_CHECKING:
    from main_pygame import MainApp


class ButtonStatusRenderer:
    """Renders the top buttons and compact status block."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("status", pygame.font.Font(None, 28))
        self.status_label_font = fonts.get(
            "notification_label", pygame.font.Font(None, 16)
        )
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.input_handler_ref: Optional[InputHandler] = None
        self.app_ref: Optional["MainApp"] = None

    def _draw_button(
        self,
        rect: pygame.Rect,
        text: str,
        base_color: Tuple[int, int, int],
        active_color: Optional[Tuple[int, int, int]] = None,
        is_active: bool = False,
        enabled: bool = True,
    ):
        """Helper to draw a single button, optionally grayed out or in active state."""
        final_color = base_color
        if not enabled:
            final_color = tuple(max(30, c // 2) for c in base_color[:3])
        elif is_active and active_color:
            final_color = active_color

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
    ) -> int:
        """Renders the compact status block below buttons."""
        x_margin, current_y = 10, y_start
        line_height_status = self.status_font.get_linesize()
        line_height_label = self.status_label_font.get_linesize()

        status_text = f"Status: {status}"
        status_color = YELLOW
        if "Error" in status:
            status_color = RED
        elif "Ready" in status:
            status_color = WHITE
        elif "Debugging" in status:
            status_color = (200, 100, 200)
        elif "Playing" in status:
            status_color = (100, 150, 200)
        elif "Initializing" in status:
            status_color = LIGHTG
        elif "Cleaning" in status:
            status_color = (200, 100, 100)
        elif "Confirm" in status:
            status_color = (200, 150, 50)
        elif "Running AlphaZero" in status:
            status_color = GREEN

        try:
            status_surface = self.status_font.render(status_text, True, status_color)
            status_rect = status_surface.get_rect(topleft=(x_margin, current_y))
            self.screen.blit(status_surface, status_rect)
            current_y += line_height_status
        except Exception as e:
            print(f"Error rendering status text: {e}")
            current_y += 20  # Fallback increment

        global_step = stats_summary.get("global_step", 0)
        total_episodes = stats_summary.get("total_episodes", 0)

        global_step_str = f"{global_step:,}".replace(",", "_")
        eps_str = f"~{total_episodes} Eps"

        line2_text = f"{global_step_str} Steps | {eps_str}"
        try:
            line2_surface = self.status_label_font.render(line2_text, True, LIGHTG)
            line2_rect = line2_surface.get_rect(topleft=(x_margin, current_y))
            self.screen.blit(line2_surface, line2_rect)
            current_y += line_height_label + 2
        except Exception as e:
            print(f"Error rendering step/ep text: {e}")
            current_y += 15  # Fallback increment

        return int(current_y)  # Ensure return is int

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        update_progress_details: Dict[str, Any],
    ) -> int:  # Ensure return type hint is int
        """Renders buttons and status. Returns next_y."""
        from app_state import AppState

        next_y = y_start  # Start with an int

        # Get button rects from InputHandler
        run_stop_btn_rect = (
            self.input_handler_ref.run_stop_btn_rect
            if self.input_handler_ref
            else pygame.Rect(0, 0, 0, 0)
        )
        cleanup_btn_rect = (
            self.input_handler_ref.cleanup_btn_rect
            if self.input_handler_ref
            else pygame.Rect(0, 0, 0, 0)
        )
        demo_btn_rect = (
            self.input_handler_ref.demo_btn_rect
            if self.input_handler_ref
            else pygame.Rect(0, 0, 0, 0)
        )
        debug_btn_rect = (
            self.input_handler_ref.debug_btn_rect
            if self.input_handler_ref
            else pygame.Rect(0, 0, 0, 0)
        )

        # Check worker status
        is_running = (
            self.app_ref.worker_manager.is_any_worker_running()
            if self.app_ref
            else False
        )

        # Render Combined Run/Stop Button
        run_stop_text = "Stop Run" if is_running else "Run AlphaZero"
        run_stop_base_color = (40, 80, 40)
        run_stop_active_color = (100, 40, 40)
        self._draw_button(
            run_stop_btn_rect,
            run_stop_text,
            run_stop_base_color,
            active_color=run_stop_active_color,
            is_active=is_running,
            enabled=(app_state == AppState.MAIN_MENU.value),
        )

        # Render Other Buttons
        other_buttons_enabled = (
            app_state == AppState.MAIN_MENU.value
        ) and not is_running
        self._draw_button(
            cleanup_btn_rect,
            "Cleanup This Run",
            (100, 40, 40),
            enabled=other_buttons_enabled,
        )
        self._draw_button(
            demo_btn_rect, "Play Demo", (40, 100, 40), enabled=other_buttons_enabled
        )
        self._draw_button(
            debug_btn_rect, "Debug Mode", (100, 40, 100), enabled=other_buttons_enabled
        )

        button_bottom = max(
            run_stop_btn_rect.bottom,
            cleanup_btn_rect.bottom,
            demo_btn_rect.bottom,
            debug_btn_rect.bottom,
        )
        # Ensure button_bottom is an int before adding
        next_y = int(button_bottom) + 10

        # Render Status Block
        status_block_y = next_y
        next_y = self._render_compact_status(  # This now returns int
            status_block_y, panel_width, status, stats_summary
        )

        return int(
            next_y
        )  # Explicitly cast just in case, though _render_compact_status should return int
