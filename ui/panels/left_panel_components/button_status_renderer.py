# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
import time
import math
from typing import Dict, Tuple, Any, Optional

from config import WHITE, YELLOW, RED, GOOGLE_COLORS, LIGHTG
from utils.helpers import format_eta
from ui.input_handler import InputHandler


class ButtonStatusRenderer:
    """Renders the top buttons (excluding Run/Stop), and compact status block."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("status", pygame.font.Font(None, 28))
        self.status_label_font = fonts.get(
            "notification_label", pygame.font.Font(None, 16)
        )
        # Removed self.progress_font
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.input_handler_ref: Optional[InputHandler] = None

    def _draw_button(
        self,
        rect: pygame.Rect,
        text: str,
        color: Tuple[int, int, int],
        enabled: bool = True,
    ):
        """Helper to draw a single button, optionally grayed out."""
        final_color = color if enabled else tuple(max(30, c // 2) for c in color[:3])
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

        status_surface = self.status_font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(status_surface, status_rect)
        current_y += line_height_status

        # Display basic info like total episodes/games played
        global_step = stats_summary.get(
            "global_step", 0
        )  # Step might mean games or NN steps
        total_episodes = stats_summary.get("total_episodes", 0)

        global_step_str = f"{global_step:,}".replace(",", "_")
        eps_str = f"~{total_episodes} Eps"  # Or Games

        line2_text = f"{global_step_str} Steps | {eps_str}"  # Simplified info line
        line2_surface = self.status_label_font.render(line2_text, True, LIGHTG)
        line2_rect = line2_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(line2_surface, line2_rect)

        current_y += line_height_label + 2
        return current_y

    # Removed _render_single_progress_bar
    # Removed _render_detailed_progress_bars

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_process_running: bool,  # Keep for potential future use (e.g., MCTS running)
        status: str,
        stats_summary: Dict[str, Any],
        update_progress_details: Dict[str, Any],  # Keep for potential NN progress
    ) -> int:
        """Renders buttons (excluding Run/Stop) and status. Returns next_y."""
        from app_state import AppState

        next_y = y_start

        # Get button rects from InputHandler
        # Removed run_btn_rect
        cleanup_btn_rect = (
            self.input_handler_ref.cleanup_btn_rect
            if self.input_handler_ref
            else pygame.Rect(10, y_start, 160, 40)
        )
        demo_btn_rect = (
            self.input_handler_ref.demo_btn_rect
            if self.input_handler_ref
            else pygame.Rect(cleanup_btn_rect.right + 10, y_start, 120, 40)
        )
        debug_btn_rect = (
            self.input_handler_ref.debug_btn_rect
            if self.input_handler_ref
            else pygame.Rect(demo_btn_rect.right + 10, y_start, 120, 40)
        )

        # Removed Run/Stop button rendering

        # Render other buttons
        buttons_enabled = app_state == AppState.MAIN_MENU.value
        self._draw_button(
            cleanup_btn_rect, "Cleanup This Run", (100, 40, 40), enabled=buttons_enabled
        )
        self._draw_button(
            demo_btn_rect, "Play Demo", (40, 100, 40), enabled=buttons_enabled
        )
        self._draw_button(
            debug_btn_rect, "Debug Mode", (100, 40, 100), enabled=buttons_enabled
        )
        # Set next_y below the buttons
        button_bottom = max(
            cleanup_btn_rect.bottom, demo_btn_rect.bottom, debug_btn_rect.bottom
        )
        next_y = button_bottom + 10

        # Render Status Block
        status_block_y = next_y
        # Removed check for "Updating Agent" status to render progress bars
        # Always render the compact status block now
        next_y = self._render_compact_status(
            status_block_y, panel_width, status, stats_summary
        )

        # Render NN training progress bar if applicable (using update_progress_details)
        # Example placeholder:
        # if update_progress_details and update_progress_details.get('phase') == 'Training NN':
        #     next_y = self._render_detailed_progress_bars(...) # Adapt this call

        return next_y
