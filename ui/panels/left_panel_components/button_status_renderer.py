# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
import time
import math
from typing import Dict, Tuple, Any, Optional, TYPE_CHECKING

from config import (
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    LIGHTG,
    GREEN,
    DARK_GREEN,
    BLUE,
    GRAY,
)

# TrainConfig might not be needed if min_buffer is passed in render_data
# from config.core import TrainConfig
from utils.helpers import format_eta
from ui.input_handler import InputHandler

if TYPE_CHECKING:
    # from main_pygame import MainApp # Avoid direct import
    pass


class ButtonStatusRenderer:
    """Renders the top buttons and compact status block based on provided data."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("status", pygame.font.Font(None, 28))
        self.detail_font = fonts.get("detail", pygame.font.Font(None, 16))
        self.progress_font = fonts.get("detail", pygame.font.Font(None, 14))
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.input_handler_ref: Optional[InputHandler] = None
        # self.app_ref: Optional["MainApp"] = None # Less relevant now
        # self.train_config = TrainConfig() # Get min_buffer from render_data

    # _draw_button remains the same
    def _draw_button(
        self,
        rect: pygame.Rect,
        text: str,
        base_color: Tuple[int, int, int],
        active_color: Optional[Tuple[int, int, int]] = None,
        is_active: bool = False,
        enabled: bool = True,
    ):
        final_color = base_color
        if not enabled:
            final_color = GRAY
        elif is_active and active_color:
            final_color = active_color
        pygame.draw.rect(self.screen, final_color, rect, border_radius=5)
        text_color = WHITE if enabled else (150, 150, 150)
        if self.ui_font:
            label_surface = self.ui_font.render(text, True, text_color)
            self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        else:  # Fallback
            pygame.draw.line(self.screen, RED, rect.topleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, RED, rect.topright, rect.bottomleft, 2)

    # _render_progress_bar remains the same
    def _render_progress_bar(
        self,
        y_pos: int,
        panel_width: int,
        current_value: int,
        target_value: int,
        label: str,
    ) -> int:
        if not self.progress_font:
            return y_pos
        bar_height = 18
        bar_width = panel_width * 0.8
        bar_x = (panel_width - bar_width) / 2
        bar_rect = pygame.Rect(bar_x, y_pos, bar_width, bar_height)
        progress = 0.0
        if target_value > 0:
            progress = min(1.0, max(0.0, current_value / target_value))
        bg_color, border_color = (50, 50, 50), LIGHTG
        pygame.draw.rect(self.screen, bg_color, bar_rect, border_radius=3)
        fill_width = int(bar_width * progress)
        fill_rect = pygame.Rect(bar_x, y_pos, fill_width, bar_height)
        fill_color = BLUE
        pygame.draw.rect(
            self.screen,
            fill_color,
            fill_rect,
            border_top_left_radius=3,
            border_bottom_left_radius=3,
            border_top_right_radius=3 if progress >= 1.0 else 0,
            border_bottom_right_radius=3 if progress >= 1.0 else 0,
        )
        pygame.draw.rect(self.screen, border_color, bar_rect, 1, border_radius=3)
        progress_text = f"{label}: {current_value:,}/{target_value:,}".replace(",", "_")
        text_surf = self.progress_font.render(progress_text, True, WHITE)
        text_rect = text_surf.get_rect(center=bar_rect.center)
        self.screen.blit(text_surf, text_rect)
        return int(bar_rect.bottom)

    # _render_compact_status remains the same
    def _render_compact_status(
        self,
        y_start: int,
        panel_width: int,
        status: str,
        stats_summary: Dict[str, Any],
        is_running: bool,
        min_buffer: int,
    ) -> int:
        x_margin, current_y = 10, y_start
        line_height_status = self.status_font.get_linesize()
        line_height_detail = self.detail_font.get_linesize()
        # 1. Render Status Text
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
            current_y += 20
        # 2. Render Global Step/Eps OR Buffering Progress
        global_step = stats_summary.get("global_step", 0)
        total_episodes = stats_summary.get("total_episodes", 0)
        buffer_size = stats_summary.get("buffer_size", 0)
        is_buffering = is_running and global_step == 0 and buffer_size < min_buffer
        if not is_buffering:
            global_step_str = f"{global_step:,}".replace(",", "_")
            eps_str = f"{total_episodes:,}".replace(",", "_")
            line2_text = f"Step: {global_step_str} | Episodes: {eps_str}"
            try:
                line2_surface = self.detail_font.render(line2_text, True, LIGHTG)
                line2_rect = line2_surface.get_rect(topleft=(x_margin, current_y))
                clip_width = max(0, panel_width - line2_rect.left - x_margin)
                blit_area = (
                    pygame.Rect(0, 0, clip_width, line2_rect.height)
                    if line2_rect.width > clip_width
                    else None
                )
                self.screen.blit(line2_surface, line2_rect, area=blit_area)
                current_y += line_height_detail + 2
            except Exception as e:
                print(f"Error rendering step/ep text: {e}")
                current_y += 15
        else:
            current_y += 2
            next_y_after_bar = self._render_progress_bar(
                current_y, panel_width, buffer_size, min_buffer, "Buffering"
            )
            current_y = next_y_after_bar + 5
        return int(current_y)

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        update_progress_details: Dict[
            str, Any
        ],  # Keep if used by progress bar logic later
    ) -> int:
        """Renders buttons and status based on provided data. Returns next_y."""
        from app_state import AppState  # Local import

        next_y = y_start
        is_running = is_process_running  # Use the passed flag

        # Get button rects from the input handler (which should have up-to-date rects)
        run_stop_btn_rect = (
            self.input_handler_ref.run_stop_btn_rect
            if self.input_handler_ref
            else pygame.Rect(10, y_start, 150, 40)
        )
        cleanup_btn_rect = (
            self.input_handler_ref.cleanup_btn_rect
            if self.input_handler_ref
            else pygame.Rect(run_stop_btn_rect.right + 10, y_start, 160, 40)
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

        # Render Buttons
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
        next_y = int(button_bottom) + 10

        # Render Status Block
        status_block_y = next_y
        min_buffer = stats_summary.get(
            "min_buffer_size", 1000
        )  # Get min buffer from summary if available
        next_y = self._render_compact_status(
            status_block_y, panel_width, status, stats_summary, is_running, min_buffer
        )

        return int(next_y)
