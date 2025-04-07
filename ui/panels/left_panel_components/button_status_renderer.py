# File: ui/panels/left_panel_components/button_status_renderer.py
# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
import time
from typing import Dict, Tuple, Any

from config import (
    # TOTAL_TRAINING_STEPS, # Removed import, use dynamic target
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    LIGHTG,
)
from utils.helpers import format_eta


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
        elif "Debugging" in status:
            status_color = (200, 100, 200)  # Magenta-ish
        elif "Initializing" in status:
            status_color = LIGHTG
        elif "Training Complete" in status:  # Added color for complete
            status_color = (100, 200, 100)  # Light Green

        status_surface = self.status_font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(status_surface, status_rect)
        stat_rects["Status"] = status_rect
        current_y += line_height_status

        # Line 2: Steps | Episodes | SPS
        global_step = stats_summary.get("global_step", 0)
        # Get the dynamic target step from the summary
        training_target_step = stats_summary.get("training_target_step", 0)
        total_episodes = stats_summary.get("total_episodes", 0)
        sps = stats_summary.get("steps_per_second", 0.0)

        # Display steps relative to the dynamic target
        if training_target_step > 0:
            steps_str = f"{global_step/1e6:.2f}M/{training_target_step/1e6:.1f}M Steps"
        else:  # Fallback if target is 0 or not set
            steps_str = f"{global_step/1e6:.2f}M Steps"

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
        bar_height: int = 16,
        x_offset: int = 10,
        custom_text: str = "",
    ) -> pygame.Rect:
        """Renders a single progress bar component."""
        x_margin = x_offset
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
            progress_text_str = (
                custom_text if custom_text else f"{text_prefix}{clamped_progress:.0%}"
            )
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
        """Renders two progress bars (overall, epoch) and epoch text, including ETAs."""
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
        update_start_time = progress_details.get("update_start_time", 0.0)
        num_minibatches_per_epoch = progress_details.get("num_minibatches_per_epoch", 0)
        current_minibatch_index = progress_details.get("current_minibatch_index", 0)

        bar_width = panel_width - 2 * x_margin
        if bar_width <= 0:
            return current_y, stat_rects

        # --- Calculate ETAs ---
        overall_eta_str = "N/A"
        epoch_eta_str = "N/A"
        time_elapsed = time.time() - update_start_time if update_start_time > 0 else 0.0

        if time_elapsed > 1.0:
            if overall_progress > 1e-6:
                total_estimated_time = time_elapsed / overall_progress
                remaining_time_overall = max(0.0, total_estimated_time - time_elapsed)
                overall_eta_str = format_eta(remaining_time_overall)

            if epoch_progress > 1e-6 and num_minibatches_per_epoch > 0:
                # Estimate time per epoch based on overall progress through epochs
                # Avoid division by zero if current_epoch is 1 and epoch_progress is small
                effective_epochs_done = max(1e-6, (current_epoch - 1) + epoch_progress)
                time_per_epoch_estimate = time_elapsed / effective_epochs_done
                # Estimate remaining time for the current epoch
                remaining_epoch_fraction = 1.0 - epoch_progress
                remaining_time_epoch = max(
                    0.0, time_per_epoch_estimate * remaining_epoch_fraction
                )
                epoch_eta_str = format_eta(remaining_time_epoch)
            elif epoch_progress == 0.0:
                epoch_eta_str = "Starting..."

        # 1. Epoch Text
        epoch_text = f"Epoch {current_epoch}/{total_epochs}"
        if self.progress_font:
            text_surface = self.progress_font.render(epoch_text, True, WHITE)
            text_rect = text_surface.get_rect(topleft=(x_margin, current_y))
            self.screen.blit(text_surface, text_rect)
            current_y += text_height + 2
            stat_rects[f"{tooltip_key_prefix} Epoch Info"] = text_rect

        # 2. Epoch Progress Bar with ETA
        epoch_bar_text = f"Epoch: {epoch_progress:.0%} | ETA: {epoch_eta_str}"
        epoch_bar_rect = self._render_single_progress_bar(
            current_y,
            panel_width,
            epoch_progress,
            bar_color,
            custom_text=epoch_bar_text,
        )
        stat_rects[f"{tooltip_key_prefix} Epoch Progress"] = epoch_bar_rect
        current_y += bar_height + bar_spacing

        # 3. Overall Progress Bar with ETA
        overall_bar_color = tuple(min(255, max(0, int(c * 0.7))) for c in bar_color[:3])
        overall_bar_text = (
            f"Overall Update: {overall_progress:.0%} | ETA: {overall_eta_str}"
        )
        overall_bar_rect = self._render_single_progress_bar(
            current_y,
            panel_width,
            overall_progress,
            overall_bar_color,
            custom_text=overall_bar_text,
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

        button_height = 40
        button_y_pos = y_start
        run_button_width = 100
        button_spacing = 10

        # --- Always render Run/Stop button ---
        run_button_rect = pygame.Rect(
            button_spacing, button_y_pos, run_button_width, button_height
        )
        run_stop_enabled = app_state == "MainMenu"
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
        stat_rects["Run Button"] = run_button_rect
        current_x = run_button_rect.right + button_spacing

        # --- Conditionally render other buttons OR progress bar ---
        if is_process_running:
            # Render Training Progress Bar
            progress_bar_x = current_x
            progress_bar_width = panel_width - progress_bar_x - button_spacing
            progress_bar_height = button_height
            progress_bar_y = button_y_pos

            global_step = stats_summary.get("global_step", 0)
            # Get the dynamic target step from the summary
            training_target_step = stats_summary.get("training_target_step", 0)
            start_time = stats_summary.get("start_time", 0.0)
            progress = (
                global_step / max(1, training_target_step)
                if training_target_step > 0
                else 0.0
            )
            progress_bar_color = (20, 100, 20)  # Dark green

            # Determine text based on whether target is reached
            if global_step >= training_target_step and training_target_step > 0:
                progress_text = (
                    f"Target Reached ({global_step/1e6:.2f}M Steps) - Continuing..."
                )
                progress = 1.0  # Keep bar full
                progress_bar_color = (100, 150, 100)  # Lighter green when complete
            elif training_target_step > 0:
                eta_str = "N/A"
                time_elapsed = time.time() - start_time if start_time > 0 else 0.0
                if progress > 1e-6 and progress < 1.0 and time_elapsed > 1.0:
                    total_estimated_time = time_elapsed / progress
                    remaining_time = max(0.0, total_estimated_time - time_elapsed)
                    eta_str = format_eta(remaining_time)
                elif progress >= 1.0:
                    eta_str = "Done"
                else:
                    eta_str = "Calculating..." if global_step > 0 else "N/A"
                progress_text = f"{global_step/1e6:.2f}M / {training_target_step/1e6:.1f}M Steps ({progress:.1%}) | ETA: {eta_str}"
            else:  # Handle case where target step is 0 (e.g., before loading)
                progress_text = f"{global_step/1e6:.2f}M Steps (No Target)"
                progress = 0.0  # No progress if no target

            progress_rect = self._render_single_progress_bar(
                progress_bar_y,
                progress_bar_width + 2 * button_spacing,
                progress,  # Use potentially clamped progress for bar fill
                progress_bar_color,
                bar_height=progress_bar_height,
                x_offset=progress_bar_x,
                custom_text=progress_text,  # Use the determined text
            )
            stat_rects["Training Progress"] = progress_rect
            next_y = run_button_rect.bottom + 10

        else:
            # Render Cleanup, Demo, Debug buttons (only if not running and in MainMenu)
            buttons_enabled = app_state == "MainMenu"
            cleanup_button_width = 160
            demo_button_width = 120
            debug_button_width = 120

            cleanup_button_rect = pygame.Rect(
                current_x, button_y_pos, cleanup_button_width, button_height
            )
            self._draw_button(
                cleanup_button_rect,
                "Cleanup This Run",
                (100, 40, 40),
                enabled=buttons_enabled,
            )
            stat_rects["Cleanup Button"] = cleanup_button_rect
            current_x = cleanup_button_rect.right + button_spacing

            demo_button_rect = pygame.Rect(
                current_x, button_y_pos, demo_button_width, button_height
            )
            self._draw_button(
                demo_button_rect, "Play Demo", (40, 100, 40), enabled=buttons_enabled
            )
            stat_rects["Play Demo Button"] = demo_button_rect
            current_x = demo_button_rect.right + button_spacing

            debug_button_rect = pygame.Rect(
                current_x, button_y_pos, debug_button_width, button_height
            )
            self._draw_button(
                debug_button_rect, "Debug Mode", (100, 40, 100), enabled=buttons_enabled
            )
            stat_rects["Debug Mode Button"] = debug_button_rect
            next_y = run_button_rect.bottom + 10

        # Render Compact Status Block (always below buttons/progress bar)
        status_block_y = next_y
        next_y, status_rects_compact = self._render_compact_status(
            status_block_y, panel_width, status, stats_summary
        )
        stat_rects.update(status_rects_compact)

        # Render Detailed Update Progress Bars if Updating
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
