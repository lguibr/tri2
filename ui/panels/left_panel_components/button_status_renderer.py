# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
import time
import math  # Added for pulsing effect
from typing import Dict, Tuple, Any, Optional  # Added Optional

from config import WHITE, YELLOW, RED, GOOGLE_COLORS, LIGHTG
from utils.helpers import format_eta
from ui.input_handler import InputHandler  # Import for type hint


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
        # Store reference to input handler to get button rects
        self.input_handler_ref: Optional[InputHandler] = None  # Set externally

    def _draw_button(
        self,
        rect: pygame.Rect,
        text: str,
        color: Tuple[int, int, int],
        enabled: bool = True,
    ):
        """Helper to draw a single button, optionally grayed out."""
        final_color = (
            color
            if enabled
            else (
                tuple(max(30, c // 2) for c in color[:3])
                if isinstance(color, tuple) and len(color) >= 3
                else (50, 50, 50)
            )
        )
        pygame.draw.rect(self.screen, final_color, rect, border_radius=5)
        text_color = WHITE if enabled else (150, 150, 150)
        if self.ui_font:
            label_surface = self.ui_font.render(text, True, text_color)
            self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        else:  # Fallback if font fails
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
        elif "Collecting" in status:
            status_color = GOOGLE_COLORS[0]
        elif "Updating" in status:  # Should not be called here now, but keep for safety
            status_color = GOOGLE_COLORS[2]
        elif "Ready" in status:
            status_color = WHITE
        elif "Debugging" in status:
            status_color = (200, 100, 200)
        elif "Initializing" in status:
            status_color = LIGHTG
        elif "Training Complete" in status:
            status_color = (100, 200, 100)  # Green for complete

        status_surface = self.status_font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(status_surface, status_rect)
        current_y += line_height_status

        # --- Fetch steps from summary ---
        global_step = stats_summary.get("global_step", 0)
        training_target_step = stats_summary.get("training_target_step", 0)
        # --- End Fetch steps ---
        total_episodes = stats_summary.get("total_episodes", 0)
        minibatch_sps = stats_summary.get(
            "avg_minibatch_sps_window", 0.0
        )  # Use avg window

        # Format steps with underscores
        global_step_str = f"{global_step:,}".replace(",", "_")
        target_step_str = f"{training_target_step:,}".replace(",", "_")

        # Use fetched target step for display
        steps_str = (
            f"{global_step_str}/{target_step_str} Steps"
            if training_target_step > 0
            else f"{global_step_str} Steps"
        )
        eps_str, sps_str = (
            f"~{total_episodes} Eps",  # Use ~ as it might be slightly behind
            f"~{minibatch_sps:.0f} SPS (MB)",  # Display minibatch SPS
        )
        line2_text = f"{steps_str} | {eps_str} | {sps_str}"
        line2_surface = self.status_label_font.render(line2_text, True, LIGHTG)
        line2_rect = line2_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(line2_surface, line2_rect)

        current_y += line_height_label + 2

        return current_y

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
        pulsing: bool = False,  # Add pulsing flag
    ) -> pygame.Rect:
        """Renders a single progress bar component with optional pulsing effect."""
        x_margin, bar_width = x_offset, panel_width - 2 * x_offset
        if bar_width <= 0:
            return pygame.Rect(x_margin, y_pos, 0, 0)

        background_rect = pygame.Rect(x_margin, y_pos, bar_width, bar_height)
        pygame.draw.rect(self.screen, (60, 60, 80), background_rect, border_radius=3)
        clamped_progress = max(0.0, min(1.0, progress))
        progress_width = int(bar_width * clamped_progress)

        # --- Pulsing Effect ---
        final_bar_color = bar_color
        if pulsing:
            pulse_speed = 3.0  # Adjust speed of pulse
            pulse_range = 0.2  # Adjust intensity of pulse (0.0 to 1.0)
            pulse_factor = (1.0 - pulse_range) + pulse_range * (
                0.5 * (1 + math.sin(time.time() * pulse_speed))
            )
            final_bar_color = tuple(
                min(255, max(0, int(c * pulse_factor))) for c in bar_color[:3]
            )
        # --- End Pulsing Effect ---

        if progress_width > 0:
            pygame.draw.rect(
                self.screen,
                final_bar_color,  # Use potentially modified color
                pygame.Rect(x_margin, y_pos, progress_width, bar_height),
                border_radius=3,
            )
        pygame.draw.rect(self.screen, LIGHTG, background_rect, 1, border_radius=3)

        if self.progress_font:
            progress_text_str = (
                custom_text if custom_text else f"{text_prefix}{clamped_progress:.0%}"
            )
            text_surface = self.progress_font.render(progress_text_str, True, WHITE)
            self.screen.blit(
                text_surface, text_surface.get_rect(center=background_rect.center)
            )
        return background_rect

    def _render_detailed_progress_bars(
        self,
        y_start: int,
        panel_width: int,
        progress_details: Dict[str, Any],
        stats_summary: Dict[str, Any],  # Add stats_summary for SPS
        bar_color: Tuple[int, int, int],
        tooltip_key_prefix: str = "Update",
    ) -> int:
        """Renders two progress bars (overall, epoch) and epoch text, including ETAs."""
        x_margin, bar_height, bar_spacing = 10, 16, 2
        text_height = self.progress_font.get_linesize() if self.progress_font else 14
        current_y = y_start

        overall_progress = progress_details.get("overall_progress", 0.0)
        epoch_progress = progress_details.get("epoch_progress", 0.0)
        current_epoch = progress_details.get("current_epoch", 0)
        total_epochs = progress_details.get("total_epochs", 0)
        update_start_time = progress_details.get("update_start_time", 0.0)
        num_minibatches_per_epoch = progress_details.get("num_minibatches_per_epoch", 0)
        current_minibatch_index = progress_details.get("current_minibatch_index", 0)

        # Get current minibatch SPS from summary
        current_minibatch_sps = stats_summary.get(
            "avg_minibatch_sps_window", 0.0
        )  # Use avg

        bar_width = panel_width - 2 * x_margin
        if bar_width <= 0:
            return current_y

        overall_eta_str, epoch_eta_str = "N/A", "N/A"
        time_elapsed = time.time() - update_start_time if update_start_time > 0 else 0.0

        # --- Calculate ETAs using Minibatch SPS ---
        if time_elapsed > 1.0 and num_minibatches_per_epoch > 0:
            total_minibatches_overall = total_epochs * num_minibatches_per_epoch
            minibatches_done_overall = max(
                0,
                (current_epoch - 1) * num_minibatches_per_epoch
                + current_minibatch_index,
            )
            minibatches_remaining_overall = max(
                0, total_minibatches_overall - minibatches_done_overall
            )

            if current_minibatch_sps > 1e-3:  # Use minibatch SPS if available
                remaining_time_overall = (
                    minibatches_remaining_overall / current_minibatch_sps
                )
                overall_eta_str = format_eta(remaining_time_overall)
            elif overall_progress > 1e-6:  # Fallback to old method if SPS is zero
                remaining_time_overall = max(
                    0.0, (time_elapsed / overall_progress) - time_elapsed
                )
                overall_eta_str = format_eta(remaining_time_overall)

            minibatches_remaining_epoch = max(
                0, num_minibatches_per_epoch - current_minibatch_index
            )
            if current_minibatch_sps > 1e-3:
                remaining_time_epoch = (
                    minibatches_remaining_epoch / current_minibatch_sps
                )
                epoch_eta_str = format_eta(remaining_time_epoch)
            elif epoch_progress > 1e-6 and num_minibatches_per_epoch > 0:  # Fallback
                effective_epochs_done = max(1e-6, (current_epoch - 1) + epoch_progress)
                time_per_epoch_estimate = time_elapsed / effective_epochs_done
                remaining_time_epoch = max(
                    0.0, time_per_epoch_estimate * (1.0 - epoch_progress)
                )
                epoch_eta_str = format_eta(remaining_time_epoch)
            elif epoch_progress == 0.0:
                epoch_eta_str = "Starting..."
        # --- End ETA Calculation ---

        epoch_text = f"Epoch {current_epoch}/{total_epochs} (Minibatch {current_minibatch_index}/{num_minibatches_per_epoch})"  # Added minibatch info
        if self.progress_font:
            text_surface = self.progress_font.render(epoch_text, True, WHITE)
            text_rect = text_surface.get_rect(topleft=(x_margin, current_y))
            self.screen.blit(text_surface, text_rect)
            current_y += text_height + 2

        epoch_bar_text = f"Epoch: {epoch_progress:.0%} | ETA: {epoch_eta_str}"
        epoch_bar_rect = self._render_single_progress_bar(
            current_y,
            panel_width,
            epoch_progress,
            bar_color,
            custom_text=epoch_bar_text,
            pulsing=True,  # Add pulsing effect
        )
        current_y += bar_height + bar_spacing

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
            pulsing=True,  # Add pulsing effect
        )
        current_y += bar_height + 5

        return current_y

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        update_progress_details: Dict[str, Any],
    ) -> int:
        """Renders buttons, status, and progress bar(s). Returns next_y."""
        from app_state import AppState  # Local import for Enum comparison

        next_y = y_start

        # Get button rects from InputHandler if available
        run_btn_rect = (
            self.input_handler_ref.run_btn_rect
            if self.input_handler_ref
            else pygame.Rect(10, y_start, 100, 40)
        )
        cleanup_btn_rect = (
            self.input_handler_ref.cleanup_btn_rect
            if self.input_handler_ref
            else pygame.Rect(run_btn_rect.right + 10, y_start, 160, 40)
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

        # --- Run/Stop Button Logic ---
        run_stop_enabled = app_state == AppState.MAIN_MENU.value
        run_button_text = "Run"
        run_button_color = (40, 40, 80)  # Default Run color

        if is_process_running:
            run_button_text = "Stop"
            run_button_color = (80, 40, 40)  # Default Stop color
            if "Collecting" in status:
                run_button_color = (40, 80, 40)  # Greenish when collecting
            elif "Updating" in status:
                run_button_color = (40, 40, 80)  # Blueish when updating
            elif status.startswith("Training Complete"):
                run_button_color = (80, 80, 40)  # Yellowish when complete but running
        elif status.startswith("Training Complete"):
            run_button_text = "Run"  # Show Run even if complete, but disabled
            run_stop_enabled = False  # Disable Run if complete and stopped

        self._draw_button(
            run_btn_rect, run_button_text, run_button_color, enabled=run_stop_enabled
        )
        # --- End Run/Stop Button Logic ---

        # --- Conditional Rendering: Progress Bar OR Other Buttons ---
        if is_process_running:
            # --- Training Progress Bar (uses stats_summary) ---
            progress_bar_x = run_btn_rect.right + 10
            progress_bar_width = panel_width - progress_bar_x - 10
            progress_bar_height = run_btn_rect.height
            progress_bar_y = run_btn_rect.y

            # --- Fetch steps from summary ---
            current_global_step = stats_summary.get("global_step", 0)
            training_target_step = stats_summary.get("training_target_step", 0)
            # --- End Fetch steps ---
            start_time = stats_summary.get("start_time", 0.0)

            progress = (
                current_global_step / max(1, training_target_step)
                if training_target_step > 0
                else 0.0
            )
            progress_bar_color = (20, 100, 20)

            # Format steps with underscores using the current step
            global_step_str = f"{current_global_step:,}".replace(",", "_")
            target_step_str = f"{training_target_step:,}".replace(",", "_")

            # Handle cases where target step might be 0 or less
            if training_target_step <= 0:
                progress_text = f"{global_step_str} Steps (No Target)"
                progress = 0.0  # No progress if no target
            elif current_global_step >= training_target_step:
                # Use fetched target step for display
                progress_text = (
                    f"Target Reached ({global_step_str} / {target_step_str})"
                )
                progress, progress_bar_color = 1.0, (100, 150, 100)
            else:
                eta_str = "N/A"
                time_elapsed = time.time() - start_time if start_time > 0 else 0.0
                if progress > 1e-6 and progress < 1.0 and time_elapsed > 1.0:
                    eta_str = format_eta(
                        max(0.0, (time_elapsed / progress) - time_elapsed)
                    )
                elif progress >= 1.0:
                    eta_str = "Done"
                else:
                    eta_str = "Calculating..." if current_global_step > 0 else "N/A"
                # Use fetched target step for display
                progress_text = f"{global_step_str} / {target_step_str} Steps ({progress:.1%}) | ETA: {eta_str}"

            progress_rect = self._render_single_progress_bar(
                progress_bar_y,
                panel_width,
                progress,
                progress_bar_color,
                bar_height=progress_bar_height,
                x_offset=progress_bar_x,
                custom_text=progress_text,
            )
            next_y = run_btn_rect.bottom + 10
            # --- End Training Progress Bar ---
        else:
            # Render other buttons only if not running
            buttons_enabled = app_state == AppState.MAIN_MENU.value
            self._draw_button(
                cleanup_btn_rect,
                "Cleanup This Run",
                (100, 40, 40),
                enabled=buttons_enabled,
            )
            self._draw_button(
                demo_btn_rect, "Play Demo", (40, 100, 40), enabled=buttons_enabled
            )
            self._draw_button(
                debug_btn_rect, "Debug Mode", (100, 40, 100), enabled=buttons_enabled
            )
            next_y = run_btn_rect.bottom + 10  # Set next_y after buttons
        # --- End Conditional Rendering ---

        # --- Render Status Block OR Update Progress ---
        status_block_y = next_y
        if status == "Updating Agent":
            # Render detailed progress bars instead of compact status
            next_y = self._render_detailed_progress_bars(
                status_block_y,
                panel_width,
                update_progress_details,  # Pass the fetched details
                stats_summary,  # Pass stats summary for SPS
                GOOGLE_COLORS[2],  # Use blue color for update
                tooltip_key_prefix="Update",
            )
        else:
            # Render the normal compact status block
            next_y = self._render_compact_status(
                status_block_y, panel_width, status, stats_summary
            )
        # --- End Status Block / Update Progress ---

        return next_y
