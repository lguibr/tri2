import pygame
import time
import traceback
import os
import shutil
from typing import TYPE_CHECKING, Tuple

from app_state import AppState
from config.general import get_run_checkpoint_dir 

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppLogic:
    """Handles the core application logic and state transitions."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def check_initial_completion_status(self):
        """Checks if training target was met upon loading (placeholder)."""
        pass 

    def update_status_and_check_completion(self):
        """Updates the status text based on application state."""
        is_running = self.app.worker_manager.is_any_worker_running()
        state = self.app.app_state
        if state == AppState.MAIN_MENU:
            self.app.status = (
                "Confirm Cleanup"
                if self.app.cleanup_confirmation_active
                else "Running AlphaZero" if is_running else "Ready"
            )
        elif state == AppState.PLAYING:
            self.app.status = "Playing Demo"
        elif state == AppState.DEBUG:
            self.app.status = "Debugging Grid"
        elif state == AppState.INITIALIZING:
            self.app.status = "Initializing..."

    # --- Worker Control ---
    def start_run(self):
        """Starts both self-play and training workers."""
        if (
            self.app.app_state != AppState.MAIN_MENU
            or self.app.worker_manager.is_any_worker_running()
        ):
            print("Cannot start run: Not in Main Menu or already running.")
            return
        print("Starting AlphaZero Run (Self-Play & Training)...")
        self.app.worker_manager.start_all_workers()
        self.update_status_and_check_completion()

    def stop_run(self):
        """Stops both self-play and training workers."""
        if not self.app.worker_manager.is_any_worker_running():
            print("Run not currently active.")
            return
        print("Stopping AlphaZero Run...")
        self.app.worker_manager.stop_all_workers()
        self.update_status_and_check_completion()

    # --- Mode Transitions & Cleanup ---
    def request_cleanup(self):
        if self.app.app_state != AppState.MAIN_MENU:
            return
        if self.app.worker_manager.is_any_worker_running():
            self._set_temp_message("Stop Run before Cleanup!")
            return
        self.app.cleanup_confirmation_active = True
        self.app.status = "Confirm Cleanup"
        print("Cleanup requested. Confirm action.")

    def start_demo_mode(self):
        if self._can_start_mode("Demo"):
            print("Entering Demo Mode...")
            self.try_save_checkpoint()
            self.app.app_state = AppState.PLAYING
            self.app.status = "Playing Demo"
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()

    def start_debug_mode(self):
        if self._can_start_mode("Debug"):
            print("Entering Debug Mode...")
            self.try_save_checkpoint()
            self.app.app_state = AppState.DEBUG
            self.app.status = "Debugging Grid"
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()

    def _can_start_mode(self, mode_name: str) -> bool:
        """Checks if demo/debug mode can be started."""
        if self.app.initializer.demo_env is None:
            print(f"Cannot start {mode_name}: Env not initialized.")
            return False
        if self.app.app_state != AppState.MAIN_MENU:
            print(f"Cannot start {mode_name} mode outside MainMenu.")
            return False
        if self.app.worker_manager.is_any_worker_running():
            self._set_temp_message(f"Stop Run before {mode_name}!")
            return False
        return True

    def exit_demo_mode(self):
        if self.app.app_state == AppState.PLAYING:
            print("Exiting Demo Mode...")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.deselect_dragged_shape()
            self._return_to_main_menu()

    def exit_debug_mode(self):
        if self.app.app_state == AppState.DEBUG:
            print("Exiting Debug Mode...")
            self._return_to_main_menu()

    def _return_to_main_menu(self):
        """Helper to transition back to the main menu state."""
        self.app.app_state = AppState.MAIN_MENU
        self.check_initial_completion_status()
        self.update_status_and_check_completion()

    def cancel_cleanup(self):
        self.app.cleanup_confirmation_active = False
        self._set_temp_message("Cleanup cancelled.")
        self.update_status_and_check_completion()
        print("Cleanup cancelled by user.")

    def confirm_cleanup(self):
        print("Cleanup confirmed by user. Starting process...")
        try:
            self._cleanup_data()
        except Exception as e:
            print(f"FATAL ERROR during cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Cleanup Failed Critically"
            self.app.app_state = AppState.ERROR
        finally:
            self.app.cleanup_confirmation_active = False
            print(
                f"Cleanup process finished. State: {self.app.app_state}, Status: {self.app.status}"
            )

    def exit_app(self) -> bool:
        print("Exit requested.")
        self.app.stop_event.set()
        self.app.worker_manager.stop_all_workers()
        return False  # Signal main loop to stop

    # --- Input Handling Callbacks ---
    def handle_demo_mouse_motion(self, mouse_pos: Tuple[int, int]):
        if self.app.app_state != AppState.PLAYING or not self.app.initializer.demo_env:
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if demo_env.demo_dragged_shape_idx is None:
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        demo_env.update_snapped_position(grid_coords)

    def handle_demo_mouse_button_down(self, event: pygame.event.Event):
        if self.app.app_state != AppState.PLAYING or not self.app.initializer.demo_env:
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over() or event.button != 1:
            return

        mouse_pos = event.pos
        clicked_preview = self.app.ui_utils.map_screen_to_preview(mouse_pos)
        if clicked_preview is not None:
            action = (
                demo_env.deselect_dragged_shape
                if clicked_preview == demo_env.demo_dragged_shape_idx
                else lambda: demo_env.select_shape_for_drag(clicked_preview)
            )
            action()
            return

        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        if (
            grid_coords is not None
            and demo_env.demo_dragged_shape_idx is not None
            and demo_env.demo_snapped_position == grid_coords
        ):
            placed = demo_env.place_dragged_shape()
            if placed and demo_env.is_over():
                print("[Demo] Game Over! Press ESC to exit.")
        else:
            demo_env.deselect_dragged_shape()

    def handle_debug_input(self, event: pygame.event.Event):
        if self.app.app_state != AppState.DEBUG or not self.app.initializer.demo_env:
            return
        demo_env = self.app.initializer.demo_env
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            print("[Debug] Resetting grid...")
            demo_env.reset()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            clicked_coords = self.app.ui_utils.map_screen_to_grid(event.pos)
            if clicked_coords:
                demo_env.toggle_triangle_debug(*clicked_coords)

    # --- Internal Helpers ---
    def _set_temp_message(self, message: str):
        """Sets a temporary message to be displayed."""
        self.app.cleanup_message = message
        self.app.last_cleanup_message_time = time.time()

    def _cleanup_data(self):
        """Deletes current run's checkpoint and re-initializes components."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.app.app_state = AppState.INITIALIZING
        self.app.status = "Cleaning"
        messages = []
        self._render_during_cleanup()

        print("[Cleanup] Stopping existing worker threads (if any)...")
        self.app.worker_manager.stop_all_workers()
        print("[Cleanup] Existing worker threads stopped.")
        print("[Cleanup] Closing stats recorder...")
        self.app.initializer.close_stats_recorder(is_cleanup=True)
        print("[Cleanup] Stats recorder closed.")

        messages.append(self._delete_checkpoint_dir())
        time.sleep(0.1)

        print("[Cleanup] Re-initializing components...")
        try:
            self.app.initializer.initialize_rl_components(
                is_reinit=True, checkpoint_to_load=None
            )
            print("[Cleanup] Components re-initialized.")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
            self.app.initializer.initialize_workers()
            print("[Cleanup] Workers re-initialized (not started).")
            messages.append("Components re-initialized.")
            self.app.status = "Ready"
            self.app.app_state = AppState.MAIN_MENU
        except Exception as e:
            print(f"FATAL ERROR during re-initialization after cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Re-init Failed"
            self.app.app_state = AppState.ERROR
            messages.append("ERROR RE-INITIALIZING COMPONENTS!")
            if self.app.renderer:
                self.app.renderer._render_error_screen(self.app.status)

        self._set_temp_message("\n".join(messages))
        print(f"--- CLEANUP DATA COMPLETE (Final State: {self.app.app_state}) ---")

    def _render_during_cleanup(self):
        """Renders the screen while cleanup is in progress."""
        if self.app.renderer:
            try:
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=False,
                    status=self.app.status,
                    stats_summary={},
                    envs=[],
                    num_envs=0,
                    env_config=self.app.env_config,
                    cleanup_confirmation_active=False,
                    cleanup_message="",
                    last_cleanup_message_time=0,
                    plot_data={},
                    demo_env=self.app.initializer.demo_env,
                    update_progress_details={},
                    agent_param_count=getattr(
                        self.app.initializer, "agent_param_count", 0
                    ),
                    worker_counts={},
                    best_game_state_data=None,
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

    def _delete_checkpoint_dir(self) -> str:
        """Deletes the checkpoint directory and returns a status message."""
        print("[Cleanup] Deleting agent checkpoint file/dir...")
        msg = ""
        try:
            save_dir = get_run_checkpoint_dir()
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
                msg = f"Run checkpoint directory deleted: {save_dir}"
            else:
                msg = f"Run checkpoint directory not found: {save_dir}"
        except OSError as e:
            msg = f"Error deleting checkpoint dir: {e}"
        print(f"  - {msg}")
        print("[Cleanup] Checkpoint deletion attempt finished.")
        return msg

    def try_save_checkpoint(self):
        """Saves checkpoint if in main menu and workers are not running."""
        if (
            self.app.app_state != AppState.MAIN_MENU
            or self.app.worker_manager.is_any_worker_running()
        ):
            return
        if (
            not self.app.initializer.checkpoint_manager
            or not self.app.initializer.stats_aggregator
        ):
            return

        print("Saving checkpoint...")
        try:
            agg_storage = self.app.initializer.stats_aggregator.storage
            current_step = getattr(agg_storage, "current_global_step", 0)
            episode_count = getattr(agg_storage, "total_episodes", 0)
            target_step = getattr(
                self.app.initializer.checkpoint_manager, "training_target_step", 0
            )
            self.app.initializer.checkpoint_manager.save_checkpoint(
                current_step,
                episode_count,
                training_target_step=target_step,
                is_final=False,
            )
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            traceback.print_exc()

    def save_final_checkpoint(self):
        """Saves the final checkpoint."""
        if (
            not self.app.initializer.checkpoint_manager
            or not self.app.initializer.stats_aggregator
        ):
            return
        save_on_exit = (
            self.app.status != "Cleaning" and self.app.app_state != AppState.ERROR
        )
        if save_on_exit:
            print("Performing final checkpoint save...")
            try:
                agg_storage = self.app.initializer.stats_aggregator.storage
                current_step = getattr(agg_storage, "current_global_step", 0)
                episode_count = getattr(agg_storage, "total_episodes", 0)
                target_step = getattr(
                    self.app.initializer.checkpoint_manager, "training_target_step", 0
                )
                self.app.initializer.checkpoint_manager.save_checkpoint(
                    current_step,
                    episode_count,
                    training_target_step=target_step,
                    is_final=True,
                )
            except Exception as final_save_err:
                print(f"Error during final checkpoint save: {final_save_err}")
                traceback.print_exc()
        else:
            print("Skipping final checkpoint save.")
