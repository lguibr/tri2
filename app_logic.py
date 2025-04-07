# File: app_logic.py
# File: app_logic.py
import pygame
import time
import traceback
import os
import shutil
from typing import TYPE_CHECKING, Tuple

from app_state import AppState

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppLogic:
    """Handles the core application logic and state transitions."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def check_initial_completion_status(self):
        """Checks if training was already complete upon loading."""
        if (
            self.app.initializer.checkpoint_manager
            and self.app.initializer.checkpoint_manager.training_target_step > 0
            and self.app.initializer.checkpoint_manager.global_step
            >= self.app.initializer.checkpoint_manager.training_target_step
        ):
            self.app.status = "Training Complete"
            print(
                f"Training already completed ({self.app.initializer.checkpoint_manager.global_step:,}/{self.app.initializer.checkpoint_manager.training_target_step:,} steps). Ready."
            )
            # Don't automatically pause here, let the user decide

    def update_status_and_check_completion(self):
        """Updates the status text based on worker state and checks for training completion."""
        if (
            not self.app.initializer.stats_recorder
            or not hasattr(self.app.initializer.stats_recorder, "aggregator")
            or not self.app.initializer.checkpoint_manager
        ):
            return

        current_step = getattr(
            self.app.initializer.stats_recorder.aggregator.storage,
            "current_global_step",
            0,  # Access via storage
        )
        target_step = self.app.initializer.checkpoint_manager.training_target_step

        if target_step > 0 and current_step >= target_step:
            if not self.app.status.startswith("Training Complete"):
                print(
                    f"\n--- Training Complete ({current_step:,}/{target_step:,} steps) ---"
                )
                self.app.status = "Training Complete"
            elif self.app.is_process_running and self.app.status == "Training Complete":
                self.app.status = "Training Complete (Running)"
            return

        # If not complete, update status based on running state
        if self.app.is_process_running:
            # Check if update progress details indicate an active update
            is_updating = (
                hasattr(self.app, "update_progress_details")
                and self.app.update_progress_details.get("current_epoch", 0) > 0
            )
            if is_updating:
                self.app.status = "Updating Agent"
            elif self.app.experience_queue.qsize() > 0:  # Check queue as fallback
                self.app.status = "Updating Agent"
            else:
                self.app.status = "Collecting Experience"
        elif self.app.app_state == AppState.MAIN_MENU:
            if self.app.cleanup_confirmation_active:
                self.app.status = "Confirm Cleanup"
            elif not self.app.status.startswith(
                "Training Complete"
            ):  # Avoid overwriting completion
                self.app.status = "Ready"
        elif self.app.app_state == AppState.PLAYING:
            self.app.status = "Playing Demo"
        elif self.app.app_state == AppState.DEBUG:
            self.app.status = "Debugging Grid"
        elif self.app.app_state == AppState.INITIALIZING:
            self.app.status = "Initializing..."

    def toggle_training_run(self):
        """Starts or stops the worker threads."""
        print(
            f"[AppLogic] toggle_training_run called. Current state: {self.app.app_state.value}, is_process_running: {self.app.is_process_running}"
        )
        if self.app.app_state != AppState.MAIN_MENU:
            print(
                f"[AppLogic] Cannot toggle run outside MainMenu (State: {self.app.app_state.value})."
            )
            return
        if (
            not self.app.worker_manager.env_runner_thread
            or not self.app.worker_manager.training_worker_thread
        ):
            print("[AppLogic] Cannot toggle run: Workers not initialized.")
            return

        if not self.app.is_process_running:
            print("[AppLogic] Attempting to START workers...")
            print("[AppLogic] Setting is_process_running = True")
            self.app.is_process_running = True
            print(
                f"[AppLogic] Clearing pause event (Current is_set: {self.app.pause_event.is_set()})..."
            )
            self.app.pause_event.clear()
            print(
                f"[AppLogic] Pause event cleared (New is_set: {self.app.pause_event.is_set()})."
            )
            # Update status based on completion
            self.update_status_and_check_completion()  # This will set status correctly
            print("[AppLogic] Workers should start running.")
        else:
            print("[AppLogic] Attempting to PAUSE workers...")
            print("[AppLogic] Setting is_process_running = False")
            self.app.is_process_running = False
            print(
                f"[AppLogic] Setting pause event (Current is_set: {self.app.pause_event.is_set()})..."
            )
            self.app.pause_event.set()
            print(
                f"[AppLogic] Pause event set (New is_set: {self.app.pause_event.is_set()})."
            )
            self.try_save_checkpoint()
            self.check_initial_completion_status()  # Re-check completion status on pause
            if not self.app.status.startswith("Training Complete"):
                self.app.status = "Ready"
            self.app.app_state = AppState.MAIN_MENU
            print("[AppLogic] Workers should pause.")

    def request_cleanup(self):
        if self.app.is_process_running:
            print("Cannot request cleanup while process is running. Pause first.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot request cleanup outside MainMenu.")
            return
        self.app.cleanup_confirmation_active = True
        print("Cleanup requested. Confirm action.")

    def start_demo_mode(self):
        if self.app.is_process_running:
            print("Cannot start demo mode while process is running. Pause first.")
            return
        if self.app.initializer.demo_env is None:
            print("Cannot start demo mode: Demo environment failed to initialize.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot start demo mode outside MainMenu.")
            return
        print("Entering Demo Mode...")
        self.try_save_checkpoint()
        self.app.app_state = AppState.PLAYING
        self.app.status = "Playing Demo"
        self.app.initializer.demo_env.reset()

    def start_debug_mode(self):
        if self.app.is_process_running:
            print("Cannot start debug mode while process is running. Pause first.")
            return
        if self.app.initializer.demo_env is None:
            print("Cannot start debug mode: Demo environment failed to initialize.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot start debug mode outside MainMenu.")
            return
        print("Entering Debug Mode...")
        self.try_save_checkpoint()
        self.app.app_state = AppState.DEBUG
        self.app.status = "Debugging Grid"
        self.app.initializer.demo_env.reset()

    def exit_debug_mode(self):
        if self.app.app_state == AppState.DEBUG:
            print("Exiting Debug Mode...")
            self.app.app_state = AppState.MAIN_MENU
            # --- Ensure process is marked as stopped/paused ---
            self.app.is_process_running = False
            self.app.pause_event.set()
            # --- End Ensure ---
            self.check_initial_completion_status()  # Check completion status on exit
            if not self.app.status.startswith("Training Complete"):
                self.app.status = "Ready"

    def cancel_cleanup(self):
        self.app.cleanup_confirmation_active = False
        self.app.cleanup_message = "Cleanup cancelled."
        self.app.last_cleanup_message_time = time.time()
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
        return False

    def exit_demo_mode(self):
        if self.app.app_state == AppState.PLAYING:
            print("Exiting Demo Mode...")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.deselect_dragged_shape()
            self.app.app_state = AppState.MAIN_MENU
            # --- Ensure process is marked as stopped/paused ---
            self.app.is_process_running = False
            self.app.pause_event.set()
            # --- End Ensure ---
            self.check_initial_completion_status()  # Check completion status on exit
            if not self.app.status.startswith("Training Complete"):
                self.app.status = "Ready"

    def handle_demo_mouse_motion(self, mouse_pos: Tuple[int, int]):
        if (
            self.app.app_state != AppState.PLAYING
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if demo_env.demo_dragged_shape_idx is None:
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        demo_env.update_snapped_position(grid_coords)

    def handle_demo_mouse_button_down(self, event: pygame.event.Event):
        if (
            self.app.app_state != AppState.PLAYING
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if event.button != 1:
            return
        mouse_pos = event.pos
        clicked_preview_index = self.app.ui_utils.map_screen_to_preview(mouse_pos)
        if clicked_preview_index is not None:
            if clicked_preview_index == demo_env.demo_dragged_shape_idx:
                demo_env.deselect_dragged_shape()
            else:
                demo_env.select_shape_for_drag(clicked_preview_index)
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        if grid_coords is not None:
            if (
                demo_env.demo_dragged_shape_idx is not None
                and demo_env.demo_snapped_position == grid_coords
            ):
                placed = demo_env.place_dragged_shape()
                if placed and demo_env.is_over():
                    print("[Demo] Game Over! Press ESC to exit.")
            else:
                demo_env.deselect_dragged_shape()
            return
        demo_env.deselect_dragged_shape()

    def handle_debug_input(self, event: pygame.event.Event):
        if (
            self.app.app_state != AppState.DEBUG
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                print("[Debug] Resetting grid...")
                demo_env.reset()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            clicked_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
            if clicked_coords:
                row, col = clicked_coords
                demo_env.toggle_triangle_debug(row, col)

    def _cleanup_data(self):
        """Deletes current run's checkpoint and re-initializes."""
        from config.general import get_run_checkpoint_dir  # Local import

        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.app.pause_event.set()  # Ensure workers are paused if running
        self.app.is_process_running = False
        self.app.app_state = AppState.INITIALIZING
        self.app.status = "Cleaning"
        messages = []

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
                    tensorboard_log_dir=None,
                    plot_data={},
                    demo_env=self.app.initializer.demo_env,
                    update_progress_details={},
                    agent_param_count=getattr(
                        self.app.initializer, "agent_param_count", 0
                    ),
                    worker_counts={
                        "env_runners": 0,
                        "trainers": 0,
                    },  # Pass default counts
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

        # --- Stop existing worker threads FIRST ---
        print("[Cleanup] Stopping existing worker threads...")
        self.app.worker_manager.stop_worker_threads()  # This calls join()
        print("[Cleanup] Existing worker threads stopped.")
        # --- End Stop Workers ---

        # --- Close Stats Recorder AFTER stopping workers ---
        print("[Cleanup] Closing stats recorder...")
        # Pass is_cleanup=True to prevent final hparam logging during cleanup
        self.app.initializer.close_stats_recorder(is_cleanup=True)
        print("[Cleanup] Stats recorder closed.")
        # --- End Close Stats ---

        print("[Cleanup] Deleting agent checkpoint file/dir...")
        try:
            save_dir = get_run_checkpoint_dir()
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
                msg = f"Run checkpoint directory deleted: {save_dir}"
            else:
                msg = f"Run checkpoint directory not found: {save_dir}"
            print(f"  - {msg}")
            messages.append(msg)
        except OSError as e:
            msg = f"Error deleting checkpoint dir: {e}"
            print(f"  - {msg}")
            messages.append(msg)
        print("[Cleanup] Checkpoint deletion attempt finished.")

        time.sleep(0.1)
        print("[Cleanup] Re-initializing RL components...")
        try:
            # Re-initialize RL components (creates new agent, stats recorder etc.)
            self.app.initializer.initialize_rl_components(
                is_reinit=True, checkpoint_to_load=None
            )
            print("[Cleanup] RL components re-initialized.")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
                print("[Cleanup] Demo env reset.")

            # --- Set pause event BEFORE starting new workers ---
            print("[Cleanup] Setting pause event before starting new workers...")
            self.app.pause_event.set()
            # --- End Set Pause ---

            # Start NEW worker threads with the re-initialized components
            print("[Cleanup] Starting new worker threads...")
            self.app.worker_manager.start_worker_threads()
            print("[Cleanup] New worker threads started.")

            print(
                "[Cleanup] RL components re-initialization and worker start successful."
            )
            messages.append("RL components re-initialized.")

            # --- Ensure state is PAUSED after cleanup ---
            self.app.is_process_running = False
            # self.app.pause_event.set() # Moved before starting workers
            self.app.status = "Ready"
            self.app.app_state = AppState.MAIN_MENU
            print("[Cleanup] Application state set to MAIN_MENU (Paused).")
            # --- End Ensure Paused State ---

        except Exception as e:
            print(f"FATAL ERROR during RL re-initialization after cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Re-init Failed"
            self.app.app_state = AppState.ERROR
            messages.append("ERROR RE-INITIALIZING RL COMPONENTS!")
            if self.app.renderer:
                try:
                    self.app.renderer._render_error_screen(self.app.status)
                except Exception as render_err_final:
                    print(f"Warning: Failed to render error screen: {render_err_final}")

        self.app.cleanup_message = "\n".join(messages)
        self.app.last_cleanup_message_time = time.time()
        print(
            f"--- CLEANUP DATA COMPLETE (Final State: {self.app.app_state}, Status: {self.app.status}) ---"
        )

    def try_save_checkpoint(self):
        """Saves checkpoint if paused and checkpoint manager exists."""
        if (
            self.app.app_state == AppState.MAIN_MENU
            and not self.app.is_process_running
            and self.app.initializer.checkpoint_manager
            and self.app.initializer.stats_recorder  # Ensure stats recorder exists
            and hasattr(
                self.app.initializer.stats_recorder, "aggregator"
            )  # Ensure aggregator exists
        ):
            print("Saving checkpoint on pause...")
            try:
                current_step = getattr(
                    self.app.initializer.stats_recorder.aggregator.storage,  # Access via storage
                    "current_global_step",
                    0,
                )
                target_step = (
                    self.app.initializer.checkpoint_manager.training_target_step
                )
                episode_count = getattr(
                    self.app.initializer.stats_recorder.aggregator.storage,
                    "total_episodes",
                    0,  # Access via storage
                )

                self.app.initializer.checkpoint_manager.save_checkpoint(
                    current_step,
                    episode_count,
                    training_target_step=target_step,
                    is_final=False,
                )
            except Exception as e:
                print(f"Error saving checkpoint on pause: {e}")
                traceback.print_exc()  # Print traceback for debugging

    def save_final_checkpoint(self):
        """Saves the final checkpoint if conditions are met."""
        if (
            self.app.initializer.checkpoint_manager
            and self.app.initializer.stats_recorder
            and hasattr(self.app.initializer.stats_recorder, "aggregator")
        ):
            current_step = getattr(
                self.app.initializer.stats_recorder.aggregator.storage,
                "current_global_step",
                0,  # Access via storage
            )
            target_step = getattr(
                self.app.initializer.checkpoint_manager, "training_target_step", 0
            )
            is_complete = target_step > 0 and current_step >= target_step
            save_on_exit = (
                self.app.status != "Cleaning" and self.app.app_state != AppState.ERROR
            )  # Always save unless cleaning or error

            if save_on_exit:
                print("Performing final checkpoint save...")
                try:
                    episode_count = getattr(
                        self.app.initializer.stats_recorder.aggregator.storage,  # Access via storage
                        "total_episodes",
                        0,
                    )
                    self.app.initializer.checkpoint_manager.save_checkpoint(
                        current_step,
                        episode_count,
                        training_target_step=target_step,
                        is_final=True,
                    )
                except Exception as final_save_err:
                    print(f"Error during final checkpoint save: {final_save_err}")
                    traceback.print_exc()  # Print traceback for debugging
            else:
                print("Skipping final checkpoint save.")
