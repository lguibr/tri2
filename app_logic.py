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
        """Checks if training target (e.g., games played) was met upon loading."""
        # This needs adaptation based on how AlphaZero training progress is tracked.
        # For now, assume completion isn't automatically checked this way.
        # Example: Check against games played or NN training steps in aggregator
        # if (
        #     self.app.initializer.stats_recorder
        #     and hasattr(self.app.initializer.stats_recorder, "aggregator")
        # ):
        #     aggregator = self.app.initializer.stats_recorder.aggregator
        #     target_games = getattr(aggregator.storage, "training_target_games", 0) # Example target
        #     current_games = getattr(aggregator.storage, "total_episodes", 0)
        #     if target_games > 0 and current_games >= target_games:
        #         self.app.status = "Target Reached"
        #         print(f"Target games already reached ({current_games:,}/{target_games:,}). Ready.")
        pass  # Keep simple for now

    def update_status_and_check_completion(self):
        """Updates the status text based on application state."""
        # Removed PPO-specific status logic (Collecting, Updating)
        # Removed completion check based on PPO steps

        # Update status based on AppState
        if self.app.app_state == AppState.MAIN_MENU:
            if self.app.cleanup_confirmation_active:
                self.app.status = "Confirm Cleanup"
            else:
                # Check if training is "complete" based on new criteria if needed
                # For now, just set to Ready if not confirming cleanup
                self.app.status = "Ready"
        elif self.app.app_state == AppState.PLAYING:
            self.app.status = "Playing Demo"
        elif self.app.app_state == AppState.DEBUG:
            self.app.status = "Debugging Grid"
        elif self.app.app_state == AppState.INITIALIZING:
            self.app.status = "Initializing..."
        elif self.app.app_state == AppState.ERROR:
            # Status should already be set by the error handler
            pass

    # Removed toggle_training_run method

    def request_cleanup(self):
        # Removed check for is_process_running
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot request cleanup outside MainMenu.")
            return
        self.app.cleanup_confirmation_active = True
        self.app.status = "Confirm Cleanup"  # Update status
        print("Cleanup requested. Confirm action.")

    def start_demo_mode(self):
        # Removed check for is_process_running
        if self.app.initializer.demo_env is None:
            print("Cannot start demo mode: Demo environment failed to initialize.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot start demo mode outside MainMenu.")
            return
        print("Entering Demo Mode...")
        self.try_save_checkpoint()  # Save NN weights before switching mode?
        self.app.app_state = AppState.PLAYING
        self.app.status = "Playing Demo"
        self.app.initializer.demo_env.reset()

    def start_debug_mode(self):
        # Removed check for is_process_running
        if self.app.initializer.demo_env is None:
            print("Cannot start debug mode: Demo environment failed to initialize.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot start debug mode outside MainMenu.")
            return
        print("Entering Debug Mode...")
        self.try_save_checkpoint()  # Save NN weights before switching mode?
        self.app.app_state = AppState.DEBUG
        self.app.status = "Debugging Grid"
        self.app.initializer.demo_env.reset()

    def exit_debug_mode(self):
        if self.app.app_state == AppState.DEBUG:
            print("Exiting Debug Mode...")
            self.app.app_state = AppState.MAIN_MENU
            # Removed setting is_process_running and pause_event
            self.check_initial_completion_status()  # Re-check status
            self.app.status = "Ready"  # Set status back to Ready

    def cancel_cleanup(self):
        self.app.cleanup_confirmation_active = False
        self.app.cleanup_message = "Cleanup cancelled."
        self.app.last_cleanup_message_time = time.time()
        self.app.status = "Ready"  # Set status back
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
            # Status is set within _cleanup_data or error handling
            print(
                f"Cleanup process finished. State: {self.app.app_state}, Status: {self.app.status}"
            )

    def exit_app(self) -> bool:
        print("Exit requested.")
        self.app.stop_event.set()  # Keep stop event for main loop exit
        return False

    def exit_demo_mode(self):
        if self.app.app_state == AppState.PLAYING:
            print("Exiting Demo Mode...")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.deselect_dragged_shape()
            self.app.app_state = AppState.MAIN_MENU
            # Removed setting is_process_running and pause_event
            self.check_initial_completion_status()  # Re-check status
            self.app.status = "Ready"  # Set status back to Ready

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
        """Deletes current run's checkpoint and re-initializes components."""
        from config.general import get_run_checkpoint_dir  # Local import

        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        # Removed pause_event setting and is_process_running
        self.app.app_state = AppState.INITIALIZING
        self.app.status = "Cleaning"
        messages = []

        # Render cleaning status
        if self.app.renderer:
            try:
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=False,  # No process running now
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
                    worker_counts={},  # Removed worker counts
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

        # --- Stop existing worker threads (if any) ---
        # Keep this structure in case new workers (MCTS/NN) are added later
        print("[Cleanup] Stopping existing worker threads (if any)...")
        self.app.worker_manager.stop_worker_threads()
        print("[Cleanup] Existing worker threads stopped.")
        # --- End Stop Workers ---

        # --- Close Stats Recorder ---
        print("[Cleanup] Closing stats recorder...")
        self.app.initializer.close_stats_recorder(is_cleanup=True)
        print("[Cleanup] Stats recorder closed.")
        # --- End Close Stats ---

        # --- Delete Checkpoint Directory ---
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
        # --- End Delete Checkpoint ---

        time.sleep(0.1)
        print("[Cleanup] Re-initializing components...")
        try:
            # Re-initialize components (NN, Stats, Checkpoint Manager)
            self.app.initializer.initialize_rl_components(
                is_reinit=True, checkpoint_to_load=None
            )
            print("[Cleanup] Components re-initialized.")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
                print("[Cleanup] Demo env reset.")

            # --- Start new worker threads (if any) ---
            # Keep this structure for future workers
            print("[Cleanup] Starting new worker threads (if any)...")
            self.app.worker_manager.start_worker_threads()
            print("[Cleanup] New worker threads started.")
            # --- End Start Workers ---

            print("[Cleanup] Component re-initialization and worker start successful.")
            messages.append("Components re-initialized.")

            # --- Set state after cleanup ---
            # Removed is_process_running and pause_event
            self.app.status = "Ready"
            self.app.app_state = AppState.MAIN_MENU
            print("[Cleanup] Application state set to MAIN_MENU.")
            # --- End Set State ---

        except Exception as e:
            print(f"FATAL ERROR during re-initialization after cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Re-init Failed"
            self.app.app_state = AppState.ERROR
            messages.append("ERROR RE-INITIALIZING COMPONENTS!")
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
        """Saves checkpoint (e.g., NN weights) if in main menu."""
        # Removed check for is_process_running
        if (
            self.app.app_state == AppState.MAIN_MENU
            and self.app.initializer.checkpoint_manager
            and self.app.initializer.stats_recorder  # Check if stats exist
            and hasattr(self.app.initializer.stats_recorder, "aggregator")
        ):
            print("Saving checkpoint...")
            try:
                # Get step/episode count from aggregator
                current_step = getattr(
                    self.app.initializer.stats_recorder.aggregator.storage,
                    "current_global_step",
                    0,
                )
                episode_count = getattr(
                    self.app.initializer.stats_recorder.aggregator.storage,
                    "total_episodes",
                    0,
                )
                # Target step is now managed within aggregator/checkpoint manager
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
        """Saves the final checkpoint (e.g., NN weights)."""
        if (
            self.app.initializer.checkpoint_manager
            and self.app.initializer.stats_recorder
            and hasattr(self.app.initializer.stats_recorder, "aggregator")
        ):
            save_on_exit = (
                self.app.status != "Cleaning" and self.app.app_state != AppState.ERROR
            )

            if save_on_exit:
                print("Performing final checkpoint save...")
                try:
                    current_step = getattr(
                        self.app.initializer.stats_recorder.aggregator.storage,
                        "current_global_step",
                        0,
                    )
                    episode_count = getattr(
                        self.app.initializer.stats_recorder.aggregator.storage,
                        "total_episodes",
                        0,
                    )
                    target_step = getattr(
                        self.app.initializer.checkpoint_manager,
                        "training_target_step",
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
                    traceback.print_exc()
            else:
                print("Skipping final checkpoint save.")
