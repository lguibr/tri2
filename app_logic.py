import pygame
import time
import traceback
import os
import shutil
from typing import TYPE_CHECKING, Tuple, Dict, Any, Optional
import logging
import ray  # Added Ray

from app_state import AppState
from config.general import get_run_checkpoint_dir

if TYPE_CHECKING:
    LogicAppState = Any
    StatsAggregatorHandle = ray.actor.ActorHandle  # Type hint

logger = logging.getLogger(__name__)


class AppLogic:
    """Handles the core application logic and state transitions within the Logic Process."""

    def __init__(self, app: "LogicAppState"):
        self.app = app

    def check_initial_completion_status(self):
        pass

    def update_status_and_check_completion(self):
        is_running = self.app.worker_manager.is_any_worker_running()
        state = self.app.app_state
        new_status = self.app.status

        if state == AppState.MAIN_MENU:
            new_status = (
                "Confirm Cleanup"
                if self.app.cleanup_confirmation_active
                else "Running AlphaZero" if is_running else "Ready"
            )
        elif state == AppState.PLAYING:
            new_status = "Playing Demo"
        elif state == AppState.DEBUG:
            new_status = "Debugging Grid"
        elif state == AppState.INITIALIZING:
            new_status = "Initializing..."
        elif state == AppState.ERROR:
            new_status = self.app.status
        elif state == AppState.CLEANING:
            new_status = "Cleaning"

        if new_status != self.app.status:
            self.app.set_status(new_status)

    def start_run(self):
        if (
            self.app.app_state != AppState.MAIN_MENU
            or self.app.worker_manager.is_any_worker_running()
        ):
            logger.warning(
                "[AppLogic] Cannot start run: Not in Main Menu or already running."
            )
            return
        logger.info("[AppLogic] Starting AlphaZero Run (Self-Play & Training)...")
        self.app.worker_manager.start_all_workers()  # Starts actor loops
        self.update_status_and_check_completion()

    def stop_run(self):
        if not self.app.worker_manager.is_any_worker_running():
            logger.info("[AppLogic] Run not currently active.")
            return
        logger.info("[AppLogic] Stop Run command received. Initiating worker stop...")
        self.app.worker_manager.stop_all_workers()  # Stops actors
        self.update_status_and_check_completion()

    def request_cleanup(self):
        if self.app.app_state != AppState.MAIN_MENU:
            return
        if self.app.worker_manager.is_any_worker_running():
            self._set_temp_message("Stop Run before Cleanup!")
            return
        self.app.set_cleanup_confirmation(True)
        self.update_status_and_check_completion()
        logger.info("[AppLogic] Cleanup requested. Confirm action.")

    def start_demo_mode(self):
        if self._can_start_mode("Demo"):
            logger.info("[AppLogic] Entering Demo Mode...")
            self.try_save_checkpoint()
            self.app.set_state(AppState.PLAYING)
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
            self.update_status_and_check_completion()

    def start_debug_mode(self):
        if self._can_start_mode("Debug"):
            logger.info("[AppLogic] Entering Debug Mode...")
            self.try_save_checkpoint()
            self.app.set_state(AppState.DEBUG)
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
            self.update_status_and_check_completion()

    def _can_start_mode(self, mode_name: str) -> bool:
        if self.app.initializer.demo_env is None:
            logger.warning(f"[AppLogic] Cannot start {mode_name}: Env not initialized.")
            return False
        if self.app.app_state != AppState.MAIN_MENU:
            logger.warning(
                f"[AppLogic] Cannot start {mode_name} mode outside MainMenu."
            )
            return False
        if self.app.worker_manager.is_any_worker_running():
            self._set_temp_message(f"Stop Run before {mode_name}!")
            return False
        return True

    def exit_demo_mode(self):
        if self.app.app_state == AppState.PLAYING:
            logger.info("[AppLogic] Exiting Demo Mode...")
            self._return_to_main_menu()

    def exit_debug_mode(self):
        if self.app.app_state == AppState.DEBUG:
            logger.info("[AppLogic] Exiting Debug Mode...")
            self._return_to_main_menu()

    def _return_to_main_menu(self):
        self.app.set_state(AppState.MAIN_MENU)
        self.check_initial_completion_status()
        self.update_status_and_check_completion()

    def cancel_cleanup(self):
        self.app.set_cleanup_confirmation(False)
        self._set_temp_message("Cleanup cancelled.")
        self.update_status_and_check_completion()
        logger.info("[AppLogic] Cleanup cancelled by user.")

    def confirm_cleanup(self):
        logger.info("[AppLogic] Cleanup confirmed by user. Starting process...")
        try:
            self._cleanup_data()
        except Exception as e:
            logger.error(f"[AppLogic] FATAL ERROR during cleanup: {e}", exc_info=True)
            self.app.set_status("Error: Cleanup Failed Critically")
            self.app.set_state(AppState.ERROR)
        finally:
            self.app.set_cleanup_confirmation(False)
            logger.info(
                f"[AppLogic] Cleanup process finished. State: {self.app.app_state}, Status: {self.app.status}"
            )

    def handle_demo_mouse_motion(self, payload: Optional[Dict]):
        if (
            self.app.app_state != AppState.PLAYING
            or not self.app.initializer.demo_env
            or not payload
        ):
            return
        grid_coords = payload.get("pos")
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if demo_env.demo_dragged_shape_idx is None:
            return
        demo_env.update_snapped_position(grid_coords)

    def handle_demo_mouse_button_down(self, payload: Optional[Dict]):
        if (
            self.app.app_state != AppState.PLAYING
            or not self.app.initializer.demo_env
            or not payload
        ):
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        click_type = payload.get("type")
        if click_type == "preview":
            clicked_preview = payload.get("index")
            if clicked_preview is not None:
                action = (
                    demo_env.deselect_dragged_shape
                    if clicked_preview == demo_env.demo_dragged_shape_idx
                    else lambda: demo_env.select_shape_for_drag(clicked_preview)
                )
                action()
        elif click_type == "grid":
            grid_coords = payload.get("grid_coords")
            if (
                grid_coords is not None
                and demo_env.demo_dragged_shape_idx is not None
                and demo_env.demo_snapped_position == grid_coords
            ):
                placed = demo_env.place_dragged_shape()
                if placed and demo_env.is_over():
                    logger.info("[Demo] Game Over! (UI handles exit prompt)")
            else:
                demo_env.deselect_dragged_shape()
        elif click_type == "outside":
            demo_env.deselect_dragged_shape()

    def handle_debug_input(self, payload: Optional[Dict]):
        if (
            self.app.app_state != AppState.DEBUG
            or not self.app.initializer.demo_env
            or not payload
        ):
            return
        demo_env = self.app.initializer.demo_env
        input_type = payload.get("type")
        if input_type == "reset":
            logger.info("[Debug] Resetting grid...")
            demo_env.reset()
        elif input_type == "toggle_triangle":
            clicked_coords = payload.get("grid_coords")
            if clicked_coords:
                demo_env.toggle_triangle_debug(*clicked_coords)

    def _set_temp_message(self, message: str):
        self.app.set_cleanup_message(message, time.time())

    def _cleanup_data(self):
        logger.info("\n[AppLogic] --- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.app.set_state(AppState.CLEANING)
        self.app.set_status("Cleaning")
        messages = []
        logger.info("[AppLogic Cleanup] Stopping existing worker actors (if any)...")
        self.app.worker_manager.stop_all_workers()  # Stops actors
        logger.info("[AppLogic Cleanup] Existing worker actors stopped.")
        logger.info("[AppLogic Cleanup] Closing stats recorder...")
        self.app.initializer.close_stats_recorder(is_cleanup=True)
        logger.info("[AppLogic Cleanup] Stats recorder closed.")
        messages.append(self._delete_checkpoint_dir())
        time.sleep(0.1)
        logger.info("[AppLogic Cleanup] Re-initializing components...")
        try:
            # Re-init actors (handled by initializer)
            self.app.initializer._init_ray_actors()
            self.app.initializer.initialize_rl_components(
                is_reinit=True, checkpoint_to_load=None
            )
            logger.info("[AppLogic Cleanup] RL Components re-initialized.")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
            self.app.worker_manager.initialize_actors()  # Re-init worker actors
            logger.info("[AppLogic Cleanup] Workers re-initialized (not started).")
            messages.append("Components re-initialized.")
            self.app.set_status("Ready")
            self.app.set_state(AppState.MAIN_MENU)
        except Exception as e:
            logger.error(
                f"[AppLogic] FATAL ERROR during re-initialization after cleanup: {e}",
                exc_info=True,
            )
            self.app.set_status("Error: Re-init Failed")
            self.app.set_state(AppState.ERROR)
            messages.append("ERROR RE-INITIALIZING COMPONENTS!")
        self._set_temp_message("\n".join(messages))
        logger.info(
            f"[AppLogic] --- CLEANUP DATA COMPLETE (Final State: {self.app.app_state}) ---"
        )

    def _delete_checkpoint_dir(self) -> str:
        logger.info("[AppLogic Cleanup] Deleting agent checkpoint file/dir...")
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
        logger.info(f"  - {msg}")
        logger.info("[AppLogic Cleanup] Checkpoint deletion attempt finished.")
        return msg

    def try_save_checkpoint(self):
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
        logger.info("[AppLogic] Saving checkpoint...")
        try:
            # Fetch stats from aggregator actor
            agg_actor: "StatsAggregatorHandle" = self.app.initializer.stats_aggregator
            step_ref = agg_actor.get_current_global_step.remote()
            ep_ref = agg_actor.get_total_episodes.remote()
            target_ref = agg_actor.get_training_target_step.remote()  # Use new getter
            current_step, episode_count, target_step = ray.get(
                [step_ref, ep_ref, target_ref]
            )

            self.app.initializer.checkpoint_manager.save_checkpoint(
                current_step,
                episode_count,
                training_target_step=target_step,
                is_final=False,
            )
        except Exception as e:
            logger.error(f"[AppLogic] Error saving checkpoint: {e}", exc_info=True)

    def save_final_checkpoint(self):
        if (
            not hasattr(self.app, "initializer")
            or not self.app.initializer.checkpoint_manager
            or not self.app.initializer.stats_aggregator
        ):
            logger.warning(
                "[AppLogic] Cannot save final checkpoint: components missing."
            )
            return
        save_on_exit = (
            self.app.app_state != AppState.CLEANING
            and self.app.app_state != AppState.ERROR
        )
        if save_on_exit:
            logger.info("[AppLogic] Performing final checkpoint save...")
            try:
                # Fetch stats from aggregator actor
                agg_actor: "StatsAggregatorHandle" = (
                    self.app.initializer.stats_aggregator
                )
                step_ref = agg_actor.get_current_global_step.remote()
                ep_ref = agg_actor.get_total_episodes.remote()
                target_ref = (
                    agg_actor.get_training_target_step.remote()
                )  # Use new getter
                current_step, episode_count, target_step = ray.get(
                    [step_ref, ep_ref, target_ref]
                )

                self.app.initializer.checkpoint_manager.save_checkpoint(
                    current_step,
                    episode_count,
                    training_target_step=target_step,
                    is_final=True,
                )
                logger.info("[AppLogic] Final checkpoint save successful.")
            except AttributeError as ae:
                # This specific error might be less likely now, but keep general exception handling
                logger.error(
                    f"[AppLogic] Attribute error during final checkpoint save: {ae}",
                    exc_info=True,
                )
            except Exception as final_save_err:
                logger.error(
                    f"[AppLogic] Error during final checkpoint save: {final_save_err}",
                    exc_info=True,
                )
        else:
            logger.info(
                f"[AppLogic] Skipping final checkpoint save due to state: {self.app.app_state}"
            )
