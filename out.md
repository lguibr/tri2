File: analyze_profile.py
# File: analyze_profile_v2.py
import pstats
from pstats import SortKey

profile_file = "profile_output.prof"
output_file_cumulative = "profile_summary_cumulative.txt"
output_file_tottime = "profile_summary_tottime.txt"
num_lines_to_print = 50  # You can adjust how many lines to show

try:
    # --- Sort by Cumulative Time ---
    print(
        f"Saving top {num_lines_to_print} cumulative time stats to {output_file_cumulative}..."
    )
    with open(output_file_cumulative, "w") as f_cum:
        # Pass the file handle directly as the stream
        stats_cum = pstats.Stats(profile_file, stream=f_cum)
        stats_cum.sort_stats(SortKey.CUMULATIVE).print_stats(num_lines_to_print)
        # 'with open' handles closing/flushing
    print("Done.")

    # --- Sort by Total Time (Internal) ---
    print(
        f"Saving top {num_lines_to_print} total time (tottime) stats to {output_file_tottime}..."
    )
    with open(output_file_tottime, "w") as f_tot:
        # Pass the file handle directly as the stream
        stats_tot = pstats.Stats(profile_file, stream=f_tot)
        stats_tot.sort_stats(SortKey.TIME).print_stats(
            num_lines_to_print
        )  # SortKey.TIME is 'tottime'
        # 'with open' handles closing/flushing
    print("Done.")

    print(
        f"\nAnalysis complete. Check '{output_file_cumulative}' and '{output_file_tottime}'."
    )

except FileNotFoundError:
    print(f"ERROR: Profile file '{profile_file}' not found.")
except Exception as e:
    print(f"An error occurred during profile analysis: {e}")


File: app_init.py
import pygame
import time
import traceback
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import TYPE_CHECKING, List, Optional, Any
import multiprocessing as mp
import ray
from ray.util.queue import Queue as RayQueue
import logging  # Added logging

from config import (
    ModelConfig,
    StatsConfig,
    DemoConfig,
    MCTSConfig,
    EnvConfig,
    TrainConfig,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
)
from environment.game_state import GameState
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregatorActor  # Import Actor
from stats.simple_stats_recorder import SimpleStatsRecorder

from training.checkpoint_manager import CheckpointManager
from app_state import AppState
from mcts import MCTS
from agent.alphazero_net import AlphaZeroNet, AgentPredictor

# Workers managed by AppWorkerManager

if TYPE_CHECKING:
    LogicAppState = Any
    from torch.optim.lr_scheduler import _LRScheduler

    AgentPredictorHandle = ray.actor.ActorHandle
    SelfPlayWorkerHandle = ray.actor.ActorHandle
    TrainingWorkerHandle = ray.actor.ActorHandle
    StatsAggregatorHandle = ray.actor.ActorHandle  # Use Actor Handle type

logger = logging.getLogger(__name__)  # Added logger


class AppInitializer:
    """Handles the initialization of core RL application components in the Logic Process."""

    def __init__(self, app: "LogicAppState"):
        self.app = app
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        self.train_config = app.train_config_instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.demo_config = DemoConfig()
        self.mcts_config = MCTSConfig()
        self.worker_stop_event: mp.Event = app.worker_stop_event

        # Components to be initialized
        self.agent: Optional[AlphaZeroNet] = None
        self.agent_predictor: Optional["AgentPredictorHandle"] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional["_LRScheduler"] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.stats_aggregator: Optional["StatsAggregatorHandle"] = (
            None  # Now an Actor Handle
        )
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        # MCTS instance not central anymore

    def initialize_logic_components(self):
        """Initializes only the RL and logic-related components, including Ray actors."""
        try:
            self._check_gpu_memory()
            self._init_ray_actors()  # Initialize actors first
            self.initialize_rl_components(
                is_reinit=False,
                checkpoint_to_load=self.app.checkpoint_to_load,
            )
            self.initialize_demo_env()
            self._calculate_agent_params()
            self.app.worker_manager.initialize_actors()  # Initialize worker actors

        except Exception as init_err:
            self._handle_init_error(init_err)

    def _init_ray_actors(self):
        """Initializes core Ray actors like AgentPredictor and StatsAggregatorActor."""
        logger.info("[AppInitializer] Initializing Ray Actors...")
        # --- Agent Predictor Actor ---
        try:
            self.agent_predictor = AgentPredictor.options(
                name="AgentPredictorActor",  # Optional: give it a name
                lifetime="detached",  # Optional: keep actor alive if main script exits abnormally
            ).remote(
                env_config=self.env_config, model_config=self.model_config.Network()
            )
            ray.get(self.agent_predictor.health_check.remote())  # Wait for actor
            logger.info("[AppInitializer] AgentPredictor actor created.")
            self.app.agent_predictor = self.agent_predictor
        except Exception as e:
            logger.error(
                f"[AppInitializer] Failed to create AgentPredictor actor: {e}",
                exc_info=True,
            )
            raise RuntimeError("AgentPredictor actor initialization failed") from e

        # --- Stats Aggregator Actor ---
        try:
            self.stats_aggregator = StatsAggregatorActor.options(
                name="StatsAggregatorActor", lifetime="detached"
            ).remote(
                avg_windows=self.stats_config.STATS_AVG_WINDOW,
                plot_window=self.stats_config.PLOT_DATA_WINDOW,
            )
            ray.get(self.stats_aggregator.health_check.remote())  # Wait for actor
            logger.info("[AppInitializer] StatsAggregatorActor created.")
            self.app.stats_aggregator = self.stats_aggregator  # Store actor handle
        except Exception as e:
            logger.error(
                f"[AppInitializer] Failed to create StatsAggregator actor: {e}",
                exc_info=True,
            )
            raise RuntimeError("StatsAggregator actor initialization failed") from e

        logger.info("[AppInitializer] Ray Actors initialized.")

    def _check_gpu_memory(self):
        """Checks and prints total GPU memory if available."""
        if self.app.device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(self.app.device)
                self.app.total_gpu_memory_bytes = props.total_memory
                print(f"Total GPU Memory: {props.total_memory / (1024**3):.2f} GB")
            except Exception as e:
                print(f"Warning: Could not get total GPU memory: {e}")

    def _calculate_agent_params(self):
        """Calculates agent parameters by calling the AgentPredictor actor."""
        if self.agent_predictor:
            try:
                param_count_ref = self.agent_predictor.get_param_count.remote()
                self.agent_param_count = ray.get(param_count_ref)
                logger.info(
                    f"[AppInitializer] Agent Parameters: {self.agent_param_count:,}"
                )
            except Exception as e:
                logger.error(f"Warning: Could not get agent parameters from actor: {e}")
                self.agent_param_count = 0
        else:
            logger.warning("AgentPredictor actor not available for param count.")
            self.agent_param_count = 0

    def _handle_init_error(self, error: Exception):
        """Handles fatal errors during component initialization."""
        print(f"FATAL ERROR during component initialization: {error}")
        traceback.print_exc()
        self.app.set_state(AppState.ERROR)
        self.app.set_status(f"Logic Init Failed: {error}")
        self.app.stop_event.set()

    def initialize_rl_components(
        self, is_reinit: bool = False, checkpoint_to_load: Optional[str] = None
    ):
        """Initializes local NN Agent (for checkpointing), Optimizer, Scheduler, StatsRecorder, Checkpoint Manager."""
        print(f"Initializing AlphaZero components... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            self._init_local_agent_for_checkpointing()
            self._init_optimizer_and_scheduler()
            self._init_stats_recorder()  # Uses stats_aggregator handle now
            self._init_checkpoint_manager(
                checkpoint_to_load
            )  # Interacts with stats_aggregator handle

            print(
                f"AlphaZero components initialized in {time.time() - start_time:.2f}s"
            )
            self.app.optimizer = self.optimizer
            self.app.scheduler = self.scheduler
            self.app.stats_recorder = self.stats_recorder
            self.app.checkpoint_manager = self.checkpoint_manager

        except Exception as e:
            print(f"Error during AlphaZero component initialization: {e}")
            traceback.print_exc()
            raise e

    def _init_local_agent_for_checkpointing(self):
        """Initializes a local copy of the agent for saving/loading checkpoints."""
        self.agent = AlphaZeroNet(
            env_config=self.env_config, model_config=self.model_config.Network()
        ).to(self.app.device)
        print(
            f"Local AlphaZeroNet (for checkpointing) initialized on device: {self.app.device}."
        )

    def _init_optimizer_and_scheduler(self):
        """Initializes the optimizer and scheduler using the local agent copy."""
        if not self.agent:
            raise RuntimeError("Local Agent must be initialized before Optimizer.")
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.train_config.LEARNING_RATE,
            weight_decay=self.train_config.WEIGHT_DECAY,
        )
        print(
            f"Optimizer initialized (Adam, LR={self.train_config.LEARNING_RATE}) for local agent."
        )

        if self.train_config.USE_LR_SCHEDULER:
            if self.train_config.SCHEDULER_TYPE == "CosineAnnealingLR":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.train_config.SCHEDULER_T_MAX,
                    eta_min=self.train_config.SCHEDULER_ETA_MIN,
                )
                print(
                    f"LR Scheduler initialized (CosineAnnealingLR, T_max={self.train_config.SCHEDULER_T_MAX}, eta_min={self.train_config.SCHEDULER_ETA_MIN})."
                )
            else:
                print(
                    f"Warning: Unknown scheduler type '{self.train_config.SCHEDULER_TYPE}'. No scheduler initialized."
                )
                self.scheduler = None
        else:
            print("LR Scheduler is DISABLED.")
            self.scheduler = None

    def _init_stats_recorder(self):
        """Initializes the local StatsRecorder, passing the StatsAggregatorActor handle."""
        if not self.stats_aggregator:  # Check if handle exists
            raise RuntimeError(
                "StatsAggregatorActor handle must be initialized before StatsRecorder."
            )
        print("Initializing SimpleStatsRecorder...")
        self.stats_recorder = SimpleStatsRecorder(
            aggregator=self.stats_aggregator,  # Pass actor handle
            console_log_interval=self.stats_config.CONSOLE_LOG_FREQ,
            train_config=self.train_config,
        )
        print("SimpleStatsRecorder initialized.")

    def _init_checkpoint_manager(self, checkpoint_to_load: Optional[str]):
        """Initializes the CheckpointManager using local agent/optimizer and StatsAggregatorActor handle."""
        if not self.agent or not self.optimizer or not self.stats_aggregator:
            raise RuntimeError(
                "Local Agent, Optimizer, and StatsAggregatorActor handle needed for CheckpointManager."
            )
        self.checkpoint_manager = CheckpointManager(
            agent=self.agent,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            stats_aggregator=self.stats_aggregator,  # Pass actor handle
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            run_checkpoint_dir=get_run_checkpoint_dir(),
            load_checkpoint_path_config=checkpoint_to_load,
            device=self.app.device,
        )
        if self.checkpoint_manager.get_checkpoint_path_to_load():
            # Load checkpoint into local agent/optimizer/scheduler
            # CheckpointManager's load_checkpoint method now handles loading stats into the actor
            self.checkpoint_manager.load_checkpoint()
            # Push loaded weights to the AgentPredictor actor
            if self.agent_predictor:
                try:
                    loaded_weights = self.agent.state_dict()
                    set_ref = self.agent_predictor.set_weights.remote(loaded_weights)
                    ray.get(set_ref)
                    logger.info(
                        "[AppInitializer] Pushed loaded checkpoint weights to AgentPredictor actor."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to push loaded weights to AgentPredictor: {e}"
                    )
            # Update App state based on loaded checkpoint AFTER loading
            # Get step count from the aggregator actor
            if self.stats_aggregator:
                try:
                    step_ref = self.stats_aggregator.get_current_global_step.remote()
                    self.app.current_global_step = ray.get(step_ref)
                except Exception as e:
                    logger.error(
                        f"Failed to get global step from StatsAggregator actor: {e}"
                    )
                    self.app.current_global_step = 0  # Fallback
            else:
                self.app.current_global_step = 0

    def initialize_demo_env(self):
        """Initializes the separate environment for demo/debug if needed by logic."""
        print("Initializing Demo/Debug Environment (in Logic Process)...")
        try:
            self.demo_env = GameState()
            self.demo_env.reset()
            print("Demo/Debug environment initialized.")
            self.app.demo_env = self.demo_env
        except Exception as e:
            print(f"ERROR initializing demo/debug environment: {e}")
            traceback.print_exc()
            self.demo_env = None
            self.app.demo_env = None

    def close_stats_recorder(self, is_cleanup: bool = False):
        """Safely closes the stats recorder (if it exists)."""
        print(
            f"[AppInitializer] close_stats_recorder called (is_cleanup={is_cleanup})..."
        )
        # StatsAggregator is now an actor, termination handled elsewhere (e.g., AppWorkerManager or main shutdown)
        if self.stats_recorder and hasattr(self.stats_recorder, "close"):
            print("[AppInitializer] Stats recorder exists, attempting close...")
            try:
                # Recorder might need to make final calls to the aggregator actor before closing
                self.stats_recorder.close(is_cleanup=is_cleanup)
                print("[AppInitializer] stats_recorder.close() executed.")
            except Exception as log_e:
                print(f"[AppInitializer] Error closing stats recorder on exit: {log_e}")
                traceback.print_exc()
        else:
            print("[AppInitializer] No stats recorder instance or close method.")
        print("[AppInitializer] close_stats_recorder finished.")


File: app_logic.py
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


File: app_setup.py
import os
import pygame
from typing import Tuple, Dict, Any

from config import (
    VisConfig,
    get_run_checkpoint_dir,
    get_run_log_dir,
    get_config_dict,
    print_config_info_and_validate,
)


def initialize_pygame(
    vis_config: VisConfig,
) -> Tuple[pygame.Surface, pygame.time.Clock]:
    """Initializes Pygame, sets up the screen and clock."""
    print("Initializing Pygame...")
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (vis_config.SCREEN_WIDTH, vis_config.SCREEN_HEIGHT), pygame.RESIZABLE
    )
    pygame.display.set_caption("AlphaTri Trainer")  # Updated caption
    clock = pygame.time.Clock()
    print("Pygame initialized.")
    return screen, clock


def initialize_directories():
    """Creates necessary runtime directories using dynamic paths."""
    run_checkpoint_dir = get_run_checkpoint_dir()
    run_log_dir = get_run_log_dir()
    # Console log dir is created within main_pygame before logger init

    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"Ensured directories exist: {run_checkpoint_dir}, {run_log_dir}")


def load_and_validate_configs() -> Dict[str, Any]:
    """Loads all config classes and returns the combined config dictionary."""
    config_dict = get_config_dict()
    print_config_info_and_validate()
    return config_dict


File: app_state.py
from enum import Enum


class AppState(Enum):
    INITIALIZING = "Initializing"
    MAIN_MENU = "MainMenu"
    PLAYING = "Playing" 
    DEBUG = "Debug"
    CLEANUP_CONFIRM = "Confirm Cleanup"
    CLEANING = "Cleaning"
    ERROR = "Error"
    UNKNOWN = "Unknown"


File: app_ui_utils.py
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from main_pygame import MainApp
    from environment.game_state import GameState
    from ui.renderer import UIRenderer


class AppUIUtils:
    """Utility functions related to mapping screen coordinates to game elements."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def map_screen_to_grid(
        self, screen_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Maps screen coordinates to grid row/column for demo/debug."""
        if (
            self.app.renderer is None
            or self.app.initializer.demo_env is None
            or self.app.renderer.demo_renderer is None
        ):
            return None
        if self.app.app_state not in [
            self.app.app_state.PLAYING,
            self.app.app_state.DEBUG,
        ]:
            return None

        demo_env: "GameState" = self.app.initializer.demo_env
        renderer: "UIRenderer" = self.app.renderer

        screen_width, screen_height = self.app.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        _, clipped_game_rect = renderer.demo_renderer._calculate_game_area_rect(
            screen_width,
            screen_height,
            padding,
            hud_height,
            help_height,
            self.app.env_config,
        )

        if not clipped_game_rect.collidepoint(screen_pos):
            return None

        relative_x = screen_pos[0] - clipped_game_rect.left
        relative_y = screen_pos[1] - clipped_game_rect.top

        tri_cell_w, tri_cell_h = renderer.demo_renderer._calculate_demo_triangle_size(
            clipped_game_rect.width, clipped_game_rect.height, self.app.env_config
        )
        grid_ox, grid_oy = renderer.demo_renderer._calculate_grid_offset(
            clipped_game_rect.width, clipped_game_rect.height, self.app.env_config
        )

        if tri_cell_w <= 0 or tri_cell_h <= 0:
            return None

        grid_relative_x = relative_x - grid_ox
        grid_relative_y = relative_y - grid_oy

        # Approximate calculation (might need refinement based on triangle geometry)
        approx_row = int(grid_relative_y / tri_cell_h)
        approx_col = int(grid_relative_x / (tri_cell_w * 0.75))

        if (
            0 <= approx_row < self.app.env_config.ROWS
            and 0 <= approx_col < self.app.env_config.COLS
        ):
            if (
                demo_env.grid.valid(approx_row, approx_col)
                and not demo_env.grid.triangles[approx_row][approx_col].is_death
            ):
                return approx_row, approx_col
        return None

    def map_screen_to_preview(self, screen_pos: Tuple[int, int]) -> Optional[int]:
        """Maps screen coordinates to a shape preview index."""
        if (
            self.app.renderer is None
            or self.app.initializer.demo_env is None
            or self.app.input_handler is None
        ):
            return None
        if self.app.app_state != self.app.app_state.PLAYING:
            return None

        # Access preview rects directly from the input handler
        if hasattr(self.app.input_handler, "shape_preview_rects"):
            for idx, rect in self.app.input_handler.shape_preview_rects.items():
                if rect.collidepoint(screen_pos):
                    return idx
        return None


File: app_workers.py
# File: app_workers.py
import threading
import queue
import time
from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Any
import logging
import multiprocessing as mp
import ray
import asyncio
import torch

# Import Ray actor classes
from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker

# Import Actor Handles for type hinting
if TYPE_CHECKING:
    LogicAppState = Any
    SelfPlayWorkerHandle = ray.actor.ActorHandle
    TrainingWorkerHandle = ray.actor.ActorHandle
    AgentPredictorHandle = ray.actor.ActorHandle
    StatsAggregatorHandle = ray.actor.ActorHandle
    from ray.util.queue import Queue as RayQueue

logger = logging.getLogger(__name__)


class AppWorkerManager:
    """Manages the creation, starting, and stopping of Ray worker actors."""

    DEFAULT_KILL_TIMEOUT = 5.0

    def __init__(self, app: "LogicAppState"):
        self.app = app
        self.self_play_worker_actors: List["SelfPlayWorkerHandle"] = []
        self.training_worker_actor: Optional["TrainingWorkerHandle"] = None
        self.agent_predictor_actor: Optional["AgentPredictorHandle"] = None
        self._workers_running = False
        logger.info("[AppWorkerManager] Initialized for Ray Actors.")

    def initialize_actors(self):
        """Initializes Ray worker actors (SelfPlay, Training). Does NOT start their loops."""
        logger.info("[AppWorkerManager] Initializing worker actors...")
        if not self.app.agent_predictor:
            logger.error(
                "[AppWorkerManager] ERROR: AgentPredictor actor not initialized in AppInitializer."
            )
            self.app.set_state(self.app.app_state.ERROR)
            self.app.set_status("Worker Init Failed: Missing AgentPredictor")
            return
        if not self.app.stats_aggregator:
            logger.error(
                "[AppWorkerManager] ERROR: StatsAggregator actor handle not initialized in AppInitializer."
            )
            self.app.set_state(self.app.app_state.ERROR)
            self.app.set_status("Worker Init Failed: Missing StatsAggregator")
            return

        self.agent_predictor_actor = self.app.agent_predictor

        self._init_self_play_actors()
        self._init_training_actor()

        num_sp = len(self.self_play_worker_actors)
        num_tr = 1 if self.training_worker_actor else 0
        logger.info(
            f"Worker actors initialized ({num_sp} Self-Play, {num_tr} Training)."
        )

    def _init_self_play_actors(self):
        """Creates SelfPlayWorker Ray actors."""
        self.self_play_worker_actors = []
        num_sp_workers = self.app.train_config_instance.NUM_SELF_PLAY_WORKERS
        logger.info(f"Initializing {num_sp_workers} SelfPlayWorker actor(s)...")
        for i in range(num_sp_workers):
            try:
                actor = SelfPlayWorker.remote(
                    worker_id=i,
                    agent_predictor=self.agent_predictor_actor,
                    mcts_config=self.app.mcts_config,
                    env_config=self.app.env_config,
                    experience_queue=self.app.experience_queue,
                    stats_aggregator=self.app.stats_aggregator,
                    max_game_steps=None,
                )
                self.self_play_worker_actors.append(actor)
                logger.info(f"  SelfPlayWorker-{i} actor created.")
            except Exception as e:
                logger.error(
                    f"  ERROR creating SelfPlayWorker-{i} actor: {e}", exc_info=True
                )

    def _init_training_actor(self):
        """Creates the TrainingWorker Ray actor."""
        logger.info("Initializing TrainingWorker actor...")
        if not self.app.optimizer or not self.app.train_config_instance:
            logger.error(
                "[AppWorkerManager] ERROR: Optimizer or TrainConfig missing for TrainingWorker init."
            )
            return

        optimizer_cls = type(self.app.optimizer)
        optimizer_kwargs = self.app.optimizer.defaults

        scheduler_cls = type(self.app.scheduler) if self.app.scheduler else None
        scheduler_kwargs = {}
        if self.app.scheduler and hasattr(self.app.scheduler, "state_dict"):
            sd = self.app.scheduler.state_dict()
            if isinstance(
                self.app.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
            ):
                scheduler_kwargs = {
                    "T_max": sd.get("T_max", 1000),
                    "eta_min": sd.get("eta_min", 0),
                }
            else:
                logger.warning(
                    f"Cannot automatically determine kwargs for scheduler type {scheduler_cls}. Scheduler might not be correctly re-initialized in actor."
                )
                scheduler_cls = None

        try:
            actor = TrainingWorker.remote(
                agent_predictor=self.agent_predictor_actor,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                scheduler_cls=scheduler_cls,
                scheduler_kwargs=scheduler_kwargs,
                experience_queue=self.app.experience_queue,
                stats_aggregator=self.app.stats_aggregator,
                train_config=self.app.train_config_instance,
            )
            self.training_worker_actor = actor
            logger.info("  TrainingWorker actor created.")
        except Exception as e:
            logger.error(f"  ERROR creating TrainingWorker actor: {e}", exc_info=True)

    def get_active_worker_counts(self) -> Dict[str, int]:
        """Returns the count of initialized worker actors."""
        sp_count = len(self.self_play_worker_actors)
        tr_count = 1 if self.training_worker_actor else 0
        return {"SelfPlay": sp_count, "Training": tr_count}

    def is_any_worker_running(self) -> bool:
        """Checks the internal flag indicating if workers have been started."""
        return self._workers_running

    async def get_worker_render_data_async(
        self, max_envs: int
    ) -> List[Optional[Dict[str, Any]]]:
        """Retrieves render data from active self-play actors asynchronously."""
        if not self.self_play_worker_actors:
            return [None] * max_envs

        tasks = []
        num_to_fetch = min(len(self.self_play_worker_actors), max_envs)
        for i in range(num_to_fetch):
            actor = self.self_play_worker_actors[i]
            tasks.append(actor.get_current_render_data.remote())

        render_data_list: List[Optional[Dict[str, Any]]] = []
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting render data from worker {i}: {result}")
                    render_data_list.append(None)
                else:
                    render_data_list.append(result)
        except Exception as e:
            logger.error(f"Error gathering render data: {e}")
            render_data_list = [None] * num_to_fetch

        while len(render_data_list) < max_envs:
            render_data_list.append(None)
        return render_data_list

    def get_worker_render_data(self, max_envs: int) -> List[Optional[Dict[str, Any]]]:
        """Synchronous wrapper for get_worker_render_data_async."""
        if not self.self_play_worker_actors:
            return [None] * max_envs

        refs = []
        num_to_fetch = min(len(self.self_play_worker_actors), max_envs)
        for i in range(num_to_fetch):
            actor = self.self_play_worker_actors[i]
            refs.append(actor.get_current_render_data.remote())

        render_data_list: List[Optional[Dict[str, Any]]] = []
        try:
            results = ray.get(refs)
            render_data_list.extend(results)
        except Exception as e:
            logger.error(f"Error getting render data via ray.get: {e}")
            render_data_list = [None] * num_to_fetch

        while len(render_data_list) < max_envs:
            render_data_list.append(None)
        return render_data_list

    def start_all_workers(self):
        """Starts the main loops of all initialized worker actors."""
        if self._workers_running:
            logger.warning("[AppWorkerManager] Workers already started.")
            return
        if not self.self_play_worker_actors and not self.training_worker_actor:
            logger.error("[AppWorkerManager] No worker actors initialized to start.")
            return

        logger.info("[AppWorkerManager] Starting all worker actor loops...")
        self._workers_running = True

        for i, actor in enumerate(self.self_play_worker_actors):
            try:
                actor.run_loop.remote()
                logger.info(f"  SelfPlayWorker-{i} actor loop started.")
            except Exception as e:
                logger.error(f"  ERROR starting SelfPlayWorker-{i} actor loop: {e}")

        if self.training_worker_actor:
            try:
                self.training_worker_actor.run_loop.remote()
                logger.info("  TrainingWorker actor loop started.")
            except Exception as e:
                logger.error(f"  ERROR starting TrainingWorker actor loop: {e}")

        if self.is_any_worker_running():
            self.app.set_status("Running AlphaZero")
            num_sp = len(self.self_play_worker_actors)
            num_tr = 1 if self.training_worker_actor else 0
            logger.info(
                f"[AppWorkerManager] Worker loops started ({num_sp} SP, {num_tr} TR)."
            )

    def stop_all_workers(self, timeout: float = DEFAULT_KILL_TIMEOUT):
        """Signals all worker actors to stop and attempts to terminate them."""
        if (
            not self._workers_running
            and not self.self_play_worker_actors
            and not self.training_worker_actor
        ):
            logger.info("[AppWorkerManager] No workers running or initialized to stop.")
            return

        logger.info("[AppWorkerManager] Stopping ALL worker actors...")
        self._workers_running = False

        actors_to_stop: List[ray.actor.ActorHandle] = []
        actors_to_stop.extend(self.self_play_worker_actors)
        if self.training_worker_actor:
            actors_to_stop.append(self.training_worker_actor)

        if not actors_to_stop:
            logger.info("[AppWorkerManager] No active actor handles found to stop.")
            return

        logger.info(
            f"[AppWorkerManager] Sending stop signal to {len(actors_to_stop)} actors..."
        )
        for actor in actors_to_stop:
            try:
                actor.stop.remote()
            except Exception as e:
                logger.warning(f"Error sending stop signal to actor {actor}: {e}")

        time.sleep(0.5)

        logger.info(f"[AppWorkerManager] Killing actors...")
        for actor in actors_to_stop:
            try:
                ray.kill(actor, no_restart=True)
                logger.info(f"  Killed actor {actor}.")
            except Exception as e:
                logger.error(f"  Error killing actor {actor}: {e}")

        self.self_play_worker_actors = []
        self.training_worker_actor = None

        self._clear_experience_queue()

        logger.info("[AppWorkerManager] All worker actors stopped/killed.")
        self.app.set_status("Ready")

    def _clear_experience_queue(self):
        """Safely clears the experience queue (assuming Ray Queue)."""
        logger.info("[AppWorkerManager] Clearing experience queue...")
        # Check if it's a RayQueue instance (which acts as a handle)
        if hasattr(self.app, "experience_queue") and isinstance(
            self.app.experience_queue, ray.util.queue.Queue
        ):
            try:
                # Call qsize() directly, it returns an ObjectRef
                qsize_ref = self.app.experience_queue.qsize()
                qsize = ray.get(qsize_ref)  # Use ray.get() to resolve the ObjectRef
                logger.info(
                    f"[AppWorkerManager] Experience queue size before potential drain: {qsize}"
                )
                # Optional drain logic can be added here if needed
                # Example: Drain items if size is large
                # if qsize > 100:
                #     logger.info("[AppWorkerManager] Draining experience queue...")
                #     while qsize > 0:
                #         try:
                #             # Use get_nowait_batch to drain efficiently
                #             items_ref = self.app.experience_queue.get_nowait_batch(100)
                #             items = ray.get(items_ref)
                #             if not items: break
                #             qsize_ref = self.app.experience_queue.qsize()
                #             qsize = ray.get(qsize_ref)
                #         except ray.exceptions.RayActorError: # Handle queue actor potentially gone
                #             logger.warning("[AppWorkerManager] Queue actor error during drain.")
                #             break
                #         except Exception as drain_e:
                #             logger.error(f"Error draining queue: {drain_e}")
                #             break
                #     logger.info("[AppWorkerManager] Experience queue drained.")

            except Exception as e:
                logger.error(f"Error accessing Ray queue size: {e}")
        else:
            logger.warning(
                "[AppWorkerManager] Experience queue not found or not a Ray Queue during clearing."
            )


File: check_gpu.py
import torch

print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is NOT available to PyTorch.")
    # You can add checks for drivers here if needed, but PyTorch check is primary
    try:
        import subprocess
        print("\nAttempting to run nvidia-smi...")
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout)
        else:
            print(f"nvidia-smi command failed or not found (return code {result.returncode}). Ensure NVIDIA drivers are installed.")
            print(f"stderr: {result.stderr}")
    except FileNotFoundError:
         print("nvidia-smi command not found. Ensure NVIDIA drivers are installed and in PATH.")
    except Exception as e:
         print(f"Error running nvidia-smi: {e}")

File: logger.py
import os
from typing import TextIO, Optional


class TeeLogger:
    """Redirects stdout/stderr to both the console and a log file."""

    def __init__(self, filepath: str, original_stream: TextIO):
        self.terminal = original_stream
        self.log_file: Optional[TextIO] = None
        try:
            log_dir = os.path.dirname(filepath)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            # Use buffering=1 for line buffering
            self.log_file = open(filepath, "w", encoding="utf-8", buffering=1)
            print(f"[TeeLogger] Logging console output to: {filepath}")
        except Exception as e:
            self.terminal.write(
                f"FATAL ERROR: Could not open log file {filepath}: {e}\n"
            )
            # Continue without file logging if opening fails

    def write(self, message: str):
        self.terminal.write(message)
        if self.log_file:
            try:
                self.log_file.write(message)
            except Exception:
                # Silently ignore errors writing to log file to avoid loops
                pass

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass  # Silently ignore errors flushing log file

    def close(self):
        if self.log_file:
            try:
                self.log_file.close()
                self.log_file = None
            except Exception as e:
                self.terminal.write(f"Warning: Error closing log file: {e}\n")

    def __del__(self):
        # Ensure file is closed if logger object is garbage collected
        self.close()


File: logic_process.py
import time
import queue
import logging
import logging.handlers
import multiprocessing as mp
import traceback
import sys
from typing import Optional, Dict, Any
import ray
from ray.util.queue import Queue as RayQueue
import asyncio  # Added asyncio

try:
    from config import VisConfig, EnvConfig, TrainConfig, MCTSConfig, set_device
    from utils.helpers import get_device as get_torch_device, set_random_seeds
    from utils.init_checks import run_pre_checks
    from app_state import AppState
    from app_init import AppInitializer
    from app_logic import AppLogic
    from app_workers import AppWorkerManager
    from app_setup import initialize_directories
    from environment.game_state import GameState
except ImportError as e:
    print(f"[Logic Process Import Error] {e}", file=sys.stderr)
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    sys.exit(1)

RENDER_DATA_SENTINEL = "RENDER_DATA"
COMMAND_SENTINEL = "COMMAND"
STOP_SENTINEL = "STOP"
ERROR_SENTINEL = "ERROR"
PAYLOAD_KEY = "payload"


def run_logic_process(
    stop_event: mp.Event,
    command_queue: mp.Queue,
    render_data_queue: mp.Queue,
    checkpoint_to_load: Optional[str],
    log_queue: Optional[mp.Queue] = None,
):
    ray_initialized = False
    try:
        if not ray.is_initialized():
            ray.init(logging_level=logging.WARNING, ignore_reinit_error=True)
            ray_initialized = True
            print("[Logic Process] Ray initialized.")
        else:
            print("[Logic Process] Ray already initialized.")
            ray_initialized = True
    except Exception as ray_init_err:
        print(
            f"[Logic Process] FATAL: Ray initialization failed: {ray_init_err}",
            file=sys.stderr,
        )
        stop_event.set()
        try:
            render_data_queue.put({ERROR_SENTINEL: f"Ray Init Failed: {ray_init_err}"})
        except Exception:
            pass
        return

    if log_queue:
        qh = logging.handlers.QueueHandler(log_queue)
        root = logging.getLogger()
        if not any(isinstance(h, logging.handlers.QueueHandler) for h in root.handlers):
            root.addHandler(qh)
            root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Logic Process starting...")
    logic_start_time = time.time()

    logic_app_state = None
    try:
        vis_config = VisConfig()
        env_config = EnvConfig()
        train_config_instance = TrainConfig()
        mcts_config = MCTSConfig()
        worker_stop_event = mp.Event()

        # Use Ray Queue Actor
        experience_ray_queue = RayQueue(
            maxsize=train_config_instance.BUFFER_CAPACITY * 2
        )
        logger.info(
            f"[Logic Process] Ray Experience Queue created (maxsize={train_config_instance.BUFFER_CAPACITY * 2})."
        )

        logic_app_state = type(
            "LogicAppState",
            (object,),
            {
                "vis_config": vis_config,
                "env_config": env_config,
                "train_config_instance": train_config_instance,
                "mcts_config": mcts_config,
                "app_state": AppState.INITIALIZING,
                "status": "Initializing...",
                "stop_event": stop_event,
                "worker_stop_event": worker_stop_event,
                "experience_queue": experience_ray_queue,  # Use Ray Queue handle
                "device": get_torch_device(),
                "checkpoint_to_load": checkpoint_to_load,
                "initializer": None,
                "logic": None,
                "worker_manager": None,
                "agent_predictor": None,
                "stats_aggregator": None,  # Actor handles
                "ui_utils": None,
                "cleanup_confirmation_active": False,
                "cleanup_message": "",
                "last_cleanup_message_time": 0.0,
                "total_gpu_memory_bytes": None,
                "current_global_step": 0,
                "set_state": lambda self, new_state: setattr(
                    self, "app_state", new_state
                ),
                "set_status": lambda self, new_status: setattr(
                    self, "status", new_status
                ),
                "set_cleanup_confirmation": lambda self, active: setattr(
                    self, "cleanup_confirmation_active", active
                ),
                "set_cleanup_message": lambda self, msg, msg_time: (
                    setattr(self, "cleanup_message", msg),
                    setattr(self, "last_cleanup_message_time", msg_time),
                ),
                "get_render_data": None,
            },
        )()
        set_device(logic_app_state.device)

        initializer = AppInitializer(logic_app_state)
        logic = AppLogic(logic_app_state)
        worker_manager = AppWorkerManager(logic_app_state)
        logic_app_state.initializer = initializer
        logic_app_state.logic = logic
        logic_app_state.worker_manager = worker_manager

        logger.info("Initializing directories...")
        initialize_directories()
        set_random_seeds(
            mcts_config.RANDOM_SEED if hasattr(mcts_config, "RANDOM_SEED") else 42
        )
        logger.info("Running pre-checks...")
        run_pre_checks()
        logger.info("Initializing RL components and Ray actors...")
        initializer.initialize_logic_components()  # Initializes actors

        logic_app_state.set_state(AppState.MAIN_MENU)
        logic_app_state.set_status("Ready")
        logic.check_initial_completion_status()
        logger.info("--- Logic Initialization Complete ---")

        # Define async get_render_data
        async def _get_render_data_async(app_obj) -> Dict[str, Any]:
            worker_render_task = None
            if (
                app_obj.worker_manager.is_any_worker_running()
                and app_obj.app_state == AppState.MAIN_MENU
            ):
                num_to_render = app_obj.vis_config.NUM_ENVS_TO_RENDER
                if num_to_render > 0:
                    worker_render_task = (
                        app_obj.worker_manager.get_worker_render_data_async(
                            num_to_render
                        )
                    )

            # Fetch stats data from StatsAggregatorActor
            plot_data_ref, summary_ref, best_game_ref = None, None, None
            if app_obj.stats_aggregator:  # Check if handle exists
                plot_data_ref = app_obj.stats_aggregator.get_plot_data.remote()
                summary_ref = app_obj.stats_aggregator.get_summary.remote(
                    app_obj.current_global_step
                )
                best_game_ref = (
                    app_obj.stats_aggregator.get_best_game_state_data.remote()
                )

            # Gather results concurrently
            results = await asyncio.gather(
                (
                    worker_render_task
                    if worker_render_task
                    else asyncio.sleep(0, result=[])
                ),  # Handle no task case
                plot_data_ref if plot_data_ref else asyncio.sleep(0, result={}),
                summary_ref if summary_ref else asyncio.sleep(0, result={}),
                best_game_ref if best_game_ref else asyncio.sleep(0, result=None),
                return_exceptions=True,  # Handle potential errors from remote calls
            )

            # Process results, handling potential errors
            worker_render_data_result = (
                results[0]
                if not isinstance(results[0], Exception)
                else ([None] * app_obj.vis_config.NUM_ENVS_TO_RENDER)
            )
            plot_data = results[1] if not isinstance(results[1], Exception) else {}
            stats_summary = results[2] if not isinstance(results[2], Exception) else {}
            best_game_state_data = (
                results[3] if not isinstance(results[3], Exception) else None
            )

            # Log errors if any occurred during gather
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    task_name = ["worker_render", "plot_data", "summary", "best_game"][
                        i
                    ]
                    logger.error(f"Error fetching {task_name} from actor: {res}")

            data = {
                "app_state": app_obj.app_state.value,
                "status": app_obj.status,
                "cleanup_confirmation_active": app_obj.cleanup_confirmation_active,
                "cleanup_message": app_obj.cleanup_message,
                "last_cleanup_message_time": app_obj.last_cleanup_message_time,
                "update_progress_details": {},
                "demo_env_state": (
                    app_obj.demo_env.get_state() if app_obj.demo_env else None
                ),
                "demo_env_is_over": (
                    app_obj.demo_env.is_over() if app_obj.demo_env else False
                ),
                "demo_env_score": (
                    app_obj.demo_env.game_score if app_obj.demo_env else 0
                ),
                "demo_env_dragged_shape_idx": (
                    app_obj.demo_env.demo_dragged_shape_idx
                    if app_obj.demo_env
                    else None
                ),
                "demo_env_snapped_pos": (
                    app_obj.demo_env.demo_snapped_position if app_obj.demo_env else None
                ),
                "demo_env_selected_shape_idx": (
                    app_obj.demo_env.demo_selected_shape_idx if app_obj.demo_env else -1
                ),
                "env_config_rows": app_obj.env_config.ROWS,
                "env_config_cols": app_obj.env_config.COLS,
                "env_config_num_shape_slots": app_obj.env_config.NUM_SHAPE_SLOTS,
                "num_envs": app_obj.train_config_instance.NUM_SELF_PLAY_WORKERS,
                "plot_data": plot_data,
                "stats_summary": stats_summary,
                "best_game_state_data": best_game_state_data,
                "agent_param_count": app_obj.initializer.agent_param_count,
                "worker_counts": app_obj.worker_manager.get_active_worker_counts(),
                "is_process_running": app_obj.worker_manager.is_any_worker_running(),
                "worker_render_data": worker_render_data_result,
            }
            return data

        logic_app_state.get_render_data = _get_render_data_async.__get__(
            logic_app_state
        )

    except Exception as init_err:
        logger.critical(f"Logic Initialization failed: {init_err}", exc_info=True)
        stop_event.set()
        try:
            render_data_queue.put({ERROR_SENTINEL: f"Logic Init Failed: {init_err}"})
        except Exception:
            pass
        if ray_initialized:
            ray.shutdown()
        return

    # --- Main Logic Loop (Async) ---
    last_render_send_time = 0
    render_send_interval = 1.0 / 30.0

    async def main_loop():
        nonlocal last_render_send_time
        while not stop_event.is_set():
            loop_start = time.monotonic()

            # Process Commands (Synchronous)
            try:
                command_data = command_queue.get_nowait()
                if isinstance(command_data, dict) and COMMAND_SENTINEL in command_data:
                    command = command_data[COMMAND_SENTINEL]
                    logger.info(f"Received command from UI: {command}")
                    if command == STOP_SENTINEL:
                        stop_event.set()
                        break
                    logic_method_name = command_data.get(COMMAND_SENTINEL)
                    logic_method = getattr(
                        logic_app_state.logic, logic_method_name, None
                    )
                    if callable(logic_method):
                        payload = command_data.get(PAYLOAD_KEY)
                        if payload is not None:
                            logic_method(payload)
                        else:
                            logic_method()
                    else:
                        logger.warning(f"Unknown command: {logic_method_name}")
                elif command_data is not None:
                    logger.warning(f"Invalid data on command queue: {command_data}")
            except queue.Empty:
                pass
            except (EOFError, BrokenPipeError):
                logger.warning("Command queue connection lost.")
                stop_event.set()
                break
            except Exception as cmd_err:
                logger.error(f"Error processing command: {cmd_err}", exc_info=True)

            # Update Logic State (Synchronous)
            logic_app_state.logic.update_status_and_check_completion()

            # Send Render Data (Async)
            current_time = time.monotonic()
            if current_time - last_render_send_time > render_send_interval:
                try:
                    render_data = await logic_app_state.get_render_data()
                    render_data_queue.put(
                        {RENDER_DATA_SENTINEL: render_data}, block=False
                    )
                    last_render_send_time = current_time
                except queue.Full:
                    logger.debug("Render data queue full.")
                    last_render_send_time = current_time
                except (EOFError, BrokenPipeError):
                    logger.warning("Render data queue connection lost.")
                    stop_event.set()
                    break
                except Exception as send_err:
                    logger.error(
                        f"Error sending render data: {send_err}", exc_info=True
                    )

            # Loop Timing
            loop_duration = time.monotonic() - loop_start
            sleep_time = max(0, 0.005 - loop_duration)
            await asyncio.sleep(sleep_time)

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.warning("Logic process received KeyboardInterrupt.")
        stop_event.set()
    except Exception as loop_err:
        logger.critical(f"Critical error in logic main loop: {loop_err}", exc_info=True)
        stop_event.set()

    # --- Shutdown Logic ---
    logger.info("Logic Process shutting down...")
    try:
        if logic_app_state:
            if logic_app_state.worker_manager:
                logic_app_state.worker_manager.stop_all_workers()  # Stops Ray actors
            if logic_app_state.logic:
                logic_app_state.logic.save_final_checkpoint()  # CheckpointManager interacts with actors
            if logic_app_state.initializer:
                logic_app_state.initializer.close_stats_recorder()
            # Terminate other actors if needed (e.g., AgentPredictor, StatsAggregatorActor)
            if logic_app_state.agent_predictor:
                try:
                    ray.kill(logic_app_state.agent_predictor)
                except Exception as e:
                    logger.error(f"Error killing AgentPredictor: {e}")
            if logic_app_state.stats_aggregator and isinstance(
                logic_app_state.stats_aggregator, ray.actor.ActorHandle
            ):
                try:
                    ray.kill(logic_app_state.stats_aggregator)
                except Exception as e:
                    logger.error(f"Error killing StatsAggregatorActor: {e}")
        else:
            logger.warning("logic_app_state not initialized during shutdown sequence.")
    except Exception as shutdown_err:
        logger.error(
            f"Error during logic process shutdown: {shutdown_err}", exc_info=True
        )
    finally:
        try:
            render_data_queue.put(STOP_SENTINEL)
        except Exception as q_err_final:
            logger.warning(f"Could not send final STOP sentinel to UI: {q_err_final}")
        if ray_initialized:
            logger.info("Shutting down Ray...")
            try:
                ray.shutdown()
            except Exception as ray_down_err:
                logger.error(f"Error during Ray shutdown: {ray_down_err}")
        logger.info(
            f"Logic Process finished. Runtime: {time.time() - logic_start_time:.2f}s"
        )


File: main_pygame.py
# File: main_pygame.py
import sys
import time
import threading
import logging
import logging.handlers
import argparse
import os
import traceback
import multiprocessing as mp
from typing import Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from config import BASE_CHECKPOINT_DIR, set_run_id, get_run_id, get_run_log_dir
    from training.checkpoint_manager import find_latest_run_and_checkpoint
    from logger import TeeLogger
    from ui_process import run_ui_process
    from logic_process import run_logic_process
except ImportError as e:
    print(f"Error importing core modules/functions: {e}\n{traceback.format_exc()}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s|%(processName)s] %(message)s"
)
logger = logging.getLogger(__name__)

tee_logger_instance: Optional[TeeLogger] = None
log_listener_thread: Optional[threading.Thread] = None


# --- Logging Setup Functions (remain the same) ---
def setup_logging_queue_listener(log_queue: mp.Queue):
    global log_listener_thread

    def listener_process():
        listener_logger = logging.getLogger("LogListener")
        listener_logger.info("Log listener started.")
        while True:
            try:
                record = log_queue.get()
                if record is None:
                    break
                logger_handler = logging.getLogger(record.name)
                logger_handler.handle(record)
            except (EOFError, OSError):
                listener_logger.warning("Log queue closed or broken pipe.")
                break
            except Exception as e:
                print(f"Log listener error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        listener_logger.info("Log listener stopped.")

    log_listener_thread = threading.Thread(
        target=listener_process, daemon=True, name="LogListener"
    )
    log_listener_thread.start()
    return log_listener_thread


def setup_logging_and_run_id(args: argparse.Namespace):
    global tee_logger_instance
    run_id_source = "New"
    if args.load_checkpoint:
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                run_id_source = f"Explicit Checkpoint ({get_run_id()})"
            else:
                get_run_id()
                run_id_source = (
                    f"New (Explicit Ckpt Path Invalid: {args.load_checkpoint})"
                )
        except Exception as e:
            logger.warning(
                f"Could not determine run_id from checkpoint path '{args.load_checkpoint}': {e}. Generating new."
            )
            get_run_id()
            run_id_source = f"New (Error parsing ckpt path)"
    else:
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            run_id_source = f"Resumed Latest ({get_run_id()})"
        else:
            get_run_id()
            run_id_source = f"New (No previous runs found)"
    current_run_id = get_run_id()
    print(f"Run ID: {current_run_id} (Source: {run_id_source})")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        log_file_dir = get_run_log_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, "console_output.log")
        tee_logger_instance = TeeLogger(log_file_path, sys.stdout)
        sys.stdout = tee_logger_instance
        sys.stderr = tee_logger_instance
        print(f"Main process console output will be mirrored to: {log_file_path}")
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logger.info(f"Main process logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {current_run_id}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")
    return original_stdout, original_stderr


def cleanup_logging(
    original_stdout, original_stderr, log_queue: Optional[mp.Queue], exit_code
):
    print("[Main Finally] Restoring stdout/stderr and closing logger...")
    if log_queue:
        try:
            log_queue.put(None)
            log_queue.close()
            log_queue.join_thread()
        except Exception as qe:
            print(f"Error closing log queue: {qe}", file=original_stderr)
    if log_listener_thread:
        try:
            log_listener_thread.join(timeout=2.0)
            if log_listener_thread.is_alive():
                print(
                    "Warning: Log listener thread did not join cleanly.",
                    file=original_stderr,
                )
        except Exception as le:
            print(f"Error joining log listener thread: {le}", file=original_stderr)
    if tee_logger_instance:
        try:
            if isinstance(sys.stdout, TeeLogger):
                sys.stdout.flush()
            if isinstance(sys.stderr, TeeLogger):
                sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            tee_logger_instance.close()
            print("[Main Finally] TeeLogger closed and streams restored.")
        except Exception as log_close_err:
            original_stdout.write(f"ERROR closing TeeLogger: {log_close_err}\n")
            traceback.print_exc(file=original_stderr)
    print(f"[Main Finally] Exiting with code {exit_code}.")
    sys.exit(exit_code)


# =========================================================================
# Main Execution Block
# =========================================================================
if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Could not set start method to 'spawn', using default.")

    parser = argparse.ArgumentParser(description="AlphaZero Trainer - Multiprocess")
    parser.add_argument(
        "--load-checkpoint", type=str, default=None, help="Path to checkpoint."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    args = parser.parse_args()

    original_stdout, original_stderr = setup_logging_and_run_id(args)

    ui_to_logic_queue = mp.Queue()
    logic_to_ui_queue = mp.Queue(maxsize=10)
    stop_event = mp.Event()
    log_queue = mp.Queue()
    log_listener = setup_logging_queue_listener(log_queue)

    # Set daemon=True for simpler exit handling, rely on stop_event and timeouts
    ui_process = mp.Process(
        target=run_ui_process,
        args=(stop_event, ui_to_logic_queue, logic_to_ui_queue, log_queue),
        name="UIProcess",
        daemon=True,
    )
    logic_process = mp.Process(
        target=run_logic_process,
        args=(
            stop_event,
            ui_to_logic_queue,
            logic_to_ui_queue,
            args.load_checkpoint,
            log_queue,
        ),
        name="LogicProcess",
        daemon=True,
    )

    exit_code = 0
    try:
        logger.info("Starting UI process...")
        ui_process.start()
        logger.info("Starting Logic process...")
        logic_process.start()

        # --- Wait for Processes ---
        while True:  # Loop indefinitely until stop_event or error
            if stop_event.is_set():
                logger.info("Stop event detected by main process. Exiting wait loop.")
                break
            if not logic_process.is_alive():
                logger.warning("Logic process terminated unexpectedly. Signaling stop.")
                stop_event.set()
                exit_code = 1  # Indicate error
                break
            if not ui_process.is_alive():
                logger.warning("UI process terminated unexpectedly. Signaling stop.")
                stop_event.set()
                exit_code = 1  # Indicate error
                break
            try:
                # Sleep briefly to prevent busy-waiting
                time.sleep(0.2)
            except KeyboardInterrupt:
                logger.warning(
                    "Main process received KeyboardInterrupt. Signaling stop..."
                )
                stop_event.set()
                exit_code = 130
                break  # Exit the waiting loop

    except Exception as main_err:
        logger.critical(
            f"Error in main process coordination: {main_err}", exc_info=True
        )
        stop_event.set()
        exit_code = 1

    finally:
        logger.info("Main process initiating cleanup...")
        if not stop_event.is_set():
            stop_event.set()  # Ensure stop is signaled

        time.sleep(0.5)  # Allow processes to potentially react

        # --- Join Processes with Timeouts ---
        join_timeout_logic = 10.0  # More time for logic to save
        join_timeout_ui = 3.0

        logger.info(
            f"Waiting for Logic process to join (timeout: {join_timeout_logic}s)..."
        )
        if logic_process.is_alive():
            logic_process.join(timeout=join_timeout_logic)
        if logic_process.is_alive():
            logger.warning("Logic process did not join cleanly. Terminating.")
            try:
                logic_process.terminate()
                logic_process.join(1.0)
            except Exception as term_err:
                logger.error(f"Error terminating Logic process: {term_err}")

        logger.info(f"Waiting for UI process to join (timeout: {join_timeout_ui}s)...")
        if ui_process.is_alive():
            ui_process.join(timeout=join_timeout_ui)
        if ui_process.is_alive():
            logger.warning("UI process did not join cleanly. Terminating.")
            try:
                ui_process.terminate()
                ui_process.join(1.0)
            except Exception as term_err:
                logger.error(f"Error terminating UI process: {term_err}")

        logger.info("Processes joined or terminated.")
        cleanup_logging(original_stdout, original_stderr, log_queue, exit_code)


File: out.md
File: analyze_profile.py
# File: analyze_profile_v2.py
import pstats
from pstats import SortKey

profile_file = "profile_output.prof"
output_file_cumulative = "profile_summary_cumulative.txt"
output_file_tottime = "profile_summary_tottime.txt"
num_lines_to_print = 50  # You can adjust how many lines to show

try:
    # --- Sort by Cumulative Time ---
    print(
        f"Saving top {num_lines_to_print} cumulative time stats to {output_file_cumulative}..."
    )
    with open(output_file_cumulative, "w") as f_cum:
        # Pass the file handle directly as the stream
        stats_cum = pstats.Stats(profile_file, stream=f_cum)
        stats_cum.sort_stats(SortKey.CUMULATIVE).print_stats(num_lines_to_print)
        # 'with open' handles closing/flushing
    print("Done.")

    # --- Sort by Total Time (Internal) ---
    print(
        f"Saving top {num_lines_to_print} total time (tottime) stats to {output_file_tottime}..."
    )
    with open(output_file_tottime, "w") as f_tot:
        # Pass the file handle directly as the stream
        stats_tot = pstats.Stats(profile_file, stream=f_tot)
        stats_tot.sort_stats(SortKey.TIME).print_stats(
            num_lines_to_print
        )  # SortKey.TIME is 'tottime'
        # 'with open' handles closing/flushing
    print("Done.")

    print(
        f"\nAnalysis complete. Check '{output_file_cumulative}' and '{output_file_tottime}'."
    )

except FileNotFoundError:
    print(f"ERROR: Profile file '{profile_file}' not found.")
except Exception as e:
    print(f"An error occurred during profile analysis: {e}")


File: app_init.py
import pygame
import time
import traceback
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import TYPE_CHECKING, List, Optional, Any
import multiprocessing as mp
import ray
from ray.util.queue import Queue as RayQueue
import logging  # Added logging

from config import (
    ModelConfig,
    StatsConfig,
    DemoConfig,
    MCTSConfig,
    EnvConfig,
    TrainConfig,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
)
from environment.game_state import GameState
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregatorActor  # Import Actor
from stats.simple_stats_recorder import SimpleStatsRecorder

from training.checkpoint_manager import CheckpointManager
from app_state import AppState
from mcts import MCTS
from agent.alphazero_net import AlphaZeroNet, AgentPredictor

# Workers managed by AppWorkerManager

if TYPE_CHECKING:
    LogicAppState = Any
    from torch.optim.lr_scheduler import _LRScheduler

    AgentPredictorHandle = ray.actor.ActorHandle
    SelfPlayWorkerHandle = ray.actor.ActorHandle
    TrainingWorkerHandle = ray.actor.ActorHandle
    StatsAggregatorHandle = ray.actor.ActorHandle  # Use Actor Handle type

logger = logging.getLogger(__name__)  # Added logger


class AppInitializer:
    """Handles the initialization of core RL application components in the Logic Process."""

    def __init__(self, app: "LogicAppState"):
        self.app = app
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        self.train_config = app.train_config_instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.demo_config = DemoConfig()
        self.mcts_config = MCTSConfig()
        self.worker_stop_event: mp.Event = app.worker_stop_event

        # Components to be initialized
        self.agent: Optional[AlphaZeroNet] = None
        self.agent_predictor: Optional["AgentPredictorHandle"] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional["_LRScheduler"] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.stats_aggregator: Optional["StatsAggregatorHandle"] = (
            None  # Now an Actor Handle
        )
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        # MCTS instance not central anymore

    def initialize_logic_components(self):
        """Initializes only the RL and logic-related components, including Ray actors."""
        try:
            self._check_gpu_memory()
            self._init_ray_actors()  # Initialize actors first
            self.initialize_rl_components(
                is_reinit=False,
                checkpoint_to_load=self.app.checkpoint_to_load,
            )
            self.initialize_demo_env()
            self._calculate_agent_params()
            self.app.worker_manager.initialize_actors()  # Initialize worker actors

        except Exception as init_err:
            self._handle_init_error(init_err)

    def _init_ray_actors(self):
        """Initializes core Ray actors like AgentPredictor and StatsAggregatorActor."""
        logger.info("[AppInitializer] Initializing Ray Actors...")
        # --- Agent Predictor Actor ---
        try:
            self.agent_predictor = AgentPredictor.options(
                name="AgentPredictorActor",  # Optional: give it a name
                lifetime="detached",  # Optional: keep actor alive if main script exits abnormally
            ).remote(
                env_config=self.env_config, model_config=self.model_config.Network()
            )
            ray.get(self.agent_predictor.health_check.remote())  # Wait for actor
            logger.info("[AppInitializer] AgentPredictor actor created.")
            self.app.agent_predictor = self.agent_predictor
        except Exception as e:
            logger.error(
                f"[AppInitializer] Failed to create AgentPredictor actor: {e}",
                exc_info=True,
            )
            raise RuntimeError("AgentPredictor actor initialization failed") from e

        # --- Stats Aggregator Actor ---
        try:
            self.stats_aggregator = StatsAggregatorActor.options(
                name="StatsAggregatorActor", lifetime="detached"
            ).remote(
                avg_windows=self.stats_config.STATS_AVG_WINDOW,
                plot_window=self.stats_config.PLOT_DATA_WINDOW,
            )
            ray.get(self.stats_aggregator.health_check.remote())  # Wait for actor
            logger.info("[AppInitializer] StatsAggregatorActor created.")
            self.app.stats_aggregator = self.stats_aggregator  # Store actor handle
        except Exception as e:
            logger.error(
                f"[AppInitializer] Failed to create StatsAggregator actor: {e}",
                exc_info=True,
            )
            raise RuntimeError("StatsAggregator actor initialization failed") from e

        logger.info("[AppInitializer] Ray Actors initialized.")

    def _check_gpu_memory(self):
        """Checks and prints total GPU memory if available."""
        if self.app.device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(self.app.device)
                self.app.total_gpu_memory_bytes = props.total_memory
                print(f"Total GPU Memory: {props.total_memory / (1024**3):.2f} GB")
            except Exception as e:
                print(f"Warning: Could not get total GPU memory: {e}")

    def _calculate_agent_params(self):
        """Calculates agent parameters by calling the AgentPredictor actor."""
        if self.agent_predictor:
            try:
                param_count_ref = self.agent_predictor.get_param_count.remote()
                self.agent_param_count = ray.get(param_count_ref)
                logger.info(
                    f"[AppInitializer] Agent Parameters: {self.agent_param_count:,}"
                )
            except Exception as e:
                logger.error(f"Warning: Could not get agent parameters from actor: {e}")
                self.agent_param_count = 0
        else:
            logger.warning("AgentPredictor actor not available for param count.")
            self.agent_param_count = 0

    def _handle_init_error(self, error: Exception):
        """Handles fatal errors during component initialization."""
        print(f"FATAL ERROR during component initialization: {error}")
        traceback.print_exc()
        self.app.set_state(AppState.ERROR)
        self.app.set_status(f"Logic Init Failed: {error}")
        self.app.stop_event.set()

    def initialize_rl_components(
        self, is_reinit: bool = False, checkpoint_to_load: Optional[str] = None
    ):
        """Initializes local NN Agent (for checkpointing), Optimizer, Scheduler, StatsRecorder, Checkpoint Manager."""
        print(f"Initializing AlphaZero components... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            self._init_local_agent_for_checkpointing()
            self._init_optimizer_and_scheduler()
            self._init_stats_recorder()  # Uses stats_aggregator handle now
            self._init_checkpoint_manager(
                checkpoint_to_load
            )  # Interacts with stats_aggregator handle

            print(
                f"AlphaZero components initialized in {time.time() - start_time:.2f}s"
            )
            self.app.optimizer = self.optimizer
            self.app.scheduler = self.scheduler
            self.app.stats_recorder = self.stats_recorder
            self.app.checkpoint_manager = self.checkpoint_manager

        except Exception as e:
            print(f"Error during AlphaZero component initialization: {e}")
            traceback.print_exc()
            raise e

    def _init_local_agent_for_checkpointing(self):
        """Initializes a local copy of the agent for saving/loading checkpoints."""
        self.agent = AlphaZeroNet(
            env_config=self.env_config, model_config=self.model_config.Network()
        ).to(self.app.device)
        print(
            f"Local AlphaZeroNet (for checkpointing) initialized on device: {self.app.device}."
        )

    def _init_optimizer_and_scheduler(self):
        """Initializes the optimizer and scheduler using the local agent copy."""
        if not self.agent:
            raise RuntimeError("Local Agent must be initialized before Optimizer.")
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.train_config.LEARNING_RATE,
            weight_decay=self.train_config.WEIGHT_DECAY,
        )
        print(
            f"Optimizer initialized (Adam, LR={self.train_config.LEARNING_RATE}) for local agent."
        )

        if self.train_config.USE_LR_SCHEDULER:
            if self.train_config.SCHEDULER_TYPE == "CosineAnnealingLR":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.train_config.SCHEDULER_T_MAX,
                    eta_min=self.train_config.SCHEDULER_ETA_MIN,
                )
                print(
                    f"LR Scheduler initialized (CosineAnnealingLR, T_max={self.train_config.SCHEDULER_T_MAX}, eta_min={self.train_config.SCHEDULER_ETA_MIN})."
                )
            else:
                print(
                    f"Warning: Unknown scheduler type '{self.train_config.SCHEDULER_TYPE}'. No scheduler initialized."
                )
                self.scheduler = None
        else:
            print("LR Scheduler is DISABLED.")
            self.scheduler = None

    def _init_stats_recorder(self):
        """Initializes the local StatsRecorder, passing the StatsAggregatorActor handle."""
        if not self.stats_aggregator:  # Check if handle exists
            raise RuntimeError(
                "StatsAggregatorActor handle must be initialized before StatsRecorder."
            )
        print("Initializing SimpleStatsRecorder...")
        self.stats_recorder = SimpleStatsRecorder(
            aggregator=self.stats_aggregator,  # Pass actor handle
            console_log_interval=self.stats_config.CONSOLE_LOG_FREQ,
            train_config=self.train_config,
        )
        print("SimpleStatsRecorder initialized.")

    def _init_checkpoint_manager(self, checkpoint_to_load: Optional[str]):
        """Initializes the CheckpointManager using local agent/optimizer and StatsAggregatorActor handle."""
        if not self.agent or not self.optimizer or not self.stats_aggregator:
            raise RuntimeError(
                "Local Agent, Optimizer, and StatsAggregatorActor handle needed for CheckpointManager."
            )
        self.checkpoint_manager = CheckpointManager(
            agent=self.agent,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            stats_aggregator=self.stats_aggregator,  # Pass actor handle
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            run_checkpoint_dir=get_run_checkpoint_dir(),
            load_checkpoint_path_config=checkpoint_to_load,
            device=self.app.device,
        )
        if self.checkpoint_manager.get_checkpoint_path_to_load():
            # Load checkpoint into local agent/optimizer/scheduler
            # CheckpointManager's load_checkpoint method now handles loading stats into the actor
            self.checkpoint_manager.load_checkpoint()
            # Push loaded weights to the AgentPredictor actor
            if self.agent_predictor:
                try:
                    loaded_weights = self.agent.state_dict()
                    set_ref = self.agent_predictor.set_weights.remote(loaded_weights)
                    ray.get(set_ref)
                    logger.info(
                        "[AppInitializer] Pushed loaded checkpoint weights to AgentPredictor actor."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to push loaded weights to AgentPredictor: {e}"
                    )
            # Update App state based on loaded checkpoint AFTER loading
            # Get step count from the aggregator actor
            if self.stats_aggregator:
                try:
                    step_ref = self.stats_aggregator.get_current_global_step.remote()
                    self.app.current_global_step = ray.get(step_ref)
                except Exception as e:
                    logger.error(
                        f"Failed to get global step from StatsAggregator actor: {e}"
                    )
                    self.app.current_global_step = 0  # Fallback
            else:
                self.app.current_global_step = 0

    def initialize_demo_env(self):
        """Initializes the separate environment for demo/debug if needed by logic."""
        print("Initializing Demo/Debug Environment (in Logic Process)...")
        try:
            self.demo_env = GameState()
            self.demo_env.reset()
            print("Demo/Debug environment initialized.")
            self.app.demo_env = self.demo_env
        except Exception as e:
            print(f"ERROR initializing demo/debug environment: {e}")
            traceback.print_exc()
            self.demo_env = None
            self.app.demo_env = None

    def close_stats_recorder(self, is_cleanup: bool = False):
        """Safely closes the stats recorder (if it exists)."""
        print(
            f"[AppInitializer] close_stats_recorder called (is_cleanup={is_cleanup})..."
        )
        # StatsAggregator is now an actor, termination handled elsewhere (e.g., AppWorkerManager or main shutdown)
        if self.stats_recorder and hasattr(self.stats_recorder, "close"):
            print("[AppInitializer] Stats recorder exists, attempting close...")
            try:
                # Recorder might need to make final calls to the aggregator actor before closing
                self.stats_recorder.close(is_cleanup=is_cleanup)
                print("[AppInitializer] stats_recorder.close() executed.")
            except Exception as log_e:
                print(f"[AppInitializer] Error closing stats recorder on exit: {log_e}")
                traceback.print_exc()
        else:
            print("[AppInitializer] No stats recorder instance or close method.")
        print("[AppInitializer] close_stats_recorder finished.")


File: app_logic.py
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


File: app_setup.py
import os
import pygame
from typing import Tuple, Dict, Any

from config import (
    VisConfig,
    get_run_checkpoint_dir,
    get_run_log_dir,
    get_config_dict,
    print_config_info_and_validate,
)


def initialize_pygame(
    vis_config: VisConfig,
) -> Tuple[pygame.Surface, pygame.time.Clock]:
    """Initializes Pygame, sets up the screen and clock."""
    print("Initializing Pygame...")
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (vis_config.SCREEN_WIDTH, vis_config.SCREEN_HEIGHT), pygame.RESIZABLE
    )
    pygame.display.set_caption("AlphaTri Trainer")  # Updated caption
    clock = pygame.time.Clock()
    print("Pygame initialized.")
    return screen, clock


def initialize_directories():
    """Creates necessary runtime directories using dynamic paths."""
    run_checkpoint_dir = get_run_checkpoint_dir()
    run_log_dir = get_run_log_dir()
    # Console log dir is created within main_pygame before logger init

    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"Ensured directories exist: {run_checkpoint_dir}, {run_log_dir}")


def load_and_validate_configs() -> Dict[str, Any]:
    """Loads all config classes and returns the combined config dictionary."""
    config_dict = get_config_dict()
    print_config_info_and_validate()
    return config_dict


File: app_state.py
from enum import Enum


class AppState(Enum):
    INITIALIZING = "Initializing"
    MAIN_MENU = "MainMenu"
    PLAYING = "Playing" 
    DEBUG = "Debug"
    CLEANUP_CONFIRM = "Confirm Cleanup"
    CLEANING = "Cleaning"
    ERROR = "Error"
    UNKNOWN = "Unknown"


File: app_ui_utils.py
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from main_pygame import MainApp
    from environment.game_state import GameState
    from ui.renderer import UIRenderer


class AppUIUtils:
    """Utility functions related to mapping screen coordinates to game elements."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def map_screen_to_grid(
        self, screen_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Maps screen coordinates to grid row/column for demo/debug."""
        if (
            self.app.renderer is None
            or self.app.initializer.demo_env is None
            or self.app.renderer.demo_renderer is None
        ):
            return None
        if self.app.app_state not in [
            self.app.app_state.PLAYING,
            self.app.app_state.DEBUG,
        ]:
            return None

        demo_env: "GameState" = self.app.initializer.demo_env
        renderer: "UIRenderer" = self.app.renderer

        screen_width, screen_height = self.app.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        _, clipped_game_rect = renderer.demo_renderer._calculate_game_area_rect(
            screen_width,
            screen_height,
            padding,
            hud_height,
            help_height,
            self.app.env_config,
        )

        if not clipped_game_rect.collidepoint(screen_pos):
            return None

        relative_x = screen_pos[0] - clipped_game_rect.left
        relative_y = screen_pos[1] - clipped_game_rect.top

        tri_cell_w, tri_cell_h = renderer.demo_renderer._calculate_demo_triangle_size(
            clipped_game_rect.width, clipped_game_rect.height, self.app.env_config
        )
        grid_ox, grid_oy = renderer.demo_renderer._calculate_grid_offset(
            clipped_game_rect.width, clipped_game_rect.height, self.app.env_config
        )

        if tri_cell_w <= 0 or tri_cell_h <= 0:
            return None

        grid_relative_x = relative_x - grid_ox
        grid_relative_y = relative_y - grid_oy

        # Approximate calculation (might need refinement based on triangle geometry)
        approx_row = int(grid_relative_y / tri_cell_h)
        approx_col = int(grid_relative_x / (tri_cell_w * 0.75))

        if (
            0 <= approx_row < self.app.env_config.ROWS
            and 0 <= approx_col < self.app.env_config.COLS
        ):
            if (
                demo_env.grid.valid(approx_row, approx_col)
                and not demo_env.grid.triangles[approx_row][approx_col].is_death
            ):
                return approx_row, approx_col
        return None

    def map_screen_to_preview(self, screen_pos: Tuple[int, int]) -> Optional[int]:
        """Maps screen coordinates to a shape preview index."""
        if (
            self.app.renderer is None
            or self.app.initializer.demo_env is None
            or self.app.input_handler is None
        ):
            return None
        if self.app.app_state != self.app.app_state.PLAYING:
            return None

        # Access preview rects directly from the input handler
        if hasattr(self.app.input_handler, "shape_preview_rects"):
            for idx, rect in self.app.input_handler.shape_preview_rects.items():
                if rect.collidepoint(screen_pos):
                    return idx
        return None


File: app_workers.py
# File: app_workers.py
import threading
import queue
import time
from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Any
import logging
import multiprocessing as mp
import ray
import asyncio
import torch

# Import Ray actor classes
from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker

# Import Actor Handles for type hinting
if TYPE_CHECKING:
    LogicAppState = Any
    SelfPlayWorkerHandle = ray.actor.ActorHandle
    TrainingWorkerHandle = ray.actor.ActorHandle
    AgentPredictorHandle = ray.actor.ActorHandle
    StatsAggregatorHandle = ray.actor.ActorHandle
    from ray.util.queue import Queue as RayQueue

logger = logging.getLogger(__name__)


class AppWorkerManager:
    """Manages the creation, starting, and stopping of Ray worker actors."""

    DEFAULT_KILL_TIMEOUT = 5.0

    def __init__(self, app: "LogicAppState"):
        self.app = app
        self.self_play_worker_actors: List["SelfPlayWorkerHandle"] = []
        self.training_worker_actor: Optional["TrainingWorkerHandle"] = None
        self.agent_predictor_actor: Optional["AgentPredictorHandle"] = None
        self._workers_running = False
        logger.info("[AppWorkerManager] Initialized for Ray Actors.")

    def initialize_actors(self):
        """Initializes Ray worker actors (SelfPlay, Training). Does NOT start their loops."""
        logger.info("[AppWorkerManager] Initializing worker actors...")
        if not self.app.agent_predictor:
            logger.error(
                "[AppWorkerManager] ERROR: AgentPredictor actor not initialized in AppInitializer."
            )
            self.app.set_state(self.app.app_state.ERROR)
            self.app.set_status("Worker Init Failed: Missing AgentPredictor")
            return
        if not self.app.stats_aggregator:
            logger.error(
                "[AppWorkerManager] ERROR: StatsAggregator actor handle not initialized in AppInitializer."
            )
            self.app.set_state(self.app.app_state.ERROR)
            self.app.set_status("Worker Init Failed: Missing StatsAggregator")
            return

        self.agent_predictor_actor = self.app.agent_predictor

        self._init_self_play_actors()
        self._init_training_actor()

        num_sp = len(self.self_play_worker_actors)
        num_tr = 1 if self.training_worker_actor else 0
        logger.info(
            f"Worker actors initialized ({num_sp} Self-Play, {num_tr} Training)."
        )

    def _init_self_play_actors(self):
        """Creates SelfPlayWorker Ray actors."""
        self.self_play_worker_actors = []
        num_sp_workers = self.app.train_config_instance.NUM_SELF_PLAY_WORKERS
        logger.info(f"Initializing {num_sp_workers} SelfPlayWorker actor(s)...")
        for i in range(num_sp_workers):
            try:
                actor = SelfPlayWorker.remote(
                    worker_id=i,
                    agent_predictor=self.agent_predictor_actor,
                    mcts_config=self.app.mcts_config,
                    env_config=self.app.env_config,
                    experience_queue=self.app.experience_queue,
                    stats_aggregator=self.app.stats_aggregator,
                    max_game_steps=None,
                )
                self.self_play_worker_actors.append(actor)
                logger.info(f"  SelfPlayWorker-{i} actor created.")
            except Exception as e:
                logger.error(
                    f"  ERROR creating SelfPlayWorker-{i} actor: {e}", exc_info=True
                )

    def _init_training_actor(self):
        """Creates the TrainingWorker Ray actor."""
        logger.info("Initializing TrainingWorker actor...")
        if not self.app.optimizer or not self.app.train_config_instance:
            logger.error(
                "[AppWorkerManager] ERROR: Optimizer or TrainConfig missing for TrainingWorker init."
            )
            return

        optimizer_cls = type(self.app.optimizer)
        optimizer_kwargs = self.app.optimizer.defaults

        scheduler_cls = type(self.app.scheduler) if self.app.scheduler else None
        scheduler_kwargs = {}
        if self.app.scheduler and hasattr(self.app.scheduler, "state_dict"):
            sd = self.app.scheduler.state_dict()
            if isinstance(
                self.app.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
            ):
                scheduler_kwargs = {
                    "T_max": sd.get("T_max", 1000),
                    "eta_min": sd.get("eta_min", 0),
                }
            else:
                logger.warning(
                    f"Cannot automatically determine kwargs for scheduler type {scheduler_cls}. Scheduler might not be correctly re-initialized in actor."
                )
                scheduler_cls = None

        try:
            actor = TrainingWorker.remote(
                agent_predictor=self.agent_predictor_actor,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                scheduler_cls=scheduler_cls,
                scheduler_kwargs=scheduler_kwargs,
                experience_queue=self.app.experience_queue,
                stats_aggregator=self.app.stats_aggregator,
                train_config=self.app.train_config_instance,
            )
            self.training_worker_actor = actor
            logger.info("  TrainingWorker actor created.")
        except Exception as e:
            logger.error(f"  ERROR creating TrainingWorker actor: {e}", exc_info=True)

    def get_active_worker_counts(self) -> Dict[str, int]:
        """Returns the count of initialized worker actors."""
        sp_count = len(self.self_play_worker_actors)
        tr_count = 1 if self.training_worker_actor else 0
        return {"SelfPlay": sp_count, "Training": tr_count}

    def is_any_worker_running(self) -> bool:
        """Checks the internal flag indicating if workers have been started."""
        return self._workers_running

    async def get_worker_render_data_async(
        self, max_envs: int
    ) -> List[Optional[Dict[str, Any]]]:
        """Retrieves render data from active self-play actors asynchronously."""
        if not self.self_play_worker_actors:
            return [None] * max_envs

        tasks = []
        num_to_fetch = min(len(self.self_play_worker_actors), max_envs)
        for i in range(num_to_fetch):
            actor = self.self_play_worker_actors[i]
            tasks.append(actor.get_current_render_data.remote())

        render_data_list: List[Optional[Dict[str, Any]]] = []
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting render data from worker {i}: {result}")
                    render_data_list.append(None)
                else:
                    render_data_list.append(result)
        except Exception as e:
            logger.error(f"Error gathering render data: {e}")
            render_data_list = [None] * num_to_fetch

        while len(render_data_list) < max_envs:
            render_data_list.append(None)
        return render_data_list

    def get_worker_render_data(self, max_envs: int) -> List[Optional[Dict[str, Any]]]:
        """Synchronous wrapper for get_worker_render_data_async."""
        if not self.self_play_worker_actors:
            return [None] * max_envs

        refs = []
        num_to_fetch = min(len(self.self_play_worker_actors), max_envs)
        for i in range(num_to_fetch):
            actor = self.self_play_worker_actors[i]
            refs.append(actor.get_current_render_data.remote())

        render_data_list: List[Optional[Dict[str, Any]]] = []
        try:
            results = ray.get(refs)
            render_data_list.extend(results)
        except Exception as e:
            logger.error(f"Error getting render data via ray.get: {e}")
            render_data_list = [None] * num_to_fetch

        while len(render_data_list) < max_envs:
            render_data_list.append(None)
        return render_data_list

    def start_all_workers(self):
        """Starts the main loops of all initialized worker actors."""
        if self._workers_running:
            logger.warning("[AppWorkerManager] Workers already started.")
            return
        if not self.self_play_worker_actors and not self.training_worker_actor:
            logger.error("[AppWorkerManager] No worker actors initialized to start.")
            return

        logger.info("[AppWorkerManager] Starting all worker actor loops...")
        self._workers_running = True

        for i, actor in enumerate(self.self_play_worker_actors):
            try:
                actor.run_loop.remote()
                logger.info(f"  SelfPlayWorker-{i} actor loop started.")
            except Exception as e:
                logger.error(f"  ERROR starting SelfPlayWorker-{i} actor loop: {e}")

        if self.training_worker_actor:
            try:
                self.training_worker_actor.run_loop.remote()
                logger.info("  TrainingWorker actor loop started.")
            except Exception as e:
                logger.error(f"  ERROR starting TrainingWorker actor loop: {e}")

        if self.is_any_worker_running():
            self.app.set_status("Running AlphaZero")
            num_sp = len(self.self_play_worker_actors)
            num_tr = 1 if self.training_worker_actor else 0
            logger.info(
                f"[AppWorkerManager] Worker loops started ({num_sp} SP, {num_tr} TR)."
            )

    def stop_all_workers(self, timeout: float = DEFAULT_KILL_TIMEOUT):
        """Signals all worker actors to stop and attempts to terminate them."""
        if (
            not self._workers_running
            and not self.self_play_worker_actors
            and not self.training_worker_actor
        ):
            logger.info("[AppWorkerManager] No workers running or initialized to stop.")
            return

        logger.info("[AppWorkerManager] Stopping ALL worker actors...")
        self._workers_running = False

        actors_to_stop: List[ray.actor.ActorHandle] = []
        actors_to_stop.extend(self.self_play_worker_actors)
        if self.training_worker_actor:
            actors_to_stop.append(self.training_worker_actor)

        if not actors_to_stop:
            logger.info("[AppWorkerManager] No active actor handles found to stop.")
            return

        logger.info(
            f"[AppWorkerManager] Sending stop signal to {len(actors_to_stop)} actors..."
        )
        for actor in actors_to_stop:
            try:
                actor.stop.remote()
            except Exception as e:
                logger.warning(f"Error sending stop signal to actor {actor}: {e}")

        time.sleep(0.5)

        logger.info(f"[AppWorkerManager] Killing actors...")
        for actor in actors_to_stop:
            try:
                ray.kill(actor, no_restart=True)
                logger.info(f"  Killed actor {actor}.")
            except Exception as e:
                logger.error(f"  Error killing actor {actor}: {e}")

        self.self_play_worker_actors = []
        self.training_worker_actor = None

        self._clear_experience_queue()

        logger.info("[AppWorkerManager] All worker actors stopped/killed.")
        self.app.set_status("Ready")

    def _clear_experience_queue(self):
        """Safely clears the experience queue (assuming Ray Queue)."""
        logger.info("[AppWorkerManager] Clearing experience queue...")
        # Check if it's a RayQueue instance (which acts as a handle)
        if hasattr(self.app, "experience_queue") and isinstance(
            self.app.experience_queue, ray.util.queue.Queue
        ):
            try:
                # Call qsize() directly, it returns an ObjectRef
                qsize_ref = self.app.experience_queue.qsize()
                qsize = ray.get(qsize_ref)  # Use ray.get() to resolve the ObjectRef
                logger.info(
                    f"[AppWorkerManager] Experience queue size before potential drain: {qsize}"
                )
                # Optional drain logic can be added here if needed
                # Example: Drain items if size is large
                # if qsize > 100:
                #     logger.info("[AppWorkerManager] Draining experience queue...")
                #     while qsize > 0:
                #         try:
                #             # Use get_nowait_batch to drain efficiently
                #             items_ref = self.app.experience_queue.get_nowait_batch(100)
                #             items = ray.get(items_ref)
                #             if not items: break
                #             qsize_ref = self.app.experience_queue.qsize()
                #             qsize = ray.get(qsize_ref)
                #         except ray.exceptions.RayActorError: # Handle queue actor potentially gone
                #             logger.warning("[AppWorkerManager] Queue actor error during drain.")
                #             break
                #         except Exception as drain_e:
                #             logger.error(f"Error draining queue: {drain_e}")
                #             break
                #     logger.info("[AppWorkerManager] Experience queue drained.")

            except Exception as e:
                logger.error(f"Error accessing Ray queue size: {e}")
        else:
            logger.warning(
                "[AppWorkerManager] Experience queue not found or not a Ray Queue during clearing."
            )


File: check_gpu.py
import torch

print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is NOT available to PyTorch.")
    # You can add checks for drivers here if needed, but PyTorch check is primary
    try:
        import subprocess
        print("\nAttempting to run nvidia-smi...")
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout)
        else:
            print(f"nvidia-smi command failed or not found (return code {result.returncode}). Ensure NVIDIA drivers are installed.")
            print(f"stderr: {result.stderr}")
    except FileNotFoundError:
         print("nvidia-smi command not found. Ensure NVIDIA drivers are installed and in PATH.")
    except Exception as e:
         print(f"Error running nvidia-smi: {e}")

File: logger.py
import os
from typing import TextIO, Optional


class TeeLogger:
    """Redirects stdout/stderr to both the console and a log file."""

    def __init__(self, filepath: str, original_stream: TextIO):
        self.terminal = original_stream
        self.log_file: Optional[TextIO] = None
        try:
            log_dir = os.path.dirname(filepath)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            # Use buffering=1 for line buffering
            self.log_file = open(filepath, "w", encoding="utf-8", buffering=1)
            print(f"[TeeLogger] Logging console output to: {filepath}")
        except Exception as e:
            self.terminal.write(
                f"FATAL ERROR: Could not open log file {filepath}: {e}\n"
            )
            # Continue without file logging if opening fails

    def write(self, message: str):
        self.terminal.write(message)
        if self.log_file:
            try:
                self.log_file.write(message)
            except Exception:
                # Silently ignore errors writing to log file to avoid loops
                pass

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass  # Silently ignore errors flushing log file

    def close(self):
        if self.log_file:
            try:
                self.log_file.close()
                self.log_file = None
            except Exception as e:
                self.terminal.write(f"Warning: Error closing log file: {e}\n")

    def __del__(self):
        # Ensure file is closed if logger object is garbage collected
        self.close()


File: logic_process.py
import time
import queue
import logging
import logging.handlers
import multiprocessing as mp
import traceback
import sys
from typing import Optional, Dict, Any
import ray
from ray.util.queue import Queue as RayQueue
import asyncio  # Added asyncio

try:
    from config import VisConfig, EnvConfig, TrainConfig, MCTSConfig, set_device
    from utils.helpers import get_device as get_torch_device, set_random_seeds
    from utils.init_checks import run_pre_checks
    from app_state import AppState
    from app_init import AppInitializer
    from app_logic import AppLogic
    from app_workers import AppWorkerManager
    from app_setup import initialize_directories
    from environment.game_state import GameState
except ImportError as e:
    print(f"[Logic Process Import Error] {e}", file=sys.stderr)
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    sys.exit(1)

RENDER_DATA_SENTINEL = "RENDER_DATA"
COMMAND_SENTINEL = "COMMAND"
STOP_SENTINEL = "STOP"
ERROR_SENTINEL = "ERROR"
PAYLOAD_KEY = "payload"


def run_logic_process(
    stop_event: mp.Event,
    command_queue: mp.Queue,
    render_data_queue: mp.Queue,
    checkpoint_to_load: Optional[str],
    log_queue: Optional[mp.Queue] = None,
):
    ray_initialized = False
    try:
        if not ray.is_initialized():
            ray.init(logging_level=logging.WARNING, ignore_reinit_error=True)
            ray_initialized = True
            print("[Logic Process] Ray initialized.")
        else:
            print("[Logic Process] Ray already initialized.")
            ray_initialized = True
    except Exception as ray_init_err:
        print(
            f"[Logic Process] FATAL: Ray initialization failed: {ray_init_err}",
            file=sys.stderr,
        )
        stop_event.set()
        try:
            render_data_queue.put({ERROR_SENTINEL: f"Ray Init Failed: {ray_init_err}"})
        except Exception:
            pass
        return

    if log_queue:
        qh = logging.handlers.QueueHandler(log_queue)
        root = logging.getLogger()
        if not any(isinstance(h, logging.handlers.QueueHandler) for h in root.handlers):
            root.addHandler(qh)
            root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Logic Process starting...")
    logic_start_time = time.time()

    logic_app_state = None
    try:
        vis_config = VisConfig()
        env_config = EnvConfig()
        train_config_instance = TrainConfig()
        mcts_config = MCTSConfig()
        worker_stop_event = mp.Event()

        # Use Ray Queue Actor
        experience_ray_queue = RayQueue(
            maxsize=train_config_instance.BUFFER_CAPACITY * 2
        )
        logger.info(
            f"[Logic Process] Ray Experience Queue created (maxsize={train_config_instance.BUFFER_CAPACITY * 2})."
        )

        logic_app_state = type(
            "LogicAppState",
            (object,),
            {
                "vis_config": vis_config,
                "env_config": env_config,
                "train_config_instance": train_config_instance,
                "mcts_config": mcts_config,
                "app_state": AppState.INITIALIZING,
                "status": "Initializing...",
                "stop_event": stop_event,
                "worker_stop_event": worker_stop_event,
                "experience_queue": experience_ray_queue,  # Use Ray Queue handle
                "device": get_torch_device(),
                "checkpoint_to_load": checkpoint_to_load,
                "initializer": None,
                "logic": None,
                "worker_manager": None,
                "agent_predictor": None,
                "stats_aggregator": None,  # Actor handles
                "ui_utils": None,
                "cleanup_confirmation_active": False,
                "cleanup_message": "",
                "last_cleanup_message_time": 0.0,
                "total_gpu_memory_bytes": None,
                "current_global_step": 0,
                "set_state": lambda self, new_state: setattr(
                    self, "app_state", new_state
                ),
                "set_status": lambda self, new_status: setattr(
                    self, "status", new_status
                ),
                "set_cleanup_confirmation": lambda self, active: setattr(
                    self, "cleanup_confirmation_active", active
                ),
                "set_cleanup_message": lambda self, msg, msg_time: (
                    setattr(self, "cleanup_message", msg),
                    setattr(self, "last_cleanup_message_time", msg_time),
                ),
                "get_render_data": None,
            },
        )()
        set_device(logic_app_state.device)

        initializer = AppInitializer(logic_app_state)
        logic = AppLogic(logic_app_state)
        worker_manager = AppWorkerManager(logic_app_state)
        logic_app_state.initializer = initializer
        logic_app_state.logic = logic
        logic_app_state.worker_manager = worker_manager

        logger.info("Initializing directories...")
        initialize_directories()
        set_random_seeds(
            mcts_config.RANDOM_SEED if hasattr(mcts_config, "RANDOM_SEED") else 42
        )
        logger.info("Running pre-checks...")
        run_pre_checks()
        logger.info("Initializing RL components and Ray actors...")
        initializer.initialize_logic_components()  # Initializes actors

        logic_app_state.set_state(AppState.MAIN_MENU)
        logic_app_state.set_status("Ready")
        logic.check_initial_completion_status()
        logger.info("--- Logic Initialization Complete ---")

        # Define async get_render_data
        async def _get_render_data_async(app_obj) -> Dict[str, Any]:
            worker_render_task = None
            if (
                app_obj.worker_manager.is_any_worker_running()
                and app_obj.app_state == AppState.MAIN_MENU
            ):
                num_to_render = app_obj.vis_config.NUM_ENVS_TO_RENDER
                if num_to_render > 0:
                    worker_render_task = (
                        app_obj.worker_manager.get_worker_render_data_async(
                            num_to_render
                        )
                    )

            # Fetch stats data from StatsAggregatorActor
            plot_data_ref, summary_ref, best_game_ref = None, None, None
            if app_obj.stats_aggregator:  # Check if handle exists
                plot_data_ref = app_obj.stats_aggregator.get_plot_data.remote()
                summary_ref = app_obj.stats_aggregator.get_summary.remote(
                    app_obj.current_global_step
                )
                best_game_ref = (
                    app_obj.stats_aggregator.get_best_game_state_data.remote()
                )

            # Gather results concurrently
            results = await asyncio.gather(
                (
                    worker_render_task
                    if worker_render_task
                    else asyncio.sleep(0, result=[])
                ),  # Handle no task case
                plot_data_ref if plot_data_ref else asyncio.sleep(0, result={}),
                summary_ref if summary_ref else asyncio.sleep(0, result={}),
                best_game_ref if best_game_ref else asyncio.sleep(0, result=None),
                return_exceptions=True,  # Handle potential errors from remote calls
            )

            # Process results, handling potential errors
            worker_render_data_result = (
                results[0]
                if not isinstance(results[0], Exception)
                else ([None] * app_obj.vis_config.NUM_ENVS_TO_RENDER)
            )
            plot_data = results[1] if not isinstance(results[1], Exception) else {}
            stats_summary = results[2] if not isinstance(results[2], Exception) else {}
            best_game_state_data = (
                results[3] if not isinstance(results[3], Exception) else None
            )

            # Log errors if any occurred during gather
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    task_name = ["worker_render", "plot_data", "summary", "best_game"][
                        i
                    ]
                    logger.error(f"Error fetching {task_name} from actor: {res}")

            data = {
                "app_state": app_obj.app_state.value,
                "status": app_obj.status,
                "cleanup_confirmation_active": app_obj.cleanup_confirmation_active,
                "cleanup_message": app_obj.cleanup_message,
                "last_cleanup_message_time": app_obj.last_cleanup_message_time,
                "update_progress_details": {},
                "demo_env_state": (
                    app_obj.demo_env.get_state() if app_obj.demo_env else None
                ),
                "demo_env_is_over": (
                    app_obj.demo_env.is_over() if app_obj.demo_env else False
                ),
                "demo_env_score": (
                    app_obj.demo_env.game_score if app_obj.demo_env else 0
                ),
                "demo_env_dragged_shape_idx": (
                    app_obj.demo_env.demo_dragged_shape_idx
                    if app_obj.demo_env
                    else None
                ),
                "demo_env_snapped_pos": (
                    app_obj.demo_env.demo_snapped_position if app_obj.demo_env else None
                ),
                "demo_env_selected_shape_idx": (
                    app_obj.demo_env.demo_selected_shape_idx if app_obj.demo_env else -1
                ),
                "env_config_rows": app_obj.env_config.ROWS,
                "env_config_cols": app_obj.env_config.COLS,
                "env_config_num_shape_slots": app_obj.env_config.NUM_SHAPE_SLOTS,
                "num_envs": app_obj.train_config_instance.NUM_SELF_PLAY_WORKERS,
                "plot_data": plot_data,
                "stats_summary": stats_summary,
                "best_game_state_data": best_game_state_data,
                "agent_param_count": app_obj.initializer.agent_param_count,
                "worker_counts": app_obj.worker_manager.get_active_worker_counts(),
                "is_process_running": app_obj.worker_manager.is_any_worker_running(),
                "worker_render_data": worker_render_data_result,
            }
            return data

        logic_app_state.get_render_data = _get_render_data_async.__get__(
            logic_app_state
        )

    except Exception as init_err:
        logger.critical(f"Logic Initialization failed: {init_err}", exc_info=True)
        stop_event.set()
        try:
            render_data_queue.put({ERROR_SENTINEL: f"Logic Init Failed: {init_err}"})
        except Exception:
            pass
        if ray_initialized:
            ray.shutdown()
        return

    # --- Main Logic Loop (Async) ---
    last_render_send_time = 0
    render_send_interval = 1.0 / 30.0

    async def main_loop():
        nonlocal last_render_send_time
        while not stop_event.is_set():
            loop_start = time.monotonic()

            # Process Commands (Synchronous)
            try:
                command_data = command_queue.get_nowait()
                if isinstance(command_data, dict) and COMMAND_SENTINEL in command_data:
                    command = command_data[COMMAND_SENTINEL]
                    logger.info(f"Received command from UI: {command}")
                    if command == STOP_SENTINEL:
                        stop_event.set()
                        break
                    logic_method_name = command_data.get(COMMAND_SENTINEL)
                    logic_method = getattr(
                        logic_app_state.logic, logic_method_name, None
                    )
                    if callable(logic_method):
                        payload = command_data.get(PAYLOAD_KEY)
                        if payload is not None:
                            logic_method(payload)
                        else:
                            logic_method()
                    else:
                        logger.warning(f"Unknown command: {logic_method_name}")
                elif command_data is not None:
                    logger.warning(f"Invalid data on command queue: {command_data}")
            except queue.Empty:
                pass
            except (EOFError, BrokenPipeError):
                logger.warning("Command queue connection lost.")
                stop_event.set()
                break
            except Exception as cmd_err:
                logger.error(f"Error processing command: {cmd_err}", exc_info=True)

            # Update Logic State (Synchronous)
            logic_app_state.logic.update_status_and_check_completion()

            # Send Render Data (Async)
            current_time = time.monotonic()
            if current_time - last_render_send_time > render_send_interval:
                try:
                    render_data = await logic_app_state.get_render_data()
                    render_data_queue.put(
                        {RENDER_DATA_SENTINEL: render_data}, block=False
                    )
                    last_render_send_time = current_time
                except queue.Full:
                    logger.debug("Render data queue full.")
                    last_render_send_time = current_time
                except (EOFError, BrokenPipeError):
                    logger.warning("Render data queue connection lost.")
                    stop_event.set()
                    break
                except Exception as send_err:
                    logger.error(
                        f"Error sending render data: {send_err}", exc_info=True
                    )

            # Loop Timing
            loop_duration = time.monotonic() - loop_start
            sleep_time = max(0, 0.005 - loop_duration)
            await asyncio.sleep(sleep_time)

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.warning("Logic process received KeyboardInterrupt.")
        stop_event.set()
    except Exception as loop_err:
        logger.critical(f"Critical error in logic main loop: {loop_err}", exc_info=True)
        stop_event.set()

    # --- Shutdown Logic ---
    logger.info("Logic Process shutting down...")
    try:
        if logic_app_state:
            if logic_app_state.worker_manager:
                logic_app_state.worker_manager.stop_all_workers()  # Stops Ray actors
            if logic_app_state.logic:
                logic_app_state.logic.save_final_checkpoint()  # CheckpointManager interacts with actors
            if logic_app_state.initializer:
                logic_app_state.initializer.close_stats_recorder()
            # Terminate other actors if needed (e.g., AgentPredictor, StatsAggregatorActor)
            if logic_app_state.agent_predictor:
                try:
                    ray.kill(logic_app_state.agent_predictor)
                except Exception as e:
                    logger.error(f"Error killing AgentPredictor: {e}")
            if logic_app_state.stats_aggregator and isinstance(
                logic_app_state.stats_aggregator, ray.actor.ActorHandle
            ):
                try:
                    ray.kill(logic_app_state.stats_aggregator)
                except Exception as e:
                    logger.error(f"Error killing StatsAggregatorActor: {e}")
        else:
            logger.warning("logic_app_state not initialized during shutdown sequence.")
    except Exception as shutdown_err:
        logger.error(
            f"Error during logic process shutdown: {shutdown_err}", exc_info=True
        )
    finally:
        try:
            render_data_queue.put(STOP_SENTINEL)
        except Exception as q_err_final:
            logger.warning(f"Could not send final STOP sentinel to UI: {q_err_final}")
        if ray_initialized:
            logger.info("Shutting down Ray...")
            try:
                ray.shutdown()
            except Exception as ray_down_err:
                logger.error(f"Error during Ray shutdown: {ray_down_err}")
        logger.info(
            f"Logic Process finished. Runtime: {time.time() - logic_start_time:.2f}s"
        )


File: main_pygame.py
# File: main_pygame.py
import sys
import time
import threading
import logging
import logging.handlers
import argparse
import os
import traceback
import multiprocessing as mp
from typing import Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from config import BASE_CHECKPOINT_DIR, set_run_id, get_run_id, get_run_log_dir
    from training.checkpoint_manager import find_latest_run_and_checkpoint
    from logger import TeeLogger
    from ui_process import run_ui_process
    from logic_process import run_logic_process
except ImportError as e:
    print(f"Error importing core modules/functions: {e}\n{traceback.format_exc()}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s|%(processName)s] %(message)s"
)
logger = logging.getLogger(__name__)

tee_logger_instance: Optional[TeeLogger] = None
log_listener_thread: Optional[threading.Thread] = None


# --- Logging Setup Functions (remain the same) ---
def setup_logging_queue_listener(log_queue: mp.Queue):
    global log_listener_thread

    def listener_process():
        listener_logger = logging.getLogger("LogListener")
        listener_logger.info("Log listener started.")
        while True:
            try:
                record = log_queue.get()
                if record is None:
                    break
                logger_handler = logging.getLogger(record.name)
                logger_handler.handle(record)
            except (EOFError, OSError):
                listener_logger.warning("Log queue closed or broken pipe.")
                break
            except Exception as e:
                print(f"Log listener error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        listener_logger.info("Log listener stopped.")

    log_listener_thread = threading.Thread(
        target=listener_process, daemon=True, name="LogListener"
    )
    log_listener_thread.start()
    return log_listener_thread


def setup_logging_and_run_id(args: argparse.Namespace):
    global tee_logger_instance
    run_id_source = "New"
    if args.load_checkpoint:
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                run_id_source = f"Explicit Checkpoint ({get_run_id()})"
            else:
                get_run_id()
                run_id_source = (
                    f"New (Explicit Ckpt Path Invalid: {args.load_checkpoint})"
                )
        except Exception as e:
            logger.warning(
                f"Could not determine run_id from checkpoint path '{args.load_checkpoint}': {e}. Generating new."
            )
            get_run_id()
            run_id_source = f"New (Error parsing ckpt path)"
    else:
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            run_id_source = f"Resumed Latest ({get_run_id()})"
        else:
            get_run_id()
            run_id_source = f"New (No previous runs found)"
    current_run_id = get_run_id()
    print(f"Run ID: {current_run_id} (Source: {run_id_source})")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        log_file_dir = get_run_log_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, "console_output.log")
        tee_logger_instance = TeeLogger(log_file_path, sys.stdout)
        sys.stdout = tee_logger_instance
        sys.stderr = tee_logger_instance
        print(f"Main process console output will be mirrored to: {log_file_path}")
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logger.info(f"Main process logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {current_run_id}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")
    return original_stdout, original_stderr


def cleanup_logging(
    original_stdout, original_stderr, log_queue: Optional[mp.Queue], exit_code
):
    print("[Main Finally] Restoring stdout/stderr and closing logger...")
    if log_queue:
        try:
            log_queue.put(None)
            log_queue.close()
            log_queue.join_thread()
        except Exception as qe:
            print(f"Error closing log queue: {qe}", file=original_stderr)
    if log_listener_thread:
        try:
            log_listener_thread.join(timeout=2.0)
            if log_listener_thread.is_alive():
                print(
                    "Warning: Log listener thread did not join cleanly.",
                    file=original_stderr,
                )
        except Exception as le:
            print(f"Error joining log listener thread: {le}", file=original_stderr)
    if tee_logger_instance:
        try:
            if isinstance(sys.stdout, TeeLogger):
                sys.stdout.flush()
            if isinstance(sys.stderr, TeeLogger):
                sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            tee_logger_instance.close()
            print("[Main Finally] TeeLogger closed and streams restored.")
        except Exception as log_close_err:
            original_stdout.write(f"ERROR closing TeeLogger: {log_close_err}\n")
            traceback.print_exc(file=original_stderr)
    print(f"[Main Finally] Exiting with code {exit_code}.")
    sys.exit(exit_code)


# =========================================================================
# Main Execution Block
# =========================================================================
if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Could not set start method to 'spawn', using default.")

    parser = argparse.ArgumentParser(description="AlphaZero Trainer - Multiprocess")
    parser.add_argument(
        "--load-checkpoint", type=str, default=None, help="Path to checkpoint."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    args = parser.parse_args()

    original_stdout, original_stderr = setup_logging_and_run_id(args)

    ui_to_logic_queue = mp.Queue()
    logic_to_ui_queue = mp.Queue(maxsize=10)
    stop_event = mp.Event()
    log_queue = mp.Queue()
    log_listener = setup_logging_queue_listener(log_queue)

    # Set daemon=True for simpler exit handling, rely on stop_event and timeouts
    ui_process = mp.Process(
        target=run_ui_process,
        args=(stop_event, ui_to_logic_queue, logic_to_ui_queue, log_queue),
        name="UIProcess",
        daemon=True,
    )
    logic_process = mp.Process(
        target=run_logic_process,
        args=(
            stop_event,
            ui_to_logic_queue,
            logic_to_ui_queue,
            args.load_checkpoint,
            log_queue,
        ),
        name="LogicProcess",
        daemon=True,
    )

    exit_code = 0
    try:
        logger.info("Starting UI process...")
        ui_process.start()
        logger.info("Starting Logic process...")
        logic_process.start()

        # --- Wait for Processes ---
        while True:  # Loop indefinitely until stop_event or error
            if stop_event.is_set():
                logger.info("Stop event detected by main process. Exiting wait loop.")
                break
            if not logic_process.is_alive():
                logger.warning("Logic process terminated unexpectedly. Signaling stop.")
                stop_event.set()
                exit_code = 1  # Indicate error
                break
            if not ui_process.is_alive():
                logger.warning("UI process terminated unexpectedly. Signaling stop.")
                stop_event.set()
                exit_code = 1  # Indicate error
                break
            try:
                # Sleep briefly to prevent busy-waiting
                time.sleep(0.2)
            except KeyboardInterrupt:
                logger.warning(
                    "Main process received KeyboardInterrupt. Signaling stop..."
                )
                stop_event.set()
                exit_code = 130
                break  # Exit the waiting loop

    except Exception as main_err:
        logger.critical(
            f"Error in main process coordination: {main_err}", exc_info=True
        )
        stop_event.set()
        exit_code = 1

    finally:
        logger.info("Main process initiating cleanup...")
        if not stop_event.is_set():
            stop_event.set()  # Ensure stop is signaled

        time.sleep(0.5)  # Allow processes to potentially react

        # --- Join Processes with Timeouts ---
        join_timeout_logic = 10.0  # More time for logic to save
        join_timeout_ui = 3.0

        logger.info(
            f"Waiting for Logic process to join (timeout: {join_timeout_logic}s)..."
        )
        if logic_process.is_alive():
            logic_process.join(timeout=join_timeout_logic)
        if logic_process.is_alive():
            logger.warning("Logic process did not join cleanly. Terminating.")
            try:
                logic_process.terminate()
                logic_process.join(1.0)
            except Exception as term_err:
                logger.error(f"Error terminating Logic process: {term_err}")

        logger.info(f"Waiting for UI process to join (timeout: {join_timeout_ui}s)...")
        if ui_process.is_alive():
            ui_process.join(timeout=join_timeout_ui)
        if ui_process.is_alive():
            logger.warning("UI process did not join cleanly. Terminating.")
            try:
                ui_process.terminate()
                ui_process.join(1.0)
            except Exception as term_err:
                logger.error(f"Error terminating UI process: {term_err}")

        logger.info("Processes joined or terminated.")
        cleanup_logging(original_stdout, original_stderr, log_queue, exit_code)


File: requirements.txt
pygame>=2.1.0
numpy>=1.20.0
torch>=1.10.0
tensorboard
cloudpickle
matplotlib
psutil
numba>=0.55.0
ray[default]>=2.0.0 # Added Ray

File: ui_process.py
# File: ui_process.py
# Contains the run_ui_process function and necessary UI-related imports

import pygame
import sys
import time
import queue
import logging
import logging.handlers
import multiprocessing as mp
from typing import Optional, Dict, Any

try:
    from config import VisConfig, BLACK, RED, WHITE  # Import necessary constants
    from app_state import AppState
    from ui.renderer import UIRenderer
    from ui.input_handler import InputHandler
    from app_setup import initialize_pygame
except ImportError as e:
    print(f"[UI Process Import Error] {e}", file=sys.stderr)

RENDER_DATA_SENTINEL = "RENDER_DATA"
STOP_SENTINEL = "STOP"
ERROR_SENTINEL = "ERROR"
UI_QUEUE_GET_TIMEOUT = 0.01


def run_ui_process(
    stop_event: mp.Event,
    command_queue: mp.Queue,
    render_data_queue: mp.Queue,
    log_queue: Optional[mp.Queue] = None,
):
    if log_queue:
        qh = logging.handlers.QueueHandler(log_queue)
        root = logging.getLogger()
        if not any(isinstance(h, logging.handlers.QueueHandler) for h in root.handlers):
            root.addHandler(qh)
            root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("UI Process starting...")

    vis_config = VisConfig()
    screen: Optional[pygame.Surface] = None
    clock: Optional[pygame.time.Clock] = None
    renderer: Optional[UIRenderer] = None
    input_handler: Optional[InputHandler] = None
    last_render_data: Dict[str, Any] = {}
    running = True

    try:
        screen, clock = initialize_pygame(vis_config)
        renderer = UIRenderer(screen, vis_config)
        input_handler = InputHandler(screen, renderer, command_queue, stop_event)
        renderer.set_input_handler(input_handler)
        logger.info("Pygame and UI components initialized.")
    except Exception as init_err:
        logger.critical(f"UI Initialization failed: {init_err}", exc_info=True)
        stop_event.set()
        running = False

    while running and not stop_event.is_set():
        if not clock or not screen or not renderer or not input_handler:
            logger.error("Critical UI components not initialized. Exiting UI loop.")
            stop_event.set()
            break

        # --- Handle Input ---
        try:
            input_handler.update_state(
                last_render_data.get("app_state", AppState.INITIALIZING.value),
                last_render_data.get("cleanup_confirmation_active", False),
                last_render_data.get("is_process_running", False),  # Pass worker status
            )
            running = input_handler.handle_input()  # This might set stop_event
            if not running:
                logger.info("Input handler requested exit.")
                break
        except Exception as input_err:
            logger.error(f"Error handling input: {input_err}", exc_info=True)
            stop_event.set()
            running = False
            break

        # --- Get Render Data ---
        try:
            item = render_data_queue.get(timeout=UI_QUEUE_GET_TIMEOUT)
            if isinstance(item, dict) and RENDER_DATA_SENTINEL in item:
                last_render_data = item[RENDER_DATA_SENTINEL]
            elif item == STOP_SENTINEL:
                logger.info("Received STOP sentinel from logic process.")
                running = False
                break
            elif isinstance(item, dict) and ERROR_SENTINEL in item:
                logger.error(f"Received ERROR sentinel: {item[ERROR_SENTINEL]}")
                last_render_data = last_render_data.copy()
                last_render_data["app_state"] = AppState.ERROR.value
                last_render_data["status"] = item[ERROR_SENTINEL]
        except queue.Empty:
            pass  # No new data
        except (EOFError, BrokenPipeError):
            logger.warning("Render data queue connection lost.")
            running = False
            stop_event.set()
        except Exception as queue_err:
            logger.error(f"Error getting render data: {queue_err}", exc_info=True)

        # --- Render Frame ---
        if last_render_data:
            try:
                renderer.render_all(**last_render_data)
            except Exception as render_err:
                logger.error(f"Error rendering frame: {render_err}", exc_info=True)
                try:  # Fallback render
                    screen.fill(BLACK)
                    font = pygame.font.Font(None, 30)
                    err_surf = font.render("Error during rendering!", True, RED)
                    screen.blit(
                        err_surf, err_surf.get_rect(center=screen.get_rect().center)
                    )
                    pygame.display.flip()
                except Exception:
                    pass
        else:  # Render waiting screen
            try:
                screen.fill(BLACK)
                font = pygame.font.Font(None, 30)
                wait_surf = font.render("Waiting for data...", True, WHITE)
                screen.blit(
                    wait_surf, wait_surf.get_rect(center=screen.get_rect().center)
                )
                pygame.display.flip()
            except Exception:
                pass

        clock.tick(vis_config.FPS if vis_config.FPS > 0 else 60)

    logger.info("UI Process shutting down...")
    pygame.quit()
    logger.info("UI Process finished.")


File: agent\alphazero_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import ray
import logging

from config import ModelConfig, EnvConfig
from utils.types import StateType, ActionType

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Basic Residual Block for CNN."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """
    Neural Network for AlphaZero.
    Takes game state features and outputs policy logits and a value estimate.
    Includes methods for single and batched predictions.
    """

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        model_config: Optional[ModelConfig.Network] = None,
    ):
        super().__init__()
        self.env_cfg = env_config if env_config else EnvConfig()
        self.model_cfg = model_config if model_config else ModelConfig.Network()

        # --- Input Processing Layers ---
        grid_input_channels = self.env_cfg.GRID_STATE_SHAPE[0]
        conv_channels = self.model_cfg.CONV_CHANNELS
        current_channels = grid_input_channels
        conv_layers = []
        for out_channels in conv_channels:
            conv_layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=self.model_cfg.CONV_KERNEL_SIZE,
                    stride=self.model_cfg.CONV_STRIDE,
                    padding=self.model_cfg.CONV_PADDING,
                    bias=not self.model_cfg.USE_BATCHNORM_CONV,
                )
            )
            if self.model_cfg.USE_BATCHNORM_CONV:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(self.model_cfg.CONV_ACTIVATION())
            # Add ResidualBlock *after* activation
            conv_layers.append(ResidualBlock(out_channels))
            current_channels = out_channels
        self.conv_backbone = nn.Sequential(*conv_layers)

        conv_output_size = self._calculate_conv_output_size(
            (grid_input_channels, self.env_cfg.ROWS, self.env_cfg.COLS)
        )

        shape_input_dim = self.env_cfg.SHAPE_STATE_DIM
        shape_mlp_dims = self.model_cfg.SHAPE_FEATURE_MLP_DIMS
        shape_layers = []
        current_shape_dim = shape_input_dim
        for dim in shape_mlp_dims:
            shape_layers.append(nn.Linear(current_shape_dim, dim))
            shape_layers.append(self.model_cfg.SHAPE_MLP_ACTIVATION())
            current_shape_dim = dim
        self.shape_mlp = nn.Sequential(*shape_layers)
        shape_output_dim = current_shape_dim if shape_mlp_dims else shape_input_dim

        other_features_dim = (
            self.env_cfg.SHAPE_AVAILABILITY_DIM + self.env_cfg.EXPLICIT_FEATURES_DIM
        )

        combined_input_dim = conv_output_size + shape_output_dim + other_features_dim
        combined_fc_dims = self.model_cfg.COMBINED_FC_DIMS
        fusion_layers = []
        current_combined_dim = combined_input_dim
        for dim in combined_fc_dims:
            fusion_layers.append(
                nn.Linear(
                    current_combined_dim, dim, bias=not self.model_cfg.USE_BATCHNORM_FC
                )
            )
            if self.model_cfg.USE_BATCHNORM_FC:
                fusion_layers.append(nn.BatchNorm1d(dim))
            fusion_layers.append(self.model_cfg.COMBINED_ACTIVATION())
            if self.model_cfg.DROPOUT_FC > 0:
                fusion_layers.append(nn.Dropout(self.model_cfg.DROPOUT_FC))
            current_combined_dim = dim
        self.fusion_mlp = nn.Sequential(*fusion_layers)
        fusion_output_dim = current_combined_dim

        self.policy_head = nn.Linear(fusion_output_dim, self.env_cfg.ACTION_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh()
        )

    def _calculate_conv_output_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Helper to calculate the flattened output size of the conv backbone."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.conv_backbone(dummy_input)
            return int(torch.flatten(output, 1).shape[1])

    def forward(
        self, state: StateType  # Expects Tensors in the dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        Assumes input state dictionary contains tensors.
        """
        grid = state["grid"]
        shapes = state["shapes"]
        shape_availability = state["shape_availability"]
        explicit_features = state["explicit_features"]

        conv_out = self.conv_backbone(grid)
        flat_conv_out = torch.flatten(conv_out, 1)

        if self.model_cfg.SHAPE_FEATURE_MLP_DIMS:
            shape_out = self.shape_mlp(shapes)
        else:
            shape_out = shapes

        other_features = torch.cat([shape_availability, explicit_features], dim=-1)
        combined_features = torch.cat([flat_conv_out, shape_out, other_features], dim=1)

        fused_out = self.fusion_mlp(combined_features)

        policy_logits = self.policy_head(fused_out)
        value = self.value_head(fused_out)

        return policy_logits, value

    def _prepare_state_batch(
        self, state_numpy_list: List[StateType], device: torch.device
    ) -> StateType:
        """Converts a list of numpy state dicts to a batched tensor state dict on the specified device."""
        batched_tensors = {key: [] for key in state_numpy_list[0].keys()}
        for state_numpy in state_numpy_list:
            for key, value in state_numpy.items():
                if key in batched_tensors:
                    batched_tensors[key].append(value)
                else:
                    logger.warning(
                        f"Warning: Key {key} not found in initial state dict during batching."
                    )
        final_batched_states = {
            k: torch.from_numpy(np.stack(v)).to(device)
            for k, v in batched_tensors.items()
        }
        return final_batched_states

    def predict_batch(
        self, state_numpy_list: List[StateType], device: torch.device
    ) -> Tuple[List[Dict[ActionType, float]], List[float]]:
        """
        Performs batched prediction for MCTS.
        """
        if not state_numpy_list:
            return [], []
        batched_state_tensors = self._prepare_state_batch(state_numpy_list, device)
        self.eval()
        with torch.no_grad():
            policy_logits_batch, value_batch = self.forward(batched_state_tensors)
        policy_probs_batch = F.softmax(policy_logits_batch, dim=-1).cpu().numpy()
        values = value_batch.squeeze(-1).cpu().numpy().tolist()
        policy_dicts = []
        for policy_probs in policy_probs_batch:
            policy_dicts.append({i: float(prob) for i, prob in enumerate(policy_probs)})
        return policy_dicts, values

    def predict(
        self, state_numpy: StateType, device: torch.device
    ) -> Tuple[Dict[ActionType, float], float]:
        """
        Convenience method for single prediction.
        """
        policy_list, value_list = self.predict_batch([state_numpy], device)
        return policy_list[0], value_list[0]

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns the model's state dictionary."""
        return self.state_dict()

    def load_state_dict_custom(self, state_dict: Dict[str, Any]):
        """Loads the model's state dictionary."""
        super().load_state_dict(state_dict)


# --- Ray Actor Wrapper ---
@ray.remote(num_cpus=0, num_gpus=1 if torch.cuda.is_available() else 0)
class AgentPredictor:
    """Ray actor to handle batched predictions using the AlphaZeroNet."""

    def __init__(self, env_config: EnvConfig, model_config: ModelConfig.Network):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroNet(env_config=env_config, model_config=model_config).to(
            self.device
        )
        self.model.eval()
        logger.info(f"[AgentPredictor] Initialized on device: {self.device}")

    def predict_batch(
        self, state_numpy_list: List[StateType]
    ) -> Tuple[List[Dict[ActionType, float]], List[float]]:
        """Performs batched prediction using the internal model."""
        if not state_numpy_list:
            return [], []
        return self.model.predict_batch(state_numpy_list, self.device)

    def get_weights(self) -> Dict[str, Any]:
        """Returns the current model weights."""
        return self.model.state_dict()

    def set_weights(self, weights: Dict[str, Any]):
        """Sets the model weights."""
        self.model.load_state_dict(weights)
        self.model.eval()
        logger.info("[AgentPredictor] Weights updated.")

    def get_param_count(self) -> int:
        """Calculates and returns the number of trainable parameters."""
        try:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        except Exception as e:
            logger.error(f"[AgentPredictor] Error calculating param count: {e}")
            return 0

    def health_check(self):
        """Basic health check method for Ray."""
        return "OK"


File: agent\__init__.py


File: config\constants.py
"""
Defines constants shared across different modules, primarily visual elements,
to avoid circular imports and keep configuration clean.
"""

# Colors (RGB tuples 0-255)
WHITE: tuple[int, int, int] = (255, 255, 255)
BLACK: tuple[int, int, int] = (0, 0, 0)
LIGHTG: tuple[int, int, int] = (140, 140, 140)
GRAY: tuple[int, int, int] = (50, 50, 50)
DARK_GRAY: tuple[int, int, int] = (30, 30, 30)
RED: tuple[int, int, int] = (255, 50, 50)
DARK_RED: tuple[int, int, int] = (80, 10, 10)
BLUE: tuple[int, int, int] = (50, 50, 255)
YELLOW: tuple[int, int, int] = (255, 255, 100)
GREEN: tuple[int, int, int] = (50, 200, 50)
DARK_GREEN: tuple[int, int, int] = (20, 80, 20)  # Added Dark Green
ORANGE: tuple[int, int, int] = (255, 165, 0)
PURPLE: tuple[int, int, int] = (128, 0, 128)
CYAN: tuple[int, int, int] = (0, 255, 255)

GOOGLE_COLORS: list[tuple[int, int, int]] = [
    (15, 157, 88),  # Green
    (244, 180, 0),  # Yellow/Orange
    (66, 133, 244),  # Blue
    (219, 68, 55),  # Red
]
LINE_CLEAR_FLASH_COLOR: tuple[int, int, int] = (180, 180, 220)
LINE_CLEAR_HIGHLIGHT_COLOR: tuple[int, int, int, int] = (255, 255, 0, 180)  # RGBA
GAME_OVER_FLASH_COLOR: tuple[int, int, int] = (255, 0, 0)

# MCTS Visualization Colors
MCTS_NODE_WIN_COLOR: tuple[int, int, int] = DARK_GREEN  # Use darker green for node fill
MCTS_NODE_LOSS_COLOR: tuple[int, int, int] = DARK_RED
MCTS_NODE_NEUTRAL_COLOR: tuple[int, int, int] = DARK_GRAY  # Use darker gray
MCTS_NODE_BORDER_COLOR: tuple[int, int, int] = GRAY  # Lighter border
MCTS_NODE_SELECTED_BORDER_COLOR: tuple[int, int, int] = YELLOW
MCTS_EDGE_COLOR: tuple[int, int, int] = GRAY  # Lighter edge color
MCTS_EDGE_HIGHLIGHT_COLOR: tuple[int, int, int] = WHITE
MCTS_INFO_TEXT_COLOR: tuple[int, int, int] = WHITE
MCTS_NODE_TEXT_COLOR: tuple[int, int, int] = WHITE
MCTS_NODE_PRIOR_COLOR: tuple[int, int, int] = CYAN
MCTS_NODE_SCORE_COLOR: tuple[int, int, int] = ORANGE
MCTS_MINI_GRID_BG_COLOR: tuple[int, int, int] = (40, 40, 40)  # Background for mini-grid
MCTS_MINI_GRID_LINE_COLOR: tuple[int, int, int] = (70, 70, 70)  # Lines for mini-grid
MCTS_MINI_GRID_OCCUPIED_COLOR: tuple[int, int, int] = (
    200,
    200,
    200,
)  # Occupied cells in mini-grid


File: config\core.py
# File: config/core.py
import torch
from typing import List, Tuple, Optional

from .constants import (
    WHITE,
    BLACK,
    LIGHTG,
    GRAY,
    DARK_GRAY,
    RED,
    DARK_RED,
    BLUE,
    YELLOW,
    GREEN,
    DARK_GREEN,
    ORANGE,
    PURPLE,
    CYAN,
    GOOGLE_COLORS,
    LINE_CLEAR_FLASH_COLOR,
    LINE_CLEAR_HIGHLIGHT_COLOR,
    GAME_OVER_FLASH_COLOR,
    MCTS_NODE_WIN_COLOR,
    MCTS_NODE_LOSS_COLOR,
    MCTS_NODE_NEUTRAL_COLOR,
    MCTS_NODE_BORDER_COLOR,
    MCTS_NODE_SELECTED_BORDER_COLOR,
    MCTS_EDGE_COLOR,
    MCTS_EDGE_HIGHLIGHT_COLOR,
    MCTS_INFO_TEXT_COLOR,
    MCTS_NODE_TEXT_COLOR,
    MCTS_NODE_PRIOR_COLOR,
    MCTS_NODE_SCORE_COLOR,
    MCTS_MINI_GRID_BG_COLOR,
    MCTS_MINI_GRID_LINE_COLOR,
    MCTS_MINI_GRID_OCCUPIED_COLOR,
)


class MCTSConfig:
    """Configuration parameters for the Monte Carlo Tree Search."""

    PUCT_C: float = 1.25  # Slightly lower exploration emphasis
    NUM_SIMULATIONS: int = 25  # Significantly reduced simulations per move
    TEMPERATURE_INITIAL: float = 1.0
    TEMPERATURE_FINAL: float = 0.1  # Anneal faster
    TEMPERATURE_ANNEAL_STEPS: int = 15  # Anneal over fewer steps
    DIRICHLET_ALPHA: float = 0.3
    DIRICHLET_EPSILON: float = 0.25
    MAX_SEARCH_DEPTH: int = 50  # Reduced max depth


class VisConfig:
    NUM_ENVS_TO_RENDER = 2  # Reduced to match NUM_SELF_PLAY_WORKERS
    FPS = 30  # Limit FPS for smoother UI during short runs
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00  # Keep at 0 for workers
    LEFT_PANEL_RATIO = 0.5
    ENV_SPACING = 2
    ENV_GRID_PADDING = 1

    WHITE = WHITE
    BLACK = BLACK
    LIGHTG = LIGHTG
    GRAY = GRAY
    DARK_GRAY = DARK_GRAY
    RED = RED
    DARK_RED = DARK_RED
    BLUE = BLUE
    YELLOW = YELLOW
    GREEN = GREEN
    DARK_GREEN = DARK_GREEN
    ORANGE = ORANGE
    PURPLE = PURPLE
    CYAN = CYAN
    GOOGLE_COLORS = GOOGLE_COLORS
    LINE_CLEAR_FLASH_COLOR = LINE_CLEAR_FLASH_COLOR
    LINE_CLEAR_HIGHLIGHT_COLOR = LINE_CLEAR_HIGHLIGHT_COLOR
    GAME_OVER_FLASH_COLOR = GAME_OVER_FLASH_COLOR
    MCTS_NODE_WIN_COLOR = MCTS_NODE_WIN_COLOR
    MCTS_NODE_LOSS_COLOR = MCTS_NODE_LOSS_COLOR
    MCTS_NODE_NEUTRAL_COLOR = MCTS_NODE_NEUTRAL_COLOR
    MCTS_NODE_BORDER_COLOR = MCTS_NODE_BORDER_COLOR
    MCTS_NODE_SELECTED_BORDER_COLOR = MCTS_NODE_SELECTED_BORDER_COLOR
    MCTS_EDGE_COLOR = MCTS_EDGE_COLOR
    MCTS_EDGE_HIGHLIGHT_COLOR = MCTS_EDGE_HIGHLIGHT_COLOR
    MCTS_INFO_TEXT_COLOR = MCTS_INFO_TEXT_COLOR
    MCTS_NODE_TEXT_COLOR = MCTS_NODE_TEXT_COLOR
    MCTS_NODE_PRIOR_COLOR = MCTS_NODE_PRIOR_COLOR
    MCTS_NODE_SCORE_COLOR = MCTS_NODE_SCORE_COLOR
    MCTS_MINI_GRID_BG_COLOR = MCTS_MINI_GRID_BG_COLOR
    MCTS_MINI_GRID_LINE_COLOR = MCTS_MINI_GRID_LINE_COLOR
    MCTS_MINI_GRID_OCCUPIED_COLOR = MCTS_MINI_GRID_OCCUPIED_COLOR


class EnvConfig:
    ROWS = 8
    COLS = 15
    GRID_FEATURES_PER_CELL = 2
    SHAPE_FEATURES_PER_SHAPE = 5
    NUM_SHAPE_SLOTS = 3
    EXPLICIT_FEATURES_DIM = 10
    CALCULATE_POTENTIAL_OUTCOMES_IN_STATE = False

    @property
    def GRID_STATE_SHAPE(self) -> Tuple[int, int, int]:
        return (self.GRID_FEATURES_PER_CELL, self.ROWS, self.COLS)

    @property
    def SHAPE_STATE_DIM(self) -> int:
        return self.NUM_SHAPE_SLOTS * self.SHAPE_FEATURES_PER_SHAPE

    @property
    def SHAPE_AVAILABILITY_DIM(self) -> int:
        return self.NUM_SHAPE_SLOTS

    @property
    def ACTION_DIM(self) -> int:
        return self.NUM_SHAPE_SLOTS * (self.ROWS * self.COLS)


class RNNConfig:
    USE_RNN = False
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2


class TransformerConfig:
    USE_TRANSFORMER = False
    TRANSFORMER_D_MODEL = 256
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_DIM_FEEDFORWARD = 512
    TRANSFORMER_NUM_LAYERS = 3
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "relu"


class TrainConfig:
    """Configuration parameters for the Training Worker."""

    CHECKPOINT_SAVE_FREQ = 200  # Save more frequently for short runs
    LOAD_CHECKPOINT_PATH: Optional[str] = None
    NUM_SELF_PLAY_WORKERS: int = 2# Reduced workers
    BATCH_SIZE: int = 64  # Smaller batch size
    LEARNING_RATE: float = 3e-4  # Slightly higher LR might help faster convergence
    WEIGHT_DECAY: float = 1e-5
    NUM_TRAINING_STEPS_PER_ITER: int = 20  # Fewer training steps per iteration
    MIN_BUFFER_SIZE_TO_TRAIN: int = 200  # Start training sooner
    BUFFER_CAPACITY: int = 2000  # Smaller buffer
    POLICY_LOSS_WEIGHT: float = 1.0
    VALUE_LOSS_WEIGHT: float = 1.0
    USE_LR_SCHEDULER: bool = True
    SCHEDULER_TYPE: str = "CosineAnnealingLR"
    SCHEDULER_T_MAX: int = 5000  # Reduced scheduler cycle for faster testing
    SCHEDULER_ETA_MIN: float = 1e-6


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        CONV_CHANNELS = [16, 32]  # Reduced channels
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True
        SHAPE_FEATURE_MLP_DIMS = [32]  # Reduced MLP dims
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU
        COMBINED_FC_DIMS = [64]  # Reduced FC dims
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.1


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [50, 200]  # Smaller windows for faster feedback
    CONSOLE_LOG_FREQ = 50  # Log more frequently
    PLOT_DATA_WINDOW = 1000  # Reduced plot window


class DemoConfig:
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Click Preview]=Select/Deselect | [Click Grid]=Place | [ESC]=Exit"
    DEBUG_HELP_TEXT = "[Click]=Toggle Triangle | [R]=Reset Grid | [ESC]=Exit"


File: config\general.py
import torch
import os
import time
from typing import Optional

# --- Base Directories ---
BASE_CHECKPOINT_DIR = "checkpoints"
BASE_LOG_DIR = "logs"

# --- Device ---
DEVICE: Optional[torch.device] = None


def set_device(device: torch.device):
    """Sets the global DEVICE variable."""
    global DEVICE
    DEVICE = device
    print(f"[Config] Global DEVICE set to: {DEVICE}")


# --- Random Seed ---
RANDOM_SEED = 42

# --- Run ID and Paths (Dynamically Determined) ---
_current_run_id: Optional[str] = None


def get_run_id() -> str:
    """Gets the current run ID, generating one if not set."""
    global _current_run_id
    if _current_run_id is None:
        _current_run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"[Config] Generated new RUN_ID: {_current_run_id}")
    return _current_run_id


def set_run_id(run_id: str):
    """Sets the run ID, typically when resuming a run."""
    global _current_run_id
    if _current_run_id is not None and _current_run_id != run_id:
        print(
            f"[Config] WARNING: Overwriting existing RUN_ID '{_current_run_id}' with '{run_id}'."
        )
    elif _current_run_id is None:
        print(f"[Config] Setting RUN_ID to resumed ID: {run_id}")
    _current_run_id = run_id


def get_run_checkpoint_dir() -> str:
    """Gets the checkpoint directory for the current run."""
    # Checkpoints will now likely store NN weights, maybe MCTS stats
    return os.path.join(BASE_CHECKPOINT_DIR, get_run_id())


def get_run_log_dir() -> str:
    """Gets the TensorBoard log directory for the current run."""
    # Ensure the base 'tensorboard' subdirectory exists within BASE_LOG_DIR
    tb_base = os.path.join(BASE_LOG_DIR, "tensorboard")
    return os.path.join(tb_base, get_run_id())


def get_console_log_dir() -> str:
    """Gets the directory for console logs for the current run."""
    # Place console logs directly within the run-specific log directory
    return get_run_log_dir()


def get_model_save_path() -> str:
    """Gets the base model save path for the current run (adapt name later)."""
    # Updated filename for AlphaZero NN
    return os.path.join(get_run_checkpoint_dir(), "alphazero_nn.pth")


File: config\utils.py
import torch
from typing import Dict, Any
from .core import (
    VisConfig,
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    DemoConfig,
    TransformerConfig,
    MCTSConfig,  
)

from .general import DEVICE, RANDOM_SEED, get_run_id


def get_config_dict() -> Dict[str, Any]:
    """Returns a flat dictionary of all relevant config values for logging."""
    all_configs = {}

    def flatten_class(cls, prefix=""):
        d = {}
        instance = None
        try:
            instance = cls()
        except Exception:
            instance = None

        for k, v in vars(cls).items():
            if (
                not k.startswith("__")
                and not callable(v)
                and not isinstance(v, type)
                and not hasattr(v, "__module__")
            ):
                is_property = isinstance(getattr(cls, k, None), property)
                if is_property and instance:
                    try:
                        v = getattr(instance, k)
                    except Exception:
                        continue
                elif is_property and not instance:
                    continue
                d[f"{prefix}{k}"] = v
        return d

    # Flatten core config classes
    all_configs.update(flatten_class(VisConfig, "Vis."))
    all_configs.update(flatten_class(EnvConfig, "Env."))
    all_configs.update(flatten_class(RNNConfig, "RNN."))
    all_configs.update(flatten_class(TrainConfig, "Train."))
    all_configs.update(flatten_class(ModelConfig.Network, "Model.Net."))
    all_configs.update(flatten_class(StatsConfig, "Stats."))
    # all_configs.update(flatten_class(TensorBoardConfig, "TB.")) # Removed
    all_configs.update(flatten_class(DemoConfig, "Demo."))
    all_configs.update(flatten_class(TransformerConfig, "Transformer."))
    all_configs.update(flatten_class(MCTSConfig, "MCTS."))  # Flatten MCTSConfig

    # Add general config values
    all_configs["General.DEVICE"] = str(DEVICE) if DEVICE else "None"
    all_configs["General.RANDOM_SEED"] = RANDOM_SEED
    all_configs["General.RUN_ID"] = get_run_id()

    all_configs = {
        k: v for k, v in all_configs.items() if not (k.endswith("_PATH") and v is None)
    }

    # Convert non-basic types to strings for logging
    for key, value in all_configs.items():
        if isinstance(value, type) and issubclass(value, torch.nn.Module):
            all_configs[key] = value.__name__
        elif isinstance(value, (list, tuple)):
            all_configs[key] = str(value)
        elif not isinstance(value, (int, float, str, bool)):
            all_configs[key] = str(value) if value is not None else "None"

    return all_configs


File: config\validation.py
# File: config/validation.py
from .core import (
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    VisConfig,
    TransformerConfig,
    MCTSConfig,
)
from .general import (
    DEVICE,
    get_run_id,
    get_run_log_dir,
    get_run_checkpoint_dir,
)


def print_config_info_and_validate():
    env_config_instance = EnvConfig()
    rnn_config_instance = RNNConfig()
    transformer_config_instance = TransformerConfig()
    train_config_instance = TrainConfig()
    mcts_config_instance = MCTSConfig()
    vis_config_instance = VisConfig()  # Get instance for NUM_ENVS_TO_RENDER
    stats_config_instance = StatsConfig()  # Get instance for logging frequency

    run_id = get_run_id()
    run_log_dir = get_run_log_dir()
    run_checkpoint_dir = get_run_checkpoint_dir()

    print("-" * 70)
    print(f"RUN ID: {run_id}")
    print(f"Log Directory: {run_log_dir}")
    print(f"Checkpoint Directory: {run_checkpoint_dir}")
    print(f"Device: {DEVICE}")

    if train_config_instance.LOAD_CHECKPOINT_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD CHECKPOINT specified: {train_config_instance.LOAD_CHECKPOINT_PATH} ***\n"
            "*** CheckpointManager will attempt to load this path (NN weights, Optimizer, Stats). ***\n"
            + "*" * 70
        )
    else:
        print(
            "--- No explicit checkpoint path. CheckpointManager will attempt auto-resume if applicable. ---"
        )

    print("--- Training Algorithm: AlphaZero (MCTS + NN) ---")

    if rnn_config_instance.USE_RNN:
        print(
            f"--- Warning: RNN configured ON ({rnn_config_instance.LSTM_HIDDEN_SIZE}, {rnn_config_instance.LSTM_NUM_LAYERS}) but not used by AlphaZeroNet ---"
        )
    if transformer_config_instance.USE_TRANSFORMER:
        print(
            f"--- Warning: Transformer configured ON ({transformer_config_instance.TRANSFORMER_D_MODEL}, {transformer_config_instance.TRANSFORMER_NHEAD}, {transformer_config_instance.TRANSFORMER_NUM_LAYERS}) but not used by AlphaZeroNet ---"
        )

    print(
        f"Config: Env=(R={env_config_instance.ROWS}, C={env_config_instance.COLS}), "
        f"GridState={env_config_instance.GRID_STATE_SHAPE}, "
        f"ShapeState={env_config_instance.SHAPE_STATE_DIM}, "
        f"ActionDim={env_config_instance.ACTION_DIM}"
    )
    cnn_str = str(ModelConfig.Network.CONV_CHANNELS).replace(" ", "")
    mlp_str = str(ModelConfig.Network.COMBINED_FC_DIMS).replace(" ", "")
    shape_mlp_cfg_str = str(ModelConfig.Network.SHAPE_FEATURE_MLP_DIMS).replace(" ", "")
    dropout = ModelConfig.Network.DROPOUT_FC
    print(
        f"Network Base: CNN={cnn_str}, ShapeMLP={shape_mlp_cfg_str}, Fusion={mlp_str}, Dropout={dropout}"
    )

    print(
        f"MCTS: Sims={mcts_config_instance.NUM_SIMULATIONS}, "
        f"PUCT_C={mcts_config_instance.PUCT_C:.2f}, "
        f"Temp={mcts_config_instance.TEMPERATURE_INITIAL:.2f}->{mcts_config_instance.TEMPERATURE_FINAL:.2f} (Steps: {mcts_config_instance.TEMPERATURE_ANNEAL_STEPS}), "
        f"Dirichlet(α={mcts_config_instance.DIRICHLET_ALPHA:.2f}, ε={mcts_config_instance.DIRICHLET_EPSILON:.2f})"
    )

    scheduler_info = "DISABLED"
    if train_config_instance.USE_LR_SCHEDULER:
        scheduler_info = f"{train_config_instance.SCHEDULER_TYPE} (T_max={train_config_instance.SCHEDULER_T_MAX:,}, eta_min={train_config_instance.SCHEDULER_ETA_MIN:.1e})"
    print(
        f"Training: Batch={train_config_instance.BATCH_SIZE}, LR={train_config_instance.LEARNING_RATE:.1e}, "
        f"WD={train_config_instance.WEIGHT_DECAY:.1e}, Scheduler={scheduler_info}"
    )
    print(
        f"Buffer: Capacity={train_config_instance.BUFFER_CAPACITY:,}, MinSize={train_config_instance.MIN_BUFFER_SIZE_TO_TRAIN:,}, Steps/Iter={train_config_instance.NUM_TRAINING_STEPS_PER_ITER}"
    )
    print(
        f"Workers: Self-Play={train_config_instance.NUM_SELF_PLAY_WORKERS}, Training=1"
    )
    print(
        f"Stats: AVG_WINDOWS={stats_config_instance.STATS_AVG_WINDOW}, Console Log Freq={stats_config_instance.CONSOLE_LOG_FREQ} (updates/episodes)"
    )

    render_info = (
        f"Live Self-Play Workers (Max: {vis_config_instance.NUM_ENVS_TO_RENDER})"
    )
    print(f"--- Rendering {render_info} in Game Area ---")
    print("-" * 70)


File: config\__init__.py
# File: config/__init__.py
# config/__init__.py
# This file marks the 'config' directory as a Python package.

# Import core configuration classes to make them available directly under 'config'
from .core import (
    VisConfig,
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    # TensorBoardConfig removed
    DemoConfig,
    TransformerConfig,
    MCTSConfig,
)

# Import general configuration settings and functions
from .general import (
    DEVICE,
    RANDOM_SEED,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    set_device,
    get_run_id,
    set_run_id,
    get_run_checkpoint_dir,
    get_run_log_dir,
    get_console_log_dir,
    get_model_save_path,
)

# Import utility functions
from .utils import get_config_dict

# Import validation function
from .validation import print_config_info_and_validate

# Import constants
from .constants import (
    WHITE,
    BLACK,
    LIGHTG,
    GRAY,
    DARK_GRAY,
    RED,
    DARK_RED,
    BLUE,
    YELLOW,
    GREEN,
    DARK_GREEN,  # Added DARK_GREEN import
    ORANGE,
    PURPLE,
    CYAN,
    GOOGLE_COLORS,
    LINE_CLEAR_FLASH_COLOR,
    LINE_CLEAR_HIGHLIGHT_COLOR,
    GAME_OVER_FLASH_COLOR,
    # MCTS Colors (also available directly)
    MCTS_NODE_WIN_COLOR,
    MCTS_NODE_LOSS_COLOR,
    MCTS_NODE_NEUTRAL_COLOR,
    MCTS_NODE_BORDER_COLOR,
    MCTS_NODE_SELECTED_BORDER_COLOR,
    MCTS_EDGE_COLOR,
    MCTS_EDGE_HIGHLIGHT_COLOR,
    MCTS_INFO_TEXT_COLOR,
    MCTS_NODE_TEXT_COLOR,
    MCTS_NODE_PRIOR_COLOR,
    MCTS_NODE_SCORE_COLOR,
    MCTS_MINI_GRID_BG_COLOR,
    MCTS_MINI_GRID_LINE_COLOR,
    MCTS_MINI_GRID_OCCUPIED_COLOR,
)


# Define __all__ to control what 'from config import *' imports
__all__ = [
    # Core Configs
    "VisConfig",
    "EnvConfig",
    "RNNConfig",
    "TrainConfig",
    "ModelConfig",
    "StatsConfig",
    # "TensorBoardConfig", # Removed
    "DemoConfig",
    "TransformerConfig",
    "MCTSConfig",
    # General Configs
    "DEVICE",
    "RANDOM_SEED",
    "BASE_CHECKPOINT_DIR",
    "BASE_LOG_DIR",
    "set_device",
    "get_run_id",
    "set_run_id",
    "get_run_checkpoint_dir",
    "get_run_log_dir",
    "get_console_log_dir",
    "get_model_save_path",
    # Utils
    "get_config_dict",
    "print_config_info_and_validate",
    # Constants
    "WHITE",
    "BLACK",
    "LIGHTG",
    "GRAY",
    "DARK_GRAY",
    "RED",
    "DARK_RED",
    "BLUE",
    "YELLOW",
    "GREEN",
    "DARK_GREEN",  # Added DARK_GREEN export
    "ORANGE",
    "PURPLE",
    "CYAN",
    "GOOGLE_COLORS",
    "LINE_CLEAR_FLASH_COLOR",
    "LINE_CLEAR_HIGHLIGHT_COLOR",
    "GAME_OVER_FLASH_COLOR",
    # MCTS Colors
    "MCTS_NODE_WIN_COLOR",
    "MCTS_NODE_LOSS_COLOR",
    "MCTS_NODE_NEUTRAL_COLOR",
    "MCTS_NODE_BORDER_COLOR",
    "MCTS_NODE_SELECTED_BORDER_COLOR",
    "MCTS_EDGE_COLOR",
    "MCTS_EDGE_HIGHLIGHT_COLOR",
    "MCTS_INFO_TEXT_COLOR",
    "MCTS_NODE_TEXT_COLOR",
    "MCTS_NODE_PRIOR_COLOR",
    "MCTS_NODE_SCORE_COLOR",
    "MCTS_MINI_GRID_BG_COLOR",
    "MCTS_MINI_GRID_LINE_COLOR",
    "MCTS_MINI_GRID_OCCUPIED_COLOR",
]


File: environment\game_demo_logic.py
# File: environment/game_demo_logic.py
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .game_state import GameState
    from .shape import Shape


class GameDemoLogic:
    """Handles logic specific to the interactive Demo/Debug mode."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def select_shape_for_drag(self, shape_index: int):
        """Selects a shape for dragging if available."""
        if (
            0 <= shape_index < len(self.gs.shapes)
            and self.gs.shapes[shape_index] is not None
        ):
            # If clicking the already selected shape, deselect it
            if self.gs.demo_selected_shape_idx == shape_index:
                self.gs.demo_selected_shape_idx = -1  # Indicate no selection
                self.gs.demo_dragged_shape_idx = None
                self.gs.demo_snapped_position = None
            else:
                self.gs.demo_selected_shape_idx = shape_index
                self.gs.demo_dragged_shape_idx = shape_index
                self.gs.demo_snapped_position = None  # Reset snap on new selection

    def deselect_dragged_shape(self):
        """Deselects any currently dragged shape."""
        self.gs.demo_dragged_shape_idx = None
        self.gs.demo_snapped_position = None
        # Keep demo_selected_shape_idx as is, maybe user wants to re-drag later

    def update_snapped_position(self, grid_pos: Optional[Tuple[int, int]]):
        """Updates the snapped position based on grid hover."""
        if self.gs.demo_dragged_shape_idx is None:
            self.gs.demo_snapped_position = None
            return

        shape_to_drag = self.gs.shapes[self.gs.demo_dragged_shape_idx]
        if shape_to_drag is None:
            self.gs.demo_snapped_position = None
            return

        if grid_pos is not None and self.gs.grid.can_place(
            shape_to_drag, grid_pos[0], grid_pos[1]
        ):
            self.gs.demo_snapped_position = grid_pos
        else:
            self.gs.demo_snapped_position = None

    def place_dragged_shape(self) -> bool:
        """Attempts to place the currently dragged shape at the snapped position."""
        if (
            self.gs.demo_dragged_shape_idx is not None
            and self.gs.demo_snapped_position is not None
        ):
            shape_idx = self.gs.demo_dragged_shape_idx
            r, c = self.gs.demo_snapped_position

            # Encode the action based on the demo state
            action_index = self.gs.logic.encode_action(shape_idx, r, c)

            # Use the core game logic step function
            _, done = self.gs.logic.step(action_index)

            # Reset demo drag state after placement attempt
            self.gs.demo_dragged_shape_idx = None
            self.gs.demo_snapped_position = None
            self.gs.demo_selected_shape_idx = -1  # Deselect after placement

            return not done  # Return True if placement was successful (game not over)
        return False

    def get_dragged_shape_info(
        self,
    ) -> Tuple[Optional["Shape"], Optional[Tuple[int, int]]]:
        """Returns the currently dragged shape and its snapped position."""
        if self.gs.demo_dragged_shape_idx is None:
            return None, None
        shape = self.gs.shapes[self.gs.demo_dragged_shape_idx]
        return shape, self.gs.demo_snapped_position

    def toggle_triangle_debug(self, row: int, col: int):
        """Toggles the occupied state of a triangle in debug mode and checks for line clears."""
        if not self.gs.grid.valid(row, col):
            return

        tri = self.gs.grid.triangles[row][col]
        if tri.is_death:
            return  # Cannot toggle death cells

        # Toggle state
        tri.is_occupied = not tri.is_occupied
        self.gs.grid._occupied_np[row, col] = tri.is_occupied  # Update numpy array
        tri.color = self.gs.vis_config.WHITE if tri.is_occupied else None

        # Manually trigger line clear check for the toggled triangle
        toggled_triangle_set = {tri}
        lines_cleared, tris_cleared, cleared_coords = self.gs.grid.clear_lines(
            newly_occupied_triangles=toggled_triangle_set  # Pass the toggled tri
        )

        if lines_cleared > 0:
            # Update score and visual timers if lines were cleared
            self.gs.triangles_cleared_this_episode += tris_cleared
            score_increase = (lines_cleared**2) * 10 + tris_cleared
            self.gs.game_score += score_increase
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.6
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                tris_cleared,
                score_increase,
            )
            print(f"[Debug] Cleared {lines_cleared} lines ({tris_cleared} tris).")


File: environment\game_logic.py
# File: environment/game_logic.py
import random
from typing import TYPE_CHECKING, List, Tuple, Optional, Set

from .shape import Shape
from .triangle import Triangle

if TYPE_CHECKING:
    from .game_state import GameState


class GameLogic:
    """Handles core game mechanics like placing shapes, clearing lines, and game over conditions."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def step(self, action_index: int) -> Tuple[Optional[float], bool]:
        """
        Performs one game step based on the action index.
        Returns (reward, is_game_over). Reward is currently always None.
        """
        shape_slot_index, target_row, target_col = self.decode_action(action_index)

        # Validate shape index and shape existence
        if not (0 <= shape_slot_index < len(self.gs.shapes)):
            self.gs._last_action_valid = False
            self.gs.game_over = True
            self.gs.game_over_flash_time = 0.5
            return None, True
        shape_to_place = self.gs.shapes[shape_slot_index]

        if shape_to_place is None or not self.gs.grid.can_place(
            shape_to_place, target_row, target_col
        ):
            self.gs._last_action_valid = False
            self.gs.game_over = True
            self.gs.game_over_flash_time = 0.5  # Visual effect timer
            return None, True

        # Place the shape and get the set of newly occupied triangles
        newly_occupied = self.gs.grid.place(shape_to_place, target_row, target_col)
        self.gs.pieces_placed_this_episode += 1
        self.gs.shapes[shape_slot_index] = None  # Remove shape from slot

        # Clear lines using the optimized method
        lines_cleared, tris_cleared, cleared_coords = self.gs.grid.clear_lines(
            newly_occupied_triangles=newly_occupied
        )

        if lines_cleared > 0:
            self.gs.triangles_cleared_this_episode += tris_cleared
            # Update score (simple scoring for now)
            score_increase = (lines_cleared**2) * 10 + tris_cleared
            self.gs.game_score += score_increase
            # Set visual effect timers
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.6
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                tris_cleared,
                score_increase,
            )

        # Refill shape slots if needed (now checks if all are empty)
        self._refill_shape_slots()

        # Check game over condition (if any slot cannot place its shape)
        # This check needs to happen *after* potential refill
        if self._check_game_over():
            self.gs.game_over = True
            self.gs.game_over_flash_time = 0.5  # Visual effect timer
            return None, True

        self.gs._last_action_valid = True
        return None, False  # Reward is None, game not over

    def _refill_shape_slots(self):
        """Refills shape slots with new random shapes ONLY if all slots are empty."""
        # Check if all slots are currently empty (None)
        if all(s is None for s in self.gs.shapes):
            num_slots = self.gs.env_config.NUM_SHAPE_SLOTS
            self.gs.shapes = [Shape() for _ in range(num_slots)]

    def _check_game_over(self) -> bool:
        """Checks if the game is over because no available shape can be placed."""
        # If all slots are empty, the game is definitely not over yet (wait for refill)
        if all(s is None for s in self.gs.shapes):
            return False

        for shape in self.gs.shapes:
            if shape is not None:
                # Check if this shape has at least one valid placement anywhere
                for r in range(self.gs.grid.rows):
                    for c in range(self.gs.grid.cols):
                        if self.gs.grid.can_place(shape, r, c):
                            return False  # Found a valid move, game not over
        # If we get here, it means there was at least one shape, but none could be placed
        return True

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        valid_action_indices = []
        for shape_slot_index, shape in enumerate(self.gs.shapes):
            if shape is None:
                continue
            for r in range(self.gs.grid.rows):
                for c in range(self.gs.grid.cols):
                    if self.gs.grid.can_place(shape, r, c):
                        action_index = self.encode_action(shape_slot_index, r, c)
                        valid_action_indices.append(action_index)
        return valid_action_indices

    def encode_action(self, shape_slot_index: int, row: int, col: int) -> int:
        """Encodes (shape_slot, row, col) into a single action index."""
        return (
            shape_slot_index * (self.gs.grid.rows * self.gs.grid.cols)
            + row * self.gs.grid.cols
            + col
        )

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        grid_size = self.gs.grid.rows * self.gs.grid.cols
        shape_slot_index = action_index // grid_size
        remainder = action_index % grid_size
        row = remainder // self.gs.grid.cols
        col = remainder % self.gs.grid.cols
        return shape_slot_index, row, col


File: environment\game_state.py
import time
import numpy as np
from typing import List, Optional, Tuple, Dict
import copy  # Keep copy for grid deepcopy if needed

from config import EnvConfig, VisConfig
from .grid import Grid
from .shape import Shape
from .game_logic import GameLogic
from .game_state_features import GameStateFeatures
from .game_demo_logic import GameDemoLogic

StateType = Dict[str, np.ndarray]


class GameState:
    """
    Represents the state of a single game instance.
    Delegates logic to helper classes: GameLogic, GameStateFeatures, GameDemoLogic.
    Visual effect timers are managed but DO NOT block core logic execution.
    _update_timers() should only be called externally for UI/Demo rendering.
    """

    def __init__(self):
        self.env_config = EnvConfig()
        self.vis_config = VisConfig()

        self.grid = Grid(self.env_config)
        # Initialize shapes list with correct number of None slots
        self.shapes: List[Optional[Shape]] = [None] * self.env_config.NUM_SHAPE_SLOTS
        self.game_score: int = 0
        self.triangles_cleared_this_episode: int = 0
        self.pieces_placed_this_episode: int = 0

        # Timers for VISUAL effects only
        self.blink_time: float = 0.0
        self._last_timer_update_time: float = time.monotonic()
        self.freeze_time: float = 0.0
        self.line_clear_flash_time: float = 0.0
        self.line_clear_highlight_time: float = 0.0
        self.game_over_flash_time: float = 0.0
        self.cleared_triangles_coords: List[Tuple[int, int]] = []
        self.last_line_clear_info: Optional[Tuple[int, int, float]] = None

        self.game_over: bool = False
        self._last_action_valid: bool = True

        # Demo state
        self.demo_selected_shape_idx: int = -1  # Start with no selection
        self.demo_dragged_shape_idx: Optional[int] = None
        self.demo_snapped_position: Optional[Tuple[int, int]] = None

        # Helper classes
        self.logic = GameLogic(self)
        self.features = GameStateFeatures(self)
        self.demo_logic = GameDemoLogic(self)

        self.reset()

    def reset(self) -> StateType:
        """Resets the game to its initial state."""
        self.grid = Grid(self.env_config)  # Recreate grid
        # Ensure shapes list has the correct number of None slots initially
        self.shapes = [None] * self.env_config.NUM_SHAPE_SLOTS
        self.game_score = 0
        self.triangles_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0

        # Reset visual timers
        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.line_clear_flash_time = 0.0
        self.line_clear_highlight_time = 0.0
        self.game_over_flash_time = 0.0
        self.cleared_triangles_coords = []
        self.last_line_clear_info = None

        self.game_over = False
        self._last_action_valid = True
        self._last_timer_update_time = time.monotonic()

        self.demo_selected_shape_idx = -1
        self.demo_dragged_shape_idx = None
        self.demo_snapped_position = None

        # Generate the initial batch of shapes
        self.logic._refill_shape_slots()

        return self.get_state()

    def step(self, action_index: int) -> Tuple[Optional[StateType], bool]:
        """
        Performs one game step based on the action index using GameLogic.
        Returns (None, is_game_over). State should be fetched via get_state().
        """
        _, done = self.logic.step(action_index)
        return None, done

    def get_state(self) -> StateType:
        """Returns the current game state as a dictionary of numpy arrays."""
        return self.features.get_state()

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        return self.logic.valid_actions()

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        return self.logic.decode_action(action_index)

    def is_over(self) -> bool:
        return self.game_over

    # --- Visual State Check Methods (Used by UI/Demo) ---
    def is_frozen(self) -> bool:
        return self.freeze_time > 0

    def is_line_clearing(self) -> bool:
        return self.line_clear_flash_time > 0

    def is_highlighting_cleared(self) -> bool:
        return self.line_clear_highlight_time > 0

    def is_game_over_flashing(self) -> bool:
        return self.game_over_flash_time > 0

    def is_blinking(self) -> bool:
        return self.blink_time > 0

    def get_cleared_triangle_coords(self) -> List[Tuple[int, int]]:
        return self.cleared_triangles_coords

    def get_shapes(self) -> List[Optional[Shape]]:
        return self.shapes

    def get_outcome(self) -> float:
        """Determines the outcome of the game. Returns 0 for now."""
        return 0.0  # Simple outcome for now

    def _update_timers(self):
        """Updates timers for visual effects based on elapsed time."""
        now = time.monotonic()
        delta_time = now - self._last_timer_update_time
        self._last_timer_update_time = now
        delta_time = max(0.0, delta_time)

        self.freeze_time = max(0, self.freeze_time - delta_time)
        self.blink_time = max(0, self.blink_time - delta_time)
        self.line_clear_flash_time = max(0, self.line_clear_flash_time - delta_time)
        self.line_clear_highlight_time = max(
            0, self.line_clear_highlight_time - delta_time
        )
        self.game_over_flash_time = max(0, self.game_over_flash_time - delta_time)

        if self.line_clear_highlight_time <= 0 and self.cleared_triangles_coords:
            self.cleared_triangles_coords = []

    # --- Demo Mode Methods (Delegated) ---
    def select_shape_for_drag(self, shape_index: int):
        self.demo_logic.select_shape_for_drag(shape_index)

    def deselect_dragged_shape(self):
        self.demo_logic.deselect_dragged_shape()

    def update_snapped_position(self, grid_pos: Optional[Tuple[int, int]]):
        self.demo_logic.update_snapped_position(grid_pos)

    def place_dragged_shape(self) -> bool:
        return self.demo_logic.place_dragged_shape()

    def get_dragged_shape_info(
        self,
    ) -> Tuple[Optional[Shape], Optional[Tuple[int, int]]]:
        return self.demo_logic.get_dragged_shape_info()

    def toggle_triangle_debug(self, row: int, col: int):
        self.demo_logic.toggle_triangle_debug(row, col)


File: environment\game_state_features.py
import numpy as np
from typing import TYPE_CHECKING, Dict, List
import copy

if TYPE_CHECKING:
    from .game_state import GameState

StateType = Dict[str, np.ndarray]


class GameStateFeatures:
    """Handles calculation of state features and potential outcomes."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    # Removed calculate_potential (PBRS logic)

    def _calculate_potential_placement_outcomes(self) -> Dict[str, float]:
        """Calculates potential outcomes (tris cleared, holes, height, bumpiness) for valid moves."""
        valid_actions = self.gs.logic.valid_actions()  # Use logic helper
        if not valid_actions:
            return {
                "max_tris_cleared": 0.0,
                "min_holes": 0.0,
                "min_height": float(self.gs.grid.get_max_height()),
                "min_bump": float(self.gs.grid.get_bumpiness()),
            }

        max_triangles_cleared = 0
        min_new_holes = float("inf")
        min_resulting_height = float("inf")
        min_resulting_bumpiness = float("inf")
        initial_holes = self.gs.grid.count_holes()

        for action_index in valid_actions:
            shape_slot_index, target_row, target_col = self.gs.logic.decode_action(
                action_index
            )
            shape_to_place = self.gs.shapes[shape_slot_index]
            if shape_to_place is None:
                continue

            # Use grid's deepcopy method for simulation
            temp_grid = self.gs.grid.deepcopy_grid()
            # Need to simulate placement and clearing on the copy
            newly_occupied_sim = set()
            for dr, dc, _ in shape_to_place.triangles:
                nr, nc = target_row + dr, target_col + dc
                if temp_grid.valid(nr, nc):
                    tri = temp_grid.triangles[nr][nc]
                    if not tri.is_death and not tri.is_occupied:
                        tri.is_occupied = True
                        temp_grid._occupied_np[nr, nc] = True  # Update numpy array too
                        newly_occupied_sim.add(tri)

            _, triangles_cleared, _ = temp_grid.clear_lines(
                newly_occupied_triangles=newly_occupied_sim
            )
            holes_after = temp_grid.count_holes()
            height_after = temp_grid.get_max_height()
            bumpiness_after = temp_grid.get_bumpiness()
            new_holes_created = max(0, holes_after - initial_holes)

            max_triangles_cleared = max(max_triangles_cleared, triangles_cleared)
            min_new_holes = min(min_new_holes, new_holes_created)
            min_resulting_height = min(min_resulting_height, height_after)
            min_resulting_bumpiness = min(min_resulting_bumpiness, bumpiness_after)
            del temp_grid  # Clean up copy

        if min_new_holes == float("inf"):
            min_new_holes = 0.0
        if min_resulting_height == float("inf"):
            min_resulting_height = float(self.gs.grid.get_max_height())
        if min_resulting_bumpiness == float("inf"):
            min_resulting_bumpiness = float(self.gs.grid.get_bumpiness())

        return {
            "max_tris_cleared": float(max_triangles_cleared),
            "min_holes": float(min_new_holes),
            "min_height": float(min_resulting_height),
            "min_bump": float(min_resulting_bumpiness),
        }

    def get_state(self) -> StateType:
        """Returns the current game state as a dictionary of numpy arrays."""
        grid_state = self.gs.grid.get_feature_matrix()
        death_mask_state = self.gs.grid.get_death_data()  # Get death mask

        shape_features_per = self.gs.env_config.SHAPE_FEATURES_PER_SHAPE
        num_shapes_expected = self.gs.env_config.NUM_SHAPE_SLOTS
        shape_feature_matrix = np.zeros(
            (num_shapes_expected, shape_features_per), dtype=np.float32
        )
        max_tris_norm = 6.0
        max_h_norm = float(self.gs.grid.rows)
        max_w_norm = float(self.gs.grid.cols)
        for i in range(num_shapes_expected):
            s = self.gs.shapes[i] if i < len(self.gs.shapes) else None
            if s:
                tri_list = s.triangles
                n_tris = len(tri_list)
                ups = sum(1 for (_, _, is_up) in tri_list if is_up)
                downs = n_tris - ups
                min_r, min_c, max_r, max_c = s.bbox()
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                shape_feature_matrix[i, 0] = np.clip(
                    float(n_tris) / max_tris_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 1] = np.clip(
                    float(ups) / max_tris_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 2] = np.clip(
                    float(downs) / max_tris_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 3] = np.clip(
                    float(height) / max_h_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 4] = np.clip(
                    float(width) / max_w_norm, 0.0, 1.0
                )

        shape_availability_dim = self.gs.env_config.SHAPE_AVAILABILITY_DIM
        shape_availability_vector = np.zeros(shape_availability_dim, dtype=np.float32)
        for i in range(min(num_shapes_expected, shape_availability_dim)):
            if i < len(self.gs.shapes) and self.gs.shapes[i] is not None:
                shape_availability_vector[i] = 1.0

        explicit_features_dim = self.gs.env_config.EXPLICIT_FEATURES_DIM
        explicit_features_vector = np.zeros(explicit_features_dim, dtype=np.float32)
        num_holes = self.gs.grid.count_holes()
        col_heights = self.gs.grid.get_column_heights()
        avg_height = np.mean(col_heights) if col_heights else 0
        max_height = max(col_heights) if col_heights else 0
        bumpiness = self.gs.grid.get_bumpiness()
        max_possible_holes = self.gs.env_config.ROWS * self.gs.env_config.COLS
        max_possible_bumpiness = self.gs.env_config.ROWS * (self.gs.env_config.COLS - 1)
        explicit_features_vector[0] = np.clip(
            num_holes / max(1, max_possible_holes), 0.0, 1.0
        )
        explicit_features_vector[1] = np.clip(
            avg_height / self.gs.env_config.ROWS, 0.0, 1.0
        )
        explicit_features_vector[2] = np.clip(
            max_height / self.gs.env_config.ROWS, 0.0, 1.0
        )
        explicit_features_vector[3] = np.clip(
            bumpiness / max(1, max_possible_bumpiness), 0.0, 1.0
        )
        explicit_features_vector[4] = np.clip(
            self.gs.triangles_cleared_this_episode / 500.0, 0.0, 1.0
        )
        explicit_features_vector[5] = np.clip(
            self.gs.pieces_placed_this_episode / 500.0, 0.0, 1.0
        )

        if self.gs.env_config.CALCULATE_POTENTIAL_OUTCOMES_IN_STATE:
            potential_outcomes = self._calculate_potential_placement_outcomes()
            max_possible_tris_cleared = (
                self.gs.env_config.ROWS * self.gs.env_config.COLS
            )
            max_possible_new_holes = max_possible_holes
            explicit_features_vector[6] = np.clip(
                potential_outcomes["max_tris_cleared"]
                / max(1, max_possible_tris_cleared),
                0.0,
                1.0,
            )
            explicit_features_vector[7] = np.clip(
                potential_outcomes["min_holes"] / max(1, max_possible_new_holes),
                0.0,
                1.0,
            )
            explicit_features_vector[8] = np.clip(
                potential_outcomes["min_height"] / self.gs.env_config.ROWS, 0.0, 1.0
            )
            explicit_features_vector[9] = np.clip(
                potential_outcomes["min_bump"] / max(1, max_possible_bumpiness),
                0.0,
                1.0,
            )
        else:
            explicit_features_vector[6:10] = 0.0

        state_dict: StateType = {
            "grid": grid_state.astype(np.float32),
            "shapes": shape_feature_matrix.reshape(-1).astype(np.float32),
            "shape_availability": shape_availability_vector.astype(np.float32),
            "explicit_features": explicit_features_vector.astype(np.float32),
            "death_mask": death_mask_state.astype(np.bool_),  # Add death mask
        }
        return state_dict


File: environment\grid.py
import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Deque
from collections import deque  # Import deque for BFS
import numba  # Import Numba
import logging  # Import logging

from config import EnvConfig
from .triangle import Triangle
from .shape import Shape

logger = logging.getLogger(__name__)  # Add logger


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig):
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.triangles: List[List[Triangle]] = self._create(env_config)
        self._link_neighbors()

        # --- Internal NumPy arrays for Numba ---
        # Initialize these arrays based on the created grid
        self._occupied_np = np.array(
            [[tri.is_occupied for tri in row] for row in self.triangles], dtype=np.bool_
        )
        self._death_np = np.array(
            [[tri.is_death for tri in row] for row in self.triangles], dtype=np.bool_
        )
        # --- End Internal NumPy arrays ---

        # Store potential lines as frozensets for easy hashing/set operations
        self.potential_lines: Set[frozenset[Triangle]] = set()
        # Index mapping Triangle object to the set of lines it belongs to
        self._triangle_to_lines_map: Dict[Triangle, Set[frozenset[Triangle]]] = {}
        self._initialize_lines_and_index()  # This needs to run after _create and _link_neighbors

    def _create(self, env_config: EnvConfig) -> List[List[Triangle]]:
        """
        Initializes the grid with playable and death cells.
        The playable area defined by cols_per_row is further reduced by
        making the first and last cell of that range into death cells.
        """
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]
        if len(cols_per_row) != self.rows:
            raise ValueError("cols_per_row length mismatch")
        if max(cols_per_row) > self.cols:
            raise ValueError("cols_per_row exceeds EnvConfig.COLS")

        grid: List[List[Triangle]] = []
        for r in range(self.rows):
            row_tris: List[Triangle] = []
            intended_playable_width = cols_per_row[r]
            total_padding = self.cols - intended_playable_width
            pad_l = total_padding // 2
            pad_r = self.cols - (total_padding - pad_l)
            playable_start_col = pad_l + 1
            playable_end_col = pad_r - 1

            for c in range(self.cols):
                is_playable = (
                    intended_playable_width > 2
                    and playable_start_col <= c < playable_end_col
                )
                is_death = not is_playable
                is_up = (r + c) % 2 == 0
                row_tris.append(Triangle(r, c, is_up=is_up, is_death=is_death))
            grid.append(row_tris)
        return grid

    def _link_neighbors(self) -> None:
        """Sets neighbor references for each triangle."""
        for r in range(self.rows):
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if self.valid(r, c - 1):
                    tri.neighbor_left = self.triangles[r][c - 1]
                if self.valid(r, c + 1):
                    tri.neighbor_right = self.triangles[r][c + 1]
                nr, nc = (r + 1, c) if tri.is_up else (r - 1, c)
                if self.valid(nr, nc):
                    tri.neighbor_vert = self.triangles[nr][nc]

    def _get_line_neighbors(self, tri: Triangle, direction: str) -> List[Triangle]:
        """Helper to get relevant neighbors for line tracing in a specific direction."""
        neighbors = []
        if direction == "horizontal":
            if tri.neighbor_left:
                neighbors.append(tri.neighbor_left)
            if tri.neighbor_right:
                neighbors.append(tri.neighbor_right)
        elif direction == "diag1":
            if tri.is_up:
                if tri.neighbor_left and not tri.neighbor_left.is_up:
                    neighbors.append(tri.neighbor_left)
                if tri.neighbor_vert and not tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
            else:
                if tri.neighbor_right and tri.neighbor_right.is_up:
                    neighbors.append(tri.neighbor_right)
                if tri.neighbor_vert and tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
        elif direction == "diag2":
            if tri.is_up:
                if tri.neighbor_right and not tri.neighbor_right.is_up:
                    neighbors.append(tri.neighbor_right)
                if tri.neighbor_vert and not tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
            else:
                if tri.neighbor_left and tri.neighbor_left.is_up:
                    neighbors.append(tri.neighbor_left)
                if tri.neighbor_vert and tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
        return [n for n in neighbors if not n.is_death]

    def _initialize_lines_and_index(self) -> None:
        """
        Identifies all sets of playable triangles forming potential lines
        by tracing connections along horizontal and diagonal axes using BFS.
        Populates self.potential_lines and self._triangle_to_lines_map.
        """
        self.potential_lines = set()
        self._triangle_to_lines_map = {}
        visited_in_direction: Dict[str, Set[Triangle]] = {
            "horizontal": set(),
            "diag1": set(),
            "diag2": set(),
        }
        min_line_length = 3

        for r in range(self.rows):
            for c in range(self.cols):
                start_node = self.triangles[r][c]
                if start_node.is_death:
                    continue

                for direction in ["horizontal", "diag1", "diag2"]:
                    if start_node not in visited_in_direction[direction]:
                        current_line: Set[Triangle] = set()
                        queue: Deque[Triangle] = deque([start_node])
                        visited_this_bfs: Set[Triangle] = {start_node}

                        while queue:
                            tri = queue.popleft()
                            if not tri.is_death:
                                current_line.add(tri)
                            visited_in_direction[direction].add(tri)

                            neighbors = self._get_line_neighbors(tri, direction)
                            for neighbor in neighbors:
                                if neighbor not in visited_this_bfs:
                                    visited_this_bfs.add(neighbor)
                                    queue.append(neighbor)

                        if len(current_line) >= min_line_length:
                            line_frozenset = frozenset(current_line)
                            self.potential_lines.add(line_frozenset)
                            for tri_in_line in current_line:
                                if tri_in_line not in self._triangle_to_lines_map:
                                    self._triangle_to_lines_map[tri_in_line] = set()
                                self._triangle_to_lines_map[tri_in_line].add(
                                    line_frozenset
                                )

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(self, shape: Shape, r: int, c: int) -> bool:
        """Checks if a shape can be placed at the target location."""
        for dr, dc, is_up_shape in shape.triangles:
            nr, nc = r + dr, c + dc
            if not self.valid(nr, nc):
                return False
            # Use pre-computed numpy arrays for faster checks
            if (
                self._death_np[nr, nc]
                or self._occupied_np[nr, nc]
                or (self.triangles[nr][nc].is_up != is_up_shape)
            ):
                return False
        return True

    def place(self, shape: Shape, r: int, c: int) -> Set[Triangle]:
        """
        Places a shape onto the grid (assumes can_place was checked).
        Updates internal numpy arrays and returns the set of occupied Triangles.
        """
        newly_occupied: Set[Triangle] = set()
        for dr, dc, _ in shape.triangles:
            nr, nc = r + dr, c + dc
            # Check validity again just in case, though can_place should precede
            if self.valid(nr, nc):
                tri = self.triangles[nr][nc]
                # Check using numpy arrays first for speed
                if not self._death_np[nr, nc] and not self._occupied_np[nr, nc]:
                    # Update Triangle object
                    tri.is_occupied = True
                    tri.color = shape.color
                    # Update internal numpy array
                    self._occupied_np[nr, nc] = True
                    newly_occupied.add(tri)
        return newly_occupied

    def clear_lines(
        self, newly_occupied_triangles: Optional[Set[Triangle]] = None
    ) -> Tuple[int, int, List[Tuple[int, int]]]:
        """
        Checks and clears completed lines using the optimized index approach.
        Updates internal numpy arrays.
        Returns lines cleared count, total triangles cleared count, and their coordinates.
        """
        lines_to_check: Set[frozenset[Triangle]] = set()
        if newly_occupied_triangles:
            for tri in newly_occupied_triangles:
                if tri in self._triangle_to_lines_map:
                    lines_to_check.update(self._triangle_to_lines_map[tri])
        else:
            lines_to_check = self.potential_lines

        cleared_tris_total: Set[Triangle] = set()
        lines_cleared_count = 0

        for line_set in lines_to_check:
            if not line_set:
                continue
            # Check occupancy using the numpy array for speed
            if all(self._occupied_np[tri.row, tri.col] for tri in line_set):
                if not line_set.issubset(cleared_tris_total):
                    cleared_tris_total.update(line_set)
                    lines_cleared_count += 1

        tris_cleared_count = 0
        coords: List[Tuple[int, int]] = []
        if not cleared_tris_total:
            return 0, 0, []

        for tri in cleared_tris_total:
            # Check using numpy array first
            if (
                not self._death_np[tri.row, tri.col]
                and self._occupied_np[tri.row, tri.col]
            ):
                tris_cleared_count += 1
                # Update Triangle object
                tri.is_occupied = False
                tri.color = None
                # Update internal numpy array
                self._occupied_np[tri.row, tri.col] = False
                coords.append((tri.row, tri.col))

        return lines_cleared_count, tris_cleared_count, coords

    @staticmethod
    @numba.njit(cache=True)
    def _numba_get_column_heights(
        rows: int, cols: int, is_occupied: np.ndarray, is_death: np.ndarray
    ) -> np.ndarray:
        """Numba-accelerated calculation of column heights."""
        heights = np.zeros(cols, dtype=np.int32)
        for c in range(cols):
            col_max_r = -1
            for r in range(rows):
                if not is_death[r, c] and is_occupied[r, c]:
                    col_max_r = max(col_max_r, r)
            heights[c] = col_max_r + 1
        return heights

    def get_column_heights(self) -> List[int]:
        """Calculates the height of occupied cells in each column using Numba and internal arrays."""
        # Use the internal numpy arrays directly
        heights_np = self._numba_get_column_heights(
            self.rows, self.cols, self._occupied_np, self._death_np
        )
        return heights_np.tolist()

    def get_max_height(self) -> int:
        """Returns the maximum height across all columns."""
        heights = self.get_column_heights()
        return max(heights) if heights else 0

    @staticmethod
    @numba.njit(cache=True)
    def _numba_get_bumpiness(heights: np.ndarray) -> int:
        """Numba-accelerated calculation of bumpiness."""
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def get_bumpiness(self) -> int:
        """Calculates the total absolute difference between adjacent column heights using Numba."""
        heights_np = np.array(self.get_column_heights(), dtype=np.int32)
        return self._numba_get_bumpiness(heights_np)

    @staticmethod
    @numba.njit(cache=True)
    def _numba_count_holes(
        rows: int,
        cols: int,
        heights: np.ndarray,
        is_occupied: np.ndarray,
        is_death: np.ndarray,
    ) -> int:
        """Numba-accelerated calculation of holes."""
        holes = 0
        for c in range(cols):
            height = heights[c]
            if height > 0:
                for r in range(height):  # Iterate up to height-1
                    if not is_death[r, c] and not is_occupied[r, c]:
                        holes += 1
        return holes

    def count_holes(self) -> int:
        """Counts empty, non-death cells below the highest occupied cell in the same column using Numba."""
        heights_np = np.array(self.get_column_heights(), dtype=np.int32)
        # Use the internal numpy arrays directly
        return self._numba_count_holes(
            self.rows, self.cols, heights_np, self._occupied_np, self._death_np
        )

    def get_feature_matrix(self) -> np.ndarray:
        """Returns the grid state as a 2-channel numpy array (Occupancy, Orientation)."""
        # Use internal _occupied_np for the first channel
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        grid_state[0, :, :] = self._occupied_np.astype(np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if not self._death_np[r, c]:
                    grid_state[1, r, c] = 1.0 if self.triangles[r][c].is_up else -1.0
        return grid_state

    def get_color_data(self) -> List[List[Optional[Tuple[int, int, int]]]]:
        """Returns a 2D list of colors for occupied cells."""
        color_data = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                # Check internal numpy array first
                if self._occupied_np[r, c] and not self._death_np[r, c]:
                    color_data[r][c] = self.triangles[r][c].color
        return color_data

    def get_death_data(self) -> np.ndarray:
        """Returns a boolean numpy array indicating death cells (uses internal array)."""
        return self._death_np.copy()  # Return a copy to prevent external modification

    def deepcopy_grid(self) -> "Grid":
        """Creates a deep copy of the grid, including Triangle objects and numpy arrays."""
        new_grid = Grid.__new__(Grid)  # Create instance without calling __init__
        new_grid.rows = self.rows
        new_grid.cols = self.cols

        # Deep copy the list of lists of Triangles
        new_grid.triangles = [[tri.copy() for tri in row] for row in self.triangles]

        # Re-link neighbors within the new grid
        new_grid._link_neighbors()

        # Copy the numpy arrays
        new_grid._occupied_np = self._occupied_np.copy()
        new_grid._death_np = self._death_np.copy()

        # Rebuild the lines index based on the *new* triangle objects
        new_grid._initialize_lines_and_index()

        return new_grid


File: environment\shape.py
import random
from typing import List, Tuple

from config.constants import GOOGLE_COLORS


class Shape:
    """Represents a polyomino-like shape made of triangles."""

    def __init__(self) -> None:
        # List of (relative_row, relative_col, is_up) tuples defining the shape
        self.triangles: List[Tuple[int, int, bool]] = []
        # GOOGLE_COLORS is now imported from constants
        self.color: Tuple[int, int, int] = random.choice(GOOGLE_COLORS)
        self._generate()  # Generate the shape structure

    def _generate(self) -> None:
        """Generates a random shape by adding adjacent triangles."""
        num_triangles_in_shape = random.randint(1, 5)
        first_triangle_is_up = random.choice([True, False])
        # Add the root triangle at relative coordinates (0,0)
        self.triangles.append((0, 0, first_triangle_is_up))

        # Add remaining triangles adjacent to existing ones
        for _ in range(num_triangles_in_shape - 1):
            # Find valid neighbors of the *last added* triangle
            if not self.triangles:
                break  # Should not happen
            last_rel_row, last_rel_col, last_is_up = self.triangles[-1]
            valid_neighbors = self._find_valid_neighbors(
                last_rel_row, last_rel_col, last_is_up
            )
            if valid_neighbors:
                self.triangles.append(random.choice(valid_neighbors))
            # else: Could break early if no valid neighbors found, shape < n

    def _find_valid_neighbors(
        self, r: int, c: int, is_up: bool
    ) -> List[Tuple[int, int, bool]]:
        """Finds potential neighbor triangles that are not already part of the shape."""
        potential_neighbors: List[Tuple[int, int, bool]]
        if is_up:  # Neighbors of an UP triangle are DOWN triangles
            potential_neighbors = [
                (r, c - 1, False),
                (r, c + 1, False),
                (r + 1, c, False),
            ]
        else:  # Neighbors of a DOWN triangle are UP triangles
            potential_neighbors = [(r, c - 1, True), (r, c + 1, True), (r - 1, c, True)]
        # Return only neighbors that are not already in self.triangles
        valid_neighbors = [n for n in potential_neighbors if n not in self.triangles]
        return valid_neighbors

    def bbox(self) -> Tuple[int, int, int, int]:
        """Calculates the bounding box (min_r, min_c, max_r, max_c) of the shape."""
        if not self.triangles:
            return (0, 0, 0, 0)
        rows = [t[0] for t in self.triangles]
        cols = [t[1] for t in self.triangles]
        return (min(rows), min(cols), max(rows), max(cols))

    def copy(self) -> "Shape":
        """Creates a shallow copy of the Shape object."""
        new_shape = Shape.__new__(
            Shape
        )  # Create a new instance without calling __init__
        new_shape.triangles = list(
            self.triangles
        )  # Copy the list (tuples inside are immutable)
        new_shape.color = self.color  # Copy the color tuple reference
        return new_shape


File: environment\triangle.py
from typing import Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .grid import Grid


class Triangle:
    """Represents a single triangular cell on the grid."""

    def __init__(self, row: int, col: int, is_up: bool, is_death: bool = False):
        self.row = row
        self.col = col
        self.is_up = is_up  # True if pointing up, False if pointing down
        self.is_death = is_death  # True if part of the unplayable border
        self.is_occupied = is_death  # Occupied if it's a death cell initially
        self.color: Optional[Tuple[int, int, int]] = (
            None  # Color if occupied by a shape
        )
        # Neighbors based on shared edges - these will be linked by the Grid
        self.neighbor_left: Optional["Triangle"] = None
        self.neighbor_right: Optional["Triangle"] = None
        self.neighbor_vert: Optional["Triangle"] = None

    def get_points(
        self, ox: int, oy: int, cw: int, ch: int
    ) -> List[Tuple[float, float]]:
        """Calculates the vertex points for drawing the triangle."""
        x = ox + self.col * (
            cw * 0.75
        )  # Horizontal position based on column and overlap
        y = oy + self.row * ch  # Vertical position based on row
        if self.is_up:
            # Points for an upward-pointing triangle
            return [(x, y + ch), (x + cw, y + ch), (x + cw / 2, y)]
        else:
            # Points for a downward-pointing triangle
            return [(x, y), (x + cw, y), (x + cw / 2, y + ch)]

    def get_line_neighbors(
        self,
    ) -> Tuple[Optional["Triangle"], Optional["Triangle"], Optional["Triangle"]]:
        """Returns neighbors relevant for line checking (left, right, vertical)."""
        return self.neighbor_left, self.neighbor_right, self.neighbor_vert

    def copy(self) -> "Triangle":
        """Creates a copy of the Triangle object's state, excluding neighbors."""
        new_tri = Triangle.__new__(Triangle)  # Create instance without calling __init__
        new_tri.row = self.row
        new_tri.col = self.col
        new_tri.is_up = self.is_up
        new_tri.is_death = self.is_death
        new_tri.is_occupied = self.is_occupied
        new_tri.color = self.color  # Copy color reference (tuple is immutable)
        # Neighbors are intentionally set to None, they will be re-linked by Grid.deepcopy_grid
        new_tri.neighbor_left = None
        new_tri.neighbor_right = None
        new_tri.neighbor_vert = None
        return new_tri


File: environment\__init__.py


File: mcts\config.py
class MCTSConfig:
    """Configuration parameters for the Monte Carlo Tree Search."""

    # Exploration constant (c_puct in PUCT formula)
    # Balances exploitation (Q value) and exploration (prior P and visit counts N)
    # Higher values encourage exploring less-visited actions with high priors.
    PUCT_C: float = 1.5

    # Number of MCTS simulations to run for each move decision.
    # More simulations generally lead to stronger play but take more time.
    NUM_SIMULATIONS: int = 100

    # Temperature parameter for action selection during self-play.
    # Controls the randomness of move selection based on visit counts.
    # Higher temperature -> more exploration (sample proportionally to N^(1/temp))
    # Lower temperature -> more exploitation (closer to choosing the most visited action)
    # Often starts high (e.g., 1.0) and anneals to a small value (e.g., 0.1 or 0) during the game.
    TEMPERATURE_INITIAL: float = 1.0
    TEMPERATURE_FINAL: float = 0.01
    TEMPERATURE_ANNEAL_STEPS: int = (
        30  # Number of game steps over which to anneal temperature
    )

    # Dirichlet noise parameters for exploration at the root node during self-play.
    # Adds noise to the prior probabilities from the network to encourage exploration,
    # especially early in training.
    # Alpha determines the shape of the distribution, Epsilon the weight of the noise.
    DIRICHLET_ALPHA: float = 0.3
    DIRICHLET_EPSILON: float = 0.25

    # Maximum depth for the MCTS search tree (optional, can prevent excessive depth)
    MAX_SEARCH_DEPTH: int = 100


File: mcts\node.py
import math
import numpy as np
from typing import Dict, Optional, TYPE_CHECKING, Any, List

# Assuming GameState is hashable or identifiable
from environment.game_state import GameState
from utils.types import ActionType
from config import MCTSConfig  # Import from config package


class MCTSNode:
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(
        self,
        game_state: GameState,
        parent: Optional["MCTSNode"] = None,
        action_taken: Optional[ActionType] = None,
        prior: float = 0.0,
        config: Optional[MCTSConfig] = None,  # Pass config for PUCT_C
    ):
        self.game_state = game_state
        self.parent = parent
        self.action_taken = action_taken

        self.children: Dict[ActionType, "MCTSNode"] = {}
        self.is_expanded: bool = False
        self.is_terminal: bool = game_state.is_over()

        self.visit_count: int = 0
        self.total_action_value: float = 0.0
        self.mean_action_value: float = 0.0
        self.prior: float = prior

        self._config = config if config else MCTSConfig()  # Use default if None

    def get_ucb_score(self) -> float:
        """Calculates the PUCT score for this node (from the perspective of its parent)."""
        if self.parent is None:
            return self.mean_action_value  # Root node score

        exploration_bonus = (
            self._config.PUCT_C
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        q_value = self.mean_action_value
        return q_value + exploration_bonus

    def select_best_child(self) -> "MCTSNode":
        """Selects the child with the highest UCB score."""
        if not self.children:
            raise ValueError("Cannot select best child from a node with no children.")
        # Simple way to handle potential ties: add small random noise or just pick first max
        best_score = -float("inf")
        best_children = []
        for child in self.children.values():
            score = child.get_ucb_score()
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        if not best_children:
            raise RuntimeError("Could not select a best child node.")
        # Randomly pick among the best children in case of ties
        return np.random.choice(best_children)

    def backpropagate(self, value: float):
        """Updates the visit count and action value of this node and its ancestors."""
        node: Optional[MCTSNode] = self
        while node is not None:
            node.visit_count += 1
            node.total_action_value += value
            node.mean_action_value = node.total_action_value / node.visit_count
            node = node.parent


File: mcts\search.py
# File: mcts/search.py
import numpy as np
import time
import copy
from typing import Dict, Optional, Tuple, Callable, Any, List
import logging
import torch
import threading
import multiprocessing as mp
import ray  # Added Ray

from environment.game_state import GameState
from utils.types import ActionType, StateType
from .node import MCTSNode
from config import MCTSConfig, EnvConfig

# Removed NetworkPredictor type hint, using ActorHandle now
# from agent.alphazero_net import AgentPredictor # Import Actor type if needed for hinting

logger = logging.getLogger(__name__)


class MCTS:
    """Monte Carlo Tree Search implementation based on AlphaZero principles with batching."""

    MCTS_NN_BATCH_SIZE = 8  # Default internal batch size for NN predictions

    def __init__(
        self,
        # network_predictor: NetworkPredictor, # Replaced with actor handle
        agent_predictor: ray.actor.ActorHandle,  # Actor handle for AgentPredictor
        config: Optional[MCTSConfig] = None,
        env_config: Optional[EnvConfig] = None,
        batch_size: int = MCTS_NN_BATCH_SIZE,
        stop_event: Optional[mp.Event] = None,
    ):
        # self.network_predictor = network_predictor # Removed
        self.agent_predictor = agent_predictor  # Store actor handle
        self.config = config if config else MCTSConfig()
        self.env_config = env_config if env_config else EnvConfig()
        self.batch_size = max(1, batch_size)
        self.stop_event = stop_event
        self.log_prefix = "[MCTS]"
        logger.info(
            f"{self.log_prefix} Initialized with AgentPredictor actor. NN batch size: {self.batch_size}"
        )

    def _select_leaf(self, root_node: MCTSNode) -> Tuple[MCTSNode, int]:
        """Selects a leaf node using PUCT criteria, checking stop_event."""
        node = root_node
        depth = 0
        while node.is_expanded and not node.is_terminal:
            if self.stop_event and self.stop_event.is_set():
                raise InterruptedError("MCTS selection interrupted.")

            if depth >= self.config.MAX_SEARCH_DEPTH:
                break
            if not node.children:
                break

            try:
                node = node.select_best_child()
                depth += 1
            except ValueError:
                logger.warning(
                    f"{self.log_prefix} Node claims expanded but has no selectable children."
                )
                break

        return node, depth

    def _expand_and_backpropagate_batch(
        self, nodes_to_expand: List[MCTSNode]
    ) -> Tuple[float, int]:
        """
        Expands a batch of leaf nodes using batched NN prediction via Ray actor and backpropagates results.
        Checks stop_event more frequently.
        Returns (total_nn_prediction_time, nodes_created_count).
        """
        if not nodes_to_expand:
            return 0.0, 0

        if self.stop_event and self.stop_event.is_set():
            raise InterruptedError(
                "MCTS expansion interrupted by stop event (before NN)."
            )

        batch_states = [node.game_state.get_state() for node in nodes_to_expand]
        total_nn_prediction_time = 0.0
        nodes_created_count = 0
        policy_probs_list = []
        predicted_values = []

        try:
            start_pred_time = time.monotonic()
            # --- Call the AgentPredictor actor ---
            prediction_ref = self.agent_predictor.predict_batch.remote(batch_states)
            policy_probs_list, predicted_values = ray.get(prediction_ref)
            # --- End Actor Call ---
            total_nn_prediction_time = time.monotonic() - start_pred_time
            logger.debug(
                f"{self.log_prefix} Batched NN Prediction ({len(batch_states)} states) via Actor took {total_nn_prediction_time:.4f}s."
            )
        except ray.exceptions.RayActorError as rae:
            logger.error(
                f"{self.log_prefix} RayActorError during prediction: {rae}",
                exc_info=True,
            )
            # Backpropagate 0 if NN fails, mark as expanded to avoid re-selection
            for node in nodes_to_expand:
                if self.stop_event and self.stop_event.is_set():
                    break
                if not node.is_expanded:
                    node.is_expanded = True
                    node.backpropagate(0.0)
            return total_nn_prediction_time, 0
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Error during batched network prediction: {e}",
                exc_info=True,
            )
            # Backpropagate 0 if NN fails, mark as expanded to avoid re-selection
            for node in nodes_to_expand:
                if self.stop_event and self.stop_event.is_set():
                    break
                if not node.is_expanded:
                    node.is_expanded = True
                    node.backpropagate(0.0)
            return total_nn_prediction_time, 0

        if self.stop_event and self.stop_event.is_set():
            raise InterruptedError(
                "MCTS expansion interrupted by stop event (after NN)."
            )

        for i, node in enumerate(nodes_to_expand):
            if self.stop_event and self.stop_event.is_set():
                logger.info(
                    f"{self.log_prefix} Stop event detected during batch expansion processing."
                )
                break

            if node.is_expanded or node.is_terminal:
                if node.visit_count == 0:
                    value_to_prop = (
                        predicted_values[i] if i < len(predicted_values) else 0.0
                    )
                    node.backpropagate(value_to_prop)
                continue

            policy_probs_dict = (
                policy_probs_list[i] if i < len(policy_probs_list) else {}
            )
            predicted_value = predicted_values[i] if i < len(predicted_values) else 0.0
            children_created_count_node = 0

            valid_actions = node.game_state.valid_actions()
            if not valid_actions:
                node.is_expanded = True
                node.is_terminal = True
                node.backpropagate(predicted_value)
                continue

            parent_state = node.game_state
            start_expand_time = time.monotonic()
            for action in valid_actions:
                if self.stop_event and self.stop_event.is_set():
                    logger.info(
                        f"{self.log_prefix} Stop event detected during child creation loop for node {id(node)}."
                    )
                    if not node.is_expanded:
                        node.is_expanded = True
                        node.backpropagate(predicted_value)
                    break

                try:
                    child_state = GameState()
                    child_state.grid = parent_state.grid.deepcopy_grid()
                    child_state.shapes = [
                        s.copy() if s else None for s in parent_state.shapes
                    ]
                    child_state.game_score = parent_state.game_score
                    child_state.triangles_cleared_this_episode = (
                        parent_state.triangles_cleared_this_episode
                    )
                    child_state.pieces_placed_this_episode = (
                        parent_state.pieces_placed_this_episode
                    )
                    _, done = child_state.step(action)

                    prior_prob = policy_probs_dict.get(action, 0.0)
                    child_node = MCTSNode(
                        game_state=child_state,
                        parent=node,
                        action_taken=action,
                        prior=prior_prob,
                        config=self.config,
                    )
                    if done:
                        child_node.is_terminal = True
                    node.children[action] = child_node
                    children_created_count_node += 1
                except Exception as child_creation_err:
                    logger.error(
                        f"{self.log_prefix} Error creating child for action {action}: {child_creation_err}",
                        exc_info=True,
                    )
                    continue

            if self.stop_event and self.stop_event.is_set():
                logger.info(
                    f"{self.log_prefix} Stop event detected after child creation loop for node {id(node)}."
                )
                if not node.is_expanded:
                    node.is_expanded = True
                    node.backpropagate(predicted_value)
                break

            expand_duration = time.monotonic() - start_expand_time
            node.is_expanded = True
            nodes_created_count += children_created_count_node

            if self.stop_event and self.stop_event.is_set():
                logger.info(
                    f"{self.log_prefix} Stop event detected before backpropagation for node {id(node)}."
                )
                break
            node.backpropagate(predicted_value)

        return total_nn_prediction_time, nodes_created_count

    # run_simulations, _add_dirichlet_noise, get_policy_target, choose_action remain largely the same
    # but they now rely on _expand_and_backpropagate_batch which uses the Ray actor.

    def run_simulations(
        self, root_state: GameState, num_simulations: int
    ) -> Tuple[MCTSNode, Dict[str, Any]]:
        """
        Runs the MCTS process for a given number of simulations using batching.
        Returns the root node and a dictionary of simulation statistics.
        Handles InterruptedError from stop_event checks.
        """
        if self.stop_event and self.stop_event.is_set():
            logger.warning(
                f"{self.log_prefix} Stop event set before starting simulations."
            )
            return MCTSNode(game_state=root_state, config=self.config), {
                "simulations_run": 0,
                "mcts_total_duration": 0.0,
                "total_nn_prediction_time": 0.0,
                "nodes_created": 1,
                "avg_leaf_depth": 0.0,
                "root_visits": 0,
            }

        root_node = MCTSNode(game_state=root_state, config=self.config)
        sim_start_time = time.monotonic()
        total_nn_prediction_time = 0.0
        nodes_created_this_run = 1
        total_leaf_depth = 0
        simulations_run_attempted = 0
        simulations_completed_full = 0

        if root_node.is_terminal:
            logger.warning(
                f"{self.log_prefix} Root node is terminal. No simulations run."
            )
            return root_node, {
                "simulations_run": 0,
                "mcts_total_duration": 0.0,
                "total_nn_prediction_time": 0.0,
                "nodes_created": 1,
                "avg_leaf_depth": 0.0,
                "root_visits": 0,
            }

        try:
            if not root_node.is_expanded:
                if self.stop_event and self.stop_event.is_set():
                    raise InterruptedError("Stop event before initial root expansion.")
                initial_batch_time, initial_nodes_created = (
                    self._expand_and_backpropagate_batch([root_node])
                )
                total_nn_prediction_time += initial_batch_time
                nodes_created_this_run += initial_nodes_created
                simulations_completed_full += 1
                if root_node.is_expanded and not root_node.is_terminal:
                    self._add_dirichlet_noise(root_node)

            leaves_to_expand: List[MCTSNode] = []
            simulations_run_attempted = 1

            for sim_num in range(simulations_run_attempted, num_simulations):
                simulations_run_attempted += 1
                if self.stop_event and self.stop_event.is_set():
                    logger.info(
                        f"{self.log_prefix} Stop event detected before simulation {sim_num+1}. Stopping MCTS."
                    )
                    break

                sim_start_step = time.monotonic()
                leaf_node, depth = self._select_leaf(root_node)
                total_leaf_depth += depth

                if leaf_node.is_terminal:
                    value = leaf_node.game_state.get_outcome()
                    leaf_node.backpropagate(value)
                    sim_duration_step = time.monotonic() - sim_start_step
                    simulations_completed_full += 1
                    continue

                leaves_to_expand.append(leaf_node)

                if (
                    len(leaves_to_expand) >= self.batch_size
                    or sim_num == num_simulations - 1
                ):
                    if leaves_to_expand:
                        if self.stop_event and self.stop_event.is_set():
                            logger.info(
                                f"{self.log_prefix} Stop event detected before expanding batch."
                            )
                            break

                        batch_nn_time, batch_nodes_created = (
                            self._expand_and_backpropagate_batch(leaves_to_expand)
                        )
                        total_nn_prediction_time += batch_nn_time
                        nodes_created_this_run += batch_nodes_created
                        simulations_completed_full += len(leaves_to_expand)
                        leaves_to_expand = []

            if leaves_to_expand and not (self.stop_event and self.stop_event.is_set()):
                logger.info(
                    f"{self.log_prefix} Processing remaining {len(leaves_to_expand)} leaves after loop exit."
                )
                batch_nn_time, batch_nodes_created = (
                    self._expand_and_backpropagate_batch(leaves_to_expand)
                )
                total_nn_prediction_time += batch_nn_time
                nodes_created_this_run += batch_nodes_created
                simulations_completed_full += len(leaves_to_expand)

        except InterruptedError as e:
            logger.warning(f"{self.log_prefix} MCTS run interrupted gracefully: {e}")
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Error during MCTS run_simulations: {e}",
                exc_info=True,
            )

        sim_duration_total = time.monotonic() - sim_start_time
        effective_sims = max(1, root_node.visit_count)
        avg_leaf_depth = (
            total_leaf_depth / effective_sims if effective_sims > 0 else 0.0
        )

        logger.info(
            f"{self.log_prefix} Finished {root_node.visit_count} effective simulations ({simulations_run_attempted} attempted) in {sim_duration_total:.4f}s. "
            f"Nodes created: {nodes_created_this_run}, "
            f"Total NN time: {total_nn_prediction_time:.4f}s, Avg Depth: {avg_leaf_depth:.1f}"
        )

        mcts_stats = {
            "simulations_run": root_node.visit_count,
            "mcts_total_duration": sim_duration_total,
            "total_nn_prediction_time": total_nn_prediction_time,
            "nodes_created": nodes_created_this_run,
            "avg_leaf_depth": avg_leaf_depth,
            "root_visits": root_node.visit_count,
        }
        return root_node, mcts_stats

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Adds Dirichlet noise to the prior probabilities of the root node's children."""
        if not node.children or self.config.DIRICHLET_ALPHA <= 0:
            return
        child_actions = [a for a in node.children.keys() if a in node.children]
        if not child_actions:
            return

        num_children = len(child_actions)
        noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA] * num_children)
        eps = self.config.DIRICHLET_EPSILON
        for i, action in enumerate(child_actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]
        logger.debug(f"{self.log_prefix} Applied Dirichlet noise to root node priors.")

    def get_policy_target(
        self, root_node: MCTSNode, temperature: float
    ) -> Dict[ActionType, float]:
        """Calculates the improved policy distribution based on visit counts."""
        if not root_node.children:
            return {}

        existing_children = {a: c for a, c in root_node.children.items() if c}
        if not existing_children:
            return {}

        total_visits = sum(child.visit_count for child in existing_children.values())
        if total_visits == 0:
            num_children = len(existing_children)
            logger.warning(
                f"{self.log_prefix} Root node has 0 total visits across children. Returning uniform policy."
            )
            return (
                {a: 1.0 / num_children for a in existing_children}
                if num_children > 0
                else {}
            )

        policy_target: Dict[ActionType, float] = {}
        if temperature == 0:
            best_action = max(
                existing_children, key=lambda a: existing_children[a].visit_count
            )
            for action in existing_children:
                policy_target[action] = 1.0 if action == best_action else 0.0
        else:
            total_power, powered_counts = 0.0, {}
            max_power_val = np.finfo(np.float64).max / (len(existing_children) + 1)

            for action, child in existing_children.items():
                visit_count = max(0, child.visit_count)
                try:
                    powered_count = np.power(
                        np.float64(visit_count), 1.0 / temperature, dtype=np.float64
                    )
                    if np.isinf(powered_count) or np.isnan(powered_count):
                        logger.warning(
                            f"{self.log_prefix} Infinite/NaN powered count for action {action}. Clamping."
                        )
                        powered_count = max_power_val
                except (OverflowError, ValueError):
                    logger.warning(
                        f"{self.log_prefix} Power calc overflow/error for action {action}. Setting large value."
                    )
                    powered_count = max_power_val

                powered_counts[action] = powered_count
                if not np.isinf(powered_count) and not np.isnan(powered_count):
                    total_power += powered_count

            if total_power <= 1e-9 or np.isinf(total_power) or np.isnan(total_power):
                num_valid_children = len(powered_counts)
                if num_valid_children > 0:
                    prob = 1.0 / num_valid_children
                    for action in existing_children:
                        policy_target[action] = prob
                    logger.warning(
                        f"{self.log_prefix} Total power invalid ({total_power:.2e}), assigned uniform prob {prob:.3f}."
                    )
                else:
                    policy_target = {}
            else:
                for action, powered_count in powered_counts.items():
                    policy_target[action] = float(powered_count / total_power)

        full_policy = np.zeros(self.env_config.ACTION_DIM, dtype=np.float32)
        policy_sum_check = 0.0
        for action, prob in policy_target.items():
            if 0 <= action < self.env_config.ACTION_DIM:
                full_policy[action] = prob
                policy_sum_check += prob
            else:
                logger.warning(
                    f"{self.log_prefix} MCTS produced invalid action index {action} in policy target."
                )

        if not np.isclose(policy_sum_check, 1.0, atol=1e-4):
            logger.warning(
                f"{self.log_prefix} Policy target sum is {policy_sum_check:.4f} before final conversion. Renormalizing."
            )
            current_sum = np.sum(full_policy)
            if current_sum > 1e-6:
                full_policy /= current_sum
            else:
                valid_actions = root_node.game_state.valid_actions()
                num_valid = len(valid_actions)
                if num_valid > 0:
                    prob = 1.0 / num_valid
                    full_policy.fill(0.0)
                    for action in valid_actions:
                        if 0 <= action < self.env_config.ACTION_DIM:
                            full_policy[action] = prob

        return {i: float(prob) for i, prob in enumerate(full_policy)}

    def choose_action(self, root_node: MCTSNode, temperature: float) -> ActionType:
        """Chooses an action based on MCTS visit counts and temperature."""
        policy_dict = self.get_policy_target(root_node, temperature)
        valid_actions_list = root_node.game_state.valid_actions()
        valid_actions_set = set(valid_actions_list)

        if not policy_dict or not valid_actions_list:
            if valid_actions_list:
                logger.warning(
                    f"{self.log_prefix} Policy dict empty/invalid, choosing random valid action."
                )
                return np.random.choice(valid_actions_list)
            else:
                logger.error(
                    f"{self.log_prefix} MCTS failed: no policy and no valid actions."
                )
                return -1

        filtered_actions, filtered_probs = [], []
        for action, prob in policy_dict.items():
            if action in valid_actions_set and prob > 1e-7:
                filtered_actions.append(action)
                filtered_probs.append(prob)

        if not filtered_actions:
            logger.warning(
                f"{self.log_prefix} MCTS policy effectively zero for all valid actions. Choosing uniformly among valid."
            )
            return np.random.choice(valid_actions_list)

        actions = np.array(filtered_actions)
        probabilities = np.array(filtered_probs, dtype=np.float64)

        prob_sum = np.sum(probabilities)
        if prob_sum <= 1e-7:
            logger.warning(
                f"{self.log_prefix} Filtered policy sum near zero ({prob_sum}). Choosing uniformly among filtered."
            )
            return np.random.choice(actions)

        probabilities /= prob_sum

        try:
            chosen_action = np.random.choice(actions, p=probabilities)
            return int(chosen_action)
        except ValueError as e:
            logger.error(
                f"{self.log_prefix} Error during np.random.choice: {e}. Prob sum: {np.sum(probabilities)}. Choosing uniformly."
            )
            return np.random.choice(actions)


File: mcts\__init__.py
from .config import MCTSConfig
from .node import MCTSNode
from .search import MCTS

__all__ = ["MCTSConfig", "MCTSNode", "MCTS"]


File: stats\aggregator.py
import time
from typing import Deque, Dict, Any, Optional, List, TYPE_CHECKING
import threading
import logging
import numpy as np
import ray

from config import StatsConfig
from .aggregator_storage import AggregatorStorage
from .aggregator_logic import AggregatorLogic

if TYPE_CHECKING:
    from environment.game_state import GameState

logger = logging.getLogger(__name__)


# --- Ray Actor Version ---
@ray.remote
class StatsAggregatorActor:
    """
    Ray Actor version of StatsAggregator. Handles aggregation and storage
    of training statistics using deques within the actor process.
    """

    def __init__(
        self,
        avg_windows: List[int] = StatsConfig.STATS_AVG_WINDOW,
        plot_window: int = StatsConfig.PLOT_DATA_WINDOW,
    ):
        if not avg_windows or not all(
            isinstance(w, int) and w > 0 for w in avg_windows
        ):
            print("Warning: Invalid avg_windows list. Using default [100].")
            self.avg_windows = [100]
        else:
            self.avg_windows = sorted(list(set(avg_windows)))
        self.plot_window = max(1, plot_window)
        self.summary_avg_window = self.avg_windows[0]
        self.storage = AggregatorStorage(self.plot_window)
        self.logic = AggregatorLogic(self.storage)
        logger.info(
            f"[StatsAggregatorActor] Initialized. Avg Windows: {self.avg_windows}, Plot Window: {self.plot_window}"
        )

    def record_episode(
        self,
        episode_outcome: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Records episode stats and checks for new bests."""
        current_step = (
            global_step if global_step is not None else self.storage.current_global_step
        )
        update_info = self.logic.update_episode_stats(
            episode_outcome,
            episode_length,
            episode_num,
            current_step,
            game_score,
            triangles_cleared,
            game_state_for_best,
        )
        return update_info

    def record_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Records step stats."""
        g_step = step_data.get("global_step")
        if g_step is not None and g_step > self.storage.current_global_step:
            self.storage.current_global_step = g_step
        elif g_step is None:
            g_step = self.storage.current_global_step

        update_info = self.logic.update_step_stats(
            step_data,
            g_step,
            mcts_sim_time=step_data.get("mcts_sim_time"),
            mcts_nn_time=step_data.get("mcts_nn_time"),
            mcts_nodes_explored=step_data.get("mcts_nodes_explored"),
            mcts_avg_depth=step_data.get("mcts_avg_depth"),
        )
        return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        """Calculates and returns the summary dictionary."""
        if current_global_step is None:
            current_global_step = self.storage.current_global_step
        summary = self.logic.calculate_summary(
            current_global_step, self.summary_avg_window
        )
        summary["device"] = "Actor"
        try:
            from config.core import TrainConfig

            summary["min_buffer_size"] = TrainConfig.MIN_BUFFER_SIZE_TO_TRAIN
        except ImportError:
            summary["min_buffer_size"] = 0
        return summary

    def get_plot_data(self) -> Dict[str, List]:  # Return List for serialization
        """Returns copies of data deques as lists for plotting."""
        plot_deques = self.storage.get_all_plot_deques()
        return {name: list(dq) for name, dq in plot_deques.items()}

    def get_best_game_state_data(self) -> Optional[Dict[str, Any]]:
        """Returns the serializable data needed to render the best game state found."""
        return self.storage.best_game_state_data

    def state_dict(self) -> Dict[str, Any]:
        """Returns the internal state for saving."""
        state = self.storage.state_dict()
        state["plot_window"] = self.plot_window
        state["avg_windows"] = self.avg_windows
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the internal state from a dictionary."""
        logger.info("[StatsAggregatorActor] Loading state...")
        self.plot_window = state_dict.get("plot_window", self.plot_window)
        self.avg_windows = state_dict.get("avg_windows", self.avg_windows)
        self.summary_avg_window = self.avg_windows[0] if self.avg_windows else 100
        self.storage.load_state_dict(state_dict, self.plot_window)
        logger.info("[StatsAggregatorActor] State loaded.")
        logger.info(f"  -> Loaded total_episodes: {self.storage.total_episodes}")
        logger.info(f"  -> Loaded best_game_score: {self.storage.best_game_score}")
        logger.info(
            f"  -> Loaded start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.storage.start_time))}"
        )
        logger.info(
            f"  -> Loaded current_global_step: {self.storage.current_global_step}"
        )
        if self.storage.best_game_state_data:
            logger.info(
                f"  -> Loaded best_game_state_data (Score: {self.storage.best_game_state_data.get('score', 'N/A')})"
            )
        else:
            logger.info("  -> No best_game_state_data found in loaded state.")

    def get_total_episodes(self) -> int:
        """Returns the total number of episodes recorded."""
        return self.storage.total_episodes

    def get_current_global_step(self) -> int:
        """Returns the current global step."""
        return self.storage.current_global_step

    def set_training_target_step(self, target_step: int):
        """Sets the training target step."""
        self.storage.training_target_step = target_step

    def get_training_target_step(self) -> int:
        """Returns the training target step."""
        return self.storage.training_target_step

    def health_check(self):
        """Basic health check method for Ray."""
        return "OK"


File: stats\aggregator_logic.py
# File: stats/aggregator_logic.py
import time
import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging

from .aggregator_storage import AggregatorStorage

if TYPE_CHECKING:
    from environment.game_state import GameState

logger = logging.getLogger(__name__)


class AggregatorLogic:
    """Handles the logic for updating stats and calculating summaries."""

    def __init__(self, storage: AggregatorStorage):
        self.storage = storage

    def update_episode_stats(
        self,
        episode_outcome: float,
        episode_length: int,
        episode_num: int,
        current_step: int,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional[
            Dict[str, np.ndarray]
        ] = None,  # Expect state dict
    ) -> Dict[str, Any]:
        """Updates deques and best values related to episode completion."""
        update_info = {"new_best_game": False}
        self.storage.total_episodes += 1
        self.storage.episode_outcomes.append(episode_outcome)
        self.storage.episode_lengths.append(episode_length)

        if game_score is not None:
            self.storage.game_scores.append(game_score)
            if game_score > self.storage.best_game_score:
                self.storage.previous_best_game_score = self.storage.best_game_score
                self.storage.best_game_score = game_score
                self.storage.best_game_score_step = current_step
                self.storage.best_game_score_history.append(game_score)
                update_info["new_best_game"] = True
                # Store best game state data if provided (now expects state dict)
                if game_state_for_best and isinstance(game_state_for_best, dict):
                    self.storage.best_game_state_data = {
                        "score": game_score,
                        "step": current_step,
                        "game_state_dict": game_state_for_best,  # Store the state dict directly
                    }
                elif game_state_for_best:  # Log warning if wrong type passed
                    logger.warning(
                        f"Received game_state_for_best of type {type(game_state_for_best)}, expected dict. Skipping storage."
                    )
                    self.storage.best_game_state_data = (
                        None  # Clear previous best if new best has invalid state
                    )
                else:
                    self.storage.best_game_state_data = (
                        None  # Clear if no state provided for new best
                    )

            elif self.storage.best_game_score > -float("inf"):
                self.storage.best_game_score_history.append(
                    self.storage.best_game_score
                )

        if triangles_cleared is not None:
            self.storage.episode_triangles_cleared.append(triangles_cleared)
            self.storage.total_triangles_cleared += triangles_cleared

        # Update best outcome (less relevant now)
        if episode_outcome > self.storage.best_outcome:
            self.storage.previous_best_outcome = self.storage.best_outcome
            self.storage.best_outcome = episode_outcome
            self.storage.best_outcome_step = current_step
            update_info["new_best_outcome"] = True

        return update_info

    def update_step_stats(
        self,
        step_data: Dict[str, Any],
        g_step: int,
        mcts_sim_time: Optional[float] = None,
        mcts_nn_time: Optional[float] = None,
        mcts_nodes_explored: Optional[int] = None,
        mcts_avg_depth: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Updates deques and best values related to individual steps (training or MCTS)."""
        update_info = {
            "new_best_value_loss": False,
            "new_best_policy_loss": False,
            "new_best_mcts_sim_time": False,
        }

        # --- Log Global Step Update ---
        # Use INFO level as requested
        logger.info(
            f"[AggLogic] update_step_stats called. Incoming g_step: {g_step}, Stored step before update: {self.storage.current_global_step}"
        )
        if g_step > self.storage.current_global_step:
            self.storage.current_global_step = g_step
            logger.info(
                f"[AggLogic] Updated storage.current_global_step to: {self.storage.current_global_step}"
            )
        # --- End Log ---

        # Training Stats
        policy_loss = step_data.get("policy_loss")
        value_loss = step_data.get("value_loss")
        lr = step_data.get("lr")

        if policy_loss is not None and np.isfinite(policy_loss):
            self.storage.policy_losses.append(policy_loss)
            if policy_loss < self.storage.best_policy_loss:
                self.storage.previous_best_policy_loss = self.storage.best_policy_loss
                self.storage.best_policy_loss = policy_loss
                self.storage.best_policy_loss_step = g_step
                update_info["new_best_policy_loss"] = True

        if value_loss is not None and np.isfinite(value_loss):
            self.storage.value_losses.append(value_loss)
            if value_loss < self.storage.best_value_loss:
                self.storage.previous_best_value_loss = self.storage.best_value_loss
                self.storage.best_value_loss = value_loss
                self.storage.best_value_loss_step = g_step
                update_info["new_best_value_loss"] = True

        if lr is not None and np.isfinite(lr):
            self.storage.lr_values.append(lr)
            self.storage.current_lr = lr

        # MCTS Stats
        if mcts_sim_time is not None and np.isfinite(mcts_sim_time):
            self.storage.mcts_simulation_times.append(mcts_sim_time)
            if mcts_sim_time < self.storage.best_mcts_sim_time:
                self.storage.previous_best_mcts_sim_time = (
                    self.storage.best_mcts_sim_time
                )
                self.storage.best_mcts_sim_time = mcts_sim_time
                self.storage.best_mcts_sim_time_step = g_step
                update_info["new_best_mcts_sim_time"] = True

        if mcts_nn_time is not None and np.isfinite(mcts_nn_time):
            self.storage.mcts_nn_prediction_times.append(mcts_nn_time)
        if mcts_nodes_explored is not None and np.isfinite(mcts_nodes_explored):
            self.storage.mcts_nodes_explored.append(mcts_nodes_explored)
        if mcts_avg_depth is not None and np.isfinite(mcts_avg_depth):
            self.storage.mcts_avg_depths.append(mcts_avg_depth)

        # System Stats
        buffer_size = step_data.get("buffer_size")
        if buffer_size is not None:
            self.storage.buffer_sizes.append(buffer_size)
            self.storage.current_buffer_size = buffer_size

        self.storage.update_steps_per_second(g_step)

        # Intermediate Progress
        current_game = step_data.get("current_self_play_game_number")
        current_game_step = step_data.get("current_self_play_game_steps")
        training_steps = step_data.get("training_steps_performed")

        if current_game is not None:
            self.storage.current_self_play_game_number = current_game
        if current_game_step is not None:
            self.storage.current_self_play_game_steps = current_game_step
        if training_steps is not None:
            self.storage.training_steps_performed = training_steps

        return update_info

    def _calculate_average(self, deque, window_size):
        """Calculates the average of the last 'window_size' elements in a deque."""
        if not deque:
            return 0.0
        actual_window = min(window_size, len(deque))
        items = list(deque)[-actual_window:]
        return np.mean(items) if items else 0.0

    def calculate_summary(
        self, current_global_step: int, avg_window: int
    ) -> Dict[str, Any]:
        """Calculates the summary dictionary based on stored data."""
        summary = {}
        summary["global_step"] = current_global_step
        summary["total_episodes"] = self.storage.total_episodes
        summary["start_time"] = self.storage.start_time
        summary["runtime_seconds"] = time.time() - self.storage.start_time
        summary["buffer_size"] = self.storage.current_buffer_size
        summary["current_lr"] = self.storage.current_lr
        summary["summary_avg_window_size"] = avg_window

        summary["avg_episode_length"] = self._calculate_average(
            self.storage.episode_lengths, avg_window
        )
        summary["avg_game_score_window"] = self._calculate_average(
            self.storage.game_scores, avg_window
        )
        summary["avg_triangles_cleared"] = self._calculate_average(
            self.storage.episode_triangles_cleared, avg_window
        )
        summary["policy_loss"] = self._calculate_average(
            self.storage.policy_losses, avg_window
        )
        summary["value_loss"] = self._calculate_average(
            self.storage.value_losses, avg_window
        )
        summary["mcts_simulation_time_avg"] = self._calculate_average(
            self.storage.mcts_simulation_times, avg_window
        )
        summary["mcts_nn_prediction_time_avg"] = self._calculate_average(
            self.storage.mcts_nn_prediction_times, avg_window
        )
        summary["mcts_nodes_explored_avg"] = self._calculate_average(
            self.storage.mcts_nodes_explored, avg_window
        )
        summary["steps_per_second_avg"] = self._calculate_average(
            self.storage.steps_per_second, avg_window
        )

        summary["best_game_score"] = self.storage.best_game_score
        summary["best_game_score_step"] = self.storage.best_game_score_step
        summary["best_value_loss"] = self.storage.best_value_loss
        summary["best_value_loss_step"] = self.storage.best_value_loss_step
        summary["best_policy_loss"] = self.storage.best_policy_loss
        summary["best_policy_loss_step"] = self.storage.best_policy_loss_step
        summary["best_mcts_sim_time"] = self.storage.best_mcts_sim_time
        summary["best_mcts_sim_time_step"] = self.storage.best_mcts_sim_time_step

        summary["current_self_play_game_number"] = (
            self.storage.current_self_play_game_number
        )
        summary["current_self_play_game_steps"] = (
            self.storage.current_self_play_game_steps
        )
        summary["training_steps_performed"] = self.storage.training_steps_performed
        summary["training_target_step"] = self.storage.training_target_step

        return summary


File: stats\aggregator_storage.py
# File: stats/aggregator_storage.py
from collections import deque
from typing import Deque, Dict, Any, List, Optional
import time
import numpy as np
import logging
import pickle  # Needed for potential complex data in best_game_state_data

logger = logging.getLogger(__name__)


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator.
    Ensures best_game_state_data is stored as a serializable dictionary.
    """

    def __init__(self, plot_window: int):
        self.plot_window = plot_window
        # ... (Deque definitions remain the same) ...
        # Training Stats
        self.policy_losses: Deque[float] = deque(maxlen=plot_window)
        self.value_losses: Deque[float] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(maxlen=plot_window)
        # Game Stats
        self.episode_outcomes: Deque[float] = deque(maxlen=plot_window)  # -1, 0, 1
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_triangles_cleared: Deque[int] = deque(maxlen=plot_window)
        self.best_game_score_history: Deque[int] = deque(maxlen=plot_window)
        # MCTS Stats
        self.mcts_simulation_times: Deque[float] = deque(maxlen=plot_window)
        self.mcts_nn_prediction_times: Deque[float] = deque(maxlen=plot_window)
        self.mcts_nodes_explored: Deque[int] = deque(maxlen=plot_window)
        self.mcts_avg_depths: Deque[float] = deque(maxlen=plot_window)
        # System Stats
        self.buffer_sizes: Deque[int] = deque(maxlen=plot_window)
        self.steps_per_second: Deque[float] = deque(maxlen=plot_window)
        self._last_step_time: Optional[float] = None
        self._last_step_count: Optional[int] = None

        # --- Scalar State Variables ---
        # ... (remain the same) ...
        self.total_episodes: int = 0
        self.total_triangles_cleared: int = 0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0
        self.current_lr: float = 0.0
        self.start_time: float = time.time()
        self.training_target_step: int = 0

        # --- Intermediate Progress Tracking ---
        # ... (remain the same) ...
        self.current_self_play_game_number: int = 0
        self.current_self_play_game_steps: int = 0
        self.training_steps_performed: int = 0

        # --- Best Value Tracking ---
        # ... (remain the same) ...
        self.best_outcome: float = -float("inf")
        self.previous_best_outcome: float = -float("inf")
        self.best_outcome_step: int = 0
        self.best_game_score: float = -float("inf")
        self.previous_best_game_score: float = -float("inf")
        self.best_game_score_step: int = 0
        self.best_value_loss: float = float("inf")
        self.previous_best_value_loss: float = float("inf")
        self.best_value_loss_step: int = 0
        self.best_policy_loss: float = float("inf")
        self.previous_best_policy_loss: float = float("inf")
        self.best_policy_loss_step: int = 0
        self.best_mcts_sim_time: float = float("inf")
        self.previous_best_mcts_sim_time: float = float("inf")
        self.best_mcts_sim_time_step: int = 0

        # --- Best Game State Data ---
        # Now stores the already processed serializable dict
        self.best_game_state_data: Optional[Dict[str, Any]] = None

    # get_deque remains the same
    def get_deque(self, name: str) -> Deque:
        return getattr(self, name, deque(maxlen=self.plot_window))

    # get_all_plot_deques remains the same
    def get_all_plot_deques(self) -> Dict[str, Deque]:
        deque_names = [
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "best_game_score_history",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "mcts_nodes_explored",
            "mcts_avg_depths",
            "buffer_sizes",
            "steps_per_second",
        ]
        return {
            name: self.get_deque(name).copy()
            for name in deque_names
            if hasattr(self, name)
        }

    # update_steps_per_second remains the same
    def update_steps_per_second(self, global_step: int):
        current_time = time.time()
        if self._last_step_time is not None and self._last_step_count is not None:
            time_diff = current_time - self._last_step_time
            step_diff = global_step - self._last_step_count
            if time_diff > 1e-3 and step_diff > 0:
                sps = step_diff / time_diff
                self.steps_per_second.append(sps)
            elif step_diff <= 0 and time_diff > 1.0:
                self.steps_per_second.append(0.0)
        self._last_step_time = current_time
        self._last_step_count = global_step

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the storage for saving."""
        state = {}
        # ... (Deque serialization remains the same) ...
        deque_names = [
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "best_game_score_history",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "mcts_nodes_explored",
            "mcts_avg_depths",
            "buffer_sizes",
            "steps_per_second",
        ]
        for name in deque_names:
            if hasattr(self, name):
                deque_instance = getattr(self, name, None)
                if deque_instance is not None:
                    state[name] = list(deque_instance)

        # ... (Scalar serialization remains the same) ...
        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_buffer_size",
            "current_global_step",
            "current_lr",
            "start_time",
            "training_target_step",
            "current_self_play_game_number",
            "current_self_play_game_steps",
            "training_steps_performed",
            "_last_step_time",
            "_last_step_count",
        ]
        for key in scalar_keys:
            state[key] = getattr(self, key, None if key.startswith("_last") else 0)

        # ... (Best value serialization remains the same) ...
        best_value_keys = [
            "best_outcome",
            "previous_best_outcome",
            "best_outcome_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
            "best_policy_loss",
            "previous_best_policy_loss",
            "best_policy_loss_step",
            "best_mcts_sim_time",
            "previous_best_mcts_sim_time",
            "best_mcts_sim_time_step",
        ]
        for key in best_value_keys:
            default = (
                0
                if "step" in key
                else (
                    float("inf") if ("loss" in key or "time" in key) else -float("inf")
                )
            )
            state[key] = getattr(self, key, default)

        # Serialize best game state data directly (it should already be a dict)
        # We might need pickle for numpy arrays within the dict
        if self.best_game_state_data:
            try:
                # Ensure numpy arrays are handled by pickle if torch save fails
                state["best_game_state_data_pkl"] = pickle.dumps(
                    self.best_game_state_data
                )
            except Exception as e:
                logger.error(f"Could not pickle best_game_state_data: {e}")
                state["best_game_state_data_pkl"] = None
        else:
            state["best_game_state_data_pkl"] = None

        # Deprecated direct storage in torch checkpoint:
        # state["best_game_state_data"] = self.best_game_state_data

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        # ... (Deque loading remains the same) ...
        self.plot_window = plot_window
        deque_names = [
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "best_game_score_history",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "mcts_nodes_explored",
            "mcts_avg_depths",
            "buffer_sizes",
            "steps_per_second",
        ]
        for key in deque_names:
            data = state_dict.get(key)
            if isinstance(data, (list, tuple)):
                setattr(self, key, deque(data, maxlen=self.plot_window))
            else:
                setattr(self, key, deque(maxlen=self.plot_window))
                if data is not None:
                    logger.warning(
                        f"Invalid data type for deque '{key}' in loaded state: {type(data)}. Init empty."
                    )

        # ... (Scalar loading remains the same) ...
        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_buffer_size",
            "current_global_step",
            "current_lr",
            "start_time",
            "training_target_step",
            "current_self_play_game_number",
            "current_self_play_game_steps",
            "training_steps_performed",
            "_last_step_time",
            "_last_step_count",
        ]
        defaults = {
            "start_time": time.time(),
            "training_target_step": 0,
            "current_global_step": 0,
            "total_episodes": 0,
            "total_triangles_cleared": 0,
            "current_buffer_size": 0,
            "current_lr": 0.0,
            "current_self_play_game_number": 0,
            "current_self_play_game_steps": 0,
            "training_steps_performed": 0,
            "_last_step_time": None,
            "_last_step_count": None,
        }
        for key in scalar_keys:
            setattr(self, key, state_dict.get(key, defaults.get(key)))

        # ... (Best value loading remains the same) ...
        best_value_keys = [
            "best_outcome",
            "previous_best_outcome",
            "best_outcome_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
            "best_policy_loss",
            "previous_best_policy_loss",
            "best_policy_loss_step",
            "best_mcts_sim_time",
            "previous_best_mcts_sim_time",
            "best_mcts_sim_time_step",
        ]
        best_defaults = {
            "best_outcome": -float("inf"),
            "previous_best_outcome": -float("inf"),
            "best_outcome_step": 0,
            "best_game_score": -float("inf"),
            "previous_best_game_score": -float("inf"),
            "best_game_score_step": 0,
            "best_value_loss": float("inf"),
            "previous_best_value_loss": float("inf"),
            "best_value_loss_step": 0,
            "best_policy_loss": float("inf"),
            "previous_best_policy_loss": float("inf"),
            "best_policy_loss_step": 0,
            "best_mcts_sim_time": float("inf"),
            "previous_best_mcts_sim_time": float("inf"),
            "best_mcts_sim_time_step": 0,
        }
        for key in best_value_keys:
            setattr(self, key, state_dict.get(key, best_defaults.get(key)))

        # Deserialize best game state data using pickle
        best_game_data_pkl = state_dict.get("best_game_state_data_pkl")
        if best_game_data_pkl:
            try:
                self.best_game_state_data = pickle.loads(best_game_data_pkl)
                if not isinstance(self.best_game_state_data, dict):
                    logger.warning(
                        "Loaded best_game_state_data is not a dict, resetting."
                    )
                    self.best_game_state_data = None
                else:
                    # Basic validation
                    if (
                        "score" not in self.best_game_state_data
                        or "step" not in self.best_game_state_data
                        or "game_state_dict" not in self.best_game_state_data
                    ):
                        logger.warning(
                            "Loaded best_game_state_data dict missing keys, resetting."
                        )
                        self.best_game_state_data = None
            except Exception as e:
                logger.error(f"Error unpickling best_game_state_data: {e}")
                self.best_game_state_data = None
        else:
            # Fallback to old direct storage method if pkl missing
            loaded_best_data = state_dict.get("best_game_state_data")
            if isinstance(loaded_best_data, dict):
                self.best_game_state_data = loaded_best_data
                # Basic validation
                if (
                    "score" not in self.best_game_state_data
                    or "step" not in self.best_game_state_data
                    or "game_state_dict" not in self.best_game_state_data
                ):
                    logger.warning(
                        "Loaded legacy best_game_state_data dict missing keys, resetting."
                    )
                    self.best_game_state_data = None
            else:
                self.best_game_state_data = None

        # Ensure critical attributes exist after loading
        for attr, default_factory in [
            ("current_global_step", lambda: 0),
            ("best_game_score", lambda: -float("inf")),
            ("best_game_state_data", lambda: None),
            ("training_steps_performed", lambda: 0),
            ("current_self_play_game_number", lambda: 0),
            ("current_self_play_game_steps", lambda: 0),
            ("best_mcts_sim_time", lambda: float("inf")),
            ("steps_per_second", lambda: deque(maxlen=self.plot_window)),
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, default_factory())


File: stats\simple_stats_recorder.py
# File: stats/simple_stats_recorder.py
import time
from typing import Deque, Dict, Any, Optional, Union, List, TYPE_CHECKING
import numpy as np
import torch
import threading
import logging
import ray # Added Ray

# from .stats_recorder import StatsRecorderBase # Keep Base class import
from .aggregator import StatsAggregatorActor # Import Actor for type hint
from config import StatsConfig, TrainConfig
from utils.helpers import format_eta

if TYPE_CHECKING:
    from environment.game_state import GameState
    StatsAggregatorHandle = ray.actor.ActorHandle # Type hint for handle

# Import base class correctly
from .stats_recorder import StatsRecorderBase

logger = logging.getLogger(__name__)


class SimpleStatsRecorder(StatsRecorderBase):
    """Logs aggregated statistics fetched from StatsAggregatorActor to the console periodically."""

    def __init__(
        self,
        aggregator: "StatsAggregatorHandle", # Expect Actor Handle
        console_log_interval: int = StatsConfig.CONSOLE_LOG_FREQ,
        train_config: Optional[TrainConfig] = None,
    ):
        self.aggregator_handle = aggregator # Store actor handle
        self.console_log_interval = (
            max(1, console_log_interval) if console_log_interval > 0 else -1
        )
        self.train_config = train_config if train_config else TrainConfig()
        self.last_log_time: float = time.time()
        # Get summary window size from config, actor doesn't store it directly this way
        self.summary_avg_window = StatsConfig.STATS_AVG_WINDOW[0] if StatsConfig.STATS_AVG_WINDOW else 100
        self.updates_since_last_log = 0
        self._lock = threading.Lock() # Lock for updates_since_last_log counter
        logger.info(
            f"[SimpleStatsRecorder] Initialized. Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'} updates/episodes. Avg Window: {self.summary_avg_window}"
        )
        # Store last known best values locally to detect changes
        self._last_best_score = -float('inf')
        self._last_best_vloss = float('inf')
        self._last_best_ploss = float('inf')
        self._last_best_mcts_time = float('inf')


    def _log_new_best(
        self,
        metric_name: str,
        current_best: float,
        previous_best: float,
        step: int,
        is_loss: bool,
        is_time: bool = False,
    ):
        """Logs a new best value achieved."""
        # This logic remains the same, uses passed values
        improvement_made = False
        if is_loss:
            if np.isfinite(current_best) and current_best < previous_best: improvement_made = True
        else:
            if np.isfinite(current_best) and current_best > previous_best: improvement_made = True
        if not improvement_made: return

        if is_time:
            format_str = "{:.3f}s"
            prev_str = format_str.format(previous_best) if np.isfinite(previous_best) and previous_best != float("inf") else "N/A"
            current_str = format_str.format(current_best)
            prefix = "⏱️"
        elif is_loss:
            format_str = "{:.4f}"
            prev_str = format_str.format(previous_best) if np.isfinite(previous_best) and previous_best != float("inf") else "N/A"
            current_str = format_str.format(current_best)
            prefix = "📉"
        else:
            format_str = "{:.0f}"
            prev_str = format_str.format(previous_best) if np.isfinite(previous_best) and previous_best != -float("inf") else "N/A"
            current_str = format_str.format(current_best)
            prefix = "🎮"

        step_info = f"at Step ~{step/1e6:.1f}M" if step > 0 else "at Start"
        logger.info(f"--- {prefix} New Best {metric_name}: {current_str} {step_info} (Prev: {prev_str}) ---")

    def record_episode(
        self,
        episode_outcome: float, # These args are now less relevant as data comes from aggregator
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional["GameState"] = None, # This arg is also less relevant
    ):
        """Checks for new bests by fetching summary from aggregator."""
        # This method is now primarily a trigger based on episode completion
        # The actual recording happens via remote calls from workers
        # We just need to check if the interval requires logging a summary
        current_step = global_step # Use passed step if available
        if current_step is None:
             # Fetch step from aggregator if not passed (blocking)
             try:
                  step_ref = self.aggregator_handle.get_current_global_step.remote()
                  current_step = ray.get(step_ref)
             except Exception as e:
                  logger.error(f"Error fetching global step from aggregator: {e}")
                  current_step = 0 # Fallback

        self._check_and_log_summary(current_step)


    def record_step(self, step_data: Dict[str, Any]):
        """Checks for new bests by fetching summary from aggregator."""
        # This method is now primarily a trigger based on step completion
        # The actual recording happens via remote calls from workers
        g_step = step_data.get("global_step")
        if g_step is None:
             # Fetch step from aggregator if not passed (blocking)
             try:
                  step_ref = self.aggregator_handle.get_current_global_step.remote()
                  g_step = ray.get(step_ref)
             except Exception as e:
                  logger.error(f"Error fetching global step from aggregator: {e}")
                  g_step = 0 # Fallback

        self._check_and_log_summary(g_step)


    def _check_and_log_summary(self, global_step: int):
        """Checks if the logging interval is met and logs summary by fetching from actor."""
        log_now = False
        with self._lock:
            self.updates_since_last_log += 1
            if (
                self.console_log_interval > 0
                and self.updates_since_last_log >= self.console_log_interval
            ):
                log_now = True
                self.updates_since_last_log = 0
        if log_now:
            self.log_summary(global_step) # Fetch data and log

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Gets the summary dictionary from the aggregator actor (blocking)."""
        if not self.aggregator_handle: return {}
        try:
            summary_ref = self.aggregator_handle.get_summary.remote(current_global_step)
            summary = ray.get(summary_ref)
            return summary
        except Exception as e:
            logger.error(f"Error getting summary from StatsAggregatorActor: {e}")
            return {"error": str(e)}

    def get_plot_data(self) -> Dict[str, Deque]:
        """Gets the plot data deques from the aggregator actor (blocking)."""
        # Note: Returns lists, not deques, due to serialization
        if not self.aggregator_handle: return {}
        try:
            plot_data_ref = self.aggregator_handle.get_plot_data.remote()
            plot_data_list_dict = ray.get(plot_data_ref)
            # Convert lists back to deques locally if needed by UI plotter
            # For now, return the dict of lists as received
            return plot_data_list_dict
        except Exception as e:
            logger.error(f"Error getting plot data from StatsAggregatorActor: {e}")
            return {"error": str(e)}

    def log_summary(self, global_step: int):
        """Logs the current summary statistics fetched from the aggregator actor."""
        summary = self.get_summary(global_step)
        if not summary or "error" in summary:
             logger.error(f"Could not log summary, failed to fetch data: {summary.get('error', 'Unknown error')}")
             return

        # --- Check for New Bests ---
        # Compare fetched best values with locally stored last known bests
        new_best_score = summary.get("best_game_score", -float('inf'))
        new_best_vloss = summary.get("best_value_loss", float('inf'))
        new_best_ploss = summary.get("best_policy_loss", float('inf'))
        new_best_mcts_time = summary.get("best_mcts_sim_time", float('inf'))

        if new_best_score > self._last_best_score:
             self._log_new_best("Game Score", new_best_score, self._last_best_score, summary.get("best_game_score_step", 0), is_loss=False)
             self._last_best_score = new_best_score
        if new_best_vloss < self._last_best_vloss:
             self._log_new_best("V.Loss", new_best_vloss, self._last_best_vloss, summary.get("best_value_loss_step", 0), is_loss=True)
             self._last_best_vloss = new_best_vloss
        if new_best_ploss < self._last_best_ploss:
             self._log_new_best("P.Loss", new_best_ploss, self._last_best_ploss, summary.get("best_policy_loss_step", 0), is_loss=True)
             self._last_best_ploss = new_best_ploss
        if new_best_mcts_time < self._last_best_mcts_time:
             self._log_new_best("MCTS Sim Time", new_best_mcts_time, self._last_best_mcts_time, summary.get("best_mcts_sim_time_step", 0), is_loss=True, is_time=True)
             self._last_best_mcts_time = new_best_mcts_time
        # --- End New Best Check ---


        runtime_hrs = (time.time() - summary.get("start_time", time.time())) / 3600
        best_score_str = f"{new_best_score:.0f}" if new_best_score > -float("inf") else "N/A"
        avg_win = summary.get("summary_avg_window_size", self.summary_avg_window)
        buf_size = summary.get("buffer_size", 0)
        min_buf = summary.get("min_buffer_size", self.train_config.MIN_BUFFER_SIZE_TO_TRAIN)
        phase = "Buffering" if buf_size < min_buf and global_step == 0 else "Training"
        steps_sec = summary.get("steps_per_second_avg", 0.0)

        current_game = summary.get("current_self_play_game_number", 0)
        current_game_step = summary.get("current_self_play_game_steps", 0)
        game_prog_str = f"Game: {current_game}({current_game_step})" if current_game > 0 else ""

        log_items = [
            f"[{runtime_hrs:.1f}h|{phase}]",
            f"Step: {global_step/1e6:<6.2f}M ({steps_sec:.1f}/s)",
            f"Ep: {summary.get('total_episodes', 0):<7,}".replace(",", "_"),
            f"Buf: {buf_size:,}/{min_buf:,}".replace(",", "_"),
            f"Score(Avg{avg_win}): {summary.get('avg_game_score_window', 0.0):<6.0f} (Best: {best_score_str})",
        ]

        if global_step > 0 or phase == "Training":
            log_items.extend([
                f"VLoss(Avg{avg_win}): {summary.get('value_loss', 0.0):.4f}",
                f"PLoss(Avg{avg_win}): {summary.get('policy_loss', 0.0):.4f}",
                f"LR: {summary.get('current_lr', 0.0):.1e}",
            ])
        else: log_items.append("Loss: N/A")

        mcts_sim_time_avg = summary.get("mcts_simulation_time_avg", 0.0)
        mcts_nn_time_avg = summary.get("mcts_nn_prediction_time_avg", 0.0)
        mcts_nodes_avg = summary.get("mcts_nodes_explored_avg", 0.0)
        if mcts_sim_time_avg > 0 or mcts_nn_time_avg > 0 or mcts_nodes_avg > 0:
            mcts_str = f"MCTS(Avg{avg_win}): SimT={mcts_sim_time_avg*1000:.1f}ms | NNT={mcts_nn_time_avg*1000:.1f}ms | Nodes={mcts_nodes_avg:.0f}"
            log_items.append(mcts_str)

        if game_prog_str: log_items.append(game_prog_str)

        training_target_step = summary.get("training_target_step", 0)
        if training_target_step > 0 and steps_sec > 0:
            steps_remaining = training_target_step - global_step
            if steps_remaining > 0:
                eta_seconds = steps_remaining / steps_sec
                eta_str = format_eta(eta_seconds)
                log_items.append(f"ETA: {eta_str}")

        logger.info(" | ".join(log_items))
        self.last_log_time = time.time()

    # --- No-op methods for other recording types ---
    def record_histogram(self, tag: str, values: Union[np.ndarray, torch.Tensor, List[float]], global_step: int): pass
    def record_image(self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int): pass
    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]): pass
    def record_graph(self, model: torch.nn.Module, input_to_model: Optional[Any] = None): pass

    def close(self, is_cleanup: bool = False):
        # Ensure final summary is logged if interval logging is enabled
        if self.console_log_interval > 0 and self.updates_since_last_log > 0:
            logger.info("[SimpleStatsRecorder] Logging final summary before closing...")
            # Fetch final step count before logging
            final_step = 0
            if self.aggregator_handle:
                 try:
                      step_ref = self.aggregator_handle.get_current_global_step.remote()
                      final_step = ray.get(step_ref)
                 except Exception: pass # Ignore error on close
            self.log_summary(final_step)
        logger.info(f"[SimpleStatsRecorder] Closed (is_cleanup={is_cleanup}).")

File: stats\stats_recorder.py
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import (
    Deque,
    List,
    Dict,
    Any,
    Optional,
    Union,
    TYPE_CHECKING,
)
import numpy as np
import torch

if TYPE_CHECKING:
    from environment.game_state import GameState  # Import for type hinting


class StatsRecorderBase(ABC):
    """Base class for recording training statistics."""

    @abstractmethod
    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional["GameState"] = None,
    ):
        """Record stats for a completed episode."""
        pass

    @abstractmethod
    def record_step(self, step_data: Dict[str, Any]):
        """Record stats from a training or environment step."""
        pass

    @abstractmethod
    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Record a histogram of values."""
        pass

    @abstractmethod
    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Record an image."""
        pass

    @abstractmethod
    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Record hyperparameters and final/key metrics."""
        pass

    @abstractmethod
    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[Any] = None
    ):
        """Record the model graph."""
        pass

    @abstractmethod
    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Return a dictionary containing summary statistics."""
        pass

    @abstractmethod
    def get_plot_data(self) -> Dict[str, Deque]:
        """Return copies of data deques for plotting."""
        pass

    @abstractmethod
    def log_summary(self, global_step: int):
        """Trigger the logging action (e.g., print to console)."""
        pass

    @abstractmethod
    def close(self, is_cleanup: bool = False):
        """Perform any necessary cleanup."""
        pass


File: stats\__init__.py
from .stats_recorder import StatsRecorderBase

# from .aggregator import StatsAggregator # Original class name removed/renamed
from .aggregator import StatsAggregatorActor  # Import the Actor class
from .simple_stats_recorder import SimpleStatsRecorder


__all__ = [
    "StatsRecorderBase",
    # "StatsAggregator", # Remove old export
    "StatsAggregatorActor",  # Export the Actor class
    "SimpleStatsRecorder",
]


File: training\checkpoint_manager.py
import os
import torch
import torch.optim as optim
import traceback
import re
import time
from typing import Optional, Tuple, Any, Dict, TYPE_CHECKING
import pickle
import ray
import logging # Added logging

# StatsAggregator is now an Actor Handle
from agent.alphazero_net import AlphaZeroNet
from torch.optim.lr_scheduler import _LRScheduler

if TYPE_CHECKING:
    # from stats.aggregator import StatsAggregatorActor # Import Actor for type hint
    StatsAggregatorHandle = ray.actor.ActorHandle

logger = logging.getLogger(__name__) # Added logger

# --- Checkpoint Finding Logic (remains the same) ---
def find_latest_run_and_checkpoint(
    base_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Finds the latest run directory and the latest checkpoint within it."""
    latest_run_id, latest_run_mtime = None, 0
    if not os.path.isdir(base_dir): return None, None
    try:
        for item in os.listdir(base_dir):
            path = os.path.join(base_dir, item)
            if os.path.isdir(path) and item.startswith("run_"):
                try:
                    mtime = os.path.getmtime(path)
                    if mtime > latest_run_mtime: latest_run_mtime, latest_run_id = mtime, item
                except OSError: continue
    except OSError as e: print(f"[CheckpointFinder] Error listing {base_dir}: {e}"); return None, None
    if latest_run_id is None: print(f"[CheckpointFinder] No runs found in {base_dir}."); return None, None
    latest_run_dir = os.path.join(base_dir, latest_run_id)
    print(f"[CheckpointFinder] Latest run directory: {latest_run_dir}")
    latest_checkpoint = find_latest_checkpoint_in_dir(latest_run_dir)
    if latest_checkpoint: print(f"[CheckpointFinder] Found checkpoint: {os.path.basename(latest_checkpoint)}")
    else: print(f"[CheckpointFinder] No valid checkpoints found in {latest_run_dir}")
    return latest_run_id, latest_checkpoint

def find_latest_checkpoint_in_dir(ckpt_dir: str) -> Optional[str]:
    """Finds the latest checkpoint file in a specific directory."""
    if not os.path.isdir(ckpt_dir): return None
    checkpoints, final_ckpt = [], None
    step_pattern = re.compile(r"step_(\d+)_alphazero_nn\.pth")
    final_name = "FINAL_alphazero_nn.pth"
    try:
        for fname in os.listdir(ckpt_dir):
            fpath = os.path.join(ckpt_dir, fname)
            if not os.path.isfile(fpath): continue
            if fname == final_name: final_ckpt = fpath
            else:
                match = step_pattern.match(fname)
                if match: checkpoints.append((int(match.group(1)), fpath))
    except OSError as e: print(f"[CheckpointFinder] Error listing {ckpt_dir}: {e}"); return None
    if final_ckpt:
        try:
            final_mtime = os.path.getmtime(final_ckpt)
            if not any(os.path.getmtime(cp) > final_mtime for _, cp in checkpoints): return final_ckpt
        except OSError: pass
    if not checkpoints: return final_ckpt
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]

# --- Checkpoint Manager Class ---
class CheckpointManager:
    """Handles loading and saving of agent, optimizer, scheduler, and stats states (interacts with StatsAggregatorActor)."""

    def __init__(
        self,
        agent: Optional[AlphaZeroNet],
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[_LRScheduler],
        stats_aggregator: "StatsAggregatorHandle", # Actor Handle
        base_checkpoint_dir: str,
        run_checkpoint_dir: str,
        load_checkpoint_path_config: Optional[str],
        device: torch.device,
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stats_aggregator = stats_aggregator
        self.base_checkpoint_dir = base_checkpoint_dir
        self.run_checkpoint_dir = run_checkpoint_dir
        self.device = device
        self.global_step = 0
        self.episode_count = 0
        self.training_target_step = 0
        self.run_id_to_load_from, self.checkpoint_path_to_load = (
            self._determine_checkpoint_to_load(load_checkpoint_path_config)
        )
        if self.stats_aggregator:
            self.stats_aggregator.set_training_target_step.remote(self.training_target_step)

    def _determine_checkpoint_to_load(
        self, config_path: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Determines which checkpoint to load based on config or latest run."""
        if config_path:
            print(f"[CheckpointManager] Using explicit checkpoint path: {config_path}")
            if os.path.isfile(config_path):
                run_id = None
                try: run_id = os.path.basename(os.path.dirname(config_path)) if os.path.basename(os.path.dirname(config_path)).startswith("run_") else None
                except Exception: pass
                print(f"[CheckpointManager] Extracted run_id '{run_id}' from path." if run_id else "[CheckpointManager] Could not determine run_id from path.")
                return run_id, config_path
            else: print(f"[CheckpointManager] WARNING: Explicit path not found: {config_path}. Starting fresh."); return None, None
        else:
            print(f"[CheckpointManager] Searching for latest run in: {self.base_checkpoint_dir}")
            run_id, ckpt_path = find_latest_run_and_checkpoint(self.base_checkpoint_dir)
            if run_id and ckpt_path: print(f"[CheckpointManager] Found latest run '{run_id}' with checkpoint.")
            elif run_id: print(f"[CheckpointManager] Found latest run '{run_id}' but no checkpoint. Starting fresh.")
            else: print(f"[CheckpointManager] No previous runs found. Starting fresh.")
            return run_id, ckpt_path

    def get_run_id_to_load_from(self) -> Optional[str]:
        return self.run_id_to_load_from

    def get_checkpoint_path_to_load(self) -> Optional[str]:
        return self.checkpoint_path_to_load

    def load_checkpoint(self):
        """Loads agent, optimizer, scheduler state locally and stats state into the StatsAggregatorActor."""
        if not self.checkpoint_path_to_load or not os.path.isfile(self.checkpoint_path_to_load):
            print(f"[CheckpointManager] Checkpoint not found or not specified: {self.checkpoint_path_to_load}. Skipping load.")
            self._reset_local_states()
            return
        print(f"[CheckpointManager] Loading checkpoint: {self.checkpoint_path_to_load}")
        try:
            checkpoint = torch.load(self.checkpoint_path_to_load, map_location=self.device, weights_only=False)
            agent_ok = self._load_agent_state(checkpoint)
            opt_ok = self._load_optimizer_state(checkpoint)
            sched_ok = self._load_scheduler_state(checkpoint)
            stats_ok, loaded_target = self._load_stats_state_actor(checkpoint) # Load into actor
            self.global_step = checkpoint.get("global_step", 0)
            print(f"  -> Loaded Global Step: {self.global_step}")

            if stats_ok and self.stats_aggregator:
                try:
                    ep_count_ref = self.stats_aggregator.get_total_episodes.remote()
                    self.episode_count = ray.get(ep_count_ref)
                except Exception as e:
                    print(f"  -> ERROR getting episode count from StatsAggregatorActor: {e}")
                    self.episode_count = checkpoint.get("episode_count", 0)
            else:
                self.episode_count = checkpoint.get("episode_count", 0)

            print(f"  -> Resuming from Step: {self.global_step}, Ep: {self.episode_count}")
            self.training_target_step = loaded_target if loaded_target is not None else checkpoint.get("training_target_step", 0)
            if self.stats_aggregator:
                self.stats_aggregator.set_training_target_step.remote(self.training_target_step)

            print("[CheckpointManager] Checkpoint loading finished.")
            if not agent_ok: print("[CheckpointManager] Agent load was unsuccessful.")
            if not opt_ok: print("[CheckpointManager] Optimizer load was unsuccessful.")
            if not sched_ok: print("[CheckpointManager] Scheduler load was unsuccessful.")
            if not stats_ok: print("[CheckpointManager] Stats load was unsuccessful.")

        except (pickle.UnpicklingError, KeyError, Exception) as e:
            print(f"  -> ERROR loading checkpoint ('{e}'). State reset.")
            traceback.print_exc()
            self._reset_local_states()
        print(f"[CheckpointManager] Final Training Target Step set to: {self.training_target_step}")

    def _load_agent_state(self, checkpoint: Dict[str, Any]) -> bool:
        """Loads the agent state dictionary into the local agent."""
        if "agent_state_dict" not in checkpoint: print("  -> WARNING: 'agent_state_dict' missing."); return False
        if not self.agent: print("  -> WARNING: Local Agent not initialized."); return False
        try:
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            print("  -> Local Agent state loaded.")
            return True
        except Exception as e: print(f"  -> ERROR loading Local Agent state: {e}."); return False

    def _load_optimizer_state(self, checkpoint: Dict[str, Any]) -> bool:
        """Loads the optimizer state dictionary into the local optimizer."""
        if "optimizer_state_dict" not in checkpoint: print("  -> WARNING: 'optimizer_state_dict' missing."); return False
        if not self.optimizer: print("  -> WARNING: Optimizer not initialized."); return False
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor): state[k] = v.to(self.device)
            print("  -> Optimizer state loaded.")
            return True
        except Exception as e: print(f"  -> ERROR loading Optimizer state: {e}."); return False

    def _load_scheduler_state(self, checkpoint: Dict[str, Any]) -> bool:
        """Loads the scheduler state dictionary into the local scheduler."""
        if "scheduler_state_dict" not in checkpoint: print("  -> INFO: 'scheduler_state_dict' missing (may be expected)."); return False
        if not self.scheduler: print("  -> INFO: Scheduler not initialized for current run, skipping load."); return False
        try:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("  -> Scheduler state loaded.")
            return True
        except Exception as e: print(f"  -> ERROR loading Scheduler state: {e}."); return False

    def _load_stats_state_actor(self, checkpoint: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
        """Loads the stats aggregator state into the StatsAggregatorActor."""
        loaded_target = None
        if "stats_aggregator_state_dict" not in checkpoint:
            print("  -> WARNING: 'stats_aggregator_state_dict' missing.")
            return False, loaded_target
        if not self.stats_aggregator:
            print("  -> WARNING: Stats Aggregator Actor handle not available.")
            return False, loaded_target
        try:
            stats_state_dict = checkpoint["stats_aggregator_state_dict"]
            load_ref = self.stats_aggregator.load_state_dict.remote(stats_state_dict)
            ray.get(load_ref) # Wait for the actor to load the state
            print("  -> Stats Aggregator Actor state loaded.")
            # Get target step from actor after loading using the new getter method
            target_ref = self.stats_aggregator.get_training_target_step.remote()
            loaded_target = ray.get(target_ref)
            if loaded_target is not None: print(f"  -> Loaded Training Target Step from Stats Actor: {loaded_target}")
            return True, loaded_target
        except Exception as e:
            print(f"  -> ERROR loading Stats Aggregator Actor state: {e}.")
            return False, loaded_target

    def _reset_local_states(self):
        """Resets only the local agent/optimizer/scheduler states."""
        print("[CheckpointManager] Resetting local states due to load failure.")
        self.global_step = 0
        self.episode_count = 0
        self.training_target_step = 0
        if self.optimizer:
            self.optimizer.state = {}
            if self.scheduler:
                try: self.scheduler = type(self.scheduler)(self.optimizer, **self.scheduler.state_dict())
                except Exception as e: print(f"  -> WARNING: Failed to re-initialize scheduler: {e}"); self.scheduler = None
            print("  -> Optimizer state reset.")

    def save_checkpoint(
        self,
        global_step: int,
        episode_count: int, # This is now less reliable, fetch from actor
        training_target_step: int,
        is_final: bool = False,
    ):
        """Saves local agent/optimizer/scheduler state and fetches/saves state from StatsAggregatorActor."""
        prefix = "FINAL" if is_final else f"step_{global_step}"
        save_dir = self.run_checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{prefix}_alphazero_nn.pth"
        full_path = os.path.join(save_dir, filename)
        print(f"[CheckpointManager] Saving checkpoint ({prefix}) to {save_dir}...")
        temp_path = full_path + ".tmp"
        try:
            agent_sd = self.agent.state_dict() if self.agent else {}
            opt_sd = self.optimizer.state_dict() if self.optimizer else {}
            sched_sd = self.scheduler.state_dict() if self.scheduler else {}

            stats_sd = {}
            agg_ep_count = 0 # Default if fetch fails
            agg_target_step = training_target_step # Use passed value as fallback
            if self.stats_aggregator:
                try:
                    # Set target step before getting state (fire-and-forget)
                    self.stats_aggregator.set_training_target_step.remote(training_target_step)
                    # Get state dict, episode count, and target step from actor
                    state_ref = self.stats_aggregator.state_dict.remote()
                    ep_count_ref = self.stats_aggregator.get_total_episodes.remote()
                    target_ref = self.stats_aggregator.get_training_target_step.remote() # Use getter
                    # Fetch concurrently
                    stats_sd, agg_ep_count, agg_target_step = ray.get([state_ref, ep_count_ref, target_ref])
                    logger.info(f"Fetched state from StatsAggregatorActor for saving (Ep: {agg_ep_count}, Target: {agg_target_step}).")
                except Exception as e:
                    print(f"  -> WARNING: Failed to get state from StatsAggregatorActor: {e}. Saving potentially incomplete stats.")
                    stats_sd = {}

            checkpoint_data = {
                "global_step": global_step,
                "episode_count": agg_ep_count, # Use value from actor
                "training_target_step": agg_target_step, # Use value from actor
                "agent_state_dict": agent_sd,
                "optimizer_state_dict": opt_sd,
                "scheduler_state_dict": sched_sd,
                "stats_aggregator_state_dict": stats_sd,
            }
            torch.save(checkpoint_data, temp_path)
            os.replace(temp_path, full_path)
            print(f"  -> Checkpoint saved: {filename}")
        except Exception as e:
            print(f"  -> ERROR saving checkpoint: {e}")
            traceback.print_exc()
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except OSError: pass

    def get_initial_state(self) -> Tuple[int, int]:
        """Returns the initial global step and episode count after potential loading."""
        return self.global_step, self.episode_count

File: training\training_utils.py
import pygame
import numpy as np
from typing import Optional
from config import EnvConfig, VisConfig


def get_env_image_as_numpy(
    env, env_config: EnvConfig, vis_config: VisConfig
) -> Optional[np.ndarray]:
    """Renders a single environment state to a NumPy array for logging."""
    img_h = 300
    aspect_ratio = (env_config.COLS * 0.75 + 0.25) / max(1, env_config.ROWS)
    img_w = int(img_h * aspect_ratio)
    if img_w <= 0 or img_h <= 0:
        return None
    try:
        temp_surf = pygame.Surface((img_w, img_h))
        cell_w_px = img_w / (env_config.COLS * 0.75 + 0.25)
        cell_h_px = img_h / max(1, env_config.ROWS)
        temp_surf.fill(vis_config.BLACK)
        if hasattr(env, "grid") and hasattr(env.grid, "triangles"):
            for r in range(env.grid.rows):
                for c in range(env.grid.cols):
                    if r < len(env.grid.triangles) and c < len(env.grid.triangles[r]):
                        t = env.grid.triangles[r][c]
                        if t.is_death:
                            continue
                        pts = t.get_points(
                            ox=0, oy=0, cw=int(cell_w_px), ch=int(cell_h_px)
                        )
                        color = vis_config.GRAY
                        if t.is_occupied:
                            color = t.color if t.color else vis_config.RED
                        pygame.draw.polygon(temp_surf, color, pts)
        img_array = pygame.surfarray.array3d(temp_surf)
        return np.transpose(img_array, (1, 0, 2))
    except Exception as e:
        print(f"Error generating environment image for TB: {e}")
        return None


File: training\__init__.py
from .checkpoint_manager import CheckpointManager

__all__ = ["CheckpointManager"]


File: ui\demo_renderer.py
# File: ui/demo_renderer.py
import pygame
import math
import traceback
from typing import Tuple, Dict, Any, Optional

from config import VisConfig, EnvConfig, DemoConfig, RED

# from environment.game_state import GameState # No longer use GameState directly
from .panels.game_area import GameAreaRenderer  # Keep for rendering logic if needed
from .demo_components.grid_renderer import DemoGridRenderer
from .demo_components.preview_renderer import DemoPreviewRenderer
from .demo_components.hud_renderer import DemoHudRenderer


class DemoRenderer:
    """
    Handles rendering for Demo/Debug Mode based on data received from logic process.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        # GameAreaRenderer might not be needed if all logic is self-contained
        self.game_area_renderer = game_area_renderer

        # Initialize sub-renderers (pass screen, configs)
        self.grid_renderer = DemoGridRenderer(
            screen, vis_config, demo_config, game_area_renderer
        )
        self.preview_renderer = DemoPreviewRenderer(
            screen, vis_config, demo_config, game_area_renderer
        )
        self.hud_renderer = DemoHudRenderer(
            screen, vis_config, demo_config, game_area_renderer
        )

        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

    def render(
        self,
        demo_env_data: Dict[str, Any],  # Now accepts a data dictionary
        env_config: Optional[Dict[str, Any]] = None,  # Env config values as dict
        is_debug: bool = False,
    ):
        """Renders the entire demo/debug mode screen using provided data."""
        if not demo_env_data or not env_config:
            print("Error: DemoRenderer called with missing data or env_config")
            # Optionally render an error message
            return

        # Extract necessary info from demo_env_data
        # This replaces accessing demo_env object attributes
        is_over = demo_env_data.get("demo_env_is_over", False)
        score = demo_env_data.get("demo_env_score", 0)
        state_dict = demo_env_data.get("demo_env_state")  # The StateType dict
        dragged_shape_idx = demo_env_data.get("demo_env_dragged_shape_idx")
        snapped_pos = demo_env_data.get("demo_env_snapped_pos")
        selected_shape_idx = demo_env_data.get("demo_env_selected_shape_idx", -1)
        # Get shape data (assuming it's part of the state_dict or stats)
        available_shapes_data = []
        if (
            state_dict and "shapes" in state_dict
        ):  # Placeholder: Need actual shape info passed
            pass  # Need to reconstruct shape info for previews if not passed separately

        # Determine background color based on state flags (passed in demo_env_data)
        # bg_color = self.hud_renderer.determine_background_color(demo_env_data) # Adapt this method
        bg_color = self.demo_config.BACKGROUND_COLOR  # Simplified for now
        self.screen.fill(bg_color)

        screen_width, screen_height = self.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        # Calculate game area using env_config dict
        game_rect, clipped_game_rect = self.grid_renderer.calculate_game_area_rect(
            screen_width, screen_height, padding, hud_height, help_height, env_config
        )

        if clipped_game_rect.width > 10 and clipped_game_rect.height > 10:
            # Pass necessary data down to grid renderer
            self.grid_renderer.render_game_area(
                demo_env_data,  # Pass the data dict
                env_config,
                clipped_game_rect,
                bg_color,
                is_debug,
            )
        else:
            self.hud_renderer.render_too_small_message(
                "Demo Area Too Small", clipped_game_rect
            )

        if not is_debug:
            # Pass necessary data down to preview renderer
            self.shape_preview_rects = self.preview_renderer.render_shape_previews_area(
                demo_env_data,  # Pass the data dict
                screen_width,
                clipped_game_rect,
                padding,
            )
        else:
            self.shape_preview_rects.clear()

        # Pass necessary data down to HUD renderer
        self.hud_renderer.render_hud(
            demo_env_data,  # Pass the data dict
            screen_width,
            game_rect.bottom + 10,
            is_debug,
        )
        self.hud_renderer.render_help_text(screen_width, screen_height, is_debug)

    # Expose calculation methods if needed by InputHandler (unlikely now)
    # ...

    def get_shape_preview_rects(self) -> Dict[int, pygame.Rect]:
        """Returns the dictionary of screen-relative shape preview rects."""
        return self.preview_renderer.get_shape_preview_rects()


File: ui\input_handler.py
# File: ui/input_handler.py
import pygame
import multiprocessing as mp
from typing import Tuple, Dict, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .renderer import UIRenderer
    from app_state import AppState

COMMAND_SENTINEL = "COMMAND"
STOP_SENTINEL = "STOP"
PAYLOAD_KEY = "payload"  # Use this key for data


class InputHandler:
    """Handles Pygame events and sends commands to the Logic process via a queue."""

    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",
        command_queue: mp.Queue,
        stop_event: mp.Event,
    ):
        self.screen = screen
        self.renderer = renderer
        self.command_queue = command_queue
        self.stop_event = stop_event
        self.app_state_str = "Initializing"
        self.cleanup_confirmation_active = False
        self.is_process_running_cache = False  # Cache running state

        self.run_stop_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.cleanup_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.demo_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.debug_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_yes_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_no_rect = pygame.Rect(0, 0, 0, 0)
        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

        self._button_renderer = getattr(
            renderer.left_panel, "button_status_renderer", None
        )
        self._update_button_rects()

    def _update_ui_screen_references(self, new_screen: pygame.Surface):
        self.screen = new_screen
        components_to_update = [
            self.renderer,
            getattr(self.renderer, "left_panel", None),
            getattr(self.renderer, "game_area", None),
            getattr(self.renderer, "overlays", None),
            getattr(self.renderer, "demo_renderer", None),
        ]
        for component in components_to_update:
            if component and hasattr(component, "screen"):
                component.screen = new_screen

    def _update_button_rects(self):
        button_height = 40
        button_y_pos = 10
        run_stop_button_width = 150
        cleanup_button_width = 160
        demo_button_width = 120
        debug_button_width = 120
        button_spacing = 10
        current_x = button_spacing
        self.run_stop_btn_rect = pygame.Rect(
            current_x, button_y_pos, run_stop_button_width, button_height
        )
        current_x = self.run_stop_btn_rect.right + button_spacing * 2
        self.cleanup_btn_rect = pygame.Rect(
            current_x, button_y_pos, cleanup_button_width, button_height
        )
        current_x = self.cleanup_btn_rect.right + button_spacing
        self.demo_btn_rect = pygame.Rect(
            current_x, button_y_pos, demo_button_width, button_height
        )
        current_x = self.demo_btn_rect.right + button_spacing
        self.debug_btn_rect = pygame.Rect(
            current_x, button_y_pos, debug_button_width, button_height
        )
        sw, sh = self.screen.get_size()
        self.confirm_yes_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_no_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

    def update_state(
        self,
        app_state_str: str,
        cleanup_confirmation_active: bool,
        is_process_running: bool = False,
    ):
        self.app_state_str = app_state_str
        self.cleanup_confirmation_active = cleanup_confirmation_active
        self.is_process_running_cache = is_process_running  # Update cache
        if (
            self.app_state_str == "Playing"
            and self.renderer
            and self.renderer.demo_renderer
        ):
            self.shape_preview_rects = (
                self.renderer.demo_renderer.get_shape_preview_rects()
            )
        else:
            self.shape_preview_rects.clear()

    def _send_command(self, command: str, payload: Optional[Dict] = None):
        cmd_dict = {COMMAND_SENTINEL: command}
        if payload:
            cmd_dict[PAYLOAD_KEY] = payload
        try:
            self.command_queue.put(cmd_dict)
        except Exception as e:
            print(f"Error sending command '{command}' to queue: {e}")

    def handle_input(self) -> bool:
        from app_state import AppState

        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            mouse_pos = (0, 0)

        self._update_button_rects()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop_event.set()
                self._send_command(STOP_SENTINEL)
                return False

            if event.type == pygame.VIDEORESIZE:
                try:
                    new_w, new_h = max(320, event.w), max(240, event.h)
                    new_screen = pygame.display.set_mode(
                        (new_w, new_h), pygame.RESIZABLE
                    )
                    self._update_ui_screen_references(new_screen)
                    self._update_button_rects()
                    if hasattr(self.renderer, "force_redraw"):
                        self.renderer.force_redraw()
                    print(f"Window resized: {new_w}x{new_h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")
                continue

            if self.cleanup_confirmation_active:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._send_command("cancel_cleanup")
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.confirm_yes_rect.collidepoint(mouse_pos):
                        self._send_command("confirm_cleanup")
                    elif self.confirm_no_rect.collidepoint(mouse_pos):
                        self._send_command("cancel_cleanup")
                continue

            current_app_state = (
                AppState(self.app_state_str)
                if self.app_state_str in AppState._value2member_map_
                else AppState.UNKNOWN
            )

            if current_app_state == AppState.PLAYING:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._send_command("exit_demo_mode")
                elif event.type == pygame.MOUSEMOTION:
                    grid_coords = None  # TODO: Implement UI-side mapping
                    self._send_command(
                        "demo_mouse_motion", payload={"pos": grid_coords}
                    )
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked_preview = None
                    for idx, rect in self.shape_preview_rects.items():
                        if rect.collidepoint(event.pos):
                            clicked_preview = idx
                            break
                    if clicked_preview is not None:
                        self._send_command(
                            "demo_mouse_button_down",
                            payload={"type": "preview", "index": clicked_preview},
                        )
                    else:
                        grid_coords = None  # TODO: Implement UI-side mapping
                        if grid_coords:
                            self._send_command(
                                "demo_mouse_button_down",
                                payload={"type": "grid", "grid_coords": grid_coords},
                            )
                        else:
                            self._send_command(
                                "demo_mouse_button_down", payload={"type": "outside"}
                            )

            elif current_app_state == AppState.DEBUG:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._send_command("exit_debug_mode")
                    elif event.key == pygame.K_r:
                        self._send_command("debug_input", payload={"type": "reset"})
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    grid_coords = None  # TODO: Implement UI-side mapping
                    if grid_coords:
                        self._send_command(
                            "debug_input",
                            payload={
                                "type": "toggle_triangle",
                                "grid_coords": grid_coords,
                            },
                        )

            elif current_app_state == AppState.MAIN_MENU:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.stop_event.set()
                    self._send_command(STOP_SENTINEL)
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    is_running = (
                        self.is_process_running_cache
                    )  # Use cached value from render data

                    if self.run_stop_btn_rect.collidepoint(mouse_pos):
                        cmd = "stop_run" if is_running else "start_run"
                        if is_running:
                            self.stop_event.set()  # Set stop event immediately on UI action
                        self._send_command(cmd)
                    elif not is_running:
                        if self.cleanup_btn_rect.collidepoint(mouse_pos):
                            self._send_command("request_cleanup")
                        elif self.demo_btn_rect.collidepoint(mouse_pos):
                            self._send_command("start_demo_mode")
                        elif self.debug_btn_rect.collidepoint(mouse_pos):
                            self._send_command("start_debug_mode")

            elif current_app_state == AppState.ERROR:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.stop_event.set()
                    self._send_command(STOP_SENTINEL)
                    return False

        return True


File: ui\overlays.py
# File: ui/overlays.py
import pygame
import time
import traceback
from typing import Tuple
from config import VisConfig


class OverlayRenderer:
    """Renders overlay elements like confirmation dialogs and status messages."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()

    def _init_fonts(self):
        """Initializes fonts used for overlays."""
        fonts = {}
        try:
            fonts["overlay_title"] = pygame.font.SysFont(None, 36)
            fonts["overlay_text"] = pygame.font.SysFont(None, 24)
        except Exception as e:
            print(f"Warning: SysFont error for overlay fonts: {e}. Using default.")
            fonts["overlay_title"] = pygame.font.Font(None, 36)
            fonts["overlay_text"] = pygame.font.Font(None, 24)
        return fonts

    def render_cleanup_confirmation(self):
        """Renders the confirmation dialog for cleanup. Does not flip display."""
        try:
            current_width, current_height = self.screen.get_size()

            # Semi-transparent background overlay
            overlay_surface = pygame.Surface(
                (current_width, current_height), pygame.SRCALPHA
            )
            overlay_surface.fill((0, 0, 0, 200))  # Black with alpha
            self.screen.blit(overlay_surface, (0, 0))

            center_x, center_y = current_width // 2, current_height // 2

            # --- Render Text Lines ---
            if "overlay_title" not in self.fonts or "overlay_text" not in self.fonts:
                print("ERROR: Overlay fonts not loaded!")
                # Draw basic fallback text
                fallback_font = pygame.font.Font(None, 30)
                err_surf = fallback_font.render("CONFIRM CLEANUP?", True, VisConfig.RED)
                self.screen.blit(
                    err_surf, err_surf.get_rect(center=(center_x, center_y - 30))
                )
                yes_surf = fallback_font.render("YES", True, VisConfig.WHITE)
                no_surf = fallback_font.render("NO", True, VisConfig.WHITE)
                self.screen.blit(
                    yes_surf, yes_surf.get_rect(center=(center_x - 60, center_y + 50))
                )
                self.screen.blit(
                    no_surf, no_surf.get_rect(center=(center_x + 60, center_y + 50))
                )
                return  # Stop here if fonts failed

            # Use loaded fonts
            prompt_l1 = self.fonts["overlay_title"].render(
                "DELETE CURRENT RUN DATA?", True, VisConfig.RED
            )
            prompt_l2 = self.fonts["overlay_text"].render(
                "(NN Checkpoint & Stats)", True, VisConfig.WHITE
            )  # Updated text
            prompt_l3 = self.fonts["overlay_text"].render(
                "This action cannot be undone!", True, VisConfig.YELLOW
            )

            # Position and blit text
            self.screen.blit(
                prompt_l1, prompt_l1.get_rect(center=(center_x, center_y - 60))
            )
            self.screen.blit(
                prompt_l2, prompt_l2.get_rect(center=(center_x, center_y - 25))
            )
            self.screen.blit(prompt_l3, prompt_l3.get_rect(center=(center_x, center_y)))

            # --- Render Buttons ---
            # Recalculate rects based on current screen size for responsiveness
            confirm_yes_rect = pygame.Rect(center_x - 110, center_y + 30, 100, 40)
            confirm_no_rect = pygame.Rect(center_x + 10, center_y + 30, 100, 40)

            pygame.draw.rect(
                self.screen, (0, 150, 0), confirm_yes_rect, border_radius=5
            )  # Green YES
            pygame.draw.rect(
                self.screen, (150, 0, 0), confirm_no_rect, border_radius=5
            )  # Red NO

            yes_text = self.fonts["overlay_text"].render("YES", True, VisConfig.WHITE)
            no_text = self.fonts["overlay_text"].render("NO", True, VisConfig.WHITE)

            self.screen.blit(
                yes_text, yes_text.get_rect(center=confirm_yes_rect.center)
            )
            self.screen.blit(no_text, no_text.get_rect(center=confirm_no_rect.center))

        except pygame.error as pg_err:
            print(f"Pygame Error in render_cleanup_confirmation: {pg_err}")
            traceback.print_exc()
        except Exception as e:
            print(f"Error in render_cleanup_confirmation: {e}")
            traceback.print_exc()

    def render_status_message(self, message: str, last_message_time: float) -> bool:
        """
        Renders a status message (e.g., after cleanup) temporarily at the bottom center.
        Does not flip display. Returns True if a message was rendered.
        """
        # Check if message exists and hasn't timed out
        if not message or (time.time() - last_message_time >= 5.0):
            return False

        try:
            if "overlay_text" not in self.fonts:  # Check if font loaded
                print(
                    "Warning: Cannot render status message, overlay_text font missing."
                )
                return False

            current_width, current_height = self.screen.get_size()
            lines = message.split("\n")
            max_width = 0
            msg_surfs = []

            # Render each line and find max width
            for line in lines:
                msg_surf = self.fonts["overlay_text"].render(
                    line,
                    True,
                    VisConfig.YELLOW,
                    VisConfig.BLACK,  # Yellow text on black bg
                )
                msg_surfs.append(msg_surf)
                max_width = max(max_width, msg_surf.get_width())

            if not msg_surfs:
                return False  # No lines to render

            # Calculate background size and position
            total_height = (
                sum(s.get_height() for s in msg_surfs) + max(0, len(lines) - 1) * 2
            )
            padding = 5
            bg_rect = pygame.Rect(
                0, 0, max_width + padding * 2, total_height + padding * 2
            )
            bg_rect.midbottom = (
                current_width // 2,
                current_height - 10,
            )  # Position at bottom center

            # Draw background and border
            pygame.draw.rect(self.screen, VisConfig.BLACK, bg_rect, border_radius=3)
            pygame.draw.rect(self.screen, VisConfig.YELLOW, bg_rect, 1, border_radius=3)

            # Draw text lines centered within the background
            current_y = bg_rect.top + padding
            for msg_surf in msg_surfs:
                msg_rect = msg_surf.get_rect(midtop=(bg_rect.centerx, current_y))
                self.screen.blit(msg_surf, msg_rect)
                current_y += msg_surf.get_height() + 2  # Move Y for next line

            return True  # Message was rendered
        except Exception as e:
            print(f"Error rendering status message: {e}")
            traceback.print_exc()
            return False  # Message render failed


File: ui\plotter.py
# File: ui/plotter.py
import pygame
from typing import Dict, Optional, Deque, Tuple
from collections import deque
import matplotlib
import time
from io import BytesIO
import logging
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, StatsConfig
from .plot_utils import render_single_plot, normalize_color_for_matplotlib

logger = logging.getLogger(__name__)


class Plotter:
    """Handles creation and caching of the multi-plot Matplotlib surface."""

    def __init__(self):
        self.plot_surface_cache: Optional[pygame.Surface] = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = (
            2.0  # Faster update for more responsive plots
        )
        self.rolling_window_sizes = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW
        self.colors = self._init_colors()

        self.fig: Optional[plt.Figure] = None
        self.axes: Optional[np.ndarray] = None  # Will be a 2D numpy array of axes
        self.last_target_size: Tuple[int, int] = (0, 0)
        self.last_data_hash: Optional[int] = None

        logger.info(
            f"[Plotter] Initialized with update interval: {self.plot_update_interval}s"
        )

    def _init_colors(self) -> Dict[str, Tuple[float, float, float]]:
        """Initializes plot colors."""
        return {
            "game_scores": normalize_color_for_matplotlib(VisConfig.GREEN),
            "policy_losses": normalize_color_for_matplotlib(VisConfig.RED),
            "value_losses": normalize_color_for_matplotlib(VisConfig.ORANGE),
            "episode_lengths": normalize_color_for_matplotlib(VisConfig.BLUE),
            "episode_outcomes": normalize_color_for_matplotlib(VisConfig.YELLOW),
            "episode_triangles_cleared": normalize_color_for_matplotlib(VisConfig.CYAN),
            "lr_values": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[2]),
            "buffer_sizes": normalize_color_for_matplotlib(VisConfig.PURPLE),
            "best_game_score_history": normalize_color_for_matplotlib(VisConfig.GREEN),
            # MCTS Stats Colors
            "mcts_simulation_times": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[1]
            ),
            "mcts_nn_prediction_times": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[3]
            ),
            "mcts_nodes_explored": normalize_color_for_matplotlib(VisConfig.LIGHTG),
            "mcts_avg_depths": normalize_color_for_matplotlib(VisConfig.WHITE),
            # System Stats Colors
            "steps_per_second": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[0]
            ),
            # Placeholder Color
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
        }

    def _init_figure(self, target_width: int, target_height: int):
        """Initializes the Matplotlib figure and axes."""
        logger.info(
            f"[Plotter] Initializing Matplotlib figure for size {target_width}x{target_height}"
        )
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                logger.warning(f"Error closing previous figure: {e}")

        dpi = 96  # Standard DPI
        fig_width_in = max(1, target_width / dpi)
        fig_height_in = max(1, target_height / dpi)

        try:
            # 5 rows, 3 columns grid = 15 plots
            self.fig, self.axes = plt.subplots(
                5, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
            )
            self.fig.subplots_adjust(
                hspace=0.4,
                wspace=0.3,  # Increased spacing slightly
                left=0.08,
                right=0.97,
                bottom=0.06,
                top=0.96,  # Adjusted margins
            )
            self.last_target_size = (target_width, target_height)
            logger.info("[Plotter] Matplotlib figure initialized (5x3 grid).")
        except Exception as e:
            logger.error(f"Error creating Matplotlib figure: {e}", exc_info=True)
            self.fig = None
            self.axes = None
            self.last_target_size = (0, 0)

    def _get_data_hash(self, plot_data: Dict[str, Deque]) -> int:
        """Generates a simple hash based on data lengths and last elements."""
        hash_val = 0
        # Include all keys used in plot_defs
        keys_to_hash = [
            "game_scores",
            "episode_outcomes",
            "episode_lengths",
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_triangles_cleared",
            "best_game_score_history",
            "buffer_sizes",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "steps_per_second",
            "mcts_nodes_explored",
            "mcts_avg_depths",
        ]
        for key in sorted(keys_to_hash):
            dq = plot_data.get(key)  # Use .get() to handle missing keys gracefully
            if dq is None or not dq:  # Check if deque exists and is not empty
                continue  # Skip if key doesn't exist or deque is empty

            hash_val ^= hash(key)
            hash_val ^= len(dq)
            try:
                # Hash based on the last element to detect changes
                last_elem = dq[-1]
                if isinstance(last_elem, (int, float)):
                    hash_val ^= hash(f"{last_elem:.6f}")  # Format floats consistently
                else:
                    hash_val ^= hash(str(last_elem))
            except IndexError:
                pass  # Should not happen if dq is not empty
        return hash_val

    def _update_plot_data(self, plot_data: Dict[str, Deque]):
        """Updates the data on the existing Matplotlib axes."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()  # Flatten the 2D array of axes
            # --- Define Plot Order (5x3 Grid) ---
            plot_defs = [
                # Row 1: Game Performance
                ("game_scores", "Game Score", self.colors["game_scores"], False),
                ("episode_lengths", "Ep Length", self.colors["episode_lengths"], False),
                (
                    "episode_triangles_cleared",
                    "Tris Cleared/Ep",
                    self.colors["episode_triangles_cleared"],
                    False,
                ),
                # Row 2: Losses & LR
                (
                    "policy_losses",
                    "Policy Loss",
                    self.colors["policy_losses"],
                    True,
                ),  # Log scale often helpful for losses
                (
                    "value_losses",
                    "Value Loss",
                    self.colors["value_losses"],
                    True,
                ),  # Log scale often helpful for losses
                (
                    "lr_values",
                    "Learning Rate",
                    self.colors["lr_values"],
                    True,
                ),  # Log scale for LR is common
                # Row 3: Buffer & History
                ("buffer_sizes", "Buffer Size", self.colors["buffer_sizes"], False),
                (
                    "best_game_score_history",
                    "Best Score Hist",
                    self.colors["best_game_score_history"],
                    False,
                ),
                (
                    "episode_outcomes",
                    "Ep Outcome (-1,0,1)",
                    self.colors["episode_outcomes"],
                    False,
                ),  # Added outcome plot
                # Row 4: MCTS Timings / System
                (
                    "mcts_simulation_times",
                    "MCTS Sim Time (s)",
                    self.colors["mcts_simulation_times"],
                    False,
                ),
                (
                    "mcts_nn_prediction_times",
                    "MCTS NN Time (s)",
                    self.colors["mcts_nn_prediction_times"],
                    False,
                ),
                (
                    "steps_per_second",
                    "Steps/Sec",
                    self.colors["steps_per_second"],
                    False,
                ),
                # Row 5: MCTS Structure
                (
                    "mcts_nodes_explored",
                    "MCTS Nodes Explored",
                    self.colors["mcts_nodes_explored"],
                    False,
                ),
                (
                    "mcts_avg_depths",
                    "MCTS Avg Depth",
                    self.colors["mcts_avg_depths"],
                    False,
                ),
                (
                    "placeholder",
                    "Future Plot",
                    self.colors["placeholder"],
                    False,
                ),  # Keep one placeholder
            ]

            # Convert deques to lists for plotting (avoids issues with some matplotlib versions)
            data_lists = {
                key: list(plot_data.get(key, deque())) for key, _, _, _ in plot_defs
            }

            for i, (data_key, label, color, log_scale) in enumerate(plot_defs):
                if i >= len(axes_flat):
                    break  # Stop if we run out of axes
                ax = axes_flat[i]
                ax.clear()  # Clear previous plot content

                # Check if data exists before plotting
                current_data = data_lists.get(data_key, [])
                show_placeholder = (data_key == "placeholder") or not current_data

                render_single_plot(
                    ax,
                    current_data,
                    label,
                    color,
                    self.rolling_window_sizes,
                    show_placeholder=show_placeholder,
                    placeholder_text=label,  # Use label as placeholder text
                    y_log_scale=log_scale,
                )
                # Hide x-labels for plots not in the last row
                if i < len(axes_flat) - 3:  # Assuming 3 columns
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=0)  # Keep rotation at 0

            plot_update_duration = time.monotonic() - plot_update_start
            logger.info(f"[Plotter] Plot data updated in {plot_update_duration:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error updating plot data: {e}", exc_info=True)
            # Attempt to clear axes on error to avoid showing stale incorrect plots
            try:
                if self.axes is not None:
                    for ax in self.axes.flatten():
                        ax.clear()
            except Exception:
                pass  # Ignore errors during cleanup
            return False

    def _render_figure_to_surface(
        self, target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Renders the current Matplotlib figure to a Pygame surface."""
        if self.fig is None:
            logger.warning("[Plotter] Cannot render figure, not initialized.")
            return None

        render_start = time.monotonic()
        try:
            # Explicitly draw the canvas
            self.fig.canvas.draw()

            # Use BytesIO buffer to avoid disk I/O
            buf = BytesIO()
            self.fig.savefig(
                buf,
                format="png",
                transparent=False,  # Keep background
                facecolor=plt.rcParams[
                    "figure.facecolor"
                ],  # Use defined background color
            )
            buf.seek(0)

            # Load buffer into Pygame surface
            plot_img_surface = pygame.image.load(
                buf, "png"
            ).convert()  # Use convert() for performance
            buf.close()

            # --- Scaling ---
            # Scale the rendered surface to the exact target size if needed
            current_size = plot_img_surface.get_size()
            if current_size != (target_width, target_height):
                scale_diff = abs(current_size[0] - target_width) + abs(
                    current_size[1] - target_height
                )
                # Use smoothscale for larger differences, faster scale otherwise
                if scale_diff > 10:  # Arbitrary threshold
                    plot_img_surface = pygame.transform.smoothscale(
                        plot_img_surface, (target_width, target_height)
                    )
                else:
                    plot_img_surface = pygame.transform.scale(
                        plot_img_surface, (target_width, target_height)
                    )

            render_duration = time.monotonic() - render_start
            logger.info(
                f"[Plotter] Figure rendered to surface in {render_duration:.4f}s"
            )
            return plot_img_surface

        except Exception as e:
            logger.error(
                f"Error rendering Matplotlib figure to surface: {e}", exc_info=True
            )
            return None

    def get_cached_or_updated_plot(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Returns the cached plot surface or creates/updates one if needed."""
        current_time = time.time()
        has_data = any(d for d in plot_data.values())  # Check if any data exists
        target_size = (target_width, target_height)

        # Check if figure needs reinitialization (first time, or size changed)
        needs_reinit = (
            self.fig is None
            or self.axes is None
            or self.last_target_size != target_size
        )

        # Check if data has changed or enough time has passed
        current_data_hash = self._get_data_hash(plot_data)
        data_changed = self.last_data_hash != current_data_hash
        time_elapsed = (
            current_time - self.last_plot_update_time
        ) > self.plot_update_interval
        needs_update = data_changed or time_elapsed

        # Check if target size is large enough to render plots meaningfully
        can_create_plot = target_width > 50 and target_height > 50

        # Handle cases where plotting is not possible or not needed
        if not can_create_plot:
            if self.plot_surface_cache is not None:
                logger.info(
                    "[Plotter] Target size too small, clearing cache and figure."
                )
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None  # Return None if area is too small

        if not has_data:
            if self.plot_surface_cache is not None:
                logger.info(
                    "[Plotter] No plot data available, clearing cache and figure."
                )
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None  # Return None if there's no data

        # --- Update or Reinitialize ---
        cache_status = "HIT"
        try:
            if needs_reinit:
                cache_status = "MISS (Re-init)"
                logger.info(
                    f"[Plotter] {cache_status}. Reason: fig/axes is None or size changed ({self.last_target_size} != {target_size})"
                )
                self._init_figure(target_width, target_height)
                if self.fig:
                    if self._update_plot_data(plot_data):
                        self.plot_surface_cache = self._render_figure_to_surface(
                            target_width, target_height
                        )
                        self.last_plot_update_time = current_time
                        self.last_data_hash = current_data_hash
                    else:
                        self.plot_surface_cache = None  # Update failed
                else:
                    self.plot_surface_cache = None  # Init failed

            elif needs_update:
                cache_status = (
                    f"MISS (Update - Data: {data_changed}, Time: {time_elapsed})"
                )
                logger.info(f"[Plotter] {cache_status}")
                if self._update_plot_data(plot_data):  # Update existing figure data
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    logger.warning(
                        "[Plotter] Plot data update failed, returning potentially stale cache."
                    )
                    cache_status = "ERROR (Update Failed)"

            elif self.plot_surface_cache is None:
                # Cache is None, but figure might exist and data hasn't changed recently
                # This can happen if the first render failed
                cache_status = "MISS (Cache None)"
                logger.info(f"[Plotter] {cache_status}, attempting re-render.")
                if self._update_plot_data(plot_data):  # Try updating data again
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = (
                        current_time  # Update time even on re-render
                    )
                    self.last_data_hash = current_data_hash

            # Log cache status for debugging performance
            # logger.info(f"[Plotter] Cache status: {cache_status}")

        except Exception as e:
            logger.error(
                f"[Plotter] Unexpected error in get_cached_or_updated_plot: {e}",
                exc_info=True,
            )
            self.plot_surface_cache = None
            if self.fig:
                plt.close(self.fig)  # Clean up figure on error
            self.fig, self.axes = None, None
            self.last_target_size = (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        """Ensure Matplotlib figure is closed when the plotter is garbage collected."""
        if self.fig:
            try:
                plt.close(self.fig)
                logger.info("[Plotter] Matplotlib figure closed in destructor.")
            except Exception as e:
                # Log error but don't crash during GC
                logger.error(f"[Plotter] Error closing figure in destructor: {e}")


File: ui\plot_utils.py
# File: ui/plot_utils.py
import numpy as np
from typing import Optional, List, Union, Tuple
import matplotlib
import traceback
import math

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig

# --- Constants ---
TREND_SLOPE_TOLERANCE = 1e-5
TREND_MIN_LINEWIDTH = 1
TREND_MAX_LINEWIDTH = 2
TREND_COLOR_STABLE = (1.0, 1.0, 0.0)  # Yellow
TREND_COLOR_INCREASING = (0.0, 0.8, 0.0)  # Green
TREND_COLOR_DECREASING = (0.8, 0.0, 0.0)  # Red
TREND_SLOPE_SCALE_FACTOR = 5.0
TREND_BACKGROUND_ALPHA = 0.15
TREND_LINE_COLOR = (1.0, 1.0, 1.0)
TREND_LINE_STYLE = (0, (5, 10))
TREND_LINE_WIDTH = 0.75
TREND_LINE_ALPHA = 0.7
TREND_LINE_ZORDER = 10
MIN_ALPHA = 0.4
MAX_ALPHA = 1.0
MIN_DATA_AVG_LINEWIDTH = 1
MAX_DATA_AVG_LINEWIDTH = 2


# --- Helper Functions ---
def normalize_color_for_matplotlib(
    color_tuple_0_255: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        return tuple(c / 255.0 for c in color_tuple_0_255)
    return (0.0, 0.0, 0.0)  # Default black


# --- Matplotlib Style Setup ---
try:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "#262626",
            "axes.facecolor": "#303030",
            "axes.edgecolor": "#707070",
            "axes.labelcolor": "#D0D0D0",
            "xtick.color": "#C0C0C0",
            "ytick.color": "#C0C0C0",
            "grid.color": "#505050",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "axes.titlepad": 6,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.facecolor": "#202020",
            "legend.title_fontsize": 8,
        }
    )
except Exception as e:
    print(f"Warning: Failed to set Matplotlib style: {e}")


# --- Trend Calculation ---
def calculate_trend_line(data: np.ndarray) -> Optional[Tuple[float, float]]:
    """Calculates the slope and intercept of the linear regression line."""
    n = len(data)
    x = np.arange(n)
    mask = np.isfinite(data)
    if np.sum(mask) < 2:
        return None
    try:
        coeffs = np.polyfit(x[mask], data[mask], 1)
        if not all(np.isfinite(c) for c in coeffs):
            return None
        return coeffs[0], coeffs[1]  # slope, intercept
    except (np.linalg.LinAlgError, ValueError):
        return None


def get_trend_color(slope: float, lower_is_better: bool) -> Tuple[float, float, float]:
    """Maps slope to color (Red -> Yellow -> Green)."""
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_COLOR_STABLE
    eff_slope = -slope if lower_is_better else slope
    norm_slope = np.clip(
        math.atan(eff_slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0), -1.0, 1.0
    )
    t = abs(norm_slope)
    base, target = (
        (TREND_COLOR_STABLE, TREND_COLOR_INCREASING)
        if norm_slope > 0
        else (TREND_COLOR_STABLE, TREND_COLOR_DECREASING)
    )
    color = tuple(base[i] * (1 - t) + target[i] * t for i in range(3))
    return tuple(np.clip(c, 0.0, 1.0) for c in color)


def get_trend_linewidth(slope: float) -> float:
    """Maps slope magnitude to border linewidth."""
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_MIN_LINEWIDTH
    norm_mag = np.clip(
        abs(math.atan(slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0)), 0.0, 1.0
    )
    return TREND_MIN_LINEWIDTH + norm_mag * (TREND_MAX_LINEWIDTH - TREND_MIN_LINEWIDTH)


# --- Visual Property Interpolation ---
def _interpolate_visual_property(
    rank: int, total_ranks: int, min_val: float, max_val: float
) -> float:
    """Linearly interpolates alpha/linewidth based on rank."""
    if total_ranks <= 1:
        return float(max_val)
    inv_rank = (total_ranks - 1) - rank
    fraction = inv_rank / max(1, total_ranks - 1)
    value = float(min_val) + (float(max_val) - float(min_val)) * fraction
    return float(np.clip(value, min_val, max_val))


# --- Value Formatting ---
def _format_value(value: float, is_loss: bool) -> str:
    """Formats value based on magnitude and whether it's a loss."""
    if not np.isfinite(value):
        return "N/A"
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.1e}"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", "_")
    if is_loss:
        return f"{value:.3f}"
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _format_slope(slope: float) -> str:
    """Formats slope value for display in the legend."""
    if not np.isfinite(slope):
        return "N/A"
    sign = "+" if slope >= 0 else ""
    abs_slope = abs(slope)
    if abs_slope < 1e-4:
        return f"{sign}{slope:.1e}"
    if abs_slope < 0.1:
        return f"{sign}{slope:.3f}"
    return f"{sign}{slope:.2f}"


# --- Main Plotting Function ---
def render_single_plot(
    ax,
    data: List[Union[float, int]],
    label: str,
    color: Tuple[float, float, float],
    rolling_window_sizes: List[int],
    show_placeholder: bool = True,
    placeholder_text: Optional[str] = None,
    y_log_scale: bool = False,
):
    """Renders data with rolling averages, trend line, and informative legend."""
    try:
        data_np = np.array(data, dtype=float)
        valid_data = data_np[np.isfinite(data_np)]
    except (ValueError, TypeError):
        valid_data = np.array([])
    n_points = len(valid_data)
    is_lower_better = "loss" in label.lower()

    if n_points == 0:  # Handle empty data
        if show_placeholder:
            p_text = placeholder_text if placeholder_text else f"{label}\n(No data)"
            ax.text(
                0.5,
                0.5,
                p_text,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color=normalize_color_for_matplotlib(VisConfig.GRAY),
            )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"{label} (N/A)")
        ax.grid(False)
        ax.patch.set_facecolor(plt.rcParams["axes.facecolor"])
        ax.patch.set_edgecolor(plt.rcParams["axes.edgecolor"])
        ax.patch.set_linewidth(0.5)
        return

    trend_params = calculate_trend_line(valid_data)
    trend_slope = trend_params[0] if trend_params else 0.0
    trend_color = get_trend_color(trend_slope, is_lower_better)
    trend_lw = get_trend_linewidth(trend_slope)
    plotted_windows = sorted([w for w in rolling_window_sizes if n_points >= w])
    total_ranks = 1 + len(plotted_windows)
    current_val = valid_data[-1]
    best_val = np.min(valid_data) if is_lower_better else np.max(valid_data)
    best_val_str = f"Best: {_format_value(best_val, is_lower_better)}"
    ax.set_title(label, loc="left")

    try:
        x_coords = np.arange(n_points)
        plotted_legend = False
        min_y, max_y = float("inf"), float("-inf")
        plot_raw = len(plotted_windows) == 0 or n_points < min(
            rolling_window_sizes, default=10
        )
        if plot_raw:
            rank = 0
            alpha = _interpolate_visual_property(
                rank, total_ranks, MIN_ALPHA, MAX_ALPHA
            )
            lw = _interpolate_visual_property(
                rank, total_ranks, MIN_DATA_AVG_LINEWIDTH, MAX_DATA_AVG_LINEWIDTH
            )
            raw_label = f"Val: {_format_value(current_val, is_lower_better)}"
            ax.plot(
                x_coords,
                valid_data,
                color=color,
                linewidth=lw,
                label=raw_label,
                alpha=alpha,
            )
            min_y = min(min_y, np.min(valid_data))
            max_y = max(max_y, np.max(valid_data))
            plotted_legend = True

        for i, avg_win in enumerate(reversed(plotted_windows)):
            rank = i
            alpha = _interpolate_visual_property(
                rank, total_ranks, MIN_ALPHA, MAX_ALPHA
            )
            lw = _interpolate_visual_property(
                rank, total_ranks, MIN_DATA_AVG_LINEWIDTH, MAX_DATA_AVG_LINEWIDTH
            )
            weights = np.ones(avg_win) / avg_win
            rolling_avg = np.convolve(valid_data, weights, mode="valid")
            avg_x = np.arange(avg_win - 1, n_points)
            if len(avg_x) == len(rolling_avg):
                last_avg = rolling_avg[-1] if len(rolling_avg) > 0 else np.nan
                avg_label = f"Avg {avg_win}: {_format_value(last_avg, is_lower_better)}"
                ax.plot(
                    avg_x,
                    rolling_avg,
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                    linestyle="-",
                    label=avg_label,
                )
                if len(rolling_avg) > 0:
                    min_y = min(min_y, np.min(rolling_avg))
                    max_y = max(max_y, np.max(rolling_avg))
                plotted_legend = True

        if trend_params and n_points >= 2:
            slope, intercept = trend_params
            x_trend = np.array([0, n_points - 1])
            y_trend = slope * x_trend + intercept
            trend_label = f"Trend: {_format_slope(slope)}"
            ax.plot(
                x_trend,
                y_trend,
                color=TREND_LINE_COLOR,
                linestyle=TREND_LINE_STYLE,
                linewidth=TREND_LINE_WIDTH,
                alpha=TREND_LINE_ALPHA,
                label=trend_label,
                zorder=TREND_LINE_ZORDER,
            )
            plotted_legend = True

        ax.tick_params(axis="both", which="major")
        ax.grid(
            True,
            linestyle=plt.rcParams["grid.linestyle"],
            alpha=plt.rcParams["grid.alpha"],
        )
        if np.isfinite(min_y) and np.isfinite(max_y):
            yrange = max(max_y - min_y, 1e-6)
            pad = yrange * 0.05
            ax.set_ylim(min_y - pad, max_y + pad)
        if y_log_scale and min_y > 1e-9:
            ax.set_yscale("log")
            bottom, top = ax.get_ylim()
            new_bottom = max(bottom, 1e-9)
            if new_bottom >= top:
                new_bottom = top / 10
            ax.set_ylim(bottom=new_bottom, top=top)
        else:
            ax.set_yscale("linear")
        if n_points > 1:
            ax.set_xlim(0, n_points - 1)
        elif n_points == 1:
            ax.set_xlim(-0.5, 0.5)
        if n_points > 1000:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

            def fmt_func(v, _):
                val = int(v)
                return (
                    f"{val/1e6:.1f}M"
                    if val >= 1e6
                    else (f"{val/1e3:.0f}k" if val >= 1e3 else f"{val}")
                )

            ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_func))
        elif n_points > 10:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        if plotted_legend:
            ax.legend(loc="center left", bbox_to_anchor=(0, 0.5), title=best_val_str)

    except Exception as plot_err:
        print(f"ERROR during render_single_plot for '{label}': {plot_err}")
        traceback.print_exc()
        ax.text(
            0.5,
            0.5,
            f"Plot Error\n({label})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color=normalize_color_for_matplotlib(VisConfig.RED),
        )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)

    ax.patch.set_facecolor((*trend_color, TREND_BACKGROUND_ALPHA))
    ax.patch.set_edgecolor(trend_color)
    ax.patch.set_linewidth(trend_lw)


File: ui\renderer.py
# File: ui/renderer.py
import pygame
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque
import logging

from config import VisConfig, EnvConfig, DemoConfig

# from environment.game_state import GameState # No longer needed directly
from .panels import LeftPanelRenderer, GameAreaRenderer
from .overlays import OverlayRenderer
from .plotter import Plotter
from .demo_renderer import DemoRenderer
from .input_handler import InputHandler
from app_state import AppState

logger = logging.getLogger(__name__)


class UIRenderer:
    """Orchestrates rendering of all UI components based on data received from the logic process."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        # Plotter is specific to the UI process
        self.plotter = Plotter()
        # Sub-renderers are initialized here
        self.left_panel = LeftPanelRenderer(screen, vis_config, self.plotter)
        self.game_area = GameAreaRenderer(screen, vis_config)
        self.overlays = OverlayRenderer(screen, vis_config)
        self.demo_config = DemoConfig()
        # DemoRenderer needs access to GameAreaRenderer's fonts/methods if shared
        self.demo_renderer = DemoRenderer(
            screen, vis_config, self.demo_config, self.game_area
        )
        self.input_handler_ref: Optional[InputHandler] = None

    def set_input_handler(self, input_handler: InputHandler):
        """Sets the InputHandler reference for components that need it (e.g., buttons)."""
        self.input_handler_ref = input_handler
        self.left_panel.input_handler = input_handler
        # Pass references down if needed
        if hasattr(self.left_panel, "button_status_renderer"):
            self.left_panel.button_status_renderer.input_handler_ref = input_handler
            # Pass app_ref if needed (though app_ref is less relevant now)
            # self.left_panel.button_status_renderer.app_ref = input_handler.app_ref

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0
        # Clear caches that might depend on screen size or old data
        self.game_area.best_state_surface_cache = None
        self.game_area.placeholder_surface_cache = None
        # Maybe force re-calc of button rects? (Handled by input handler resize)
        logger.info("[Renderer] Forced redraw triggered.")

    def render_all(self, **render_data: Dict[str, Any]):
        """
        Renders UI based on the application state dictionary received from the logic process.
        """
        try:
            app_state_str = render_data.get("app_state", AppState.UNKNOWN.value)
            current_app_state = (
                AppState(app_state_str)
                if app_state_str in AppState._value2member_map_
                else AppState.UNKNOWN
            )

            # --- Main Rendering Logic ---
            if current_app_state == AppState.MAIN_MENU:
                self._render_main_menu(**render_data)
            elif current_app_state == AppState.PLAYING:
                self._render_demo_mode(is_debug=False, **render_data)
            elif current_app_state == AppState.DEBUG:
                self._render_demo_mode(is_debug=True, **render_data)
            elif current_app_state == AppState.INITIALIZING:
                # Use status from render_data
                self._render_initializing_screen(
                    render_data.get("status", "Initializing...")
                )
            elif current_app_state == AppState.ERROR:
                self._render_error_screen(render_data.get("status", "Unknown Error"))
            else:  # Handle other potential states or default view
                self._render_simple_message(f"State: {app_state_str}", VisConfig.WHITE)

            # Render overlays on top (e.g., cleanup confirmation)
            if (
                render_data.get("cleanup_confirmation_active")
                and current_app_state != AppState.ERROR
            ):
                self.overlays.render_cleanup_confirmation()
            elif not render_data.get("cleanup_confirmation_active"):
                # Render temporary status messages (like 'Cleanup complete')
                self.overlays.render_status_message(
                    render_data.get("cleanup_message", ""),
                    render_data.get("last_cleanup_message_time", 0.0),
                )

            pygame.display.flip()  # Flip the display once after all rendering

        except pygame.error as e:
            logger.error(f"Pygame rendering error in render_all: {e}", exc_info=True)
            try:
                self._render_simple_message("Pygame Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception:
                pass
        except Exception as e:
            logger.critical(
                f"Unexpected critical rendering error in render_all: {e}", exc_info=True
            )
            try:
                self._render_simple_message("Critical Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception:
                pass

    def _render_main_menu(self, **render_data: Dict[str, Any]):
        """Renders the main dashboard view with live worker envs."""
        self.screen.fill(VisConfig.BLACK)
        current_width, current_height = self.screen.get_size()
        lp_width, ga_width = self._calculate_panel_widths(current_width)

        # --- Render Left Panel ---
        # Pass only the necessary data from render_data
        self.left_panel.render(
            panel_width=lp_width,
            is_process_running=render_data.get("is_process_running", False),
            status=render_data.get("status", ""),
            stats_summary=render_data.get("stats_summary", {}),
            plot_data=render_data.get("plot_data", {}),
            app_state=render_data.get("app_state", ""),
            update_progress_details={},  # Maybe remove if not used
            agent_param_count=render_data.get("agent_param_count", 0),
            worker_counts=render_data.get("worker_counts", {}),
        )

        # --- Render Game Area Panel ---
        if ga_width > 0:
            # Recreate minimalist EnvConfig for rendering if needed
            # Or pass necessary values directly
            # env_config_render = EnvConfig() # Avoid creating full config if possible
            env_config_render_dict = {
                "ROWS": render_data.get("env_config_rows", 8),
                "COLS": render_data.get("env_config_cols", 15),
                # Add other needed EnvConfig values if GameAreaRenderer uses them
            }
            self.game_area.render(
                panel_width=ga_width,
                panel_x_offset=lp_width,
                worker_render_data=render_data.get("worker_render_data", []),
                num_envs=render_data.get("num_envs", 0),
                env_config=env_config_render_dict,  # Pass dict or minimalist object
                best_game_state_data=render_data.get("best_game_state_data"),
            )

    def _calculate_panel_widths(self, current_width: int) -> Tuple[int, int]:
        """Calculates the widths for the left and game area panels."""
        left_panel_ratio = max(0.2, min(0.8, self.vis_config.LEFT_PANEL_RATIO))
        lp_width = int(current_width * left_panel_ratio)
        ga_width = current_width - lp_width
        min_lp_width = 400
        if lp_width < min_lp_width and current_width > min_lp_width:
            lp_width = min_lp_width
            ga_width = max(0, current_width - lp_width)
        elif current_width <= min_lp_width:
            lp_width = current_width
            ga_width = 0
        return lp_width, ga_width

    def _render_demo_mode(self, is_debug: bool, **render_data: Dict[str, Any]):
        """Renders the demo or debug mode using data from the queue."""
        if not self.demo_renderer:
            self._render_simple_message("Demo Renderer Missing!", VisConfig.RED)
            return

        # Reconstruct a temporary GameState or pass data directly
        # Passing data directly avoids creating GameState in UI process
        # Requires DemoRenderer to accept data dictionary instead of GameState object
        demo_env_state = render_data.get("demo_env_state")
        if demo_env_state:
            # DemoRenderer needs to be adapted to use this data
            self.demo_renderer.render(
                demo_env_data=render_data,  # Pass the relevant subset
                env_config=None,  # Pass necessary config values instead
                is_debug=is_debug,
            )
        else:
            mode = "Debug" if is_debug else "Demo"
            self._render_simple_message(f"{mode} Env Data Missing!", VisConfig.RED)

    def _render_initializing_screen(self, status_message: str = "Initializing..."):
        """Renders a simple initializing message."""
        self._render_simple_message(status_message, VisConfig.WHITE)

    def _render_error_screen(self, status_message: str):
        """Renders the error screen."""
        try:
            self.screen.fill((40, 0, 0))
            font_title = pygame.font.SysFont(None, 70)
            font_msg = pygame.font.SysFont(None, 30)
            if not font_title or not font_msg:
                font_title = pygame.font.Font(None, 70)
                font_msg = pygame.font.Font(None, 30)

            title_surf = font_title.render("APPLICATION ERROR", True, VisConfig.RED)
            msg_surf = font_msg.render(
                f"Status: {status_message}", True, VisConfig.YELLOW
            )
            exit_surf = font_msg.render(
                "Press ESC or close window to exit.", True, VisConfig.WHITE
            )

            title_rect = title_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() // 3)
            )
            msg_rect = msg_surf.get_rect(
                center=(self.screen.get_width() // 2, title_rect.bottom + 30)
            )
            exit_rect = exit_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() * 0.8)
            )

            self.screen.blit(title_surf, title_rect)
            self.screen.blit(msg_surf, msg_rect)
            self.screen.blit(exit_surf, exit_rect)

        except Exception as e:
            logger.error(f"Error rendering error screen itself: {e}")
            self._render_simple_message(f"Error State: {status_message}", VisConfig.RED)

    def _render_simple_message(self, message: str, color: Tuple[int, int, int]):
        """Renders a simple centered message."""
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)
            if not font:
                font = pygame.font.Font(None, 50)
            text_surf = font.render(message, True, color)
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
        except Exception as e:
            logger.error(f"Error rendering simple message '{message}': {e}")


File: ui\tooltips.py
import pygame
from typing import Tuple, Dict, Optional
from config import VisConfig


class TooltipRenderer:
    """Handles rendering of tooltips when hovering over specific UI elements."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.font_tooltip = self._init_font()
        self.hovered_stat_key: Optional[str] = None
        self.stat_rects: Dict[str, pygame.Rect] = {}  # Rects to check for hover
        self.tooltip_texts: Dict[str, str] = {}  # Text corresponding to each rect key

    def _init_font(self):
        """Initializes the font used for tooltips."""
        try:
            # Smaller font for tooltips
            return pygame.font.SysFont(None, 18)
        except Exception as e:
            print(f"Warning: SysFont error for tooltip font: {e}. Using default.")
            return pygame.font.Font(None, 18)

    def check_hover(self, mouse_pos: Tuple[int, int]):
        """Checks if the mouse is hovering over any registered stat rect."""
        self.hovered_stat_key = None
        # Iterate in reverse order of drawing to prioritize top elements
        for key, rect in reversed(self.stat_rects.items()):
            # Ensure rect is valid before checking collision
            if (
                rect
                and rect.width > 0
                and rect.height > 0
                and rect.collidepoint(mouse_pos)
            ):
                self.hovered_stat_key = key
                return  # Found one, stop checking

    def render_tooltip(self):
        """Renders the tooltip if a stat element is being hovered over. Does not flip display."""
        if not self.hovered_stat_key or self.hovered_stat_key not in self.tooltip_texts:
            return  # No active hover or no text defined for this key

        tooltip_text = self.tooltip_texts[self.hovered_stat_key]
        mouse_pos = pygame.mouse.get_pos()

        # --- Text Wrapping Logic ---
        lines = []
        max_width = 300  # Max tooltip width in pixels
        words = tooltip_text.split(" ")
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}" if current_line else word
            try:
                test_surf = self.font_tooltip.render(test_line, True, VisConfig.BLACK)
                if test_surf.get_width() <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            except Exception as e:
                print(f"Warning: Font render error during tooltip wrap: {e}")
                lines.append(current_line)  # Add what we had
                current_line = word  # Start new line
        lines.append(current_line)  # Add the last line

        # --- Render Wrapped Text ---
        line_surfs = []
        total_height = 0
        max_line_width = 0
        try:
            for line in lines:
                if not line:
                    continue  # Skip empty lines
                surf = self.font_tooltip.render(line, True, VisConfig.BLACK)
                line_surfs.append(surf)
                total_height += surf.get_height()
                max_line_width = max(max_line_width, surf.get_width())
        except Exception as e:
            print(f"Warning: Font render error creating tooltip surfaces: {e}")
            return  # Cannot render tooltip

        if not line_surfs:
            return  # No valid lines to render

        # --- Calculate Tooltip Rect and Draw ---
        padding = 5
        tooltip_rect = pygame.Rect(
            mouse_pos[0] + 15,  # Offset from cursor x
            mouse_pos[1] + 10,  # Offset from cursor y
            max_line_width + padding * 2,
            total_height + padding * 2,
        )

        # Clamp tooltip rect to stay within screen bounds
        tooltip_rect.clamp_ip(self.screen.get_rect())

        try:
            # Draw background and border
            pygame.draw.rect(
                self.screen, VisConfig.YELLOW, tooltip_rect, border_radius=3
            )
            pygame.draw.rect(
                self.screen, VisConfig.BLACK, tooltip_rect, 1, border_radius=3
            )

            # Draw text lines onto the screen
            current_y = tooltip_rect.y + padding
            for surf in line_surfs:
                self.screen.blit(surf, (tooltip_rect.x + padding, current_y))
                current_y += surf.get_height()
        except Exception as e:
            print(f"Warning: Error drawing tooltip background/text: {e}")

    def update_rects_and_texts(
        self, rects: Dict[str, pygame.Rect], texts: Dict[str, str]
    ):
        """Updates the dictionaries used for hover detection and text lookup. Called by UIRenderer."""
        self.stat_rects = rects
        self.tooltip_texts = texts


File: ui\__init__.py
from .renderer import UIRenderer
from .input_handler import InputHandler

__all__ = ["UIRenderer", "InputHandler"]


File: ui\demo_components\grid_renderer.py
# File: ui/demo_components/grid_renderer.py
import pygame
import math
import traceback
from typing import Tuple

from config import (
    VisConfig,
    EnvConfig,
    DemoConfig,
    RED,
    BLUE,
    WHITE,
    GRAY,
    BLACK,
)  # Added WHITE, GRAY, BLACK
from config.constants import LINE_CLEAR_FLASH_COLOR, GAME_OVER_FLASH_COLOR
from environment.game_state import GameState
from environment.triangle import Triangle
from ui.panels.game_area import GameAreaRenderer  # Import for base rendering


class DemoGridRenderer:
    """Renders the main game grid area for Demo/Debug mode."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,  # Use fonts/methods from GameAreaRenderer
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer  # Store reference
        self.overlay_font = self.game_area_renderer.fonts.get(
            "env_overlay", pygame.font.Font(None, 36)
        )
        self.invalid_placement_color = (0, 0, 0, 150)

    def calculate_game_area_rect(
        self,
        screen_width: int,
        screen_height: int,
        padding: int,
        hud_height: int,
        help_height: int,
        env_config: EnvConfig,
    ) -> Tuple[pygame.Rect, pygame.Rect]:
        """Calculates the main game area rectangle, maintaining aspect ratio."""
        max_game_h = screen_height - 2 * padding - hud_height - help_height
        max_game_w = screen_width - 2 * padding
        aspect_ratio = (env_config.COLS * 0.75 + 0.25) / max(1, env_config.ROWS)

        game_w = max_game_w
        game_h = game_w / aspect_ratio if aspect_ratio > 0 else max_game_h
        if game_h > max_game_h:
            game_h = max_game_h
            game_w = game_h * aspect_ratio

        game_w = math.floor(min(game_w, max_game_w))
        game_h = math.floor(min(game_h, max_game_h))
        game_x = (screen_width - game_w) // 2
        game_y = padding
        game_rect = pygame.Rect(game_x, game_y, game_w, game_h)
        clipped_game_rect = game_rect.clip(self.screen.get_rect())
        return game_rect, clipped_game_rect

    def render_game_area(
        self,
        demo_env: GameState,
        env_config: EnvConfig,
        clipped_game_rect: pygame.Rect,
        bg_color: Tuple[int, int, int],
        is_debug: bool,
    ):
        """Renders the central game grid and placement preview."""
        try:
            game_surf = self.screen.subsurface(clipped_game_rect)
            game_surf.fill(bg_color)

            # Calculate grid rendering parameters
            tri_cell_w, tri_cell_h = self.calculate_demo_triangle_size(
                clipped_game_rect.width, clipped_game_rect.height, env_config
            )
            if tri_cell_w > 0 and tri_cell_h > 0:
                grid_ox, grid_oy = self.calculate_grid_offset(
                    clipped_game_rect.width, clipped_game_rect.height, env_config
                )

                # Render the grid directly using GameState object
                self._render_demo_grid_from_gamestate(
                    game_surf, demo_env, tri_cell_w, tri_cell_h, grid_ox, grid_oy
                )

                # Render dragged shape if not in debug mode
                if not is_debug:
                    self._render_dragged_shape(
                        game_surf,
                        demo_env,
                        tri_cell_w,
                        tri_cell_h,
                        grid_ox,
                        grid_oy,
                        clipped_game_rect.topleft,
                    )

            # Render overlays (Game Over, Line Clear) - using HUD renderer's logic if needed
            if demo_env.is_over():
                self._render_demo_overlay_text(game_surf, "GAME OVER", RED)
            elif demo_env.is_line_clearing() and demo_env.last_line_clear_info:
                lines, tris, score = demo_env.last_line_clear_info
                line_str = "Line" if lines == 1 else "Lines"
                clear_msg = (
                    f"{lines} {line_str} Cleared! ({tris} Tris, +{score:.2f} pts)"
                )
                self._render_demo_overlay_text(game_surf, clear_msg, BLUE)

        except ValueError as e:
            print(f"Error subsurface demo game ({clipped_game_rect}): {e}")
            pygame.draw.rect(self.screen, RED, clipped_game_rect, 1)
        except Exception as render_e:
            print(f"Error rendering demo game area: {render_e}")
            traceback.print_exc()
            pygame.draw.rect(self.screen, RED, clipped_game_rect, 1)

    def _render_demo_grid_from_gamestate(
        self,
        surf: pygame.Surface,
        env: GameState,
        cell_w: int,
        cell_h: int,
        grid_offset_x: float,
        grid_offset_y: float,
    ):
        """Renders the grid directly from the GameState object."""
        grid = env.grid
        for r in range(grid.rows):
            for c in range(grid.cols):
                if not (
                    0 <= r < len(grid.triangles) and 0 <= c < len(grid.triangles[r])
                ):
                    continue
                t = grid.triangles[r][c]
                if t.is_death:
                    continue  # Don't draw death cells

                try:
                    pts = t.get_points(
                        ox=grid_offset_x, oy=grid_offset_y, cw=cell_w, ch=cell_h
                    )
                    color = self.vis_config.LIGHTG  # Default empty color
                    if t.is_occupied:
                        color = (
                            t.color if t.color else self.vis_config.RED
                        )  # Use shape color or fallback

                    # Highlight cleared triangles
                    is_highlighted = (
                        env.is_highlighting_cleared()
                        and (r, c) in env.cleared_triangles_coords
                    )
                    if is_highlighted:
                        highlight_color = self.vis_config.LINE_CLEAR_HIGHLIGHT_COLOR
                        # Draw slightly larger background polygon
                        center_x = sum(p[0] for p in pts) / 3
                        center_y = sum(p[1] for p in pts) / 3
                        scale_factor = 1.2
                        highlight_pts = [
                            (
                                center_x + (p[0] - center_x) * scale_factor,
                                center_y + (p[1] - center_y) * scale_factor,
                            )
                            for p in pts
                        ]
                        pygame.draw.polygon(
                            surf,
                            (
                                highlight_color[0],
                                highlight_color[1],
                                highlight_color[2],
                            ),
                            highlight_pts,
                        )  # Use RGB for solid bg

                    # Draw the main triangle
                    pygame.draw.polygon(surf, color, pts)
                    # Draw border
                    pygame.draw.polygon(surf, self.vis_config.GRAY, pts, 1)

                except Exception as tri_err:
                    # logger.info(f"Error drawing triangle ({r},{c}): {tri_err}") # Use debug level
                    pass  # Ignore errors for single triangles

    def calculate_demo_triangle_size(
        self, surf_w: int, surf_h: int, env_config: EnvConfig
    ) -> Tuple[int, int]:
        """Calculates the size of triangles for rendering within the demo area."""
        padding = self.vis_config.ENV_GRID_PADDING
        drawable_w = max(1, surf_w - 2 * padding)
        drawable_h = max(1, surf_h - 2 * padding)
        grid_rows = env_config.ROWS
        grid_cols_eff_width = env_config.COLS * 0.75 + 0.25
        if grid_rows <= 0 or grid_cols_eff_width <= 0:
            return 0, 0

        scale_w = drawable_w / grid_cols_eff_width
        scale_h = drawable_h / grid_rows
        final_scale = min(scale_w, scale_h)
        if final_scale <= 0:
            return 0, 0
        tri_cell_size = max(1, int(final_scale))
        return tri_cell_size, tri_cell_size

    def calculate_grid_offset(
        self, surf_w: int, surf_h: int, env_config: EnvConfig
    ) -> Tuple[float, float]:
        """Calculates the top-left offset for centering the grid rendering."""
        padding = self.vis_config.ENV_GRID_PADDING
        drawable_w = max(1, surf_w - 2 * padding)
        drawable_h = max(1, surf_h - 2 * padding)
        grid_rows = env_config.ROWS
        grid_cols_eff_width = env_config.COLS * 0.75 + 0.25
        if grid_rows <= 0 or grid_cols_eff_width <= 0:
            return float(padding), float(padding)

        scale_w = drawable_w / grid_cols_eff_width
        scale_h = drawable_h / grid_rows
        final_scale = min(scale_w, scale_h)
        final_grid_pixel_w = max(1, grid_cols_eff_width * final_scale)
        final_grid_pixel_h = max(1, grid_rows * final_scale)
        grid_ox = padding + (drawable_w - final_grid_pixel_w) / 2
        grid_oy = padding + (drawable_h - final_grid_pixel_h) / 2
        return grid_ox, grid_oy

    def _render_dragged_shape(
        self,
        surf: pygame.Surface,
        env: GameState,
        cell_w: int,
        cell_h: int,
        grid_offset_x: float,
        grid_offset_y: float,
        game_area_offset: Tuple[int, int],
    ):
        """Renders the shape being dragged, either snapped or following the mouse."""
        if cell_w <= 0 or cell_h <= 0:
            return
        dragged_shape, snapped_pos = env.get_dragged_shape_info()
        if dragged_shape is None:
            return

        is_valid_placement = snapped_pos is not None
        preview_alpha = 150
        if is_valid_placement:
            shape_rgb = dragged_shape.color
            preview_color_rgba = (
                shape_rgb[0],
                shape_rgb[1],
                shape_rgb[2],
                preview_alpha,
            )
        else:
            preview_color_rgba = (50, 50, 50, 100)  # Invalid placement color

        temp_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))

        ref_x, ref_y = 0, 0
        if snapped_pos:
            snap_r, snap_c = snapped_pos
            ref_x = grid_offset_x + snap_c * (cell_w * 0.75)
            ref_y = grid_offset_y + snap_r * cell_h
        else:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            mouse_x -= game_area_offset[0]
            mouse_y -= game_area_offset[1]
            min_r, min_c, max_r, max_c = dragged_shape.bbox()
            shape_h_cells = max_r - min_r + 1
            shape_w_cells_eff = (max_c - min_c + 1) * 0.75 + 0.25
            shape_pixel_w = shape_w_cells_eff * cell_w
            shape_pixel_h = shape_h_cells * cell_h
            ref_x = mouse_x - (shape_pixel_w / 2) - (min_c * cell_w * 0.75)
            ref_y = mouse_y - (shape_pixel_h / 2) - (min_r * cell_h)

        for dr, dc, is_up in dragged_shape.triangles:
            tri_x = ref_x + dc * (cell_w * 0.75)
            tri_y = ref_y + dr * cell_h
            temp_tri = Triangle(0, 0, is_up)
            try:
                points = temp_tri.get_points(ox=tri_x, oy=tri_y, cw=cell_w, ch=cell_h)
                pygame.draw.polygon(temp_surface, preview_color_rgba, points)
                # Optionally draw border for preview
                # pygame.draw.polygon(temp_surface, (200, 200, 200, 100), points, 1)
            except Exception:
                pass

        surf.blit(temp_surface, (0, 0))

    def _render_demo_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders centered overlay text (e.g., GAME OVER)."""
        try:
            font = self.overlay_font  # Use the font initialized for overlays
            if not font:
                return

            max_w = surf.get_width() * 0.9
            original_size = font.get_height()
            current_size = original_size

            surf_txt = font.render(text, True, WHITE)
            while surf_txt.get_width() > max_w and current_size > 8:
                current_size -= 2
                try:
                    font = pygame.font.SysFont(None, current_size)
                except:
                    font = pygame.font.Font(None, current_size)
                surf_txt = font.render(text, True, WHITE)

            bg_rgba = (color[0] // 2, color[1] // 2, color[2] // 2, 220)
            surf_bg = font.render(text, True, WHITE, bg_rgba)
            rect = surf_bg.get_rect(center=surf.get_rect().center)
            surf.blit(surf_bg, rect)
        except Exception as e:
            print(f"Error rendering overlay '{text}': {e}")


File: ui\demo_components\hud_renderer.py
# File: ui/demo_components/hud_renderer.py
import pygame
from typing import Tuple

from config import VisConfig, DemoConfig, RED, BLUE, WHITE, LIGHTG, DARK_RED
from config.constants import LINE_CLEAR_FLASH_COLOR, GAME_OVER_FLASH_COLOR
from environment.game_state import GameState
from ui.panels.game_area import GameAreaRenderer  # Import for fonts


class DemoHudRenderer:
    """Renders the HUD, help text, and overlays for Demo/Debug mode."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,  # Use fonts from GameAreaRenderer
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer  # Store reference
        self._init_demo_fonts()  # Initialize specific demo fonts

    def _init_demo_fonts(self):
        """Initializes fonts used specifically in demo mode HUD/Help."""
        try:
            self.demo_hud_font = pygame.font.SysFont(
                None, self.demo_config.HUD_FONT_SIZE
            )
            self.demo_help_font = pygame.font.SysFont(
                None, self.demo_config.HELP_FONT_SIZE
            )
        except Exception as e:
            print(f"Warning: SysFont error for demo fonts: {e}. Using default.")
            self.demo_hud_font = pygame.font.Font(None, self.demo_config.HUD_FONT_SIZE)
            self.demo_help_font = pygame.font.Font(
                None, self.demo_config.HELP_FONT_SIZE
            )

    def determine_background_color(self, demo_env: GameState) -> Tuple[int, int, int]:
        """Determines the background color based on game state."""
        if demo_env.is_line_clearing():
            return LINE_CLEAR_FLASH_COLOR
        if demo_env.is_game_over_flashing():
            return GAME_OVER_FLASH_COLOR
        if demo_env.is_over():
            return DARK_RED
        if demo_env.is_frozen():
            return (30, 30, 100)
        return self.demo_config.BACKGROUND_COLOR

    def render_hud(
        self, demo_env: GameState, screen_width: int, hud_y: int, is_debug: bool
    ):
        """Renders the score and triangles cleared HUD."""
        if is_debug:
            hud_text = (
                f"DEBUG MODE | Tris Cleared: {demo_env.triangles_cleared_this_episode}"
            )
        else:
            hud_text = f"Score: {demo_env.game_score} | Tris Cleared: {demo_env.triangles_cleared_this_episode}"
        try:
            hud_surf = self.demo_hud_font.render(hud_text, True, WHITE)
            hud_rect = hud_surf.get_rect(midtop=(screen_width // 2, hud_y))
            self.screen.blit(hud_surf, hud_rect)
        except Exception as e:
            print(f"HUD render error: {e}")

    def render_help_text(self, screen_width: int, screen_height: int, is_debug: bool):
        """Renders the control help text at the bottom."""
        if is_debug:
            help_text = self.demo_config.DEBUG_HELP_TEXT
        else:
            help_text = (
                "[Click Preview]=Select/Deselect | [Click Grid]=Place | [ESC]=Exit"
            )
        try:
            help_surf = self.demo_help_font.render(help_text, True, LIGHTG)
            help_rect = help_surf.get_rect(
                centerx=screen_width // 2, bottom=screen_height - 10
            )
            self.screen.blit(help_surf, help_rect)
        except Exception as e:
            print(f"Help render error: {e}")

    def render_demo_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders centered overlay text (e.g., GAME OVER)."""
        # This method might be better placed here if it's specific to demo overlays
        # Or keep it in the grid renderer if it's used there too.
        # For now, assuming it's called by the grid renderer.
        # If called directly by DemoRenderer, it needs access to the game_surf.
        pass  # Logic moved to GridRenderer for now, can be moved back if needed.

    def render_too_small_message(self, text: str, area_rect: pygame.Rect):
        """Renders a message indicating the area is too small."""
        try:
            font = self.game_area_renderer.fonts.get("ui", pygame.font.Font(None, 24))
            err_surf = font.render(
                text, True, LIGHTG
            )  # Use LIGHTG for less alarming message
            target_rect = err_surf.get_rect(center=area_rect.center)
            self.screen.blit(err_surf, target_rect)
        except Exception as e:
            print(f"Error rendering 'too small' message: {e}")


File: ui\demo_components\preview_renderer.py
# File: ui/demo_components/preview_renderer.py
import pygame
from typing import Tuple, Dict

from config import VisConfig, DemoConfig, RED, BLUE, GRAY
from environment.game_state import GameState
from ui.panels.game_area import GameAreaRenderer  # Import for base rendering


class DemoPreviewRenderer:
    """Renders the shape preview area in Demo mode."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,  # Use fonts/methods from GameAreaRenderer
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer  # Store reference
        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

    def render_shape_previews_area(
        self,
        demo_env: GameState,
        screen_width: int,
        clipped_game_rect: pygame.Rect,
        padding: int,
    ) -> Dict[int, pygame.Rect]:
        """Renders the shape preview area. Returns dict of preview rects."""
        self.shape_preview_rects.clear()  # Clear previous rects
        preview_area_w = min(150, screen_width - clipped_game_rect.right - padding // 2)
        if preview_area_w <= 20:
            return self.shape_preview_rects

        preview_area_rect = pygame.Rect(
            clipped_game_rect.right + padding // 2,
            clipped_game_rect.top,
            preview_area_w,
            clipped_game_rect.height,
        )
        clipped_preview_area_rect = preview_area_rect.clip(self.screen.get_rect())
        if (
            clipped_preview_area_rect.width <= 0
            or clipped_preview_area_rect.height <= 0
        ):
            return self.shape_preview_rects

        try:
            preview_area_surf = self.screen.subsurface(clipped_preview_area_rect)
            self.shape_preview_rects = self._render_demo_shape_previews(
                preview_area_surf, demo_env, preview_area_rect.topleft
            )
        except ValueError as e:
            print(f"Error subsurface demo shape preview area: {e}")
            pygame.draw.rect(self.screen, RED, clipped_preview_area_rect, 1)
        except Exception as e:
            print(f"Error rendering demo shape previews: {e}")
            traceback.print_exc()
        return self.shape_preview_rects

    def _render_demo_shape_previews(
        self, surf: pygame.Surface, env: GameState, area_topleft: Tuple[int, int]
    ) -> Dict[int, pygame.Rect]:
        """Renders the small previews of available shapes. Returns dict of screen rects."""
        calculated_rects: Dict[int, pygame.Rect] = {}
        surf.fill((25, 25, 25))
        all_slots = env.shapes
        selected_idx = env.demo_selected_shape_idx
        dragged_idx = env.demo_dragged_shape_idx

        num_slots = env.env_config.NUM_SHAPE_SLOTS
        surf_w, surf_h = surf.get_size()
        preview_padding = 5

        if num_slots <= 0:
            return calculated_rects

        preview_h = max(20, (surf_h - (num_slots + 1) * preview_padding) / num_slots)
        preview_w = max(20, surf_w - 2 * preview_padding)
        current_preview_y = preview_padding

        for i in range(num_slots):
            shape_in_slot = all_slots[i] if i < len(all_slots) else None
            preview_rect_local = pygame.Rect(
                preview_padding, current_preview_y, preview_w, preview_h
            )
            preview_rect_screen = preview_rect_local.move(area_topleft)
            calculated_rects[i] = preview_rect_screen

            clipped_preview_rect = preview_rect_local.clip(surf.get_rect())
            if clipped_preview_rect.width <= 0 or clipped_preview_rect.height <= 0:
                current_preview_y += preview_h + preview_padding
                continue

            bg_color = (40, 40, 40)
            border_color = GRAY
            border_width = 1
            if i == selected_idx and shape_in_slot is not None:
                border_color = self.demo_config.SELECTED_SHAPE_HIGHLIGHT_COLOR
                border_width = 3
            elif i == dragged_idx:
                border_color = (100, 100, 255)
                border_width = 2

            pygame.draw.rect(surf, bg_color, clipped_preview_rect, border_radius=3)
            pygame.draw.rect(
                surf, border_color, clipped_preview_rect, border_width, border_radius=3
            )

            if shape_in_slot is not None:
                self._render_single_shape_in_preview_box(
                    surf, shape_in_slot, preview_rect_local, clipped_preview_rect
                )

            current_preview_y += preview_h + preview_padding
        return calculated_rects

    def _render_single_shape_in_preview_box(
        self,
        surf: pygame.Surface,
        shape_obj,
        preview_rect: pygame.Rect,
        clipped_preview_rect: pygame.Rect,
    ):
        """Renders a single shape scaled to fit within its preview box."""
        try:
            inner_padding = 2
            shape_render_area_rect = pygame.Rect(
                inner_padding,
                inner_padding,
                clipped_preview_rect.width - 2 * inner_padding,
                clipped_preview_rect.height - 2 * inner_padding,
            )
            if shape_render_area_rect.width <= 0 or shape_render_area_rect.height <= 0:
                return

            temp_surf = pygame.Surface(shape_render_area_rect.size, pygame.SRCALPHA)
            temp_surf.fill((0, 0, 0, 0))

            min_r, min_c, max_r, max_c = shape_obj.bbox()
            shape_h_cells = max(1, max_r - min_r + 1)
            shape_w_cells_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)
            scale_h = shape_render_area_rect.height / shape_h_cells
            scale_w = shape_render_area_rect.width / shape_w_cells_eff
            cell_size = max(1, min(scale_h, scale_w))

            # Use the GameAreaRenderer's shape rendering logic
            self.game_area_renderer._render_single_shape(
                temp_surf, shape_obj, int(cell_size)
            )

            surf.blit(
                temp_surf, shape_render_area_rect.move(preview_rect.topleft).topleft
            )

        except ValueError as sub_err:
            print(f"Error subsurface shape preview: {sub_err}")
            pygame.draw.rect(surf, RED, clipped_preview_rect, 1)
        except Exception as e:
            print(f"Error rendering demo shape preview: {e}")
            pygame.draw.rect(surf, RED, clipped_preview_rect, 1)

    def get_shape_preview_rects(self) -> Dict[int, pygame.Rect]:
        """Returns the dictionary of screen-relative shape preview rects."""
        return self.shape_preview_rects.copy()


File: ui\mcts_visualizer\node_renderer.py
# File: ui/mcts_visualizer/node_renderer.py
import pygame
import math
from typing import Tuple, Optional, TYPE_CHECKING

from config import (
    VisConfig,
    WHITE,
    BLACK,
    RED,
    BLUE,
    YELLOW,
    GRAY,
    LIGHTG,
    CYAN,
    ORANGE,
)
from config.constants import (
    MCTS_NODE_WIN_COLOR,
    MCTS_NODE_LOSS_COLOR,
    MCTS_NODE_NEUTRAL_COLOR,
    MCTS_NODE_BORDER_COLOR,
    MCTS_NODE_SELECTED_BORDER_COLOR,
    MCTS_NODE_TEXT_COLOR,
    MCTS_NODE_PRIOR_COLOR,
    MCTS_NODE_SCORE_COLOR,
    MCTS_MINI_GRID_BG_COLOR,
    MCTS_MINI_GRID_LINE_COLOR,
    MCTS_MINI_GRID_OCCUPIED_COLOR,
    MCTS_EDGE_COLOR,
)
from mcts.node import MCTSNode

if TYPE_CHECKING:
    from ui.panels.game_area import GameAreaRenderer  # For type hinting


class MCTSNodeRenderer:
    """Renders a single MCTS node for visualization, including a mini-grid."""

    BASE_NODE_RADIUS = 25  # Increased base radius
    BASE_NODE_BORDER_WIDTH = 1
    BASE_FONT_SIZE = 10  # Smaller base font for more info
    MIN_NODE_RADIUS = 8
    MAX_NODE_RADIUS = 80
    MIN_FONT_SIZE = 6
    MAX_FONT_SIZE = 14
    GRID_RENDER_THRESHOLD_RADIUS = 15  # Only render grid if node radius is above this

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.font: Optional[pygame.font.Font] = None
        self.game_area_renderer: Optional["GameAreaRenderer"] = (
            None  # To render mini-grid
        )

    def set_game_area_renderer(self, renderer: "GameAreaRenderer"):
        """Sets the reference to the GameAreaRenderer."""
        self.game_area_renderer = renderer

    def _get_scaled_font(self, zoom: float) -> pygame.font.Font:
        """Gets a font scaled based on the zoom level."""
        scaled_size = int(self.BASE_FONT_SIZE * math.sqrt(zoom))
        clamped_size = max(self.MIN_FONT_SIZE, min(self.MAX_FONT_SIZE, scaled_size))
        try:
            # Consider caching fonts if performance becomes an issue
            return pygame.font.SysFont(None, clamped_size)
        except Exception:
            return pygame.font.Font(None, clamped_size)

    def _render_mini_grid(self, node: MCTSNode, surface: pygame.Surface):
        """Renders the game state grid onto the provided surface."""
        if not self.game_area_renderer:
            pygame.draw.line(surface, RED, (0, 0), surface.get_size(), 1)
            return

        # Use a simplified version of GameAreaRenderer's grid rendering
        try:
            padding = 1  # Minimal padding inside the node
            drawable_w = max(1, surface.get_width() - 2 * padding)
            drawable_h = max(1, surface.get_height() - 2 * padding)
            env_config = node.game_state.env_config  # Get config from node's state
            grid_rows, grid_cols_eff_width = (
                env_config.ROWS,
                env_config.COLS * 0.75 + 0.25,
            )
            if grid_rows <= 0 or grid_cols_eff_width <= 0:
                return

            scale_w = drawable_w / grid_cols_eff_width
            scale_h = drawable_h / grid_rows
            final_scale = min(scale_w, scale_h)
            if final_scale <= 0:
                return

            tri_cell_w, tri_cell_h = max(1, final_scale), max(1, final_scale)
            final_grid_pixel_w = grid_cols_eff_width * final_scale
            final_grid_pixel_h = grid_rows * final_scale
            grid_ox = padding + (drawable_w - final_grid_pixel_w) / 2
            grid_oy = padding + (drawable_h - final_grid_pixel_h) / 2

            surface.fill(MCTS_MINI_GRID_BG_COLOR)  # Background for the grid area

            grid = node.game_state.grid
            for r in range(grid.rows):
                for c in range(grid.cols):
                    if not (
                        0 <= r < len(grid.triangles) and 0 <= c < len(grid.triangles[r])
                    ):
                        continue
                    t = grid.triangles[r][c]
                    if not t.is_death and hasattr(t, "get_points"):
                        try:
                            pts = t.get_points(
                                ox=grid_ox,
                                oy=grid_oy,
                                cw=int(tri_cell_w),
                                ch=int(tri_cell_h),
                            )
                            color = MCTS_MINI_GRID_BG_COLOR  # Default empty color
                            if t.is_occupied:
                                # Use a simple bright color for occupied cells in mini-grid
                                color = MCTS_MINI_GRID_OCCUPIED_COLOR
                            pygame.draw.polygon(surface, color, pts)
                            # Draw subtle grid lines
                            pygame.draw.polygon(
                                surface, MCTS_MINI_GRID_LINE_COLOR, pts, 1
                            )
                        except Exception:
                            pass  # Ignore drawing errors for single triangles
        except Exception as e:
            print(f"Error rendering mini-grid: {e}")
            pygame.draw.line(surface, RED, (0, 0), surface.get_size(), 1)

    def render(
        self,
        node: MCTSNode,
        pos: Tuple[int, int],
        zoom: float,
        is_selected: bool = False,
    ):
        """Draws the node circle, mini-grid, and info, scaled by zoom."""
        self.font = self._get_scaled_font(zoom)
        if not self.font:
            return

        scaled_radius = int(self.BASE_NODE_RADIUS * zoom)
        node_radius = max(
            self.MIN_NODE_RADIUS, min(self.MAX_NODE_RADIUS, scaled_radius)
        )
        border_width = max(1, int(self.BASE_NODE_BORDER_WIDTH * zoom))

        value = node.mean_action_value
        if value > 0.1:
            color = MCTS_NODE_WIN_COLOR
        elif value < -0.1:
            color = MCTS_NODE_LOSS_COLOR
        else:
            color = MCTS_NODE_NEUTRAL_COLOR

        # Create surface for the node content (grid + border)
        node_diameter = node_radius * 2
        node_surface = pygame.Surface((node_diameter, node_diameter), pygame.SRCALPHA)
        node_surface.fill((0, 0, 0, 0))  # Transparent background

        # Render mini-grid if node is large enough
        if node_radius >= self.GRID_RENDER_THRESHOLD_RADIUS:
            grid_surface = pygame.Surface(
                (node_diameter, node_diameter), pygame.SRCALPHA
            )
            self._render_mini_grid(node, grid_surface)
            # Clip the grid to a circle
            pygame.draw.circle(
                grid_surface,
                (255, 255, 255, 0),
                (node_radius, node_radius),
                node_radius,
            )  # Transparent circle mask
            grid_surface.set_colorkey(
                (255, 255, 255, 0)
            )  # Make transparent area the colorkey
            node_surface.blit(grid_surface, (0, 0))
        else:
            # Draw solid color if too small for grid
            pygame.draw.circle(
                node_surface, color, (node_radius, node_radius), node_radius
            )

        # Draw border
        border_color = (
            MCTS_NODE_SELECTED_BORDER_COLOR if is_selected else MCTS_NODE_BORDER_COLOR
        )
        pygame.draw.circle(
            node_surface,
            border_color,
            (node_radius, node_radius),
            node_radius,
            border_width,
        )

        # Blit the node surface onto the main screen
        node_rect = node_surface.get_rect(center=pos)
        self.screen.blit(node_surface, node_rect)

        # Render text info below the node if radius is sufficient
        if node_radius > 10:
            visits_str = f"N:{node.visit_count}"
            value_str = f"Q:{value:.2f}"
            prior_str = f"P:{node.prior:.2f}"
            score_str = f"S:{node.game_state.game_score}"

            text_y_offset = node_rect.bottom + 2  # Start text below node rect
            line_height = self.font.get_linesize()

            visits_surf = self.font.render(visits_str, True, MCTS_NODE_TEXT_COLOR)
            value_surf = self.font.render(value_str, True, MCTS_NODE_TEXT_COLOR)
            prior_surf = self.font.render(prior_str, True, MCTS_NODE_PRIOR_COLOR)
            score_surf = self.font.render(score_str, True, MCTS_NODE_SCORE_COLOR)

            # Center text horizontally below the node
            self.screen.blit(
                visits_surf, visits_surf.get_rect(midtop=(pos[0], text_y_offset))
            )
            self.screen.blit(
                value_surf,
                value_surf.get_rect(midtop=(pos[0], text_y_offset + line_height)),
            )
            self.screen.blit(
                prior_surf,
                prior_surf.get_rect(midtop=(pos[0], text_y_offset + 2 * line_height)),
            )
            self.screen.blit(
                score_surf,
                score_surf.get_rect(midtop=(pos[0], text_y_offset + 3 * line_height)),
            )

    def draw_edge(
        self,
        parent_pos: Tuple[int, int],
        child_pos: Tuple[int, int],
        line_width: int = 1,
        color: Tuple[int, int, int] = MCTS_EDGE_COLOR,
    ):
        """Draws a line connecting parent and child nodes with variable width/color."""
        clamped_width = max(1, min(line_width, 5))
        pygame.draw.aaline(
            self.screen, color, parent_pos, child_pos
        )  # Use anti-aliased line


File: ui\mcts_visualizer\renderer.py
# File: ui/mcts_visualizer/renderer.py
import pygame
import math
from typing import Optional, Dict, Tuple, TYPE_CHECKING

from config import VisConfig, BLACK, WHITE, GRAY, YELLOW
from config.constants import (
    MCTS_INFO_TEXT_COLOR,
    MCTS_EDGE_COLOR,
    MCTS_EDGE_HIGHLIGHT_COLOR,
)
from mcts.node import MCTSNode
from .node_renderer import MCTSNodeRenderer
from .tree_layout import TreeLayout

if TYPE_CHECKING:
    from ui.panels.game_area import GameAreaRenderer


class MCTSVisualizer:
    """Renders the MCTS tree visualization with pan and zoom."""

    MIN_ZOOM = 0.1
    MAX_ZOOM = 5.0
    EDGE_HIGHLIGHT_THRESHOLD = 0.7  # Fraction of max visits to highlight edge

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        fonts: Dict[str, pygame.font.Font],
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = fonts
        self.node_renderer = MCTSNodeRenderer(screen, vis_config)
        self.info_font = fonts.get("ui", pygame.font.Font(None, 24))

        self.camera_offset_x = 0
        self.camera_offset_y = 0
        self.zoom_level = 1.0

        self.layout: Optional[TreeLayout] = None
        self.positions: Dict[MCTSNode, Tuple[int, int]] = {}

    def set_game_area_renderer(self, renderer: "GameAreaRenderer"):
        """Provides the GameAreaRenderer to the NodeRenderer for mini-grid drawing."""
        self.node_renderer.set_game_area_renderer(renderer)

    def reset_camera(self):
        """Resets camera pan and zoom."""
        self.camera_offset_x = 0
        self.camera_offset_y = 0
        self.zoom_level = 1.0
        print("MCTS Camera Reset")

    def pan_camera(self, delta_x: int, delta_y: int):
        """Pans the camera view."""
        self.camera_offset_x += delta_x
        self.camera_offset_y += delta_y

    def zoom_camera(self, factor: float, mouse_pos: Tuple[int, int]):
        """Zooms the camera view towards/away from the mouse position."""
        old_zoom = self.zoom_level
        self.zoom_level *= factor
        self.zoom_level = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.zoom_level))
        zoom_change = self.zoom_level / old_zoom

        world_mouse_x = (mouse_pos[0] - self.camera_offset_x) / old_zoom
        world_mouse_y = (mouse_pos[1] - self.camera_offset_y) / old_zoom

        new_offset_x = mouse_pos[0] - world_mouse_x * self.zoom_level
        new_offset_y = mouse_pos[1] - world_mouse_y * self.zoom_level

        self.camera_offset_x = new_offset_x
        self.camera_offset_y = new_offset_y

    def _world_to_screen(self, world_x: int, world_y: int) -> Tuple[int, int]:
        """Converts world coordinates (from layout) to screen coordinates."""
        screen_x = int(world_x * self.zoom_level + self.camera_offset_x)
        screen_y = int(world_y * self.zoom_level + self.camera_offset_y)
        return screen_x, screen_y

    def render(self, root_node: Optional[MCTSNode]):
        """Draws the MCTS tree and related info, applying camera transforms."""
        self.screen.fill(BLACK)

        if root_node is None:
            self._render_message("No MCTS data available.")
            return
        if not root_node.children and root_node.is_terminal:
            self._render_message("Root node is terminal.")
            pos = self._world_to_screen(self.screen.get_width() // 2, 100)
            self.node_renderer.render(root_node, pos, self.zoom_level, is_selected=True)
            self._render_info(root_node)
            return
        if not root_node.children and not root_node.is_expanded:
            self._render_message("MCTS Root not expanded (0 simulations?).")
            return

        if self.layout is None or self.layout.root != root_node:
            self.layout = TreeLayout(
                root_node, self.screen.get_width(), self.screen.get_height()
            )
            self.positions = self.layout.calculate_layout()

        max_child_visits = 0
        best_child_node: Optional[MCTSNode] = None
        if root_node.children:
            try:
                # Find best child based on visits for highlighting
                best_child_node = max(
                    root_node.children.values(), key=lambda n: n.visit_count
                )
                max_child_visits = best_child_node.visit_count
            except ValueError:  # Handle empty children dict case
                max_child_visits = 0
                best_child_node = None

        # Render edges first
        edges_to_render = []
        for node, world_pos in self.positions.items():
            if node.parent and node.parent in self.positions:
                parent_world_pos = self.positions[node.parent]
                parent_screen_pos = self._world_to_screen(*parent_world_pos)
                child_screen_pos = self._world_to_screen(*world_pos)

                line_width = 1
                edge_color = MCTS_EDGE_COLOR
                is_best_edge = False

                # Highlight edge from root to best child (based on visits)
                if (
                    node.parent == root_node
                    and node == best_child_node
                    and max_child_visits > 0
                ):
                    line_width = 3
                    edge_color = MCTS_EDGE_HIGHLIGHT_COLOR
                    is_best_edge = True

                edges_to_render.append(
                    (
                        (parent_screen_pos, child_screen_pos, line_width, edge_color),
                        is_best_edge,
                    )
                )

        # Sort edges to draw non-highlighted ones first
        edges_to_render.sort(
            key=lambda x: x[1]
        )  # False (non-best) comes before True (best)

        for edge_params, _ in edges_to_render:
            self.node_renderer.draw_edge(*edge_params)

        # Render nodes on top
        for node, world_pos in self.positions.items():
            screen_pos = self._world_to_screen(*world_pos)
            # Basic visibility check (culling) - expand bounds slightly
            render_radius = int(MCTSNodeRenderer.MAX_NODE_RADIUS * self.zoom_level)
            if (
                -render_radius < screen_pos[0] < self.screen.get_width() + render_radius
                and -render_radius
                < screen_pos[1]
                < self.screen.get_height() + render_radius
            ):
                self.node_renderer.render(
                    node, screen_pos, self.zoom_level, is_selected=(node == root_node)
                )

        self._render_info(root_node)

    def _render_message(self, message: str):
        """Displays a message centered on the screen."""
        if not self.info_font:
            return
        text_surf = self.info_font.render(message, True, WHITE)
        text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
        self.screen.blit(text_surf, text_rect)

    def _render_info(self, root_node: MCTSNode):
        """Displays information about the MCTS search and controls."""
        if not self.info_font:
            return
        sims = root_node.visit_count
        info_text = f"MCTS | Sims: {sims} | Zoom: {self.zoom_level:.2f}x | Drag=Pan | Scroll=Zoom | ESC=Exit"
        text_surf = self.info_font.render(info_text, True, MCTS_INFO_TEXT_COLOR)
        self.screen.blit(text_surf, (10, 10))

        if root_node.children:
            try:
                best_action_visits_node = max(
                    root_node.children.values(), key=lambda n: n.visit_count
                )
                best_action_visits = best_action_visits_node.action_taken

                best_action_q_node = max(
                    root_node.children.values(), key=lambda n: n.mean_action_value
                )
                best_action_q = best_action_q_node.action_taken

                best_action_text = f"Best Action (Visits): {best_action_visits} | Best Action (Q-Value): {best_action_q}"
                action_surf = self.info_font.render(best_action_text, True, YELLOW)
                self.screen.blit(action_surf, (10, 10 + self.info_font.get_linesize()))
            except ValueError:
                pass


File: ui\mcts_visualizer\tree_layout.py
# File: ui/mcts_visualizer/tree_layout.py
from typing import Dict, Tuple, Optional, List
from mcts.node import MCTSNode
import math


class TreeLayout:
    """Calculates positions for nodes in the MCTS tree for visualization."""

    HORIZONTAL_SPACING = 50
    VERTICAL_SPACING = 80
    SUBTREE_HORIZONTAL_PADDING = 10

    def __init__(self, root_node: MCTSNode, canvas_width: int, canvas_height: int):
        self.root = root_node
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.positions: Dict[MCTSNode, Tuple[int, int]] = {}
        self.subtree_widths: Dict[MCTSNode, int] = {}

    def calculate_layout(self) -> Dict[MCTSNode, Tuple[int, int]]:
        """Calculates and returns the positions for all nodes."""
        self._calculate_subtree_widths(self.root)
        self._calculate_positions(
            self.root, self.canvas_width // 2, 50
        )  # Start root at top-center
        return self.positions

    def _calculate_subtree_widths(self, node: MCTSNode):
        """Recursively calculates the horizontal space needed for each subtree."""
        if not node.children:
            self.subtree_widths[node] = self.HORIZONTAL_SPACING
            return

        total_width = 0
        for child in node.children.values():
            self._calculate_subtree_widths(child)
            total_width += self.subtree_widths[child]

        # Add padding between subtrees
        total_width += max(0, len(node.children) - 1) * self.SUBTREE_HORIZONTAL_PADDING
        # Ensure node itself has minimum spacing
        self.subtree_widths[node] = max(total_width, self.HORIZONTAL_SPACING)

    def _calculate_positions(self, node: MCTSNode, x: int, y: int):
        """Recursively calculates the (x, y) position for each node."""
        self.positions[node] = (x, y)

        if not node.children:
            return

        num_children = len(node.children)
        total_children_width = (
            self.subtree_widths[node] - self.HORIZONTAL_SPACING
        )  # Width excluding node itself
        current_x = x - total_children_width // 2

        child_list = list(node.children.values())  # Consistent order
        for i, child in enumerate(child_list):
            child_subtree_width = self.subtree_widths[child]
            child_x = current_x + child_subtree_width // 2
            child_y = y + self.VERTICAL_SPACING
            self._calculate_positions(child, child_x, child_y)
            current_x += child_subtree_width
            if i < num_children - 1:
                current_x += self.SUBTREE_HORIZONTAL_PADDING  # Add padding


File: ui\panels\game_area.py
# File: ui/panels/game_area.py
import pygame
import math
import traceback
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import copy

from config import (
    VisConfig,
    EnvConfig,
    TrainConfig,
    BLACK,
    BLUE,
    RED,
    GRAY,
    YELLOW,
    LIGHTG,
    WHITE,
    DARK_RED,
    DARK_GREEN,
    GREEN,
    LINE_CLEAR_FLASH_COLOR,
    GAME_OVER_FLASH_COLOR,
    DARK_GRAY,
)

# GameState not needed directly, use dicts
# from environment.game_state import GameState, StateType
from environment.shape import Shape  # Still useful for rendering shapes
from environment.triangle import Triangle  # Still useful for rendering triangles

logger = logging.getLogger(__name__)


class GameAreaRenderer:
    """Renders the right panel based on data received from the logic process."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()
        # Caches remain UI-side
        self.best_state_surface_cache: Optional[pygame.Surface] = None
        self.last_best_state_size: Tuple[int, int] = (0, 0)
        self.last_best_state_score: Optional[int] = None
        self.last_best_state_step: Optional[int] = None
        self.placeholder_surface_cache: Optional[pygame.Surface] = None
        self.last_placeholder_size: Tuple[int, int] = (0, 0)
        self.last_placeholder_message_key: str = ""

    def _init_fonts(self):
        """Initializes fonts used in the game area."""
        # ... (font init remains the same)
        fonts = {}
        font_configs = {
            "env_score": 16,
            "env_overlay": 24,
            "env_info": 14,
            "ui": 24,
            "placeholder": 30,
            "placeholder_detail": 22,
            "best_state_title": 20,
            "best_state_score": 18,
            "best_state_step": 16,
        }
        for key, size in font_configs.items():
            try:
                fonts[key] = pygame.font.SysFont(None, size)
            except Exception:
                try:
                    fonts[key] = pygame.font.Font(None, size)
                except Exception as e:
                    logger.error(f"ERROR: Font '{key}' failed to load: {e}")
                    fonts[key] = None
        # Fallbacks (ensure keys exist)
        if fonts.get("ui") is None:
            fonts["ui"] = pygame.font.Font(None, 24)
        if fonts.get("placeholder") is None:
            fonts["placeholder"] = pygame.font.Font(None, 30)
        if fonts.get("env_score") is None:
            fonts["env_score"] = pygame.font.Font(None, 16)
        if fonts.get("env_overlay") is None:
            fonts["env_overlay"] = pygame.font.Font(None, 24)
        if fonts.get("env_info") is None:
            fonts["env_info"] = pygame.font.Font(None, 14)
        if fonts.get("best_state_title") is None:
            fonts["best_state_title"] = pygame.font.Font(None, 20)
        if fonts.get("best_state_score") is None:
            fonts["best_state_score"] = pygame.font.Font(None, 18)
        if fonts.get("best_state_step") is None:
            fonts["best_state_step"] = pygame.font.Font(None, 16)
        return fonts

    def render(
        self,
        panel_width: int,
        panel_x_offset: int,
        worker_render_data: List[Optional[Dict[str, Any]]],  # Receives list of dicts
        num_envs: int,
        env_config: Dict[str, Any],  # Receives config values as dict
        best_game_state_data: Optional[Dict[str, Any]],  # Receives dict
    ):
        """Renders the grid of live self-play worker environments and best game found."""
        current_height = self.screen.get_height()
        ga_rect = pygame.Rect(panel_x_offset, 0, panel_width, current_height)
        if ga_rect.width <= 0 or ga_rect.height <= 0:
            return

        pygame.draw.rect(self.screen, VisConfig.DARK_GRAY, ga_rect)

        best_game_display_height = (
            int(current_height * 0.2) if best_game_state_data else 0
        )
        best_game_rect = pygame.Rect(
            ga_rect.x, ga_rect.y, ga_rect.width, best_game_display_height
        )
        env_grid_rect = pygame.Rect(
            ga_rect.x,
            ga_rect.y + best_game_display_height,
            ga_rect.width,
            ga_rect.height - best_game_display_height,
        )

        if best_game_display_height > 0 and best_game_rect.height > 20:
            self._render_best_game(best_game_rect, best_game_state_data, env_config)
        elif best_game_state_data and best_game_rect.height <= 20:
            logger.warning("Best game display area too small to render.")

        num_to_render = self.vis_config.NUM_ENVS_TO_RENDER
        actual_render_data = worker_render_data[:num_to_render]

        if not actual_render_data or not env_config:
            self._render_placeholder(env_grid_rect, "Waiting for Self-Play Workers...")
            return

        cols_env, rows_env, cell_w, cell_h_total = self._calculate_grid_layout(
            env_grid_rect, len(actual_render_data)
        )

        if cell_w > 10 and cell_h_total > 40:
            self._render_env_grid(
                actual_render_data,
                env_config,
                env_grid_rect,
                cols_env,
                rows_env,
                cell_w,
                cell_h_total,
            )
        else:
            self._render_too_small_message(env_grid_rect, cell_w, cell_h_total)

        if len(actual_render_data) < num_envs:
            self._render_render_limit_text(ga_rect, len(actual_render_data), num_envs)

    def _render_placeholder(self, area_rect: pygame.Rect, message: str):
        """Renders a simple placeholder message."""
        # ... (remains the same)
        pygame.draw.rect(self.screen, (60, 60, 70), area_rect, 1)
        font = self.fonts.get("placeholder")
        if font:
            text_surf = font.render(message, True, LIGHTG)
            self.screen.blit(text_surf, text_surf.get_rect(center=area_rect.center))

    def _calculate_grid_layout(
        self, available_rect: pygame.Rect, num_items: int
    ) -> Tuple[int, int, int, int]:
        """Calculates layout for multiple small environment grids."""
        # ... (remains the same)
        if available_rect.width <= 0 or available_rect.height <= 0 or num_items <= 0:
            return 0, 0, 0, 0
        aspect = available_rect.width / max(1, available_rect.height)
        cols = max(1, int(math.sqrt(num_items * aspect)))
        rows = max(1, math.ceil(num_items / cols))
        while cols * rows < num_items:
            if (cols + 1) / rows < aspect:
                cols += 1
            else:
                rows += 1
        while cols * (rows - 1) >= num_items and rows > 1:
            rows -= 1
        while (cols - 1) * rows >= num_items and cols > 1:
            cols -= 1
        sp = self.vis_config.ENV_SPACING
        cw_total = max(1, (available_rect.width - (cols + 1) * sp) // cols)
        ch_total = max(1, (available_rect.height - (rows + 1) * sp) // rows)
        return cols, rows, cw_total, ch_total

    def _render_env_grid(
        self,
        worker_render_data: List[Optional[Dict[str, Any]]],
        env_config: Dict[str, Any],  # Use dict
        grid_area_rect: pygame.Rect,
        cols: int,
        rows: int,
        cell_w: int,
        cell_h_total: int,
    ):
        """Renders the grid of small environment previews."""
        # ... (loop structure remains the same)
        env_idx = 0
        sp = self.vis_config.ENV_SPACING
        info_h, shapes_h = 12, 22
        grid_cell_h = max(1, cell_h_total - shapes_h - info_h - 2)

        for r in range(rows):
            for c in range(cols):
                if env_idx >= len(worker_render_data):
                    break
                env_x = grid_area_rect.x + sp * (c + 1) + c * cell_w
                env_y = grid_area_rect.y + sp * (r + 1) + r * cell_h_total
                env_rect_total = pygame.Rect(env_x, env_y, cell_w, cell_h_total)
                clip_rect_total = env_rect_total.clip(self.screen.get_rect())
                if clip_rect_total.width <= 0 or clip_rect_total.height <= 0:
                    env_idx += 1
                    continue

                render_data = worker_render_data[env_idx]
                grid_rect_local = pygame.Rect(0, 0, cell_w, grid_cell_h)
                shapes_rect_local = pygame.Rect(0, grid_cell_h + 1, cell_w, shapes_h)
                info_rect_local = pygame.Rect(
                    0, grid_cell_h + shapes_h + 2, cell_w, info_h
                )

                try:
                    cell_surf = self.screen.subsurface(clip_rect_total)
                    cell_surf.fill(VisConfig.DARK_GRAY)

                    # Check if render_data and state_dict exist
                    if (
                        render_data
                        and isinstance(render_data, dict)
                        and render_data.get("state_dict")
                    ):
                        env_state_dict = render_data["state_dict"]
                        env_stats = render_data.get("stats", {})

                        # Render Grid
                        if grid_rect_local.height > 0:
                            clipped_grid_rect_local = grid_rect_local.clip(
                                cell_surf.get_rect()
                            )
                            if (
                                clipped_grid_rect_local.width > 0
                                and clipped_grid_rect_local.height > 0
                            ):
                                grid_surf = cell_surf.subsurface(
                                    clipped_grid_rect_local
                                )
                                # Check if state_dict has grid data
                                if (
                                    isinstance(env_state_dict, dict)
                                    and "grid" in env_state_dict
                                ):
                                    grid_array = env_state_dict["grid"]
                                    if (
                                        isinstance(grid_array, np.ndarray)
                                        and grid_array.ndim == 3
                                        and grid_array.shape[0] >= 2
                                    ):
                                        occupancy_grid = grid_array[0]
                                        orientation_grid = grid_array[1]
                                        death_mask = env_state_dict.get(
                                            "death_mask",
                                            np.zeros_like(occupancy_grid, dtype=bool),
                                        )
                                        self._render_single_env_grid_from_arrays(
                                            grid_surf,
                                            occupancy_grid,
                                            orientation_grid,
                                            death_mask,
                                            env_config,
                                            env_stats,
                                        )
                                    else:
                                        self._render_placeholder(
                                            grid_surf, "Invalid Grid Data"
                                        )
                                else:
                                    self._render_placeholder(grid_surf, "No Grid Data")

                        # Render Shapes
                        if shapes_rect_local.height > 0:
                            clipped_shapes_rect_local = shapes_rect_local.clip(
                                cell_surf.get_rect()
                            )
                            if (
                                clipped_shapes_rect_local.width > 0
                                and clipped_shapes_rect_local.height > 0
                            ):
                                shapes_surf = cell_surf.subsurface(
                                    clipped_shapes_rect_local
                                )
                                available_shapes_data = env_stats.get(
                                    "available_shapes_data", []
                                )
                                self._render_env_shapes_from_data(
                                    shapes_surf, available_shapes_data, env_config
                                )

                        # Render Info
                        if info_rect_local.height > 0:
                            clipped_info_rect_local = info_rect_local.clip(
                                cell_surf.get_rect()
                            )
                            if (
                                clipped_info_rect_local.width > 0
                                and clipped_info_rect_local.height > 0
                            ):
                                info_surf = cell_surf.subsurface(
                                    clipped_info_rect_local
                                )
                                self._render_env_info(info_surf, env_idx, env_stats)
                    else:  # Placeholder rendering
                        # ... (placeholder rendering remains the same)
                        pygame.draw.rect(cell_surf, (20, 20, 20), cell_surf.get_rect())
                        pygame.draw.rect(
                            cell_surf, (60, 60, 60), cell_surf.get_rect(), 1
                        )
                        font = self.fonts.get("env_info")
                        if font:
                            status = (
                                render_data.get("stats", {}).get("status", "N/A")
                                if render_data
                                else "N/A"
                            )
                            text_surf = font.render(status, True, GRAY)
                            cell_surf.blit(
                                text_surf,
                                text_surf.get_rect(center=cell_surf.get_rect().center),
                            )

                    pygame.draw.rect(
                        cell_surf, VisConfig.LIGHTG, cell_surf.get_rect(), 1
                    )
                except ValueError as e:
                    logger.error(
                        f"Error creating subsurface for env cell {env_idx} ({clip_rect_total}): {e}"
                    )
                    pygame.draw.rect(self.screen, (50, 0, 50), clip_rect_total, 1)
                except Exception as e:
                    logger.error(
                        f"Error rendering env cell {env_idx}: {e}", exc_info=True
                    )
                    pygame.draw.rect(self.screen, (50, 0, 50), clip_rect_total, 1)
                env_idx += 1
            if env_idx >= len(worker_render_data):
                break

    def _render_single_env_grid_from_arrays(
        self,
        surf: pygame.Surface,
        occupancy_grid: np.ndarray,
        orientation_grid: np.ndarray,
        death_mask: np.ndarray,
        env_config: Dict[str, Any],
        stats: Dict[str, Any],
    ):
        """Renders the grid portion using occupancy, orientation, and death arrays."""
        # ... (grid rendering logic remains mostly the same, use env_config dict)
        cw, ch = surf.get_width(), surf.get_height()
        bg = VisConfig.GRAY
        surf.fill(bg)
        try:
            pad = self.vis_config.ENV_GRID_PADDING
            dw, dh = max(1, cw - 2 * pad), max(1, ch - 2 * pad)
            gr, gc = env_config.get("ROWS", 8), env_config.get("COLS", 15)
            gcw_eff = gc * 0.75 + 0.25
            if gr <= 0 or gcw_eff <= 0:
                return

            scale = min(dw / gcw_eff, dh / gr) if gr > 0 and gcw_eff > 0 else 0
            if scale <= 0:
                return

            tcw, tch = max(1, scale), max(1, scale)
            fpw, fph = gcw_eff * scale, gr * scale
            ox, oy = pad + (dw - fpw) / 2, pad + (dh - fph) / 2

            rows, cols = occupancy_grid.shape
            for r in range(rows):
                for c in range(cols):
                    if not (
                        0 <= r < death_mask.shape[0] and 0 <= c < death_mask.shape[1]
                    ):
                        continue
                    is_death = death_mask[r, c] > 0.5
                    is_occupied = occupancy_grid[r, c] > 0.5
                    expected_is_up = (r + c) % 2 == 0
                    temp_tri = Triangle(r, c, expected_is_up)
                    try:
                        pts = temp_tri.get_points(
                            ox=ox, oy=oy, cw=int(tcw), ch=int(tch)
                        )
                        color = VisConfig.LIGHTG
                        if is_death:
                            color = BLACK
                        elif is_occupied:
                            color = VisConfig.WHITE
                        pygame.draw.polygon(surf, color, pts)
                        if not is_death:
                            pygame.draw.polygon(surf, VisConfig.DARK_GRAY, pts, 1)
                    except Exception:
                        pass  # Ignore triangle errors
        except Exception as grid_err:
            logger.error(f"Error rendering grid from arrays: {grid_err}")
            pygame.draw.rect(surf, RED, surf.get_rect(), 2)

        # Score Overlay
        try:
            score_font = self.fonts.get("env_score")
            score = stats.get("game_score", "?")
            if score_font:
                score_surf = score_font.render(
                    f"Score: {score}", True, WHITE, (0, 0, 0, 180)
                )
                surf.blit(score_surf, (2, 2))
        except Exception as e:
            logger.error(f"Error rendering score overlay: {e}")

        # State Overlays
        status = stats.get("status", "")
        if "Error" in status:
            self._render_overlay_text(surf, "ERROR", RED)
        elif "Finished" in status:
            self._render_overlay_text(surf, "DONE", BLUE)
        elif "Stopped" in status:
            self._render_overlay_text(surf, "STOPPED", YELLOW)

    def _render_env_shapes_from_data(
        self,
        surf: pygame.Surface,
        available_shapes_data: List[Optional[Dict[str, Any]]],
        env_config: Dict[str, Any],
    ):
        """Renders the available shapes using triangle list and color data."""
        # ... (shape rendering logic remains mostly the same, use env_config dict)
        sw, sh = surf.get_width(), surf.get_height()
        if sw <= 0 or sh <= 0:
            return
        num_slots = env_config.get("NUM_SHAPE_SLOTS", 3)
        pad = 2
        total_pad_w = (num_slots + 1) * pad
        avail_w = sw - total_pad_w
        if avail_w <= 0:
            return
        w_per = avail_w / num_slots if num_slots > 0 else avail_w
        h_lim = sh - 2 * pad
        dim = max(5, int(min(w_per, h_lim)))
        start_x = pad + (sw - (num_slots * dim + (num_slots - 1) * pad)) / 2
        start_y = pad + (sh - dim) / 2
        curr_x = start_x

        for i in range(num_slots):
            shape_data = (
                available_shapes_data[i] if i < len(available_shapes_data) else None
            )
            rect = pygame.Rect(int(curr_x), int(start_y), dim, dim)
            if rect.right > sw - pad:
                break
            pygame.draw.rect(surf, (50, 50, 50), rect, border_radius=2)
            if shape_data and isinstance(shape_data, dict):
                try:
                    # Reconstruct temporary Shape object for rendering
                    temp_shape = Shape()
                    temp_shape.triangles = shape_data.get("triangles", [])
                    temp_shape.color = shape_data.get("color", WHITE)
                    self._render_single_shape_in_preview_box(surf, temp_shape, rect)
                except Exception as e:
                    logger.error(f"Error rendering shape preview from data: {e}")
                    pygame.draw.line(surf, RED, rect.topleft, rect.bottomright, 1)
            else:
                pygame.draw.line(surf, GRAY, rect.topleft, rect.bottomright, 1)
                pygame.draw.line(surf, GRAY, rect.topright, rect.bottomleft, 1)
            curr_x += dim + pad

    def _render_single_shape_in_preview_box(
        self, surf: pygame.Surface, shape_obj: Shape, preview_rect: pygame.Rect
    ):
        """Renders a single shape scaled to fit within its preview box."""
        # ... (remains the same)
        try:
            inner_padding = 2
            clipped_preview_rect = preview_rect.clip(surf.get_rect())
            if clipped_preview_rect.width <= 0 or clipped_preview_rect.height <= 0:
                return
            shape_surf = surf.subsurface(clipped_preview_rect)
            render_area_w = clipped_preview_rect.width - 2 * inner_padding
            render_area_h = clipped_preview_rect.height - 2 * inner_padding
            if render_area_w <= 0 or render_area_h <= 0:
                return
            temp_shape_surf = pygame.Surface(
                (render_area_w, render_area_h), pygame.SRCALPHA
            )
            temp_shape_surf.fill((0, 0, 0, 0))
            self._render_single_shape(
                temp_shape_surf, shape_obj, min(render_area_w, render_area_h)
            )
            shape_surf.blit(temp_shape_surf, (inner_padding, inner_padding))
        except ValueError as sub_err:
            logger.error(f"Error creating subsurface for shape preview: {sub_err}")
            pygame.draw.rect(surf, RED, preview_rect, 1)
        except Exception as e:
            logger.error(f"Error rendering single shape preview: {e}")
            pygame.draw.rect(surf, RED, preview_rect, 1)

    def _render_env_info(
        self, surf: pygame.Surface, worker_idx: int, stats: Dict[str, Any]
    ):
        """Renders worker ID and current game step with descriptive labels."""
        # ... (remains the same)
        font = self.fonts.get("env_info")
        if not font:
            return
        game_step = stats.get("game_steps", "?")
        info_text = f"Worker: {worker_idx} | Step: {game_step}"
        try:
            text_surf = font.render(info_text, True, LIGHTG)
            text_rect = text_surf.get_rect(centerx=surf.get_width() // 2, top=0)
            surf.blit(text_surf, text_rect)
        except Exception as e:
            logger.error(f"Error rendering env info: {e}")

    def _render_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders overlay text, scaling font if needed."""
        # ... (remains the same)
        try:
            font = self.fonts.get("env_overlay")
            if not font:
                return
            max_w = surf.get_width() * 0.9
            original_size = font.get_height()
            current_size = original_size
            surf_txt = font.render(text, True, WHITE)
            while surf_txt.get_width() > max_w and current_size > 8:
                current_size -= 2
                try:
                    font = pygame.font.SysFont(None, current_size)
                except:
                    font = pygame.font.Font(None, current_size)
                surf_txt = font.render(text, True, WHITE)
            bg_rgba = (color[0] // 2, color[1] // 2, color[2] // 2, 220)
            surf_bg = font.render(text, True, WHITE, bg_rgba)
            rect = surf_bg.get_rect(center=surf.get_rect().center)
            surf.blit(surf_bg, rect)
        except Exception as e:
            logger.error(f"Error rendering overlay '{text}': {e}")

    def _render_single_shape(self, surf: pygame.Surface, shape: Shape, target_dim: int):
        """Renders a single shape scaled to fit within the target surface."""
        # ... (remains the same)
        if not shape or not shape.triangles or target_dim <= 0:
            return
        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h_cells = max(1, max_r - min_r + 1)
        shape_w_cells_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)
        surf_w, surf_h = surf.get_size()
        scale_h = surf_h / shape_h_cells if shape_h_cells > 0 else surf_h
        scale_w = surf_w / shape_w_cells_eff if shape_w_cells_eff > 0 else surf_w
        scale = max(1, min(scale_h, scale_w))
        total_w_pixels = shape_w_cells_eff * scale
        total_h_pixels = shape_h_cells * scale
        ox = (surf_w - total_w_pixels) / 2 - min_c * (scale * 0.75)
        oy = (surf_h - total_h_pixels) / 2 - min_r * scale
        for dr, dc, is_up in shape.triangles:
            temp_tri = Triangle(0, 0, is_up)
            try:
                tri_x = ox + dc * (scale * 0.75)
                tri_y = oy + dr * scale
                pts = [
                    (int(p[0]), int(p[1]))
                    for p in temp_tri.get_points(
                        ox=tri_x, oy=tri_y, cw=int(scale), ch=int(scale)
                    )
                ]
                pygame.draw.polygon(surf, shape.color, pts)
                pygame.draw.polygon(surf, BLACK, pts, 1)
            except Exception:
                pass

    def _render_too_small_message(
        self, area_rect: pygame.Rect, cell_w: int, cell_h: int
    ):
        """Renders a message if the env cells are too small."""
        # ... (remains the same)
        font = self.fonts.get("ui")
        if font:
            surf = font.render(f"Envs Too Small ({cell_w}x{cell_h})", True, GRAY)
            self.screen.blit(surf, surf.get_rect(center=area_rect.center))

    def _render_render_limit_text(
        self, ga_rect: pygame.Rect, num_rendered: int, num_total: int
    ):
        """Renders text indicating not all envs are shown."""
        # ... (remains the same)
        font = self.fonts.get("ui")
        if font:
            surf = font.render(
                f"Rendering {num_rendered}/{num_total} Workers", True, YELLOW, BLACK
            )
            self.screen.blit(
                surf, surf.get_rect(bottomright=(ga_rect.right - 5, ga_rect.bottom - 5))
            )

    def _render_best_game(
        self,
        area_rect: pygame.Rect,
        best_game_data: Optional[Dict[str, Any]],
        env_config: Dict[str, Any],
    ):
        """Renders the best game state found so far."""
        # ... (remains mostly the same, uses dicts)
        pygame.draw.rect(self.screen, (20, 40, 20), area_rect)
        pygame.draw.rect(self.screen, GREEN, area_rect, 1)

        if not best_game_data or not env_config:
            self._render_placeholder(area_rect, "No Best Game Yet")
            return

        title_font = self.fonts.get("best_state_title")
        score_font = self.fonts.get("best_state_score")
        step_font = self.fonts.get("best_state_step")

        padding = 5
        text_area_width = 100
        grid_area_width = area_rect.width - text_area_width - 3 * padding
        grid_area_height = area_rect.height - 2 * padding
        text_area_rect = pygame.Rect(
            area_rect.left + padding,
            area_rect.top + padding,
            text_area_width,
            grid_area_height,
        )
        grid_area_rect = pygame.Rect(
            text_area_rect.right + padding,
            area_rect.top + padding,
            grid_area_width,
            grid_area_height,
        )

        if title_font and score_font and step_font:
            score = best_game_data.get("score", "N/A")
            step = best_game_data.get("step", "N/A")
            title_surf = title_font.render("Best Game", True, YELLOW)
            score_surf = score_font.render(f"Score: {score}", True, WHITE)
            step_surf = step_font.render(f"Step: {step}", True, LIGHTG)
            title_rect = title_surf.get_rect(
                midtop=(text_area_rect.centerx, text_area_rect.top + 2)
            )
            score_rect = score_surf.get_rect(
                midtop=(text_area_rect.centerx, title_rect.bottom + 4)
            )
            step_rect = step_surf.get_rect(
                midtop=(text_area_rect.centerx, score_rect.bottom + 2)
            )
            self.screen.blit(title_surf, title_rect)
            self.screen.blit(score_surf, score_rect)
            self.screen.blit(step_surf, step_rect)

        if grid_area_rect.width > 10 and grid_area_rect.height > 10:
            game_state_dict = best_game_data.get(
                "game_state_dict"
            )  # Get the state dict
            if game_state_dict and isinstance(game_state_dict, dict):
                try:
                    if "grid" in game_state_dict:
                        grid_state_array = game_state_dict["grid"]
                        death_mask = game_state_dict.get("death_mask", None)
                        if (
                            isinstance(grid_state_array, np.ndarray)
                            and grid_state_array.ndim == 3
                            and grid_state_array.shape[0] >= 2
                        ):
                            # Create death mask if missing
                            if (
                                death_mask is None
                                or death_mask.shape != grid_state_array[0].shape
                            ):
                                death_mask = np.zeros_like(
                                    grid_state_array[0], dtype=bool
                                )
                                logger.warning(
                                    "Best game state missing valid death_mask, using default."
                                )

                            self._render_grid_from_array(
                                grid_area_rect,
                                grid_state_array[0],
                                grid_state_array[1],
                                death_mask,
                                env_config,
                            )
                        else:
                            logger.warning(
                                f"Best game state 'grid' data invalid shape/type: {grid_state_array.shape if isinstance(grid_state_array, np.ndarray) else type(grid_state_array)}"
                            )
                            self._render_placeholder(
                                grid_area_rect, "Grid Data Invalid"
                            )
                    else:
                        logger.warning("Best game state dict missing 'grid' key.")
                        self._render_placeholder(grid_area_rect, "Grid Data Missing")
                except Exception as e:
                    logger.error(f"Error rendering best game grid: {e}", exc_info=True)
                    self._render_placeholder(grid_area_rect, "Grid Render Error")
            else:
                self._render_placeholder(grid_area_rect, "State Missing")

    def _render_grid_from_array(
        self,
        area_rect: pygame.Rect,
        occupancy_grid: np.ndarray,
        orientation_grid: np.ndarray,
        death_mask: np.ndarray,
        env_config: Dict[str, Any],
    ):
        """Simplified grid rendering directly from occupancy, orientation, and death arrays."""
        # ... (remains the same, uses env_config dict)
        try:
            clipped_area_rect = area_rect.clip(self.screen.get_rect())
            if clipped_area_rect.width <= 0 or clipped_area_rect.height <= 0:
                return
            grid_surf = self.screen.subsurface(clipped_area_rect)
            grid_surf.fill(VisConfig.GRAY)
            rows, cols = occupancy_grid.shape
            if rows == 0 or cols == 0:
                return
            pad = 1
            dw, dh = max(1, clipped_area_rect.width - 2 * pad), max(
                1, clipped_area_rect.height - 2 * pad
            )
            gc, gr = env_config.get("COLS", 15), env_config.get("ROWS", 8)
            gcw_eff = gc * 0.75 + 0.25
            if gr <= 0 or gcw_eff <= 0:
                return
            scale = min(dw / gcw_eff, dh / gr) if gr > 0 and gcw_eff > 0 else 0
            if scale <= 0:
                return
            tcw, tch = max(1, scale), max(1, scale)
            fpw, fph = gcw_eff * scale, gr * scale
            ox, oy = pad + (dw - fpw) / 2, pad + (dh - fph) / 2

            for r in range(rows):
                for c in range(cols):
                    if not (
                        0 <= r < death_mask.shape[0] and 0 <= c < death_mask.shape[1]
                    ):
                        continue
                    is_death = death_mask[r, c] > 0.5
                    is_occupied = occupancy_grid[r, c] > 0.5
                    expected_is_up = (r + c) % 2 == 0
                    temp_tri = Triangle(r, c, expected_is_up)
                    try:
                        pts = temp_tri.get_points(
                            ox=ox, oy=oy, cw=int(tcw), ch=int(tch)
                        )
                        color = VisConfig.LIGHTG
                        if is_death:
                            color = BLACK
                        elif is_occupied:
                            color = WHITE
                        pygame.draw.polygon(grid_surf, color, pts)
                        if not is_death:
                            pygame.draw.polygon(grid_surf, VisConfig.DARK_GRAY, pts, 1)
                    except Exception:
                        pass
        except ValueError as e:
            logger.error(
                f"Error creating subsurface for best grid ({clipped_area_rect}): {e}"
            )
            pygame.draw.rect(self.screen, RED, clipped_area_rect, 1)
        except Exception as e:
            logger.error(f"Error in _render_grid_from_array: {e}", exc_info=True)
            pygame.draw.rect(self.screen, RED, clipped_area_rect, 1)


File: ui\panels\left_panel.py
import pygame
from typing import Optional, Tuple, Dict, Any, List
import logging

from config import VisConfig
from ui.plotter import Plotter
from ui.input_handler import InputHandler  # Keep for type hint if needed
from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    PlotAreaRenderer,
    NotificationRenderer,
)
from app_state import AppState

logger = logging.getLogger(__name__)


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components based on provided data."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.input_handler: Optional[InputHandler] = None  # Reference set by UIRenderer

        # Initialize components
        self.button_status_renderer = ButtonStatusRenderer(self.screen, self.fonts)
        self.info_text_renderer = InfoTextRenderer(self.screen, self.fonts)
        self.notification_renderer = NotificationRenderer(self.screen, self.fonts)
        self.plot_area_renderer = PlotAreaRenderer(
            self.screen, self.fonts, self.plotter
        )

    def _init_fonts(self):
        """Initializes fonts used in the left panel."""
        # ... (font init remains the same)
        fonts = {}
        font_configs = {
            "ui": 24,
            "status": 28,
            "detail": 16,
            "resource": 16,
            "notification_label": 16,
            "notification": 18,
            "plot_placeholder": 20,
            "plot_title_values": 8,
            "mcts_stats_label": 18,
            "mcts_stats_value": 18,
        }
        for key, size in font_configs.items():
            try:
                fonts[key] = pygame.font.SysFont(None, size)
            except Exception:
                try:
                    fonts[key] = pygame.font.Font(None, size)
                except Exception as e:
                    logger.error(f"ERROR: Font '{key}' failed: {e}")
                    fonts[key] = None
        # Fallbacks
        if fonts.get("ui") is None:
            fonts["ui"] = pygame.font.Font(None, 24)
        if fonts.get("status") is None:
            fonts["status"] = pygame.font.Font(None, 28)
        if fonts.get("mcts_stats_label") is None:
            fonts["mcts_stats_label"] = pygame.font.Font(None, 18)
        if fonts.get("mcts_stats_value") is None:
            fonts["mcts_stats_value"] = pygame.font.Font(None, 18)
        return fonts

    def _get_background_color(self, status: str) -> Tuple[int, int, int]:
        """Determines background color based on status."""
        # ... (background color logic remains the same)
        status_color_map = {
            "Ready": (30, 30, 30),
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Debugging Grid": (40, 30, 40),
            "Initializing": (40, 40, 40),
            "Running AlphaZero": (30, 50, 30),
        }
        base_status = status.split(" (")[0] if "(" in status else status
        return status_color_map.get(base_status, (30, 30, 30))

    def render(self, panel_width: int, **render_data: Dict[str, Any]):
        """Renders the entire left panel based on the provided render_data dictionary."""
        current_height = self.screen.get_height()
        lp_rect = pygame.Rect(0, 0, panel_width, current_height)
        status = render_data.get("status", "")
        bg_color = self._get_background_color(status)
        pygame.draw.rect(self.screen, bg_color, lp_rect)

        current_y = 10
        # Define render order and estimated heights, pass data down
        render_order: List[Tuple[callable, int, Dict[str, Any]]] = [
            (
                self.button_status_renderer.render,
                60,
                {
                    k: render_data.get(k)
                    for k in [
                        "app_state",
                        "is_process_running",
                        "status",
                        "stats_summary",
                        "update_progress_details",
                    ]
                },
            ),
            (
                self.info_text_renderer.render,
                120,
                {
                    k: render_data.get(k)
                    for k in ["stats_summary", "agent_param_count", "worker_counts"]
                },
            ),
            (
                self.notification_renderer.render,
                70,
                {k: render_data.get(k) for k in ["stats_summary"]},
            ),
        ]

        # Render static components sequentially
        for render_func, fallback_height, func_kwargs in render_order:
            try:
                # Pass specific arguments required by each component
                if render_func == self.notification_renderer.render:
                    notification_rect = pygame.Rect(
                        10, current_y + 5, panel_width - 20, fallback_height
                    )
                    render_func(notification_rect, func_kwargs.get("stats_summary", {}))
                    next_y = notification_rect.bottom
                else:
                    next_y = render_func(
                        y_start=current_y + 5,
                        panel_width=panel_width,
                        **func_kwargs,  # Pass the specific kwargs for this function
                    )

                current_y = (
                    next_y
                    if isinstance(next_y, (int, float))
                    else current_y + fallback_height + 5
                )
            except Exception as e:
                logger.error(
                    f"Error rendering component {render_func.__name__}: {e}",
                    exc_info=True,
                )
                error_rect = pygame.Rect(
                    10, current_y + 5, panel_width - 20, fallback_height
                )
                pygame.draw.rect(self.screen, VisConfig.RED, error_rect, 1)
                current_y += fallback_height + 5

        # --- Render Plots Area ---
        app_state_str = render_data.get("app_state", AppState.UNKNOWN.value)
        should_render_plots = app_state_str == AppState.MAIN_MENU.value

        plot_y_start = current_y + 5
        try:
            self.plot_area_renderer.render(
                y_start=plot_y_start,
                panel_width=panel_width,
                screen_height=current_height,
                plot_data=render_data.get("plot_data", {}),
                status=status,
                render_enabled=should_render_plots,
            )
        except Exception as e:
            logger.error(f"Error in plot_area_renderer: {e}", exc_info=True)
            plot_area_height = current_height - plot_y_start - 10
            plot_area_width = panel_width - 20
            if plot_area_width > 10 and plot_area_height > 10:
                plot_area_rect = pygame.Rect(
                    10, plot_y_start, plot_area_width, plot_area_height
                )
                pygame.draw.rect(self.screen, (80, 0, 0), plot_area_rect, 2)


File: ui\panels\__init__.py
from .left_panel import LeftPanelRenderer
from .game_area import GameAreaRenderer

__all__ = ["LeftPanelRenderer", "GameAreaRenderer"]


File: ui\panels\left_panel_components\button_status_renderer.py
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


File: ui\panels\left_panel_components\info_text_renderer.py
# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple
import logging

from config import WHITE, LIGHTG, GRAY, YELLOW, GREEN

# Don't import config.general directly in UI process
# import config.general as config_general

logger = logging.getLogger(__name__)


class InfoTextRenderer:
    """Renders essential non-plotted information text based on provided data."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.detail_font = fonts.get("detail", pygame.font.Font(None, 16))
        # stats_summary_cache is not needed as data is passed in each frame

    def _get_network_description(self) -> str:
        """Builds a description string based on network components."""
        return "AlphaZero Neural Network"  # Keep simple for now

    # _render_key_value_line remains the same
    def _render_key_value_line(
        self,
        y_pos: int,
        panel_width: int,
        key: str,
        value: str,
        key_font: pygame.font.Font,
        value_font: pygame.font.Font,
        key_color=LIGHTG,
        value_color=WHITE,
    ) -> int:
        x_pos_key = 10
        x_pos_val_offset = 5
        try:
            key_surf = key_font.render(f"{key}:", True, key_color)
            key_rect = key_surf.get_rect(topleft=(x_pos_key, y_pos))
            self.screen.blit(key_surf, key_rect)
            value_surf = value_font.render(f"{value}", True, value_color)
            value_rect = value_surf.get_rect(
                topleft=(key_rect.right + x_pos_val_offset, y_pos)
            )
            clip_width = max(0, panel_width - value_rect.left - 10)
            blit_area = (
                pygame.Rect(0, 0, clip_width, value_rect.height)
                if value_rect.width > clip_width
                else None
            )
            self.screen.blit(value_surf, value_rect, area=blit_area)
            return max(key_rect.bottom, value_rect.bottom)
        except Exception as e:
            logger.error(f"Error rendering info line '{key}': {e}")
            return y_pos + key_font.get_linesize()

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],
        panel_width: int,
        agent_param_count: int,
        worker_counts: Dict[str, int],
    ) -> int:
        """Renders the info text block based on provided data. Returns next_y."""
        if not self.ui_font or not self.detail_font:
            logger.warning("Missing fonts for InfoTextRenderer.")
            return y_start

        current_y = y_start

        # --- General Info (Extract from stats_summary or passed args) ---
        device_type_str = stats_summary.get(
            "device", "N/A"
        )  # Expect device in summary now
        network_desc = self._get_network_description()
        param_str = (
            f"{agent_param_count / 1e6:.2f} M" if agent_param_count > 0 else "N/A"
        )
        start_time_unix = stats_summary.get("start_time", 0.0)
        start_time_str = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_unix))
            if start_time_unix > 0
            else "N/A"
        )
        sp_workers = worker_counts.get("SelfPlay", 0)
        tr_workers = worker_counts.get("Training", 0)
        worker_str = f"SP: {sp_workers}, TR: {tr_workers}"
        steps_sec = stats_summary.get("steps_per_second_avg", 0.0)
        steps_sec_str = f"{steps_sec:.1f}"

        general_info_lines = [
            ("Device", device_type_str),
            ("Network", network_desc),
            ("Params", param_str),
            ("Workers", worker_str),
            ("Run Started", start_time_str),
            ("Steps/Sec (Avg)", steps_sec_str),
        ]

        # Render lines using the helper function
        line_spacing = 2
        for key, value_str in general_info_lines:
            current_y = (
                self._render_key_value_line(
                    current_y, panel_width, key, value_str, self.ui_font, self.ui_font
                )
                + line_spacing
            )

        return int(current_y) + 5


File: ui\panels\left_panel_components\notification_renderer.py
# File: ui/panels/left_panel_components/notification_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple, Optional
from config import VisConfig, StatsConfig, WHITE, LIGHTG, GRAY, YELLOW, RED, GREEN
import numpy as np


class NotificationRenderer:
    """Renders the notification area based on provided data."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.label_font = fonts.get("notification_label", pygame.font.Font(None, 16))
        self.value_font = fonts.get("notification", pygame.font.Font(None, 18))

    def render(
        self, area_rect: pygame.Rect, stats_summary: Dict[str, Any]
    ) -> Dict[str, pygame.Rect]:
        """Renders the simplified notification content (e.g., total episodes) based on provided data."""
        stat_rects: Dict[str, pygame.Rect] = {}
        pygame.draw.rect(self.screen, (45, 45, 45), area_rect, border_radius=3)
        pygame.draw.rect(self.screen, LIGHTG, area_rect, 1, border_radius=3)
        stat_rects["Notification Area"] = area_rect

        if not self.value_font or not self.label_font:
            return stat_rects

        padding = 5
        line_height = self.value_font.get_linesize() + 2
        y = area_rect.top + padding

        # --- Display Total Episodes ---
        total_episodes = stats_summary.get("total_episodes", 0)
        label_surf = self.label_font.render("Total Episodes:", True, LIGHTG)
        value_surf = self.value_font.render(
            f"{total_episodes:,}".replace(",", "_"), True, WHITE
        )
        label_rect = label_surf.get_rect(topleft=(area_rect.left + padding, y))
        value_rect = value_surf.get_rect(topleft=(label_rect.right + 4, y))
        self.screen.blit(label_surf, label_rect)
        self.screen.blit(value_surf, value_rect)
        stat_rects["Total Episodes Info"] = label_rect.union(value_rect)
        y += line_height

        # --- Display Best Game Score ---
        best_score = stats_summary.get("best_game_score", -float("inf"))
        best_score_step = stats_summary.get("best_game_score_step", 0)
        best_score_str = "N/A"
        if best_score > -float("inf"):
            best_score_str = f"{best_score:.0f} (at step {best_score_step:,})".replace(
                ",", "_"
            )

        label_surf_bs = self.label_font.render("Best Score:", True, LIGHTG)
        value_surf_bs = self.value_font.render(
            best_score_str, True, GREEN if best_score > -float("inf") else WHITE
        )

        label_rect_bs = label_surf_bs.get_rect(topleft=(area_rect.left + padding, y))
        value_rect_bs = value_surf_bs.get_rect(topleft=(label_rect_bs.right + 4, y))
        self.screen.blit(label_surf_bs, label_rect_bs)
        self.screen.blit(value_surf_bs, value_rect_bs)
        stat_rects["Best Score Info"] = label_rect_bs.union(value_rect_bs)
        # y += line_height # Add if more lines needed

        # Best value rendering removed for simplification, can be added back similarly

        return stat_rects


File: ui\panels\left_panel_components\plot_area_renderer.py
# File: ui/panels/left_panel_components/plot_area_renderer.py
import pygame
from typing import Dict, Deque, Any, Optional, Tuple
import numpy as np
import logging

from config import (
    VisConfig,
    LIGHTG,
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    GRAY,
)
from ui.plotter import Plotter

logger = logging.getLogger(__name__)


class PlotAreaRenderer:
    """Renders the plot area using a Plotter instance based on provided data."""

    def __init__(
        self,
        screen: pygame.Surface,
        fonts: Dict[str, pygame.font.Font],
        plotter: Plotter,
    ):
        self.screen = screen
        self.fonts = fonts
        self.plotter = plotter
        self.placeholder_font = fonts.get(
            "plot_placeholder", pygame.font.Font(None, 20)
        )

    def render(
        self,
        y_start: int,
        panel_width: int,
        screen_height: int,
        plot_data: Dict[str, Deque],
        status: str,
        render_enabled: bool = True,
    ):
        """Renders the plot area, conditionally based on render_enabled."""
        # ... (Logic remains the same, it already uses passed-in data)
        plot_area_y_start = y_start
        plot_area_height = screen_height - plot_area_y_start - 10
        plot_area_width = panel_width - 20

        if plot_area_width <= 50 or plot_area_height <= 50:
            return

        plot_area_rect = pygame.Rect(
            10, plot_area_y_start, plot_area_width, plot_area_height
        )

        if not render_enabled:
            self._render_placeholder(plot_area_rect, "Plots Disabled")
            return

        plot_surface = self.plotter.get_cached_or_updated_plot(
            plot_data, plot_area_width, plot_area_height
        )

        if plot_surface:
            self.screen.blit(plot_surface, plot_area_rect.topleft)
        else:
            placeholder_text = "Waiting for plot data..."
            if status == "Error":
                placeholder_text = "Plotting disabled due to error."
            elif not plot_data or not any(plot_data.values()):
                placeholder_text = "No plot data yet..."
            self._render_placeholder(plot_area_rect, placeholder_text)

    def _render_placeholder(self, plot_area_rect: pygame.Rect, message: str):
        """Renders a placeholder message within the plot area."""
        # ... (remains the same)
        pygame.draw.rect(self.screen, (40, 40, 40), plot_area_rect, 1)
        if self.placeholder_font:
            placeholder_surf = self.placeholder_font.render(message, True, GRAY)
            placeholder_rect = placeholder_surf.get_rect(center=plot_area_rect.center)
            blit_pos = (
                max(plot_area_rect.left, placeholder_rect.left),
                max(plot_area_rect.top, placeholder_rect.top),
            )
            clip_area_rect = plot_area_rect.clip(placeholder_rect)
            blit_area = clip_area_rect.move(
                -placeholder_rect.left, -placeholder_rect.top
            )
            if blit_area.width > 0 and blit_area.height > 0:
                self.screen.blit(placeholder_surf, blit_pos, area=blit_area)
        else:  # Fallback cross
            pygame.draw.line(
                self.screen, GRAY, plot_area_rect.topleft, plot_area_rect.bottomright
            )
            pygame.draw.line(
                self.screen, GRAY, plot_area_rect.topright, plot_area_rect.bottomleft
            )


File: ui\panels\left_panel_components\__init__.py
from .button_status_renderer import ButtonStatusRenderer
from .notification_renderer import NotificationRenderer
from .info_text_renderer import InfoTextRenderer

from .plot_area_renderer import PlotAreaRenderer


__all__ = [
    "ButtonStatusRenderer",
    "NotificationRenderer",
    "InfoTextRenderer",
    "PlotAreaRenderer",
]


File: utils\helpers.py
import torch
import numpy as np
import random
import os
import pickle
import cloudpickle
import math  # Added for format_eta
from typing import Union, Any, Optional  # Added Optional for format_eta


def get_device() -> torch.device:
    """Gets the appropriate torch device (MPS, CUDA, or CPU)."""
    force_cpu = os.environ.get("FORCE_CPU", "false").lower() == "true"
    if force_cpu:
        print("Forcing CPU device based on environment variable.")
        return torch.device("cpu")

    # Check MPS first (for Macs) - This will be false on your PC
    if torch.backends.mps.is_available():
        device_str = "mps"
    # Check CUDA next (for NVIDIA GPUs) - This SHOULD become true
    elif torch.cuda.is_available():
        device_str = "cuda"
    # Fallback to CPU
    else:
        device_str = "cpu"

    print(f"Using device: {device_str.upper()}")
    if device_str == "cuda":
        # This line should execute once fixed
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    elif device_str == "mps":
        print("MPS device found on MacOS.")  # Won't execute on PC
    else:
        print(
            "No CUDA or MPS device found, falling back to CPU."
        )  # This is what's happening now

    return torch.device(device_str)


def set_random_seeds(seed: int = 42):
    """Sets random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Note: Setting deterministic algorithms can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Set random seeds to {seed}")


def ensure_numpy(data: Union[np.ndarray, list, tuple, torch.Tensor]) -> np.ndarray:
    """Ensures the input data is a numpy array with float32 type."""
    try:
        if isinstance(data, np.ndarray):
            if data.dtype != np.float32:
                return data.astype(np.float32)
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float32)
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=np.float32)
            if arr.dtype == np.object_:  # Indicates ragged array
                raise ValueError(
                    "Cannot convert ragged list/tuple to float32 numpy array."
                )
            return arr
        else:
            # Attempt conversion for single numbers or other types
            return np.array([data], dtype=np.float32)
    except (ValueError, TypeError, RuntimeError) as e:
        print(
            f"CRITICAL ERROR in ensure_numpy conversion: {e}. Input type: {type(data)}. Data (partial): {str(data)[:100]}"
        )
        raise ValueError(f"ensure_numpy failed: {e}") from e


def save_object(obj: Any, filepath: str):
    """Saves an arbitrary Python object to a file using cloudpickle."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            cloudpickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error saving object to {filepath}: {e}")
        raise e  # Re-raise after logging


def load_object(filepath: str) -> Any:
    """Loads a Python object from a file using cloudpickle."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found for loading: {filepath}")
    try:
        with open(filepath, "rb") as f:
            obj = cloudpickle.load(f)
        return obj
    except Exception as e:
        print(f"Error loading object from {filepath}: {e}")
        raise e  # Re-raise after logging


def format_eta(seconds: Optional[float]) -> str:
    """Formats seconds into a human-readable HH:MM:SS or MM:SS string."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "N/A"
    if seconds > 3600 * 24 * 30:  # Cap at roughly a month
        return ">1 month"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"


File: utils\init_checks.py
# File: utils/init_checks.py
import sys
import traceback
import numpy as np

from environment.game_state import GameState


def run_pre_checks() -> bool:
    """Performs basic checks on GameState and configuration compatibility."""
    try:
        from config import EnvConfig
    except ImportError as e:
        print(f"FATAL ERROR: Could not import EnvConfig during pre-check: {e}")
        print(
            "This might indicate an issue with the config package structure or an ongoing import cycle."
        )
        sys.exit(1)

    print("--- Pre-Run Checks ---")
    try:
        print("Checking GameState and Configuration Compatibility...")
        env_config_instance = EnvConfig()

        gs_test = GameState()
        gs_test.reset()
        s_test_dict = gs_test.get_state()

        if not isinstance(s_test_dict, dict):
            raise TypeError(
                f"GameState.get_state() should return a dict, but got {type(s_test_dict)}"
            )
        print("GameState state type check PASSED (returned dict).")

        if "grid" not in s_test_dict:
            raise KeyError("State dictionary missing 'grid' key.")
        grid_state = s_test_dict["grid"]
        expected_grid_shape = env_config_instance.GRID_STATE_SHAPE
        if not isinstance(grid_state, np.ndarray):
            raise TypeError(
                f"State 'grid' component should be numpy array, but got {type(grid_state)}"
            )
        if grid_state.shape != expected_grid_shape:
            raise ValueError(
                f"State 'grid' shape mismatch! GameState:{grid_state.shape}, EnvConfig:{expected_grid_shape}"
            )
        print(f"GameState 'grid' state shape check PASSED (Shape: {grid_state.shape}).")

        if "shapes" not in s_test_dict:
            raise KeyError("State dictionary missing 'shapes' key.")
        shape_state = s_test_dict["shapes"]
        expected_shape_shape = (env_config_instance.SHAPE_STATE_DIM,)
        if not isinstance(shape_state, np.ndarray):
            raise TypeError(
                f"State 'shapes' component should be numpy array, but got {type(shape_state)}"
            )
        if shape_state.shape != expected_shape_shape:
            raise ValueError(
                f"State 'shapes' feature shape mismatch! GameState:{shape_state.shape}, EnvConfig:{expected_shape_shape}"
            )
        print(
            f"GameState 'shapes' feature shape check PASSED (Shape: {shape_state.shape})."
        )

        if "shape_availability" not in s_test_dict:
            raise KeyError("State dictionary missing 'shape_availability' key.")
        availability_state = s_test_dict["shape_availability"]
        expected_availability_shape = (env_config_instance.SHAPE_AVAILABILITY_DIM,)
        if not isinstance(availability_state, np.ndarray):
            raise TypeError(
                f"State 'shape_availability' component should be numpy array, but got {type(availability_state)}"
            )
        if availability_state.shape != expected_availability_shape:
            raise ValueError(
                f"State 'shape_availability' shape mismatch! GameState:{availability_state.shape}, EnvConfig:{expected_availability_shape}"
            )
        print(
            f"GameState 'shape_availability' state shape check PASSED (Shape: {availability_state.shape})."
        )

        if "explicit_features" not in s_test_dict:
            raise KeyError("State dictionary missing 'explicit_features' key.")
        explicit_features_state = s_test_dict["explicit_features"]
        expected_explicit_features_shape = (env_config_instance.EXPLICIT_FEATURES_DIM,)
        if not isinstance(explicit_features_state, np.ndarray):
            raise TypeError(
                f"State 'explicit_features' component should be numpy array, but got {type(explicit_features_state)}"
            )
        if explicit_features_state.shape != expected_explicit_features_shape:
            raise ValueError(
                f"State 'explicit_features' shape mismatch! GameState:{explicit_features_state.shape}, EnvConfig:{expected_explicit_features_shape}"
            )
        print(
            f"GameState 'explicit_features' state shape check PASSED (Shape: {explicit_features_state.shape})."
        )

        if env_config_instance.CALCULATE_POTENTIAL_OUTCOMES_IN_STATE:
            print("Potential outcome calculation is ENABLED in EnvConfig.")
        else:
            print("Potential outcome calculation is DISABLED in EnvConfig.")

        # Removed PBRS check

        _ = gs_test.valid_actions()
        print("GameState valid_actions check PASSED.")
        if not hasattr(gs_test, "game_score"):
            raise AttributeError("GameState missing 'game_score' attribute!")
        print("GameState 'game_score' attribute check PASSED.")
        if not hasattr(gs_test, "triangles_cleared_this_episode"):
            raise AttributeError(
                "GameState missing 'triangles_cleared_this_episode' attribute!"
            )
        print("GameState 'triangles_cleared_this_episode' attribute check PASSED.")

        del gs_test
        print("--- Pre-Run Checks Complete ---")
        return True
    except (NameError, ImportError) as e:
        print(f"FATAL ERROR: Import/Name error during pre-check: {e}")
    except (ValueError, AttributeError, TypeError, KeyError) as e:
        print(f"FATAL ERROR during pre-run checks: {e}")
    except Exception as e:
        print(f"FATAL ERROR during GameState pre-check: {e}")
        traceback.print_exc()
    sys.exit(1)


File: utils\types.py
from typing import Dict, Any
import numpy as np

StateType = Dict[str, np.ndarray]
ActionType = int
AgentStateDict = Dict[str, Any]


File: utils\__init__.py
# File: utils/__init__.py
from .helpers import (
    get_device,
    set_random_seeds,
    ensure_numpy,
    save_object,
    load_object,
    format_eta,  # Added format_eta
)
from .init_checks import run_pre_checks
from .types import StateType, ActionType, AgentStateDict


__all__ = [
    "get_device",
    "set_random_seeds",
    "ensure_numpy",
    "save_object",
    "load_object",
    "format_eta",  # Added format_eta
    "run_pre_checks",
    "StateType",
    "ActionType",
    "AgentStateDict",
]


File: visualization\__init__.py


File: workers\self_play_worker.py
# File: workers/self_play_worker.py
import threading
import time
import queue
import traceback
import torch
import copy
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
import logging
import numpy as np
import multiprocessing as mp
import ray
import asyncio

from environment.game_state import GameState, StateType
from environment.shape import Shape
from mcts import MCTS
from config import EnvConfig, MCTSConfig, TrainConfig
from utils.types import ActionType

if TYPE_CHECKING:
    from ray.util.queue import Queue as RayQueue

    AgentPredictorHandle = ray.actor.ActorHandle
    StatsAggregatorHandle = ray.actor.ActorHandle

ExperienceTuple = Tuple[StateType, Dict[ActionType, float], int]
ProcessedExperienceBatch = List[Tuple[StateType, Dict[ActionType, float], float]]

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class SelfPlayWorker:
    """Plays games using MCTS to generate training data (Ray Actor)."""

    INTERMEDIATE_STATS_INTERVAL_SEC = 5.0

    def __init__(
        self,
        worker_id: int,
        agent_predictor: "AgentPredictorHandle",
        mcts_config: MCTSConfig,
        env_config: EnvConfig,
        experience_queue: "RayQueue",
        stats_aggregator: "StatsAggregatorHandle",
        max_game_steps: Optional[int] = None,
    ):
        self.worker_id = worker_id
        self.agent_predictor = agent_predictor
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.mcts = MCTS(
            agent_predictor=self.agent_predictor,
            config=self.mcts_config,
            env_config=self.env_config,
        )
        self.experience_queue = experience_queue
        self.stats_aggregator = stats_aggregator
        self.max_game_steps = max_game_steps if max_game_steps else float("inf")
        self.log_prefix = f"[SelfPlayWorker-{self.worker_id}]"
        self.last_intermediate_stats_time = 0.0

        self._current_render_state_dict: Optional[StateType] = None
        self._last_stats: Dict[str, Any] = {
            "status": "Initialized",
            "game_steps": 0,
            "game_score": 0,
            "mcts_sim_time": 0.0,
            "mcts_nn_time": 0.0,
            "mcts_nodes_explored": 0,
            "mcts_avg_depth": 0.0,
            "available_shapes_data": [],
        }
        self._stop_requested = False
        logger.info(f"{self.log_prefix} Initialized as Ray Actor.")

    async def get_current_render_data(self) -> Optional[Dict[str, Any]]:
        """Returns a serializable dictionary for rendering (async)."""
        if self._current_render_state_dict:
            return {
                "state_dict": self._current_render_state_dict,
                "stats": self._last_stats.copy(),
            }
        else:
            return {"state_dict": None, "stats": self._last_stats.copy()}

    def _update_render_state(
        self, game_state: Optional[GameState], stats: Dict[str, Any]
    ):
        """Updates the state dictionary and stats exposed for rendering."""
        if game_state:
            try:
                self._current_render_state_dict = game_state.get_state()
                available_shapes_data = []
                for shape_obj in game_state.shapes:
                    if shape_obj:
                        available_shapes_data.append(
                            {"triangles": shape_obj.triangles, "color": shape_obj.color}
                        )
                    else:
                        available_shapes_data.append(None)
            except Exception as e:
                logger.error(f"{self.log_prefix} Error getting game state dict: {e}")
                self._current_render_state_dict = None
                available_shapes_data = []
        else:
            self._current_render_state_dict = None
            available_shapes_data = []

        current_game_score = (
            game_state.game_score
            if game_state
            else self._last_stats.get("game_score", 0)
        )

        ui_stats = {
            "status": stats.get("status", self._last_stats.get("status", "Unknown")),
            "game_steps": stats.get(
                "game_steps", self._last_stats.get("game_steps", 0)
            ),
            "game_score": current_game_score,
            "mcts_sim_time": stats.get(
                "mcts_total_duration", self._last_stats.get("mcts_sim_time", 0.0)
            ),
            "mcts_nn_time": stats.get(
                "total_nn_prediction_time", self._last_stats.get("mcts_nn_time", 0.0)
            ),
            "mcts_nodes_explored": stats.get(
                "nodes_created", self._last_stats.get("mcts_nodes_explored", 0)
            ),
            "mcts_avg_depth": stats.get(
                "avg_leaf_depth", self._last_stats.get("mcts_avg_depth", 0.0)
            ),
            "available_shapes_data": available_shapes_data,
        }
        self._last_stats.update(ui_stats)

    def _get_temperature(self, game_step: int) -> float:
        """Calculates the MCTS temperature based on the game step."""
        if game_step < self.mcts_config.TEMPERATURE_ANNEAL_STEPS:
            progress = game_step / max(1, self.mcts_config.TEMPERATURE_ANNEAL_STEPS)
            return (
                self.mcts_config.TEMPERATURE_INITIAL * (1 - progress)
                + self.mcts_config.TEMPERATURE_FINAL * progress
            )
        return self.mcts_config.TEMPERATURE_FINAL

    async def _play_one_game(self) -> Optional[ProcessedExperienceBatch]:
        """Plays a single game and returns the processed experience (async)."""
        current_game_num_ref = self.stats_aggregator.get_total_episodes.remote()
        current_game_num = await current_game_num_ref + 1
        logger.info(f"{self.log_prefix} Starting game {current_game_num}")
        start_time = time.monotonic()
        game_data: List[ExperienceTuple] = []
        game = GameState()
        current_state_features = game.reset()
        game_steps = 0
        self.last_intermediate_stats_time = time.monotonic()
        self._update_render_state(game, {"status": "Starting", "game_steps": 0})

        # Initial Stats Update (Async)
        # Call qsize() directly, it returns an ObjectRef
        qsize_ref = self.experience_queue.qsize()
        buffer_size = await qsize_ref  # Await the ObjectRef
        recording_step = {
            "current_self_play_game_number": current_game_num,
            "current_self_play_game_steps": 0,
            "buffer_size": buffer_size,
        }
        self.stats_aggregator.record_step.remote(recording_step)
        logger.info(
            f"{self.log_prefix} Game {current_game_num} started. Buffer size: {buffer_size}"
        )

        while not game.is_over() and game_steps < self.max_game_steps:
            if self._stop_requested:
                logger.info(
                    f"{self.log_prefix} Stop requested during game {current_game_num}. Aborting."
                )
                self._update_render_state(
                    game, {"status": "Stopped", "game_steps": game_steps}
                )
                return None

            current_time = time.monotonic()
            if (
                current_time - self.last_intermediate_stats_time
                > self.INTERMEDIATE_STATS_INTERVAL_SEC
            ):
                self._update_render_state(
                    game, {"status": "Running", "game_steps": game_steps}
                )
                # Call qsize() directly, it returns an ObjectRef
                qsize_ref = self.experience_queue.qsize()
                buffer_size = await qsize_ref  # Await the ObjectRef
                recording_step = {
                    "current_self_play_game_number": current_game_num,
                    "current_self_play_game_steps": game_steps,
                    "buffer_size": buffer_size,
                }
                self.stats_aggregator.record_step.remote(recording_step)
                self.last_intermediate_stats_time = current_time

            mcts_start_time = time.monotonic()
            try:
                root_node, mcts_stats = self.mcts.run_simulations(
                    root_state=game, num_simulations=self.mcts_config.NUM_SIMULATIONS
                )
            except Exception as mcts_err:
                if self._stop_requested:
                    logger.info(
                        f"{self.log_prefix} MCTS interrupted (stop requested) for game {current_game_num}, step {game_steps}. Aborting game."
                    )
                    self._update_render_state(
                        game, {"status": "Stopped", "game_steps": game_steps}
                    )
                    return None
                else:
                    logger.error(
                        f"{self.log_prefix} MCTS failed for game {current_game_num}, step {game_steps}: {mcts_err}",
                        exc_info=True,
                    )
                    game.game_over = True
                    break

            mcts_duration = time.monotonic() - mcts_start_time
            mcts_stats["game_steps"] = game_steps
            mcts_stats["status"] = "Running (MCTS)"
            logger.debug(
                f"{self.log_prefix} Game {current_game_num} Step {game_steps}: MCTS took {mcts_duration:.4f}s"
            )
            self._update_render_state(game, mcts_stats)

            temperature = self._get_temperature(game_steps)
            policy_target = self.mcts.get_policy_target(root_node, temperature)
            game_data.append((copy.deepcopy(current_state_features), policy_target, 1))

            action = self.mcts.choose_action(root_node, temperature)
            if action == -1:
                logger.error(
                    f"{self.log_prefix} MCTS failed to choose an action. Aborting game {current_game_num}."
                )
                game.game_over = True
                break

            step_start_time = time.monotonic()
            _, done = game.step(action)
            step_duration = time.monotonic() - step_start_time
            logger.debug(
                f"{self.log_prefix} Game {current_game_num} Step {game_steps}: Game step took {step_duration:.4f}s"
            )
            current_state_features = game.get_state()
            game_steps += 1

            # Record MCTS Stats (Async Fire-and-forget)
            # Call qsize() directly, it returns an ObjectRef
            qsize_ref = self.experience_queue.qsize()
            buffer_size = await qsize_ref  # Await the ObjectRef
            step_stats_for_aggregator = {
                "mcts_sim_time": mcts_stats.get("mcts_total_duration", 0.0),
                "mcts_nn_time": mcts_stats.get("total_nn_prediction_time", 0.0),
                "mcts_nodes_explored": mcts_stats.get("nodes_created", 0),
                "mcts_avg_depth": mcts_stats.get("avg_leaf_depth", 0.0),
                "buffer_size": buffer_size,
            }
            self.stats_aggregator.record_step.remote(step_stats_for_aggregator)

        status = (
            "Finished (Max Steps)"
            if game_steps >= self.max_game_steps and not game.is_over()
            else "Finished"
        )
        self._update_render_state(game, {"status": status, "game_steps": game_steps})

        if self._stop_requested:
            logger.info(
                f"{self.log_prefix} Stop requested after game {current_game_num} finished. Not processing."
            )
            return None

        final_outcome = game.get_outcome()
        processed_data: ProcessedExperienceBatch = [
            (state, policy, final_outcome * player)
            for state, policy, player in game_data
        ]
        game_duration = time.monotonic() - start_time
        logger.info(
            f"{self.log_prefix} Game {current_game_num} finished in {game_duration:.2f}s "
            f"({game_steps} steps). Outcome: {final_outcome}, Score: {game.game_score}. "
            f"Queueing {len(processed_data)} experiences."
        )

        final_state_for_best = game.get_state()
        self.stats_aggregator.record_episode.remote(
            episode_outcome=final_outcome,
            episode_length=game_steps,
            episode_num=current_game_num,
            game_score=game.game_score,
            triangles_cleared=game.triangles_cleared_this_episode,
            game_state_for_best=final_state_for_best,
        )
        return processed_data

    async def run_loop(self):
        """Main loop for the self-play worker actor (async)."""
        logger.info(f"{self.log_prefix} Starting run loop.")
        while not self._stop_requested:
            try:
                processed_data = await self._play_one_game()
                if processed_data is None:
                    if self._stop_requested:
                        break
                    else:
                        logger.warning(
                            f"{self.log_prefix} Game play returned None without stop signal. Continuing."
                        )
                        await asyncio.sleep(0.5)
                        continue

                if processed_data:
                    try:
                        q_put_start = time.monotonic()
                        await self.experience_queue.put_async(
                            processed_data, timeout=1.0
                        )
                        q_put_duration = time.monotonic() - q_put_start
                        # Call qsize() directly, it returns an ObjectRef
                        qsize_ref = self.experience_queue.qsize()
                        qsize = await qsize_ref  # Await the ObjectRef
                        logger.debug(
                            f"{self.log_prefix} Added game data to queue (qsize: {qsize}) in {q_put_duration:.4f}s."
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"{self.log_prefix} Experience queue put timed out. Discarding game data."
                        )
                    except Exception as q_err:
                        logger.error(
                            f"{self.log_prefix} Error putting data in queue: {q_err}"
                        )
                        if self._stop_requested:
                            break
                        await asyncio.sleep(0.5)  # Corrected indentation

            except Exception as e:
                logger.critical(
                    f"{self.log_prefix} CRITICAL ERROR in run loop: {e}", exc_info=True
                )
                self._update_render_state(None, {"status": "Error"})
                if self._stop_requested:
                    break
                await asyncio.sleep(5.0)  # Corrected indentation

        logger.info(f"{self.log_prefix} Run loop finished.")
        self._update_render_state(None, {"status": "Stopped"})

    def stop(self):
        """Signals the actor to stop gracefully."""
        logger.info(f"{self.log_prefix} Stop requested.")
        self._stop_requested = True

    def health_check(self):
        """Ray health check method."""
        return "OK"


File: workers\training_worker.py
# File: workers/training_worker.py
import threading
import time
import queue
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
import logging
import multiprocessing as mp
import ray
import asyncio

from config import TrainConfig
from utils.types import StateType, ActionType

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import _LRScheduler
    from ray.util.queue import Queue as RayQueue

    AgentPredictorHandle = ray.actor.ActorHandle
    StatsAggregatorHandle = ray.actor.ActorHandle

ExperienceData = Tuple[StateType, Dict[ActionType, float], float]
ProcessedExperienceBatch = List[ExperienceData]
logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class TrainingWorker:
    """Samples experience and trains the neural network (Ray Actor)."""

    def __init__(
        self,
        agent_predictor: "AgentPredictorHandle",
        optimizer_cls: type,
        optimizer_kwargs: dict,
        scheduler_cls: Optional[type],
        scheduler_kwargs: Optional[dict],
        experience_queue: "RayQueue",
        stats_aggregator: "StatsAggregatorHandle",
        train_config: TrainConfig,
    ):
        self.agent_predictor = agent_predictor
        self.experience_queue = experience_queue
        self.stats_aggregator = stats_aggregator
        self.train_config = train_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_prefix = "[TrainingWorker]"

        from config import EnvConfig, ModelConfig
        from agent.alphazero_net import AlphaZeroNet

        self.local_agent = AlphaZeroNet(EnvConfig(), ModelConfig.Network()).to(
            self.device
        )

        self.optimizer = optimizer_cls(
            self.local_agent.parameters(), **optimizer_kwargs
        )
        self.scheduler = None
        if scheduler_cls and scheduler_kwargs:
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_kwargs)

        self.steps_done = 0
        self._stop_requested = False

        logger.info(
            f"{self.log_prefix} Initialized as Ray Actor. Device: {self.device}."
        )
        logger.info(
            f"{self.log_prefix} Config: Batch={self.train_config.BATCH_SIZE}, LR={self.train_config.LEARNING_RATE}, MinBuffer={self.train_config.MIN_BUFFER_SIZE_TO_TRAIN}"
        )
        if self.scheduler:
            logger.info(
                f"{self.log_prefix} LR Scheduler Type: {type(self.scheduler).__name__}"
            )
        else:
            logger.info(f"{self.log_prefix} LR Scheduler: DISABLED")

    async def _get_initial_state(self):
        """Asynchronously fetches initial weights and global step."""
        try:
            weights_ref = self.agent_predictor.get_weights.remote()
            step_ref = self.stats_aggregator.get_current_global_step.remote()
            initial_weights, initial_step = await asyncio.gather(weights_ref, step_ref)
            self.local_agent.load_state_dict(initial_weights)
            self.steps_done = initial_step
            logger.info(
                f"{self.log_prefix} Initial weights loaded. Initial global step: {self.steps_done}"
            )
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Failed to get initial state: {e}. Starting from scratch."
            )
            self.steps_done = 0

    def _prepare_batch(
        self, batch_data: ProcessedExperienceBatch
    ) -> Optional[Tuple[StateType, torch.Tensor, torch.Tensor]]:
        """Converts a list of experience tuples into batched tensors."""
        try:
            if (
                not batch_data
                or not isinstance(batch_data[0], tuple)
                or len(batch_data[0]) != 3
            ):
                logger.error(f"{self.log_prefix} Invalid batch data structure (outer).")
                return None
            if not isinstance(batch_data[0][0], dict):
                logger.error(
                    f"{self.log_prefix} Invalid batch data structure (state dict)."
                )
                return None

            states = {key: [] for key in batch_data[0][0].keys()}
            policy_targets, value_targets = [], []
            valid_items = 0

            for item in batch_data:
                if not isinstance(item, tuple) or len(item) != 3:
                    logger.warning(
                        f"{self.log_prefix} Skipping invalid item in batch (wrong structure)."
                    )
                    continue
                state_dict, policy_dict, outcome = item
                if not isinstance(state_dict, dict) or not isinstance(
                    policy_dict, dict
                ):
                    logger.warning(
                        f"{self.log_prefix} Skipping invalid item in batch (wrong inner types)."
                    )
                    continue
                if not (isinstance(outcome, (float, int)) and np.isfinite(outcome)):
                    logger.warning(
                        f"{self.log_prefix} Skipping invalid item in batch (invalid outcome: {outcome})."
                    )
                    continue

                temp_state, valid_state = {}, True
                for key, value in state_dict.items():
                    if key in states and isinstance(value, np.ndarray):
                        temp_state[key] = value
                    else:
                        if key == "death_mask" and key not in states:
                            logger.warning(
                                f"{self.log_prefix} State dict missing expected 'death_mask' key initially."
                            )
                        elif key != "death_mask":
                            logger.warning(
                                f"{self.log_prefix} Skipping invalid item in batch (invalid state key/value: {key}, type: {type(value)})."
                            )
                            valid_state = False
                            break
                        else:
                            temp_state[key] = value
                if not valid_state:
                    continue

                policy_array = np.zeros(
                    self.local_agent.env_cfg.ACTION_DIM, dtype=np.float32
                )
                policy_sum = 0.0
                valid_policy_entries = 0
                for action, prob in policy_dict.items():
                    if (
                        isinstance(action, int)
                        and 0 <= action < self.local_agent.env_cfg.ACTION_DIM
                        and isinstance(prob, (float, int))
                        and np.isfinite(prob)
                        and prob >= 0
                    ):
                        policy_array[action] = prob
                        policy_sum += prob
                        valid_policy_entries += 1
                if valid_policy_entries > 0 and not np.isclose(
                    policy_sum, 1.0, atol=1e-4
                ):
                    logger.warning(
                        f"{self.log_prefix} Policy target sum is {policy_sum:.4f}, expected ~1.0. Using as is."
                    )

                for key in states.keys():
                    states[key].append(temp_state[key])
                policy_targets.append(policy_array)
                value_targets.append(outcome)
                valid_items += 1

            if valid_items == 0:
                logger.error(f"{self.log_prefix} No valid items found in the batch.")
                return None

            batched_states = {
                k: torch.from_numpy(np.stack(v)).to(self.device)
                for k, v in states.items()
            }
            batched_policy = torch.from_numpy(np.stack(policy_targets)).to(self.device)
            batched_value = (
                torch.tensor(value_targets, dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device)
            )

            if (
                batched_policy.shape[0] != valid_items
                or batched_value.shape[0] != valid_items
            ):
                logger.error(
                    f"{self.log_prefix} Shape mismatch after stacking tensors."
                )
                return None

            return batched_states, batched_policy, batched_value
        except Exception as e:
            logger.error(f"{self.log_prefix} Error preparing batch: {e}", exc_info=True)
            return None

    async def _perform_training_step(
        self, batch_data: ProcessedExperienceBatch
    ) -> Optional[Dict[str, float]]:
        """Performs a single training step (async for stats)."""
        prep_start = time.monotonic()
        prepared_batch = self._prepare_batch(batch_data)
        prep_duration = time.monotonic() - prep_start
        if prepared_batch is None:
            logger.warning(
                f"{self.log_prefix} Failed to prepare batch (took {prep_duration:.4f}s). Skipping step."
            )
            return None
        batch_states, batch_policy_targets, batch_value_targets = prepared_batch
        logger.debug(f"{self.log_prefix} Batch preparation took {prep_duration:.4f}s.")

        try:
            step_start_time = time.monotonic()
            self.local_agent.train()
            self.optimizer.zero_grad()
            policy_logits, value_preds = self.local_agent(batch_states)

            if (
                policy_logits.shape[0] != batch_policy_targets.shape[0]
                or value_preds.shape[0] != batch_value_targets.shape[0]
            ):
                logger.error(
                    f"{self.log_prefix} Batch size mismatch after forward pass! Skipping. Logits: {policy_logits.shape}, Targets: {batch_policy_targets.shape}, Values: {value_preds.shape}, Targets: {batch_value_targets.shape}"
                )
                return None
            if policy_logits.shape[1] != batch_policy_targets.shape[1]:
                logger.error(
                    f"{self.log_prefix} Action dim mismatch after forward pass! Skipping. Logits: {policy_logits.shape[1]}, Targets: {batch_policy_targets.shape[1]}"
                )
                return None

            log_policy_preds = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(
                batch_policy_targets * log_policy_preds, dim=1
            ).mean()
            value_loss = F.mse_loss(value_preds, batch_value_targets)
            total_loss = (
                self.train_config.POLICY_LOSS_WEIGHT * policy_loss
                + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            )

            total_loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            step_duration = time.monotonic() - step_start_time
            logger.debug(f"{self.log_prefix} Training step took {step_duration:.4f}s.")
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.steps_done % 10 == 0:
                weights = self.local_agent.state_dict()
                self.agent_predictor.set_weights.remote(weights)
                logger.debug(
                    f"{self.log_prefix} Sent updated weights to AgentPredictor."
                )

            return {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "update_time": step_duration,
                "lr": current_lr,
            }
        except Exception as e:
            logger.critical(
                f"{self.log_prefix} CRITICAL ERROR during training step {self.steps_done}: {e}",
                exc_info=True,
            )
            return None

    async def run_loop(self):
        """Main training loop (async)."""
        logger.info(f"{self.log_prefix} Starting run loop.")
        await self._get_initial_state()
        logger.info(
            f"{self.log_prefix} Starting training from Global Step: {self.steps_done}"
        )

        last_buffer_update_time = 0
        buffer_update_interval = 1.0

        while not self._stop_requested:
            # Call qsize() directly, it returns an ObjectRef
            qsize_ref = self.experience_queue.qsize()
            buffer_size = await qsize_ref  # Await the ObjectRef

            if buffer_size < self.train_config.MIN_BUFFER_SIZE_TO_TRAIN:
                if time.time() - last_buffer_update_time > buffer_update_interval:
                    self.stats_aggregator.record_step.remote(
                        {"buffer_size": buffer_size}
                    )
                    last_buffer_update_time = time.time()
                    logger.info(
                        f"{self.log_prefix} Waiting for buffer... Size: {buffer_size}/{self.train_config.MIN_BUFFER_SIZE_TO_TRAIN}"
                    )
                await asyncio.sleep(0.1)
                continue

            logger.debug(
                f"{self.log_prefix} Starting training iteration. Buffer size: {buffer_size}"
            )
            steps_this_iter, iter_policy_loss, iter_value_loss = 0, 0.0, 0.0
            iter_start_time = time.monotonic()

            for _ in range(self.train_config.NUM_TRAINING_STEPS_PER_ITER):
                if self._stop_requested:
                    break

                batch_data_list: Optional[ProcessedExperienceBatch] = None
                try:
                    q_get_start = time.monotonic()
                    batch_data_list = await self.experience_queue.get_async(timeout=1.0)
                    q_get_duration = time.monotonic() - q_get_start
                    logger.debug(
                        f"{self.log_prefix} Queue get (batch size {len(batch_data_list) if batch_data_list else 0}) took {q_get_duration:.4f}s."
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"{self.log_prefix} Queue empty during training iteration, waiting..."
                    )
                    await asyncio.sleep(0.1)
                    break
                except Exception as e:
                    logger.error(
                        f"{self.log_prefix} Error getting data from queue: {e}",
                        exc_info=True,
                    )
                    break

                if not batch_data_list:
                    continue

                actual_batch_size = min(
                    len(batch_data_list), self.train_config.BATCH_SIZE
                )
                if actual_batch_size < 1:
                    continue

                step_result = await self._perform_training_step(
                    batch_data_list[:actual_batch_size]
                )
                if step_result is None:
                    logger.warning(
                        f"{self.log_prefix} Training step failed, ending iteration early."
                    )
                    break

                self.steps_done += 1
                steps_this_iter += 1
                iter_policy_loss += step_result["policy_loss"]
                iter_value_loss += step_result["value_loss"]

                # Call qsize() directly, it returns an ObjectRef
                qsize_ref = self.experience_queue.qsize()
                current_buffer_size = await qsize_ref  # Await the ObjectRef
                step_stats = {
                    "global_step": self.steps_done,
                    "buffer_size": current_buffer_size,
                    "training_steps_performed": self.steps_done,
                    **step_result,
                }
                self.stats_aggregator.record_step.remote(step_stats)

            iter_duration = time.monotonic() - iter_start_time
            if steps_this_iter > 0:
                avg_p = iter_policy_loss / steps_this_iter
                avg_v = iter_value_loss / steps_this_iter
                logger.info(
                    f"{self.log_prefix} Iteration complete. Steps: {steps_this_iter}, Duration: {iter_duration:.2f}s, Avg P.Loss: {avg_p:.4f}, Avg V.Loss: {avg_v:.4f}"
                )
            else:
                logger.info(
                    f"{self.log_prefix} Iteration finished with 0 steps performed (Duration: {iter_duration:.2f}s)."
                )

            await asyncio.sleep(0.01)

        logger.info(f"{self.log_prefix} Run loop finished.")

    def stop(self):
        """Signals the actor to stop gracefully."""
        logger.info(f"{self.log_prefix} Stop requested.")
        self._stop_requested = True

    def health_check(self):
        """Ray health check method."""
        return "OK"

File: workers\__init__.py
# File: workers/__init__.py
# This file makes the 'workers' directory a Python package.

from .self_play_worker import SelfPlayWorker
from .training_worker import TrainingWorker  # Added TrainingWorker

__all__ = ["SelfPlayWorker", "TrainingWorker"]


