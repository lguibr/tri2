# File: main_pygame.py
# main_pygame.py
import pygame
import sys
import time
import threading
import logging
import argparse
import os
import traceback  # Added for error handling
from typing import Optional, List, Dict, Any
import torch  # <<< ADDED IMPORT HERE

# --- Resource Monitoring Import ---
try:
    import psutil
except ImportError:
    print("Warning: psutil not found. CPU/Memory usage monitoring will be disabled.")
    print("Install it using: pip install psutil")
    psutil = None
# --- End Resource Monitoring Import ---


# --- Path Adjustment ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# --- End Path Adjustment ---

# Corrected import: Import specific config classes from the 'config' package
try:
    from config import (
        VisConfig,
        EnvConfig,
        PPOConfig,
        ModelConfig,  # Changed from NetworkConfig
        StatsConfig,
        TrainConfig,  # Changed from CheckpointConfig
        TensorBoardConfig,  # Changed from LogConfig
        DemoConfig,  # Changed from DebugConfig
        RNNConfig,  # Added
        TransformerConfig,  # Added
        ObsNormConfig,  # Added
        RewardConfig,  # Added
        DEVICE,  # Import directly
        RANDOM_SEED,  # Import directly
        TOTAL_TRAINING_STEPS,  # Import directly
        BASE_CHECKPOINT_DIR,  # Import directly
        BASE_LOG_DIR,  # Import directly
        set_device,  # Import function
        get_run_id,  # Import function
        set_run_id,  # Import function
        get_run_checkpoint_dir,  # Import function
        get_run_log_dir,  # Import function
        get_console_log_dir,  # Import function
        print_config_info_and_validate,  # Import function
        get_config_dict,  # Import function
    )
    from utils.helpers import (
        get_device as get_torch_device,
    )  # Use helper for device detection
    from utils.helpers import set_random_seeds  # Use helper for seeds

    from logger import TeeLogger  # Import TeeLogger from logger.py
    from utils.init_checks import run_pre_checks  # Use helper for pre-checks
    from utils.types import AgentStateDict  # Import type if needed

except ImportError as e:
    print(f"Error importing configuration classes or utils: {e}")
    print(
        "Please ensure 'config/__init__.py' is set up correctly and imports necessary classes from config.core and config.general."
    )
    traceback.print_exc()  # Print detailed traceback
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during config/util import: {e}")
    traceback.print_exc()  # Print detailed traceback
    sys.exit(1)

# --- Component Imports (Assuming these paths are correct relative to main_pygame.py) ---
# Note: The original main_pygame.py had different paths (e.g., game.*, rl_components.*)
# compared to the app_*.py files (e.g., environment.*, agent.*).
# Using the paths from app_*.py as they seem more consistent with the provided codebase.
try:
    from environment.game_state import GameState
    from ui.renderer import UIRenderer  # Assuming this is the main renderer now
    from agent.ppo_agent import PPOAgent
    from training.rollout_storage import RolloutStorage  # Assuming this is the buffer
    from training.rollout_collector import RolloutCollector
    from workers import (
        EnvironmentRunner,
        TrainingWorker,
    )  # Assuming these are the workers

    # Corrected stats imports: Import from the package level
    from stats import (
        StatsRecorderBase,
        SimpleStatsRecorder,
        TensorBoardStatsRecorder,
        StatsAggregator,
    )
    from training.checkpoint_manager import (
        CheckpointManager,
        find_latest_run_and_checkpoint,
    )
    from app_state import AppState  # Use the AppState enum
    from app_init import AppInitializer  # Use the AppInitializer
    from app_logic import AppLogic  # Use the AppLogic
    from app_workers import AppWorkerManager  # Use the AppWorkerManager
    from app_setup import (
        initialize_pygame,
        initialize_directories,
        load_and_validate_configs,
    )  # Use app_setup helpers
    from app_ui_utils import AppUIUtils  # Use UI utils
    from ui.input_handler import InputHandler  # Use the InputHandler
    import queue  # For experience queue

except ImportError as e:
    print(f"Error importing application components: {e}")
    print(
        "Please ensure component paths (environment, agent, ui, training, workers, stats, app_*) are correct."
    )
    traceback.print_exc()
    sys.exit(1)


# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MainApp:
    """Main application class orchestrating Pygame UI, RL components, and threads."""

    def __init__(self, checkpoint_to_load: Optional[str] = None):
        # --- Configuration ---
        # Instantiate config classes directly
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        self.ppo_config = PPOConfig()
        self.rnn_config = RNNConfig()
        self.train_config_instance = TrainConfig()  # Store instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        self.reward_config = RewardConfig()
        self.obs_norm_config = ObsNormConfig()
        self.transformer_config = TransformerConfig()
        self.config_dict = get_config_dict()  # Get combined dict for logging

        # --- Core Components ---
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None

        # --- State ---
        self.app_state: AppState = AppState.INITIALIZING
        self.is_process_running: bool = False  # Training/collection active
        self.status: str = "Initializing..."
        self.running: bool = True  # Main loop flag
        self.update_progress_details: Dict[str, Any] = {}  # Added attribute

        # --- Threading & Communication ---
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # Start paused
        self.experience_queue = queue.Queue(maxsize=10)  # Experience buffer queue

        # --- RL Components (Managed by Initializer) ---
        # Placeholders, will be populated by AppInitializer
        self.envs: List[GameState] = []
        self.agent: Optional[PPOAgent] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.rollout_collector: Optional[RolloutCollector] = None
        self.demo_env: Optional[GameState] = None  # For demo/debug mode

        # --- Helper Classes ---
        self.device = get_torch_device()
        set_device(self.device)  # Set global device variable
        self.checkpoint_to_load = checkpoint_to_load  # Store path from args
        self.initializer = AppInitializer(self)
        self.logic = AppLogic(self)
        self.worker_manager = AppWorkerManager(self)
        self.ui_utils = AppUIUtils(self)

        # --- UI State ---
        self.cleanup_confirmation_active: bool = False
        self.cleanup_message: str = ""
        self.last_cleanup_message_time: float = 0.0
        self.total_gpu_memory_bytes: Optional[int] = None

        # --- Resource Monitoring ---
        self.last_resource_update_time: float = 0.0
        self.resource_update_interval: float = 1.0  # Update every second

    def initialize(self):
        """Initializes Pygame, directories, configs, and core components."""
        logger.info("--- Application Initialization ---")
        self.screen, self.clock = initialize_pygame(self.vis_config)
        initialize_directories()  # Creates checkpoint/log dirs for current run
        # Configs are already instantiated in __init__
        # print_config_info_and_validate() # Validate the instantiated configs

        set_random_seeds(RANDOM_SEED)
        run_pre_checks()  # Basic GameState checks

        # Initialize core RL and UI components via AppInitializer
        self.app_state = AppState.INITIALIZING
        self.initializer.initialize_all()

        # Set input handler reference in renderer after init
        if self.renderer and self.initializer.app.input_handler:
            self.renderer.set_input_handler(self.initializer.app.input_handler)

        # Start worker threads (will wait for pause_event to clear)
        self.worker_manager.start_worker_threads()

        # Check initial completion status after loading checkpoint
        self.logic.check_initial_completion_status()
        if not self.status.startswith(
            "Training Complete"
        ):  # Avoid overwriting completion status
            self.status = "Ready"
        self.app_state = AppState.MAIN_MENU

        logger.info("--- Initialization Complete ---")
        if self.tensorboard_config.LOG_DIR:
            tb_path = os.path.abspath(get_run_log_dir())
            logger.info(f"--- TensorBoard logs: tensorboard --logdir {tb_path} ---")

    def _update_resource_stats(self):
        """Updates CPU, Memory, and GPU usage in the StatsAggregator."""
        current_time = time.time()
        if (
            current_time - self.last_resource_update_time
            < self.resource_update_interval
        ):
            return

        if not self.initializer.stats_recorder or not hasattr(
            self.initializer.stats_recorder, "aggregator"
        ):
            return

        aggregator = self.initializer.stats_recorder.aggregator
        storage = aggregator.storage

        cpu_percent = 0.0
        mem_percent = 0.0
        gpu_mem_percent = 0.0

        if psutil:
            try:
                cpu_percent = psutil.cpu_percent()
                mem_percent = psutil.virtual_memory().percent
            except Exception as e:
                logger.warning(f"Error getting CPU/Mem usage: {e}")

        if self.device.type == "cuda" and self.total_gpu_memory_bytes:
            try:
                # Use torch here, which is now imported at the top
                allocated = torch.cuda.memory_allocated(self.device)
                # reserved = torch.cuda.memory_reserved(self.device) # Alternative
                gpu_mem_percent = (allocated / self.total_gpu_memory_bytes) * 100.0
            except Exception as e:
                logger.warning(f"Error getting GPU memory usage: {e}")
                gpu_mem_percent = 0.0  # Reset if error occurs

        # Update aggregator storage directly (requires lock)
        with aggregator._lock:
            storage.current_cpu_usage = cpu_percent
            storage.current_memory_usage = mem_percent
            storage.current_gpu_memory_usage_percent = gpu_mem_percent
            storage.cpu_usage.append(cpu_percent)
            storage.memory_usage.append(mem_percent)
            storage.gpu_memory_usage_percent.append(gpu_mem_percent)

        self.last_resource_update_time = current_time

    def run_main_loop(self):
        """The main application loop."""
        logger.info("Starting main application loop...")
        while self.running:
            dt = self.clock.tick(self.vis_config.FPS) / 1000.0  # Delta time in seconds

            # Handle Input
            if self.input_handler:
                self.running = self.input_handler.handle_input(
                    self.app_state.value, self.cleanup_confirmation_active
                )
                if not self.running:  # Exit requested via input handler
                    self.stop_event.set()
                    break
            else:  # Fallback exit if input handler fails
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        self.stop_event.set()
                        break
                if not self.running:
                    break

            # --- Fetch Update Progress Details ---
            # Fetch details *before* updating status logic
            if (
                self.worker_manager.training_worker_thread
                and self.worker_manager.training_worker_thread.is_alive()
            ):
                try:
                    self.update_progress_details = (
                        self.worker_manager.training_worker_thread.get_update_progress_details()
                    )
                except Exception as e:
                    logger.warning(f"Could not get update progress details: {e}")
                    self.update_progress_details = {}  # Reset on error
            else:
                self.update_progress_details = {}  # Clear if worker not running
            # --- End Fetch Update Progress Details ---

            # Update Logic & State (uses self.update_progress_details)
            self.logic.update_status_and_check_completion()

            # --- Update Resource Stats ---
            self._update_resource_stats()
            # --- End Resource Stats ---

            # --- Update Demo Env Timers ---
            # Ensure demo env timers decrement even when paused or in demo/debug mode
            if self.initializer.demo_env:
                try:
                    self.initializer.demo_env._update_timers()
                except Exception as timer_err:
                    # Log error but don't crash the main loop
                    logger.error(
                        f"Error updating demo env timers: {timer_err}", exc_info=False
                    )
            # --- End Timer Update ---

            # Render UI
            if self.renderer:
                plot_data = {}
                stats_summary = {}
                tb_log_dir = None
                # update_details = {} # Use self.update_progress_details now
                agent_params = 0
                worker_counts = {"env_runners": 0, "trainers": 0}

                if self.initializer.stats_recorder:
                    plot_data = self.initializer.stats_recorder.get_plot_data()
                    # --- Get latest global step directly from aggregator ---
                    current_step = 0
                    if hasattr(self.initializer.stats_recorder, "aggregator"):
                        current_step = getattr(
                            self.initializer.stats_recorder.aggregator.storage,
                            "current_global_step",
                            0,
                        )
                    # --- End get latest global step ---
                    stats_summary = self.initializer.stats_recorder.get_summary(
                        current_step  # Pass the latest step to get_summary
                    )
                    if isinstance(
                        self.initializer.stats_recorder, TensorBoardStatsRecorder
                    ):
                        tb_log_dir = self.initializer.stats_recorder.log_dir

                # --- Get Worker Counts ---
                if (
                    self.worker_manager.env_runner_thread
                    and self.worker_manager.env_runner_thread.is_alive()
                ):
                    worker_counts["env_runners"] = 1
                if (
                    self.worker_manager.training_worker_thread
                    and self.worker_manager.training_worker_thread.is_alive()
                ):
                    worker_counts["trainers"] = 1
                # --- End Worker Counts ---

                if self.initializer.agent:
                    agent_params = self.initializer.agent_param_count

                self.renderer.render_all(
                    app_state=self.app_state.value,
                    is_process_running=self.is_process_running,
                    status=self.status,
                    stats_summary=stats_summary,  # Pass the summary with latest step
                    envs=self.initializer.envs,  # Use initialized envs
                    num_envs=self.env_config.NUM_ENVS,
                    env_config=self.env_config,
                    cleanup_confirmation_active=self.cleanup_confirmation_active,
                    cleanup_message=self.cleanup_message,
                    last_cleanup_message_time=self.last_cleanup_message_time,
                    tensorboard_log_dir=tb_log_dir,
                    plot_data=plot_data,
                    demo_env=self.initializer.demo_env,
                    update_progress_details=self.update_progress_details,  # Pass stored details
                    agent_param_count=agent_params,
                    worker_counts=worker_counts,  # Pass worker counts
                )
            else:
                # Basic render if main renderer failed
                self.screen.fill((20, 0, 0))
                font = pygame.font.Font(None, 30)
                text_surf = font.render("Renderer Error", True, (255, 50, 50))
                self.screen.blit(
                    text_surf, text_surf.get_rect(center=self.screen.get_rect().center)
                )
                pygame.display.flip()

        logger.info("Main application loop exited.")
        # Shutdown is now called explicitly after the loop or in exception handlers

    def shutdown(self):
        """Cleans up resources and exits."""
        logger.info("Initiating shutdown sequence...")

        # Signal threads to stop
        logger.info("Setting stop event for worker threads.")
        self.stop_event.set()
        self.pause_event.clear()  # Ensure threads aren't stuck paused

        # Stop and join worker threads
        logger.info("Stopping worker threads...")
        self.worker_manager.stop_worker_threads()
        logger.info("Worker threads stopped.")

        # Save final checkpoint (if applicable)
        logger.info("Attempting final checkpoint save...")
        self.logic.save_final_checkpoint()
        logger.info("Final checkpoint save attempt finished.")

        # Close stats recorder (handles TensorBoard writer)
        logger.info("Closing stats recorder (before pygame.quit)...")
        self.initializer.close_stats_recorder()
        logger.info("Stats recorder closed.")

        # Quit Pygame
        logger.info("Quitting Pygame...")
        pygame.quit()
        logger.info("Pygame quit.")
        logger.info("Shutdown complete.")
        # sys.exit(0) # Exit is handled in the main block's finally clause


# --- Global variable for TeeLogger instance ---
tee_logger_instance: Optional[TeeLogger] = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriCrack PPO Trainer")
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint file to load (e.g., checkpoints/run_xyz/step_1000_agent_state.pth). Overrides auto-resume.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    # --- Setup Logging ---
    # Determine run ID early (needed for log file path)
    # CheckpointManager will handle resuming the correct ID later if loading
    if args.load_checkpoint:
        # Try to extract run_id from path, otherwise generate new one
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                print(f"Using Run ID from checkpoint path: {get_run_id()}")
            else:
                get_run_id()  # Generate new one
        except Exception:
            get_run_id()  # Generate new one if path parsing fails
    else:
        # Search for latest run to potentially resume ID, otherwise generate new
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            print(f"Resuming Run ID: {get_run_id()}")
        else:
            get_run_id()  # Generate new one

    # Setup TeeLogger
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        log_file_dir = (
            get_console_log_dir()
        )  # Use function from config.general via __init__
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, "console_output.log")
        tee_logger_instance = TeeLogger(log_file_path, sys.stdout)  # Redirect stdout
        sys.stdout = tee_logger_instance
        sys.stderr = tee_logger_instance  # Redirect stderr as well
    except Exception as e:
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)
        # Continue without TeeLogger if setup fails

    # Set logging level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.info(f"Logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {get_run_id()}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")

    # --- Create and Run Application ---
    app = None
    exit_code = 0
    try:
        app = MainApp(checkpoint_to_load=args.load_checkpoint)
        app.initialize()
        app.run_main_loop()  # This loop now exits cleanly via self.running = False
        app.shutdown()  # Call shutdown explicitly after loop finishes normally
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down.")
        if app:
            app.shutdown()
        else:
            pygame.quit()  # Ensure pygame quits even if app init failed
        exit_code = 130  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in main: {e}", exc_info=True)
        if app:
            app.shutdown()
        else:
            pygame.quit()  # Ensure pygame quits even if app init failed
        exit_code = 1
    finally:
        # Ensure logs are flushed and stdout/stderr restored *before* exiting
        print("[Main Finally] Restoring stdout/stderr and closing logger...")
        if tee_logger_instance:
            try:
                # Flush before closing and restoring
                if isinstance(sys.stdout, TeeLogger):
                    sys.stdout.flush()
                if isinstance(sys.stderr, TeeLogger):
                    sys.stderr.flush()

                # Restore original streams
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                # Now close the file handle
                tee_logger_instance.close()
                print("[Main Finally] TeeLogger closed and streams restored.")
            except Exception as log_close_err:
                # Use original stdout/stderr for this final error message
                original_stdout.write(f"ERROR closing TeeLogger: {log_close_err}\n")
                traceback.print_exc(file=original_stderr)

        print(f"[Main Finally] Exiting with code {exit_code}.")
        sys.exit(exit_code)
