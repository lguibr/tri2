# File: main_pygame.py
import pygame
import sys
import time
import threading
import logging
import argparse
import os
import traceback
from typing import Optional, Dict, Any, List
import queue
import numpy as np
from collections import deque

script_dir = os.path.dirname(os.path.abspath(__file__))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# --- Config and Utils Imports ---
try:
    from config import (
        VisConfig,
        EnvConfig,
        TrainConfig,
        MCTSConfig,
        RANDOM_SEED,
        BASE_CHECKPOINT_DIR,
        set_device,
        get_run_id,
        set_run_id,
        get_run_log_dir,
        get_console_log_dir,
        get_config_dict,
    )
    from utils.helpers import get_device as get_torch_device, set_random_seeds
    from logger import TeeLogger
    from utils.init_checks import run_pre_checks
except ImportError as e:
    print(f"Error importing config/utils: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# --- App Component Imports ---
try:
    from environment.game_state import GameState
    from ui.renderer import UIRenderer
    from stats import StatsAggregator
    from training.checkpoint_manager import (
        find_latest_run_and_checkpoint,
    )
    from app_state import AppState
    from app_init import AppInitializer
    from app_logic import AppLogic
    from app_workers import AppWorkerManager
    from app_setup import (
        initialize_pygame,
        initialize_directories,
    )
    from app_ui_utils import AppUIUtils
    from ui.input_handler import InputHandler
    from agent.alphazero_net import AlphaZeroNet
    from workers.training_worker import ProcessedExperienceBatch
except ImportError as e:
    print(f"Error importing app components: {e}\n{traceback.format_exc()}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
# Reduce default logging noise
logging.getLogger("mcts").setLevel(logging.WARNING)
logging.getLogger("workers.self_play_worker").setLevel(logging.INFO)
logging.getLogger("workers.training_worker").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)  # Main app logger

# --- Constants ---
LOOP_TIMING_INTERVAL = 60


class MainApp:
    """Main application class orchestrating Pygame UI and AlphaZero components."""

    def __init__(self, checkpoint_to_load: Optional[str] = None):
        # Config Instances
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        self.train_config_instance = TrainConfig()
        self.mcts_config = MCTSConfig()

        # Core Components
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None

        # State
        self.app_state: AppState = AppState.INITIALIZING
        self.status: str = "Initializing..."
        self.running: bool = True
        self.update_progress_details: Dict[str, Any] = {}

        # Threading & Communication
        self.stop_event = threading.Event()
        self.experience_queue: queue.Queue[ProcessedExperienceBatch] = queue.Queue(
            maxsize=self.train_config_instance.BUFFER_CAPACITY
        )
        logger.info(
            f"Experience Queue Size: {self.train_config_instance.BUFFER_CAPACITY}"
        )

        # RL Components (Managed by Initializer)
        self.agent: Optional[AlphaZeroNet] = None
        self.stats_aggregator: Optional[StatsAggregator] = None
        self.demo_env: Optional[GameState] = None

        # Helper Classes
        self.device = get_torch_device()
        set_device(self.device)
        self.checkpoint_to_load = checkpoint_to_load
        self.initializer = AppInitializer(self)
        self.logic = AppLogic(self)
        self.worker_manager = AppWorkerManager(self)
        self.ui_utils = AppUIUtils(self)

        # UI State
        self.cleanup_confirmation_active: bool = False
        self.cleanup_message: str = ""
        self.last_cleanup_message_time: float = 0.0
        self.total_gpu_memory_bytes: Optional[int] = None

        # Timing
        self.frame_count = 0
        self.loop_times = deque(maxlen=LOOP_TIMING_INTERVAL)

    def initialize(self):
        """Initializes Pygame, directories, configs, and core components."""
        logger.info("--- Application Initialization ---")
        self.screen, self.clock = initialize_pygame(self.vis_config)
        initialize_directories()
        set_random_seeds(RANDOM_SEED)
        run_pre_checks()

        self.app_state = AppState.INITIALIZING
        self.initializer.initialize_all()

        self.agent = self.initializer.agent
        self.stats_aggregator = self.initializer.stats_aggregator
        self.demo_env = self.initializer.demo_env

        if self.renderer and self.input_handler:
            self.renderer.set_input_handler(self.input_handler)
            if hasattr(self.input_handler, "app_ref"):
                self.input_handler.app_ref = self

        self.logic.check_initial_completion_status()
        self.status = "Ready"
        self.app_state = AppState.MAIN_MENU
        logger.info("--- Initialization Complete ---")

    def _handle_input(self) -> bool:
        """Handles user input using the InputHandler."""
        if self.input_handler:
            return self.input_handler.handle_input(
                self.app_state.value, self.cleanup_confirmation_active
            )
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.logic.exit_app()
                    return False
            return True

    def _update_state(self):
        """Updates application logic and status."""
        self.update_progress_details = {}
        self.logic.update_status_and_check_completion()

        # Update demo env timers if in demo/debug mode
        if self.app_state in [AppState.PLAYING, AppState.DEBUG] and self.demo_env:
            try:
                self.demo_env._update_timers()
            except Exception as e:
                logger.error(f"Error updating demo env timers: {e}")

    def _prepare_render_data(self) -> Dict[str, Any]:
        """Gathers all necessary data for rendering the current frame."""
        render_data = {
            "app_state": self.app_state.value,
            "status": self.status,
            "cleanup_confirmation_active": self.cleanup_confirmation_active,
            "cleanup_message": self.cleanup_message,
            "last_cleanup_message_time": self.last_cleanup_message_time,
            "update_progress_details": self.update_progress_details,
            "demo_env": self.demo_env,  # Pass the demo env object itself
            "env_config": self.env_config,
            "num_envs": self.train_config_instance.NUM_SELF_PLAY_WORKERS,
            "plot_data": {},
            "stats_summary": {},
            "best_game_state_data": None,
            "agent_param_count": 0,
            "worker_counts": {},
            "is_process_running": False,
            "worker_render_data": [],
        }

        if self.stats_aggregator:
            render_data["plot_data"] = self.stats_aggregator.get_plot_data()
            current_step = self.stats_aggregator.storage.current_global_step
            render_data["stats_summary"] = self.stats_aggregator.get_summary(
                current_step
            )
            render_data["best_game_state_data"] = (
                self.stats_aggregator.get_best_game_state_data()
            )

        if self.initializer:
            render_data["agent_param_count"] = self.initializer.agent_param_count

        if self.worker_manager:
            render_data["worker_counts"] = (
                self.worker_manager.get_active_worker_counts()
            )
            render_data["is_process_running"] = (
                self.worker_manager.is_any_worker_running()
            )
            if (
                render_data["is_process_running"]
                and self.app_state == AppState.MAIN_MENU
            ):
                num_to_render = self.vis_config.NUM_ENVS_TO_RENDER
                if num_to_render > 0:
                    render_data["worker_render_data"] = (
                        self.worker_manager.get_worker_render_data(num_to_render)
                    )

        return render_data

    def _render_frame(self, render_data: Dict[str, Any]):
        """Renders the UI frame using the collected data."""
        if self.renderer:
            self.renderer.render_all(**render_data)
        else:
            try:
                self.screen.fill((20, 0, 0))
                font = pygame.font.Font(None, 30)
                text_surf = font.render(
                    "Renderer Initialization Failed!", True, (255, 50, 50)
                )
                self.screen.blit(
                    text_surf, text_surf.get_rect(center=self.screen.get_rect().center)
                )
                pygame.display.flip()
            except Exception:
                pass

    def _log_loop_timing(self, loop_start_time: float):
        """Logs average loop time periodically."""
        self.loop_times.append(time.monotonic() - loop_start_time)
        self.frame_count += 1

    def run_main_loop(self):
        """The main application loop."""
        logger.info("Starting main application loop...")
        while self.running:
            loop_start_time = time.monotonic()

            if not self.clock or not self.screen:
                logger.error("Pygame clock or screen not initialized. Exiting.")
                break

            dt = self.clock.tick(self.vis_config.FPS) / 1000.0

            self.running = self._handle_input()
            if not self.running:
                break

            self._update_state()  # Updates demo timers if needed
            render_data = self._prepare_render_data()
            self._render_frame(render_data)

            self._log_loop_timing(loop_start_time)

        logger.info("Main application loop exited.")

    def shutdown(self):
        """Cleans up resources and exits."""
        logger.info("Initiating shutdown sequence...")
        logger.info("Stopping worker threads...")
        self.worker_manager.stop_all_workers()
        logger.info("Worker threads stopped.")
        logger.info("Attempting final checkpoint save...")
        self.logic.save_final_checkpoint()
        logger.info("Final checkpoint save attempt finished.")
        logger.info("Closing stats recorder (before pygame.quit)...")
        self.initializer.close_stats_recorder()
        logger.info("Stats recorder closed.")
        logger.info("Quitting Pygame...")
        pygame.quit()
        logger.info("Pygame quit.")
        logger.info("Shutdown complete.")


# --- Main Execution ---
tee_logger_instance: Optional[TeeLogger] = None


def setup_logging_and_run_id(args: argparse.Namespace):
    """Sets up logging (including TeeLogger) and determines the run ID."""
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
        print(f"Console output will be mirrored to: {log_file_path}")
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)

    # Configure Python's logging system
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    # Set default levels for noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    # Set default levels for our workers unless overridden by args.log_level
    if log_level <= logging.INFO:
        logging.getLogger("mcts").setLevel(logging.WARNING)  # Keep MCTS quieter
        logging.getLogger("workers.self_play_worker").setLevel(logging.INFO)
        logging.getLogger("workers.training_worker").setLevel(logging.INFO)
    # If DEBUG is requested, set workers/MCTS to DEBUG too
    if log_level == logging.DEBUG:
        logging.getLogger("mcts").setLevel(logging.DEBUG)
        logging.getLogger("workers.self_play_worker").setLevel(logging.DEBUG)
        logging.getLogger("workers.training_worker").setLevel(logging.DEBUG)

    logger.info(f"Logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {current_run_id}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")

    return original_stdout, original_stderr


def cleanup_logging(original_stdout, original_stderr, exit_code):
    """Restores standard output/error and closes logger."""
    print("[Main Finally] Restoring stdout/stderr and closing logger...")
    logger = logging.getLogger(__name__)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Trainer")
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint file (.pth) to load.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    original_stdout, original_stderr = setup_logging_and_run_id(args)

    app = None
    exit_code = 0
    try:
        app = MainApp(checkpoint_to_load=args.load_checkpoint)
        app.initialize()
        app.run_main_loop()

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Shutting down gracefully...")
        if app:
            app.logic.exit_app()
        else:
            pygame.quit()
        exit_code = 130

    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        if app:
            app.logic.exit_app()
        else:
            pygame.quit()
        exit_code = 1

    finally:
        if app:
            app.shutdown()
        cleanup_logging(original_stdout, original_stderr, exit_code)
