import pygame
import sys
import time
import threading
import logging
import argparse
import os
import traceback
from typing import Optional, Dict, Any
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
except ImportError as e:
    print(f"Error importing app components: {e}\n{traceback.format_exc()}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
LOOP_TIMING_INTERVAL = 60  # Log loop timing every N frames


class MainApp:
    """Main application class orchestrating Pygame UI and AlphaZero components."""

    def __init__(self, checkpoint_to_load: Optional[str] = None):
        # Config Instances (Keep essential ones)
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
        self.initializer.initialize_all()  # Delegates complex init

        # Get references after initialization
        self.agent = self.initializer.agent
        self.stats_aggregator = self.initializer.stats_aggregator
        self.demo_env = self.initializer.demo_env

        if self.renderer and self.input_handler:
            self.renderer.set_input_handler(self.input_handler)

        self.logic.check_initial_completion_status()
        self.status = "Ready"
        self.app_state = AppState.MAIN_MENU
        logger.info("--- Initialization Complete ---")

    def _handle_input(self) -> bool:
        """Handles user input."""
        if self.input_handler:
            return self.input_handler.handle_input(
                self.app_state.value, self.cleanup_confirmation_active
            )
        else:
            # Basic quit handling if input handler fails
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop_event.set()
                    self.worker_manager.stop_all_workers()
                    return False
            return True

    def _update_state(self):
        """Updates application logic and status."""
        self.update_progress_details = {}  # Reset progress details
        self.logic.update_status_and_check_completion()

    def _prepare_render_data(self) -> Dict[str, Any]:
        """Gathers data needed for rendering."""
        render_data = {
            "plot_data": {},
            "stats_summary": {},
            "agent_params": 0,
            "best_game_state_data": None,
            "worker_counts": {},
            "is_process_running": False,
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

        if self.agent:
            render_data["agent_params"] = self.initializer.agent_param_count

        render_data["worker_counts"] = self.worker_manager.get_active_worker_counts()
        render_data["is_process_running"] = self.worker_manager.is_any_worker_running()
        return render_data

    def _render_frame(self, render_data: Dict[str, Any]):
        """Renders the UI frame."""
        if self.renderer:
            self.renderer.render_all(
                app_state=self.app_state.value,
                is_process_running=render_data["is_process_running"],
                status=self.status,
                stats_summary=render_data["stats_summary"],
                envs=[],  # envs list is not actively used for rendering AZ
                num_envs=self.train_config_instance.NUM_SELF_PLAY_WORKERS,
                env_config=self.env_config,
                cleanup_confirmation_active=self.cleanup_confirmation_active,
                cleanup_message=self.cleanup_message,
                last_cleanup_message_time=self.last_cleanup_message_time,
                plot_data=render_data["plot_data"],
                demo_env=self.demo_env,
                update_progress_details=self.update_progress_details,
                agent_param_count=render_data["agent_params"],
                worker_counts=render_data["worker_counts"],
                best_game_state_data=render_data["best_game_state_data"],
            )
        else:  # Fallback rendering
            self.screen.fill((20, 0, 0))
            font = pygame.font.Font(None, 30)
            text_surf = font.render("Renderer Error", True, (255, 50, 50))
            self.screen.blit(
                text_surf, text_surf.get_rect(center=self.screen.get_rect().center)
            )
            pygame.display.flip()

    def _log_loop_timing(self, loop_start_time: float):
        """Logs average loop time periodically."""
        self.loop_times.append(time.monotonic() - loop_start_time)
        self.frame_count += 1

    def run_main_loop(self):
        """The main application loop."""
        logger.info("Starting main application loop...")
        while self.running:
            loop_start_time = time.monotonic()
            if not self.clock:
                break  # Exit if clock not initialized
            dt = self.clock.tick(self.vis_config.FPS) / 1000.0

            self.running = self._handle_input()
            if not self.running:
                break

            self._update_state()
            render_data = self._prepare_render_data()
            self._render_frame(render_data)
            self._log_loop_timing(loop_start_time)

        logger.info("Main application loop exited.")

    def shutdown(self):
        """Cleans up resources and exits."""
        logger.info("Initiating shutdown sequence...")
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
    """Sets up logging and determines the run ID."""
    global tee_logger_instance
    if args.load_checkpoint:
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                print(f"Using Run ID from checkpoint path: {get_run_id()}")
            else:
                get_run_id()  # Generate new if path doesn't contain valid ID
        except Exception:
            get_run_id()
    else:
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            print(f"Resuming Run ID: {get_run_id()}")
        else:
            get_run_id()  # Generate new if no runs found

    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        log_file_dir = get_console_log_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, "console_output.log")
        tee_logger_instance = TeeLogger(log_file_path, sys.stdout)
        sys.stdout = tee_logger_instance
        sys.stderr = tee_logger_instance
    except Exception as e:
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.info(f"Logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {get_run_id()}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")
    return original_stdout, original_stderr


def cleanup_logging(original_stdout, original_stderr, exit_code):
    """Restores standard output/error and closes logger."""
    print("[Main Finally] Restoring stdout/stderr and closing logger...")
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
    parser = argparse.ArgumentParser(description="AlphaTri Trainer")
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

    app = None
    exit_code = 0
    try:
        app = MainApp(checkpoint_to_load=args.load_checkpoint)
        app.initialize()
        app.run_main_loop()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down.")
        if app:
            app.logic.exit_app()
            app.shutdown()
        else:
            pygame.quit()
        exit_code = 130
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        if app:
            app.logic.exit_app()
            app.shutdown()
        else:
            pygame.quit()
        exit_code = 1
    finally:
        cleanup_logging(original_stdout, original_stderr, exit_code)
