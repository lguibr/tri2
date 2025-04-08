# File: main_pygame.py
import pygame
import sys
import time
import threading
import logging
import argparse
import os
import traceback
from typing import Optional, List, Dict, Any
import torch
import torch.optim as optim
import queue

try:
    import psutil
except ImportError:
    print("Warning: psutil not found.")
    psutil = None

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from config import (
        VisConfig,
        EnvConfig,
        RNNConfig,
        ModelConfig,
        StatsConfig,
        TrainConfig,
        TensorBoardConfig,
        DemoConfig,
        TransformerConfig,
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
    print(f"Error importing config/utils: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during import: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from environment.game_state import GameState
    from ui.renderer import UIRenderer
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
    from app_state import AppState
    from app_init import AppInitializer
    from app_logic import AppLogic
    from app_workers import AppWorkerManager
    from app_setup import (
        initialize_pygame,
        initialize_directories,
        load_and_validate_configs,
    )
    from app_ui_utils import AppUIUtils
    from ui.input_handler import InputHandler
    from agent.alphazero_net import AlphaZeroNet
except ImportError as e:
    print(f"Error importing app components: {e}")
    traceback.print_exc()
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MainApp:
    """Main application class orchestrating Pygame UI and AlphaZero components."""

    def __init__(self, checkpoint_to_load: Optional[str] = None):
        # --- Configuration ---
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        self.rnn_config = RNNConfig()
        self.train_config_instance = TrainConfig()
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        self.transformer_config = TransformerConfig()
        self.mcts_config = MCTSConfig()
        self.config_dict = get_config_dict()

        # --- Core Components ---
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None

        # --- State ---
        self.app_state: AppState = AppState.INITIALIZING
        self.status: str = "Initializing..."
        self.running: bool = True
        self.update_progress_details: Dict[str, Any] = {}

        # --- Threading & Communication ---
        self.stop_event = threading.Event()
        self.experience_queue: queue.Queue = queue.Queue(
            maxsize=self.train_config_instance.BUFFER_CAPACITY
        )

        # --- RL Components (Managed by Initializer) ---
        self.envs: List[GameState] = []
        self.agent: Optional[AlphaZeroNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.demo_env: Optional[GameState] = None

        # --- Helper Classes ---
        self.device = get_torch_device()
        set_device(self.device)
        self.checkpoint_to_load = checkpoint_to_load
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
        self.resource_update_interval: float = 1.0

    def initialize(self):
        """Initializes Pygame, directories, configs, and core components."""
        logger.info("--- Application Initialization ---")
        self.screen, self.clock = initialize_pygame(self.vis_config)
        initialize_directories()
        set_random_seeds(RANDOM_SEED)
        run_pre_checks()

        self.app_state = AppState.INITIALIZING
        self.initializer.initialize_all()

        self.optimizer = self.initializer.optimizer

        if self.renderer and self.initializer.app.input_handler:
            self.renderer.set_input_handler(self.initializer.app.input_handler)

        self.logic.check_initial_completion_status()
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
        aggregator = self.initializer.stats_aggregator
        if not aggregator:
            return

        storage = aggregator.storage
        cpu_percent, mem_percent, gpu_mem_percent = 0.0, 0.0, 0.0
        if psutil:
            try:
                cpu_percent = psutil.cpu_percent()
                mem_percent = psutil.virtual_memory().percent
            except Exception as e:
                logger.warning(f"Error getting CPU/Mem usage: {e}")
        if self.device.type == "cuda" and self.total_gpu_memory_bytes:
            try:
                allocated = torch.cuda.memory_allocated(self.device)
                gpu_mem_percent = (allocated / self.total_gpu_memory_bytes) * 100.0
            except Exception as e:
                logger.warning(f"Error getting GPU memory usage: {e}")
                gpu_mem_percent = 0.0
        with aggregator._lock:
            storage.current_cpu_usage = cpu_percent
            storage.current_memory_usage = mem_percent
            storage.current_gpu_memory_usage_percent = gpu_mem_percent
            if hasattr(storage, "cpu_usage"):
                storage.cpu_usage.append(cpu_percent)
            if hasattr(storage, "memory_usage"):
                storage.memory_usage.append(mem_percent)
            if hasattr(storage, "gpu_memory_usage_percent"):
                storage.gpu_memory_usage_percent.append(gpu_mem_percent)
        self.last_resource_update_time = current_time

    def run_main_loop(self):
        """The main application loop."""
        logger.info("Starting main application loop...")
        while self.running:
            dt = self.clock.tick(self.vis_config.FPS) / 1000.0

            if self.input_handler:
                self.running = self.input_handler.handle_input(
                    self.app_state.value, self.cleanup_confirmation_active
                )
                if not self.running:
                    break
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        self.stop_event.set()
                        self.worker_manager.stop_all_workers()
                        break
                if not self.running:
                    break

            self.update_progress_details = {}
            self.logic.update_status_and_check_completion()
            self._update_resource_stats()
            if self.initializer.demo_env:
                try:
                    pass
                except Exception as timer_err:
                    logger.error(
                        f"Error updating demo env timers: {timer_err}", exc_info=False
                    )

            if self.renderer:
                plot_data = {}
                stats_summary = {}
                tb_log_dir = None
                agent_params = 0
                best_game_state_data = None  # Initialize

                aggregator = self.initializer.stats_aggregator
                if aggregator:
                    plot_data = aggregator.get_plot_data()
                    current_step = getattr(aggregator.storage, "current_global_step", 0)
                    stats_summary = aggregator.get_summary(current_step)
                    best_game_state_data = stats_summary.get(
                        "best_game_state_data"
                    )  # Fetch best state data

                    if self.initializer.stats_recorder and isinstance(
                        self.initializer.stats_recorder, TensorBoardStatsRecorder
                    ):
                        tb_log_dir = self.initializer.stats_recorder.log_dir
                if self.initializer.agent:
                    agent_params = self.initializer.agent_param_count

                is_process_running = self.worker_manager.is_any_worker_running()

                self.renderer.render_all(
                    app_state=self.app_state.value,
                    is_process_running=is_process_running,
                    status=self.status,
                    stats_summary=stats_summary,
                    envs=self.initializer.envs,  # Pass empty list when running
                    num_envs=self.env_config.NUM_ENVS,
                    env_config=self.env_config,
                    cleanup_confirmation_active=self.cleanup_confirmation_active,
                    cleanup_message=self.cleanup_message,
                    last_cleanup_message_time=self.last_cleanup_message_time,
                    tensorboard_log_dir=tb_log_dir,
                    plot_data=plot_data,
                    demo_env=self.initializer.demo_env,
                    update_progress_details=self.update_progress_details,
                    agent_param_count=agent_params,
                    worker_counts={},
                    best_game_state_data=best_game_state_data,  # Pass best state data
                )
            else:
                self.screen.fill((20, 0, 0))
                font = pygame.font.Font(None, 30)
                text_surf = font.render("Renderer Error", True, (255, 50, 50))
                self.screen.blit(
                    text_surf, text_surf.get_rect(center=self.screen.get_rect().center)
                )
                pygame.display.flip()
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


tee_logger_instance: Optional[TeeLogger] = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaTri Trainer")
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint file to load.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level.",
    )
    args = parser.parse_args()

    if args.load_checkpoint:
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                print(f"Using Run ID from checkpoint path: {get_run_id()}")
            else:
                get_run_id()
        except Exception:
            get_run_id()
    else:
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            print(f"Resuming Run ID: {get_run_id()}")
        else:
            get_run_id()

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
        logger.critical(f"An unhandled exception occurred in main: {e}", exc_info=True)
        if app:
            app.logic.exit_app()
            app.shutdown()
        else:
            pygame.quit()
        exit_code = 1
    finally:
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
