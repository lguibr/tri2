import pygame
import time
import traceback
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # Import scheduler
from typing import TYPE_CHECKING, List, Optional

from config import (
    ModelConfig,
    StatsConfig,
    DemoConfig,
    MCTSConfig,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
)
from environment.game_state import GameState
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregator
from stats.simple_stats_recorder import SimpleStatsRecorder
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler
from training.checkpoint_manager import CheckpointManager
from app_state import AppState
from mcts import MCTS
from agent.alphazero_net import AlphaZeroNet
from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker

if TYPE_CHECKING:
    from main_pygame import MainApp
    from torch.optim.lr_scheduler import _LRScheduler


class AppInitializer:
    """Handles the initialization of core application components."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Config instances
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        self.train_config = app.train_config_instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.demo_config = DemoConfig()
        self.mcts_config = MCTSConfig()

        # Components to be initialized
        self.agent: Optional[AlphaZeroNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional["_LRScheduler"] = None  # Add scheduler attribute
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.stats_aggregator: Optional[StatsAggregator] = None
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.mcts: Optional[MCTS] = None
        self.self_play_workers: List[SelfPlayWorker] = []
        self.training_worker: Optional[TrainingWorker] = None

    def initialize_all(self, is_reinit: bool = False):
        """Initializes all core components."""
        try:
            self._check_gpu_memory()
            if not is_reinit:
                self._initialize_ui_early()

            self.initialize_rl_components(
                is_reinit=is_reinit, checkpoint_to_load=self.app.checkpoint_to_load
            )

            if not is_reinit:
                self.initialize_demo_env()
                self.initialize_input_handler()

            self._calculate_agent_params()
            self.initialize_workers()  # Workers now need the scheduler

        except Exception as init_err:
            self._handle_init_error(init_err)

    def _check_gpu_memory(self):
        """Checks and prints total GPU memory if available."""
        if self.app.device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(self.app.device)
                self.app.total_gpu_memory_bytes = props.total_memory
                print(f"Total GPU Memory: {props.total_memory / (1024**3):.2f} GB")
            except Exception as e:
                print(f"Warning: Could not get total GPU memory: {e}")

    def _initialize_ui_early(self):
        """Initializes the renderer and performs an initial render."""
        self.app.renderer = UIRenderer(self.app.screen, self.vis_config)
        self.app.renderer.render_all(
            app_state=self.app.app_state.value,
            is_process_running=False,
            status=self.app.status,
            stats_summary={},
            envs=[],
            num_envs=0,
            env_config=self.env_config,
            cleanup_confirmation_active=False,
            cleanup_message="",
            last_cleanup_message_time=0,
            plot_data={},
            demo_env=None,
            update_progress_details={},
            agent_param_count=0,
            worker_counts={},
            best_game_state_data=None,
        )
        pygame.display.flip()
        pygame.time.delay(100)  # Allow UI to update

    def _calculate_agent_params(self):
        """Calculates the number of trainable parameters in the agent."""
        if self.agent:
            try:
                self.agent_param_count = sum(
                    p.numel() for p in self.agent.parameters() if p.requires_grad
                )
            except Exception as e:
                print(f"Warning: Could not calculate agent parameters: {e}")
                self.agent_param_count = 0

    def _handle_init_error(self, error: Exception):
        """Handles fatal errors during initialization."""
        print(f"FATAL ERROR during component initialization: {error}")
        traceback.print_exc()
        if self.app.renderer:
            try:
                self.app.app_state = AppState.ERROR
                self.app.status = "Initialization Failed"
                self.app.renderer._render_error_screen(self.app.status)
                pygame.display.flip()
                time.sleep(5)
            except Exception:
                pass
        pygame.quit()
        sys.exit(1)

    def initialize_rl_components(
        self, is_reinit: bool = False, checkpoint_to_load: Optional[str] = None
    ):
        """Initializes NN Agent, Optimizer, Scheduler, MCTS, Stats, Checkpoint Manager."""
        print(f"Initializing AlphaZero components... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            self._init_agent()
            self._init_optimizer_and_scheduler()  # Renamed method
            self._init_mcts()
            self._init_stats()
            self._init_checkpoint_manager(
                checkpoint_to_load
            )  # Checkpoint manager needs scheduler now
            print(
                f"AlphaZero components initialized in {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            print(f"Error during AlphaZero component initialization: {e}")
            traceback.print_exc()
            raise e

    def _init_agent(self):
        self.agent = AlphaZeroNet(
            env_config=self.env_config, model_config=self.model_config.Network()
        ).to(self.app.device)
        print(f"AlphaZeroNet initialized on device: {self.app.device}.")

    def _init_optimizer_and_scheduler(self):
        """Initializes the optimizer and the learning rate scheduler."""
        if not self.agent:
            raise RuntimeError("Agent must be initialized before Optimizer.")
        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.train_config.LEARNING_RATE,
            weight_decay=self.train_config.WEIGHT_DECAY,
        )
        print(f"Optimizer initialized (Adam, LR={self.train_config.LEARNING_RATE}).")

        # Initialize Scheduler (if enabled)
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
            # Add other scheduler types here if needed
            # elif self.train_config.SCHEDULER_TYPE == "OneCycleLR":
            #     self.scheduler = OneCycleLR(...)
            else:
                print(
                    f"Warning: Unknown scheduler type '{self.train_config.SCHEDULER_TYPE}'. No scheduler initialized."
                )
                self.scheduler = None
        else:
            print("LR Scheduler is DISABLED.")
            self.scheduler = None

    def _init_mcts(self):
        if not self.agent:
            raise RuntimeError("Agent must be initialized before MCTS.")
        self.mcts = MCTS(
            network_predictor=self.agent.predict,
            config=self.mcts_config,
            env_config=self.env_config,
        )
        print("MCTS initialized with AlphaZeroNet predictor.")

    def _init_stats(self):
        print("Initializing StatsAggregator and SimpleStatsRecorder...")
        self.stats_aggregator = StatsAggregator(
            avg_windows=self.stats_config.STATS_AVG_WINDOW,
            plot_window=self.stats_config.PLOT_DATA_WINDOW,
        )
        self.stats_recorder = SimpleStatsRecorder(
            aggregator=self.stats_aggregator,
            console_log_interval=self.stats_config.CONSOLE_LOG_FREQ,
            train_config=self.train_config,
        )
        print("StatsAggregator and SimpleStatsRecorder initialized.")

    def _init_checkpoint_manager(self, checkpoint_to_load: Optional[str]):
        if not self.agent or not self.optimizer or not self.stats_aggregator:
            raise RuntimeError(
                "Agent, Optimizer, StatsAggregator needed for CheckpointManager."
            )
        # Pass the scheduler to the CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            agent=self.agent,
            optimizer=self.optimizer,
            scheduler=self.scheduler,  # Pass scheduler
            stats_aggregator=self.stats_aggregator,
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            run_checkpoint_dir=get_run_checkpoint_dir(),
            load_checkpoint_path_config=checkpoint_to_load,
            device=self.app.device,
        )
        if self.checkpoint_manager.get_checkpoint_path_to_load():
            self.checkpoint_manager.load_checkpoint()

    def initialize_workers(self):
        """Initializes worker threads (Self-Play, Training). Does NOT start them."""
        print("Initializing worker threads...")
        if (
            not self.agent
            or not self.mcts
            or not self.stats_aggregator
            or not self.optimizer
            # Scheduler is optional, so don't check it here
        ):
            print("ERROR: Cannot initialize workers, core RL components missing.")
            return

        self._init_self_play_workers()
        self._init_training_worker()  # Training worker needs scheduler
        num_sp = len(self.self_play_workers)
        print(f"Worker threads initialized ({num_sp} Self-Play, 1 Training).")

    def _init_self_play_workers(self):
        self.self_play_workers = []
        num_sp_workers = self.train_config.NUM_SELF_PLAY_WORKERS
        print(f"Initializing {num_sp_workers} SelfPlayWorker(s)...")
        for i in range(num_sp_workers):
            worker = SelfPlayWorker(
                worker_id=i,
                agent=self.agent,
                mcts=self.mcts,
                experience_queue=self.app.experience_queue,
                stats_aggregator=self.stats_aggregator,
                stop_event=self.app.stop_event,
                env_config=self.env_config,
                mcts_config=self.mcts_config,
                device=self.app.device,
            )
            self.self_play_workers.append(worker)
            print(f"  SelfPlayWorker-{i} initialized.")

    def _init_training_worker(self):
        # Pass the scheduler to the TrainingWorker
        self.training_worker = TrainingWorker(
            agent=self.agent,
            optimizer=self.optimizer,
            scheduler=self.scheduler,  # Pass scheduler
            experience_queue=self.app.experience_queue,
            stats_aggregator=self.stats_aggregator,
            stop_event=self.app.stop_event,
            train_config=self.train_config,
            device=self.app.device,
        )
        print("TrainingWorker initialized.")

    def initialize_demo_env(self):
        """Initializes the separate environment for demo/debug mode."""
        print("Initializing Demo/Debug Environment...")
        try:
            self.demo_env = GameState()
            self.demo_env.reset()
            print("Demo/Debug environment initialized.")
        except Exception as e:
            print(f"ERROR initializing demo/debug environment: {e}")
            traceback.print_exc()
            self.demo_env = None

    def initialize_input_handler(self):
        """Initializes the Input Handler."""
        if not self.app.renderer:
            print("ERROR: Cannot initialize InputHandler before Renderer.")
            return
        self.app.input_handler = InputHandler(
            screen=self.app.screen,
            renderer=self.app.renderer,
            request_cleanup_cb=self.app.logic.request_cleanup,
            cancel_cleanup_cb=self.app.logic.cancel_cleanup,
            confirm_cleanup_cb=self.app.logic.confirm_cleanup,
            exit_app_cb=self.app.logic.exit_app,
            start_demo_mode_cb=self.app.logic.start_demo_mode,
            exit_demo_mode_cb=self.app.logic.exit_demo_mode,
            handle_demo_mouse_motion_cb=self.app.logic.handle_demo_mouse_motion,
            handle_demo_mouse_button_down_cb=self.app.logic.handle_demo_mouse_button_down,
            start_debug_mode_cb=self.app.logic.start_debug_mode,
            exit_debug_mode_cb=self.app.logic.exit_debug_mode,
            handle_debug_input_cb=self.app.logic.handle_debug_input,
            start_run_cb=self.app.logic.start_run,
            stop_run_cb=self.app.logic.stop_run,
        )
        if self.app.input_handler:
            self.app.input_handler.app_ref = self.app
        if self.app.renderer and self.app.renderer.left_panel:
            self.app.renderer.left_panel.input_handler = self.app.input_handler
            if hasattr(self.app.renderer.left_panel, "button_status_renderer"):
                btn_renderer = self.app.renderer.left_panel.button_status_renderer
                btn_renderer.input_handler_ref = self.app.input_handler
                btn_renderer.app_ref = self.app

    def close_stats_recorder(self, is_cleanup: bool = False):
        """Safely closes the stats recorder (if it exists)."""
        print(
            f"[AppInitializer] close_stats_recorder called (is_cleanup={is_cleanup})..."
        )
        if self.stats_recorder and hasattr(self.stats_recorder, "close"):
            print("[AppInitializer] Stats recorder exists, attempting close...")
            try:
                self.stats_recorder.close(is_cleanup=is_cleanup)
                print("[AppInitializer] stats_recorder.close() executed.")
            except Exception as log_e:
                print(f"[AppInitializer] Error closing stats recorder on exit: {log_e}")
                traceback.print_exc()
        else:
            print("[AppInitializer] No stats recorder instance or close method.")
        print("[AppInitializer] close_stats_recorder finished.")
