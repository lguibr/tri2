# File: app_init.py
# File: app_init.py
import pygame
import time
import traceback
import sys
import torch
import torch.optim as optim
import queue
from typing import TYPE_CHECKING, List, Optional, Dict, Any

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
    get_run_checkpoint_dir,
    DEVICE,
)
from environment.game_state import GameState
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregator
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler
from training.checkpoint_manager import CheckpointManager
from app_state import AppState
from mcts import MCTS  # Import MCTS
from agent.alphazero_net import AlphaZeroNet
from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker

if TYPE_CHECKING:
    from main_pygame import MainApp
    from mcts.node import MCTSNode


class AppInitializer:
    """Handles the initialization of core application components."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Config instances
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        self.rnn_config = RNNConfig()
        self.train_config = app.train_config_instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        self.transformer_config = TransformerConfig()
        self.mcts_config = MCTSConfig()

        # Components to be initialized
        self.envs: List[GameState] = []
        self.agent: Optional[AlphaZeroNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.stats_aggregator: Optional[StatsAggregator] = None
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.mcts: Optional[MCTS] = None  # Added MCTS instance
        # self.mcts_root_node: Optional["MCTSNode"] = None # MCTS Vis removed
        self.self_play_worker: Optional[SelfPlayWorker] = None
        self.training_worker: Optional[TrainingWorker] = None

    def initialize_all(self, is_reinit: bool = False):
        """Initializes all core components."""
        try:
            if self.app.device.type == "cuda":
                try:
                    self.app.total_gpu_memory_bytes = torch.cuda.get_device_properties(
                        self.app.device
                    ).total_memory
                    print(
                        f"Total GPU Memory: {self.app.total_gpu_memory_bytes / (1024**3):.2f} GB"
                    )
                except Exception as e:
                    print(f"Warning: Could not get total GPU memory: {e}")

            if not is_reinit:
                self.app.renderer = UIRenderer(self.app.screen, self.vis_config)
                # MCTS Visualizer removed
                # if self.app.renderer.mcts_visualizer and self.app.renderer.game_area:
                #     self.app.renderer.mcts_visualizer.set_game_area_renderer(
                #         self.app.renderer.game_area
                #     )

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
                    tensorboard_log_dir=None,
                    plot_data={},
                    demo_env=None,
                    update_progress_details={},
                    agent_param_count=0,
                    worker_counts={},
                    # mcts_root_node=None, # MCTS Vis removed
                )
                pygame.display.flip()
                pygame.time.delay(100)

            self.initialize_rl_components(
                is_reinit=is_reinit, checkpoint_to_load=self.app.checkpoint_to_load
            )

            if not is_reinit:
                self.initialize_demo_env()
                self.initialize_input_handler()

            if self.agent:
                try:
                    self.agent_param_count = sum(
                        p.numel() for p in self.agent.parameters() if p.requires_grad
                    )
                except Exception as e:
                    print(f"Warning: Could not calculate agent parameters: {e}")
                    self.agent_param_count = 0

            self.initialize_workers()

        except Exception as init_err:
            print(f"FATAL ERROR during component initialization: {init_err}")
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
        """Initializes NN Agent, Optimizer, MCTS, Stats, Checkpoint Manager."""
        print(f"Initializing AlphaZero components... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            self.envs = []
            self.agent = AlphaZeroNet(
                env_config=self.env_config, model_config=self.model_config.Network()
            ).to(self.app.device)
            print(f"AlphaZeroNet initialized on device: {self.app.device}.")

            self.optimizer = optim.Adam(
                self.agent.parameters(),
                lr=self.train_config.LEARNING_RATE,
                weight_decay=self.train_config.WEIGHT_DECAY,
            )
            print(
                f"Optimizer initialized (Adam, LR={self.train_config.LEARNING_RATE})."
            )

            if self.agent is None:
                raise RuntimeError("Agent (NN) must be initialized before MCTS.")
            # Instantiate MCTS with the agent's predict method
            self.mcts = MCTS(
                network_predictor=self.agent.predict,
                config=self.mcts_config,
                env_config=self.env_config,
            )
            print("MCTS initialized with AlphaZeroNet predictor.")

            self.stats_recorder = None
            self.stats_aggregator = None
            try:
                from init.stats_init import initialize_stats_recorder

                self.stats_recorder = initialize_stats_recorder(
                    stats_config=self.stats_config,
                    tb_config=self.tensorboard_config,
                    config_dict=self.app.config_dict,
                    agent=self.agent,
                    env_config=self.env_config,
                    rnn_config=self.rnn_config,
                    transformer_config=self.transformer_config,
                    is_reinit=is_reinit,
                )
                if self.stats_recorder and hasattr(self.stats_recorder, "aggregator"):
                    self.stats_aggregator = self.stats_recorder.aggregator
                    print("StatsRecorder initialized, using its aggregator.")
                else:
                    print("StatsRecorder initialized but has no aggregator attribute.")
            except ImportError:
                print(
                    "Warning: init.stats_init not found. Skipping full stats recorder."
                )
            except Exception as stats_init_err:
                print(f"Error initializing stats recorder: {stats_init_err}")
                traceback.print_exc()

            if self.stats_aggregator is None:
                print("Creating standalone StatsAggregator as fallback.")
                self.stats_aggregator = StatsAggregator(
                    avg_windows=self.stats_config.STATS_AVG_WINDOW,
                    plot_window=self.stats_config.PLOT_DATA_WINDOW,
                )

            self.checkpoint_manager = CheckpointManager(
                agent=self.agent,
                optimizer=self.optimizer,
                stats_aggregator=self.stats_aggregator,
                base_checkpoint_dir=BASE_CHECKPOINT_DIR,
                run_checkpoint_dir=get_run_checkpoint_dir(),
                load_checkpoint_path_config=checkpoint_to_load,
                device=self.app.device,
            )

            if self.checkpoint_manager.get_checkpoint_path_to_load():
                self.checkpoint_manager.load_checkpoint()
                loaded_global_step, initial_episode_count = (
                    self.checkpoint_manager.get_initial_state()
                )

            print(
                f"AlphaZero components initialized in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            print(f"Error during AlphaZero component initialization: {e}")
            traceback.print_exc()
            raise e

    def initialize_workers(self):
        """Initializes worker threads (Self-Play, Training). Does NOT start them."""
        print("Initializing worker threads...")
        if (
            not self.agent
            or not self.mcts
            or not self.stats_aggregator
            or not self.optimizer
        ):
            print("ERROR: Cannot initialize workers, core RL components missing.")
            return

        # Instantiate SelfPlayWorker
        self.self_play_worker = SelfPlayWorker(
            worker_id=0,
            agent=self.agent,
            mcts=self.mcts,
            experience_queue=self.app.experience_queue,
            stats_aggregator=self.stats_aggregator,
            stop_event=self.app.stop_event,
            env_config=self.env_config,
            mcts_config=self.mcts_config,
            device=self.app.device,
        )
        print("SelfPlayWorker initialized.")

        # Instantiate TrainingWorker
        self.training_worker = TrainingWorker(
            agent=self.agent,
            optimizer=self.optimizer,
            experience_queue=self.app.experience_queue,
            stats_aggregator=self.stats_aggregator,
            stop_event=self.app.stop_event,
            train_config=self.train_config,
            device=self.app.device,
        )
        print("TrainingWorker initialized.")
        print("Worker threads initialized.")

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
            # Basic Callbacks
            request_cleanup_cb=self.app.logic.request_cleanup,
            cancel_cleanup_cb=self.app.logic.cancel_cleanup,
            confirm_cleanup_cb=self.app.logic.confirm_cleanup,
            exit_app_cb=self.app.logic.exit_app,
            # Demo Mode Callbacks
            start_demo_mode_cb=self.app.logic.start_demo_mode,
            exit_demo_mode_cb=self.app.logic.exit_demo_mode,
            handle_demo_mouse_motion_cb=self.app.logic.handle_demo_mouse_motion,
            handle_demo_mouse_button_down_cb=self.app.logic.handle_demo_mouse_button_down,
            # Debug Mode Callbacks
            start_debug_mode_cb=self.app.logic.start_debug_mode,
            exit_debug_mode_cb=self.app.logic.exit_debug_mode,
            handle_debug_input_cb=self.app.logic.handle_debug_input,
            # MCTS Vis Callbacks (Removed)
            # start_mcts_visualization_cb=self.app.logic.start_mcts_visualization,
            # exit_mcts_visualization_cb=self.app.logic.exit_mcts_visualization,
            # handle_mcts_pan_cb=self.app.logic.handle_mcts_pan,
            # handle_mcts_zoom_cb=self.app.logic.handle_mcts_zoom,
            # Combined Worker Control Callbacks
            start_run_cb=self.app.logic.start_run,
            stop_run_cb=self.app.logic.stop_run,
        )
        # Provide app reference to input handler AFTER it's created
        if self.app.input_handler:
            self.app.input_handler.app_ref = self.app

        if self.app.renderer and self.app.renderer.left_panel:
            self.app.renderer.left_panel.input_handler = self.app.input_handler
            if hasattr(self.app.renderer.left_panel, "button_status_renderer"):
                self.app.renderer.left_panel.button_status_renderer.input_handler_ref = (
                    self.app.input_handler
                )
                # Also provide app_ref to button renderer if needed
                self.app.renderer.left_panel.button_status_renderer.app_ref = self.app

    def close_stats_recorder(self, is_cleanup: bool = False):
        """Safely closes the stats recorder (if it exists)."""
        print(
            f"[AppInitializer] close_stats_recorder called (is_cleanup={is_cleanup})..."
        )
        if self.stats_recorder:
            print("[AppInitializer] Stats recorder exists, attempting close...")
            try:
                if hasattr(self.stats_recorder, "close"):
                    self.stats_recorder.close(is_cleanup=is_cleanup)
                    print("[AppInitializer] stats_recorder.close() executed.")
                else:
                    print("[AppInitializer] stats_recorder has no close method.")
            except Exception as log_e:
                print(f"[AppInitializer] Error closing stats recorder on exit: {log_e}")
                traceback.print_exc()
        else:
            print(
                "[AppInitializer] No stats recorder instance to close (might be using standalone aggregator)."
            )
        print("[AppInitializer] close_stats_recorder finished.")
