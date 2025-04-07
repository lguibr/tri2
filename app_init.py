# File: app_init.py
import pygame
import time
import traceback
import sys
import torch
from typing import TYPE_CHECKING, List, Optional, Dict, Any

from config import (
    VisConfig,
    EnvConfig,
    # Removed PPOConfig
    RNNConfig,
    ModelConfig,
    StatsConfig,
    # Removed RewardConfig
    TensorBoardConfig,
    DemoConfig,
    # Removed ObsNormConfig
    TransformerConfig,
    RANDOM_SEED,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
    DEVICE,
)
from environment.game_state import GameState

# Removed PPOAgent import
# from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler

# Removed init.rl_components_ppo import
from training.checkpoint_manager import CheckpointManager

# Removed RolloutCollector import
from app_state import AppState

if TYPE_CHECKING:
    from main_pygame import MainApp
    from agent.base_agent import BaseAgent  # Hypothetical base class for NN agent


class AppInitializer:
    """Handles the initialization of core application components."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Config instances
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        # Removed self.ppo_config
        self.rnn_config = RNNConfig()
        self.train_config = app.train_config_instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        # Removed self.reward_config
        # Removed self.obs_norm_config
        self.transformer_config = TransformerConfig()

        # Components to be initialized
        self.envs: List[GameState] = []  # Keep for potential multi-env display
        self.agent: Optional["BaseAgent"] = None  # Agent is now the NN
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        # Removed self.rollout_collector

    def initialize_all(self, is_reinit: bool = False):
        """Initializes all core components."""
        try:
            # GPU Memory Info (Keep)
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

            # Renderer and Initial Render (Keep)
            if not is_reinit:
                self.app.renderer = UIRenderer(self.app.screen, self.vis_config)
                # Adapt render_all call later if needed
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=False,  # No PPO process running
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
                    update_progress_details={},  # Keep for potential NN training progress
                    agent_param_count=0,
                    worker_counts={},  # Remove worker counts for now
                )
                pygame.display.flip()
                pygame.time.delay(100)

            # Initialize "RL" components (NN, Stats, Checkpoint Manager)
            self.initialize_rl_components(
                is_reinit=is_reinit, checkpoint_to_load=self.app.checkpoint_to_load
            )

            # Demo Env and Input Handler (Keep)
            if not is_reinit:
                self.initialize_demo_env()
                self.initialize_input_handler()

            # Calculate NN parameter count if agent exists
            if self.agent and hasattr(self.agent, "network"):
                try:
                    self.agent_param_count = sum(
                        p.numel()
                        for p in self.agent.network.parameters()
                        if p.requires_grad
                    )
                except Exception as e:
                    print(f"Warning: Could not calculate agent parameters: {e}")
                    self.agent_param_count = 0

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
        """Initializes NN Agent, Stats Recorder, Checkpoint Manager."""
        print(f"Initializing AlphaZero components... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            # Initialize Envs (only needed for visualization now)
            # self.envs = initialize_envs(self.env_config.NUM_ENVS, self.env_config) # Removed multi-env init
            self.envs = []  # No parallel envs needed for core logic now

            # --- Initialize Agent (Neural Network) ---
            # Replace with AlphaZero NN initialization later
            # For now, set agent to None or a placeholder
            self.agent = None  # Placeholder - Initialize AlphaZero NN here later
            print("Agent (NN) initialization SKIPPED (placeholder).")
            # Example placeholder for future NN init:
            # self.agent = initialize_alphazero_agent(
            #     model_config=self.model_config,
            #     rnn_config=self.rnn_config,
            #     env_config=self.env_config,
            #     transformer_config=self.transformer_config,
            #     device=self.app.device,
            # )
            # --- End Agent Init ---

            # --- Initialize Stats Recorder ---
            # Adapt initialize_stats_recorder if needed (e.g., remove PPO hparams)
            # Need to create a simplified version or adapt existing one
            # For now, assume it's adapted or create a placeholder
            try:
                # Assuming init.stats_init exists and is adapted
                from init.stats_init import initialize_stats_recorder

                self.stats_recorder = initialize_stats_recorder(
                    stats_config=self.stats_config,
                    tb_config=self.tensorboard_config,
                    config_dict=self.app.config_dict,
                    # Pass agent=None if NN not ready, or pass the NN agent
                    agent=self.agent,
                    env_config=self.env_config,
                    rnn_config=self.rnn_config,  # Keep for potential NN config logging
                    transformer_config=self.transformer_config,  # Keep for potential NN config logging
                    is_reinit=is_reinit,
                )
            except ImportError:
                print(
                    "Warning: init.stats_init.initialize_stats_recorder not found. Skipping stats recorder init."
                )
                self.stats_recorder = None  # Fallback
            except Exception as stats_init_err:
                print(f"Error initializing stats recorder: {stats_init_err}")
                traceback.print_exc()
                self.stats_recorder = None

            if self.stats_recorder is None:
                print("Warning: Stats Recorder initialization failed or skipped.")
                # Decide if this is critical - maybe allow running without stats?
                # raise RuntimeError("Stats Recorder init failed.")
            # --- End Stats Recorder Init ---

            # --- Initialize Checkpoint Manager ---
            # Checkpoint manager now handles NN agent state and stats aggregator state
            self.checkpoint_manager = CheckpointManager(
                # Pass the NN agent (or None if not ready)
                agent=self.agent,
                # Pass stats aggregator if it exists
                stats_aggregator=getattr(self.stats_recorder, "aggregator", None),
                base_checkpoint_dir=BASE_CHECKPOINT_DIR,
                run_checkpoint_dir=get_run_checkpoint_dir(),
                load_checkpoint_path_config=checkpoint_to_load,
                device=self.app.device,
                # obs_rms_dict=None, # Removed Obs RMS
            )
            # --- End Checkpoint Manager Init ---

            # --- Load Checkpoint ---
            if self.checkpoint_manager.get_checkpoint_path_to_load():
                self.checkpoint_manager.load_checkpoint()  # Loads NN state and stats state
                # Get initial step/episode count from loaded stats
                loaded_global_step, initial_episode_count = (
                    self.checkpoint_manager.get_initial_state()
                )
                # Sync episode count if needed (e.g., if MCTS tracks episodes)
                # self.mcts_manager.state.episode_count = initial_episode_count # Example
            # --- End Load Checkpoint ---

            print(
                f"AlphaZero components initialized in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            print(f"Error during AlphaZero component initialization: {e}")
            traceback.print_exc()
            raise e  # Re-raise to be caught by initialize_all

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
            # Removed toggle_training_run_cb
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
        )
        if self.app.renderer and self.app.renderer.left_panel:
            self.app.renderer.left_panel.input_handler = self.app.input_handler
            if hasattr(self.app.renderer.left_panel, "button_status_renderer"):
                self.app.renderer.left_panel.button_status_renderer.input_handler_ref = (
                    self.app.input_handler
                )

    def close_stats_recorder(self, is_cleanup: bool = False):
        """Safely closes the stats recorder."""
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
            print("[AppInitializer] No stats recorder instance to close.")
        print("[AppInitializer] close_stats_recorder finished.")
