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
    PPOConfig,
    RNNConfig,
    ModelConfig,
    StatsConfig,
    RewardConfig,
    TensorBoardConfig,
    DemoConfig,
    ObsNormConfig,
    TransformerConfig,
    RANDOM_SEED,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
    DEVICE,
)
from environment.game_state import GameState
from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler
from init.rl_components_ppo import (
    initialize_envs,
    initialize_agent,
    initialize_stats_recorder,
)
from training.checkpoint_manager import CheckpointManager
from training.rollout_collector import RolloutCollector
from app_state import AppState

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppInitializer:
    """Handles the initialization of core application components."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Config instances (can be accessed via app as well)
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        self.ppo_config = app.ppo_config
        self.rnn_config = RNNConfig()
        # --- Access the stored instance from MainApp ---
        self.train_config = app.train_config_instance
        # --- End Access ---
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        self.reward_config = RewardConfig()
        self.obs_norm_config = ObsNormConfig()
        self.transformer_config = TransformerConfig()

        # Components to be initialized
        self.envs: List[GameState] = []
        self.agent: Optional[PPOAgent] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.rollout_collector: Optional[RolloutCollector] = None

    def initialize_all(self, is_reinit: bool = False):
        """Initializes all core components."""
        try:
            # --- GPU Memory Info ---
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

            # --- Renderer and Initial Render ---
            if not is_reinit:
                self.app.renderer = UIRenderer(self.app.screen, self.vis_config)
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=self.app.is_process_running,
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
                )
                pygame.display.flip()
                pygame.time.delay(100)

            # --- RL Components ---
            self.initialize_rl_components(
                is_reinit=is_reinit, checkpoint_to_load=self.app.checkpoint_to_load
            )

            # --- Demo Env and Input Handler ---
            if not is_reinit:
                self.initialize_demo_env()
                self.initialize_input_handler()

            if self.agent:
                self.agent_param_count = sum(
                    p.numel()
                    for p in self.agent.network.parameters()
                    if p.requires_grad
                )

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
        """Initializes RL components: Envs, Agent, Stats, Collector, CheckpointManager."""
        print(f"Initializing RL components (PPO)... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            self.envs = initialize_envs(self.env_config.NUM_ENVS, self.env_config)
            self.agent = initialize_agent(
                model_config=self.model_config,
                ppo_config=self.ppo_config,
                rnn_config=self.rnn_config,
                env_config=self.env_config,
                transformer_config=self.transformer_config,
                device=self.app.device,
            )
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
            if self.stats_recorder is None:
                raise RuntimeError("Stats Recorder init failed.")

            self.checkpoint_manager = CheckpointManager(
                agent=self.agent,
                stats_aggregator=self.stats_recorder.aggregator,
                base_checkpoint_dir=BASE_CHECKPOINT_DIR,
                run_checkpoint_dir=get_run_checkpoint_dir(),
                load_checkpoint_path_config=checkpoint_to_load,
                device=self.app.device,
                obs_rms_dict=None,  # Will be set after collector init
            )

            self.rollout_collector = RolloutCollector(
                envs=self.envs,
                agent=self.agent,
                stats_recorder=self.stats_recorder,
                env_config=self.env_config,
                ppo_config=self.ppo_config,
                rnn_config=self.rnn_config,
                reward_config=self.reward_config,
                tb_config=self.tensorboard_config,
                obs_norm_config=self.obs_norm_config,
                device=self.app.device,
            )

            # Pass the initialized RMS dict from collector to checkpoint manager
            self.checkpoint_manager.obs_rms_dict = (
                self.rollout_collector.get_obs_rms_dict()
            )

            if self.checkpoint_manager.get_checkpoint_path_to_load():
                self.checkpoint_manager.load_checkpoint()
                loaded_global_step, initial_episode_count = (
                    self.checkpoint_manager.get_initial_state()
                )
                # Sync collector's episode count (global step is managed by aggregator)
                self.rollout_collector.state.episode_count = initial_episode_count
                # Aggregator state is loaded within checkpoint_manager.load_checkpoint()

            print(f"RL components initialized in {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"Error during RL component initialization: {e}")
            traceback.print_exc()
            raise e

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
            toggle_training_run_cb=self.app.logic.toggle_training_run,
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
            # Also set the reference in the button renderer if needed
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
                    # Pass the is_cleanup flag down
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
