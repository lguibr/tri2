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
