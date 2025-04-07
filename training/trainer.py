# File: training/trainer.py
import time
import torch
import numpy as np
import traceback
import random
import math
from typing import List, Optional, Dict, Any, Union, Tuple, Deque
import gc
import threading  # Added for Lock

from collections import defaultdict

from config import (
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    ObsNormConfig,
    TransformerConfig,
    TensorBoardConfig,
    VisConfig,
    RewardConfig,
    TOTAL_TRAINING_STEPS,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
)
from environment.game_state import GameState
from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregator
from .rollout_collector import RolloutCollector
from .checkpoint_manager import CheckpointManager
from .training_utils import get_env_image_as_numpy


class Trainer:
    """
    Orchestrates the PPO training process.
    NOTE: This class is now primarily used by the TrainingWorker thread.
    The main training loop logic resides within the worker's run method.
    """

    def __init__(
        self,
        envs: List[GameState],  # Still needed for context? Maybe not directly.
        agent: PPOAgent,
        stats_recorder: StatsRecorderBase,
        env_config: EnvConfig,
        ppo_config: PPOConfig,
        rnn_config: RNNConfig,
        train_config: TrainConfig,
        model_config: ModelConfig,
        obs_norm_config: ObsNormConfig,
        transformer_config: TransformerConfig,
        device: torch.device,
        load_checkpoint_path: Optional[str] = None,
        # Removed rollout_collector and storage, managed by worker/main app
    ):
        print("[Trainer-PPO] Initializing (as component)...")
        # Store necessary components, but don't initialize workers here
        # self.envs = envs # Not directly used by trainer logic anymore
        self.agent = agent
        self.stats_recorder = stats_recorder
        if not hasattr(stats_recorder, "aggregator") or not isinstance(
            stats_recorder.aggregator, StatsAggregator
        ):
            raise TypeError(
                "StatsRecorder provided to Trainer must have a 'aggregator' attribute of type StatsAggregator."
            )
        self.stats_aggregator: StatsAggregator = stats_recorder.aggregator

        self.num_envs = env_config.NUM_ENVS
        self.device = device
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.train_config = train_config
        self.model_config = model_config
        self.obs_norm_config = obs_norm_config
        self.transformer_config = transformer_config
        self.reward_config = RewardConfig()
        self.tb_config = TensorBoardConfig()
        self.vis_config = VisConfig()

        # Checkpoint manager is now initialized and managed by MainApp
        self.checkpoint_manager: Optional[CheckpointManager] = None

        # State related to update progress (managed by worker)
        self.current_update_epoch = 0
        self.current_minibatch_index = 0
        self.update_indices: Optional[np.ndarray] = None
        self.update_metrics_accumulator: Dict[str, float] = defaultdict(float)
        self.num_minibatches_per_epoch = 0
        self.total_update_steps = 0
        self.num_updates_this_epoch = 0
        self.update_start_time: float = 0.0

        # Other state variables (managed by worker or main app)
        self.global_step = 0
        self.training_target_step = TOTAL_TRAINING_STEPS
        self.last_image_log_step = -1
        self.last_checkpoint_step = 0
        self.rollouts_completed_since_last_checkpoint = 0
        self.current_phase = "Idle"  # Worker controls its phase

        print("[Trainer-PPO] Component Initialization complete.")
        # Note: Checkpoint loading and LR scheduling are handled by MainApp/CheckpointManager/Worker

    def set_checkpoint_manager(self, ckpt_manager: CheckpointManager):
        """Allows MainApp to set the checkpoint manager after initialization."""
        self.checkpoint_manager = ckpt_manager
        # Sync initial state from manager if needed (though worker usually gets it)
        self.global_step = self.checkpoint_manager.global_step
        self.training_target_step = self.checkpoint_manager.training_target_step

    def get_current_phase(self) -> str:
        """Returns the current phase (now likely managed by the worker)."""
        # This might be better tracked within the worker itself
        return self.current_phase

    def get_update_progress_details(self) -> Dict[str, Any]:
        """Returns detailed progress information for the current update phase."""
        # This state is now managed within the TrainingWorker's run loop
        # This method might need to fetch state from the worker if called from main thread
        # For now, return placeholder or rely on worker updating shared state/queue
        details = {
            "overall_progress": 0.0,
            "epoch_progress": 0.0,
            "current_epoch": 0,
            "total_epochs": 0,
            "phase": "Idle",
            "update_start_time": 0.0,
            "num_minibatches_per_epoch": 0,
            "current_minibatch_index": 0,
        }
        # In a real implementation, this would query the TrainingWorker thread
        return details

    def _get_current_lr(self) -> float:
        """Safely gets the current learning rate from the optimizer."""
        try:
            # Use agent's lock for thread safety
            with self.agent._lock:
                if (
                    hasattr(self.agent, "optimizer")
                    and self.agent.optimizer.param_groups
                ):
                    return self.agent.optimizer.param_groups[0]["lr"]
                else:
                    return self.ppo_config.LEARNING_RATE
        except Exception:
            return self.ppo_config.LEARNING_RATE

    def _update_learning_rate(self):
        """
        Updates the learning rate based on the configured schedule.
        NOTE: This should ideally be called by the main thread or the worker
              before starting an update cycle, using the global step from the stats aggregator.
        """
        if not self.ppo_config.USE_LR_SCHEDULER:
            return

        # Use the global step from the aggregator for consistency
        current_global_step = self.stats_aggregator.current_global_step
        # Use the target step from the checkpoint manager (which should be synced)
        total_steps = max(
            1,
            (
                self.checkpoint_manager.training_target_step
                if self.checkpoint_manager
                else TOTAL_TRAINING_STEPS
            ),
        )

        current_progress = current_global_step / total_steps
        initial_lr = self.ppo_config.LEARNING_RATE
        schedule_type = getattr(self.ppo_config, "LR_SCHEDULE_TYPE", "linear")

        if schedule_type == "linear":
            end_fraction = getattr(self.ppo_config, "LR_LINEAR_END_FRACTION", 0.0)
            decay_fraction = max(0.0, 1.0 - min(1.0, current_progress))
            new_lr = initial_lr * (end_fraction + (1.0 - end_fraction) * decay_fraction)
        elif schedule_type == "cosine":
            min_factor = getattr(self.ppo_config, "LR_COSINE_MIN_FACTOR", 0.01)
            min_lr = initial_lr * min_factor
            new_lr = min_lr + 0.5 * (initial_lr - min_lr) * (
                1 + math.cos(math.pi * min(1.0, current_progress))
            )
        else:
            new_lr = self._get_current_lr()  # Keep current if unknown schedule

        min_factor_floor = getattr(self.ppo_config, "LR_COSINE_MIN_FACTOR", 0.0)
        new_lr = max(new_lr, initial_lr * min_factor_floor)

        try:
            # Use agent's lock for thread safety
            with self.agent._lock:
                if (
                    hasattr(self.agent, "optimizer")
                    and self.agent.optimizer.param_groups
                ):
                    if abs(self.agent.optimizer.param_groups[0]["lr"] - new_lr) > 1e-9:
                        # print(f"[LR Update] Step {current_global_step}: Updating LR from {self.agent.optimizer.param_groups[0]['lr']:.2e} to {new_lr:.2e}")
                        for param_group in self.agent.optimizer.param_groups:
                            param_group["lr"] = new_lr
        except Exception as e:
            print(f"Warning: Unexpected error updating LR: {e}")

    # --- Methods below are largely deprecated or adapted for worker ---

    def _prepare_regular_update(self):
        """DEPRECATED - Logic moved to TrainingWorker."""
        print("Trainer._prepare_regular_update() called - DEPRECATED")
        pass

    def _iterate_regular_update(self):
        """DEPRECATED - Logic moved to TrainingWorker."""
        print("Trainer._iterate_regular_update() called - DEPRECATED")
        pass

    def _finalize_update_phase(self):
        """DEPRECATED - Logic moved to TrainingWorker."""
        print("Trainer._finalize_update_phase() called - DEPRECATED")
        pass

    def _handle_update_error(self):
        """DEPRECATED - Logic moved to TrainingWorker."""
        print("Trainer._handle_update_error() called - DEPRECATED")
        pass

    def perform_training_iteration(self):
        """DEPRECATED - Logic moved to worker threads and MainApp loop."""
        print("Trainer.perform_training_iteration() called - DEPRECATED")
        pass

    def maybe_save_checkpoint(self, force_save=False):
        """
        Saves a checkpoint based on frequency or if forced.
        NOTE: This should be called by the main thread, not the worker.
        """
        if not self.checkpoint_manager:
            print("Warning: Checkpoint Manager not set in Trainer. Cannot save.")
            return

        # Use stats aggregator's rollout count for frequency check
        rollouts_completed = getattr(
            self.stats_aggregator,
            "rollouts_processed",
            self.rollouts_completed_since_last_checkpoint,
        )  # Fallback needed?

        save_freq_rollouts = self.train_config.CHECKPOINT_SAVE_FREQ
        should_save_freq = (
            save_freq_rollouts > 0
            and rollouts_completed
            >= save_freq_rollouts  # Check if enough rollouts passed
        )

        if force_save or should_save_freq:
            print(
                f"[Trainer->CkptMgr] Saving checkpoint. Force: {force_save}, FreqMet: {should_save_freq}, Rollouts: {rollouts_completed}"
            )
            current_global_step = self.stats_aggregator.current_global_step
            current_episode_count = self.stats_aggregator.total_episodes
            current_target_step = self.checkpoint_manager.training_target_step

            self.checkpoint_manager.save_checkpoint(
                current_global_step,
                current_episode_count,
                training_target_step=current_target_step,
                is_final=False,  # Assume not final unless called by cleanup
            )
            # Reset counter - How to sync this back to worker/aggregator?
            # Maybe the main thread tracks rollouts_since_last_checkpoint based on queue activity.
            # For now, just save. Resetting counter needs careful thought.
            # self.rollouts_completed_since_last_checkpoint = 0 # This state is local, might be wrong
            self.last_checkpoint_step = current_global_step

    def _maybe_log_image(self):
        """
        Logs a sample environment image to TensorBoard based on frequency.
        NOTE: This should be called by the main thread, not the worker.
        """
        # This logic depends on having access to envs, which the trainer might not have directly.
        # Best handled by the main thread which owns the envs list.
        print("Trainer._maybe_log_image() called - DEPRECATED (moved to MainApp)")
        pass

    def train_loop(self):
        """Main training loop (DEPRECATED - logic moved to TrainingWorker)."""
        print("[Trainer-PPO] train_loop() is deprecated. Use Worker threads.")

    def cleanup(self, save_final: bool = True):
        """
        Cleans up resources.
        NOTE: Actual saving and closing is handled by MainApp. This is mostly a placeholder.
        """
        print("[Trainer-PPO] Cleanup called (component level)...")
        # Saving and closing stats recorder is handled by MainApp during its cleanup
        # if self.checkpoint_manager and save_final and self.global_step > 0:
        #     print("[Trainer-PPO] Requesting final checkpoint save...")
        #     self.checkpoint_manager.save_checkpoint(
        #         self.global_step,
        #         self.stats_aggregator.total_episodes,
        #         is_final=True,
        #         training_target_step=self.training_target_step,
        #     )
        # if hasattr(self.stats_recorder, "close"):
        #     try:
        #         self.stats_recorder.close()
        #     except Exception as e:
        #         print(f"Error closing stats recorder: {e}")
        print("[Trainer-PPO] Component cleanup finished.")
