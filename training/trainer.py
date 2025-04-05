# File: training/trainer.py
import time
import torch
import numpy as np
import traceback
import random
from typing import List, Optional, Dict, Any, Union

from config import (
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    DEVICE,
    TensorBoardConfig,
    VisConfig,
    RewardConfig,
    TOTAL_TRAINING_STEPS,
)
from environment.game_state import GameState, StateType
from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from utils.helpers import ensure_numpy
from .rollout_storage import RolloutStorage
from .rollout_collector import RolloutCollector
from .checkpoint_manager import CheckpointManager
from .training_utils import get_env_image_as_numpy


class Trainer:
    """Orchestrates the PPO training process."""

    def __init__(
        self,
        envs: List[GameState],
        agent: PPOAgent,
        stats_recorder: StatsRecorderBase,
        env_config: EnvConfig,
        ppo_config: PPOConfig,
        rnn_config: RNNConfig,
        train_config: TrainConfig,
        model_config: ModelConfig,
        model_save_path: str,
        load_checkpoint_path: Optional[str] = None,
    ):
        print("[Trainer-PPO] Initializing...")
        self.envs = envs
        self.agent = agent
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.device = DEVICE
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.train_config = train_config
        self.model_config = model_config
        self.reward_config = RewardConfig()
        self.tb_config = TensorBoardConfig()
        self.vis_config = VisConfig()

        self.checkpoint_manager = CheckpointManager(
            agent=self.agent,
            model_save_path=model_save_path,
            load_checkpoint_path=load_checkpoint_path,
            device=self.device,
        )
        self.global_step, initial_episode_count = (
            self.checkpoint_manager.get_initial_state()
        )

        self.rollout_collector = RolloutCollector(
            envs=self.envs,
            agent=self.agent,
            stats_recorder=self.stats_recorder,
            env_config=self.env_config,
            ppo_config=self.ppo_config,
            rnn_config=self.rnn_config,
            reward_config=self.reward_config,
            tb_config=self.tb_config,
        )
        self.rollout_collector.episode_count = initial_episode_count

        self.last_image_log_step = (
            -self.tb_config.IMAGE_LOG_FREQ * self.ppo_config.NUM_STEPS_PER_ROLLOUT
        )
        self.last_checkpoint_step = 0

        # --- NEW: State for iterative training ---
        self.steps_collected_this_rollout = 0
        self.rollout_storage = (
            self.rollout_collector.rollout_storage
        )  # Use collector's storage
        # --- END NEW ---

        self._log_initial_state()
        print("[Trainer-PPO] Initialization complete.")

    def _log_initial_state(self):
        """Logs the state after initialization and potential loading."""
        initial_lr = self._get_current_lr()
        self.stats_recorder.record_step(
            {
                "lr": initial_lr,
                "global_step": self.global_step,
                "episode_count": self.rollout_collector.get_episode_count(),
            }
        )
        print(
            f"  -> Start Step={self.global_step}, Ep={self.rollout_collector.get_episode_count()}, LR={initial_lr:.1e}"
        )

    def _get_current_lr(self) -> float:
        """Retrieves the current learning rate from the optimizer."""
        return self.agent.optimizer.param_groups[0]["lr"]

    def _update_learning_rate(self):
        """Linearly decay learning rate if scheduler is enabled."""
        if not self.ppo_config.USE_LR_SCHEDULER:
            return

        frac = 1.0 - (self.global_step / TOTAL_TRAINING_STEPS)
        frac = max(self.ppo_config.LR_SCHEDULER_END_FRACTION, frac)
        new_lr = self.ppo_config.LEARNING_RATE * frac

        for param_group in self.agent.optimizer.param_groups:
            param_group["lr"] = new_lr

    # --- MODIFIED: Renamed and changed logic ---
    def perform_training_iteration(self):
        """Performs one step of environment interaction and potentially an agent update."""
        step_start_time = time.time()

        # Collect one step across all environments
        steps_collected_this_iter = self.rollout_collector.collect_one_step(
            self.global_step
        )
        self.global_step += steps_collected_this_iter
        self.steps_collected_this_rollout += 1  # Increment rollout step counter

        # Check if a full rollout is complete
        if self.steps_collected_this_rollout >= self.ppo_config.NUM_STEPS_PER_ROLLOUT:
            # --- Agent Update Phase ---
            update_start_time = time.time()

            # Compute advantages using the final value estimate
            self.rollout_collector.compute_advantages_for_storage()

            # Prepare data and update the agent
            self.rollout_storage.to(self.agent.device)
            update_data = self.rollout_storage.get_data_for_update()
            update_metrics = self.agent.update(update_data)

            # Reset storage and rollout counter
            self.rollout_storage.after_update()
            self.steps_collected_this_rollout = 0

            update_duration = time.time() - update_start_time

            # Update LR and log/save after the update
            self._update_learning_rate()
            self.maybe_save_checkpoint()
            self._maybe_log_image()

            # Record update-specific metrics
            self.stats_recorder.record_step(
                {
                    "update_time": update_duration,
                    "lr": self._get_current_lr(),
                    **update_metrics,
                    "global_step": self.global_step,  # Ensure global step is logged here too
                }
            )
            # --- End Agent Update Phase ---

        step_end_time = time.time()
        step_duration = step_end_time - step_start_time

        # Record step-timing metrics (happens every iteration)
        self.stats_recorder.record_step(
            {
                "step_time": step_duration,
                "num_steps_processed": steps_collected_this_iter,
                "global_step": self.global_step,
            }
        )

    # --- END MODIFIED ---

    def maybe_save_checkpoint(self, force_save=False):
        """Saves agent state based on frequency or if forced."""
        # --- MODIFIED: Checkpoint frequency based on agent updates (rollouts completed) ---
        # Calculate rollouts completed based on when steps_collected_this_rollout resets
        # This logic assumes maybe_save_checkpoint is called right after a potential update.
        save_freq_rollouts = self.train_config.CHECKPOINT_SAVE_FREQ
        steps_per_rollout = (
            self.ppo_config.NUM_STEPS_PER_ROLLOUT
        )  # Use steps per rollout directly

        # Check if an update just happened (steps_collected_this_rollout is 0)
        update_just_happened = self.steps_collected_this_rollout == 0

        if update_just_happened:
            # Calculate how many rollouts have passed since the last save
            rollouts_since_last_save = (
                self.global_step - self.last_checkpoint_step
            ) // (steps_per_rollout * self.num_envs)

            should_save_freq = (
                save_freq_rollouts > 0
                and rollouts_since_last_save >= save_freq_rollouts
            )

            if force_save or should_save_freq:
                self.checkpoint_manager.save_checkpoint(
                    self.global_step, self.rollout_collector.get_episode_count()
                )
                # Update last_checkpoint_step to the *start* of the rollout that just finished
                self.last_checkpoint_step = self.global_step - (
                    steps_per_rollout * self.num_envs
                )
        elif force_save:  # Allow forced save even if not exactly on rollout boundary
            self.checkpoint_manager.save_checkpoint(
                self.global_step, self.rollout_collector.get_episode_count()
            )
            self.last_checkpoint_step = self.global_step
        # --- END MODIFIED ---

    def _maybe_log_image(self):
        """Logs a sample environment state image to TensorBoard periodically."""
        if not self.tb_config.LOG_IMAGES or self.tb_config.IMAGE_LOG_FREQ <= 0:
            return

        # --- MODIFIED: Check image log frequency based on agent updates ---
        steps_per_rollout = self.ppo_config.NUM_STEPS_PER_ROLLOUT
        image_log_freq_steps = (
            self.tb_config.IMAGE_LOG_FREQ * steps_per_rollout * self.num_envs
        )

        # Check if an update just happened
        update_just_happened = self.steps_collected_this_rollout == 0

        if update_just_happened:
            steps_since_last = self.global_step - self.last_image_log_step
            if steps_since_last >= image_log_freq_steps:
                try:
                    env_idx = random.randrange(self.num_envs)
                    img_array = get_env_image_as_numpy(
                        self.envs[env_idx], self.env_config, self.vis_config
                    )
                    if img_array is not None:
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                        self.stats_recorder.record_image(
                            f"Environment/Sample State Env {env_idx}",
                            img_tensor,
                            self.global_step,
                        )
                        self.last_image_log_step = self.global_step
                except Exception as e:
                    print(f"Error logging environment image: {e}")
                    traceback.print_exc()
        # --- END MODIFIED ---

    def train_loop(self):
        """Main training loop until max steps."""
        print("[Trainer-PPO] Starting training loop...")
        try:
            while self.global_step < TOTAL_TRAINING_STEPS:
                # --- MODIFIED: Call iterative method ---
                self.perform_training_iteration()
                # --- END MODIFIED ---
        except KeyboardInterrupt:
            print("\n[Trainer-PPO] Training loop interrupted by user (Ctrl+C).")
        except Exception as e:
            print(f"\n[Trainer-PPO] CRITICAL ERROR in training loop: {e}")
            traceback.print_exc()
        finally:
            print("[Trainer-PPO] Training loop finished or terminated.")
            self.cleanup(save_final=True)

    def cleanup(self, save_final: bool = True):
        """Performs cleanup actions like saving final state and closing logger."""
        print("[Trainer-PPO] Cleaning up resources...")
        if save_final:
            print("[Trainer-PPO] Saving final checkpoint...")
            self.checkpoint_manager.save_checkpoint(
                self.global_step,
                self.rollout_collector.get_episode_count(),
                is_final=True,
            )
        else:
            print("[Trainer-PPO] Skipping final save as requested.")

        if hasattr(self.stats_recorder, "close"):
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"Error closing stats recorder: {e}")

        print("[Trainer-PPO] Cleanup complete.")
