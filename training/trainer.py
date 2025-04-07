import time
import torch
import numpy as np
import traceback
import random
import math
from typing import List, Optional, Dict, Any, Union, Tuple, Deque
import gc

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
    TOTAL_TRAINING_STEPS,  # Import the base increment value
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
    """Orchestrates the PPO training process, including LR scheduling and dynamic target steps."""

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
        obs_norm_config: ObsNormConfig,
        transformer_config: TransformerConfig,
        device: torch.device,
        load_checkpoint_path: Optional[str] = None,
    ):
        print("[Trainer-PPO] Initializing...")
        self.envs = envs
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

        self.rollout_collector = RolloutCollector(
            envs=self.envs,
            agent=self.agent,
            stats_recorder=self.stats_recorder,
            env_config=self.env_config,
            ppo_config=self.ppo_config,
            rnn_config=self.rnn_config,
            reward_config=self.reward_config,
            tb_config=self.tb_config,
            obs_norm_config=self.obs_norm_config,
            device=self.device,  # Pass target device to collector
        )
        self.rollout_storage = self.rollout_collector.rollout_storage

        self.checkpoint_manager = CheckpointManager(
            agent=self.agent,
            stats_aggregator=self.stats_aggregator,
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            run_checkpoint_dir=get_run_checkpoint_dir(),
            load_checkpoint_path_config=load_checkpoint_path,
            device=self.device,
            obs_rms_dict=self.rollout_collector.get_obs_rms_dict(),
        )

        # --- Load Checkpoint and Determine Training Target ---
        self.global_step = 0
        self.training_target_step = TOTAL_TRAINING_STEPS  # Default target for new run

        if self.checkpoint_manager.get_checkpoint_path_to_load():
            self.checkpoint_manager.load_checkpoint()
            # Get loaded step count AFTER loading
            loaded_global_step, initial_episode_count = (
                self.checkpoint_manager.get_initial_state()
            )
            self.global_step = loaded_global_step
            # Set the new target by adding the configured steps to the loaded steps
            self.training_target_step = self.global_step + TOTAL_TRAINING_STEPS
            print(
                f"[Trainer] Resumed from step {self.global_step}. New target: {self.training_target_step} steps."
            )
            # Sync aggregator's target step (loaded from checkpoint or set here)
            self.stats_aggregator.training_target_step = self.training_target_step
        else:
            print(
                f"[Trainer] No checkpoint loaded. Starting fresh. Target: {self.training_target_step} steps."
            )
            # Ensure aggregator's target step is set for a new run
            self.stats_aggregator.training_target_step = self.training_target_step
            # Ensure aggregator start time is current time if starting fresh
            self.stats_aggregator.start_time = time.time()

        # Get initial state AFTER CheckpointManager has potentially loaded a checkpoint
        # Note: global_step is already set above
        _, initial_episode_count = self.checkpoint_manager.get_initial_state()
        self.rollout_collector.episode_count = initial_episode_count
        self.stats_aggregator.current_global_step = self.global_step

        self.current_update_epoch = 0
        self.current_minibatch_index = 0
        self.update_indices: Optional[np.ndarray] = None
        self.update_metrics_accumulator: Dict[str, float] = defaultdict(float)
        self.num_minibatches_per_epoch = 0
        self.total_update_steps = 0
        self.num_updates_this_epoch = 0
        self.current_update_data_cpu: Optional[Dict[str, torch.Tensor]] = (
            None  # Data on CPU (pinned)
        )
        self.update_start_time: float = 0.0

        self.last_image_log_step = -1
        self.last_checkpoint_step = self.global_step
        self.rollouts_completed_since_last_checkpoint = 0
        self.steps_collected_this_rollout = 0
        self.current_phase = "Collecting"

        self._log_initial_state()
        print("[Trainer-PPO] Initialization complete.")

    def get_current_phase(self) -> str:
        """Returns the current phase ('Collecting' or 'Updating')."""
        if self.current_phase == "Updating":
            progress_details = self.get_update_progress_details()
            epoch = progress_details.get("current_epoch", 0)
            total_epochs = progress_details.get("total_epochs", 0)
            return f"Updating (Epoch {epoch}/{total_epochs})"
        return self.current_phase

    def get_update_progress_details(self) -> Dict[str, Any]:
        """Returns detailed progress information for the current update phase."""
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
        if self.current_phase == "Updating":
            total_epochs = self.ppo_config.PPO_EPOCHS
            total_steps_in_epoch = self.num_minibatches_per_epoch
            current_step_in_epoch = self.current_minibatch_index
            if self.total_update_steps == 0 and total_steps_in_epoch > 0:
                self.total_update_steps = total_steps_in_epoch * total_epochs

            current_total_step = (
                self.current_update_epoch * total_steps_in_epoch
            ) + current_step_in_epoch
            overall_progress = current_total_step / max(1, self.total_update_steps)
            epoch_progress = current_step_in_epoch / max(1, total_steps_in_epoch)
            details.update(
                {
                    "overall_progress": overall_progress,
                    "epoch_progress": epoch_progress,
                    "current_epoch": self.current_update_epoch + 1,
                    "total_epochs": total_epochs,
                    "phase": "Train Update",
                    "update_start_time": self.update_start_time,
                    "num_minibatches_per_epoch": self.num_minibatches_per_epoch,
                    "current_minibatch_index": self.current_minibatch_index,
                }
            )
        elif self.current_phase == "Collecting":
            details["phase"] = "Collecting"
        return details

    def _log_initial_state(self):
        """Logs the initial state after potential checkpoint loading."""
        initial_lr = self._get_current_lr()
        self.stats_recorder.record_step(
            {
                "lr": initial_lr,
                "global_step": self.global_step,
                "episode_count": self.rollout_collector.get_episode_count(),
                "training_target_step": self.training_target_step,  # Log target
            }
        )
        print(
            f"  -> Start Step={self.global_step}, Target Step={self.training_target_step}, Ep={self.rollout_collector.get_episode_count()}, LR={initial_lr:.1e}"
        )

    def _get_current_lr(self) -> float:
        """Safely gets the current learning rate from the optimizer."""
        try:
            if hasattr(self.agent, "optimizer") and self.agent.optimizer.param_groups:
                return self.agent.optimizer.param_groups[0]["lr"]
            else:
                return self.ppo_config.LEARNING_RATE
        except Exception:
            return self.ppo_config.LEARNING_RATE

    def _update_learning_rate(self):
        """Updates the learning rate based on the configured schedule."""
        if not self.ppo_config.USE_LR_SCHEDULER:
            return
        # Use the dynamic training target step for LR scheduling
        total_steps = max(1, self.training_target_step)
        current_progress = self.global_step / total_steps
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
            new_lr = self._get_current_lr()

        min_factor_floor = getattr(self.ppo_config, "LR_COSINE_MIN_FACTOR", 0.0)
        new_lr = max(new_lr, initial_lr * min_factor_floor)

        try:
            if hasattr(self.agent, "optimizer") and self.agent.optimizer.param_groups:
                if abs(self.agent.optimizer.param_groups[0]["lr"] - new_lr) > 1e-9:
                    for param_group in self.agent.optimizer.param_groups:
                        param_group["lr"] = new_lr
        except Exception as e:
            print(f"Warning: Unexpected error updating LR: {e}")

    # --- Regular Training Methods ---

    def _prepare_regular_update(self):
        """Prepares data and state for the regular PPO update phase."""
        self.current_phase = "Updating"
        self.update_start_time = time.time()
        self.rollout_collector.compute_advantages_for_storage()
        # Data is retrieved on CPU (potentially pinned)
        self.current_update_data_cpu = self.rollout_storage.get_data_for_update()

        if not self.current_update_data_cpu:
            print("[Trainer Warning] No data retrieved from storage for update.")
            self.current_phase = "Collecting"
            self.steps_collected_this_rollout = 0
            return False

        # Normalize advantages on CPU
        advantages = self.current_update_data_cpu["advantages"]
        self.current_update_data_cpu["advantages"] = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

        num_samples = self.current_update_data_cpu["actions"].shape[0]
        batch_size = self.ppo_config.MINIBATCH_SIZE
        self.num_minibatches_per_epoch = 0
        for i in range(0, num_samples, batch_size):
            if i + batch_size <= num_samples:
                self.num_minibatches_per_epoch += 1
            elif num_samples - i >= 2:  # Ensure minibatch has at least 2 samples
                self.num_minibatches_per_epoch += 1

        self.total_update_steps = (
            self.num_minibatches_per_epoch * self.ppo_config.PPO_EPOCHS
        )
        self.update_indices = np.arange(num_samples)
        self.current_update_epoch = 0
        self.current_minibatch_index = 0
        self.update_metrics_accumulator = defaultdict(float)
        self.num_updates_this_epoch = 0
        return True

    def _iterate_regular_update(self):
        """Performs one MINIBATCH update step of the regular training phase."""
        if self.current_update_data_cpu is None or self.update_indices is None:
            print("Error: Regular update called without data.")
            self.current_phase = "Collecting"
            return

        if self.current_minibatch_index == 0:
            np.random.shuffle(self.update_indices)
            self.update_metrics_accumulator = defaultdict(float)
            self.num_updates_this_epoch = 0

        start_idx = self.current_minibatch_index * self.ppo_config.MINIBATCH_SIZE
        end_idx = start_idx + self.ppo_config.MINIBATCH_SIZE
        minibatch_indices = self.update_indices[start_idx:end_idx]

        if len(minibatch_indices) < 2:  # Skip if minibatch is too small
            self.current_minibatch_index += 1
            if self.current_minibatch_index >= self.num_minibatches_per_epoch:
                self.current_update_epoch += 1
                self.current_minibatch_index = 0
                if self.current_update_epoch >= self.ppo_config.PPO_EPOCHS:
                    self._finalize_update_phase()
            return

        # Select minibatch data (still on CPU)
        minibatch_cpu = {
            key: self.current_update_data_cpu[key][minibatch_indices]
            for key in [
                "obs_grid",
                "obs_shapes",
                "obs_availability",
                "obs_explicit_features",
                "actions",
                "log_probs",
                "returns",
                "advantages",
            ]
        }

        # Move minibatch data to agent's device just before the update
        minibatch_device = {
            k: v.to(self.agent.device, non_blocking=self.rollout_storage.pin_memory)
            for k, v in minibatch_cpu.items()
        }

        try:
            # Perform update on the agent's device
            minibatch_metrics = self.agent.update_minibatch(minibatch_device)
            for k, v in minibatch_metrics.items():
                self.update_metrics_accumulator[k] += v
            self.num_updates_this_epoch += 1
        except Exception as e:
            print(
                f"CRITICAL ERROR during agent.update_minibatch (Epoch {self.current_update_epoch+1}, MB {self.current_minibatch_index}): {e}"
            )
            traceback.print_exc()
            self._handle_update_error()
            return
        finally:
            # Clean up device tensors immediately after use
            del minibatch_device
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.current_minibatch_index += 1

        if self.current_minibatch_index >= self.num_minibatches_per_epoch:
            self.current_update_epoch += 1
            self.current_minibatch_index = 0
            if self.current_update_epoch >= self.ppo_config.PPO_EPOCHS:
                self._finalize_update_phase()

    def _finalize_update_phase(self):
        """Handles logic after all update epochs are completed."""
        update_duration = time.time() - self.update_start_time
        total_updates_in_phase = self.num_updates_this_epoch
        avg_metrics = {
            k: v / max(1, total_updates_in_phase)
            for k, v in self.update_metrics_accumulator.items()
        }
        step_record_data_update = {
            "update_time": update_duration,
            "lr": self._get_current_lr(),
            "global_step": self.global_step,
            "training_target_step": self.training_target_step,  # Include target
        }
        step_record_data_update.update(avg_metrics)
        self.stats_recorder.record_step(step_record_data_update)

        self.rollout_storage.after_update()
        self.steps_collected_this_rollout = 0
        self.rollouts_completed_since_last_checkpoint += 1
        self.current_phase = "Collecting"
        self._update_learning_rate()
        self.maybe_save_checkpoint()
        self._maybe_log_image()
        self.current_update_data_cpu = None  # Clear CPU data
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _handle_update_error(self):
        """Handles errors during the update phase."""
        self.current_phase = "Collecting"
        self.steps_collected_this_rollout = 0
        self.current_update_data_cpu = None  # Clear CPU data
        self.rollout_storage.after_update()  # Reset storage even on error
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Main Loop Logic ---

    def perform_training_iteration(self):
        """Performs one iteration: one step of collection OR one minibatch of update."""
        step_start_time = time.time()

        if self.current_phase == "Collecting":
            # Check if training target is already reached before collecting
            if self.global_step >= self.training_target_step:
                self.current_phase = "Complete"  # Or some other terminal state
                print(
                    f"Training target ({self.training_target_step}) reached. Stopping collection."
                )
                return

            steps_collected_this_iter = self.rollout_collector.collect_one_step(
                self.global_step
            )
            self.global_step += steps_collected_this_iter
            self.steps_collected_this_rollout += 1

            if (
                self.steps_collected_this_rollout
                >= self.ppo_config.NUM_STEPS_PER_ROLLOUT
            ):
                if not self._prepare_regular_update():
                    # If prepare fails (e.g., no data), reset collection count
                    self.steps_collected_this_rollout = 0

            step_duration = time.time() - step_start_time
            step_record_data_timing = {
                "step_time": step_duration,
                "num_steps_processed": steps_collected_this_iter,
                "global_step": self.global_step,
                "lr": self._get_current_lr(),
                "training_target_step": self.training_target_step,  # Include target
            }
            self.stats_recorder.record_step(step_record_data_timing)

        elif self.current_phase == "Updating":
            self._iterate_regular_update()

        elif self.current_phase == "Complete":
            # Do nothing if training is complete
            pass

    def maybe_save_checkpoint(self, force_save=False):
        """Saves a checkpoint based on frequency or if forced."""
        save_freq_rollouts = self.train_config.CHECKPOINT_SAVE_FREQ
        should_save_freq = (
            save_freq_rollouts > 0
            and self.rollouts_completed_since_last_checkpoint >= save_freq_rollouts
        )
        if force_save or should_save_freq:
            print(
                f"[Trainer] Saving checkpoint. Force: {force_save}, FreqMet: {should_save_freq}, Rollouts Since Last: {self.rollouts_completed_since_last_checkpoint}"
            )
            # Pass the current training target step to the checkpoint manager
            self.checkpoint_manager.save_checkpoint(
                self.global_step,
                self.stats_aggregator.total_episodes,
                training_target_step=self.training_target_step,
            )
            self.rollouts_completed_since_last_checkpoint = 0
            self.last_checkpoint_step = self.global_step

    def _maybe_log_image(self):
        """Logs a sample environment image to TensorBoard based on frequency."""
        if not self.tb_config.LOG_IMAGES or self.tb_config.IMAGE_LOG_FREQ <= 0:
            return
        image_log_freq_rollouts = self.tb_config.IMAGE_LOG_FREQ
        if (
            self.rollouts_completed_since_last_checkpoint > 0
            and self.rollouts_completed_since_last_checkpoint % image_log_freq_rollouts
            == 0
        ):
            if self.global_step > self.last_image_log_step:
                print(f"[Trainer] Logging image at step {self.global_step}")
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

    def train_loop(self):
        """Main training loop (DEPRECATED - logic moved to MainApp._update)."""
        print("[Trainer-PPO] train_loop() is deprecated. Use MainApp loop.")

    def cleanup(self, save_final: bool = True):
        """Cleans up resources, optionally saving a final checkpoint."""
        print("[Trainer-PPO] Cleaning up resources...")
        should_save = save_final and self.global_step > 0
        if should_save:
            print("[Trainer-PPO] Saving final checkpoint...")
            # Pass the current training target step to the checkpoint manager
            self.checkpoint_manager.save_checkpoint(
                self.global_step,
                self.stats_aggregator.total_episodes,
                is_final=True,
                training_target_step=self.training_target_step,
            )
        else:
            print(
                f"[Trainer-PPO] Skipping final save (SaveFinal={save_final}, GlobalStep={self.global_step})."
            )
        if hasattr(self.stats_recorder, "close"):
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"Error closing stats recorder: {e}")
        print("[Trainer-PPO] Cleanup complete.")
