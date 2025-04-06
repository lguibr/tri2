# File: training/trainer.py
import time
import torch
import numpy as np
import traceback
import random
import math  # Added for cosine annealing
from typing import List, Optional, Dict, Any, Union

from config import (
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    ObsNormConfig,
    TransformerConfig,  # Added configs
    # DEVICE, # Removed direct import
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
    """Orchestrates the PPO training process, including LR scheduling."""

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
        model_save_path: str,
        device: torch.device,  # --- MODIFIED: Moved device parameter ---
        load_checkpoint_path: Optional[
            str
        ] = None,  # Default parameter now comes after non-default
    ):
        print("[Trainer-PPO] Initializing...")
        self.envs = envs
        self.agent = agent
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.device = device  # --- MODIFIED: Use passed device ---
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.train_config = train_config
        self.model_config = model_config
        self.obs_norm_config = obs_norm_config  # Store config
        self.transformer_config = transformer_config  # Store config
        self.reward_config = RewardConfig()
        self.tb_config = TensorBoardConfig()
        self.vis_config = VisConfig()

        # Initialize Rollout Collector (which initializes RMS if enabled)
        self.rollout_collector = RolloutCollector(
            envs=self.envs,
            agent=self.agent,
            stats_recorder=self.stats_recorder,
            env_config=self.env_config,
            ppo_config=self.ppo_config,
            rnn_config=self.rnn_config,
            reward_config=self.reward_config,
            tb_config=self.tb_config,
            obs_norm_config=self.obs_norm_config,  # Pass config
            device=self.device,  # --- MODIFIED: Pass device ---
        )
        self.rollout_storage = self.rollout_collector.rollout_storage

        # Initialize Checkpoint Manager (AFTER collector to get RMS instances)
        self.checkpoint_manager = CheckpointManager(
            agent=self.agent,
            model_save_path=model_save_path,
            load_checkpoint_path=load_checkpoint_path,
            device=self.device,
            obs_rms_dict=self.rollout_collector.get_obs_rms_dict(),  # Pass RMS dict
        )
        self.global_step, initial_episode_count = (
            self.checkpoint_manager.get_initial_state()
        )
        self.rollout_collector.episode_count = (
            initial_episode_count  # Sync episode count
        )

        self.last_image_log_step = -1
        self.last_checkpoint_step = self.global_step  # Initialize based on loaded step
        self.rollouts_completed_since_last_checkpoint = 0
        self.steps_collected_this_rollout = 0
        self.current_phase = "Collecting"  # Initial phase

        self._log_initial_state()
        print("[Trainer-PPO] Initialization complete.")

    def get_current_phase(self) -> str:
        """Returns the current phase ('Collecting' or 'Updating')."""
        return self.current_phase

    def get_update_progress(self) -> float:
        """Returns the progress of the agent update phase (0.0 to 1.0)."""
        if self.current_phase == "Updating":
            return self.agent.get_update_progress()
        return 0.0

    def _log_initial_state(self):
        """Logs the initial state after potential checkpoint loading."""
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
        """Safely gets the current learning rate from the optimizer."""
        try:
            # Ensure optimizer and param_groups exist
            if hasattr(self.agent, "optimizer") and self.agent.optimizer.param_groups:
                return self.agent.optimizer.param_groups[0]["lr"]
            else:
                print(
                    "Warning: Optimizer or param_groups not found, returning default LR."
                )
                return self.ppo_config.LEARNING_RATE
        except (AttributeError, IndexError, TypeError) as e:
            print(f"Warning: Error getting LR ({e}), returning default LR.")
            return self.ppo_config.LEARNING_RATE  # Fallback

    # --- MODIFIED: Learning Rate Scheduler ---
    def _update_learning_rate(self):
        """Updates the learning rate based on the configured schedule."""
        if not self.ppo_config.USE_LR_SCHEDULER:
            return

        total_steps = max(1, TOTAL_TRAINING_STEPS)
        current_progress = self.global_step / total_steps
        initial_lr = self.ppo_config.LEARNING_RATE

        # --- MODIFIED: Check for LR_SCHEDULE_TYPE before accessing ---
        schedule_type = getattr(
            self.ppo_config, "LR_SCHEDULE_TYPE", "linear"
        )  # Default to linear if missing

        if schedule_type == "linear":
            end_fraction = getattr(
                self.ppo_config, "LR_LINEAR_END_FRACTION", 0.0
            )  # Default if missing
            decay_fraction = max(0.0, 1.0 - current_progress)
            new_lr = initial_lr * (end_fraction + (1.0 - end_fraction) * decay_fraction)
        elif schedule_type == "cosine":
            min_factor = getattr(
                self.ppo_config, "LR_COSINE_MIN_FACTOR", 0.01
            )  # Default if missing
            # Cosine annealing formula: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
            min_lr = initial_lr * min_factor
            new_lr = min_lr + 0.5 * (initial_lr - min_lr) * (
                1 + math.cos(math.pi * current_progress)
            )
        else:
            # Default or unknown type: No change
            print(
                f"Warning: Unknown LR_SCHEDULE_TYPE '{schedule_type}'. Using current LR."
            )
            new_lr = self._get_current_lr()
        # --- END MODIFIED ---

        # Ensure LR doesn't drop below min (using cosine min factor as a general floor)
        min_factor_floor = getattr(self.ppo_config, "LR_COSINE_MIN_FACTOR", 0.0)
        new_lr = max(new_lr, initial_lr * min_factor_floor)

        # Apply the new learning rate to the optimizer
        try:
            # Ensure optimizer and param_groups exist before trying to update
            if hasattr(self.agent, "optimizer") and self.agent.optimizer.param_groups:
                for param_group in self.agent.optimizer.param_groups:
                    param_group["lr"] = new_lr
            else:
                print(
                    "Warning: Could not update LR, optimizer or param_groups not found."
                )
        except AttributeError:
            print("Warning: Could not update LR, optimizer attribute missing.")
        except Exception as e:
            print(f"Warning: Unexpected error updating LR: {e}")

    # --- END MODIFIED ---

    def perform_training_iteration(self):
        """Performs one iteration of collection and potential update."""
        step_start_time = time.time()
        if self.current_phase != "Collecting":
            self.current_phase = "Collecting"

        # Collect one step from all environments
        steps_collected_this_iter = self.rollout_collector.collect_one_step(
            self.global_step
        )
        self.global_step += steps_collected_this_iter
        self.steps_collected_this_rollout += 1

        update_metrics = {}
        # Check if rollout buffer is full
        if self.steps_collected_this_rollout >= self.ppo_config.NUM_STEPS_PER_ROLLOUT:
            self.current_phase = "Updating"
            update_start_time = time.time()

            # Compute advantages using the collected rollout
            self.rollout_collector.compute_advantages_for_storage()
            self.rollout_storage.to(
                self.agent.device
            )  # Move storage to agent device for update
            update_data = self.rollout_storage.get_data_for_update()

            if update_data:
                try:
                    update_metrics = self.agent.update(update_data)
                except Exception as agent_update_err:
                    print(f"CRITICAL ERROR during agent.update: {agent_update_err}")
                    traceback.print_exc()
                    update_metrics = {}  # Prevent crash, continue loop
            else:
                print(
                    "[Trainer Warning] No data retrieved from rollout storage for update."
                )
                update_metrics = {}

            self.rollout_storage.after_update()  # Reset storage, keep last obs/state
            self.steps_collected_this_rollout = 0
            self.rollouts_completed_since_last_checkpoint += 1
            self.current_phase = "Collecting"  # Switch back after update
            update_duration = time.time() - update_start_time

            # Update LR *after* the agent update
            self._update_learning_rate()
            # Checkpoint and log images based on completed rollouts
            self.maybe_save_checkpoint()
            self._maybe_log_image()

            # Record update-specific metrics
            step_record_data_update = {
                "update_time": update_duration,
                "lr": self._get_current_lr(),
                "global_step": self.global_step,
            }
            if isinstance(update_metrics, dict):
                step_record_data_update.update(update_metrics)
            self.stats_recorder.record_step(step_record_data_update)

        # Record timing and basic info for every step (collection or update)
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        step_record_data_timing = {
            "step_time": step_duration,
            "num_steps_processed": steps_collected_this_iter,
            "global_step": self.global_step,
            "lr": self._get_current_lr(),
        }
        # Avoid double-logging if an update happened this iteration
        if not update_metrics:
            self.stats_recorder.record_step(step_record_data_timing)

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
            self.checkpoint_manager.save_checkpoint(
                self.global_step, self.rollout_collector.get_episode_count()
            )
            self.rollouts_completed_since_last_checkpoint = 0  # Reset counter
            self.last_checkpoint_step = self.global_step

    def _maybe_log_image(self):
        """Logs a sample environment image to TensorBoard based on frequency."""
        if not self.tb_config.LOG_IMAGES or self.tb_config.IMAGE_LOG_FREQ <= 0:
            return

        image_log_freq_rollouts = self.tb_config.IMAGE_LOG_FREQ
        # Use rollouts completed since *last* checkpoint for frequency check
        current_rollout_num_since_chkpt = self.rollouts_completed_since_last_checkpoint
        # Log if frequency is met AND we haven't logged for this exact step already
        if (
            current_rollout_num_since_chkpt > 0
            and current_rollout_num_since_chkpt % image_log_freq_rollouts == 0
        ):
            if self.global_step > self.last_image_log_step:
                print(f"[Trainer] Logging image at step {self.global_step}")
                try:
                    env_idx = random.randrange(self.num_envs)
                    img_array = get_env_image_as_numpy(
                        self.envs[env_idx], self.env_config, self.vis_config
                    )
                    if img_array is not None:
                        # Convert HWC to CHW for TensorBoard
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                        self.stats_recorder.record_image(
                            f"Environment/Sample State Env {env_idx}",
                            img_tensor,
                            self.global_step,
                        )
                        self.last_image_log_step = (
                            self.global_step
                        )  # Update last log step
                except Exception as e:
                    print(f"Error logging environment image: {e}")
                    traceback.print_exc()

    def train_loop(self):
        """Main training loop."""
        print("[Trainer-PPO] Starting training loop...")
        try:
            while self.global_step < TOTAL_TRAINING_STEPS:
                self.perform_training_iteration()
        except KeyboardInterrupt:
            print("\n[Trainer-PPO] Training loop interrupted by user (Ctrl+C).")
        except Exception as e:
            print(f"\n[Trainer-PPO] CRITICAL ERROR in training loop: {e}")
            traceback.print_exc()
        finally:
            print("[Trainer-PPO] Training loop finished or terminated.")
            self.cleanup(save_final=True)  # Attempt cleanup and final save

    def cleanup(self, save_final: bool = True):
        """Cleans up resources, optionally saving a final checkpoint."""
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

        # Close stats recorder (which closes TensorBoard writer)
        if hasattr(self.stats_recorder, "close"):
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"Error closing stats recorder: {e}")
        print("[Trainer-PPO] Cleanup complete.")
