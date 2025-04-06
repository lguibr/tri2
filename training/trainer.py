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

        self.last_image_log_step = -1
        self.last_checkpoint_step = 0
        self.rollouts_completed_since_last_checkpoint = 0

        self.steps_collected_this_rollout = 0
        self.rollout_storage = self.rollout_collector.rollout_storage

        self.current_phase = "Collecting"  # Track current phase

        self._log_initial_state()
        print("[Trainer-PPO] Initialization complete.")

    def get_current_phase(self) -> str:
        """Returns the current phase: 'Collecting' or 'Updating'."""
        return self.current_phase

    def get_update_progress(self) -> float:
        """Returns the agent's update progress if in 'Updating' phase."""
        if self.current_phase == "Updating":
            return self.agent.get_update_progress()
        return 0.0

    def _log_initial_state(self):
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
        if hasattr(self.agent, "optimizer") and self.agent.optimizer.param_groups:
            return self.agent.optimizer.param_groups[0]["lr"]
        else:
            return self.ppo_config.LEARNING_RATE

    def _update_learning_rate(self):
        if not self.ppo_config.USE_LR_SCHEDULER:
            return
        total_steps = max(1, TOTAL_TRAINING_STEPS)
        frac = 1.0 - (self.global_step / total_steps)
        frac = max(self.ppo_config.LR_SCHEDULER_END_FRACTION, frac)
        new_lr = self.ppo_config.LEARNING_RATE * frac
        if hasattr(self.agent, "optimizer"):
            for param_group in self.agent.optimizer.param_groups:
                param_group["lr"] = new_lr

    def perform_training_iteration(self):
        """Performs one step of environment interaction and potentially an agent update."""
        step_start_time = time.time()

        # Ensure phase is 'Collecting' before starting collection
        if self.current_phase != "Collecting":
            self.current_phase = "Collecting"

        steps_collected_this_iter = self.rollout_collector.collect_one_step(
            self.global_step
        )
        self.global_step += steps_collected_this_iter
        self.steps_collected_this_rollout += 1

        update_metrics = {}

        if self.steps_collected_this_rollout >= self.ppo_config.NUM_STEPS_PER_ROLLOUT:
            # print(f"[Trainer Debug] Rollout complete...") # Removed
            self.current_phase = "Updating"  # Set phase before update
            update_start_time = time.time()

            self.rollout_collector.compute_advantages_for_storage()

            self.rollout_storage.to(self.agent.device)
            update_data = self.rollout_storage.get_data_for_update()

            # print(f"[Trainer Debug] update_data keys: ...") # Removed detailed print
            # for key, value in update_data.items(): ... # Removed detailed print

            if update_data:
                try:
                    update_metrics = self.agent.update(update_data)
                    # print(f"[Trainer Debug] update_metrics received...") # Removed
                except Exception as agent_update_err:
                    print(f"CRITICAL ERROR during agent.update: {agent_update_err}")
                    traceback.print_exc()
                    update_metrics = {}
            else:
                print(
                    "[Trainer Warning] No data retrieved from rollout storage for update. Skipping agent update."
                )
                update_metrics = {}

            self.rollout_storage.after_update()
            self.steps_collected_this_rollout = 0
            self.rollouts_completed_since_last_checkpoint += 1
            self.current_phase = "Collecting"  # Reset phase after update

            update_duration = time.time() - update_start_time

            self._update_learning_rate()
            self.maybe_save_checkpoint()
            self._maybe_log_image()

            step_record_data_update = {
                "update_time": update_duration,
                "lr": self._get_current_lr(),
                "global_step": self.global_step,
            }
            if isinstance(update_metrics, dict):
                step_record_data_update.update(update_metrics)
            else:
                print(
                    f"[Trainer Warning] agent.update did not return a dictionary. Received: {type(update_metrics)}"
                )

            # print(f"[Trainer Debug] Data being sent to stats_recorder...") # Removed
            self.stats_recorder.record_step(step_record_data_update)

        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        step_record_data_timing = {
            "step_time": step_duration,
            "num_steps_processed": steps_collected_this_iter,
            "global_step": self.global_step,
            "lr": self._get_current_lr(),
        }

        if not update_metrics:  # Only log step time if no update happened
            self.stats_recorder.record_step(step_record_data_timing)

    def maybe_save_checkpoint(self, force_save=False):
        """Saves agent state based on frequency or if forced."""
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
            self.rollouts_completed_since_last_checkpoint = 0
            self.last_checkpoint_step = self.global_step

    def _maybe_log_image(self):
        """Logs a sample environment state image to TensorBoard periodically."""
        if not self.tb_config.LOG_IMAGES or self.tb_config.IMAGE_LOG_FREQ <= 0:
            return

        image_log_freq_rollouts = self.tb_config.IMAGE_LOG_FREQ
        # Log based on rollouts completed *since last checkpoint*
        # Check if the *current* rollout number (1-based) is a multiple of the freq
        current_rollout_num_since_chkpt = self.rollouts_completed_since_last_checkpoint
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
        """Main training loop until max steps."""
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
