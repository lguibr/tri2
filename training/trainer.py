# File: training/trainer.py
import time
import torch
import numpy as np
import traceback
import random
from typing import List, Optional, Dict, Any

from config import (
    EnvConfig,
    DQNConfig,
    TrainConfig,
    BufferConfig,
    ModelConfig,
    DEVICE,
    TensorBoardConfig,
    VisConfig,
    RewardConfig,
    TOTAL_TRAINING_STEPS,
)
from environment.game_state import GameState
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from stats.stats_recorder import StatsRecorderBase
from utils.helpers import ensure_numpy
from .experience_collector import ExperienceCollector
from .checkpoint_manager import CheckpointManager
from .training_utils import get_env_image_as_numpy


class Trainer:
    """Orchestrates the DQN training process."""

    def __init__(
        self,
        envs: List[GameState],
        agent: DQNAgent,
        buffer: ReplayBufferBase,  # Initial buffer, might be replaced by CheckpointManager
        stats_recorder: StatsRecorderBase,
        env_config: EnvConfig,
        dqn_config: DQNConfig,
        train_config: TrainConfig,
        buffer_config: BufferConfig,
        model_config: ModelConfig,
        model_save_path: str,
        buffer_save_path: str,
        load_checkpoint_path: Optional[str] = None,
        load_buffer_path: Optional[str] = None,
    ):
        print("[Trainer] Initializing...")
        self.envs = envs
        self.agent = agent
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.device = DEVICE
        self.env_config = env_config
        self.dqn_config = dqn_config
        self.train_config = train_config
        self.buffer_config = buffer_config
        self.model_config = model_config
        self.reward_config = RewardConfig()
        self.tb_config = TensorBoardConfig()
        self.vis_config = VisConfig()

        self.checkpoint_manager = CheckpointManager(
            agent=self.agent,
            buffer=buffer,  # Pass initial buffer
            model_save_path=model_save_path,
            buffer_save_path=buffer_save_path,
            load_checkpoint_path=load_checkpoint_path,
            load_buffer_path=load_buffer_path,
            buffer_config=self.buffer_config,
            dqn_config=self.dqn_config,
            device=self.device,
        )
        self.buffer = (
            self.checkpoint_manager.get_buffer()
        )  # Get potentially loaded buffer
        self.global_step, initial_episode_count = (
            self.checkpoint_manager.get_initial_state()
        )

        self.experience_collector = ExperienceCollector(
            envs=self.envs,
            agent=self.agent,
            buffer=self.buffer,
            stats_recorder=self.stats_recorder,
            env_config=self.env_config,
            reward_config=self.reward_config,
            tb_config=self.tb_config,
        )
        # Sync episode count if collector was reset but checkpoint loaded
        self.experience_collector.episode_count = initial_episode_count

        self.last_image_log_step = -self.tb_config.IMAGE_LOG_FREQ

        self._log_initial_state()
        print("[Trainer] Initialization complete.")

    def _log_initial_state(self):
        initial_beta = self._update_beta()
        initial_lr = self._get_current_lr()
        self.stats_recorder.record_step(
            {
                "buffer_size": len(self.buffer),
                "beta": initial_beta,
                "lr": initial_lr,
                "global_step": self.global_step,
                "episode_count": self.experience_collector.get_episode_count(),
            }
        )
        print(
            f"  -> Start Step={self.global_step}, Ep={self.experience_collector.get_episode_count()}, Buf={len(self.buffer)}, Beta={initial_beta:.4f}, LR={initial_lr:.1e}"
        )

    def _get_current_lr(self) -> float:
        return (
            self.agent.optimizer.param_groups[0]["lr"]
            if hasattr(self.agent, "optimizer") and self.agent.optimizer.param_groups
            else 0.0
        )

    def step(self):
        """Performs one full step: collect experience, train, update target net."""
        steps_collected = self.experience_collector.collect(self.global_step)
        self.global_step += steps_collected

        if (
            self.global_step >= self.train_config.LEARN_START_STEP
            and self.global_step % self.train_config.LEARN_FREQ == 0
        ):
            if len(self.buffer) >= self.train_config.BATCH_SIZE:
                self._train_batch()

        target_freq = self.dqn_config.TARGET_UPDATE_FREQ
        if target_freq > 0 and self.global_step > 0:
            steps_before_this_iter = self.global_step - steps_collected
            if steps_before_this_iter // target_freq < self.global_step // target_freq:
                print(f"[Trainer] Updating target network at step {self.global_step}")
                self.agent.update_target_network()

        self.maybe_save_checkpoint()
        self._maybe_log_image()

    def _train_batch(self):
        """Samples a batch, computes loss, updates agent, and records stats."""
        beta = self._update_beta()
        indices, is_weights_np, batch_tuple = None, None, None
        try:
            if self.buffer_config.USE_PER:
                sample_result = self.buffer.sample(self.train_config.BATCH_SIZE)
                if sample_result:
                    batch_tuple, indices, is_weights_np = sample_result
                else:
                    return
            else:
                batch_tuple = self.buffer.sample(self.train_config.BATCH_SIZE)
                if batch_tuple is None:
                    return
        except Exception as e:
            print(f"ERROR sampling buffer: {e}")
            traceback.print_exc()
            return

        loss, td_errors = torch.tensor(0.0), None
        try:
            loss, td_errors = self.agent.compute_loss(
                batch_tuple, self.buffer_config.USE_N_STEP, is_weights_np
            )
        except Exception as e:
            print(f"ERROR computing loss: {e}")
            traceback.print_exc()
            return

        grad_norm = None
        try:
            grad_norm = self.agent.update(loss)
            if grad_norm is None and self.dqn_config.GRADIENT_CLIP_NORM > 0:
                return
        except Exception as e:
            print(f"ERROR updating agent: {e}")
            traceback.print_exc()
            return

        if self.buffer_config.USE_PER and indices is not None and td_errors is not None:
            try:
                td_errors_np = ensure_numpy(td_errors)
                self.buffer.update_priorities(indices, td_errors_np)
            except Exception as e:
                print(f"ERROR updating PER priorities: {e}")
                traceback.print_exc()

        train_log_data = {
            "loss": loss.item(),
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
            "avg_max_q": self.agent.get_last_avg_max_q(),
            "lr": self._get_current_lr(),
            "global_step": self.global_step,
        }
        if self.tb_config.LOG_HISTOGRAMS and td_errors is not None:
            train_log_data["batch_td_errors"] = td_errors
        self.stats_recorder.record_step(train_log_data)

    def _update_beta(self) -> float:
        """Updates PER beta and records it."""
        if not self.buffer_config.USE_PER:
            beta = 1.0
        else:
            start, end, frames = (
                self.buffer_config.PER_BETA_START,
                1.0,
                self.buffer_config.PER_BETA_FRAMES,
            )
            fraction = min(1.0, float(self.global_step) / max(1, frames))
            beta = start + fraction * (end - start)
            if hasattr(self.buffer, "set_beta"):
                self.buffer.set_beta(beta)
        self.stats_recorder.record_step({"beta": beta, "global_step": self.global_step})
        return beta

    def _maybe_log_image(self):
        """Logs an image of a random environment state periodically."""
        if not self.tb_config.LOG_IMAGES:
            return
        img_freq = self.tb_config.IMAGE_LOG_FREQ
        if img_freq <= 0:
            return
        steps_since_last = self.global_step - self.last_image_log_step
        if steps_since_last >= img_freq:
            try:
                env_idx = random.randint(0, self.num_envs - 1)
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

    def maybe_save_checkpoint(self, force_save=False):
        """Saves a checkpoint periodically or if forced."""
        save_freq = self.train_config.CHECKPOINT_SAVE_FREQ
        if save_freq <= 0 and not force_save:
            return
        should_save_freq = (
            save_freq > 0
            and self.global_step > 0
            and (
                self.global_step // save_freq
                > (self.global_step - self.num_envs) // save_freq
            )
        )
        if force_save or should_save_freq:
            self.checkpoint_manager.save_checkpoint(
                self.global_step, self.experience_collector.get_episode_count()
            )

    def train_loop(self):
        """Main training loop that runs until total steps are reached."""
        print("[Trainer] Starting training loop...")
        try:
            while self.global_step < TOTAL_TRAINING_STEPS:
                self.step()
        except KeyboardInterrupt:
            print("\n[Trainer] Training loop interrupted by user (Ctrl+C).")
        except Exception as e:
            print(f"\n[Trainer] CRITICAL ERROR in training loop: {e}")
            traceback.print_exc()
        finally:
            print("[Trainer] Training loop finished or terminated.")
            self.cleanup(save_final=True)

    def cleanup(self, save_final: bool = True):
        """Cleans up resources, optionally saving a final checkpoint."""
        print("[Trainer] Cleaning up resources...")
        if save_final:
            print("[Trainer] Saving final checkpoint...")
            self.checkpoint_manager.save_checkpoint(
                self.global_step,
                self.experience_collector.get_episode_count(),
                is_final=True,
            )
        else:
            print("[Trainer] Skipping final save as requested.")
        if hasattr(self.stats_recorder, "close"):
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"Error closing stats recorder: {e}")
        print("[Trainer] Cleanup complete.")
