# File: training/trainer.py
import time
import torch
import numpy as np
import traceback
import random
from typing import List, Optional, Dict, Any, Union

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

# --- MODIFIED: Import GameState directly for type hinting ---
from environment.game_state import GameState, StateType

# --- END MODIFIED ---
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from stats.stats_recorder import StatsRecorderBase
from utils.helpers import ensure_numpy
from utils.types import (
    PrioritizedNumpyBatch,
    PrioritizedNumpyNStepBatch,
    NumpyBatch,
    NumpyNStepBatch,
)
from .experience_collector import ExperienceCollector
from .checkpoint_manager import CheckpointManager
from .training_utils import get_env_image_as_numpy


class Trainer:
    """Orchestrates the DQN training process."""

    def __init__(
        self,
        envs: List[GameState],  # Keep GameState type hint
        agent: DQNAgent,
        buffer: ReplayBufferBase,
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
        # --- MODIFIED: num_envs is max, collector tracks active ---
        self.max_num_envs = env_config.NUM_ENVS
        # --- END MODIFIED ---
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
            buffer=buffer,
            model_save_path=model_save_path,
            buffer_save_path=buffer_save_path,
            load_checkpoint_path=load_checkpoint_path,
            load_buffer_path=load_buffer_path,
            buffer_config=self.buffer_config,
            dqn_config=self.dqn_config,
            device=self.device,
        )
        self.buffer = self.checkpoint_manager.get_buffer()
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
        # Initialize collector's episode count from checkpoint if loaded
        self.experience_collector.episode_count = initial_episode_count

        self.last_image_log_step = -self.tb_config.IMAGE_LOG_FREQ
        self.is_n_step_buffer = (
            self.buffer_config.USE_N_STEP and self.buffer_config.N_STEP > 1
        )

        self._log_initial_state()
        print("[Trainer] Initialization complete.")

    def _log_initial_state(self):
        """Logs the state after initialization and potential loading."""
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
        """Retrieves the current learning rate from the optimizer."""
        return (
            self.agent.optimizer.param_groups[0]["lr"]
            if hasattr(self.agent, "optimizer") and self.agent.optimizer.param_groups
            else 0.0
        )

    def step(self):
        """Performs one full step: collect experience, train, update target net."""
        step_start_time = time.perf_counter()

        # --- MODIFIED: Use steps_collected from collector ---
        steps_collected = self.experience_collector.collect(self.global_step)
        if steps_collected == 0:  # Handle case where no envs are active yet
            time.sleep(0.01)  # Small sleep to prevent busy-waiting
            return

        step_before_collection = self.global_step
        self.global_step += steps_collected
        # --- END MODIFIED ---

        if (
            self.global_step >= self.train_config.LEARN_START_STEP
            and self.global_step % self.train_config.LEARN_FREQ == 0
        ):
            if len(self.buffer) >= self.train_config.BATCH_SIZE:
                self._train_batch()

        step_end_time = time.perf_counter()
        step_duration = step_end_time - step_start_time

        self.stats_recorder.record_step(
            {
                "step_time": step_duration,
                "num_steps_processed": steps_collected,  # Log actual steps processed
                "global_step": self.global_step,
            }
        )

        target_freq = self.dqn_config.TARGET_UPDATE_FREQ
        if target_freq > 0 and self.global_step > 0:
            # --- MODIFIED: Use step_before_collection for comparison ---
            if step_before_collection // target_freq < self.global_step // target_freq:
                # --- END MODIFIED ---
                print(f"[Trainer] Updating target network at step {self.global_step}")
                self.agent.update_target_network()

        self.maybe_save_checkpoint(
            step_before_collection
        )  # Pass previous step for freq check
        self._maybe_log_image()

    def _train_batch(self):
        """Samples a batch, computes loss, updates agent, and updates priorities (for PER)."""
        beta = self._update_beta()
        batch_sample = None
        try:
            batch_sample = self.buffer.sample(self.train_config.BATCH_SIZE)
            if batch_sample is None:
                # print(f"[Trainer] Buffer sample returned None (Size: {len(self.buffer)}). Skipping train step.")
                return  # Not enough samples or other issue
        except Exception as e:
            print(f"ERROR sampling buffer: {e}")
            traceback.print_exc()
            return

        # Unpack sample based on whether it's prioritized
        indices, is_weights_np = None, None
        if self.buffer_config.USE_PER:
            if (
                batch_sample
                and isinstance(batch_sample, tuple)
                and len(batch_sample) == 3
            ):
                batch_tuple, indices, is_weights_np = batch_sample
            else:
                print(
                    f"Warning: Expected prioritized batch (tuple of 3), got {type(batch_sample)}. Skipping training step."
                )
                return
        else:  # Uniform sampling
            # Batch tuple is the whole sample for uniform (length 5 or 6)
            if (
                batch_sample
                and isinstance(batch_sample, tuple)
                and len(batch_sample) in [5, 6]
            ):
                batch_tuple = batch_sample
            else:
                print(
                    f"Warning: Expected uniform batch (tuple of 5 or 6), got {type(batch_sample)}. Skipping training step."
                )
                return

        loss, td_errors = torch.tensor(0.0), None
        try:
            # Agent's compute_loss handles checking buffer_config for is_n_step
            loss, td_errors = self.agent.compute_loss(batch_tuple, is_weights_np)
        except Exception as e:
            print(f"ERROR computing loss: {e}")
            traceback.print_exc()
            return

        grad_norm = None
        try:
            grad_norm = self.agent.update(loss)
            if grad_norm is None and self.dqn_config.GRADIENT_CLIP_NORM > 0:
                print("[Trainer] Skipping PER update due to gradient clipping error.")
                return  # Skip PER update if agent update failed
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
            train_log_data["batch_td_errors"] = (
                td_errors.detach() if td_errors.requires_grad else td_errors
            )

        self.stats_recorder.record_step(train_log_data)

    def _update_beta(self) -> float:
        """Updates PER beta based on annealing schedule and sets it in the buffer."""
        beta = 1.0  # Default for uniform
        if self.buffer_config.USE_PER:
            start, end, frames = (
                self.buffer_config.PER_BETA_START,
                1.0,
                self.buffer_config.PER_BETA_FRAMES,
            )
            fraction = min(1.0, float(self.global_step) / max(1, frames))
            beta = start + fraction * (end - start)
            self.buffer.set_beta(beta)  # Update buffer's beta

        # Always record beta for consistency
        self.stats_recorder.record_step({"beta": beta, "global_step": self.global_step})
        return beta

    def _maybe_log_image(self):
        """Logs a sample environment state image to TensorBoard periodically."""
        if not self.tb_config.LOG_IMAGES:
            return
        img_freq = self.tb_config.IMAGE_LOG_FREQ
        if img_freq <= 0:
            return

        steps_since_last = self.global_step - self.last_image_log_step
        if steps_since_last >= img_freq:
            try:
                # --- MODIFIED: Sample from active envs ---
                num_currently_active = self.experience_collector.num_active_envs
                if num_currently_active == 0:
                    return
                env_idx = random.randint(0, num_currently_active - 1)
                # --- END MODIFIED ---
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

    # --- MODIFIED: Pass previous step for frequency check ---
    def maybe_save_checkpoint(self, step_before_collection: int, force_save=False):
        # --- END MODIFIED ---
        """Saves agent and buffer state based on frequency or if forced."""
        save_freq = self.train_config.CHECKPOINT_SAVE_FREQ
        if save_freq <= 0 and not force_save:
            return

        # --- MODIFIED: Use step_before_collection for frequency check ---
        should_save_freq = (
            save_freq > 0
            and self.global_step > 0
            and (step_before_collection // save_freq < self.global_step // save_freq)
        )
        # --- END MODIFIED ---

        if force_save or should_save_freq:
            self.checkpoint_manager.save_checkpoint(
                self.global_step, self.experience_collector.get_episode_count()
            )

    def train_loop(self):
        """Main training loop until max steps."""
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
        """Performs cleanup actions like saving final state and closing logger."""
        print("[Trainer] Cleaning up resources...")
        if save_final:
            print("[Trainer] Saving final checkpoint...")
            if hasattr(self.buffer, "flush_pending"):
                print("[Trainer] Flushing pending buffer transitions...")
                try:
                    self.buffer.flush_pending()
                except Exception as flush_err:
                    print(f"ERROR flushing buffer: {flush_err}")
                    traceback.print_exc()
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
