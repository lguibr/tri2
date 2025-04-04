# File: training/trainer.py
import time
import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import random
import traceback
import pygame
from typing import List, Optional, Union, Tuple, Callable, Dict  # Added Dict
from collections import deque

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

# --- MODIFIED: Import specific StateType ---
from environment.game_state import GameState, StateType  # Use the Dict type

# --- END MODIFIED ---
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from agent.replay_buffer.buffer_utils import create_replay_buffer
from stats.stats_recorder import StatsRecorderBase
from utils.helpers import ensure_numpy, load_object, save_object
from utils.types import ActionType  # Removed StateType import here


class Trainer:
    """Orchestrates the DQN training process."""

    def __init__(
        self,
        envs: List[GameState],
        agent: DQNAgent,
        buffer: ReplayBufferBase,
        stats_recorder: StatsRecorderBase,
        env_config: EnvConfig,  # Keep passing instance
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
        self.buffer = buffer
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.device = DEVICE
        self.env_config = env_config  # Store instance
        self.dqn_config = dqn_config
        self.train_config = train_config
        self.buffer_config = buffer_config
        self.model_config = model_config
        self.reward_config = RewardConfig
        self.tb_config = TensorBoardConfig
        self.vis_config = VisConfig
        self.model_save_path = model_save_path
        self.buffer_save_path = buffer_save_path

        self.global_step = 0
        self.episode_count = 0
        self.last_image_log_step = -self.tb_config.IMAGE_LOG_FREQ

        # --- MODIFIED: Initialize current_states as list of dicts ---
        try:
            # Get initial state dictionaries
            self.current_states: List[StateType] = [env.reset() for env in self.envs]
        except Exception as e:
            print(f"FATAL ERROR during initial environment reset: {e}")
            traceback.print_exc()
            raise e
        # --- END MODIFIED ---

        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_game_scores = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_lines_cleared = np.zeros(self.num_envs, dtype=np.int32)

        if load_checkpoint_path:
            self._load_checkpoint(load_checkpoint_path)
        else:
            print("[Trainer] No checkpoint specified, starting agent fresh.")
            self._reset_trainer_state()

        if load_buffer_path:
            self._load_buffer_state(load_buffer_path)
        else:
            print("[Trainer] No buffer specified, starting buffer empty.")
            # Ensure buffer is created even if not loading
            if not hasattr(self, "buffer") or self.buffer is None:
                self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)

        initial_beta = self._update_beta()
        initial_lr = (
            self.agent.optimizer.param_groups[0]["lr"]
            if hasattr(self.agent, "optimizer") and self.agent.optimizer.param_groups
            else 0.0
        )
        self.stats_recorder.record_step(
            {
                "buffer_size": len(self.buffer),
                "epsilon": 0.0,  # Assuming Noisy Nets, otherwise need epsilon tracking
                "beta": initial_beta,
                "lr": initial_lr,
                "global_step": self.global_step,
                "episode_count": self.episode_count,
            }
        )

        print(
            f"[Trainer] Init complete. Start Step={self.global_step}, Ep={self.episode_count}, Buf={len(self.buffer)}, Beta={initial_beta:.4f}, LR={initial_lr:.1e}"
        )

    def _load_checkpoint(self, path_to_load: str):
        """Loads agent and trainer state from a checkpoint file."""
        if not os.path.isfile(path_to_load):
            print(
                f"[Trainer] LOAD WARNING: Checkpoint file not found at {path_to_load}. Starting fresh."
            )
            self._reset_trainer_state()
            return

        print(f"[Trainer] Loading agent checkpoint from: {path_to_load}")
        try:
            checkpoint = torch.load(path_to_load, map_location=self.device)
            self.agent.load_state_dict(
                checkpoint
            )  # Agent handles loading its components
            self.global_step = checkpoint.get("global_step", 0)
            self.episode_count = checkpoint.get("episode_count", 0)
            print(
                f"[Trainer] Checkpoint loaded. Resuming from step {self.global_step}, ep {self.episode_count}"
            )
        except FileNotFoundError:
            print(
                f"[Trainer] Checkpoint file disappeared? ({path_to_load}). Starting fresh."
            )
            self._reset_trainer_state()
        except (
            KeyError,
            AttributeError,
            TypeError,
            RuntimeError,
        ) as e:  # Catch more potential load errors
            print(
                f"[Trainer] Checkpoint load error ('{e}'). Incompatible? Starting fresh."
            )
            traceback.print_exc()
            self._reset_trainer_state()
        except Exception as e:
            print(f"[Trainer] CRITICAL ERROR loading checkpoint: {e}. Starting fresh.")
            traceback.print_exc()
            self._reset_trainer_state()

    def _reset_trainer_state(self):
        """Resets trainer-specific state variables."""
        self.global_step = 0
        self.episode_count = 0

    def _load_buffer_state(self, path_to_load: str):
        """Loads replay buffer state from a file."""
        if not os.path.isfile(path_to_load):
            print(
                f"[Trainer] LOAD WARNING: Buffer file not found at {path_to_load}. Starting empty."
            )
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
            return

        print(f"[Trainer] Attempting to load buffer state from: {path_to_load}")
        try:
            # Recreate buffer first to ensure correct type and config
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
            self.buffer.load_state(path_to_load)
            print(f"[Trainer] Buffer state loaded. Size: {len(self.buffer)}")
        except (
            FileNotFoundError,
            EOFError,
            pickle.UnpicklingError,
            ImportError,
            AttributeError,
            ValueError,
            TypeError,
        ) as e:  # Catch more load errors
            print(
                f"[Trainer] ERROR loading buffer (incompatible/corrupt?): {e}. Starting empty."
            )
            traceback.print_exc()
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
        except Exception as e:
            print(f"[Trainer] CRITICAL ERROR loading buffer: {e}. Starting empty.")
            traceback.print_exc()
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)

    def _save_checkpoint(self, is_final=False):
        """Saves agent and buffer state."""
        prefix = "FINAL" if is_final else f"step_{self.global_step}"
        save_dir = os.path.dirname(self.model_save_path)
        os.makedirs(save_dir, exist_ok=True)

        # Save Agent State
        print(
            f"[Trainer] Saving agent checkpoint ({prefix}) to: {self.model_save_path}"
        )
        try:
            agent_save_data = self.agent.get_state_dict()
            agent_save_data["global_step"] = self.global_step
            agent_save_data["episode_count"] = self.episode_count
            torch.save(agent_save_data, self.model_save_path)
            print(f"[Trainer] Agent checkpoint ({prefix}) saved.")
        except Exception as e:
            print(f"[Trainer] ERROR saving agent checkpoint ({prefix}): {e}")
            traceback.print_exc()

        # Save Buffer State
        print(
            f"[Trainer] Saving buffer state ({prefix}) to: {self.buffer_save_path} (Size: {len(self.buffer)})"
        )
        try:
            if hasattr(self.buffer, "flush_pending"):
                self.buffer.flush_pending()
            self.buffer.save_state(self.buffer_save_path)
            print(f"[Trainer] Buffer state ({prefix}) saved.")
        except Exception as e:
            print(f"[Trainer] ERROR saving buffer state ({prefix}): {e}")
            traceback.print_exc()

    def _update_beta(self) -> float:
        """Updates PER beta based on global step and records it."""
        if not self.buffer_config.USE_PER:
            beta = 1.0
        else:
            start = self.buffer_config.PER_BETA_START
            end = 1.0
            frames = self.buffer_config.PER_BETA_FRAMES
            fraction = min(1.0, float(self.global_step) / max(1, frames))
            beta = start + fraction * (end - start)
            if hasattr(self.buffer, "set_beta"):
                self.buffer.set_beta(beta)

        self.stats_recorder.record_step({"beta": beta, "global_step": self.global_step})
        return beta

    # --- MODIFIED: _collect_experience handles dictionary states ---
    def _collect_experience(self):
        """Collects one step of experience from each parallel environment."""
        actions: List[ActionType] = [-1] * self.num_envs
        step_rewards_batch = np.zeros(self.num_envs, dtype=np.float32)
        start_time = time.time()

        # 1. Select actions for all environments based on current states (dicts)
        for i in range(self.num_envs):
            current_state_dict = self.current_states[i]
            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                actions[i] = 0  # Default action if no valid moves
            else:
                try:
                    # Epsilon is ignored by agent if using Noisy Nets
                    # Pass the state dictionary directly to the agent
                    actions[i] = self.agent.select_action(
                        current_state_dict, 0.0, valid_actions
                    )
                except Exception as e:
                    print(f"ERROR: Agent select_action failed for env {i}: {e}")
                    traceback.print_exc()
                    actions[i] = random.choice(valid_actions)

        # 2. Step all environments
        next_states_list: List[StateType] = [
            {} for _ in range(self.num_envs)
        ]  # List of dicts
        rewards_list = np.zeros(self.num_envs, dtype=np.float32)
        dones_list = np.zeros(self.num_envs, dtype=bool)

        for i in range(self.num_envs):
            env = self.envs[i]
            current_state_dict = self.current_states[
                i
            ]  # State used for action selection
            action = actions[i]

            try:
                reward, done = env.step(action)
                # Get the state dictionary resulting from the action
                next_state_dict = env.get_state()

                rewards_list[i] = reward
                dones_list[i] = done
                step_rewards_batch[i] = reward

                # Push the transition (State Dict, A, R, Next State Dict, D)
                self.buffer.push(
                    current_state_dict, action, reward, next_state_dict, done
                )

                # Update episode trackers
                self.current_episode_scores[i] += reward
                self.current_episode_lengths[i] += 1
                if not done:
                    self.current_episode_game_scores[i] = env.game_score
                    self.current_episode_lines_cleared[i] = (
                        env.lines_cleared_this_episode
                    )

                # Handle environment termination
                if done:
                    self.episode_count += 1
                    self.stats_recorder.record_episode(
                        episode_score=self.current_episode_scores[i],
                        episode_length=self.current_episode_lengths[i],
                        episode_num=self.episode_count,
                        global_step=self.global_step + i + 1,
                        game_score=self.current_episode_game_scores[i],
                        lines_cleared=self.current_episode_lines_cleared[i],
                    )
                    # Reset environment and store the *new* starting state dictionary
                    next_states_list[i] = env.reset()
                    # Reset episode trackers
                    self.current_episode_scores[i] = 0.0
                    self.current_episode_lengths[i] = 0
                    self.current_episode_game_scores[i] = 0
                    self.current_episode_lines_cleared[i] = 0
                else:
                    # If not done, the next state is the one observed
                    next_states_list[i] = next_state_dict

            except Exception as e:
                print(f"ERROR: Env {i} step/reset failed (Action: {action}): {e}")
                traceback.print_exc()
                rewards_list[i] = self.reward_config.PENALTY_GAME_OVER
                dones_list[i] = True
                step_rewards_batch[i] = self.reward_config.PENALTY_GAME_OVER
                try:
                    crashed_state_reset = env.reset()  # Get state dict
                except Exception as reset_e:
                    print(
                        f"FATAL: Env {i} failed to reset after crash: {reset_e}. Creating zero state."
                    )
                    # Create a zero state dictionary matching the expected structure
                    grid_zeros = np.zeros(
                        self.env_config.GRID_STATE_SHAPE, dtype=np.float32
                    )
                    shape_zeros = np.zeros(
                        (
                            self.env_config.NUM_SHAPE_SLOTS,
                            self.env_config.SHAPE_FEATURES_PER_SHAPE,
                        ),
                        dtype=np.float32,
                    )
                    crashed_state_reset = {"grid": grid_zeros, "shapes": shape_zeros}

                self.buffer.push(
                    current_state_dict,
                    action,
                    rewards_list[i],
                    crashed_state_reset,
                    True,
                )
                self.episode_count += 1
                self.stats_recorder.record_episode(
                    episode_score=self.current_episode_scores[i] + rewards_list[i],
                    episode_length=self.current_episode_lengths[i] + 1,
                    episode_num=self.episode_count,
                    global_step=self.global_step + i + 1,
                    game_score=self.current_episode_game_scores[i],
                    lines_cleared=self.current_episode_lines_cleared[i],
                )
                next_states_list[i] = crashed_state_reset
                self.current_episode_scores[i] = 0.0
                self.current_episode_lengths[i] = 0
                self.current_episode_game_scores[i] = 0
                self.current_episode_lines_cleared[i] = 0

        # 3. Update current states for the next iteration
        self.current_states = next_states_list
        self.global_step += self.num_envs

        # 4. Record step statistics
        end_time = time.time()
        step_duration = end_time - start_time
        step_log_data = {
            "buffer_size": len(self.buffer),
            "global_step": self.global_step,
            "step_time": step_duration,
            "num_steps_processed": self.num_envs,
        }
        if self.tb_config.LOG_HISTOGRAMS:
            step_log_data["step_rewards_batch"] = step_rewards_batch
            step_log_data["action_batch"] = np.array(actions, dtype=np.int32)
        self.stats_recorder.record_step(step_log_data)

    # --- END MODIFIED ---

    def _train_batch(self):
        """Samples a batch, computes loss, updates agent, and records stats."""
        if (
            len(self.buffer) < self.train_config.BATCH_SIZE
            or self.global_step < self.train_config.LEARN_START_STEP
        ):
            return

        beta = self._update_beta()
        indices, is_weights_np, batch_tuple = None, None, None

        try:
            if self.buffer_config.USE_PER:
                sample_result = self.buffer.sample(self.train_config.BATCH_SIZE)
                if sample_result:
                    batch_tuple, indices, is_weights_np = sample_result
                else:
                    print(
                        "Warning: Buffer sample returned None (PER). Skipping training step."
                    )
                    return
            else:  # Uniform sampling
                batch_tuple = self.buffer.sample(self.train_config.BATCH_SIZE)
                if batch_tuple is None:
                    print(
                        "Warning: Buffer sample returned None (Uniform). Skipping training step."
                    )
                    return

        except Exception as e:
            print(f"ERROR sampling buffer: {e}")
            traceback.print_exc()
            return

        # Compute loss and TD errors
        loss = torch.tensor(0.0)
        td_errors = None
        raw_q_values_for_log = None  # For histogram logging

        try:
            # --- Optional: Log raw Q-values before loss computation ---
            if self.tb_config.LOG_HISTOGRAMS:
                with torch.no_grad():
                    # Convert numpy batch (with dict states) to tensors
                    tensor_batch_log = self.agent._np_batch_to_tensor(
                        batch_tuple, self.buffer_config.USE_N_STEP
                    )
                    states_tuple_log, _, _, _, _ = tensor_batch_log[:5]
                    grids_log, shapes_log = states_tuple_log

                    self.agent.online_net.eval()
                    if self.agent.use_distributional:
                        dist_logits = self.agent.online_net(grids_log, shapes_log)
                        raw_q_values_for_log = (
                            F.softmax(dist_logits, dim=2) * self.agent.support
                        ).sum(dim=2)
                    else:
                        raw_q_values_for_log = self.agent.online_net(
                            grids_log, shapes_log
                        )
                    self.agent.online_net.train()  # Switch back
            # --- End Optional Logging ---

            # Compute loss (handles distributional/standard, PER weights, dict states)
            loss, td_errors = self.agent.compute_loss(
                batch_tuple, self.buffer_config.USE_N_STEP, is_weights_np
            )

        except Exception as e:
            print(f"ERROR computing loss: {e}")
            traceback.print_exc()
            return  # Skip update if loss computation fails

        # Update agent
        grad_norm = None
        try:
            grad_norm = self.agent.update(loss)
            if grad_norm is None and self.gradient_clip_norm > 0:
                # Grad norm is None likely because clipping failed (e.g., NaN grads)
                print("Warning: Agent update skipped due to likely gradient issue.")
                return  # Skip priority update if agent update failed
        except Exception as e:
            print(f"ERROR updating agent: {e}")
            traceback.print_exc()
            return  # Skip priority update

        # Update priorities in PER buffer
        if self.buffer_config.USE_PER and indices is not None and td_errors is not None:
            try:
                # Ensure td_errors are numpy for SumTree
                td_errors_np = ensure_numpy(td_errors)
                self.buffer.update_priorities(indices, td_errors_np)
            except Exception as e:
                print(f"ERROR updating PER priorities: {e}")
                traceback.print_exc()

        # Record training statistics
        current_lr = self.agent.optimizer.param_groups[0]["lr"]
        train_log_data = {
            "loss": loss.item(),
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
            "avg_max_q": self.agent.get_last_avg_max_q(),
            "lr": current_lr,
            "global_step": self.global_step,
        }
        if self.tb_config.LOG_HISTOGRAMS:
            if raw_q_values_for_log is not None:
                train_log_data["batch_q_values"] = raw_q_values_for_log
            if td_errors is not None:
                train_log_data["batch_td_errors"] = td_errors
        self.stats_recorder.record_step(train_log_data)

    def step(self):
        """Performs one full step: collect experience, train, update target net."""
        self._collect_experience()

        if (
            self.global_step >= self.train_config.LEARN_START_STEP
            and self.global_step % self.train_config.LEARN_FREQ == 0
        ):
            if len(self.buffer) >= self.train_config.BATCH_SIZE:
                self._train_batch()

        target_freq = self.dqn_config.TARGET_UPDATE_FREQ
        if target_freq > 0 and self.global_step > 0:
            steps_before_this_iter = self.global_step - self.num_envs
            if steps_before_this_iter // target_freq < self.global_step // target_freq:
                print(f"[Trainer] Updating target network at step {self.global_step}")
                self.agent.update_target_network()

        self.maybe_save_checkpoint()
        self._maybe_log_image()

    def _maybe_log_image(self):
        """Logs an image of a random environment state to TensorBoard periodically."""
        if not self.tb_config.LOG_IMAGES:
            return

        img_freq = self.tb_config.IMAGE_LOG_FREQ
        steps_before_this_iter = self.global_step - self.num_envs
        if (
            img_freq > 0
            and steps_before_this_iter // img_freq < self.global_step // img_freq
        ):
            try:
                env_idx = random.randint(0, self.num_envs - 1)
                img_array = self._get_env_image_as_numpy(env_idx)
                if img_array is not None:
                    img_tensor = torch.from_numpy(img_array).permute(
                        2, 0, 1
                    )  # HWC to CHW
                    self.stats_recorder.record_image(
                        f"Environment/Sample State Env {env_idx}",
                        img_tensor,
                        self.global_step,
                    )
            except Exception as e:
                print(f"Error logging environment image: {e}")

    def _get_env_image_as_numpy(self, env_index: int) -> Optional[np.ndarray]:
        """Renders a single environment state to a NumPy array for logging."""
        if not (0 <= env_index < self.num_envs):
            return None

        env = self.envs[env_index]
        img_h = 300
        # Use env_config instance stored in trainer
        aspect_ratio = (self.env_config.COLS * 0.75 + 0.25) / max(
            1, self.env_config.ROWS
        )
        img_w = int(img_h * aspect_ratio)
        if img_w <= 0 or img_h <= 0:
            return None

        try:
            temp_surf = pygame.Surface((img_w, img_h))
            cell_w_px = img_w / (self.env_config.COLS * 0.75 + 0.25)
            cell_h_px = img_h / max(1, self.env_config.ROWS)
            temp_surf.fill(self.vis_config.BLACK)

            if hasattr(env, "grid") and hasattr(env.grid, "triangles"):
                for r in range(env.grid.rows):
                    for c in range(env.grid.cols):
                        if r < len(env.grid.triangles) and c < len(
                            env.grid.triangles[r]
                        ):
                            t = env.grid.triangles[r][c]
                            pts = t.get_points(
                                ox=0, oy=0, cw=int(cell_w_px), ch=int(cell_h_px)
                            )
                            color = self.vis_config.GRAY
                            if t.is_death:
                                color = (10, 10, 10)
                            elif t.is_occupied:
                                color = t.color if t.color else self.vis_config.RED
                            pygame.draw.polygon(temp_surf, color, pts)

            img_array = pygame.surfarray.array3d(temp_surf)
            return np.transpose(img_array, (1, 0, 2))  # W, H, C -> H, W, C
        except Exception as e:
            print(f"Error generating environment image for TB: {e}")
            traceback.print_exc()
            return None

    def maybe_save_checkpoint(self, force_save=False):
        """Saves a checkpoint periodically or if forced."""
        save_freq = self.train_config.CHECKPOINT_SAVE_FREQ
        if save_freq <= 0 and not force_save:
            return

        steps_before_this_iter = self.global_step - self.num_envs
        should_save_freq = (
            save_freq > 0
            and self.global_step > 0
            and steps_before_this_iter // save_freq < self.global_step // save_freq
        )

        if force_save or should_save_freq:
            self._save_checkpoint(is_final=False)

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
        if (
            hasattr(self, "buffer")
            and self.buffer
            and hasattr(self.buffer, "flush_pending")
        ):
            print("[Trainer] Flushing pending N-step transitions...")
            try:
                self.buffer.flush_pending()
            except Exception as flush_e:
                print(f"Error flushing buffer during cleanup: {flush_e}")

        if save_final:
            print("[Trainer] Saving final checkpoint...")
            self._save_checkpoint(is_final=True)
        else:
            print("[Trainer] Skipping final save as requested.")

        if (
            hasattr(self, "stats_recorder")
            and self.stats_recorder
            and hasattr(self.stats_recorder, "close")
        ):
            try:
                self.stats_recorder.close()
            except Exception as close_e:
                print(f"Error closing stats recorder during cleanup: {close_e}")

        print("[Trainer] Cleanup complete.")
