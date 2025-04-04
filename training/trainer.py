# File: training/trainer.py
import time
import torch
import torch.nn.functional as F  # Import F for potential use in logging Q-values
import numpy as np
import os
import pickle
import random
import traceback
import pygame
from typing import List, Optional, Union, Tuple, Callable
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
from environment.game_state import GameState
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from agent.replay_buffer.buffer_utils import create_replay_buffer
from stats.stats_recorder import StatsRecorderBase
from utils.helpers import ensure_numpy, load_object, save_object
from utils.types import ActionType, StateType


class Trainer:
    """Orchestrates the DQN training process."""

    def __init__(
        self,
        envs: List[GameState],
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
        self.buffer = buffer
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.device = DEVICE
        self.env_config = env_config
        self.dqn_config = dqn_config
        self.train_config = train_config
        self.buffer_config = buffer_config
        self.model_config = model_config
        self.reward_config = RewardConfig
        self.tb_config = TensorBoardConfig
        self.vis_config = VisConfig
        self.model_save_path = model_save_path
        self.buffer_save_path = buffer_save_path

        # Initialize state tracking
        self.global_step = 0
        self.episode_count = 0
        self.last_image_log_step = (
            -self.tb_config.IMAGE_LOG_FREQ
        )  # Ensure first log happens

        # Initialize environment states and episode tracking arrays
        try:
            self.current_states: List[StateType] = [
                ensure_numpy(env.reset()) for env in self.envs
            ]
        except Exception as e:
            print(f"FATAL ERROR during initial environment reset: {e}")
            traceback.print_exc()
            raise e
        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_game_scores = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_lines_cleared = np.zeros(self.num_envs, dtype=np.int32)

        # Load checkpoint and buffer if paths are provided
        if load_checkpoint_path:
            self._load_checkpoint(load_checkpoint_path)
        else:
            print("[Trainer] No checkpoint specified, starting agent fresh.")
            self._reset_trainer_state()  # Ensure state is reset if no checkpoint

        if load_buffer_path:
            self._load_buffer_state(load_buffer_path)
        else:
            print("[Trainer] No buffer specified, starting buffer empty.")
            # Ensure buffer is created even if not loading
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)

        # Initialize PER beta and record initial stats
        initial_beta = self._update_beta()
        initial_lr = (
            self.agent.optimizer.param_groups[0]["lr"]
            if hasattr(self.agent, "optimizer")
            else 0.0
        )
        self.stats_recorder.record_step(
            {
                "buffer_size": len(self.buffer),
                "epsilon": 0.0,  # Epsilon is effectively 0 with Noisy Nets
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
            # Load the entire state dict first
            checkpoint = torch.load(path_to_load, map_location=self.device)

            # Load into agent (handles net, optim, scheduler)
            self.agent.load_state_dict(checkpoint)

            # Load trainer state (global step, episode count)
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
        except KeyError as e:
            print(
                f"[Trainer] Checkpoint missing key '{e}'. Incompatible? Starting fresh."
            )
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
            self.buffer.load_state(
                path_to_load
            )  # Load state into the new buffer instance
            print(f"[Trainer] Buffer state loaded. Size: {len(self.buffer)}")
        except (
            FileNotFoundError,
            EOFError,
            pickle.UnpicklingError,
            ImportError,
            AttributeError,
            ValueError,
        ) as e:
            print(
                f"[Trainer] ERROR loading buffer (incompatible/corrupt?): {e}. Starting empty."
            )
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
            # Get agent state (includes scheduler now)
            agent_save_data = self.agent.get_state_dict()
            # Add trainer state to the same dictionary
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
            # Flush any pending N-step transitions before saving
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
            beta = 1.0  # Beta is irrelevant if PER is not used
        else:
            start = self.buffer_config.PER_BETA_START
            end = 1.0
            frames = self.buffer_config.PER_BETA_FRAMES
            fraction = min(
                1.0, float(self.global_step) / max(1, frames)
            )  # Avoid division by zero
            beta = start + fraction * (end - start)
            if hasattr(self.buffer, "set_beta"):
                self.buffer.set_beta(beta)

        # Record beta regardless of PER usage (will be 1.0 if not used)
        self.stats_recorder.record_step({"beta": beta, "global_step": self.global_step})
        return beta

    def _collect_experience(self):
        """Collects one step of experience from each parallel environment."""
        actions: List[ActionType] = [
            -1
        ] * self.num_envs  # Initialize with invalid action
        step_rewards_batch = np.zeros(self.num_envs, dtype=np.float32)
        start_time = time.time()

        # 1. Select actions for all environments
        for i in range(self.num_envs):
            # If env was done, it should have been reset in the previous iteration's step handling
            # We use self.current_states[i] which holds the state after potential reset
            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                # This happens if the game ends immediately upon reset or enters a state with no moves
                # print(f"Warning: Env {i} has no valid actions at step start. Using action 0.")
                actions[i] = (
                    0  # Choose a default action (e.g., 0) or handle appropriately
                )
            else:
                try:
                    # Epsilon is ignored by agent if using Noisy Nets
                    actions[i] = self.agent.select_action(
                        self.current_states[i], 0.0, valid_actions
                    )
                except Exception as e:
                    print(f"ERROR: Agent select_action failed for env {i}: {e}")
                    traceback.print_exc()
                    actions[i] = random.choice(
                        valid_actions
                    )  # Fallback to random valid action

        # 2. Step all environments with selected actions
        next_states_list: List[StateType] = [
            np.zeros_like(self.current_states[0]) for _ in range(self.num_envs)
        ]
        rewards_list = np.zeros(self.num_envs, dtype=np.float32)
        dones_list = np.zeros(self.num_envs, dtype=bool)

        for i in range(self.num_envs):
            env = self.envs[i]
            current_state = self.current_states[i]  # State used for action selection
            action = actions[i]

            try:
                # --- FIX: Correct Experience Collection ---
                # Step the environment
                reward, done = env.step(action)

                # Get the state resulting from the action *before* potential reset
                next_state_raw = env.get_state()

                # Store results for this step
                rewards_list[i] = reward
                dones_list[i] = done
                step_rewards_batch[i] = reward  # For logging batch rewards

                # Push the transition (S, A, R, S') to the buffer
                # S' is the state *before* reset if done is True
                self.buffer.push(
                    current_state, action, reward, ensure_numpy(next_state_raw), done
                )

                # Update episode trackers
                self.current_episode_scores[i] += reward
                self.current_episode_lengths[i] += 1
                if not done:
                    # Update game score/lines only if not done (use final values otherwise)
                    self.current_episode_game_scores[i] = env.game_score
                    self.current_episode_lines_cleared[i] = (
                        env.lines_cleared_this_episode
                    )

                # Handle environment termination
                if done:
                    self.episode_count += 1
                    # Record completed episode stats
                    self.stats_recorder.record_episode(
                        episode_score=self.current_episode_scores[i],
                        episode_length=self.current_episode_lengths[i],
                        episode_num=self.episode_count,
                        global_step=self.global_step + i + 1,  # Approximate step count
                        game_score=self.current_episode_game_scores[
                            i
                        ],  # Use final game score
                        lines_cleared=self.current_episode_lines_cleared[
                            i
                        ],  # Use final lines cleared
                    )
                    # Reset environment and store the *new* starting state for the *next* iteration
                    next_states_list[i] = ensure_numpy(env.reset())
                    # Reset episode trackers for this environment
                    self.current_episode_scores[i] = 0.0
                    self.current_episode_lengths[i] = 0
                    self.current_episode_game_scores[i] = 0
                    self.current_episode_lines_cleared[i] = 0
                else:
                    # If not done, the next state for the next iteration is the one we observed
                    next_states_list[i] = ensure_numpy(next_state_raw)
                # --- END FIX ---

            except Exception as e:
                print(f"ERROR: Env {i} step/reset failed (Action: {action}): {e}")
                traceback.print_exc()
                # Handle environment crash: record minimal reward, mark as done, reset
                rewards_list[i] = self.reward_config.PENALTY_GAME_OVER
                dones_list[i] = True
                step_rewards_batch[i] = self.reward_config.PENALTY_GAME_OVER
                # Attempt to reset the crashed environment
                try:
                    crashed_state_reset = ensure_numpy(env.reset())
                except Exception as reset_e:
                    print(
                        f"FATAL: Env {i} failed to reset after crash: {reset_e}. Replacing state with zeros."
                    )
                    crashed_state_reset = np.zeros_like(self.current_states[i])

                # Push a terminal transition for the failed step
                self.buffer.push(
                    current_state, action, rewards_list[i], crashed_state_reset, True
                )

                # Record episode as ended due to error
                self.episode_count += 1
                self.stats_recorder.record_episode(
                    episode_score=self.current_episode_scores[i]
                    + rewards_list[i],  # Include penalty
                    episode_length=self.current_episode_lengths[i] + 1,
                    episode_num=self.episode_count,
                    global_step=self.global_step + i + 1,
                    game_score=self.current_episode_game_scores[
                        i
                    ],  # Last known game score
                    lines_cleared=self.current_episode_lines_cleared[
                        i
                    ],  # Last known lines
                )

                # Store the reset state for the next iteration
                next_states_list[i] = crashed_state_reset
                # Reset episode trackers
                self.current_episode_scores[i] = 0.0
                self.current_episode_lengths[i] = 0
                self.current_episode_game_scores[i] = 0
                self.current_episode_lines_cleared[i] = 0

        # 3. Update current states for the next iteration
        self.current_states = next_states_list
        self.global_step += self.num_envs  # Increment global step count

        # 4. Record step statistics
        end_time = time.time()
        step_duration = end_time - start_time
        step_log_data = {
            "buffer_size": len(self.buffer),
            "global_step": self.global_step,
            "step_time": step_duration,
            "num_steps_processed": self.num_envs,
        }
        # Optionally log batch rewards/actions for histograms
        if self.tb_config.LOG_HISTOGRAMS:
            step_log_data["step_rewards_batch"] = step_rewards_batch
            step_log_data["action_batch"] = np.array(actions, dtype=np.int32)
        self.stats_recorder.record_step(step_log_data)

    def _train_batch(self):
        """Samples a batch, computes loss, updates agent, and records stats."""
        # Check if learning should start
        if (
            len(self.buffer) < self.train_config.BATCH_SIZE
            or self.global_step < self.train_config.LEARN_START_STEP
        ):
            return  # Not enough samples or too early to learn

        # Update PER beta
        beta = self._update_beta()

        # Sample from buffer
        is_n_step = self.buffer_config.USE_N_STEP and self.buffer_config.N_STEP > 1
        indices, is_weights_np, batch_np_tuple = None, None, None
        try:
            if self.buffer_config.USE_PER:
                sample_result = self.buffer.sample(self.train_config.BATCH_SIZE)
                if sample_result:
                    batch_np_tuple, indices, is_weights_np = sample_result
                else:
                    # Handle case where buffer sample returns None (e.g., PER error)
                    print(
                        "Warning: Buffer sample returned None. Skipping training step."
                    )
                    return
            else:  # Uniform sampling
                batch_np_tuple = self.buffer.sample(self.train_config.BATCH_SIZE)

            if batch_np_tuple is None:  # Should be caught above, but double-check
                print(
                    "Warning: Buffer sample returned None tuple. Skipping training step."
                )
                return

        except Exception as e:
            print(f"ERROR sampling buffer: {e}")
            traceback.print_exc()
            return

        # Compute loss and TD errors
        raw_q_values = None  # For potential histogram logging
        loss = torch.tensor(0.0)  # Default loss
        td_errors = None
        try:
            # --- Optional: Log raw Q-values before loss computation ---
            # This requires converting batch to tensor first
            if self.tb_config.LOG_HISTOGRAMS:
                with torch.no_grad():
                    tensor_batch_log = self.agent._np_batch_to_tensor(
                        batch_np_tuple, is_n_step
                    )
                    self.agent.online_net.eval()  # Use eval mode for consistent Q-values
                    if self.agent.use_distributional:
                        dist_logits = self.agent.online_net(tensor_batch_log[0])
                        raw_q_values = (
                            F.softmax(dist_logits, dim=2) * self.agent.support
                        ).sum(dim=2)
                    else:
                        raw_q_values = self.agent.online_net(tensor_batch_log[0])
                    self.agent.online_net.train()  # Switch back to train mode
            # --- End Optional Logging ---

            # Compute loss (handles distributional/standard, PER weights)
            loss, td_errors = self.agent.compute_loss(
                batch_np_tuple, is_n_step, is_weights_np
            )

        except Exception as e:
            print(f"ERROR computing loss: {e}")
            traceback.print_exc()
            # Potentially skip update if loss computation fails critically
            return

        # Update agent (optimizer step, scheduler step, noise reset)
        grad_norm = None
        try:
            grad_norm = self.agent.update(loss)  # Update includes scheduler step
        except Exception as e:
            print(f"ERROR updating agent: {e}")
            traceback.print_exc()
            # Potentially skip priority update if agent update fails
            return

        # Update priorities in PER buffer
        if self.buffer_config.USE_PER and indices is not None and td_errors is not None:
            try:
                # Ensure td_errors are on CPU and numpy for SumTree
                td_errors_np = td_errors.squeeze().cpu().numpy()
                self.buffer.update_priorities(indices, td_errors_np)
            except Exception as e:
                print(f"ERROR updating PER priorities: {e}")
                traceback.print_exc()  # Log error but continue

        # Record training statistics
        current_lr = self.agent.optimizer.param_groups[0]["lr"]
        train_log_data = {
            "loss": loss.item(),
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
            "avg_max_q": self.agent.get_last_avg_max_q(),  # Get Q value from agent
            "lr": current_lr,
            "global_step": self.global_step,
        }
        # Add optional histograms
        if self.tb_config.LOG_HISTOGRAMS:
            if raw_q_values is not None:
                train_log_data["batch_q_values"] = (
                    raw_q_values  # Log Q-values if calculated
                )
            if td_errors is not None:
                train_log_data["batch_td_errors"] = td_errors  # Log TD errors
        self.stats_recorder.record_step(train_log_data)

    def step(self):
        """Performs one full step: collect experience, train, update target net."""
        # 1. Collect experience from environments
        self._collect_experience()

        # 2. Perform training step(s) if conditions met
        if (
            self.global_step >= self.train_config.LEARN_START_STEP
            and self.global_step % self.train_config.LEARN_FREQ == 0
        ):
            # Check buffer size again just before training
            if len(self.buffer) >= self.train_config.BATCH_SIZE:
                self._train_batch()
            # else: print(f"Skipping train step {self.global_step}: Buffer size {len(self.buffer)} < Batch size {self.train_config.BATCH_SIZE}")

        # 3. Update target network periodically
        target_freq = self.dqn_config.TARGET_UPDATE_FREQ
        if target_freq > 0 and self.global_step > 0:
            # Check if the update threshold was crossed *during the last num_envs steps*
            steps_before_this_iter = self.global_step - self.num_envs
            if steps_before_this_iter // target_freq < self.global_step // target_freq:
                print(f"[Trainer] Updating target network at step {self.global_step}")
                self.agent.update_target_network()

        # 4. Maybe save checkpoint and log images
        self.maybe_save_checkpoint()
        self._maybe_log_image()

    def _maybe_log_image(self):
        """Logs an image of a random environment state to TensorBoard periodically."""
        if not self.tb_config.LOG_IMAGES:
            return

        img_freq = self.tb_config.IMAGE_LOG_FREQ
        # Check if the logging threshold was crossed during the last num_envs steps
        steps_before_this_iter = self.global_step - self.num_envs
        if (
            img_freq > 0
            and steps_before_this_iter // img_freq < self.global_step // img_freq
        ):
            try:
                env_idx = random.randint(0, self.num_envs - 1)
                img_array = self._get_env_image_as_numpy(env_idx)
                if img_array is not None:
                    # Convert HWC (Pygame) to CHW (TensorBoard)
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
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
        # Define desired image dimensions (can be adjusted)
        img_h = 300
        # Calculate width based on aspect ratio of the triangular grid
        aspect_ratio = (self.env_config.COLS * 0.75 + 0.25) / max(
            1, self.env_config.ROWS
        )
        img_w = int(img_h * aspect_ratio)

        if img_w <= 0 or img_h <= 0:
            return None  # Invalid dimensions

        try:
            # Create a temporary Pygame surface
            temp_surf = pygame.Surface((img_w, img_h))
            # Calculate cell dimensions for rendering on this surface
            cell_w_px = img_w / (self.env_config.COLS * 0.75 + 0.25)
            cell_h_px = img_h / max(1, self.env_config.ROWS)

            temp_surf.fill(self.vis_config.BLACK)  # Background

            # Render grid triangles
            if hasattr(env, "grid") and hasattr(env.grid, "triangles"):
                for r in range(env.grid.rows):
                    for c in range(env.grid.cols):
                        # Basic check for grid structure validity
                        if r < len(env.grid.triangles) and c < len(
                            env.grid.triangles[r]
                        ):
                            t = env.grid.triangles[r][c]
                            pts = t.get_points(
                                ox=0, oy=0, cw=int(cell_w_px), ch=int(cell_h_px)
                            )
                            color = self.vis_config.GRAY  # Default for empty
                            if t.is_death:
                                color = (10, 10, 10)  # Darker gray for death cells
                            elif t.is_occupied:
                                color = (
                                    t.color if t.color else self.vis_config.RED
                                )  # Use shape color or red fallback
                            pygame.draw.polygon(temp_surf, color, pts)
                        # else: print(f"Warning: Grid index out of bounds ({r},{c}) during image render")

            # Convert Pygame surface to NumPy array (HWC format)
            img_array = pygame.surfarray.array3d(temp_surf)
            # Pygame gives W, H, C -> Transpose to H, W, C
            return np.transpose(img_array, (1, 0, 2))

        except Exception as e:
            print(f"Error generating environment image for TB: {e}")
            traceback.print_exc()
            return None

    def maybe_save_checkpoint(self, force_save=False):
        """Saves a checkpoint periodically or if forced."""
        save_freq = self.train_config.CHECKPOINT_SAVE_FREQ
        if save_freq <= 0 and not force_save:
            return  # Saving disabled unless forced

        # Check if the save threshold was crossed during the last num_envs steps
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
            self.cleanup(save_final=True)  # Ensure cleanup and final save

    def cleanup(self, save_final: bool = True):
        """Cleans up resources, optionally saving a final checkpoint."""
        print("[Trainer] Cleaning up resources...")

        # Flush N-step buffer if applicable
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

        # Save final checkpoint
        if save_final:
            print("[Trainer] Saving final checkpoint...")
            self._save_checkpoint(is_final=True)
        else:
            print("[Trainer] Skipping final save as requested.")

        # Close stats recorder (e.g., TensorBoard writer)
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
