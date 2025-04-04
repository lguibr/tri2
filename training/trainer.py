# File: training/trainer.py
import time
import torch
import numpy as np
import os
import pickle
import random
import traceback
import pygame  # Added for image rendering helper
from typing import List, Optional, Union, Tuple
from collections import deque

from config import (
    EnvConfig,
    DQNConfig,
    TrainConfig,
    BufferConfig,
    ModelConfig,
    DEVICE,
    TensorBoardConfig,
    VisConfig,  # Import TB and Vis configs
)
from environment.game_state import GameState
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from agent.replay_buffer.buffer_utils import create_replay_buffer
from stats.stats_recorder import StatsRecorderBase  # Base class
from utils.helpers import ensure_numpy
from utils.types import ActionType, StateType


class Trainer:
    """Orchestrates the DQN training process."""

    def __init__(
        self,
        envs: List[GameState],
        agent: DQNAgent,
        buffer: ReplayBufferBase,
        stats_recorder: StatsRecorderBase,  # Should be TensorBoardStatsRecorder instance
        env_config: EnvConfig,
        dqn_config: DQNConfig,
        train_config: TrainConfig,
        buffer_config: BufferConfig,
        model_config: ModelConfig,
        model_save_path: str,  # Path to save model for THIS run
        buffer_save_path: str,  # Path to save buffer for THIS run
        load_checkpoint_path: Optional[str] = None,  # Optional path to LOAD model state
        load_buffer_path: Optional[str] = None,  # Optional path to LOAD buffer state
    ):
        print("[Trainer] Initializing...")
        self.envs = envs
        self.agent = agent
        self.buffer = buffer
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.device = DEVICE

        # Store configs
        self.env_config = env_config
        self.dqn_config = dqn_config
        self.train_config = train_config
        self.buffer_config = buffer_config
        self.model_config = model_config
        self.reward_config = envs[0].rewards  # Get reward config from an env instance

        # --- Add Config reference for logging flags ---
        self.tb_config = TensorBoardConfig
        self.vis_config = VisConfig  # Needed for rendering image helper
        # --- End Config reference ---

        # Specific paths for this run
        self.model_save_path = model_save_path
        self.buffer_save_path = buffer_save_path

        # State / Trackers
        self.global_step = 0
        self.episode_count = 0
        self.last_image_log_step = (
            -self.tb_config.IMAGE_LOG_FREQ
        )  # Track image logging frequency

        try:
            self.current_states: List[StateType] = [
                ensure_numpy(env.reset()) for env in self.envs
            ]
        except Exception as e:
            print(f"FATAL ERROR during initial reset: {e}")
            raise e
        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_game_scores = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_lines_cleared = np.zeros(self.num_envs, dtype=np.int32)

        # --- Loading State ---
        if load_checkpoint_path:
            self._load_checkpoint(load_checkpoint_path)
        else:
            print(
                "[Trainer] No checkpoint specified to load, starting agent from scratch."
            )
            self._reset_trainer_state()

        if load_buffer_path:
            self._load_buffer_state(load_buffer_path)
        else:
            print("[Trainer] No buffer specified to load, starting buffer empty.")
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)

        # Initial PER Beta update & logging
        initial_beta = self._update_beta()
        self.stats_recorder.record_step(
            {
                "buffer_size": len(self.buffer),
                "epsilon": 0.0,  # Noisy Nets
                "beta": initial_beta,
                "global_step": self.global_step,
                "episode_count": self.episode_count,
            }
        )

        print(
            f"[Trainer] Init complete. Start Step={self.global_step}, Ep={self.episode_count}, Buf={len(self.buffer)}, Beta={initial_beta:.4f}"
        )

    def _load_checkpoint(self, path_to_load: str):
        if not os.path.isfile(path_to_load):
            print(
                f"[Trainer] LOAD WARNING: Checkpoint file not found at {path_to_load}. Starting agent from scratch."
            )
            self._reset_trainer_state()
            return

        print(f"[Trainer] Loading agent checkpoint from: {path_to_load}")
        try:
            checkpoint = torch.load(path_to_load, map_location=self.device)
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
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
                f"[Trainer] Checkpoint missing key '{e}'. Incompatible format? Starting fresh."
            )
            self._reset_trainer_state()
        except Exception as e:
            print(f"[Trainer] CRITICAL ERROR loading checkpoint: {e}. Starting fresh.")
            traceback.print_exc()
            self._reset_trainer_state()

    def _reset_trainer_state(self):
        """Resets step and episode counters."""
        self.global_step = 0
        self.episode_count = 0

    def _load_buffer_state(self, path_to_load: str):
        if not os.path.isfile(path_to_load):
            print(
                f"[Trainer] LOAD WARNING: Buffer file not found at {path_to_load}. Starting empty buffer."
            )
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
            return

        print(f"[Trainer] Attempting to load buffer state from: {path_to_load}")
        try:
            if hasattr(self.buffer, "load_state"):
                self.buffer.load_state(path_to_load)
                print(f"[Trainer] Buffer state loaded. Size: {len(self.buffer)}")
            else:
                print(
                    "[Trainer] Warning: Buffer object has no 'load_state' method. Cannot load."
                )
        except FileNotFoundError:
            print(
                f"[Trainer] Buffer file disappeared? ({path_to_load}). Starting empty."
            )
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
        except (
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
        """Saves agent state and buffer state to the paths defined for THIS run."""
        prefix = "FINAL" if is_final else f"step_{self.global_step}"
        save_dir = os.path.dirname(self.model_save_path)
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        # Save Agent State
        print(
            f"[Trainer] Saving agent checkpoint ({prefix}) to: {self.model_save_path}"
        )
        try:
            save_data = {
                "global_step": self.global_step,
                "episode_count": self.episode_count,
                "agent_state_dict": self.agent.get_state_dict(),
                # Consider adding config hashes/identifiers here for compatibility checks
            }
            torch.save(save_data, self.model_save_path)
            # Save a copy as 'latest' maybe?
            # torch.save(save_data, os.path.join(save_dir, "latest_agent_state.pth"))
            print(f"[Trainer] Agent checkpoint ({prefix}) saved.")
        except Exception as e:
            print(f"[Trainer] ERROR saving agent checkpoint ({prefix}): {e}")
            traceback.print_exc()

        # Save Buffer State
        print(
            f"[Trainer] Saving buffer state ({prefix}) to: {self.buffer_save_path} (Size: {len(self.buffer)})"
        )
        try:
            if hasattr(self.buffer, "save_state"):
                self.buffer.save_state(self.buffer_save_path)
                # Save a copy as 'latest' maybe?
                # self.buffer.save_state(os.path.join(save_dir, "latest_replay_buffer_state.pkl"))
                print(f"[Trainer] Buffer state ({prefix}) saved.")
            else:
                print("[Trainer] Warning: Buffer does not support save_state.")
        except Exception as e:
            print(f"[Trainer] ERROR saving buffer state ({prefix}): {e}")
            traceback.print_exc()

    def _update_beta(self) -> float:
        """Updates PER beta based on global steps and returns current beta."""
        if not self.buffer_config.USE_PER:
            beta = 1.0
        else:
            start = self.buffer_config.PER_BETA_START
            end = 1.0
            anneal_frames = self.buffer_config.PER_BETA_FRAMES
            if anneal_frames <= 0:
                beta = end
            else:
                fraction = min(1.0, float(self.global_step) / anneal_frames)
                beta = start + fraction * (end - start)

            if hasattr(self.buffer, "set_beta"):
                self.buffer.set_beta(beta)
            else:
                print("Warning: PER enabled but buffer lacks set_beta method.")
        # Record beta via stats_recorder (which handles actual logging)
        # Note: record_step also needs global_step to associate the beta value correctly
        self.stats_recorder.record_step({"beta": beta, "global_step": self.global_step})
        return beta

    def _collect_experience(self):
        """Performs one step in each parallel env, stores transition, handles resets."""
        actions: List[ActionType] = [-1] * self.num_envs
        step_rewards_batch = np.zeros(self.num_envs, dtype=np.float32)  # For histogram

        # --- 1. Select Actions ---
        for i in range(self.num_envs):
            if self.envs[i].is_over():
                try:
                    self.current_states[i] = ensure_numpy(self.envs[i].reset())
                    self.current_episode_scores[i] = 0.0
                    self.current_episode_lengths[i] = 0
                    self.current_episode_game_scores[i] = 0
                    self.current_episode_lines_cleared[i] = 0
                except Exception as e:
                    print(f"ERROR: Env {i} failed reset: {e}")
                    self.current_states[i] = np.zeros(
                        self.env_config.STATE_DIM, dtype=np.float32
                    )

            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                actions[i] = 0
            else:
                try:
                    actions[i] = self.agent.select_action(
                        self.current_states[i], 0.0, valid_actions
                    )
                except Exception as e:
                    print(f"ERROR: Agent select_action env {i}: {e}")
                    actions[i] = random.choice(valid_actions)

        # --- 2. Step Environments & Store Transitions ---
        next_states_list: List[StateType] = [
            np.zeros_like(self.current_states[0]) for _ in range(self.num_envs)
        ]
        rewards_list = np.zeros(self.num_envs, dtype=np.float32)
        dones_list = np.zeros(self.num_envs, dtype=bool)

        for i in range(self.num_envs):
            env = self.envs[i]
            current_state = self.current_states[i]
            action = actions[i]

            try:
                reward, done = env.step(action)
                next_state = ensure_numpy(env.get_state())
            except Exception as e:
                print(f"ERROR: Env {i} step failed (Action: {action}): {e}")
                reward = self.reward_config.PENALTY_GAME_OVER
                done = True
                next_state = current_state
                if hasattr(env, "game_over"):
                    env.game_over = True

            rewards_list[i] = reward
            dones_list[i] = done
            next_states_list[i] = next_state
            step_rewards_batch[i] = reward  # Store for histogram

            try:
                self.buffer.push(current_state, action, reward, next_state, done)
            except Exception as e:
                print(f"ERROR: Buffer push env {i}: {e}")

            # --- 3. Update Trackers ---
            self.current_episode_scores[i] += reward
            self.current_episode_lengths[i] += 1
            self.current_episode_game_scores[i] = env.game_score
            self.current_episode_lines_cleared[i] = env.lines_cleared_this_episode

            # --- 4. Handle Episode End ---
            if done:
                self.episode_count += 1
                final_rl_score = self.current_episode_scores[i]
                final_length = self.current_episode_lengths[i]
                final_game_score = self.current_episode_game_scores[i]
                final_lines_cleared = self.current_episode_lines_cleared[i]

                self.stats_recorder.record_episode(
                    episode_score=final_rl_score,
                    episode_length=final_length,
                    episode_num=self.episode_count,
                    global_step=self.global_step
                    + self.num_envs,  # Step count *after* this batch
                    game_score=final_game_score,
                    lines_cleared=final_lines_cleared,
                )

        # --- 5. Update Current States ---
        self.current_states = next_states_list
        # --- 6. Increment Global Step ---
        self.global_step += self.num_envs

        # --- Log Step Data (including histograms) ---
        step_log_data = {
            "buffer_size": len(self.buffer),
            "global_step": self.global_step,
        }
        if self.tb_config.LOG_HISTOGRAMS:
            step_log_data["step_rewards_batch"] = step_rewards_batch
            step_log_data["action_batch"] = np.array(actions, dtype=np.int32)

        self.stats_recorder.record_step(step_log_data)
        # --- End Log Step Data ---

    def _train_batch(self):
        """Samples a batch, computes loss, updates agent, logs training stats."""
        if (
            len(self.buffer) < self.train_config.BATCH_SIZE
            or self.global_step < self.train_config.LEARN_START_STEP
        ):
            return

        beta = self._update_beta()
        is_n_step = self.buffer_config.USE_N_STEP and self.buffer_config.N_STEP > 1

        # Sample from buffer
        indices, is_weights_np, batch_np_tuple = None, None, None
        try:
            if self.buffer_config.USE_PER:
                sample_result = self.buffer.sample(self.train_config.BATCH_SIZE)
                if sample_result is None:
                    print("Warn: PER Buffer sample returned None.")
                    return
                batch_np_tuple, indices, is_weights_np = sample_result
            else:
                batch_np_tuple = self.buffer.sample(self.train_config.BATCH_SIZE)
                if batch_np_tuple is None:
                    print("Warn: Uniform Buffer sample returned None.")
                    return
        except Exception as e:
            print(f"ERROR sampling buffer: {e}")
            traceback.print_exc()
            return

        # Compute loss and get raw values for histograms
        raw_q_values = None
        loss = torch.tensor(0.0)  # Initialize loss
        td_errors = None
        try:
            # Get raw Q-values for logging before loss computation if needed
            if self.tb_config.LOG_HISTOGRAMS:
                with torch.no_grad():
                    tensor_batch_log = self.agent._np_batch_to_tensor(
                        batch_np_tuple, is_n_step
                    )
                    states_log = tensor_batch_log[0]
                    self.agent.online_net.eval()
                    raw_q_values = self.agent.online_net(states_log)
                    self.agent.online_net.train()  # Switch back

            # Compute actual loss
            loss, td_errors = self.agent.compute_loss(
                batch_np_tuple, is_n_step, is_weights_np
            )

        except Exception as e:
            print(f"ERROR computing loss: {e}")
            traceback.print_exc()
            return

        # Update agent
        grad_norm = None
        try:
            grad_norm = self.agent.update(loss)
        except Exception as e:
            print(f"ERROR updating agent: {e}")
            traceback.print_exc()
            return

        # Update priorities in PER buffer
        if self.buffer_config.USE_PER and indices is not None and td_errors is not None:
            try:
                td_errors_np = td_errors.squeeze().cpu().numpy()
                self.buffer.update_priorities(indices, td_errors_np)
            except Exception as e:
                print(f"ERROR updating priorities: {e}")

        # Record training step statistics (including histograms)
        train_log_data = {
            "loss": loss.item(),
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
            "avg_max_q": self.agent.get_last_avg_max_q(),  # Batch average
            # Beta already logged in _update_beta, no need to log again here
            "global_step": self.global_step,  # Associate with current step
        }
        if self.tb_config.LOG_HISTOGRAMS:
            if raw_q_values is not None:
                train_log_data["batch_q_values"] = raw_q_values  # Add raw Q values
            if td_errors is not None:
                train_log_data["batch_td_errors"] = td_errors  # Add raw TD errors (abs)

        self.stats_recorder.record_step(train_log_data)

    def step(self):
        """Performs one iteration: experience collection & potentially training."""
        step_start_time = time.time()

        # --- Experience Collection (logs step rewards/actions histograms) ---
        self._collect_experience()

        # --- Learning (logs loss, Q-value, TD-error histograms) ---
        if (
            self.global_step >= self.train_config.LEARN_START_STEP
            and
            # Learn roughly every LEARN_FREQ * NUM_ENVS steps
            (self.global_step // self.num_envs) % self.train_config.LEARN_FREQ == 0
        ):
            if len(self.buffer) >= self.train_config.BATCH_SIZE:
                self._train_batch()

        # --- Target Network Update ---
        target_freq = self.dqn_config.TARGET_UPDATE_FREQ
        if target_freq > 0:
            steps_before = self.global_step - self.num_envs
            if steps_before // target_freq < self.global_step // target_freq:
                if self.global_step > 0:
                    print(
                        f"[Trainer] Updating target network at step {self.global_step}"
                    )
                    self.agent.update_target_network()

        # --- Checkpointing ---
        self.maybe_save_checkpoint()

        # --- Log Step Time ---
        step_duration = time.time() - step_start_time
        self.stats_recorder.record_step(
            {"step_time": step_duration, "global_step": self.global_step}
        )

        # --- Optional Image Logging ---
        if (
            self.tb_config.LOG_IMAGES
            and self.global_step
            >= self.last_image_log_step + self.tb_config.IMAGE_LOG_FREQ
        ):
            try:
                env_to_log = random.randint(
                    0, self.num_envs - 1
                )  # Log a random env state
                img_array = self._get_env_image_as_numpy(env_to_log)
                if img_array is not None:
                    img_tensor = torch.from_numpy(img_array).permute(
                        2, 0, 1
                    )  # HWC to CHW for TB
                    self.stats_recorder.record_image(
                        f"Environment/Sample State Env {env_to_log}",
                        img_tensor,  # Pass tensor CHW
                        self.global_step,
                    )
                self.last_image_log_step = self.global_step
            except Exception as e:
                print(f"Error logging environment image: {e}")

    def _get_env_image_as_numpy(self, env_index: int) -> Optional[np.ndarray]:
        """Renders a single environment to a temporary surface and returns as numpy HWC."""
        if not (0 <= env_index < self.num_envs):
            return None
        env = self.envs[env_index]
        # Simple fixed size for the image
        img_h = 200
        img_w = int(img_h * (self.env_config.COLS * 0.75 + 0.25) / self.env_config.ROWS)
        try:
            temp_surf = pygame.Surface((img_w, img_h))
            # Simple rendering (modify if using UIRenderer._render_env for consistency)
            cell_w_px = img_w / (self.env_config.COLS * 0.75 + 0.25)
            cell_h_px = img_h / self.env_config.ROWS
            temp_surf.fill(self.vis_config.BLACK)  # Background

            # Render grid triangles
            for r in range(env.grid.rows):
                for c in range(env.grid.cols):
                    t = env.grid.triangles[r][c]
                    pts = t.get_points(ox=0, oy=0, cw=int(cell_w_px), ch=int(cell_h_px))
                    color = self.vis_config.GRAY
                    if t.is_death:
                        color = (10, 10, 10)
                    elif t.is_occupied:
                        color = t.color if t.color else self.vis_config.RED
                    pygame.draw.polygon(temp_surf, color, pts)
                    # pygame.draw.polygon(temp_surf, VisConfig.LIGHTG, pts, 1) # Outlines

            # Convert surface to numpy array [W, H, C] -> [H, W, C]
            img_array = pygame.surfarray.array3d(temp_surf)
            img_array = np.transpose(img_array, (1, 0, 2))  # W,H,C -> H,W,C
            return img_array
        except Exception as e:
            print(f"Error generating env image for TB: {e}")
            return None

    def maybe_save_checkpoint(self):
        """Saves checkpoint if frequency is met."""
        save_freq = self.train_config.CHECKPOINT_SAVE_FREQ
        if save_freq <= 0:
            return
        steps_before = self.global_step - self.num_envs
        if steps_before // save_freq < self.global_step // save_freq:
            if self.global_step > 0:
                self._save_checkpoint(is_final=False)

    def train_loop(self):
        """Main training loop."""
        print("[Trainer] Starting training loop...")
        try:
            while self.global_step < self.train_config.TOTAL_TRAINING_STEPS:
                self.step()
        except KeyboardInterrupt:
            print("\n[Trainer] Training loop interrupted by user.")
        except Exception as e:
            print(f"\n[Trainer] CRITICAL ERROR in training loop: {e}")
            traceback.print_exc()
        finally:
            print("[Trainer] Training loop finished or terminated.")
            self.cleanup(save_final=True)

    def cleanup(self, save_final: bool = True):
        """Cleans up resources: saves final state, flushes buffer, closes logger."""
        print("[Trainer] Cleaning up resources...")
        if hasattr(self.buffer, "flush_pending"):
            print("[Trainer] Flushing pending N-step transitions...")
            self.buffer.flush_pending()
        if save_final:
            print("[Trainer] Saving final checkpoint...")
            self._save_checkpoint(is_final=True)
        else:
            print("[Trainer] Skipping final save.")
        if hasattr(self.stats_recorder, "close"):
            self.stats_recorder.close()
        print("[Trainer] Cleanup complete.")
