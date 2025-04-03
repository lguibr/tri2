import time
import torch
import numpy as np
import os
import pickle
from typing import List, Optional, Union
from config import (
    EnvConfig,
    DQNConfig,
    TrainConfig,
    BufferConfig,
    ExplorationConfig,
    ModelConfig,
    DEVICE,
    BUFFER_SAVE_PATH,
)
from environment.game_state import GameState
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from stats.stats_recorder import StatsRecorderBase
from utils.helpers import ensure_numpy
from utils.types import (
    ActionType,
    PrioritizedNumpyBatch,
    PrioritizedNumpyNStepBatch,
    NumpyBatch,
    NumpyNStepBatch,
)


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
        exploration_config: ExplorationConfig,
        model_config: ModelConfig,
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
        self.exploration_config = exploration_config
        self.model_config = model_config

        self.global_step = 0
        self.episode_count = 0
        self.current_states = [ensure_numpy(env.reset()) for env in self.envs]
        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

        self._load_checkpoint()
        self._load_buffer_state()

        print(
            f"[Trainer] Initialization complete. Starting from global_step={self.global_step}, "
            f"Buffer size: {len(self.buffer)}"
        )

    def _load_checkpoint(self):
        """Loads agent and trainer state from checkpoint."""
        if not self.model_config.LOAD_MODEL:
            print("[Trainer] LOAD_MODEL is False. Starting fresh.")
            return

        save_path = self.model_config.SAVE_PATH
        if os.path.isfile(save_path):
            print(f"[Trainer] Loading agent checkpoint from: {save_path}")
            try:
                checkpoint = torch.load(save_path, map_location=self.device)
                self.agent.load_state_dict(checkpoint["agent_state_dict"])
                self.global_step = checkpoint.get("global_step", 0)
                self.episode_count = checkpoint.get("episode_count", 0)
                print(
                    f"[Trainer] Agent checkpoint loaded. Resuming from step {self.global_step}, episode {self.episode_count}"
                )
                # Recalculate epsilon/beta based on loaded step
                self._update_epsilon()
                self._update_beta()
            except FileNotFoundError:
                print(
                    f"[Trainer] Agent checkpoint file not found at {save_path}. Starting fresh."
                )
                self.global_step = 0
                self.episode_count = 0
            except KeyError as e:
                print(
                    f"[Trainer] Error: Missing key {e} in checkpoint. Starting fresh."
                )
                self.global_step = 0
                self.episode_count = 0
                # Consider re-initializing agent if load fails? For now, assume initial state is OK.
            except Exception as e:
                print(
                    f"[Trainer] Failed to load agent checkpoint: {e}. Starting fresh."
                )
                self.global_step = 0
                self.episode_count = 0
        else:
            print(
                f"[Trainer] No agent checkpoint found at {save_path}. Starting fresh."
            )

    def _load_buffer_state(self):
        """Loads replay buffer state if configured and file exists."""
        if not self.buffer_config.LOAD_BUFFER:
            print("[Trainer] LOAD_BUFFER is False. Not loading buffer state.")
            return

        buffer_path = BUFFER_SAVE_PATH
        if os.path.isfile(buffer_path):
            print(f"[Trainer] Attempting to load buffer state from: {buffer_path}")
            try:
                if hasattr(self.buffer, "load_state"):
                    self.buffer.load_state(buffer_path)
                    print(
                        f"[Trainer] Replay buffer state loaded. Size: {len(self.buffer)}"
                    )
                else:
                    print(
                        "[Trainer] Warning: Buffer does not support load_state method."
                    )
            except FileNotFoundError:
                print(f"[Trainer] Buffer state file not found at {buffer_path}.")
            except Exception as e:
                print(
                    f"[Trainer] Failed to load buffer state: {e}. Starting with empty buffer."
                )
                # If loading fails, potentially clear the buffer? Assume buffer init is empty.
                # self.buffer.clear() # If a clear method exists
        else:
            print(
                f"[Trainer] No buffer state file found at {buffer_path}. Starting with empty buffer."
            )

    def _save_checkpoint(self):
        """Saves the agent/trainer state and the buffer state separately."""
        # Save Agent/Trainer State
        agent_save_path = self.model_config.SAVE_PATH
        print(
            f"[Trainer] Saving agent checkpoint to: {agent_save_path} at step {self.global_step}"
        )
        try:
            os.makedirs(os.path.dirname(agent_save_path), exist_ok=True)
            save_data = {
                "global_step": self.global_step,
                "episode_count": self.episode_count,
                "agent_state_dict": self.agent.get_state_dict(),
            }
            torch.save(save_data, agent_save_path)
            print("[Trainer] Agent checkpoint saved successfully.")
        except Exception as e:
            print(f"[Trainer] Error saving agent checkpoint: {e}")

        # Save Buffer State
        buffer_save_path = BUFFER_SAVE_PATH
        print(
            f"[Trainer] Saving buffer state to: {buffer_save_path} (Size: {len(self.buffer)})"
        )
        try:
            if hasattr(self.buffer, "save_state"):
                self.buffer.save_state(buffer_save_path)
                print("[Trainer] Buffer state saved successfully.")
            else:
                print("[Trainer] Warning: Buffer does not support save_state method.")
        except Exception as e:
            print(f"[Trainer] Error saving buffer state: {e}")

    def _update_epsilon(self) -> float:
        start = self.exploration_config.EPS_START
        end = self.exploration_config.EPS_END
        decay_frames = self.exploration_config.EPS_DECAY_FRAMES
        if decay_frames <= 0:
            return end
        fraction = min(1.0, float(self.global_step) / decay_frames)
        epsilon = start + fraction * (end - start)
        # Store current epsilon for stats recorder
        self.stats_recorder.record_step({"epsilon": epsilon})
        return epsilon

    def _update_beta(self) -> float:
        if not self.buffer_config.USE_PER:
            return 1.0
        start = self.buffer_config.PER_BETA_START
        end = 1.0
        anneal_frames = self.buffer_config.PER_BETA_FRAMES
        if anneal_frames <= 0:
            beta = end
        else:
            fraction = min(1.0, float(self.global_step) / anneal_frames)
            beta = start + fraction * (end - start)
        self.buffer.set_beta(beta)
        # Store current beta for stats recorder
        self.stats_recorder.record_step({"beta": beta})
        return beta

    def _collect_experience(self):
        epsilon = self._update_epsilon()
        actions: List[ActionType] = [-1] * self.num_envs
        valid_action_lists: List[List[ActionType]] = [[] for _ in range(self.num_envs)]
        needs_final_push: List[bool] = [False] * self.num_envs

        for i in range(self.num_envs):
            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                needs_final_push[i] = True
                valid_action_lists[i] = [0]  # Dummy
                actions[i] = 0  # Dummy
            else:
                valid_action_lists[i] = valid_actions
                actions[i] = self.agent.select_action(
                    self.current_states[i], epsilon, valid_action_lists[i]
                )

        for i in range(self.num_envs):
            env = self.envs[i]
            current_state = self.current_states[i]
            action = actions[i]

            if needs_final_push[i]:
                reward = 0.0
                done = True
                next_state = current_state
            else:
                try:
                    reward, done = env.step(action)
                    next_state = ensure_numpy(env.get_state())
                except Exception as e:
                    print(f"ERROR: Env {i} step failed (Action: {action}): {e}")
                    reward = -1.0
                    done = True
                    next_state = current_state

            # Buffer handles N-step logic internally via its push method
            self.buffer.push(current_state, action, reward, next_state, done)

            self.current_episode_scores[i] += reward
            self.current_episode_lengths[i] += 1
            self.stats_recorder.record_step({"step_reward": reward})

            if done:
                self.episode_count += 1
                score = self.current_episode_scores[i]
                length = self.current_episode_lengths[i]
                self.stats_recorder.record_episode(
                    episode_score=score,
                    episode_length=length,
                    episode_num=self.episode_count,
                    global_step=self.global_step
                    + self.num_envs,  # Step count *after* this batch
                )
                try:
                    self.current_states[i] = ensure_numpy(env.reset())
                except Exception as e:
                    print(f"ERROR: Environment {i} reset failed: {e}")
                self.current_episode_scores[i] = 0.0
                self.current_episode_lengths[i] = 0
            else:
                self.current_states[i] = next_state

        self.global_step += self.num_envs
        # Update buffer size in stats recorder after pushing
        self.stats_recorder.record_step({"buffer_size": len(self.buffer)})

    def _train_batch(self):
        if len(self.buffer) < self.train_config.BATCH_SIZE:
            return

        beta = self._update_beta()  # Update beta right before sampling

        is_n_step = self.buffer_config.USE_N_STEP and self.buffer_config.N_STEP > 1
        indices = None
        is_weights_np = None

        if self.buffer_config.USE_PER:
            sample_result: Optional[
                Union[PrioritizedNumpyBatch, PrioritizedNumpyNStepBatch]
            ] = self.buffer.sample(self.train_config.BATCH_SIZE)
            if sample_result is None:
                return
            batch_np_tuple, indices, is_weights_np = sample_result
        else:
            batch_np_tuple: Optional[Union[NumpyBatch, NumpyNStepBatch]] = (
                self.buffer.sample(self.train_config.BATCH_SIZE)
            )
            if batch_np_tuple is None:
                return

        # Agent computes loss using the numpy batch directly
        loss, td_errors = self.agent.compute_loss(
            batch=batch_np_tuple,
            is_n_step=is_n_step,
            is_weights=is_weights_np,
        )

        grad_norm = self.agent.update(loss)

        if self.buffer_config.USE_PER and indices is not None:
            self.buffer.update_priorities(indices, td_errors.squeeze(1).cpu().numpy())

        # Record training stats (beta already updated and recorded)
        self.stats_recorder.record_step({"loss": loss.item(), "grad_norm": grad_norm})

    def step(self):
        """Performs one logical step: collect experience, maybe train, maybe update target."""

        # 1. Collect experience (advances global_step)
        self._collect_experience()

        # 2. Train based on frequency
        if (
            self.global_step >= self.train_config.LEARN_START_STEP
            and (self.global_step // self.num_envs) % self.train_config.LEARN_FREQ == 0
        ):
            # Train potentially multiple times if many steps collected? No, standard is once per LEARN_FREQ env steps.
            if len(self.buffer) >= self.train_config.BATCH_SIZE:
                self._train_batch()

        # 3. Update target network periodically
        # Check if target update interval boundary crossed
        # Compare steps before and after collecting experience
        steps_before = self.global_step - self.num_envs
        if (
            steps_before // self.dqn_config.TARGET_UPDATE_FREQ
            < self.global_step // self.dqn_config.TARGET_UPDATE_FREQ
        ):
            if self.global_step > 0:
                print(f"[Trainer] Updating target network at step {self.global_step}")
                self.agent.update_target_network()

        # 4. Maybe save checkpoint
        self.maybe_save_checkpoint()

        # 5. Update epsilon/beta (done within collect/train) and log summary (done by caller/stats_recorder)

    def maybe_save_checkpoint(self):
        """Saves checkpoint based on frequency."""
        save_freq = self.train_config.CHECKPOINT_SAVE_FREQ
        if save_freq <= 0:  # Saving disabled
            return

        # Check if save frequency boundary crossed
        steps_before = (
            self.global_step - self.num_envs
        )  # Steps before the last collection
        if steps_before // save_freq < self.global_step // save_freq:
            if self.global_step > 0:  # Avoid saving at step 0
                self._save_checkpoint()

    def train(self):
        """Main training loop (can be run standalone or controlled step-by-step)."""
        print("[Trainer] Starting training loop...")
        try:
            self._update_beta()  # Initial beta set
            while self.global_step < self.train_config.TOTAL_TRAINING_STEPS:
                self.step()  # Use the combined step logic
                self.stats_recorder.log_summary(self.global_step)  # Log periodically
                # Optional: Add a small sleep if CPU usage is too high in standalone mode
                # time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[Trainer] Training loop interrupted by user.")
        finally:
            print("[Trainer] Training loop finished.")
            self.cleanup()

    def cleanup(self):
        """Clean up resources: flush buffer, save final state, close logger."""
        print("[Trainer] Cleaning up...")
        if hasattr(self.buffer, "flush_pending"):
            print("[Trainer] Flushing pending N-step transitions...")
            self.buffer.flush_pending()
        self._save_checkpoint()
        if hasattr(self.stats_recorder, "close"):
            self.stats_recorder.close()
        print("[Trainer] Cleanup complete.")
