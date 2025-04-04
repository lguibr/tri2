# File: training/trainer.py
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

        # Store configs
        self.env_config = env_config
        self.dqn_config = dqn_config
        self.train_config = train_config
        self.buffer_config = buffer_config
        self.exploration_config = exploration_config
        self.model_config = model_config

        # Training state
        self.global_step = 0
        self.episode_count = 0
        try:
            # Initialize states by resetting all environments
            self.current_states = [ensure_numpy(env.reset()) for env in self.envs]
        except Exception as e:
            print(f"FATAL ERROR during initial environment reset: {e}")
            raise e  # Propagate error to stop execution

        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

        self._load_checkpoint()  # Load agent state first
        self._load_buffer_state()  # Then load buffer state

        # Ensure epsilon and beta are set correctly based on loaded global_step
        self._update_epsilon()
        self._update_beta()
        # Pass initial buffer size to stats recorder
        self.stats_recorder.record_step({"buffer_size": len(self.buffer)})

        print(
            f"[Trainer] Initialization complete. Starting from global_step={self.global_step}, "
            f"Episode: {self.episode_count}, Buffer size: {len(self.buffer)}, "
            f"Epsilon: {self._update_epsilon():.4f}, Beta: {self._update_beta():.4f}"
        )

    def _load_checkpoint(self):
        """Loads agent and trainer state from checkpoint."""
        if not self.model_config.LOAD_MODEL:
            print("[Trainer] LOAD_MODEL is False. Starting fresh agent.")
            return

        save_path = self.model_config.SAVE_PATH
        if os.path.isfile(save_path):
            print(f"[Trainer] Loading agent checkpoint from: {save_path}")
            try:
                # Load checkpoint onto the correct device
                checkpoint = torch.load(save_path, map_location=self.device)

                # Load agent state (model weights, optimizer state)
                self.agent.load_state_dict(checkpoint["agent_state_dict"])

                # Load trainer state (global step, episode count)
                self.global_step = checkpoint.get("global_step", 0)
                self.episode_count = checkpoint.get("episode_count", 0)

                print(
                    f"[Trainer] Agent checkpoint loaded successfully. Resuming from step {self.global_step}, episode {self.episode_count}"
                )
                # Epsilon/Beta will be updated based on loaded step after this function returns

            except FileNotFoundError:
                # This case should be caught by os.path.isfile, but handle defensively
                print(
                    f"[Trainer] Agent checkpoint file disappeared before loading? ({save_path}). Starting fresh."
                )
                self._reset_trainer_state()
            except KeyError as e:
                print(
                    f"[Trainer] Error: Checkpoint missing key '{e}'. Checkpoint structure may be incompatible. Starting fresh."
                )
                self._reset_trainer_state()
            except Exception as e:
                print(
                    f"[Trainer] CRITICAL ERROR loading agent checkpoint: {e}. Starting fresh."
                )
                self._reset_trainer_state()
        else:
            print(
                f"[Trainer] No agent checkpoint found at {save_path}. Starting fresh agent."
            )
            self._reset_trainer_state()  # Ensure state is reset if no file found

    def _reset_trainer_state(self):
        """Resets global step and episode count."""
        self.global_step = 0
        self.episode_count = 0

    def _load_buffer_state(self):
        """Loads replay buffer state if configured and file exists."""
        if not self.buffer_config.LOAD_BUFFER:
            print("[Trainer] LOAD_BUFFER is False. Not loading buffer state.")
            return

        buffer_path = BUFFER_SAVE_PATH
        if os.path.isfile(buffer_path):
            print(f"[Trainer] Attempting to load buffer state from: {buffer_path}")
            try:
                # Delegate loading to the buffer instance itself
                if hasattr(self.buffer, "load_state"):
                    self.buffer.load_state(buffer_path)
                    print(
                        f"[Trainer] Replay buffer state loaded successfully. Size: {len(self.buffer)}"
                    )
                else:
                    # This should not happen if using provided buffer classes
                    print(
                        "[Trainer] Warning: Buffer object does not have a 'load_state' method. Cannot load buffer."
                    )

            except FileNotFoundError:
                # This case should be caught by os.path.isfile, but handle defensively
                print(
                    f"[Trainer] Buffer state file disappeared before loading? ({buffer_path}). Starting empty buffer."
                )
            except Exception as e:
                # Catch broader exceptions during loading (e.g., pickle errors, corrupted file)
                print(
                    f"[Trainer] CRITICAL ERROR loading buffer state: {e}. Starting with empty buffer."
                )
                # Consider clearing the buffer explicitly if a load fails partially
                if hasattr(self.buffer, "tree") and hasattr(
                    self.buffer.tree, "clear"
                ):  # Example for PER
                    self.buffer.tree.clear()
                elif hasattr(self.buffer, "buffer") and hasattr(
                    self.buffer.buffer, "clear"
                ):  # Example for Uniform
                    self.buffer.buffer.clear()

        else:
            print(
                f"[Trainer] No buffer state file found at {buffer_path}. Starting with empty buffer."
            )

    def _save_checkpoint(self, is_final=False):
        """Saves the agent/trainer state and the buffer state separately."""
        prefix = "FINAL" if is_final else f"step_{self.global_step}"

        # --- Save Agent/Trainer State ---
        agent_save_path = self.model_config.SAVE_PATH
        print(f"[Trainer] Saving agent checkpoint ({prefix}) to: {agent_save_path}")
        try:
            # Ensure checkpoint directory exists
            os.makedirs(os.path.dirname(agent_save_path), exist_ok=True)
            # Gather state dictionaries
            save_data = {
                "global_step": self.global_step,
                "episode_count": self.episode_count,
                "agent_state_dict": self.agent.get_state_dict(),
                # Add model/env/training config dicts? Useful for reproducibility.
                # "model_config": vars(self.model_config), # Example
            }
            # Save using torch.save
            torch.save(save_data, agent_save_path)
            print(f"[Trainer] Agent checkpoint ({prefix}) saved successfully.")
        except Exception as e:
            print(f"[Trainer] ERROR saving agent checkpoint ({prefix}): {e}")

        # --- Save Buffer State ---
        # Avoid saving buffer frequently if it's huge and saving is slow
        # Only save periodically or finally. Checkpoint freq check handles periodic.
        buffer_save_path = BUFFER_SAVE_PATH
        print(
            f"[Trainer] Saving buffer state ({prefix}) to: {buffer_save_path} (Size: {len(self.buffer)})"
        )
        try:
            # Ensure checkpoint directory exists (might be different from agent)
            os.makedirs(os.path.dirname(buffer_save_path), exist_ok=True)
            if hasattr(self.buffer, "save_state"):
                self.buffer.save_state(buffer_save_path)
                print(f"[Trainer] Buffer state ({prefix}) saved successfully.")
            else:
                print("[Trainer] Warning: Buffer does not support save_state method.")
        except Exception as e:
            print(f"[Trainer] ERROR saving buffer state ({prefix}): {e}")

    def _update_epsilon(self) -> float:
        """Calculates epsilon based on global step and decay schedule."""
        start = self.exploration_config.EPS_START
        end = self.exploration_config.EPS_END
        decay_frames = self.exploration_config.EPS_DECAY_FRAMES

        if decay_frames <= 0:  # Avoid division by zero if decay is disabled
            epsilon = end
        else:
            # Linear decay
            fraction = min(1.0, float(self.global_step) / decay_frames)
            epsilon = start + fraction * (end - start)

        # Pass current epsilon to stats recorder
        self.stats_recorder.record_step({"epsilon": epsilon})
        return epsilon

    def _update_beta(self) -> float:
        """Calculates PER beta based on global step and annealing schedule."""
        if not self.buffer_config.USE_PER:
            # Beta is irrelevant if not using PER
            self.stats_recorder.record_step({"beta": 1.0})  # Log default value
            return 1.0

        start = self.buffer_config.PER_BETA_START
        end = 1.0  # Beta typically anneals to 1.0
        anneal_frames = self.buffer_config.PER_BETA_FRAMES

        if anneal_frames <= 0:  # Avoid division by zero
            beta = end
        else:
            # Linear annealing
            fraction = min(1.0, float(self.global_step) / anneal_frames)
            beta = start + fraction * (end - start)

        # Set beta in the buffer (important for IS weight calculation)
        self.buffer.set_beta(beta)
        # Pass current beta to stats recorder
        self.stats_recorder.record_step({"beta": beta})
        return beta

    def _collect_experience(self):
        """Performs one step in each parallel environment and stores transitions."""
        epsilon = self._update_epsilon()  # Get current epsilon for action selection

        # --- Select Actions for all environments ---
        actions: List[ActionType] = [
            -1
        ] * self.num_envs  # Initialize with invalid action
        valid_action_lists: List[List[ActionType]] = [[] for _ in range(self.num_envs)]
        needs_final_push_on_invalid: List[bool] = [False] * self.num_envs

        for i in range(self.num_envs):
            # Skip action selection if env is already done (should have been reset)
            if self.envs[i].is_over():
                # This state indicates an env didn't reset properly after last step. Log warning.
                # print(f"Warning: Environment {i} was already 'done' at start of _collect_experience.")
                # Force reset and get state again.
                try:
                    self.current_states[i] = ensure_numpy(self.envs[i].reset())
                    self.current_episode_scores[i] = 0.0
                    self.current_episode_lengths[i] = 0
                except Exception as e:
                    print(f"ERROR: Environment {i} failed reset during collect: {e}")
                    # Handle failure? Mark env as unusable? For now, proceed.

            # Get valid actions for the current state
            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                # If no valid actions, game might be over or stuck.
                # The env.step() should handle game over logic.
                # We still need to select a dummy action (e.g., 0) to proceed.
                # Mark this env to potentially push a final transition if needed.
                needs_final_push_on_invalid[i] = True
                valid_action_lists[i] = [0]  # Provide dummy list for agent call
                actions[i] = 0  # Select dummy action
            else:
                # Select action using the agent's policy
                valid_action_lists[i] = valid_actions
                try:
                    actions[i] = self.agent.select_action(
                        self.current_states[i], epsilon, valid_action_lists[i]
                    )
                except Exception as e:
                    print(f"ERROR: Agent failed select_action for env {i}: {e}")
                    # Choose random valid action as fallback
                    actions[i] = random.choice(valid_actions) if valid_actions else 0

        # --- Step Environments and Store Transitions ---
        for i in range(self.num_envs):
            env = self.envs[i]
            current_state = self.current_states[i]
            action = actions[i]

            # If we selected a dummy action because no valid moves were found initially
            if needs_final_push_on_invalid[i]:
                # Simulate a step resulting in game over immediately
                reward = (
                    env.rewards.PENALTY_GAME_OVER
                )  # Apply game over penalty directly
                done = True
                next_state = (
                    current_state  # Next state is same as current terminal state
                )
                env.game_over = True  # Ensure env state reflects this
            else:
                # Perform the chosen action in the environment
                try:
                    reward, done = env.step(action)
                    # Get the resulting next state only if not done (or if needed for buffer)
                    # If done, the 'next_state' might be terminal representation or reset state.
                    # Get state *before* potential reset for buffer consistency.
                    next_state = ensure_numpy(env.get_state())
                except Exception as e:
                    print(f"ERROR: Environment {i} step failed (Action: {action}): {e}")
                    # Penalize and mark as done on error
                    reward = env.rewards.PENALTY_GAME_OVER
                    done = True
                    next_state = current_state  # Use current state as next on error
                    env.game_over = True  # Mark env as done

            # --- Store Transition in Replay Buffer ---
            # NStepBufferWrapper handles N-step logic internally via its push method
            try:
                self.buffer.push(current_state, action, reward, next_state, done)
            except Exception as e:
                print(f"ERROR: Buffer push failed for env {i}: {e}")

            # --- Update Trackers ---
            self.current_episode_scores[i] += reward
            self.current_episode_lengths[i] += 1
            # Record step-level reward for averaging
            self.stats_recorder.record_step({"step_reward": reward})

            # --- Handle Episode End ---
            if done:
                self.episode_count += 1
                score = self.current_episode_scores[i]
                length = self.current_episode_lengths[i]
                # Record episode stats
                self.stats_recorder.record_episode(
                    episode_score=score,
                    episode_length=length,
                    episode_num=self.episode_count,
                    global_step=self.global_step
                    + self.num_envs,  # Step count *after* this batch
                )
                # Reset the environment and associated trackers
                try:
                    self.current_states[i] = ensure_numpy(env.reset())
                except Exception as e:
                    print(f"ERROR: Environment {i} reset failed after episode end: {e}")
                    # What to do here? Env might be broken. Re-init? Skip?
                self.current_episode_scores[i] = 0.0
                self.current_episode_lengths[i] = 0
            else:
                # If not done, update the current state for the next iteration
                self.current_states[i] = next_state

        # Increment global step count after processing all environments
        self.global_step += self.num_envs
        # Update buffer size in stats recorder after pushing potentially many transitions
        self.stats_recorder.record_step({"buffer_size": len(self.buffer)})

    def _train_batch(self):
        """Samples a batch, computes loss, updates agent, and updates priorities."""
        # Ensure buffer has enough samples to form a batch
        if len(self.buffer) < self.train_config.BATCH_SIZE:
            # print(f"Skipping training: Buffer size {len(self.buffer)} < Batch size {self.train_config.BATCH_SIZE}")
            return

        # Update PER beta value right before sampling
        beta = self._update_beta()

        # Determine if N-step is used based on config (buffer handles internal format)
        is_n_step = self.buffer_config.USE_N_STEP and self.buffer_config.N_STEP > 1

        # --- Sample Batch ---
        indices = None
        is_weights_np = None
        batch_np_tuple = None
        try:
            if self.buffer_config.USE_PER:
                # Sample with priorities, returns ((s,a,r,...), indices, weights)
                sample_result: Optional[
                    Union[PrioritizedNumpyBatch, PrioritizedNumpyNStepBatch]
                ] = self.buffer.sample(self.train_config.BATCH_SIZE)
                if sample_result is None:
                    return  # Buffer might be temporarily unable to sample
                batch_np_tuple, indices, is_weights_np = sample_result
            else:
                # Sample uniformly, returns (s,a,r,...)
                batch_np_tuple: Optional[Union[NumpyBatch, NumpyNStepBatch]] = (
                    self.buffer.sample(self.train_config.BATCH_SIZE)
                )
                if batch_np_tuple is None:
                    return  # Buffer might be temporarily unable to sample

            if batch_np_tuple is None:  # Final check
                print("Warning: Buffer sample returned None. Skipping training step.")
                return

        except Exception as e:
            print(f"ERROR during buffer sampling: {e}")
            return  # Skip training step on sampling error

        # --- Compute Loss and TD Errors ---
        try:
            loss, td_errors = self.agent.compute_loss(
                batch=batch_np_tuple,
                is_n_step=is_n_step,
                is_weights=is_weights_np,  # Pass IS weights if PER is used
            )
        except Exception as e:
            print(f"ERROR during agent.compute_loss: {e}")
            # Potentially log the batch data that caused the error
            return  # Skip training step on loss computation error

        # --- Update Agent Network ---
        try:
            grad_norm = self.agent.update(loss)  # Perform backprop and optimizer step
        except Exception as e:
            print(f"ERROR during agent.update: {e}")
            return  # Skip training step if update fails

        # --- Update Priorities (for PER) ---
        if self.buffer_config.USE_PER and indices is not None and td_errors is not None:
            try:
                # Ensure td_errors is numpy array on CPU for buffer update
                td_errors_np = td_errors.squeeze(1).cpu().numpy()
                self.buffer.update_priorities(indices, td_errors_np)
            except Exception as e:
                print(f"ERROR during buffer.update_priorities: {e}")

        # --- Record Training Statistics ---
        # Fetch avg_max_q calculated during compute_loss
        avg_max_q = self.agent.get_last_avg_max_q()
        self.stats_recorder.record_step(
            {
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "avg_max_q": avg_max_q,
                # Epsilon/Beta/BufferSize are updated elsewhere
            }
        )

    def step(self):
        """Performs one logical step: collect experience, maybe train, maybe update target, maybe save."""
        step_start_time = time.time()

        # 1. Collect experience from all environments
        self._collect_experience()

        # 2. Train the agent network(s) based on frequency
        # Check if ready to start learning and if enough steps have passed since last learn step
        if (
            self.global_step >= self.train_config.LEARN_START_STEP
            and (self.global_step // self.num_envs) % self.train_config.LEARN_FREQ == 0
        ):
            # Check buffer size *again* right before training attempt
            if len(self.buffer) >= self.train_config.BATCH_SIZE:
                self._train_batch()  # Perform one training update

        # 3. Update the target network periodically
        # Check if the target update interval boundary has been crossed since the last step
        steps_before_collect = (
            self.global_step - self.num_envs
        )  # Step count *before* _collect_experience
        # Use integer division to check if the interval marker changed
        if (
            steps_before_collect // self.dqn_config.TARGET_UPDATE_FREQ
            < self.global_step // self.dqn_config.TARGET_UPDATE_FREQ
        ):
            if self.global_step > 0:  # Avoid update at step 0
                print(f"[Trainer] Updating target network at step {self.global_step}")
                self.agent.update_target_network()

        # 4. Save checkpoint periodically
        self.maybe_save_checkpoint()

        # 5. Update epsilon/beta (done within collect/train)
        # 6. Log summary (handled by caller or main loop based on StatsConfig interval)
        step_duration = time.time() - step_start_time
        # Note: SPS is calculated within StatsRecorder based on logging intervals

    def maybe_save_checkpoint(self):
        """Saves checkpoint based on frequency."""
        save_freq = self.train_config.CHECKPOINT_SAVE_FREQ
        if save_freq <= 0:  # Periodic saving disabled
            return

        # Check if the save frequency interval boundary has been crossed
        steps_before_collect = self.global_step - self.num_envs
        if steps_before_collect // save_freq < self.global_step // save_freq:
            if self.global_step > 0:  # Avoid saving at step 0
                self._save_checkpoint(is_final=False)

    def train(self):
        """Main training loop (can be run standalone or controlled step-by-step)."""
        print("[Trainer] Starting training loop...")
        try:
            # Initial epsilon/beta/buffer size already set in __init__
            while self.global_step < self.train_config.TOTAL_TRAINING_STEPS:
                self.step()  # Perform collection, training, updates, saving

                # Log summary statistics based on interval defined in StatsConfig
                # Delegate logging decision to the stats_recorder itself
                self.stats_recorder.log_summary(self.global_step)

                # Optional: Add a small sleep if CPU usage is too high in standalone mode
                # time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[Trainer] Training loop interrupted by user (KeyboardInterrupt).")
        except Exception as e:
            print(f"\n[Trainer] CRITICAL ERROR encountered during training loop: {e}")
            import traceback

            traceback.print_exc()  # Print detailed traceback
        finally:
            print("[Trainer] Training loop finished or terminated.")
            self.cleanup()  # Ensure cleanup runs

    def cleanup(self):
        """Clean up resources: flush buffer, save final state, close logger."""
        print("[Trainer] Cleaning up resources...")
        # 1. Flush any pending transitions in the N-step buffer wrapper
        if hasattr(self.buffer, "flush_pending"):
            print("[Trainer] Flushing pending N-step transitions...")
            try:
                self.buffer.flush_pending()
            except Exception as e:
                print(f"ERROR during buffer flush: {e}")

        # 2. Save final agent and buffer state
        print("[Trainer] Saving final checkpoint...")
        try:
            self._save_checkpoint(is_final=True)
        except Exception as e:
            print(f"ERROR during final save: {e}")

        # 3. Close the statistics recorder (e.g., close DB connection)
        if hasattr(self.stats_recorder, "close"):
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"ERROR closing stats recorder: {e}")

        print("[Trainer] Cleanup complete.")
