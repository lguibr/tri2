# File: training/experience_collector.py
import time
import torch
import numpy as np
import random
import traceback
from typing import List, Dict, Any, Tuple

from config import EnvConfig, RewardConfig, TensorBoardConfig, DEVICE

# --- MODIFIED: Import GameState directly for type hinting ---
from environment.game_state import GameState, StateType

# --- END MODIFIED ---
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from stats.stats_recorder import StatsRecorderBase
from utils.types import ActionType


class ExperienceCollector:
    """
    Handles interaction with parallel environments to collect experience.
    Introduces environments sequentially to avoid synchronized starts.
    """

    def __init__(
        self,
        envs: List[GameState],  # Keep GameState type hint
        agent: DQNAgent,
        buffer: ReplayBufferBase,
        stats_recorder: StatsRecorderBase,
        env_config: EnvConfig,
        reward_config: RewardConfig,
        tb_config: TensorBoardConfig,
    ):
        self.envs = envs
        self.agent = agent
        self.buffer = buffer
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.env_config = env_config
        self.reward_config = reward_config
        self.tb_config = tb_config
        self.device = DEVICE

        # Initialize all states, but only a subset will be active initially
        self.current_states: List[StateType] = [env.reset() for env in self.envs]
        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_game_scores = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_lines_cleared = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_count = 0

        # --- NEW: Track active environments ---
        self.num_active_envs: int = 0
        print(
            f"[ExperienceCollector] Initialized. Staggering start for {self.num_envs} environments."
        )
        # --- END NEW ---

    def collect(self, current_global_step: int) -> int:
        """
        Collects one step of experience from each *active* environment.
        Activates one new environment per call until all are active.
        Returns the number of environment steps actually taken in this call.
        """
        # --- Activate one new environment if not all are active ---
        if self.num_active_envs < self.num_envs:
            self.num_active_envs += 1
            print(
                f"[ExperienceCollector] Activating environment {self.num_active_envs}/{self.num_envs} at global step ~{current_global_step}"
            )
        # --- END ---

        if self.num_active_envs == 0:
            return 0  # Should not happen if num_envs > 0

        # --- Only interact with the active subset ---
        num_to_process = self.num_active_envs
        actions, batch_chosen_slots, batch_shape_qs = self._select_actions(
            num_to_process
        )
        next_states, rewards, dones, step_rewards = self._step_environments(
            actions, num_to_process
        )
        self._store_transitions(actions, rewards, next_states, dones, num_to_process)
        self._handle_episode_ends(dones, current_global_step, num_to_process)
        self._update_current_states(next_states, num_to_process)
        self._log_step_stats(
            actions,
            step_rewards,
            batch_chosen_slots,
            batch_shape_qs,
            current_global_step,
            num_to_process,  # Pass the number processed
        )
        # --- Return the number of environments processed in this step ---
        return num_to_process

    def _select_actions(
        self, num_to_select: int
    ) -> Tuple[List[ActionType], np.ndarray, List[np.ndarray]]:
        """Selects actions for the first `num_to_select` environments."""
        actions = [-1] * num_to_select
        chosen_slots = np.full(num_to_select, -1, dtype=np.int32)
        shape_qs_list = []
        num_slots = self.env_config.NUM_SHAPE_SLOTS

        for i in range(num_to_select):  # Loop only over active envs
            state = self.current_states[i]
            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                actions[i] = 0  # Default action if none valid
                shape_qs_list.append(np.full(num_slots, -np.inf, dtype=np.float32))
            else:
                try:
                    actions[i] = self.agent.select_action(state, 0.0, valid_actions)
                    slot, shape_qs, _ = self.agent.get_last_shape_selection_info()
                    chosen_slots[i] = slot if slot is not None else -1
                    qs_to_log = (
                        shape_qs if shape_qs is not None else [-np.inf] * num_slots
                    )
                    # Ensure logged array has correct size
                    shape_qs_list.append(
                        np.array(qs_to_log[:num_slots], dtype=np.float32)
                    )
                except Exception as e:
                    print(f"ERROR: Agent select_action failed env {i}: {e}")
                    traceback.print_exc()
                    actions[i] = random.choice(valid_actions)
                    shape_qs_list.append(np.full(num_slots, -np.inf, dtype=np.float32))
        return actions, chosen_slots, shape_qs_list

    def _step_environments(
        self, actions: List[ActionType], num_to_step: int
    ) -> Tuple[List[StateType], np.ndarray, np.ndarray, np.ndarray]:
        """Steps the first `num_to_step` environments with the selected actions."""
        # Initialize results for the number of envs being stepped
        next_states = [{} for _ in range(num_to_step)]
        rewards = np.zeros(num_to_step, dtype=np.float32)
        dones = np.zeros(num_to_step, dtype=bool)
        step_rewards = np.zeros(num_to_step, dtype=np.float32)

        for i in range(num_to_step):  # Loop only over active envs
            env = self.envs[i]
            action = actions[i]
            try:
                reward, done = env.step(action)
                next_state = env.get_state()
                rewards[i] = reward
                dones[i] = done
                step_rewards[i] = reward
                next_states[i] = next_state
                # Update internal tracking for this env
                self.current_episode_scores[i] += reward
                self.current_episode_lengths[i] += 1
                if not done:
                    self.current_episode_game_scores[i] = env.game_score
                    self.current_episode_lines_cleared[i] = (
                        env.lines_cleared_this_episode
                    )
            except Exception as e:
                print(f"ERROR: Env {i} step failed (Action: {action}): {e}")
                traceback.print_exc()
                rewards[i] = self.reward_config.PENALTY_GAME_OVER
                dones[i] = True
                step_rewards[i] = rewards[i]
                next_states[i] = self._handle_env_crash(env, i)  # Reset and get state
                # Record episode immediately on crash
                self._record_episode_end(
                    i, rewards[i], current_global_step=0  # Step will be updated later
                )
                self._reset_episode_stats(i)

        return next_states, rewards, dones, step_rewards

    def _handle_env_crash(self, env: GameState, env_index: int) -> StateType:
        """Attempts to reset a crashed environment, returns a fallback state if needed."""
        print(f"Attempting reset for crashed env {env_index}...")
        try:
            return env.reset()
        except Exception as reset_e:
            print(
                f"FATAL: Env {env_index} failed reset after crash: {reset_e}. Creating zero state."
            )
            traceback.print_exc()
            grid_zeros = np.zeros(self.env_config.GRID_STATE_SHAPE, dtype=np.float32)
            shape_zeros = np.zeros(
                (
                    self.env_config.NUM_SHAPE_SLOTS,
                    self.env_config.SHAPE_FEATURES_PER_SHAPE,
                ),
                dtype=np.float32,
            )
            return {"grid": grid_zeros, "shapes": shape_zeros}

    def _store_transitions(
        self,
        actions: List[ActionType],
        rewards: np.ndarray,
        next_states: List[StateType],
        dones: np.ndarray,
        num_to_store: int,
    ):
        """Pushes transitions for the first `num_to_store` environments to the replay buffer."""
        for i in range(num_to_store):  # Loop only over active envs
            # Use the state *before* the step was taken
            state_before_step = self.current_states[i]
            self.buffer.push(
                state_before_step, actions[i], rewards[i], next_states[i], dones[i]
            )

    def _handle_episode_ends(
        self, dones: np.ndarray, current_global_step: int, num_to_check: int
    ):
        """Records stats for finished episodes among the first `num_to_check` environments."""
        for i in range(num_to_check):  # Loop only over active envs
            if dones[i]:
                # Pass the correct global step for this env's termination
                step_at_termination = current_global_step + i + 1
                self._record_episode_end(i, 0, step_at_termination)
                self.current_states[i] = self.envs[i].reset()  # Reset env state
                self._reset_episode_stats(i)

    def _record_episode_end(
        self, env_index: int, final_reward_adjustment: float, current_global_step: int
    ):
        """Records stats for a single finished episode."""
        self.episode_count += 1
        final_score = self.current_episode_scores[env_index] + final_reward_adjustment
        final_length = self.current_episode_lengths[env_index]
        final_game_score = self.current_episode_game_scores[env_index]
        final_lines = self.current_episode_lines_cleared[env_index]

        self.stats_recorder.record_episode(
            episode_score=final_score,
            episode_length=final_length,
            episode_num=self.episode_count,
            global_step=current_global_step,  # Use the passed step
            game_score=final_game_score,
            lines_cleared=final_lines,
        )

    def _reset_episode_stats(self, env_index: int):
        """Resets tracking stats for a specific environment index."""
        self.current_episode_scores[env_index] = 0.0
        self.current_episode_lengths[env_index] = 0
        self.current_episode_game_scores[env_index] = 0
        self.current_episode_lines_cleared[env_index] = 0

    def _update_current_states(self, next_states: List[StateType], num_to_update: int):
        """Updates the current states for the first `num_to_update` environments."""
        for i in range(num_to_update):  # Loop only over active envs
            self.current_states[i] = next_states[i]

    def _log_step_stats(
        self,
        actions: List[ActionType],
        step_rewards: np.ndarray,
        chosen_slots: np.ndarray,
        shape_qs_list: List[np.ndarray],
        current_global_step: int,
        num_envs_processed: int,  # Number of envs processed in this step
    ):
        """Logs statistics related to the collection step for the active environments."""
        step_log_data = {
            "buffer_size": len(self.buffer),
            # The global step *after* this collection step
            "global_step": current_global_step + num_envs_processed,
        }
        if self.tb_config.LOG_HISTOGRAMS:
            # Log data only from the processed environments
            step_log_data["step_rewards_batch"] = step_rewards[:num_envs_processed]
            step_log_data["action_batch"] = np.array(
                actions[:num_envs_processed], dtype=np.int32
            )

            valid_chosen_slots = chosen_slots[:num_envs_processed]
            valid_chosen_slots = valid_chosen_slots[valid_chosen_slots != -1]
            if len(valid_chosen_slots) > 0:
                step_log_data["chosen_shape_slot_batch"] = valid_chosen_slots

            if shape_qs_list:  # shape_qs_list already has length num_envs_processed
                flat_shape_qs = np.concatenate(shape_qs_list)
                valid_flat_shape_qs = flat_shape_qs[np.isfinite(flat_shape_qs)]
                if len(valid_flat_shape_qs) > 0:
                    step_log_data["shape_slot_max_q_batch"] = valid_flat_shape_qs

        if self.tb_config.LOG_SHAPE_PLACEMENT_Q_VALUES:
            # Agent's last batch Q values correspond to the last call to select_action,
            # which processed num_envs_processed environments.
            batch_q_vals = self.agent.get_last_batch_q_values_for_actions()
            if batch_q_vals is not None and len(batch_q_vals) == num_envs_processed:
                step_log_data["batch_q_values_actions_taken"] = batch_q_vals
            # else: Handle potential mismatch if needed

        self.stats_recorder.record_step(step_log_data)

    def get_episode_count(self) -> int:
        return self.episode_count
