# File: training/experience_collector.py
import time
import torch
import numpy as np
import random
import traceback
from typing import List, Dict, Any, Tuple

from config import EnvConfig, RewardConfig, TensorBoardConfig, DEVICE
from environment.game_state import GameState, StateType
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from stats.stats_recorder import StatsRecorderBase
from utils.types import ActionType


class ExperienceCollector:
    """Handles interaction with parallel environments to collect experience."""

    def __init__(
        self,
        envs: List[GameState],
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

        self.current_states: List[StateType] = [env.reset() for env in self.envs]
        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_game_scores = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_lines_cleared = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_count = 0  # Managed internally, but Trainer reads it

    def collect(self, current_global_step: int) -> int:
        """Collects one step of experience from each environment."""
        actions, batch_chosen_slots, batch_shape_qs = self._select_actions()
        next_states, rewards, dones, step_rewards = self._step_environments(actions)
        self._store_transitions(actions, rewards, next_states, dones)
        self._handle_episode_ends(dones, current_global_step)
        self._update_current_states(next_states)
        self._log_step_stats(
            actions,
            step_rewards,
            batch_chosen_slots,
            batch_shape_qs,
            current_global_step,
        )
        return self.num_envs  # Return number of steps collected

    def _select_actions(self) -> Tuple[List[ActionType], np.ndarray, List[np.ndarray]]:
        """Selects actions for all environments using the agent."""
        actions = [-1] * self.num_envs
        chosen_slots = np.full(self.num_envs, -1, dtype=np.int32)
        shape_qs_list = []
        num_slots = self.env_config.NUM_SHAPE_SLOTS

        for i in range(self.num_envs):
            state = self.current_states[i]
            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                actions[i] = 0
                shape_qs_list.append(np.full(num_slots, -np.inf, dtype=np.float32))
            else:
                try:
                    actions[i] = self.agent.select_action(state, 0.0, valid_actions)
                    slot, shape_qs, _ = self.agent.get_last_shape_selection_info()
                    chosen_slots[i] = slot if slot is not None else -1
                    qs_to_log = (
                        shape_qs if shape_qs is not None else [-np.inf] * num_slots
                    )
                    shape_qs_list.append(
                        np.array(qs_to_log[:num_slots], dtype=np.float32)
                    )
                except Exception as e:
                    print(f"ERROR: Agent select_action failed env {i}: {e}")
                    actions[i] = random.choice(valid_actions)
                    shape_qs_list.append(np.full(num_slots, -np.inf, dtype=np.float32))
        return actions, chosen_slots, shape_qs_list

    def _step_environments(
        self, actions: List[ActionType]
    ) -> Tuple[List[StateType], np.ndarray, np.ndarray, np.ndarray]:
        """Steps all environments with the selected actions."""
        next_states = [{} for _ in range(self.num_envs)]
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        step_rewards = np.zeros(self.num_envs, dtype=np.float32)

        for i in range(self.num_envs):
            env = self.envs[i]
            action = actions[i]
            try:
                reward, done = env.step(action)
                next_state = env.get_state()
                rewards[i] = reward
                dones[i] = done
                step_rewards[i] = reward
                next_states[i] = next_state
                self.current_episode_scores[i] += reward
                self.current_episode_lengths[i] += 1
                if not done:
                    self.current_episode_game_scores[i] = env.game_score
                    self.current_episode_lines_cleared[i] = (
                        env.lines_cleared_this_episode
                    )
            except Exception as e:
                print(f"ERROR: Env {i} step failed (Action: {action}): {e}")
                rewards[i] = self.reward_config.PENALTY_GAME_OVER
                dones[i] = True
                step_rewards[i] = rewards[i]
                next_states[i] = self._handle_env_crash(env, i)  # Reset and get state
                # Record episode immediately on crash
                self._record_episode_end(
                    i, rewards[i], current_global_step=0
                )  # Step will be updated later
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
    ):
        """Pushes transitions to the replay buffer."""
        for i in range(self.num_envs):
            self.buffer.push(
                self.current_states[i], actions[i], rewards[i], next_states[i], dones[i]
            )

    def _handle_episode_ends(self, dones: np.ndarray, current_global_step: int):
        """Records stats for finished episodes and resets their state."""
        for i in range(self.num_envs):
            if dones[i]:
                self._record_episode_end(i, 0, current_global_step + i + 1)  # Pass step
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
            global_step=current_global_step,
            game_score=final_game_score,
            lines_cleared=final_lines,
        )

    def _reset_episode_stats(self, env_index: int):
        """Resets tracking stats for a specific environment index."""
        self.current_episode_scores[env_index] = 0.0
        self.current_episode_lengths[env_index] = 0
        self.current_episode_game_scores[env_index] = 0
        self.current_episode_lines_cleared[env_index] = 0

    def _update_current_states(self, next_states: List[StateType]):
        """Updates the current states for the next iteration."""
        self.current_states = next_states

    def _log_step_stats(
        self,
        actions: List[ActionType],
        step_rewards: np.ndarray,
        chosen_slots: np.ndarray,
        shape_qs_list: List[np.ndarray],
        current_global_step: int,
    ):
        """Logs statistics related to the collection step."""
        step_log_data = {
            "buffer_size": len(self.buffer),
            "global_step": current_global_step + self.num_envs,  # Step after collection
        }
        if self.tb_config.LOG_HISTOGRAMS:
            step_log_data["step_rewards_batch"] = step_rewards
            step_log_data["action_batch"] = np.array(actions, dtype=np.int32)
            valid_chosen_slots = chosen_slots[chosen_slots != -1]
            if len(valid_chosen_slots) > 0:
                step_log_data["chosen_shape_slot_batch"] = valid_chosen_slots
            if shape_qs_list:
                flat_shape_qs = np.concatenate(shape_qs_list)
                valid_flat_shape_qs = flat_shape_qs[np.isfinite(flat_shape_qs)]
                if len(valid_flat_shape_qs) > 0:
                    step_log_data["shape_slot_max_q_batch"] = valid_flat_shape_qs
        if self.tb_config.LOG_SHAPE_PLACEMENT_Q_VALUES:
            batch_q_vals = self.agent.get_last_batch_q_values_for_actions()
            if batch_q_vals is not None:
                step_log_data["batch_q_values_actions_taken"] = batch_q_vals

        self.stats_recorder.record_step(step_log_data)

    def get_episode_count(self) -> int:
        return self.episode_count
