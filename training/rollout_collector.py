# File: training/rollout_collector.py
import time
import torch
import numpy as np
import random
import traceback
from typing import List, Dict, Any, Tuple, Optional

from config import (
    EnvConfig,
    RewardConfig,
    TensorBoardConfig,
    PPOConfig,
    RNNConfig,
    DEVICE,
)
from environment.game_state import GameState, StateType
from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from utils.types import ActionType
from .rollout_storage import RolloutStorage


class RolloutCollector:
    """Handles interaction with parallel environments to collect rollouts for PPO."""

    def __init__(
        self,
        envs: List[GameState],
        agent: PPOAgent,
        stats_recorder: StatsRecorderBase,
        env_config: EnvConfig,
        ppo_config: PPOConfig,
        rnn_config: RNNConfig,
        reward_config: RewardConfig,
        tb_config: TensorBoardConfig,
    ):
        self.envs = envs
        self.agent = agent
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.reward_config = reward_config
        self.tb_config = tb_config
        self.device = DEVICE

        self.rollout_storage = RolloutStorage(
            ppo_config.NUM_STEPS_PER_ROLLOUT,
            self.num_envs,
            self.env_config,
            self.rnn_config,
            self.device,
        )

        self.current_obs_grid_cpu = np.zeros(
            (self.num_envs, *self.env_config.GRID_STATE_SHAPE), dtype=np.float32
        )
        self.current_obs_shapes_cpu = np.zeros(
            (self.num_envs, self.env_config.SHAPE_STATE_DIM), dtype=np.float32
        )
        self.current_dones_cpu = np.zeros(self.num_envs, dtype=bool)

        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_game_scores = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_lines_cleared = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_count = 0

        # Holds the tuple (h, c) on the correct device
        self.current_lstm_state_device: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
            None
        )
        if self.rnn_config.USE_RNN:
            self.current_lstm_state_device = self.agent.get_initial_hidden_state(
                self.num_envs
            )

        self._reset_all_envs()

        initial_obs_grid_t = torch.from_numpy(self.current_obs_grid_cpu).to(
            self.rollout_storage.device
        )
        initial_obs_shapes_t = torch.from_numpy(self.current_obs_shapes_cpu).to(
            self.rollout_storage.device
        )
        initial_dones_t = (
            torch.from_numpy(self.current_dones_cpu)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )

        self.rollout_storage.obs_grid[0].copy_(initial_obs_grid_t)
        self.rollout_storage.obs_shapes[0].copy_(initial_obs_shapes_t)
        self.rollout_storage.dones[0].copy_(initial_dones_t)

        # MODIFIED: Copy initial LSTM state (h and c) to storage
        if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
            if (
                self.rollout_storage.hidden_states is not None
                and self.rollout_storage.cell_states is not None
            ):
                self.rollout_storage.hidden_states[0].copy_(
                    self.current_lstm_state_device[0]
                )
                self.rollout_storage.cell_states[0].copy_(
                    self.current_lstm_state_device[1]
                )

        print(f"[RolloutCollector] Initialized for {self.num_envs} environments.")

    def _reset_all_envs(self):
        """Resets all environments and updates initial CPU observations."""
        for i, env in enumerate(self.envs):
            state_dict = env.reset()
            self.current_obs_grid_cpu[i] = state_dict["grid"]
            self.current_obs_shapes_cpu[i] = state_dict["shapes"].flatten()
            self.current_dones_cpu[i] = False
            self.current_episode_scores[i] = 0.0
            self.current_episode_lengths[i] = 0
            self.current_episode_game_scores[i] = 0
            self.current_episode_lines_cleared[i] = 0
        if self.rnn_config.USE_RNN:
            # MODIFIED: Use consistent name
            self.current_lstm_state_device = self.agent.get_initial_hidden_state(
                self.num_envs
            )

    def _record_episode_stats(
        self,
        env_index: int,
        final_reward: float,
        step_count_offset: int,
        current_global_step: int,
    ):
        """Helper function to record stats for a finished episode."""
        self.episode_count += 1
        episode_score = self.current_episode_scores[env_index] + final_reward
        episode_length = self.current_episode_lengths[env_index] + step_count_offset
        game_score = self.current_episode_game_scores[env_index]
        lines_cleared = self.current_episode_lines_cleared[env_index]
        approx_global_step = current_global_step + env_index + 1

        self.stats_recorder.record_episode(
            episode_score=episode_score,
            episode_length=episode_length,
            episode_num=self.episode_count,
            global_step=approx_global_step,
            game_score=game_score,
            lines_cleared=lines_cleared,
        )

    def collect_one_step(self, current_global_step: int) -> int:
        """Collects one step of experience from all environments using batching."""
        step_start_time = time.time()

        valid_actions_list: List[Optional[List[int]]] = [None] * self.num_envs
        envs_needing_action: List[int] = []
        initial_done_indices: List[int] = []

        for i in range(self.num_envs):
            if self.current_dones_cpu[i]:
                initial_done_indices.append(i)
                continue

            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                self.current_dones_cpu[i] = True
                initial_done_indices.append(i)
            else:
                valid_actions_list[i] = valid_actions
                envs_needing_action.append(i)

        actions_np = np.zeros(self.num_envs, dtype=np.int64)
        log_probs_np = np.zeros(self.num_envs, dtype=np.float32)
        values_np = np.zeros(self.num_envs, dtype=np.float32)
        # MODIFIED: Use consistent name, will hold tuple (h, c) for *next* step
        next_lstm_state_device = None

        if envs_needing_action:
            active_indices = torch.tensor(envs_needing_action, dtype=torch.long)
            batch_obs_grid_cpu = self.current_obs_grid_cpu[active_indices]
            batch_obs_shapes_cpu = self.current_obs_shapes_cpu[active_indices]
            batch_valid_actions = [valid_actions_list[i] for i in envs_needing_action]

            batch_hidden_state = None
            # MODIFIED: Use current_lstm_state_device
            if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
                h_n, c_n = self.current_lstm_state_device
                batch_hidden_state = (
                    h_n[:, active_indices, :].contiguous(),
                    c_n[:, active_indices, :].contiguous(),
                )

            batch_obs_grid_t = torch.from_numpy(batch_obs_grid_cpu).to(
                self.agent.device
            )
            batch_obs_shapes_t = torch.from_numpy(batch_obs_shapes_cpu).to(
                self.agent.device
            )

            with torch.no_grad():
                (
                    batch_actions_t,
                    batch_log_probs_t,
                    batch_values_t,
                    # MODIFIED: Receive tuple (h, c) for next state
                    batch_next_lstm_state,
                ) = self.agent.select_action_batch(
                    batch_obs_grid_t,
                    batch_obs_shapes_t,
                    batch_hidden_state,
                    batch_valid_actions,
                )

            actions_np[active_indices] = batch_actions_t.cpu().numpy()
            log_probs_np[active_indices] = batch_log_probs_t.cpu().numpy()
            values_np[active_indices] = batch_values_t.cpu().numpy()

            # MODIFIED: Update the full LSTM state (h and c)
            if self.rnn_config.USE_RNN and batch_next_lstm_state is not None:
                if self.current_lstm_state_device is None:
                    # Should not happen if USE_RNN is true, but as fallback
                    self.current_lstm_state_device = (
                        self.agent.get_initial_hidden_state(self.num_envs)
                    )

                next_h = self.current_lstm_state_device[0].clone()
                next_c = self.current_lstm_state_device[1].clone()
                next_h[:, active_indices, :] = batch_next_lstm_state[0]
                next_c[:, active_indices, :] = batch_next_lstm_state[1]

                if initial_done_indices:
                    reset_indices = torch.tensor(initial_done_indices, dtype=torch.long)
                    reset_h, reset_c = self.agent.get_initial_hidden_state(
                        len(initial_done_indices)
                    )
                    if reset_h is not None and reset_c is not None:
                        next_h[:, reset_indices, :] = reset_h
                        next_c[:, reset_indices, :] = reset_c
                next_lstm_state_device = (next_h, next_c)

        step_dones_np = np.copy(self.current_dones_cpu)
        step_rewards_np = np.zeros(self.num_envs, dtype=np.float32)
        next_obs_grid_cpu = np.copy(self.current_obs_grid_cpu)
        next_obs_shapes_cpu = np.copy(self.current_obs_shapes_cpu)
        reset_indices_this_step: List[int] = []

        for i in range(self.num_envs):
            if i in initial_done_indices:
                final_reward = 0.0
                step_offset = 0
                if not self.current_dones_cpu[i]:
                    final_reward = self.reward_config.PENALTY_GAME_OVER
                    log_probs_np[i] = -1e9
                    values_np[i] = 0.0
                    step_offset = 1

                self._record_episode_stats(
                    i, final_reward, step_offset, current_global_step
                )
                reset_indices_this_step.append(i)

                next_state_dict = self.envs[i].reset()
                next_obs_grid_cpu[i] = next_state_dict["grid"]
                next_obs_shapes_cpu[i] = next_state_dict["shapes"].flatten()
                step_dones_np[i] = False

                self.current_episode_scores[i] = 0.0
                self.current_episode_lengths[i] = 0
                self.current_episode_game_scores[i] = 0
                self.current_episode_lines_cleared[i] = 0
                continue

            action_to_take = actions_np[i]
            try:
                reward, done = self.envs[i].step(action_to_take)
                step_rewards_np[i] = reward
                step_dones_np[i] = done

                self.current_episode_scores[i] += reward
                self.current_episode_lengths[i] += 1

                if done:
                    self._record_episode_stats(i, 0.0, 0, current_global_step)
                    reset_indices_this_step.append(i)
                    next_state_dict = self.envs[i].reset()
                    step_dones_np[i] = False

                    self.current_episode_scores[i] = 0.0
                    self.current_episode_lengths[i] = 0
                    self.current_episode_game_scores[i] = 0
                    self.current_episode_lines_cleared[i] = 0

                    # MODIFIED: Reset RNN state for this env
                    if self.rnn_config.USE_RNN and next_lstm_state_device is not None:
                        reset_h, reset_c = self.agent.get_initial_hidden_state(1)
                        if reset_h is not None and reset_c is not None:
                            next_lstm_state_device[0][:, i : i + 1, :] = reset_h
                            next_lstm_state_device[1][:, i : i + 1, :] = reset_c
                else:
                    next_state_dict = self.envs[i].get_state()
                    self.current_episode_game_scores[i] = self.envs[i].game_score
                    self.current_episode_lines_cleared[i] = self.envs[
                        i
                    ].lines_cleared_this_episode

                next_obs_grid_cpu[i] = next_state_dict["grid"]
                next_obs_shapes_cpu[i] = next_state_dict["shapes"].flatten()

            except Exception as e:
                print(f"ERROR: Env {i} step failed (Action: {action_to_take}): {e}")
                traceback.print_exc()
                step_rewards_np[i] = self.reward_config.PENALTY_GAME_OVER
                step_dones_np[i] = True
                reset_indices_this_step.append(i)

                try:
                    next_state_dict = self.envs[i].reset()
                    next_obs_grid_cpu[i] = next_state_dict["grid"]
                    next_obs_shapes_cpu[i] = next_state_dict["shapes"].flatten()
                except Exception as reset_e:
                    print(f"FATAL: Env {i} failed reset after crash: {reset_e}")
                    next_obs_grid_cpu[i].fill(0)
                    next_obs_shapes_cpu[i].fill(0)

                self._record_episode_stats(
                    i, step_rewards_np[i], 1, current_global_step
                )

                self.current_episode_scores[i] = 0.0
                self.current_episode_lengths[i] = 0
                self.current_episode_game_scores[i] = 0
                self.current_episode_lines_cleared[i] = 0

                # MODIFIED: Reset RNN state if needed
                if self.rnn_config.USE_RNN and next_lstm_state_device is not None:
                    reset_h, reset_c = self.agent.get_initial_hidden_state(1)
                    if reset_h is not None and reset_c is not None:
                        next_lstm_state_device[0][:, i : i + 1, :] = reset_h
                        next_lstm_state_device[1][:, i : i + 1, :] = reset_c

        # --- Store Data ---
        obs_grid_t = torch.from_numpy(self.current_obs_grid_cpu).to(
            self.rollout_storage.device
        )
        obs_shapes_t = torch.from_numpy(self.current_obs_shapes_cpu).to(
            self.rollout_storage.device
        )
        actions_t = (
            torch.from_numpy(actions_np)
            .long()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )
        log_probs_t = (
            torch.from_numpy(log_probs_np)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )
        values_t = (
            torch.from_numpy(values_np)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )
        rewards_t = (
            torch.from_numpy(step_rewards_np)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )
        dones_t_for_storage = (
            torch.from_numpy(self.current_dones_cpu)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )

        # MODIFIED: Get the LSTM state (h, c) corresponding to the *current* observation
        lstm_state_to_store = None
        if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
            lstm_state_to_store = (
                self.current_lstm_state_device[0].to(self.rollout_storage.device),
                self.current_lstm_state_device[1].to(self.rollout_storage.device),
            )

        # MODIFIED: Pass the lstm_state tuple to insert
        self.rollout_storage.insert(
            obs_grid_t,
            obs_shapes_t,
            actions_t,
            log_probs_t,
            values_t,
            rewards_t,
            dones_t_for_storage,
            lstm_state_to_store,  # Pass the tuple (h, c)
        )

        # --- Update Collector's State for Next Iteration ---
        self.current_obs_grid_cpu = next_obs_grid_cpu
        self.current_obs_shapes_cpu = next_obs_shapes_cpu
        self.current_dones_cpu = step_dones_np

        # MODIFIED: Update current LSTM state (h, c)
        if self.rnn_config.USE_RNN:
            self.current_lstm_state_device = next_lstm_state_device

        collection_time = time.time() - step_start_time
        sps = self.num_envs / max(1e-9, collection_time)
        self.stats_recorder.record_step(
            {"sps_collection": sps, "rollout_collection_time": collection_time}
        )

        return self.num_envs

    def compute_advantages_for_storage(self):
        """Computes GAE advantages using the data in RolloutStorage."""
        with torch.no_grad():
            final_obs_grid_t = torch.from_numpy(self.current_obs_grid_cpu).to(
                self.agent.device
            )
            final_obs_shapes_t = torch.from_numpy(self.current_obs_shapes_cpu).to(
                self.agent.device
            )
            final_lstm_state = None
            # MODIFIED: Use current LSTM state (h, c)
            if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
                final_lstm_state = (
                    self.current_lstm_state_device[0].to(self.agent.device),
                    self.current_lstm_state_device[1].to(self.agent.device),
                )

            # Add sequence dimension for final value prediction if RNN is used
            if self.rnn_config.USE_RNN:
                final_obs_grid_t = final_obs_grid_t.unsqueeze(1)
                final_obs_shapes_t = final_obs_shapes_t.unsqueeze(1)

            # MODIFIED: Pass the final LSTM state tuple (h, c)
            _, next_value, _ = self.agent.network(
                final_obs_grid_t, final_obs_shapes_t, final_lstm_state
            )

            if self.rnn_config.USE_RNN:
                next_value = next_value.squeeze(1)

            if next_value.ndim == 1:
                next_value = next_value.unsqueeze(-1)

            final_dones = (
                torch.from_numpy(self.current_dones_cpu)
                .float()
                .unsqueeze(1)
                .to(self.device)
            )

        self.rollout_storage.compute_returns_and_advantages(
            next_value, final_dones, self.ppo_config.GAMMA, self.ppo_config.GAE_LAMBDA
        )

    def get_episode_count(self) -> int:
        return self.episode_count
