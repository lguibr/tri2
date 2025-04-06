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
    """
    Handles interaction with parallel environments to collect rollouts for PPO.
    Includes staggered start of interaction to desynchronize environments.
    """

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

        # CPU Buffers for current step's observations and dones
        self.current_obs_grid_cpu = np.zeros(
            (self.num_envs, *self.env_config.GRID_STATE_SHAPE), dtype=np.float32
        )
        self.current_obs_shapes_cpu = np.zeros(
            (self.num_envs, self.env_config.SHAPE_STATE_DIM), dtype=np.float32
        )
        self.current_obs_availability_cpu = np.zeros(
            (self.num_envs, self.env_config.SHAPE_AVAILABILITY_DIM), dtype=np.float32
        )
        # --- UPDATED: Use correct dimension for explicit features ---
        self.current_obs_explicit_features_cpu = np.zeros(
            (self.num_envs, self.env_config.EXPLICIT_FEATURES_DIM), dtype=np.float32
        )
        # --- END UPDATED ---
        self.current_dones_cpu = np.zeros(self.num_envs, dtype=bool)

        # Episode trackers
        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_game_scores = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_lines_cleared = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_count = 0

        # RNN state
        self.current_lstm_state_device: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
            None
        )
        if self.rnn_config.USE_RNN:
            self.current_lstm_state_device = self.agent.get_initial_hidden_state(
                self.num_envs
            )

        # --- NEW: Staggered Start ---
        # Assign a start step delay for each environment (simple sequential stagger)
        # Ensures interaction starts spread out over the first rollout steps.
        self.env_start_step_delay = (
            np.arange(self.num_envs) % self.ppo_config.NUM_STEPS_PER_ROLLOUT
        )
        print(
            f"[RolloutCollector] Interaction start steps staggered up to step {np.max(self.env_start_step_delay)}."
        )
        # --- END NEW ---

        # Reset environments and populate initial observations
        self._reset_all_envs()
        self._copy_initial_observations_to_storage()

        print(f"[RolloutCollector] Initialized for {self.num_envs} environments.")

    def _reset_env(self, env_index: int) -> StateType:
        """Resets a single environment and returns its initial state dict."""
        try:
            state_dict = self.envs[env_index].reset()
            self.current_episode_scores[env_index] = 0.0
            self.current_episode_lengths[env_index] = 0
            self.current_episode_game_scores[env_index] = 0
            self.current_episode_lines_cleared[env_index] = 0
            return state_dict
        except KeyError as e:
            print(
                f"FATAL ERROR: Env {env_index} reset missing key '{e}'. Check GameState.reset()"
            )
            raise e
        except Exception as e:
            print(f"ERROR resetting env {env_index}: {e}")
            dummy_state: StateType = {
                "grid": np.zeros(self.env_config.GRID_STATE_SHAPE, dtype=np.float32),
                "shapes": np.zeros(
                    (
                        self.env_config.NUM_SHAPE_SLOTS,
                        self.env_config.SHAPE_FEATURES_PER_SHAPE,
                    ),
                    dtype=np.float32,
                ),
                "shape_availability": np.zeros(
                    self.env_config.SHAPE_AVAILABILITY_DIM, dtype=np.float32
                ),
                # --- UPDATED: Use correct dimension for dummy explicit features ---
                "explicit_features": np.zeros(
                    self.env_config.EXPLICIT_FEATURES_DIM, dtype=np.float32
                ),
                # --- END UPDATED ---
            }
            self.current_dones_cpu[env_index] = True
            return dummy_state

    def _update_obs_from_state_dict(self, env_index: int, state_dict: StateType):
        """Updates the CPU observation buffers for a given environment index."""
        self.current_obs_grid_cpu[env_index] = state_dict["grid"]
        self.current_obs_shapes_cpu[env_index] = state_dict["shapes"].reshape(-1)[
            : self.env_config.SHAPE_STATE_DIM
        ]
        self.current_obs_availability_cpu[env_index] = state_dict["shape_availability"]
        # --- UPDATED: Copy explicit features ---
        self.current_obs_explicit_features_cpu[env_index] = state_dict[
            "explicit_features"
        ]
        # --- END UPDATED ---

    def _reset_all_envs(self):
        """Resets all environments and updates initial observations."""
        for i in range(self.num_envs):
            initial_state = self._reset_env(i)
            self._update_obs_from_state_dict(i, initial_state)
            self.current_dones_cpu[i] = False
        if self.rnn_config.USE_RNN:
            self.current_lstm_state_device = self.agent.get_initial_hidden_state(
                self.num_envs
            )

    def _copy_initial_observations_to_storage(self):
        """Copies the initial observations from CPU buffers to the RolloutStorage."""
        initial_obs_grid_t = torch.from_numpy(self.current_obs_grid_cpu).to(
            self.rollout_storage.device
        )
        initial_obs_shapes_t = torch.from_numpy(self.current_obs_shapes_cpu).to(
            self.rollout_storage.device
        )
        initial_obs_availability_t = torch.from_numpy(
            self.current_obs_availability_cpu
        ).to(self.rollout_storage.device)
        # --- UPDATED: Copy explicit features ---
        initial_obs_explicit_features_t = torch.from_numpy(
            self.current_obs_explicit_features_cpu
        ).to(self.rollout_storage.device)
        # --- END UPDATED ---
        initial_dones_t = (
            torch.from_numpy(self.current_dones_cpu)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )

        self.rollout_storage.obs_grid[0].copy_(initial_obs_grid_t)
        self.rollout_storage.obs_shapes[0].copy_(initial_obs_shapes_t)
        self.rollout_storage.obs_availability[0].copy_(initial_obs_availability_t)
        # --- UPDATED: Copy explicit features to storage ---
        self.rollout_storage.obs_explicit_features[0].copy_(
            initial_obs_explicit_features_t
        )
        # --- END UPDATED ---
        self.rollout_storage.dones[0].copy_(initial_dones_t)

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

    def _record_episode_stats(
        self, env_index: int, final_reward_adjustment: float, current_global_step: int
    ):
        """Helper function to record stats for a finished episode."""
        # Only record stats if the environment has actually started interacting
        if current_global_step >= self.env_start_step_delay[env_index]:
            self.episode_count += 1
            final_episode_score = (
                self.current_episode_scores[env_index] + final_reward_adjustment
            )
            final_episode_length = self.current_episode_lengths[env_index]
            final_game_score = self.current_episode_game_scores[env_index]
            final_lines_cleared = self.current_episode_lines_cleared[env_index]

            approx_global_step_for_log = current_global_step + env_index + 1

            self.stats_recorder.record_episode(
                episode_score=final_episode_score,
                episode_length=final_episode_length,
                episode_num=self.episode_count,
                global_step=approx_global_step_for_log,
                game_score=final_game_score,
                lines_cleared=final_lines_cleared,
            )

    def _reset_rnn_state_for_env(self, env_index: int):
        """Resets the RNN hidden state for a specific environment index."""
        if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
            reset_h, reset_c = self.agent.get_initial_hidden_state(1)
            if reset_h is not None and reset_c is not None:
                reset_h = reset_h.to(self.current_lstm_state_device[0].device)
                reset_c = reset_c.to(self.current_lstm_state_device[1].device)
                self.current_lstm_state_device[0][
                    :, env_index : env_index + 1, :
                ] = reset_h
                self.current_lstm_state_device[1][
                    :, env_index : env_index + 1, :
                ] = reset_c

    def collect_one_step(self, current_global_step: int) -> int:
        """Collects one step of experience from all environments using batching."""
        step_start_time = time.time()
        current_rollout_step = (
            self.rollout_storage.step
        )  # Get current step within the rollout

        # --- 1. Identify active, frozen, waiting, and truly done environments ---
        active_env_indices: List[int] = []
        frozen_env_indices: List[int] = []
        waiting_env_indices: List[int] = (
            []
        )  # NEW: Envs that haven't reached start delay
        envs_done_pre_action: List[int] = (
            []
        )  # Truly done (no moves, not frozen/waiting)
        valid_actions_list: List[Optional[List[int]]] = [None] * self.num_envs

        for i in range(self.num_envs):
            self.envs[i]._update_timers()  # Update timers first

            if self.current_dones_cpu[i]:
                # Already done from previous step, will be reset later
                continue

            # --- NEW: Check if waiting to start ---
            if current_rollout_step < self.env_start_step_delay[i]:
                waiting_env_indices.append(i)
                continue  # Skip checks below if waiting
            # --- END NEW ---

            if self.envs[i].is_frozen():
                frozen_env_indices.append(i)
                continue  # Skip action selection

            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                envs_done_pre_action.append(i)
            else:
                valid_actions_list[i] = valid_actions
                active_env_indices.append(i)

        # --- 2. Select actions ONLY for active environments ---
        actions_np = np.zeros(self.num_envs, dtype=np.int64)
        log_probs_np = np.zeros(self.num_envs, dtype=np.float32)
        values_np = np.zeros(self.num_envs, dtype=np.float32)
        next_lstm_state_device = self.current_lstm_state_device

        if active_env_indices:  # Only run agent if there are active envs
            active_indices_tensor = torch.tensor(active_env_indices, dtype=torch.long)

            batch_obs_grid_cpu = self.current_obs_grid_cpu[active_env_indices]
            batch_obs_shapes_cpu = self.current_obs_shapes_cpu[active_env_indices]
            batch_obs_availability_cpu = self.current_obs_availability_cpu[
                active_env_indices
            ]
            # --- UPDATED: Get explicit features for active envs ---
            batch_obs_explicit_features_cpu = self.current_obs_explicit_features_cpu[
                active_env_indices
            ]
            # --- END UPDATED ---
            batch_valid_actions = [valid_actions_list[i] for i in active_env_indices]

            batch_hidden_state_device = None
            if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
                h_n, c_n = self.current_lstm_state_device
                batch_hidden_state_device = (
                    h_n[:, active_indices_tensor, :].contiguous(),
                    c_n[:, active_indices_tensor, :].contiguous(),
                )

            batch_obs_grid_t = torch.from_numpy(batch_obs_grid_cpu).to(
                self.agent.device
            )
            batch_obs_shapes_t = torch.from_numpy(batch_obs_shapes_cpu).to(
                self.agent.device
            )
            batch_obs_availability_t = torch.from_numpy(batch_obs_availability_cpu).to(
                self.agent.device
            )
            # --- UPDATED: Convert explicit features to tensor ---
            batch_obs_explicit_features_t = torch.from_numpy(
                batch_obs_explicit_features_cpu
            ).to(self.agent.device)
            # --- END UPDATED ---

            with torch.no_grad():
                (
                    batch_actions_t,
                    batch_log_probs_t,
                    batch_values_t,
                    batch_next_lstm_state_device,
                ) = self.agent.select_action_batch(
                    batch_obs_grid_t,
                    batch_obs_shapes_t,
                    batch_obs_availability_t,
                    # --- UPDATED: Pass explicit features tensor ---
                    batch_obs_explicit_features_t,
                    # --- END UPDATED ---
                    batch_hidden_state_device,
                    batch_valid_actions,
                )

            actions_np[active_env_indices] = batch_actions_t.cpu().numpy()
            log_probs_np[active_env_indices] = batch_log_probs_t.cpu().numpy()
            values_np[active_env_indices] = batch_values_t.cpu().numpy()

            if self.rnn_config.USE_RNN and batch_next_lstm_state_device is not None:
                next_h = self.current_lstm_state_device[0].clone()
                next_c = self.current_lstm_state_device[1].clone()
                next_h[:, active_indices_tensor, :] = batch_next_lstm_state_device[0]
                next_c[:, active_indices_tensor, :] = batch_next_lstm_state_device[1]
                next_lstm_state_device = (next_h, next_c)

        # --- 3. Step environments, handle resets, and update observations ---
        next_obs_grid_cpu = np.copy(self.current_obs_grid_cpu)
        next_obs_shapes_cpu = np.copy(self.current_obs_shapes_cpu)
        next_obs_availability_cpu = np.copy(self.current_obs_availability_cpu)
        # --- UPDATED: Copy explicit features ---
        next_obs_explicit_features_cpu = np.copy(self.current_obs_explicit_features_cpu)
        # --- END UPDATED ---
        step_rewards_np = np.zeros(self.num_envs, dtype=np.float32)
        step_dones_np = np.copy(self.current_dones_cpu)

        for i in range(self.num_envs):
            is_done_this_step = False
            final_reward_adj = 0.0

            if self.current_dones_cpu[i]:
                # --- Was already done, reset ---
                new_state_dict = self._reset_env(i)
                self._update_obs_from_state_dict(i, new_state_dict)
                self._reset_rnn_state_for_env(i)
                step_dones_np[i] = False
                is_done_this_step = True

            # --- NEW: Handle waiting environments ---
            elif i in waiting_env_indices:
                step_rewards_np[i] = 0.0  # No reward while waiting
                step_dones_np[i] = False  # Not done
                is_done_this_step = False
                # Observations and LSTM state remain the same (initial state)
            # --- END NEW ---

            elif i in frozen_env_indices:
                # --- Was frozen, do not step ---
                step_rewards_np[i] = self.reward_config.REWARD_ALIVE_STEP
                step_dones_np[i] = False
                is_done_this_step = False
                # Observations and LSTM state remain the same

            elif i in envs_done_pre_action:
                # --- Became done (no valid moves), reset ---
                final_reward_adj = self.reward_config.PENALTY_GAME_OVER
                log_probs_np[i] = -1e9
                values_np[i] = 0.0
                self.current_episode_lengths[i] += 1

                self._record_episode_stats(i, final_reward_adj, current_global_step)
                new_state_dict = self._reset_env(i)
                self._update_obs_from_state_dict(i, new_state_dict)
                self._reset_rnn_state_for_env(i)
                step_dones_np[i] = True
                is_done_this_step = True

            else:
                # --- Environment is active, perform step ---
                action_to_take = actions_np[i]
                try:
                    reward, done = self.envs[i].step(action_to_take)
                    step_rewards_np[i] = reward
                    step_dones_np[i] = done

                    # Update episode trackers only if the env has started interacting
                    if current_rollout_step >= self.env_start_step_delay[i]:
                        self.current_episode_scores[i] += reward
                        self.current_episode_lengths[i] += 1
                        self.current_episode_game_scores[i] = self.envs[i].game_score
                        self.current_episode_lines_cleared[i] = self.envs[
                            i
                        ].lines_cleared_this_episode

                    if done:
                        self._record_episode_stats(i, 0.0, current_global_step)
                        new_state_dict = self._reset_env(i)
                        self._update_obs_from_state_dict(i, new_state_dict)
                        self._reset_rnn_state_for_env(i)
                        is_done_this_step = True
                    else:
                        next_state_dict = self.envs[i].get_state()
                        self._update_obs_from_state_dict(i, next_state_dict)
                        is_done_this_step = False

                except Exception as e:
                    print(f"ERROR: Env {i} step failed (Action: {action_to_take}): {e}")
                    traceback.print_exc()
                    step_rewards_np[i] = self.reward_config.PENALTY_GAME_OVER
                    step_dones_np[i] = True
                    # Update length even on error if it was interacting
                    if current_rollout_step >= self.env_start_step_delay[i]:
                        self.current_episode_lengths[i] += 1

                    self._record_episode_stats(i, 0.0, current_global_step)
                    new_state_dict = self._reset_env(i)
                    self._update_obs_from_state_dict(i, new_state_dict)
                    self._reset_rnn_state_for_env(i)
                    is_done_this_step = True

            # Update the observation buffers for the *next* step (S_{t+1})
            next_obs_grid_cpu[i] = self.current_obs_grid_cpu[i]
            next_obs_shapes_cpu[i] = self.current_obs_shapes_cpu[i]
            next_obs_availability_cpu[i] = self.current_obs_availability_cpu[i]
            # --- UPDATED: Update next explicit features ---
            next_obs_explicit_features_cpu[i] = self.current_obs_explicit_features_cpu[
                i
            ]
            # --- END UPDATED ---

        # --- 4. Store results in RolloutStorage ---
        obs_grid_t = torch.from_numpy(self.current_obs_grid_cpu).to(
            self.rollout_storage.device
        )
        obs_shapes_t = torch.from_numpy(self.current_obs_shapes_cpu).to(
            self.rollout_storage.device
        )
        obs_availability_t = torch.from_numpy(self.current_obs_availability_cpu).to(
            self.rollout_storage.device
        )
        # --- UPDATED: Convert explicit features to tensor ---
        obs_explicit_features_t = torch.from_numpy(
            self.current_obs_explicit_features_cpu
        ).to(self.rollout_storage.device)
        # --- END UPDATED ---
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
            torch.from_numpy(step_dones_np)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )

        lstm_state_to_store = None
        if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
            lstm_state_to_store = (
                self.current_lstm_state_device[0].to(self.rollout_storage.device),
                self.current_lstm_state_device[1].to(self.rollout_storage.device),
            )

        self.rollout_storage.insert(
            obs_grid_t,
            obs_shapes_t,
            obs_availability_t,
            # --- UPDATED: Pass explicit features to storage ---
            obs_explicit_features_t,
            # --- END UPDATED ---
            actions_t,
            log_probs_t,
            values_t,
            rewards_t,
            dones_t_for_storage,
            lstm_state_to_store,
        )

        # --- 5. Update collector's current state for the *next* iteration ---
        self.current_obs_grid_cpu = next_obs_grid_cpu
        self.current_obs_shapes_cpu = next_obs_shapes_cpu
        self.current_obs_availability_cpu = next_obs_availability_cpu
        # --- UPDATED: Update current explicit features ---
        self.current_obs_explicit_features_cpu = next_obs_explicit_features_cpu
        # --- END UPDATED ---
        self.current_dones_cpu = step_dones_np
        self.current_lstm_state_device = next_lstm_state_device

        # --- 6. Record performance ---
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
            final_obs_availability_t = torch.from_numpy(
                self.current_obs_availability_cpu
            ).to(self.agent.device)
            # --- UPDATED: Get final explicit features ---
            final_obs_explicit_features_t = torch.from_numpy(
                self.current_obs_explicit_features_cpu
            ).to(self.agent.device)
            # --- END UPDATED ---

            final_lstm_state = None
            if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
                final_lstm_state = (
                    self.current_lstm_state_device[0].to(self.agent.device),
                    self.current_lstm_state_device[1].to(self.agent.device),
                )

            if self.rnn_config.USE_RNN:
                final_obs_grid_t = final_obs_grid_t.unsqueeze(1)
                final_obs_shapes_t = final_obs_shapes_t.unsqueeze(1)
                final_obs_availability_t = final_obs_availability_t.unsqueeze(1)
                # --- UPDATED: Add sequence dim if RNN ---
                final_obs_explicit_features_t = final_obs_explicit_features_t.unsqueeze(
                    1
                )
                # --- END UPDATED ---

            _, next_value, _ = self.agent.network(
                final_obs_grid_t,
                final_obs_shapes_t,
                final_obs_availability_t,
                # --- UPDATED: Pass final explicit features ---
                final_obs_explicit_features_t,
                # --- END UPDATED ---
                final_lstm_state,
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
