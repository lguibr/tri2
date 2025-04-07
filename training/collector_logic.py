# File: training/collector_logic.py
import time
import torch
import numpy as np
import traceback
from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Optional,
    TYPE_CHECKING,
)  # Added TYPE_CHECKING

from config import EnvConfig, RewardConfig, PPOConfig, RNNConfig, ObsNormConfig
from environment.game_state import GameState
from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase

# Removed: from .rollout_storage import RolloutStorage
from .collector_state import CollectorState
from .normalization import normalize_obs, update_obs_rms
from .env_interaction import (
    reset_env,
    update_raw_obs_from_state_dict,
    reset_rnn_state_for_env,
)

# Use TYPE_CHECKING for RolloutStorage hint to avoid runtime import error
if TYPE_CHECKING:
    from .rollout_storage import RolloutStorage


class CollectorLogic:
    """Handles the core logic of collecting one step of experience."""

    def __init__(
        self,
        envs: List[GameState],
        agent: PPOAgent,
        storage: "RolloutStorage",  # Use string hint or TYPE_CHECKING block
        state: CollectorState,
        stats_recorder: StatsRecorderBase,
        env_config: EnvConfig,
        reward_config: RewardConfig,
        ppo_config: PPOConfig,
        rnn_config: RNNConfig,
        obs_norm_config: ObsNormConfig,
    ):
        # --- Import RolloutStorage here ---
        from .rollout_storage import RolloutStorage

        # --- End Import ---

        self.envs = envs
        self.agent = agent
        self.storage = (
            storage  # storage is already passed in, no need to re-assign from import
        )
        self.state = state
        self.stats_recorder = stats_recorder
        self.env_config = env_config
        self.reward_config = reward_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.obs_norm_config = obs_norm_config
        self.num_envs = len(envs)

    def _record_episode_stats(
        self, env_index: int, final_reward_adjustment: float, current_global_step: int
    ):
        """Records completed episode statistics using the thread-safe stats_recorder."""
        self.state.episode_count += 1
        final_episode_score = (
            self.state.current_episode_scores[env_index] + final_reward_adjustment
        )
        final_episode_length = self.state.current_episode_lengths[env_index]
        final_game_score = self.state.current_episode_game_scores[env_index]
        final_triangles_cleared = self.state.current_episode_triangles_cleared[
            env_index
        ]

        self.stats_recorder.record_episode(
            episode_score=final_episode_score,
            episode_length=final_episode_length,
            episode_num=self.state.episode_count,
            global_step=current_global_step,
            game_score=final_game_score,
            triangles_cleared=final_triangles_cleared,
        )

    def collect_one_step(self, current_global_step: int) -> int:
        """Collects one step of experience from all environments."""
        step_start_time = time.time()
        with self.storage._lock:
            current_rollout_step = self.storage.step  # Get current storage step index

        # 1. Identify active environments and get valid actions (CPU operations)
        active_env_indices: List[int] = []
        valid_actions_list: List[Optional[List[int]]] = [None] * self.num_envs
        envs_done_pre_action: List[int] = []

        for i in range(self.num_envs):
            self.envs[i]._update_timers()  # Update internal env timers if any
            if self.state.current_dones_cpu[i]:
                continue  # Skip already done envs
            if self.envs[i].is_frozen():
                continue  # Skip frozen envs (e.g., during animation)

            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                envs_done_pre_action.append(i)  # Mark as done if no valid moves
            else:
                valid_actions_list[i] = valid_actions
                active_env_indices.append(i)

        # 2. Select actions ONLY for active environments (Agent interaction)
        actions_np = np.zeros(self.num_envs, dtype=np.int64)
        log_probs_np = np.zeros(self.num_envs, dtype=np.float32)
        values_np = np.zeros(self.num_envs, dtype=np.float32)
        next_lstm_state_device = (
            self.state.current_lstm_state_device
        )  # Start with current state

        if active_env_indices:
            active_indices_tensor = torch.tensor(active_env_indices, dtype=torch.long)

            # Prepare batch of RAW observations on CPU
            batch_obs_grid_raw = self.state.current_raw_obs_grid_cpu[active_env_indices]
            batch_obs_shapes_raw = self.state.current_raw_obs_shapes_cpu[
                active_env_indices
            ]
            batch_obs_availability_raw = self.state.current_raw_obs_availability_cpu[
                active_env_indices
            ]
            batch_obs_explicit_features_raw = (
                self.state.current_raw_obs_explicit_features_cpu[active_env_indices]
            )

            # Normalize observations on CPU using helper
            batch_obs_grid_norm = normalize_obs(
                batch_obs_grid_raw,
                self.state.obs_rms.get("grid"),
                self.obs_norm_config.OBS_CLIP,
            )
            batch_obs_shapes_norm = normalize_obs(
                batch_obs_shapes_raw,
                self.state.obs_rms.get("shapes"),
                self.obs_norm_config.OBS_CLIP,
            )
            batch_obs_availability_norm = normalize_obs(
                batch_obs_availability_raw,
                self.state.obs_rms.get("shape_availability"),
                self.obs_norm_config.OBS_CLIP,
            )
            batch_obs_explicit_features_norm = normalize_obs(
                batch_obs_explicit_features_raw,
                self.state.obs_rms.get("explicit_features"),
                self.obs_norm_config.OBS_CLIP,
            )

            # Convert normalized observations to tensors on agent's device
            batch_obs_grid_t = torch.from_numpy(batch_obs_grid_norm).to(
                self.agent.device
            )
            batch_obs_shapes_t = torch.from_numpy(batch_obs_shapes_norm).to(
                self.agent.device
            )
            batch_obs_availability_t = torch.from_numpy(batch_obs_availability_norm).to(
                self.agent.device
            )
            batch_obs_explicit_features_t = torch.from_numpy(
                batch_obs_explicit_features_norm
            ).to(self.agent.device)
            batch_valid_actions = [valid_actions_list[i] for i in active_env_indices]

            # Select corresponding hidden state (already on agent device)
            batch_hidden_state_device = None
            if (
                self.rnn_config.USE_RNN
                and self.state.current_lstm_state_device is not None
            ):
                h_n, c_n = self.state.current_lstm_state_device
                batch_hidden_state_device = (
                    h_n[:, active_indices_tensor, :].contiguous(),
                    c_n[:, active_indices_tensor, :].contiguous(),
                )

            # Call agent's select_action_batch (uses internal lock)
            (
                batch_actions_t,
                batch_log_probs_t,
                batch_values_t,
                batch_next_lstm_state_device,
            ) = self.agent.select_action_batch(
                batch_obs_grid_t,
                batch_obs_shapes_t,
                batch_obs_availability_t,
                batch_obs_explicit_features_t,
                batch_hidden_state_device,
                batch_valid_actions,
            )

            # Store results back into numpy arrays (CPU)
            actions_np[active_env_indices] = batch_actions_t.cpu().numpy()
            log_probs_np[active_env_indices] = batch_log_probs_t.cpu().numpy()
            values_np[active_env_indices] = batch_values_t.cpu().numpy()

            # Update the full hidden state (on agent device)
            if self.rnn_config.USE_RNN and batch_next_lstm_state_device is not None:
                next_h, next_c = (
                    self.state.current_lstm_state_device[0].clone(),
                    self.state.current_lstm_state_device[1].clone(),
                )
                next_h[:, active_indices_tensor, :] = batch_next_lstm_state_device[0]
                next_c[:, active_indices_tensor, :] = batch_next_lstm_state_device[1]
                next_lstm_state_device = (next_h, next_c)

        # 3. Step environments, handle resets, update RAW observations (CPU operations)
        # Prepare buffers for the *next* step's raw observations
        next_raw_obs_grid_cpu = np.copy(self.state.current_raw_obs_grid_cpu)
        next_raw_obs_shapes_cpu = np.copy(self.state.current_raw_obs_shapes_cpu)
        next_raw_obs_availability_cpu = np.copy(
            self.state.current_raw_obs_availability_cpu
        )
        next_raw_obs_explicit_features_cpu = np.copy(
            self.state.current_raw_obs_explicit_features_cpu
        )
        step_rewards_np = np.zeros(self.num_envs, dtype=np.float32)
        step_dones_np = np.copy(
            self.state.current_dones_cpu
        )  # Dones resulting from *this* step

        for i in range(self.num_envs):
            final_reward_adj = 0.0
            if self.state.current_dones_cpu[
                i
            ]:  # Env was done at the start of this step, reset it
                new_state_dict = reset_env(
                    self.envs[i],
                    self.state.current_episode_scores,
                    self.state.current_episode_lengths,
                    self.state.current_episode_game_scores,
                    self.state.current_episode_triangles_cleared,
                    i,
                    self.env_config,
                )
                update_raw_obs_from_state_dict(
                    i,
                    new_state_dict,
                    next_raw_obs_grid_cpu,
                    next_raw_obs_shapes_cpu,
                    next_raw_obs_availability_cpu,
                    next_raw_obs_explicit_features_cpu,
                )
                reset_rnn_state_for_env(
                    i, self.rnn_config.USE_RNN, next_lstm_state_device, self.agent
                )  # Reset next state
                step_dones_np[i] = False  # It's not done *after* the reset
            elif (
                i in envs_done_pre_action
            ):  # Env became done because no actions were valid
                final_reward_adj = self.reward_config.PENALTY_GAME_OVER
                log_probs_np[i], values_np[i] = (
                    -1e9,
                    0.0,
                )  # Assign dummy values for storage
                self.state.current_episode_lengths[
                    i
                ] += 1  # Increment length for the step that led to game over
                self._record_episode_stats(i, final_reward_adj, current_global_step)
                new_state_dict = reset_env(
                    self.envs[i],
                    self.state.current_episode_scores,
                    self.state.current_episode_lengths,
                    self.state.current_episode_game_scores,
                    self.state.current_episode_triangles_cleared,
                    i,
                    self.env_config,
                )
                update_raw_obs_from_state_dict(
                    i,
                    new_state_dict,
                    next_raw_obs_grid_cpu,
                    next_raw_obs_shapes_cpu,
                    next_raw_obs_availability_cpu,
                    next_raw_obs_explicit_features_cpu,
                )
                reset_rnn_state_for_env(
                    i, self.rnn_config.USE_RNN, next_lstm_state_device, self.agent
                )  # Reset next state
                step_dones_np[i] = True  # Mark as done for the *next* step
            elif i in active_env_indices:  # Env was active, take a step
                action_to_take = actions_np[i]
                try:
                    reward, done = self.envs[i].step(action_to_take)
                    step_rewards_np[i], step_dones_np[i] = reward, done
                    self.state.current_episode_scores[i] += reward
                    self.state.current_episode_lengths[i] += 1
                    self.state.current_episode_game_scores[i] = self.envs[i].game_score
                    self.state.current_episode_triangles_cleared[i] = self.envs[
                        i
                    ].triangles_cleared_this_episode
                    if done:
                        self._record_episode_stats(
                            i, 0.0, current_global_step
                        )  # Record completed episode
                        new_state_dict = reset_env(
                            self.envs[i],
                            self.state.current_episode_scores,
                            self.state.current_episode_lengths,
                            self.state.current_episode_game_scores,
                            self.state.current_episode_triangles_cleared,
                            i,
                            self.env_config,
                        )
                        update_raw_obs_from_state_dict(
                            i,
                            new_state_dict,
                            next_raw_obs_grid_cpu,
                            next_raw_obs_shapes_cpu,
                            next_raw_obs_availability_cpu,
                            next_raw_obs_explicit_features_cpu,
                        )
                        reset_rnn_state_for_env(
                            i,
                            self.rnn_config.USE_RNN,
                            next_lstm_state_device,
                            self.agent,
                        )  # Reset next state
                    else:  # Game continues, get next state
                        next_state_dict = self.envs[i].get_state()
                        update_raw_obs_from_state_dict(
                            i,
                            next_state_dict,
                            next_raw_obs_grid_cpu,
                            next_raw_obs_shapes_cpu,
                            next_raw_obs_availability_cpu,
                            next_raw_obs_explicit_features_cpu,
                        )
                except Exception as e:
                    print(f"ERROR: Env {i} step failed (Action: {action_to_take}): {e}")
                    traceback.print_exc()
                    step_rewards_np[i], step_dones_np[i] = (
                        self.reward_config.PENALTY_GAME_OVER,
                        True,
                    )
                    self.state.current_episode_lengths[i] += 1
                    self._record_episode_stats(i, 0.0, current_global_step)
                    new_state_dict = reset_env(
                        self.envs[i],
                        self.state.current_episode_scores,
                        self.state.current_episode_lengths,
                        self.state.current_episode_game_scores,
                        self.state.current_episode_triangles_cleared,
                        i,
                        self.env_config,
                    )
                    update_raw_obs_from_state_dict(
                        i,
                        new_state_dict,
                        next_raw_obs_grid_cpu,
                        next_raw_obs_shapes_cpu,
                        next_raw_obs_availability_cpu,
                        next_raw_obs_explicit_features_cpu,
                    )
                    reset_rnn_state_for_env(
                        i, self.rnn_config.USE_RNN, next_lstm_state_device, self.agent
                    )  # Reset next state

        # 4. Update RMS with the RAW observations collected *before* this step (CPU)
        update_obs_rms(
            self.state.current_raw_obs_grid_cpu, self.state.obs_rms.get("grid")
        )
        update_obs_rms(
            self.state.current_raw_obs_shapes_cpu, self.state.obs_rms.get("shapes")
        )
        update_obs_rms(
            self.state.current_raw_obs_availability_cpu,
            self.state.obs_rms.get("shape_availability"),
        )
        update_obs_rms(
            self.state.current_raw_obs_explicit_features_cpu,
            self.state.obs_rms.get("explicit_features"),
        )

        # 5. Store potentially NORMALIZED results in RolloutStorage (CPU tensors)
        # Normalize the observations *before* the step was taken
        obs_grid_norm = normalize_obs(
            self.state.current_raw_obs_grid_cpu,
            self.state.obs_rms.get("grid"),
            self.obs_norm_config.OBS_CLIP,
        )
        obs_shapes_norm = normalize_obs(
            self.state.current_raw_obs_shapes_cpu,
            self.state.obs_rms.get("shapes"),
            self.obs_norm_config.OBS_CLIP,
        )
        obs_availability_norm = normalize_obs(
            self.state.current_raw_obs_availability_cpu,
            self.state.obs_rms.get("shape_availability"),
            self.obs_norm_config.OBS_CLIP,
        )
        obs_explicit_features_norm = normalize_obs(
            self.state.current_raw_obs_explicit_features_cpu,
            self.state.obs_rms.get("explicit_features"),
            self.obs_norm_config.OBS_CLIP,
        )

        obs_grid_t = torch.from_numpy(obs_grid_norm)
        obs_shapes_t = torch.from_numpy(obs_shapes_norm)
        obs_availability_t = torch.from_numpy(obs_availability_norm)
        obs_explicit_features_t = torch.from_numpy(obs_explicit_features_norm)
        actions_t = torch.from_numpy(actions_np).long().unsqueeze(1)
        log_probs_t = torch.from_numpy(log_probs_np).float().unsqueeze(1)
        values_t = torch.from_numpy(values_np).float().unsqueeze(1)
        rewards_t = torch.from_numpy(step_rewards_np).float().unsqueeze(1)
        # Store the dones that resulted *from this step*
        dones_t_for_storage = torch.from_numpy(step_dones_np).float().unsqueeze(1)

        # Get LSTM state corresponding to the observation *before* the action (copy from device to CPU)
        lstm_state_to_store_cpu = None
        if self.rnn_config.USE_RNN and self.state.current_lstm_state_device is not None:
            lstm_state_to_store_cpu = (
                self.state.current_lstm_state_device[0].cpu(),
                self.state.current_lstm_state_device[1].cpu(),
            )

        # Insert into storage (uses internal lock)
        self.storage.insert(
            obs_grid_t,
            obs_shapes_t,
            obs_availability_t,
            obs_explicit_features_t,
            actions_t,
            log_probs_t,
            values_t,
            rewards_t,
            dones_t_for_storage,
            lstm_state_to_store_cpu,
        )

        # 6. Update collector's current RAW state (CPU) and LSTM state (Device) for the *next* iteration
        self.state.current_raw_obs_grid_cpu = next_raw_obs_grid_cpu
        self.state.current_raw_obs_shapes_cpu = next_raw_obs_shapes_cpu
        self.state.current_raw_obs_availability_cpu = next_raw_obs_availability_cpu
        self.state.current_raw_obs_explicit_features_cpu = (
            next_raw_obs_explicit_features_cpu
        )
        self.state.current_dones_cpu = (
            step_dones_np  # Update dones for the next step check
        )
        self.state.current_lstm_state_device = (
            next_lstm_state_device  # Update LSTM state (on agent device)
        )

        # 7. Record performance (thread-safe recorder)
        collection_time = time.time() - step_start_time
        sps = self.num_envs / max(1e-9, collection_time)
        self.stats_recorder.record_step(
            {"sps_collection": sps, "rollout_collection_time": collection_time}
        )

        return self.num_envs  # Return number of steps collected (always num_envs here)
