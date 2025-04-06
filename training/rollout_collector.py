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
    ObsNormConfig,
    TransformerConfig,
    # DEVICE, # Removed direct import
)
from environment.game_state import GameState, StateType
from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from utils.types import ActionType

# --- MODIFIED: Import RunningMeanStd ---
from utils.running_mean_std import RunningMeanStd

# --- END MODIFIED ---
from .rollout_storage import RolloutStorage


class RolloutCollector:
    """
    Handles interaction with parallel environments to collect rollouts for PPO.
    Includes optional observation normalization using RunningMeanStd.
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
        obs_norm_config: ObsNormConfig,  # Added
        device: torch.device,  # --- MODIFIED: Added device ---
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
        self.obs_norm_config = obs_norm_config  # Store config
        self.device = device  # --- MODIFIED: Use passed device ---

        self.rollout_storage = RolloutStorage(
            ppo_config.NUM_STEPS_PER_ROLLOUT,
            self.num_envs,
            self.env_config,
            self.rnn_config,
            self.device,
        )

        # --- MODIFIED: Observation Normalization Setup ---
        self.obs_rms: Dict[str, RunningMeanStd] = {}
        if self.obs_norm_config.ENABLE_OBS_NORMALIZATION:
            print("[RolloutCollector] Observation Normalization ENABLED.")
            if self.obs_norm_config.NORMALIZE_GRID:
                # Grid shape is (C, H, W)
                self.obs_rms["grid"] = RunningMeanStd(
                    shape=self.env_config.GRID_STATE_SHAPE,
                    epsilon=self.obs_norm_config.EPSILON,
                )
                print(
                    f"  - Normalizing Grid (shape: {self.env_config.GRID_STATE_SHAPE})"
                )
            if self.obs_norm_config.NORMALIZE_SHAPES:
                # Shape features are flattened in state dict
                self.obs_rms["shapes"] = RunningMeanStd(
                    shape=(self.env_config.SHAPE_STATE_DIM,),
                    epsilon=self.obs_norm_config.EPSILON,
                )
                print(
                    f"  - Normalizing Shapes (shape: {(self.env_config.SHAPE_STATE_DIM,)})"
                )
            if self.obs_norm_config.NORMALIZE_AVAILABILITY:
                self.obs_rms["shape_availability"] = RunningMeanStd(
                    shape=(self.env_config.SHAPE_AVAILABILITY_DIM,),
                    epsilon=self.obs_norm_config.EPSILON,
                )
                print(
                    f"  - Normalizing Availability (shape: {(self.env_config.SHAPE_AVAILABILITY_DIM,)})"
                )
            if self.obs_norm_config.NORMALIZE_EXPLICIT_FEATURES:
                self.obs_rms["explicit_features"] = RunningMeanStd(
                    shape=(self.env_config.EXPLICIT_FEATURES_DIM,),
                    epsilon=self.obs_norm_config.EPSILON,
                )
                print(
                    f"  - Normalizing Explicit Features (shape: {(self.env_config.EXPLICIT_FEATURES_DIM,)})"
                )
        else:
            print("[RolloutCollector] Observation Normalization DISABLED.")
        # --- END MODIFIED ---

        # CPU Buffers for current step's RAW observations and dones
        self.current_raw_obs_grid_cpu = np.zeros(
            (self.num_envs, *self.env_config.GRID_STATE_SHAPE), dtype=np.float32
        )
        self.current_raw_obs_shapes_cpu = np.zeros(
            (self.num_envs, self.env_config.SHAPE_STATE_DIM), dtype=np.float32
        )
        self.current_raw_obs_availability_cpu = np.zeros(
            (self.num_envs, self.env_config.SHAPE_AVAILABILITY_DIM), dtype=np.float32
        )
        self.current_raw_obs_explicit_features_cpu = np.zeros(
            (self.num_envs, self.env_config.EXPLICIT_FEATURES_DIM), dtype=np.float32
        )
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

        # Reset environments and populate initial observations
        self._reset_all_envs()
        # --- MODIFIED: Update RMS with initial obs before copying ---
        self._update_obs_rms(self.current_raw_obs_grid_cpu, "grid")
        self._update_obs_rms(self.current_raw_obs_shapes_cpu, "shapes")
        self._update_obs_rms(
            self.current_raw_obs_availability_cpu, "shape_availability"
        )
        self._update_obs_rms(
            self.current_raw_obs_explicit_features_cpu, "explicit_features"
        )
        self._copy_initial_observations_to_storage()  # Now copies potentially normalized obs
        # --- END MODIFIED ---

        print(f"[RolloutCollector] Initialized for {self.num_envs} environments.")

    # --- MODIFIED: Observation Normalization Methods ---
    def _update_obs_rms(self, obs_batch: np.ndarray, key: str):
        """Update the running mean/std for a given observation key if enabled."""
        if key in self.obs_rms:
            self.obs_rms[key].update(obs_batch)

    def _normalize_obs(self, obs_batch: np.ndarray, key: str) -> np.ndarray:
        """Normalize observations using running mean/std if enabled."""
        if key in self.obs_rms:
            normalized_obs = self.obs_rms[key].normalize(obs_batch)
            # Clip normalized observations
            clipped_obs = np.clip(
                normalized_obs,
                -self.obs_norm_config.OBS_CLIP,
                self.obs_norm_config.OBS_CLIP,
            )
            return clipped_obs.astype(np.float32)
        else:
            # Return raw observations if normalization for this key is disabled
            return obs_batch.astype(np.float32)

    def get_obs_rms_dict(self) -> Dict[str, RunningMeanStd]:
        """Returns the dictionary containing the RunningMeanStd instances for checkpointing."""
        return self.obs_rms

    # --- END MODIFIED ---

    def _reset_env(self, env_index: int) -> StateType:
        """Resets a single environment and returns its initial state dict."""
        try:
            state_dict = self.envs[env_index].reset()
            self.current_episode_scores[env_index] = 0.0
            self.current_episode_lengths[env_index] = 0
            self.current_episode_game_scores[env_index] = 0
            self.current_episode_lines_cleared[env_index] = 0
            return state_dict
        except Exception as e:
            print(f"ERROR resetting env {env_index}: {e}")
            traceback.print_exc()
            # Return a dummy state to avoid crashing, mark as done
            dummy_state: StateType = {
                "grid": np.zeros(self.env_config.GRID_STATE_SHAPE, dtype=np.float32),
                "shapes": np.zeros(self.env_config.SHAPE_STATE_DIM, dtype=np.float32),
                "shape_availability": np.zeros(
                    self.env_config.SHAPE_AVAILABILITY_DIM, dtype=np.float32
                ),
                "explicit_features": np.zeros(
                    self.env_config.EXPLICIT_FEATURES_DIM, dtype=np.float32
                ),
            }
            self.current_dones_cpu[env_index] = True  # Mark as done if reset failed
            return dummy_state

    def _update_raw_obs_from_state_dict(self, env_index: int, state_dict: StateType):
        """Updates the CPU RAW observation buffers for a given environment index."""
        self.current_raw_obs_grid_cpu[env_index] = state_dict["grid"]
        self.current_raw_obs_shapes_cpu[env_index] = state_dict[
            "shapes"
        ]  # Assumes already flat
        self.current_raw_obs_availability_cpu[env_index] = state_dict[
            "shape_availability"
        ]
        self.current_raw_obs_explicit_features_cpu[env_index] = state_dict[
            "explicit_features"
        ]

    def _reset_all_envs(self):
        """Resets all environments and updates initial raw observations."""
        for i in range(self.num_envs):
            initial_state = self._reset_env(i)
            self._update_raw_obs_from_state_dict(i, initial_state)
            self.current_dones_cpu[i] = False  # Reset done flag
        if self.rnn_config.USE_RNN:
            self.current_lstm_state_device = self.agent.get_initial_hidden_state(
                self.num_envs
            )

    def _copy_initial_observations_to_storage(self):
        """Normalizes (if enabled) and copies initial observations to RolloutStorage."""
        # --- MODIFIED: Normalize before copying ---
        obs_grid_norm = self._normalize_obs(self.current_raw_obs_grid_cpu, "grid")
        obs_shapes_norm = self._normalize_obs(self.current_raw_obs_shapes_cpu, "shapes")
        obs_availability_norm = self._normalize_obs(
            self.current_raw_obs_availability_cpu, "shape_availability"
        )
        obs_explicit_features_norm = self._normalize_obs(
            self.current_raw_obs_explicit_features_cpu, "explicit_features"
        )
        # --- END MODIFIED ---

        initial_obs_grid_t = torch.from_numpy(obs_grid_norm).to(
            self.rollout_storage.device
        )
        initial_obs_shapes_t = torch.from_numpy(obs_shapes_norm).to(
            self.rollout_storage.device
        )
        initial_obs_availability_t = torch.from_numpy(obs_availability_norm).to(
            self.rollout_storage.device
        )
        initial_obs_explicit_features_t = torch.from_numpy(
            obs_explicit_features_norm
        ).to(self.rollout_storage.device)
        initial_dones_t = (
            torch.from_numpy(self.current_dones_cpu)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )

        # Copy to storage at index 0
        self.rollout_storage.obs_grid[0].copy_(initial_obs_grid_t)
        self.rollout_storage.obs_shapes[0].copy_(initial_obs_shapes_t)
        self.rollout_storage.obs_availability[0].copy_(initial_obs_availability_t)
        self.rollout_storage.obs_explicit_features[0].copy_(
            initial_obs_explicit_features_t
        )
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
        """Records completed episode statistics."""
        self.episode_count += 1
        final_episode_score = (
            self.current_episode_scores[env_index] + final_reward_adjustment
        )
        final_episode_length = self.current_episode_lengths[env_index]
        final_game_score = self.current_episode_game_scores[env_index]
        final_lines_cleared = self.current_episode_lines_cleared[env_index]
        # Use the actual global step for logging
        self.stats_recorder.record_episode(
            episode_score=final_episode_score,
            episode_length=final_episode_length,
            episode_num=self.episode_count,
            global_step=current_global_step,
            game_score=final_game_score,
            lines_cleared=final_lines_cleared,
        )

    def _reset_rnn_state_for_env(self, env_index: int):
        """Resets the hidden state for a specific environment index if RNN is used."""
        if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
            # Get initial state for a single environment
            reset_h, reset_c = self.agent.get_initial_hidden_state(1)
            if reset_h is not None and reset_c is not None:
                # Ensure state is on the same device and assign to the specific env index
                reset_h = reset_h.to(self.current_lstm_state_device[0].device)
                reset_c = reset_c.to(self.current_lstm_state_device[1].device)
                self.current_lstm_state_device[0][
                    :, env_index : env_index + 1, :
                ] = reset_h
                self.current_lstm_state_device[1][
                    :, env_index : env_index + 1, :
                ] = reset_c

    def collect_one_step(self, current_global_step: int) -> int:
        """Collects one step of experience from all environments."""
        step_start_time = time.time()
        current_rollout_step = self.rollout_storage.step

        # 1. Identify active environments and get valid actions
        active_env_indices: List[int] = []
        valid_actions_list: List[Optional[List[int]]] = [None] * self.num_envs
        envs_done_pre_action: List[int] = []  # Envs that become done without an action

        for i in range(self.num_envs):
            self.envs[i]._update_timers()  # Update internal timers if any
            if self.current_dones_cpu[i]:
                continue  # Skip already done envs (will be reset later)
            if self.envs[i].is_frozen():
                continue  # Skip frozen envs (e.g., during animation)

            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                # If no valid actions, the env is effectively done
                envs_done_pre_action.append(i)
            else:
                valid_actions_list[i] = valid_actions
                active_env_indices.append(i)

        # 2. Select actions ONLY for active environments
        actions_np = np.zeros(self.num_envs, dtype=np.int64)
        log_probs_np = np.zeros(self.num_envs, dtype=np.float32)
        values_np = np.zeros(self.num_envs, dtype=np.float32)
        next_lstm_state_device = (
            self.current_lstm_state_device
        )  # Start with current state

        if active_env_indices:
            active_indices_tensor = torch.tensor(active_env_indices, dtype=torch.long)

            # --- MODIFIED: Normalize observations before feeding to agent ---
            batch_obs_grid_raw = self.current_raw_obs_grid_cpu[active_env_indices]
            batch_obs_shapes_raw = self.current_raw_obs_shapes_cpu[active_env_indices]
            batch_obs_availability_raw = self.current_raw_obs_availability_cpu[
                active_env_indices
            ]
            batch_obs_explicit_features_raw = (
                self.current_raw_obs_explicit_features_cpu[active_env_indices]
            )

            batch_obs_grid_norm = self._normalize_obs(batch_obs_grid_raw, "grid")
            batch_obs_shapes_norm = self._normalize_obs(batch_obs_shapes_raw, "shapes")
            batch_obs_availability_norm = self._normalize_obs(
                batch_obs_availability_raw, "shape_availability"
            )
            batch_obs_explicit_features_norm = self._normalize_obs(
                batch_obs_explicit_features_raw, "explicit_features"
            )

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
            # --- END MODIFIED ---

            batch_valid_actions = [valid_actions_list[i] for i in active_env_indices]

            # Select hidden state corresponding to active envs
            batch_hidden_state_device = None
            if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
                h_n, c_n = self.current_lstm_state_device
                batch_hidden_state_device = (
                    h_n[:, active_indices_tensor, :].contiguous(),
                    c_n[:, active_indices_tensor, :].contiguous(),
                )

            # Get actions, log_probs, values, and next hidden state from agent
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
                    batch_obs_explicit_features_t,
                    batch_hidden_state_device,
                    batch_valid_actions,
                )

            # Store results back into numpy arrays for all envs
            actions_np[active_env_indices] = batch_actions_t.cpu().numpy()
            log_probs_np[active_env_indices] = batch_log_probs_t.cpu().numpy()
            values_np[active_env_indices] = batch_values_t.cpu().numpy()

            # Update the full hidden state with results from active envs
            if self.rnn_config.USE_RNN and batch_next_lstm_state_device is not None:
                next_h = self.current_lstm_state_device[0].clone()
                next_c = self.current_lstm_state_device[1].clone()
                next_h[:, active_indices_tensor, :] = batch_next_lstm_state_device[0]
                next_c[:, active_indices_tensor, :] = batch_next_lstm_state_device[1]
                next_lstm_state_device = (next_h, next_c)

        # 3. Step environments, handle resets, update RAW observations
        next_raw_obs_grid_cpu = np.copy(self.current_raw_obs_grid_cpu)
        next_raw_obs_shapes_cpu = np.copy(self.current_raw_obs_shapes_cpu)
        next_raw_obs_availability_cpu = np.copy(self.current_raw_obs_availability_cpu)
        next_raw_obs_explicit_features_cpu = np.copy(
            self.current_raw_obs_explicit_features_cpu
        )
        step_rewards_np = np.zeros(self.num_envs, dtype=np.float32)
        step_dones_np = np.copy(self.current_dones_cpu)  # Start with previous dones

        for i in range(self.num_envs):
            final_reward_adj = 0.0  # Adjustment for game over penalty

            if self.current_dones_cpu[i]:  # If env was done at the start of this step
                new_state_dict = self._reset_env(i)
                self._update_raw_obs_from_state_dict(i, new_state_dict)
                self._reset_rnn_state_for_env(i)
                step_dones_np[i] = False  # Mark as not done for the *next* step
            elif (
                i in envs_done_pre_action
            ):  # Env became done because no actions were possible
                final_reward_adj = self.reward_config.PENALTY_GAME_OVER
                log_probs_np[i] = -1e9
                values_np[i] = 0.0  # Assign dummy values
                self.current_episode_lengths[
                    i
                ] += 1  # Increment length for the step leading to game over
                self._record_episode_stats(i, final_reward_adj, current_global_step)
                new_state_dict = self._reset_env(i)
                self._update_raw_obs_from_state_dict(i, new_state_dict)
                self._reset_rnn_state_for_env(i)
                step_dones_np[i] = True  # Mark as done for storage
            elif i in active_env_indices:  # Env was active, took an action
                action_to_take = actions_np[i]
                try:
                    reward, done = self.envs[i].step(action_to_take)
                    step_rewards_np[i] = reward
                    step_dones_np[i] = done
                    # Update episode trackers
                    self.current_episode_scores[i] += reward
                    self.current_episode_lengths[i] += 1
                    self.current_episode_game_scores[i] = self.envs[i].game_score
                    self.current_episode_lines_cleared[i] = self.envs[
                        i
                    ].lines_cleared_this_episode
                    if done:
                        self._record_episode_stats(
                            i, 0.0, current_global_step
                        )  # Record completed episode
                        new_state_dict = self._reset_env(i)
                        self._update_raw_obs_from_state_dict(i, new_state_dict)
                        self._reset_rnn_state_for_env(i)
                    else:
                        # Get the next state for non-done environments
                        next_state_dict = self.envs[i].get_state()
                        self._update_raw_obs_from_state_dict(i, next_state_dict)
                except Exception as e:
                    print(f"ERROR: Env {i} step failed (Action: {action_to_take}): {e}")
                    traceback.print_exc()
                    step_rewards_np[i] = (
                        self.reward_config.PENALTY_GAME_OVER
                    )  # Penalize
                    step_dones_np[i] = True  # Mark as done
                    self.current_episode_lengths[i] += 1
                    self._record_episode_stats(
                        i, 0.0, current_global_step
                    )  # Record failed episode
                    new_state_dict = self._reset_env(i)
                    self._update_raw_obs_from_state_dict(i, new_state_dict)
                    self._reset_rnn_state_for_env(i)
            # else: Environment was frozen, no action taken, state remains the same

            # Update the RAW observation buffers for the *next* step's input
            next_raw_obs_grid_cpu[i] = self.current_raw_obs_grid_cpu[i]
            next_raw_obs_shapes_cpu[i] = self.current_raw_obs_shapes_cpu[i]
            next_raw_obs_availability_cpu[i] = self.current_raw_obs_availability_cpu[i]
            next_raw_obs_explicit_features_cpu[i] = (
                self.current_raw_obs_explicit_features_cpu[i]
            )

        # 4. Update RMS with the RAW observations collected *before* this step
        # --- MODIFIED: Update RMS stats ---
        self._update_obs_rms(self.current_raw_obs_grid_cpu, "grid")
        self._update_obs_rms(self.current_raw_obs_shapes_cpu, "shapes")
        self._update_obs_rms(
            self.current_raw_obs_availability_cpu, "shape_availability"
        )
        self._update_obs_rms(
            self.current_raw_obs_explicit_features_cpu, "explicit_features"
        )
        # --- END MODIFIED ---

        # 5. Store potentially NORMALIZED results in RolloutStorage
        # --- MODIFIED: Normalize observations before storing ---
        obs_grid_norm = self._normalize_obs(self.current_raw_obs_grid_cpu, "grid")
        obs_shapes_norm = self._normalize_obs(self.current_raw_obs_shapes_cpu, "shapes")
        obs_availability_norm = self._normalize_obs(
            self.current_raw_obs_availability_cpu, "shape_availability"
        )
        obs_explicit_features_norm = self._normalize_obs(
            self.current_raw_obs_explicit_features_cpu, "explicit_features"
        )
        # --- END MODIFIED ---

        obs_grid_t = torch.from_numpy(obs_grid_norm).to(self.rollout_storage.device)
        obs_shapes_t = torch.from_numpy(obs_shapes_norm).to(self.rollout_storage.device)
        obs_availability_t = torch.from_numpy(obs_availability_norm).to(
            self.rollout_storage.device
        )
        obs_explicit_features_t = torch.from_numpy(obs_explicit_features_norm).to(
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
            torch.from_numpy(step_dones_np)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )

        lstm_state_to_store = None
        if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
            # Store the hidden state *before* this step was processed by the agent
            lstm_state_to_store = (
                self.current_lstm_state_device[0].to(self.rollout_storage.device),
                self.current_lstm_state_device[1].to(self.rollout_storage.device),
            )

        self.rollout_storage.insert(
            obs_grid_t,
            obs_shapes_t,
            obs_availability_t,
            obs_explicit_features_t,
            actions_t,
            log_probs_t,
            values_t,
            rewards_t,
            dones_t_for_storage,
            lstm_state_to_store,
        )

        # 6. Update collector's current RAW state for the *next* iteration
        self.current_raw_obs_grid_cpu = next_raw_obs_grid_cpu
        self.current_raw_obs_shapes_cpu = next_raw_obs_shapes_cpu
        self.current_raw_obs_availability_cpu = next_raw_obs_availability_cpu
        self.current_raw_obs_explicit_features_cpu = next_raw_obs_explicit_features_cpu
        self.current_dones_cpu = step_dones_np  # Update dones for the next step's check
        self.current_lstm_state_device = next_lstm_state_device  # Update LSTM state

        # 7. Record performance
        collection_time = time.time() - step_start_time
        sps = self.num_envs / max(1e-9, collection_time)
        self.stats_recorder.record_step(
            {"sps_collection": sps, "rollout_collection_time": collection_time}
        )

        return self.num_envs  # Return number of steps collected (one per env)

    def compute_advantages_for_storage(self):
        """Computes GAE advantages using the data in RolloutStorage."""
        with torch.no_grad():
            # --- MODIFIED: Normalize final observation before value prediction ---
            final_obs_grid_norm = self._normalize_obs(
                self.current_raw_obs_grid_cpu, "grid"
            )
            final_obs_shapes_norm = self._normalize_obs(
                self.current_raw_obs_shapes_cpu, "shapes"
            )
            final_obs_availability_norm = self._normalize_obs(
                self.current_raw_obs_availability_cpu, "shape_availability"
            )
            final_obs_explicit_features_norm = self._normalize_obs(
                self.current_raw_obs_explicit_features_cpu, "explicit_features"
            )

            final_obs_grid_t = torch.from_numpy(final_obs_grid_norm).to(
                self.agent.device
            )
            final_obs_shapes_t = torch.from_numpy(final_obs_shapes_norm).to(
                self.agent.device
            )
            final_obs_availability_t = torch.from_numpy(final_obs_availability_norm).to(
                self.agent.device
            )
            final_obs_explicit_features_t = torch.from_numpy(
                final_obs_explicit_features_norm
            ).to(self.agent.device)
            # --- END MODIFIED ---

            final_lstm_state = None
            if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
                final_lstm_state = (
                    self.current_lstm_state_device[0].to(self.agent.device),
                    self.current_lstm_state_device[1].to(self.agent.device),
                )

            # Add sequence dimension if needed by network for value prediction
            needs_sequence_dim = (
                self.rnn_config.USE_RNN or self.agent.transformer_config.USE_TRANSFORMER
            )
            if needs_sequence_dim:
                final_obs_grid_t = final_obs_grid_t.unsqueeze(1)
                final_obs_shapes_t = final_obs_shapes_t.unsqueeze(1)
                final_obs_availability_t = final_obs_availability_t.unsqueeze(1)
                final_obs_explicit_features_t = final_obs_explicit_features_t.unsqueeze(
                    1
                )

            # Get value of the final state (s_T)
            _, next_value, _ = self.agent.network(
                final_obs_grid_t,
                final_obs_shapes_t,
                final_obs_availability_t,
                final_obs_explicit_features_t,
                final_lstm_state,
                padding_mask=None,
            )

            # Remove sequence dim if added
            if needs_sequence_dim:
                next_value = next_value.squeeze(1)
            if next_value.ndim == 1:
                next_value = next_value.unsqueeze(-1)  # Ensure shape (B, 1)

            # Get dones corresponding to the final state
            final_dones = (
                torch.from_numpy(self.current_dones_cpu)
                .float()
                .unsqueeze(1)
                .to(self.device)
            )

        # Compute returns and advantages in storage
        self.rollout_storage.compute_returns_and_advantages(
            next_value, final_dones, self.ppo_config.GAMMA, self.ppo_config.GAE_LAMBDA
        )

    def get_episode_count(self) -> int:
        """Returns the total number of episodes completed across all environments."""
        return self.episode_count
