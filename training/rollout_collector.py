# File: training/rollout_collector.py
import time
import torch
import numpy as np
import traceback
import threading
from typing import List, Dict, Any, Tuple, Optional

from config import (
    EnvConfig,
    RewardConfig,
    TensorBoardConfig,
    PPOConfig,
    RNNConfig,
    ObsNormConfig,
)
from environment.game_state import GameState
from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from utils.running_mean_std import RunningMeanStd
from .rollout_storage import RolloutStorage

# Import new helper classes (except CollectorLogic)
from .collector_state import CollectorState

# Removed: from .collector_logic import CollectorLogic

# Keep normalization import for compute_advantages and add update_obs_rms
from .normalization import normalize_obs, update_obs_rms

# Keep env interaction import for reset_all_envs
from .env_interaction import reset_all_envs


class RolloutCollector:
    """
    Main interface for rollout collection. Initializes and coordinates state and logic handlers.
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
        obs_norm_config: ObsNormConfig,
        device: torch.device,
    ):
        # --- Import CollectorLogic here ---
        from .collector_logic import CollectorLogic

        # --- End Import ---

        self.envs = envs
        self.agent = agent
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.reward_config = reward_config
        self.tb_config = tb_config
        self.obs_norm_config = obs_norm_config
        self.device = device

        self._lock = threading.Lock()  # Keep lock if needed for main class state

        # Initialize Storage
        self.rollout_storage = RolloutStorage(
            ppo_config.NUM_STEPS_PER_ROLLOUT,
            self.num_envs,
            self.env_config,
            self.rnn_config,
            self.device,
        )

        # Initialize State Handler
        self.state = CollectorState(
            self.num_envs, self.env_config, self.rnn_config, self.agent
        )

        # Initialize Logic Handler (using the imported class)
        self.logic = CollectorLogic(
            self.envs,
            self.agent,
            self.rollout_storage,
            self.state,
            self.stats_recorder,
            self.env_config,
            self.reward_config,
            self.ppo_config,
            self.rnn_config,
            self.obs_norm_config,
        )

        # Initialize Observation Normalization (RMS instances stored in state)
        if self.obs_norm_config.ENABLE_OBS_NORMALIZATION:
            print("[RolloutCollector] Observation Normalization ENABLED.")
            if self.obs_norm_config.NORMALIZE_GRID:
                self.state.obs_rms["grid"] = RunningMeanStd(
                    shape=self.env_config.GRID_STATE_SHAPE,
                    epsilon=self.obs_norm_config.EPSILON,
                )
                print(
                    f"  - Normalizing Grid (shape: {self.env_config.GRID_STATE_SHAPE})"
                )
            if self.obs_norm_config.NORMALIZE_SHAPES:
                self.state.obs_rms["shapes"] = RunningMeanStd(
                    shape=(self.env_config.SHAPE_STATE_DIM,),
                    epsilon=self.obs_norm_config.EPSILON,
                )
                print(
                    f"  - Normalizing Shapes (shape: {(self.env_config.SHAPE_STATE_DIM,)})"
                )
            if self.obs_norm_config.NORMALIZE_AVAILABILITY:
                self.state.obs_rms["shape_availability"] = RunningMeanStd(
                    shape=(self.env_config.SHAPE_AVAILABILITY_DIM,),
                    epsilon=self.obs_norm_config.EPSILON,
                )
                print(
                    f"  - Normalizing Availability (shape: {(self.env_config.SHAPE_AVAILABILITY_DIM,)})"
                )
            if self.obs_norm_config.NORMALIZE_EXPLICIT_FEATURES:
                self.state.obs_rms["explicit_features"] = RunningMeanStd(
                    shape=(self.env_config.EXPLICIT_FEATURES_DIM,),
                    epsilon=self.obs_norm_config.EPSILON,
                )
                print(
                    f"  - Normalizing Explicit Features (shape: {(self.env_config.EXPLICIT_FEATURES_DIM,)})"
                )
        else:
            print("[RolloutCollector] Observation Normalization DISABLED.")

        # Reset environments and populate initial observations using helper
        reset_all_envs(
            self.envs,
            self.num_envs,
            self.state.current_raw_obs_grid_cpu,
            self.state.current_raw_obs_shapes_cpu,
            self.state.current_raw_obs_availability_cpu,
            self.state.current_raw_obs_explicit_features_cpu,
            self.state.current_dones_cpu,
            self.state.current_episode_scores,
            self.state.current_episode_lengths,
            self.state.current_episode_game_scores,
            self.state.current_episode_triangles_cleared,
            self.env_config,
        )
        # Initial RMS update using helper (accessing state.obs_rms)
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
        # Copy initial obs to storage
        self._copy_initial_observations_to_storage()  # Use internal method

        print(f"[RolloutCollector] Initialized for {self.num_envs} environments.")

    def get_obs_rms_dict(self) -> Dict[str, RunningMeanStd]:
        """Returns the dictionary containing the RunningMeanStd instances."""
        return self.state.obs_rms

    def _copy_initial_observations_to_storage(self):
        """Normalizes (if enabled) and copies initial observations to RolloutStorage."""
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

        initial_obs_grid_t = torch.from_numpy(obs_grid_norm)
        initial_obs_shapes_t = torch.from_numpy(obs_shapes_norm)
        initial_obs_availability_t = torch.from_numpy(obs_availability_norm)
        initial_obs_explicit_features_t = torch.from_numpy(obs_explicit_features_norm)
        initial_dones_t = (
            torch.from_numpy(self.state.current_dones_cpu).float().unsqueeze(1)
        )

        with self.rollout_storage._lock:
            self.rollout_storage.obs_grid[0].copy_(initial_obs_grid_t)
            self.rollout_storage.obs_shapes[0].copy_(initial_obs_shapes_t)
            self.rollout_storage.obs_availability[0].copy_(initial_obs_availability_t)
            self.rollout_storage.obs_explicit_features[0].copy_(
                initial_obs_explicit_features_t
            )
            self.rollout_storage.dones[0].copy_(initial_dones_t)

            if (
                self.rnn_config.USE_RNN
                and self.state.current_lstm_state_device is not None
            ):
                if (
                    self.rollout_storage.hidden_states is not None
                    and self.rollout_storage.cell_states is not None
                ):
                    self.rollout_storage.hidden_states[0].copy_(
                        self.state.current_lstm_state_device[0].cpu()
                    )
                    self.rollout_storage.cell_states[0].copy_(
                        self.state.current_lstm_state_device[1].cpu()
                    )

    def collect_one_step(self, current_global_step: int) -> int:
        """Delegates the collection of one step to the logic handler."""
        return self.logic.collect_one_step(current_global_step)

    def compute_advantages_for_storage(self):
        """Computes GAE advantages using the data in RolloutStorage."""
        final_obs_grid_norm = normalize_obs(
            self.state.current_raw_obs_grid_cpu,
            self.state.obs_rms.get("grid"),
            self.obs_norm_config.OBS_CLIP,
        )
        final_obs_shapes_norm = normalize_obs(
            self.state.current_raw_obs_shapes_cpu,
            self.state.obs_rms.get("shapes"),
            self.obs_norm_config.OBS_CLIP,
        )
        final_obs_availability_norm = normalize_obs(
            self.state.current_raw_obs_availability_cpu,
            self.state.obs_rms.get("shape_availability"),
            self.obs_norm_config.OBS_CLIP,
        )
        final_obs_explicit_features_norm = normalize_obs(
            self.state.current_raw_obs_explicit_features_cpu,
            self.state.obs_rms.get("explicit_features"),
            self.obs_norm_config.OBS_CLIP,
        )

        final_obs_grid_t = torch.from_numpy(final_obs_grid_norm).to(self.agent.device)
        final_obs_shapes_t = torch.from_numpy(final_obs_shapes_norm).to(
            self.agent.device
        )
        final_obs_availability_t = torch.from_numpy(final_obs_availability_norm).to(
            self.agent.device
        )
        final_obs_explicit_features_t = torch.from_numpy(
            final_obs_explicit_features_norm
        ).to(self.agent.device)
        final_lstm_state = self.state.current_lstm_state_device

        needs_sequence_dim = (
            self.rnn_config.USE_RNN or self.agent.transformer_config.USE_TRANSFORMER
        )
        if needs_sequence_dim:
            final_obs_grid_t = final_obs_grid_t.unsqueeze(1)
            final_obs_shapes_t = final_obs_shapes_t.unsqueeze(1)
            final_obs_availability_t = final_obs_availability_t.unsqueeze(1)
            final_obs_explicit_features_t = final_obs_explicit_features_t.unsqueeze(1)

        # --- Acquire agent lock and use no_grad for value prediction ---
        with self.agent._lock, torch.no_grad():
            self.agent.network.eval()  # Ensure eval mode
            _, next_value, _ = self.agent.network(
                final_obs_grid_t,
                final_obs_shapes_t,
                final_obs_availability_t,
                final_obs_explicit_features_t,
                final_lstm_state,
                padding_mask=None,
            )
        # --- Lock released ---

        if needs_sequence_dim:
            next_value = next_value.squeeze(1)
        if next_value.ndim == 1:
            next_value = next_value.unsqueeze(-1)

        final_dones = (
            torch.from_numpy(self.state.current_dones_cpu)
            .float()
            .unsqueeze(1)
            .to(self.agent.device)
        )

        self.rollout_storage.compute_returns_and_advantages(
            next_value, final_dones, self.ppo_config.GAMMA, self.ppo_config.GAE_LAMBDA
        )

    def get_episode_count(self) -> int:
        """Returns the total number of episodes completed."""
        return self.state.episode_count

    @property
    def global_step(self) -> int:
        if hasattr(self.stats_recorder, "aggregator") and hasattr(
            self.stats_recorder.aggregator, "storage"
        ):
            return getattr(
                self.stats_recorder.aggregator.storage, "current_global_step", 0
            )
        return 0

    @global_step.setter
    def global_step(self, value: int):
        if hasattr(self.stats_recorder, "aggregator") and hasattr(
            self.stats_recorder.aggregator, "storage"
        ):
            setattr(
                self.stats_recorder.aggregator.storage, "current_global_step", value
            )
