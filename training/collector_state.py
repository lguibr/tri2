# File: training/collector_state.py
import numpy as np
import torch
from typing import Dict, Optional, Tuple

from config import EnvConfig, RNNConfig
from utils.running_mean_std import RunningMeanStd
from agent.ppo_agent import PPOAgent  # For type hint


class CollectorState:
    """Holds the state variables managed by the RolloutCollector."""

    def __init__(
        self,
        num_envs: int,
        env_config: EnvConfig,
        rnn_config: RNNConfig,
        agent: PPOAgent,
    ):
        self.num_envs = num_envs
        self.env_config = env_config
        self.rnn_config = rnn_config
        self.agent = agent  # Needed to get initial RNN state

        # CPU buffers for current environment states
        self.current_raw_obs_grid_cpu = np.zeros(
            (num_envs, *env_config.GRID_STATE_SHAPE), dtype=np.float32
        )
        self.current_raw_obs_shapes_cpu = np.zeros(
            (num_envs, env_config.SHAPE_STATE_DIM), dtype=np.float32
        )
        self.current_raw_obs_availability_cpu = np.zeros(
            (num_envs, env_config.SHAPE_AVAILABILITY_DIM), dtype=np.float32
        )
        self.current_raw_obs_explicit_features_cpu = np.zeros(
            (num_envs, env_config.EXPLICIT_FEATURES_DIM), dtype=np.float32
        )
        self.current_dones_cpu = np.zeros(num_envs, dtype=bool)

        # Episode trackers
        self.current_episode_scores = np.zeros(num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.current_episode_game_scores = np.zeros(num_envs, dtype=np.int32)
        self.current_episode_triangles_cleared = np.zeros(num_envs, dtype=np.int32)
        self.episode_count = 0  # Total completed episodes tracked here

        # RNN state (managed by collector, lives on agent's device)
        self.current_lstm_state_device: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
            None
        )
        if self.rnn_config.USE_RNN:
            self.current_lstm_state_device = self.agent.get_initial_hidden_state(
                self.num_envs
            )

        # Observation normalization RMS instances (managed by main collector)
        self.obs_rms: Dict[str, RunningMeanStd] = {}

    def reset_rnn_state(self):
        """Resets the RNN state."""
        if self.rnn_config.USE_RNN:
            self.current_lstm_state_device = self.agent.get_initial_hidden_state(
                self.num_envs
            )

    def reset_episode_trackers(self, env_index: int):
        """Resets trackers for a specific environment."""
        self.current_episode_scores[env_index] = 0.0
        self.current_episode_lengths[env_index] = 0
        self.current_episode_game_scores[env_index] = 0
        self.current_episode_triangles_cleared[env_index] = 0
