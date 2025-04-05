# File: training/rollout_storage.py
import torch
from typing import Optional, Tuple, Dict, List, Any
import numpy as np

from config import EnvConfig, PPOConfig, RNNConfig, DEVICE


class RolloutStorage:
    """Stores rollout data collected from parallel environments for PPO."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        env_config: EnvConfig,
        rnn_config: RNNConfig,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.env_config = env_config
        self.rnn_config = rnn_config
        self.device = device

        grid_c, grid_h, grid_w = self.env_config.GRID_STATE_SHAPE
        shape_feat_dim = self.env_config.SHAPE_STATE_DIM

        # --- Standard PPO Data ---
        self.obs_grid = torch.zeros(
            num_steps + 1, num_envs, grid_c, grid_h, grid_w, device=self.device
        )
        self.obs_shapes = torch.zeros(
            num_steps + 1, num_envs, shape_feat_dim, device=self.device
        )
        self.actions = torch.zeros(num_steps, num_envs, 1, device=self.device).long()
        self.log_probs = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_steps + 1, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_steps + 1, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_steps, num_envs, 1, device=self.device)

        # --- RNN Specific Data ---
        self.hidden_states = None
        self.cell_states = None  # NEW: For LSTM cell state 'c'
        if self.rnn_config.USE_RNN:
            lstm_hidden_size = self.rnn_config.LSTM_HIDDEN_SIZE
            num_layers = self.rnn_config.LSTM_NUM_LAYERS
            # hidden_states[t] (h_t) corresponds to state s_t
            self.hidden_states = torch.zeros(
                num_steps + 1,
                num_layers,
                num_envs,
                lstm_hidden_size,
                device=self.device,
            )
            # NEW: cell_states[t] (c_t) corresponds to state s_t
            self.cell_states = torch.zeros(
                num_steps + 1,
                num_layers,
                num_envs,
                lstm_hidden_size,
                device=self.device,
            )

        self.step = 0

    def to(self, device: torch.device):
        """Move storage tensors to the specified device."""
        if self.device == device:
            return
        self.obs_grid = self.obs_grid.to(device)
        self.obs_shapes = self.obs_shapes.to(device)
        self.rewards = self.rewards.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.log_probs = self.log_probs.to(device)
        self.actions = self.actions.to(device)
        self.dones = self.dones.to(device)
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.to(device)
        if self.cell_states is not None:  # NEW
            self.cell_states = self.cell_states.to(device)
        self.device = device
        print(f"[RolloutStorage] Moved tensors to {device}")

    def insert(
        self,
        obs_grid: torch.Tensor,
        obs_shapes: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        # MODIFIED: Accept tuple (h, c)
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Insert one step of data. Assumes input tensors are already on self.device."""
        if self.step >= self.num_steps:
            raise IndexError(
                f"RolloutStorage step index {self.step} out of bounds (max {self.num_steps-1})"
            )

        self.obs_grid[self.step].copy_(obs_grid)
        self.obs_shapes[self.step].copy_(obs_shapes)
        self.actions[self.step].copy_(action)
        self.log_probs[self.step].copy_(log_prob)
        self.values[self.step].copy_(value)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)

        # MODIFIED: Store both h and c states
        if self.rnn_config.USE_RNN and lstm_state is not None:
            if self.hidden_states is not None and self.cell_states is not None:
                self.hidden_states[self.step].copy_(lstm_state[0])  # h_t
                self.cell_states[self.step].copy_(lstm_state[1])  # c_t
            else:
                print(
                    "Warning: LSTM state provided but storage tensors not initialized."
                )

        self.step += 1

    def after_update(self):
        """Reset storage after PPO update, keeping the last observation and state."""
        self.obs_grid[0].copy_(self.obs_grid[self.num_steps])
        self.obs_shapes[0].copy_(self.obs_shapes[self.num_steps])
        self.dones[0].copy_(self.dones[self.num_steps])

        # MODIFIED: Copy both h and c states
        if self.rnn_config.USE_RNN:
            if self.hidden_states is not None:
                self.hidden_states[0].copy_(self.hidden_states[self.num_steps])
            if self.cell_states is not None:
                self.cell_states[0].copy_(self.cell_states[self.num_steps])
        self.step = 0

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,
        final_dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """Computes returns and GAE advantages. Assumes inputs are on self.device."""
        if self.step != self.num_steps:
            print(
                f"Warning: Computing returns before storage is full (step={self.step}, num_steps={self.num_steps})"
            )

        self.values[self.num_steps] = next_value.to(self.device)
        self.dones[self.num_steps] = final_dones.to(self.device)

        gae = 0.0
        for step in reversed(range(self.num_steps)):
            delta = (
                self.rewards[step]
                + gamma * self.values[step + 1] * (1.0 - self.dones[step + 1])
                - self.values[step]
            )
            gae = delta + gamma * gae_lambda * gae * (1.0 - self.dones[step + 1])
            self.returns[step] = gae + self.values[step]

    def get_data_for_update(self) -> Dict[str, Any]:
        """
        Returns collected data prepared for PPO update iterations.
        Data is returned as flattened tensors [N = T*B, ...].
        """
        advantages = self.returns[: self.num_steps] - self.values[: self.num_steps]
        num_samples = self.num_steps * self.num_envs

        def _flatten(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(num_samples, *tensor.shape[2:])

        # MODIFIED: Return initial hidden state tuple (h_0, c_0)
        initial_lstm_state = None
        if (
            self.rnn_config.USE_RNN
            and self.hidden_states is not None
            and self.cell_states is not None
        ):
            initial_lstm_state = (self.hidden_states[0], self.cell_states[0])

        data = {
            "obs_grid": _flatten(self.obs_grid[: self.num_steps]),
            "obs_shapes": _flatten(self.obs_shapes[: self.num_steps]),
            "actions": _flatten(self.actions).squeeze(-1),
            "log_probs": _flatten(self.log_probs).squeeze(-1),
            "values": _flatten(self.values[: self.num_steps]).squeeze(-1),
            "returns": _flatten(self.returns[: self.num_steps]).squeeze(-1),
            "advantages": _flatten(advantages).squeeze(-1),
            # MODIFIED: Key now holds the tuple (h, c)
            "initial_lstm_state": initial_lstm_state,
            "dones": self.dones[: self.num_steps].permute(1, 0, 2).squeeze(-1),
        }
        return data
