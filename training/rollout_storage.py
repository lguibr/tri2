# File: training/rollout_storage.py
import torch
from typing import Optional, Tuple, Dict, List, Any
import numpy as np

from config import EnvConfig, PPOConfig, RNNConfig, DEVICE


class RolloutStorage:
    """
    Stores rollout data collected from parallel environments for PPO.
    Now includes storage for shape availability and explicit features.
    """

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
        shape_availability_dim = self.env_config.SHAPE_AVAILABILITY_DIM
        # --- UPDATED: Get the potentially larger explicit features dim ---
        explicit_features_dim = self.env_config.EXPLICIT_FEATURES_DIM
        # --- END UPDATED ---

        # --- Standard PPO Data ---
        self.obs_grid = torch.zeros(
            num_steps + 1, num_envs, grid_c, grid_h, grid_w, device=self.device
        )
        self.obs_shapes = torch.zeros(
            num_steps + 1, num_envs, shape_feat_dim, device=self.device
        )
        self.obs_availability = torch.zeros(
            num_steps + 1, num_envs, shape_availability_dim, device=self.device
        )
        # --- UPDATED: Storage for explicit features with correct dimension ---
        self.obs_explicit_features = torch.zeros(
            num_steps + 1, num_envs, explicit_features_dim, device=self.device
        )
        # --- END UPDATED ---
        self.actions = torch.zeros(num_steps, num_envs, 1, device=self.device).long()
        self.log_probs = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_steps + 1, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_steps + 1, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_steps, num_envs, 1, device=self.device)

        # --- RNN Specific Data ---
        self.hidden_states = None
        self.cell_states = None
        if self.rnn_config.USE_RNN:
            lstm_hidden_size = self.rnn_config.LSTM_HIDDEN_SIZE
            num_layers = self.rnn_config.LSTM_NUM_LAYERS
            self.hidden_states = torch.zeros(
                num_steps + 1,
                num_layers,
                num_envs,
                lstm_hidden_size,
                device=self.device,
            )
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
        self.obs_availability = self.obs_availability.to(device)
        # --- UPDATED: Move explicit features ---
        self.obs_explicit_features = self.obs_explicit_features.to(device)
        # --- END UPDATED ---
        self.rewards = self.rewards.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.log_probs = self.log_probs.to(device)
        self.actions = self.actions.to(device)
        self.dones = self.dones.to(device)
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.to(device)
        if self.cell_states is not None:
            self.cell_states = self.cell_states.to(device)
        self.device = device
        print(f"[RolloutStorage] Moved tensors to {device}")

    def insert(
        self,
        obs_grid: torch.Tensor,
        obs_shapes: torch.Tensor,
        obs_availability: torch.Tensor,
        # --- UPDATED: Add explicit features ---
        obs_explicit_features: torch.Tensor,
        # --- END UPDATED ---
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Insert one step of data. Assumes input tensors are already on self.device."""
        if self.step >= self.num_steps:
            raise IndexError(
                f"RolloutStorage step index {self.step} out of bounds (max {self.num_steps-1})"
            )

        current_step_index = self.step

        self.obs_grid[current_step_index].copy_(obs_grid)
        self.obs_shapes[current_step_index].copy_(obs_shapes)
        self.obs_availability[current_step_index].copy_(obs_availability)
        # --- UPDATED: Copy explicit features ---
        self.obs_explicit_features[current_step_index].copy_(obs_explicit_features)
        # --- END UPDATED ---
        self.actions[current_step_index].copy_(action)
        self.log_probs[current_step_index].copy_(log_prob)
        self.values[current_step_index].copy_(value)
        self.rewards[current_step_index].copy_(reward)
        self.dones[current_step_index].copy_(done)

        if self.rnn_config.USE_RNN and lstm_state is not None:
            if self.hidden_states is not None and self.cell_states is not None:
                self.hidden_states[current_step_index].copy_(lstm_state[0])
                self.cell_states[current_step_index].copy_(lstm_state[1])
            else:
                print(
                    "Warning: LSTM state provided but storage tensors not initialized."
                )

        # Store the *next* observation/done state at step+1 index
        # These will be overwritten by the next insert or used in after_update/compute_returns
        next_step_index = current_step_index + 1
        if next_step_index <= self.num_steps:  # Prevent index out of bounds
            self.obs_grid[next_step_index].copy_(
                obs_grid
            )  # These are placeholders for the *next* actual obs
            self.obs_shapes[next_step_index].copy_(obs_shapes)
            self.obs_availability[next_step_index].copy_(obs_availability)
            # --- UPDATED: Copy next explicit features ---
            self.obs_explicit_features[next_step_index].copy_(obs_explicit_features)
            # --- END UPDATED ---
            self.dones[next_step_index].copy_(
                done
            )  # Store the done state corresponding to the obs at current_step_index
            if self.rnn_config.USE_RNN and lstm_state is not None:
                # The LSTM state stored at step+1 should correspond to the state *after* processing obs at step
                if self.hidden_states is not None:
                    self.hidden_states[next_step_index].copy_(
                        lstm_state[0]
                    )  # Store the *next* hidden state
                if self.cell_states is not None:
                    self.cell_states[next_step_index].copy_(
                        lstm_state[1]
                    )  # Store the *next* cell state

        self.step += 1  # Increment step *after* storing

    def after_update(self):
        """Reset storage after PPO update, keeping the last observation and state."""
        last_step_index = self.num_steps
        # Copy the actual *last* observation (which was stored at index num_steps) to index 0
        self.obs_grid[0].copy_(self.obs_grid[last_step_index])
        self.obs_shapes[0].copy_(self.obs_shapes[last_step_index])
        self.obs_availability[0].copy_(self.obs_availability[last_step_index])
        # --- UPDATED: Copy last explicit features ---
        self.obs_explicit_features[0].copy_(self.obs_explicit_features[last_step_index])
        # --- END UPDATED ---
        self.dones[0].copy_(self.dones[last_step_index])

        if self.rnn_config.USE_RNN:
            if self.hidden_states is not None:
                self.hidden_states[0].copy_(self.hidden_states[last_step_index])
            if self.cell_states is not None:
                self.cell_states[0].copy_(self.cell_states[last_step_index])
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

        effective_num_steps = self.step  # Use actual steps filled if not full
        last_step_index = effective_num_steps

        # The value of the state *after* the last step in the rollout
        self.values[last_step_index] = next_value.to(self.device)
        # The done state *after* the last step in the rollout
        self.dones[last_step_index] = final_dones.to(self.device)

        gae = 0.0
        for step in reversed(range(effective_num_steps)):
            # delta = R_t + gamma * V(s_{t+1}) * (1-done_{t+1}) - V(s_t)
            # Note: self.dones[step + 1] is the done flag *after* taking action at step `step`
            delta = (
                self.rewards[step]
                + gamma * self.values[step + 1] * (1.0 - self.dones[step + 1])
                - self.values[step]
            )
            # gae_t = delta_t + gamma * lambda * gae_{t+1} * (1-done_{t+1})
            gae = delta + gamma * gae_lambda * gae * (1.0 - self.dones[step + 1])
            # return_t = gae_t + V(s_t)
            self.returns[step] = gae + self.values[step]

    def get_data_for_update(self) -> Dict[str, Any]:
        """
        Returns collected data prepared for PPO update iterations.
        Data is returned as flattened tensors [N = T*B, ...].
        """
        effective_num_steps = self.step
        if effective_num_steps == 0:
            return {}

        # Advantages are calculated based on returns and values up to the effective step count
        advantages = (
            self.returns[:effective_num_steps] - self.values[:effective_num_steps]
        )

        num_samples = effective_num_steps * self.num_envs

        def _flatten(tensor: torch.Tensor, steps: int) -> torch.Tensor:
            return tensor[:steps].reshape(steps * self.num_envs, *tensor.shape[2:])

        initial_lstm_state = None
        if (
            self.rnn_config.USE_RNN
            and self.hidden_states is not None
            and self.cell_states is not None
        ):
            # State at the very beginning of the rollout (index 0)
            initial_lstm_state = (self.hidden_states[0], self.cell_states[0])

        data = {
            "obs_grid": _flatten(self.obs_grid, effective_num_steps),
            "obs_shapes": _flatten(self.obs_shapes, effective_num_steps),
            "obs_availability": _flatten(self.obs_availability, effective_num_steps),
            # --- UPDATED: Flatten explicit features ---
            "obs_explicit_features": _flatten(
                self.obs_explicit_features, effective_num_steps
            ),
            # --- END UPDATED ---
            "actions": _flatten(self.actions, effective_num_steps).squeeze(-1),
            "log_probs": _flatten(self.log_probs, effective_num_steps).squeeze(-1),
            "values": _flatten(self.values, effective_num_steps).squeeze(
                -1
            ),  # Values V(s_0) to V(s_{T-1})
            "returns": _flatten(self.returns, effective_num_steps).squeeze(
                -1
            ),  # Returns GAE(s_0) to GAE(s_{T-1})
            "advantages": _flatten(advantages, effective_num_steps).squeeze(-1),
            "initial_lstm_state": initial_lstm_state,  # Pass the initial LSTM state for sequence evaluation
            # Dones corresponding to obs t=0 to t=T-1 (i.e., d_0 to d_{T-1})
            # Shape [B, T] for potential RNN sequence processing needs
            "dones": self.dones[:effective_num_steps].permute(1, 0, 2).squeeze(-1),
        }
        return data
