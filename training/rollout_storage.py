# File: training/rollout_storage.py
import torch
import threading  # Added for Lock
from typing import Optional, Tuple, Dict, Any

from config import EnvConfig, RNNConfig


class RolloutStorage:
    """
    Stores rollout data collected from parallel environments for PPO.
    Observations stored here are potentially normalized.
    Uses pinned memory for faster CPU -> GPU transfers.
    Includes locks for thread safety.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        env_config: EnvConfig,
        rnn_config: RNNConfig,
        device: torch.device,  # Target device for agent updates
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.env_config = env_config
        self.rnn_config = rnn_config
        self.device = device  # Target device for agent updates
        self.storage_device = torch.device("cpu")  # Store data on CPU
        self.pin_memory = self.device.type == "cuda"

        # Lock for protecting storage access from multiple threads
        self._lock = threading.Lock()

        grid_c, grid_h, grid_w = self.env_config.GRID_STATE_SHAPE
        shape_feat_dim = self.env_config.SHAPE_STATE_DIM
        shape_availability_dim = self.env_config.SHAPE_AVAILABILITY_DIM
        explicit_features_dim = self.env_config.EXPLICIT_FEATURES_DIM

        # Initialize storage tensors on CPU
        self.obs_grid = torch.zeros(
            num_steps + 1, num_envs, grid_c, grid_h, grid_w, device=self.storage_device
        )
        self.obs_shapes = torch.zeros(
            num_steps + 1, num_envs, shape_feat_dim, device=self.storage_device
        )
        self.obs_availability = torch.zeros(
            num_steps + 1, num_envs, shape_availability_dim, device=self.storage_device
        )
        self.obs_explicit_features = torch.zeros(
            num_steps + 1, num_envs, explicit_features_dim, device=self.storage_device
        )
        self.actions = torch.zeros(
            num_steps, num_envs, 1, device=self.storage_device
        ).long()
        self.log_probs = torch.zeros(num_steps, num_envs, 1, device=self.storage_device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=self.storage_device)
        self.dones = torch.zeros(num_steps + 1, num_envs, 1, device=self.storage_device)
        self.values = torch.zeros(
            num_steps + 1, num_envs, 1, device=self.storage_device
        )
        self.returns = torch.zeros(num_steps, num_envs, 1, device=self.storage_device)

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
                device=self.storage_device,
            )
            self.cell_states = torch.zeros(
                num_steps + 1,
                num_layers,
                num_envs,
                lstm_hidden_size,
                device=self.storage_device,
            )

        if self.pin_memory:
            print("[RolloutStorage] Pinning memory for faster CPU->GPU transfer.")
            self.obs_grid = self.obs_grid.pin_memory()
            self.obs_shapes = self.obs_shapes.pin_memory()
            self.obs_availability = self.obs_availability.pin_memory()
            self.obs_explicit_features = self.obs_explicit_features.pin_memory()
            self.actions = self.actions.pin_memory()
            self.log_probs = self.log_probs.pin_memory()
            self.rewards = self.rewards.pin_memory()
            self.dones = self.dones.pin_memory()
            self.values = self.values.pin_memory()
            self.returns = self.returns.pin_memory()
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states.pin_memory()
            if self.cell_states is not None:
                self.cell_states = self.cell_states.pin_memory()

        self.step = 0

    def to(self, device: torch.device):
        """Moves storage tensors to the specified device (use with caution)."""
        with self._lock:  # Protect during move
            if self.storage_device == device:
                return
            print(
                f"[RolloutStorage] WARNING: Explicitly moving storage to {device}. This might negate pinned memory benefits."
            )
            self.obs_grid = self.obs_grid.to(device)
            self.obs_shapes = self.obs_shapes.to(device)
            self.obs_availability = self.obs_availability.to(device)
            self.obs_explicit_features = self.obs_explicit_features.to(device)
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
            self.storage_device = device
            self.pin_memory = False

    def insert(
        self,
        obs_grid: torch.Tensor,
        obs_shapes: torch.Tensor,
        obs_availability: torch.Tensor,
        obs_explicit_features: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        lstm_state: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # Assumed CPU tensors
    ):
        """Insert one step of data. Thread-safe."""
        with self._lock:  # Protect storage modification
            if self.step >= self.num_steps:
                raise IndexError(
                    f"RolloutStorage step index {self.step} out of bounds (max {self.num_steps-1})"
                )

            current_step_index = self.step

            # Data is assumed to be on CPU already
            self.obs_grid[current_step_index].copy_(
                obs_grid, non_blocking=self.pin_memory
            )
            self.obs_shapes[current_step_index].copy_(
                obs_shapes, non_blocking=self.pin_memory
            )
            self.obs_availability[current_step_index].copy_(
                obs_availability, non_blocking=self.pin_memory
            )
            self.obs_explicit_features[current_step_index].copy_(
                obs_explicit_features, non_blocking=self.pin_memory
            )
            self.actions[current_step_index].copy_(action, non_blocking=self.pin_memory)
            self.log_probs[current_step_index].copy_(
                log_prob, non_blocking=self.pin_memory
            )
            self.values[current_step_index].copy_(value, non_blocking=self.pin_memory)
            self.rewards[current_step_index].copy_(reward, non_blocking=self.pin_memory)
            self.dones[current_step_index].copy_(done, non_blocking=self.pin_memory)

            if self.rnn_config.USE_RNN and lstm_state is not None:
                if self.hidden_states is not None and self.cell_states is not None:
                    self.hidden_states[current_step_index].copy_(
                        lstm_state[0], non_blocking=self.pin_memory
                    )
                    self.cell_states[current_step_index].copy_(
                        lstm_state[1], non_blocking=self.pin_memory
                    )

            # Store the *next* observation/done state at step+1 index
            next_step_index = current_step_index + 1
            self.obs_grid[next_step_index].copy_(obs_grid, non_blocking=self.pin_memory)
            self.obs_shapes[next_step_index].copy_(
                obs_shapes, non_blocking=self.pin_memory
            )
            self.obs_availability[next_step_index].copy_(
                obs_availability, non_blocking=self.pin_memory
            )
            self.obs_explicit_features[next_step_index].copy_(
                obs_explicit_features, non_blocking=self.pin_memory
            )
            self.dones[next_step_index].copy_(done, non_blocking=self.pin_memory)
            if self.rnn_config.USE_RNN and lstm_state is not None:
                if self.hidden_states is not None:
                    self.hidden_states[next_step_index].copy_(
                        lstm_state[0], non_blocking=self.pin_memory
                    )
                if self.cell_states is not None:
                    self.cell_states[next_step_index].copy_(
                        lstm_state[1], non_blocking=self.pin_memory
                    )

            self.step += 1

    def after_update(self):
        """Reset storage after PPO update, keeping the last observation and state. Thread-safe."""
        with self._lock:  # Protect storage modification
            last_step_index = self.num_steps
            self.obs_grid[0].copy_(
                self.obs_grid[last_step_index], non_blocking=self.pin_memory
            )
            self.obs_shapes[0].copy_(
                self.obs_shapes[last_step_index], non_blocking=self.pin_memory
            )
            self.obs_availability[0].copy_(
                self.obs_availability[last_step_index], non_blocking=self.pin_memory
            )
            self.obs_explicit_features[0].copy_(
                self.obs_explicit_features[last_step_index],
                non_blocking=self.pin_memory,
            )
            self.dones[0].copy_(
                self.dones[last_step_index], non_blocking=self.pin_memory
            )

            if self.rnn_config.USE_RNN:
                if self.hidden_states is not None:
                    self.hidden_states[0].copy_(
                        self.hidden_states[last_step_index],
                        non_blocking=self.pin_memory,
                    )
                if self.cell_states is not None:
                    self.cell_states[0].copy_(
                        self.cell_states[last_step_index], non_blocking=self.pin_memory
                    )
            self.step = 0

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,  # Value of state s_T (on agent device)
        final_dones: torch.Tensor,  # Done state after action a_{T-1} (on agent device)
        gamma: float,
        gae_lambda: float,
    ):
        """Computes returns and GAE advantages. Thread-safe."""
        with self._lock:  # Protect storage modification
            if self.step != self.num_steps:
                print(
                    f"Warning: Computing returns before storage is full (step={self.step}, num_steps={self.num_steps})"
                )

            effective_num_steps = self.step
            last_step_index = effective_num_steps

            # Move inputs to CPU for calculation with storage tensors
            next_value_cpu = next_value.cpu()
            final_dones_cpu = final_dones.cpu()

            self.values[last_step_index].copy_(
                next_value_cpu, non_blocking=self.pin_memory
            )
            self.dones[last_step_index].copy_(
                final_dones_cpu, non_blocking=self.pin_memory
            )

            gae = 0.0
            for step in reversed(range(effective_num_steps)):
                delta = (
                    self.rewards[step]
                    + gamma * self.values[step + 1] * (1.0 - self.dones[step + 1])
                    - self.values[step]
                )
                gae = delta + gamma * gae_lambda * gae * (1.0 - self.dones[step + 1])
                self.returns[step] = gae + self.values[step]

    def get_data_for_update(self) -> Dict[str, Any]:
        """
        Returns collected data prepared for PPO update iterations. Thread-safe.
        Data is returned as flattened tensors [N = T*B, ...] on CPU (potentially pinned).
        """
        with self._lock:  # Protect storage access
            effective_num_steps = self.step
            if effective_num_steps == 0:
                return {}

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
                # Return the initial state for the sequence (on CPU)
                initial_lstm_state = (
                    self.hidden_states[0].clone(),
                    self.cell_states[0].clone(),
                )

            data = {
                "obs_grid": _flatten(self.obs_grid, effective_num_steps),
                "obs_shapes": _flatten(self.obs_shapes, effective_num_steps),
                "obs_availability": _flatten(
                    self.obs_availability, effective_num_steps
                ),
                "obs_explicit_features": _flatten(
                    self.obs_explicit_features, effective_num_steps
                ),
                "actions": _flatten(self.actions, effective_num_steps).squeeze(-1),
                "log_probs": _flatten(self.log_probs, effective_num_steps).squeeze(-1),
                "values": _flatten(self.values, effective_num_steps).squeeze(-1),
                "returns": _flatten(self.returns, effective_num_steps).squeeze(-1),
                "advantages": _flatten(advantages, effective_num_steps).squeeze(-1),
                "initial_lstm_state": initial_lstm_state,  # On CPU
                "dones": self.dones[:effective_num_steps]
                .permute(1, 0, 2)
                .squeeze(-1),  # Shape (B, T)
            }
            return data
