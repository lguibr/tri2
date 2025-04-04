# File: agent/dqn_agent.py
# (Largely unchanged structurally, Noisy Nets were already core)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import traceback
from typing import Tuple, List, Dict, Any, Optional, Union

from config import EnvConfig, ModelConfig, DQNConfig, DEVICE
from agent.model_factory import create_network
from utils.types import (
    StateType,
    ActionType,
    NumpyBatch,
    NumpyNStepBatch,
    AgentStateDict,
    TensorBatch,
    TensorNStepBatch,
)
from utils.helpers import ensure_numpy
from agent.networks.noisy_layer import NoisyLinear  # Keep for type checking info


class DQNAgent:
    """DQN Agent using Noisy Nets for exploration."""

    def __init__(
        self,
        config: ModelConfig,
        dqn_config: DQNConfig,
        env_config: EnvConfig,
    ):
        print("[DQNAgent] Initializing...")
        self.device = DEVICE
        self.action_dim = env_config.ACTION_DIM
        self.gamma = dqn_config.GAMMA
        self.use_double_dqn = dqn_config.USE_DOUBLE_DQN
        self.gradient_clip_norm = dqn_config.GRADIENT_CLIP_NORM
        self.use_noisy_nets = dqn_config.USE_NOISY_NETS
        self.use_dueling = dqn_config.USE_DUELING

        if not self.use_noisy_nets:
            # This shouldn't happen based on config, but warn if it does
            print("WARNING: DQNConfig.USE_NOISY_NETS is False. Agent expects True.")

        self.online_net = create_network(
            env_config.STATE_DIM, env_config.ACTION_DIM, config, dqn_config
        ).to(self.device)
        self.target_net = create_network(
            env_config.STATE_DIM, env_config.ACTION_DIM, config, dqn_config
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target net always in eval mode

        self.optimizer = optim.AdamW(
            self.online_net.parameters(),
            lr=dqn_config.LEARNING_RATE,
            eps=dqn_config.ADAM_EPS,
            weight_decay=1e-5,  # Example weight decay
        )
        # Huber loss is generally robust for Q-learning
        self.loss_fn = nn.SmoothL1Loss(
            reduction="none", beta=1.0
        )  # Use 'none' for PER weights

        self._last_avg_max_q: float = 0.0

        print(f"[DQNAgent] Online Network: {type(self.online_net).__name__}")
        print(f"[DQNAgent] Using Double DQN: {self.use_double_dqn}")
        print(f"[DQNAgent] Using Dueling: {self.use_dueling}")
        print(f"[DQNAgent] Using Noisy Nets: {self.use_noisy_nets}")
        print(
            f"[DQNAgent] Optimizer: AdamW (LR={dqn_config.LEARNING_RATE}, EPS={dqn_config.ADAM_EPS})"
        )
        total_params = sum(
            p.numel() for p in self.online_net.parameters() if p.requires_grad
        )
        print(f"[DQNAgent] Trainable Parameters: {total_params / 1e6:.2f} M")

    @torch.no_grad()
    def select_action(
        self,
        state: StateType,
        epsilon: float,  # Epsilon is unused but kept for potential API compatibility
        valid_actions: List[ActionType],
    ) -> ActionType:
        """Selects action using the noisy online network (greedy w.r.t mean weights)."""
        if not valid_actions:
            # print("Warning: select_action called with no valid actions.")
            return 0  # Return a default action index

        state_np = ensure_numpy(state)
        state_t = torch.tensor(
            state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Use online network in eval mode for action selection.
        # NoisyLinear layers use mean weights in eval mode.
        self.online_net.eval()
        q_values = self.online_net(state_t)[0]  # Q-values for the single state

        # Mask invalid actions
        q_values_masked = torch.full_like(q_values, -float("inf"))
        valid_action_indices = torch.tensor(
            valid_actions, dtype=torch.long, device=self.device
        )
        q_values_masked[valid_action_indices] = q_values[valid_action_indices]

        best_action = torch.argmax(q_values_masked).item()

        # Note: online_net is set back to train() mode within compute_loss/update methods

        return best_action

    def _np_batch_to_tensor(
        self, batch: Union[NumpyBatch, NumpyNStepBatch], is_n_step: bool
    ) -> Union[TensorBatch, TensorNStepBatch]:
        """Converts a numpy batch (1-step or N-step) to tensors on the correct device."""
        if is_n_step:
            states, actions, rewards, next_states, dones, discounts = batch
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(
                actions, dtype=torch.long, device=self.device
            ).unsqueeze(1)
            rewards_t = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            next_states_t = torch.tensor(
                next_states, dtype=torch.float32, device=self.device
            )
            dones_t = torch.tensor(
                dones, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            discounts_t = torch.tensor(
                discounts, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            return states_t, actions_t, rewards_t, next_states_t, dones_t, discounts_t
        else:
            states, actions, rewards, next_states, dones = batch
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(
                actions, dtype=torch.long, device=self.device
            ).unsqueeze(1)
            rewards_t = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            next_states_t = torch.tensor(
                next_states, dtype=torch.float32, device=self.device
            )
            dones_t = torch.tensor(
                dones, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            return states_t, actions_t, rewards_t, next_states_t, dones_t

    def compute_loss(
        self,
        batch: Union[NumpyBatch, NumpyNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray] = None,  # PER Importance Sampling weights
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes DQN loss (Huber Loss), handles N-step and PER weights."""

        # Convert Batch to Tensors
        tensor_batch = self._np_batch_to_tensor(batch, is_n_step)
        if is_n_step:
            states, actions, rewards, next_states, dones, discounts = tensor_batch
        else:
            states, actions, rewards, next_states, dones = tensor_batch
            discounts = torch.full_like(
                rewards, self.gamma, device=self.device
            )  # gamma^1

        is_weights_t = None
        if is_weights is not None:
            is_weights_t = torch.tensor(
                is_weights, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

        # Calculate Target Q-values (using Double DQN logic)
        with torch.no_grad():
            # Target net is already in eval mode
            # Select best actions for next states using the *online* network (eval mode for consistency)
            self.online_net.eval()
            online_next_q = self.online_net(next_states)
            best_next_actions = online_next_q.argmax(dim=1, keepdim=True)

            # Get Q-values for these best actions using the *target* network
            target_next_q_values = self.target_net(next_states).gather(
                1, best_next_actions
            )

            # Calculate the TD target: R + gamma^N * Q_target(s', a') * (1 - done)
            target_q = rewards + discounts * target_next_q_values * (1.0 - dones)

        # Calculate Current Q-values (train mode for gradients and noise)
        self.online_net.train()  # Ensure train mode for Noisy Nets and gradients
        current_q = self.online_net(states).gather(1, actions)

        # Calculate Loss
        td_error = target_q - current_q
        elementwise_loss = self.loss_fn(current_q, target_q)

        # Apply PER weights
        loss = (
            (is_weights_t * elementwise_loss).mean()
            if is_weights_t is not None
            else elementwise_loss.mean()
        )

        # Update Stats (average max Q for logging - use eval for consistency)
        with torch.no_grad():
            self.online_net.eval()
            self._last_avg_max_q = self.online_net(states).max(dim=1)[0].mean().item()
            self.online_net.train()  # Switch back immediately if needed elsewhere? Update handles it.

        return loss, td_error.abs().detach()  # Return abs TD error for PER

    def update(self, loss: torch.Tensor) -> Optional[float]:
        """Performs one optimization step and returns gradient norm."""
        grad_norm = None
        try:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient Clipping
            if self.gradient_clip_norm > 0:
                # Ensure online_net is in train mode before clipping/stepping
                self.online_net.train()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.online_net.parameters(), max_norm=self.gradient_clip_norm
                ).item()

            self.optimizer.step()
        except Exception as e:
            print(f"ERROR during agent update/optimizer step: {e}")
            traceback.print_exc()
            # Return None or re-raise? Returning None indicates failure.
            return None

        return grad_norm

    def get_last_avg_max_q(self) -> float:
        """Returns the average max Q value computed during the last loss calculation."""
        return self._last_avg_max_q

    def update_target_network(self):
        """Copies weights from the online network to the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def get_state_dict(self) -> AgentStateDict:
        """Returns the agent's state for saving."""
        return {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: AgentStateDict):
        """Loads the agent's state from a dictionary."""
        self.online_net.load_state_dict(state_dict["online_net_state_dict"])

        if "target_net_state_dict" in state_dict:
            self.target_net.load_state_dict(state_dict["target_net_state_dict"])
        else:
            print(
                "Warning: Target network state missing in checkpoint, copying from online."
            )
            self.target_net.load_state_dict(self.online_net.state_dict())

        try:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        except ValueError as e:
            print(
                f"Warning: Optimizer state mismatch ({e}). Resetting optimizer state."
            )
            # Reset optimizer if loading fails (e.g., model change)
            self.optimizer = optim.AdamW(
                self.online_net.parameters(),
                lr=DQNConfig.LEARNING_RATE,  # Use current config LR
                eps=DQNConfig.ADAM_EPS,
                weight_decay=1e-5,
            )
        except Exception as e:
            print(
                f"Warning: Error loading optimizer state: {e}. Resetting optimizer state."
            )
            self.optimizer = optim.AdamW(
                self.online_net.parameters(),
                lr=DQNConfig.LEARNING_RATE,
                eps=DQNConfig.ADAM_EPS,
                weight_decay=1e-5,
            )

        # Ensure networks are in correct mode after loading
        self.online_net.train()
        self.target_net.eval()
