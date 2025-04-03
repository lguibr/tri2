import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
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


class DQNAgent:
    """DQN Agent: Creates models, selects actions, computes loss, updates model."""

    def __init__(
        self, config: ModelConfig, dqn_config: DQNConfig, env_config: EnvConfig
    ):
        print("[DQNAgent] Initializing...")
        self.device = DEVICE
        self.action_dim = env_config.ACTION_DIM
        self.gamma = dqn_config.GAMMA
        self.use_double_dqn = config.USE_DOUBLE_DQN
        self.gradient_clip_norm = dqn_config.GRADIENT_CLIP_NORM

        self.online_net = create_network(
            env_config.STATE_DIM, env_config.ACTION_DIM, config
        ).to(self.device)
        self.target_net = create_network(
            env_config.STATE_DIM, env_config.ACTION_DIM, config
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=dqn_config.LEARNING_RATE,
            eps=dqn_config.ADAM_EPS,
        )
        self.loss_fn = nn.MSELoss(reduction="none")  # For PER weighting

        print(f"[DQNAgent] Online Network: {type(self.online_net).__name__}")
        print(f"[DQNAgent] Using Double DQN: {self.use_double_dqn}")

    @torch.no_grad()
    def select_action(
        self, state: StateType, epsilon: float, valid_actions: List[ActionType]
    ) -> ActionType:
        """Selects action using epsilon-greedy, ensuring validity."""
        if not valid_actions:
            return 0  # Should be handled by trainer, but return default

        if random.random() < epsilon:
            return random.choice(valid_actions)
        else:
            state_t = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            self.online_net.eval()
            q_values = self.online_net(state_t)[0]
            self.online_net.train()

            # Filter Q-values for valid actions
            # Create tensor of -inf for invalid actions
            q_values_masked = torch.full_like(q_values, -float("inf"))
            valid_action_indices = torch.tensor(
                valid_actions, dtype=torch.long, device=self.device
            )
            q_values_masked[valid_action_indices] = q_values[valid_action_indices]

            # Find action with max Q-value among valid actions
            best_action = torch.argmax(q_values_masked).item()
            # Ensure the selected action is actually in the valid list (should be)
            return (
                best_action
                if best_action in valid_actions
                else random.choice(valid_actions)
            )

    def _np_batch_to_tensor(
        self, batch: Union[NumpyBatch, NumpyNStepBatch], is_n_step: bool
    ) -> Union[TensorBatch, TensorNStepBatch]:
        """Converts numpy batch tuple to tensor tuple on the correct device."""
        if is_n_step:
            s, a, rn, nsn, dn, gamma_n = batch
            states = torch.tensor(s, dtype=torch.float32, device=self.device)
            actions = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.tensor(
                rn, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            next_states = torch.tensor(nsn, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dn, dtype=torch.float32, device=self.device).unsqueeze(
                1
            )
            discounts = torch.tensor(
                gamma_n, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            return states, actions, rewards, next_states, dones, discounts
        else:
            s, a, r, ns, d = batch
            states = torch.tensor(s, dtype=torch.float32, device=self.device)
            actions = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.tensor(
                r, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            next_states = torch.tensor(ns, dtype=torch.float32, device=self.device)
            dones = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(
                1
            )
            return states, actions, rewards, next_states, dones

    def compute_loss(
        self,
        batch: Union[NumpyBatch, NumpyNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes DQN loss (1-step or N-step, PER weighted).
        Input batch contains numpy arrays. Returns scalar loss and TD errors tensor.
        """
        # Convert numpy batch to tensors
        tensor_batch = self._np_batch_to_tensor(batch, is_n_step)

        if is_n_step:
            states, actions, rewards, next_states, dones, discounts = tensor_batch
        else:
            states, actions, rewards, next_states, dones = tensor_batch
            # Create gamma tensor for 1-step (discounts for next state Q)
            discounts = torch.full_like(rewards, self.gamma, device=self.device)

        # Convert IS weights if provided
        is_weights_t = None
        if is_weights is not None:
            is_weights_t = torch.tensor(
                is_weights, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

        # --- Calculate Target Q-values ---
        with torch.no_grad():
            # next_states is ns_n if is_n_step=True
            if self.use_double_dqn:
                online_next_q = self.online_net(next_states)
                best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
                target_next_q_values = self.target_net(next_states).gather(
                    1, best_next_actions
                )
            else:
                target_next_q_values = self.target_net(next_states).max(
                    dim=1, keepdim=True
                )[0]

            # Target = R_n + discount * Q_target(s_n, a*) * (1 - done_n)
            # 'discounts' holds gamma^N (or gamma^k if done early) for N-step, or gamma for 1-step.
            # 'rewards' holds R_n for N-step, or R for 1-step.
            # 'dones' holds done_n for N-step, or done for 1-step.
            # If done=1, the Q-value term is zeroed out.
            target_q = rewards + discounts * target_next_q_values * (1.0 - dones)

        # --- Calculate Current Q-values ---
        current_q = self.online_net(states).gather(1, actions)

        # --- Calculate Loss ---
        td_error = target_q - current_q
        elementwise_loss = self.loss_fn(current_q, target_q)

        if is_weights_t is not None:
            loss = (is_weights_t * elementwise_loss).mean()
        else:
            loss = elementwise_loss.mean()

        return loss, td_error.abs().detach()

    def update(self, loss: torch.Tensor) -> float:
        """Performs optimization step and returns gradient norm."""
        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = 0.0
        if self.gradient_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.online_net.parameters(), max_norm=self.gradient_clip_norm
            ).item()
        else:
            for p in self.online_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = math.sqrt(grad_norm)

        self.optimizer.step()
        return grad_norm

    def update_target_network(self):
        """Copies weights from online to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_state_dict(self) -> AgentStateDict:
        """Returns state dictionary for saving."""
        return {
            "online_net_state_dict": self.online_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: AgentStateDict):
        """Loads state dictionary."""
        self.online_net.load_state_dict(state_dict["online_net_state_dict"])
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.online_net.train()
        self.target_net.eval()
