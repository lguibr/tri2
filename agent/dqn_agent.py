# File: agent/dqn_agent.py
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

        self.optimizer = optim.AdamW(
            self.online_net.parameters(),
            lr=dqn_config.LEARNING_RATE,
            eps=dqn_config.ADAM_EPS,
        )
        # Use Huber loss (SmoothL1Loss) which is robust to outliers
        self.loss_fn = nn.SmoothL1Loss(reduction="none", beta=1.0)

        # <<< NEW >>> Store last batch's avg max Q for stats
        self._last_avg_max_q: float = 0.0

        print(f"[DQNAgent] Online Network: {type(self.online_net).__name__}")
        print(f"[DQNAgent] Using Double DQN: {self.use_double_dqn}")
        print(f"[DQNAgent] Using Dueling: {config.USE_DUELING}")
        print(
            f"[DQNAgent] Optimizer: AdamW (LR={dqn_config.LEARNING_RATE}, EPS={dqn_config.ADAM_EPS})"
        )

    @torch.no_grad()
    def select_action(
        self, state: StateType, epsilon: float, valid_actions: List[ActionType]
    ) -> ActionType:
        """Selects action using epsilon-greedy, ensuring validity."""
        if not valid_actions:
            # If no valid actions, the game should be over or frozen.
            # The trainer should ideally handle this. Return a dummy action (e.g., 0).
            # print("Warning: DQNAgent.select_action called with no valid actions.")
            return 0

        if random.random() < epsilon:
            # Exploration: Choose a random action from the *valid* list
            return random.choice(valid_actions)
        else:
            # Exploitation: Choose the best action according to the online network
            state_t = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(
                0
            )  # Add batch dimension
            self.online_net.eval()  # Set to evaluation mode for inference
            q_values = self.online_net(state_t)[0]  # Get Q-values for this state
            self.online_net.train()  # Set back to training mode

            # Masking: Set Q-values of invalid actions to negative infinity
            q_values_masked = torch.full_like(q_values, -float("inf"))
            # Ensure valid_actions are tensor indices on the correct device
            valid_action_indices = torch.tensor(
                valid_actions, dtype=torch.long, device=self.device
            )
            # Assign the calculated Q-values only to the valid action indices
            q_values_masked[valid_action_indices] = q_values[valid_action_indices]

            # Find the action index with the maximum Q-value among the valid ones
            best_action = torch.argmax(q_values_masked).item()

            # Safety check: Ensure the chosen best action is indeed in the valid list
            # This should always be true due to masking, but helps catch errors.
            # if best_action not in valid_actions:
            #     print(f"Warning: Argmax selected invalid action {best_action}. Choosing random valid action.")
            #     return random.choice(valid_actions)

            return best_action

    def _np_batch_to_tensor(
        self, batch: Union[NumpyBatch, NumpyNStepBatch], is_n_step: bool
    ) -> Union[TensorBatch, TensorNStepBatch]:
        """Converts numpy batch tuple to tensor tuple on the correct device."""
        if is_n_step:
            # N-step batch: (s, a, rn, nsn, dn, gamma_n)
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
            ).unsqueeze(
                1
            )  # This is gamma^k
            return states, actions, rewards, next_states, dones, discounts
        else:
            # 1-step batch: (s, a, r, ns, d)
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
        Input batch contains numpy arrays. Returns (scalar loss, TD errors tensor).
        Also updates self._last_avg_max_q.
        """
        # 1. Convert numpy batch to tensors
        tensor_batch = self._np_batch_to_tensor(batch, is_n_step)

        if is_n_step:
            states, actions, rewards, next_states, dones, discounts = tensor_batch
        else:
            states, actions, rewards, next_states, dones = tensor_batch
            # For 1-step, the discount for the next state Q-value is gamma^1
            discounts = torch.full_like(rewards, self.gamma, device=self.device)

        # Convert Importance Sampling weights if provided (for PER)
        is_weights_t = None
        if is_weights is not None:
            is_weights_t = torch.tensor(
                is_weights, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

        # --- 2. Calculate Target Q-values (Q_target) ---
        with torch.no_grad():  # Important: Target network calculations don't need gradients
            if self.use_double_dqn:
                # Select best action using the *online* network
                online_next_q = self.online_net(next_states)
                best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
                # Evaluate the Q-value of that action using the *target* network
                target_next_q_values = self.target_net(next_states).gather(
                    1, best_next_actions
                )
            else:
                # Standard DQN: Select and evaluate using the target network
                target_next_q_values = self.target_net(next_states).max(
                    dim=1, keepdim=True
                )[
                    0
                ]  # [0] gets the values, not the indices

            # Target Q = R + discount * Q_target(S', a*) * (1 - done)
            # 'discounts' is gamma^k (from N-step buffer) or gamma (for 1-step)
            # 'rewards' is R_n (N-step reward) or R (1-step reward)
            # 'dones' is done_n (N-step done) or done (1-step done)
            target_q = rewards + discounts * target_next_q_values * (1.0 - dones)

        # --- 3. Calculate Current Q-values (Q_online) ---
        # Get Q-values from the online network for the *actions actually taken*
        current_q = self.online_net(states).gather(1, actions)

        # --- 4. Calculate Loss ---
        # TD Error = Target Q - Current Q
        td_error = target_q - current_q

        # Huber Loss element-wise
        elementwise_loss = self.loss_fn(current_q, target_q)

        # Apply Importance Sampling weights if using PER
        if is_weights_t is not None:
            # Weighted mean loss
            loss = (is_weights_t * elementwise_loss).mean()
        else:
            # Simple mean loss
            loss = elementwise_loss.mean()

        # --- 5. Update Stats ---
        # Calculate avg max Q value predicted by the online network for the *current* states
        with torch.no_grad():
            self._last_avg_max_q = self.online_net(states).max(dim=1)[0].mean().item()

        # Return the scalar loss and the absolute TD errors (for PER updates)
        # Detach TD errors as they are only used for buffer updates, not backprop
        return loss, td_error.abs().detach()

    def update(self, loss: torch.Tensor) -> float:
        """Performs optimization step and returns gradient norm."""
        self.optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients

        # Optional: Gradient Clipping
        grad_norm = 0.0
        if self.gradient_clip_norm > 0:
            # Clip gradients by norm and return the norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.online_net.parameters(), max_norm=self.gradient_clip_norm
            ).item()
        else:
            # Calculate L2 norm manually if not clipping (for logging)
            for p in self.online_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = math.sqrt(grad_norm)

        self.optimizer.step()  # Update network weights
        return grad_norm  # Return gradient norm for logging

    # <<< NEW >>> Method to get the last calculated avg max Q
    def get_last_avg_max_q(self) -> float:
        return self._last_avg_max_q

    def update_target_network(self):
        """Copies weights from online to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_state_dict(self) -> AgentStateDict:
        """Returns state dictionary for saving."""
        return {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),  # Save target too? Optional but safer.
            "optimizer_state_dict": self.optimizer.state_dict(),
            # Add other agent state if needed (e.g., training step for LR schedule)
        }

    def load_state_dict(self, state_dict: AgentStateDict):
        """Loads state dictionary."""
        self.online_net.load_state_dict(state_dict["online_net_state_dict"])
        # Load target net separately if saved, otherwise copy from loaded online net
        if "target_net_state_dict" in state_dict:
            self.target_net.load_state_dict(state_dict["target_net_state_dict"])
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())  # Fallback
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        # Ensure nets are in correct mode after loading
        self.online_net.train()
        self.target_net.eval()
