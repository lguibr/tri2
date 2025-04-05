# File: agent/loss_calculator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Dict, List  # Added List, Dict
from config import DQNConfig, TensorBoardConfig, EnvConfig
from utils.types import (
    TensorBatch,
    TensorNStepBatch,
    NumpyBatch,
    NumpyNStepBatch,
)  # Added Numpy types
from utils.helpers import ensure_numpy
from .agent_utils import np_batch_to_tensor


class LossCalculator:
    """Handles loss calculation for the DQNAgent."""

    def __init__(
        self,
        online_net: nn.Module,
        target_net: nn.Module,
        env_config: EnvConfig,
        dqn_config: DQNConfig,
        tb_config: TensorBoardConfig,
        device: torch.device,
    ):
        self.online_net = online_net
        self.target_net = target_net
        self.env_config = env_config
        self.dqn_config = dqn_config
        self.tb_config = tb_config
        self.device = device
        self.gamma = (
            dqn_config.GAMMA
        )  # Still keep gamma for potential 1-step use or fallbacks
        self.use_distributional = dqn_config.USE_DISTRIBUTIONAL
        self.num_atoms = dqn_config.NUM_ATOMS
        self.v_min = dqn_config.V_MIN
        self.v_max = dqn_config.V_MAX

        if self.use_distributional:
            if self.num_atoms <= 1:
                raise ValueError("NUM_ATOMS must be >= 2 for Distributional RL")
            self.support = torch.linspace(
                self.v_min, self.v_max, self.num_atoms, device=self.device
            )
            self.delta_z = (self.v_max - self.v_min) / max(1, self.num_atoms - 1)
        else:
            self.loss_fn = nn.SmoothL1Loss(reduction="none", beta=1.0)

        self._last_avg_max_q: float = 0.0
        self._batch_q_values_for_actions_taken: Optional[np.ndarray] = None

    def compute_loss(
        self,
        # --- MODIFIED: Type hint for numpy batch from buffer ---
        batch: Union[NumpyBatch, NumpyNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes the loss and TD errors, handling PER weights."""
        # --- Conversion now happens inside loss computation ---
        try:
            tensor_batch = np_batch_to_tensor(
                batch, is_n_step, self.env_config, self.device
            )
        except Exception as e:
            print(f"Error converting batch to tensor in compute_loss: {e}")
            # Return zero loss and no errors if conversion fails
            return torch.tensor(0.0, device=self.device, requires_grad=True), None

        if self.use_distributional:
            return self._compute_distributional_loss(
                tensor_batch, is_n_step, is_weights
            )
        else:
            return self._compute_standard_loss(tensor_batch, is_n_step, is_weights)

    def _compute_standard_loss(
        self,
        tensor_batch: Union[TensorBatch, TensorNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes standard (SmoothL1) loss, applying PER weights."""
        # --- MODIFIED: Unpack based on is_n_step ---
        if is_n_step:
            # Ensure batch has 6 elements for N-step tensor batch
            if len(tensor_batch) != 6:
                raise ValueError(
                    f"Expected 6 elements in N-step tensor batch, got {len(tensor_batch)}"
                )
            states_tuple, actions, rewards, next_states_tuple, dones, discounts = (
                tensor_batch
            )
        else:
            # Ensure batch has 5 elements for 1-step tensor batch
            if len(tensor_batch) != 5:
                raise ValueError(
                    f"Expected 5 elements in 1-step tensor batch, got {len(tensor_batch)}"
                )
            states_tuple, actions, rewards, next_states_tuple, dones = tensor_batch
            # Use standard gamma for 1-step
            discounts = torch.full_like(rewards, self.gamma)
        # --- END MODIFIED ---

        grids, shapes = states_tuple
        next_grids, next_shapes = next_states_tuple
        batch_size = grids.size(0)

        with torch.no_grad():
            self.online_net.eval()
            online_next_q = self.online_net(next_grids, next_shapes)
            best_next_actions = online_next_q.argmax(dim=1, keepdim=True)

            self.target_net.eval()
            target_next_q_values = self.target_net(next_grids, next_shapes)
            target_q_for_best_actions = target_next_q_values.gather(
                1, best_next_actions
            )
            # --- MODIFIED: Use n-step reward and discount ---
            # target_q = R_n + discount_n * Q_target(s_{t+n}, argmax_a Q_online(s_{t+n}, a))
            target_q = rewards + discounts * target_q_for_best_actions * (1.0 - dones)
            # --- END MODIFIED ---

        self.online_net.train()
        current_q_all_actions = self.online_net(grids, shapes)
        current_q = current_q_all_actions.gather(1, actions)

        elementwise_loss = self.loss_fn(current_q, target_q)
        td_errors = (target_q - current_q).abs().detach()

        self._log_q_values(grids, shapes, current_q)

        loss = self._apply_per_weights(elementwise_loss, is_weights)
        td_errors_for_per = td_errors.squeeze() if td_errors is not None else None

        return loss, td_errors_for_per

    def _compute_distributional_loss(
        self,
        tensor_batch: Union[TensorBatch, TensorNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes distributional (C51) loss using KL divergence, applying PER weights."""
        # --- MODIFIED: Unpack based on is_n_step ---
        if is_n_step:
            if len(tensor_batch) != 6:
                raise ValueError(
                    f"Expected 6 elements in N-step tensor batch, got {len(tensor_batch)}"
                )
            states_tuple, actions, _, _, _, _ = (
                tensor_batch  # Rewards/discounts used in target calculation
            )
        else:
            if len(tensor_batch) != 5:
                raise ValueError(
                    f"Expected 5 elements in 1-step tensor batch, got {len(tensor_batch)}"
                )
            states_tuple, actions, _, _, _ = tensor_batch
        # --- END MODIFIED ---

        grids, shapes = states_tuple
        batch_size = grids.size(0)

        with torch.no_grad():
            # --- Target calculation now uses n-step rewards/discounts internally ---
            target_dist = self._get_target_distribution(tensor_batch, is_n_step)

        self.online_net.train()
        current_dist_logits = self.online_net(grids, shapes)
        current_dist_log_probs = F.log_softmax(current_dist_logits, dim=2)

        actions_expanded = actions.view(batch_size, 1, 1).expand(-1, -1, self.num_atoms)
        current_log_probs_for_actions = current_dist_log_probs.gather(
            1, actions_expanded
        ).squeeze(1)

        elementwise_loss = -(target_dist * current_log_probs_for_actions).sum(dim=1)
        td_errors = elementwise_loss.detach()

        self._log_q_values(grids, shapes)

        loss = self._apply_per_weights(elementwise_loss, is_weights)
        td_errors_for_per = td_errors if td_errors is not None else None

        return loss, td_errors_for_per

    @torch.no_grad()
    def _get_target_distribution(
        self, batch: Union[TensorBatch, TensorNStepBatch], is_n_step: bool
    ) -> torch.Tensor:
        """Calculates the target distribution for C51 using Double DQN logic."""
        if not self.use_distributional:
            raise RuntimeError(
                "_get_target_distribution called when use_distributional is False"
            )

        # --- MODIFIED: Unpack rewards and discounts based on is_n_step ---
        if is_n_step:
            if len(batch) != 6:
                raise ValueError(
                    f"Expected 6 elements in N-step tensor batch, got {len(batch)}"
                )
            _, _, rewards, next_states_tuple, dones, discounts = batch
        else:
            if len(batch) != 5:
                raise ValueError(
                    f"Expected 5 elements in 1-step tensor batch, got {len(batch)}"
                )
            _, _, rewards, next_states_tuple, dones = batch[:5]
            discounts = torch.full_like(rewards, self.gamma)  # Use standard gamma
        # --- END MODIFIED ---

        next_grids, next_shapes = next_states_tuple
        batch_size = next_grids.size(0)

        self.online_net.eval()
        online_next_dist_logits = self.online_net(next_grids, next_shapes)
        online_next_probs = F.softmax(online_next_dist_logits, dim=2)
        online_expected_q = (
            online_next_probs * self.support.unsqueeze(0).unsqueeze(0)
        ).sum(dim=2)
        best_next_actions = online_expected_q.argmax(dim=1)

        self.target_net.eval()
        target_next_dist_logits = self.target_net(next_grids, next_shapes)
        target_next_probs = F.softmax(target_next_dist_logits, dim=2)
        target_next_best_dist_probs = target_next_probs[
            torch.arange(batch_size), best_next_actions
        ]

        # --- MODIFIED: Project using n-step rewards and discounts ---
        # Tz = R_n + discount_n * z
        Tz = rewards + discounts * self.support.unsqueeze(0) * (1.0 - dones)
        # --- END MODIFIED ---

        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        lower_idx = b.floor().long()
        upper_idx = b.ceil().long()

        # Handle edge cases and integer bins
        lower_idx = torch.where(lower_idx == upper_idx, lower_idx - 1, lower_idx)
        upper_idx = torch.where(lower_idx >= upper_idx, lower_idx + 1, upper_idx)
        lower_idx = lower_idx.clamp(0, self.num_atoms - 1)
        upper_idx = upper_idx.clamp(0, self.num_atoms - 1)

        weight_u = b - b.floor()
        weight_l = 1.0 - weight_u

        target_dist = torch.zeros_like(target_next_best_dist_probs)
        target_dist.scatter_add_(1, lower_idx, target_next_best_dist_probs * weight_l)
        target_dist.scatter_add_(1, upper_idx, target_next_best_dist_probs * weight_u)

        return target_dist

    def _log_q_values(
        self,
        grids: torch.Tensor,
        shapes: torch.Tensor,
        current_q: Optional[torch.Tensor] = None,
    ):
        with torch.no_grad():
            self.online_net.eval()
            q_or_dist = self.online_net(grids, shapes)
            if self.use_distributional:
                probabilities = F.softmax(q_or_dist, dim=2)
                q_values = (probabilities * self.support.unsqueeze(0).unsqueeze(0)).sum(
                    dim=2
                )
            else:
                q_values = q_or_dist
            self._last_avg_max_q = q_values.max(dim=1)[0].mean().item()

        if (
            self.tb_config.LOG_SHAPE_PLACEMENT_Q_VALUES
            and current_q is not None
            and not self.use_distributional
        ):
            self._batch_q_values_for_actions_taken = (
                current_q.detach().squeeze().cpu().numpy()
            )
        else:
            self._batch_q_values_for_actions_taken = None

    def _apply_per_weights(
        self, elementwise_loss: torch.Tensor, is_weights: Optional[np.ndarray]
    ) -> torch.Tensor:
        if is_weights is not None:
            is_weights_t = torch.tensor(
                is_weights, dtype=torch.float32, device=self.device
            )
            # Ensure dimensions match for broadcasting (elementwise_loss is likely [B])
            if (
                elementwise_loss.ndim == 1
                and is_weights_t.ndim == 1
                and elementwise_loss.shape[0] == is_weights_t.shape[0]
            ):
                loss = (is_weights_t * elementwise_loss).mean()
            elif elementwise_loss.ndim > 1 and is_weights_t.ndim == 1:
                # Try unsqueezing weights if loss has extra dimensions
                is_weights_t = is_weights_t.unsqueeze(1)
                loss = (is_weights_t * elementwise_loss).mean()
            else:
                print(
                    f"Warning: Mismatched shapes for loss ({elementwise_loss.shape}) and IS weights ({is_weights_t.shape}). Using unweighted mean."
                )
                loss = elementwise_loss.mean()
        else:
            loss = elementwise_loss.mean()
        return loss

    def get_last_avg_max_q(self) -> float:
        return self._last_avg_max_q

    def get_last_batch_q_values_for_actions(self) -> Optional[np.ndarray]:
        return self._batch_q_values_for_actions_taken
