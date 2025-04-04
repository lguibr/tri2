# File: agent/loss_calculator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union

from config import DQNConfig, TensorBoardConfig, EnvConfig
from utils.types import TensorBatch, TensorNStepBatch
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
        self.gamma = dqn_config.GAMMA
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
        batch: Union[np.ndarray, np.ndarray],
        is_n_step: bool,
        is_weights: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes the loss and TD errors."""
        tensor_batch = np_batch_to_tensor(
            batch, is_n_step, self.env_config, self.device
        )

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
        """Computes standard (SmoothL1) loss."""
        states_tuple, actions, rewards, next_states_tuple, dones = tensor_batch[:5]
        grids, shapes = states_tuple
        next_grids, next_shapes = next_states_tuple
        batch_size = grids.size(0)

        discounts = (
            tensor_batch[5] if is_n_step else torch.full_like(rewards, self.gamma)
        )

        with torch.no_grad():
            self.online_net.eval()
            online_next_q = self.online_net(next_grids, next_shapes)
            best_next_actions = online_next_q.argmax(dim=1, keepdim=True)

            self.target_net.eval()
            target_next_q_values = self.target_net(next_grids, next_shapes)
            target_q_for_best_actions = target_next_q_values.gather(
                1, best_next_actions
            )
            target_q = rewards + discounts * target_q_for_best_actions * (1.0 - dones)

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
        """Computes distributional (C51) loss using KL divergence."""
        states_tuple, actions, _, _, _ = tensor_batch[:5]
        grids, shapes = states_tuple
        batch_size = grids.size(0)

        with torch.no_grad():
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

        if is_n_step:
            _, _, rewards, next_states_tuple, dones, discounts = batch
        else:
            _, _, rewards, next_states_tuple, dones = batch[:5]
            discounts = torch.full_like(rewards, self.gamma)

        next_grids, next_shapes = next_states_tuple
        batch_size = next_grids.size(0)

        self.online_net.eval()
        online_next_dist_logits = self.online_net(next_grids, next_shapes)
        online_next_probs = F.softmax(online_next_dist_logits, dim=2)
        online_expected_q = (online_next_probs * self.support).sum(dim=2)
        best_next_actions = online_expected_q.argmax(dim=1)

        self.target_net.eval()
        target_next_dist_logits = self.target_net(next_grids, next_shapes)
        target_next_probs = F.softmax(target_next_dist_logits, dim=2)
        target_next_best_dist_probs = target_next_probs[
            torch.arange(batch_size), best_next_actions
        ]

        Tz = rewards + discounts * self.support.unsqueeze(0) * (1.0 - dones)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        lower_idx = b.floor().long()
        upper_idx = b.ceil().long()

        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=self.device
            )
            .long()
            .unsqueeze(1)
        )
        lower_idx = (lower_idx + offset).clamp(0, batch_size * self.num_atoms - 1)
        upper_idx = (upper_idx + offset).clamp(0, batch_size * self.num_atoms - 1)

        weight_u = b - b.floor()
        weight_l = 1.0 - weight_u

        target_dist = torch.zeros_like(target_next_best_dist_probs)
        target_dist.view(-1).index_add_(
            0, lower_idx.view(-1), (target_next_best_dist_probs * weight_l).view(-1)
        )
        target_dist.view(-1).index_add_(
            0, upper_idx.view(-1), (target_next_best_dist_probs * weight_u).view(-1)
        )

        return target_dist

    def _log_q_values(
        self,
        grids: torch.Tensor,
        shapes: torch.Tensor,
        current_q: Optional[torch.Tensor] = None,
    ):
        """Logs average max Q and optionally Q-values for actions taken."""
        with torch.no_grad():
            self.online_net.eval()
            q_or_dist = self.online_net(grids, shapes)
            if self.use_distributional:
                probabilities = F.softmax(q_or_dist, dim=2)
                q_values = (probabilities * self.support).sum(dim=2)
            else:
                q_values = q_or_dist
            self._last_avg_max_q = q_values.max(dim=1)[0].mean().item()

        if self.tb_config.LOG_SHAPE_PLACEMENT_Q_VALUES and current_q is not None:
            self._batch_q_values_for_actions_taken = (
                current_q.detach().squeeze().cpu().numpy()
            )
        else:
            self._batch_q_values_for_actions_taken = None

    def _apply_per_weights(
        self, elementwise_loss: torch.Tensor, is_weights: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Applies PER weights to the loss."""
        if is_weights is not None:
            is_weights_t = torch.tensor(
                is_weights, dtype=torch.float32, device=self.device
            )
            # Ensure is_weights_t matches elementwise_loss shape if needed
            if elementwise_loss.ndim > 1 and is_weights_t.ndim == 1:
                is_weights_t = is_weights_t.unsqueeze(1)
            loss = (is_weights_t * elementwise_loss).mean()
        else:
            loss = elementwise_loss.mean()
        return loss

    def get_last_avg_max_q(self) -> float:
        return self._last_avg_max_q

    def get_last_batch_q_values_for_actions(self) -> Optional[np.ndarray]:
        return self._batch_q_values_for_actions_taken
