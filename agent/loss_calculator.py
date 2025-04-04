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
            # --- Initialize support tensor ---
            self.support = torch.linspace(
                self.v_min, self.v_max, self.num_atoms, device=self.device
            )
            # Calculate delta_z based on support range and num_atoms
            self.delta_z = (self.v_max - self.v_min) / max(1, self.num_atoms - 1)
            # --- Loss for C51 is calculated differently (KL Divergence based) ---
            # No explicit loss_fn needed here, calculation happens in _compute_distributional_loss
        else:
            # Standard DQN uses SmoothL1Loss (Huber Loss)
            self.loss_fn = nn.SmoothL1Loss(
                reduction="none", beta=1.0
            )  # Keep element-wise loss

        self._last_avg_max_q: float = 0.0
        self._batch_q_values_for_actions_taken: Optional[np.ndarray] = None

    def compute_loss(
        self,
        batch: Union[
            np.ndarray, np.ndarray
        ],  # Batch from buffer (numpy, states as dicts)
        is_n_step: bool,  # Flag indicating if N-step returns were used
        is_weights: Optional[np.ndarray] = None,  # Importance Sampling weights for PER
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes the loss and TD errors, handling PER weights."""
        # Convert numpy batch (with dict states) to tensor batch (with separate grid/shape tensors)
        tensor_batch = np_batch_to_tensor(
            batch, is_n_step, self.env_config, self.device
        )

        if self.use_distributional:
            # --- Pass IS weights to distributional loss ---
            return self._compute_distributional_loss(
                tensor_batch, is_n_step, is_weights
            )
        else:
            # --- Pass IS weights to standard loss ---
            return self._compute_standard_loss(tensor_batch, is_n_step, is_weights)

    def _compute_standard_loss(
        self,
        tensor_batch: Union[TensorBatch, TensorNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray],  # Accept IS weights
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes standard (SmoothL1) loss, applying PER weights."""
        states_tuple, actions, rewards, next_states_tuple, dones = tensor_batch[:5]
        grids, shapes = states_tuple
        next_grids, next_shapes = next_states_tuple
        batch_size = grids.size(0)

        # Determine discount factor based on N-step or standard gamma
        discounts = (
            tensor_batch[5] if is_n_step else torch.full_like(rewards, self.gamma)
        )

        with torch.no_grad():
            self.online_net.eval()  # Set online net to eval mode for action selection
            online_next_q = self.online_net(next_grids, next_shapes)
            # Select best actions according to online network (Double DQN)
            best_next_actions = online_next_q.argmax(dim=1, keepdim=True)

            self.target_net.eval()  # Ensure target net is in eval mode
            target_next_q_values = self.target_net(next_grids, next_shapes)
            # Get Q-values for the best actions from the target network
            target_q_for_best_actions = target_next_q_values.gather(
                1, best_next_actions
            )
            # Calculate the TD target: R + gamma * Q_target(s', argmax_a Q_online(s', a))
            target_q = rewards + discounts * target_q_for_best_actions * (1.0 - dones)

        self.online_net.train()  # Set online net back to train mode
        # Get Q-values for the actions actually taken
        current_q_all_actions = self.online_net(grids, shapes)
        current_q = current_q_all_actions.gather(1, actions)

        # Calculate element-wise SmoothL1 loss
        elementwise_loss = self.loss_fn(current_q, target_q)
        # Calculate TD errors (absolute difference) for PER priority update
        td_errors = (target_q - current_q).abs().detach()  # Detach errors from graph

        # Log Q-values (average max Q and optionally Q for actions taken)
        self._log_q_values(grids, shapes, current_q)

        # --- Apply PER IS weights ---
        loss = self._apply_per_weights(elementwise_loss, is_weights)
        # --- Squeeze TD errors for PER update ---
        td_errors_for_per = td_errors.squeeze() if td_errors is not None else None

        return loss, td_errors_for_per

    def _compute_distributional_loss(
        self,
        tensor_batch: Union[TensorBatch, TensorNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray],  # Accept IS weights
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes distributional (C51) loss using KL divergence, applying PER weights."""
        states_tuple, actions, _, _, _ = tensor_batch[:5]  # Unpack necessary parts
        grids, shapes = states_tuple
        batch_size = grids.size(0)

        with torch.no_grad():
            # Calculate the target distribution projected onto the support
            target_dist = self._get_target_distribution(
                tensor_batch, is_n_step
            )  # Shape: [B, N_atoms]

        self.online_net.train()  # Set online net to train mode
        # Get current distribution logits from online network
        current_dist_logits = self.online_net(
            grids, shapes
        )  # Shape: [B, action_dim, N_atoms]
        # Convert logits to log probabilities
        current_dist_log_probs = F.log_softmax(current_dist_logits, dim=2)

        # Select the log probabilities for the actions actually taken
        actions_expanded = actions.view(batch_size, 1, 1).expand(-1, -1, self.num_atoms)
        current_log_probs_for_actions = current_dist_log_probs.gather(
            1, actions_expanded
        ).squeeze(
            1
        )  # Shape: [B, N_atoms]

        # Calculate element-wise KL divergence (negative cross-entropy here as target is one-hot like)
        elementwise_loss = -(target_dist * current_log_probs_for_actions).sum(
            dim=1
        )  # Shape: [B]
        # Use the element-wise loss as TD error for PER (already detached as target_dist is no_grad)
        td_errors = elementwise_loss.detach()

        # Log Q-values (average max Expected Q)
        self._log_q_values(grids, shapes)

        # --- Apply PER IS weights ---
        loss = self._apply_per_weights(elementwise_loss, is_weights)
        # --- TD errors are already element-wise losses for C51 ---
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
        else:  # Standard 1-step
            _, _, rewards, next_states_tuple, dones = batch[:5]
            discounts = torch.full_like(rewards, self.gamma)  # Use standard gamma

        next_grids, next_shapes = next_states_tuple
        batch_size = next_grids.size(0)

        # --- Double DQN Logic for selecting next actions ---
        self.online_net.eval()  # Use online net to select best next actions
        online_next_dist_logits = self.online_net(next_grids, next_shapes)  # [B, A, N]
        online_next_probs = F.softmax(online_next_dist_logits, dim=2)
        online_expected_q = (
            online_next_probs * self.support.unsqueeze(0).unsqueeze(0)
        ).sum(
            dim=2
        )  # [B, A]
        best_next_actions = online_expected_q.argmax(dim=1)  # [B]
        # --- End Double DQN Logic ---

        # Use target net to get the distribution for the selected best next actions
        self.target_net.eval()
        target_next_dist_logits = self.target_net(next_grids, next_shapes)  # [B, A, N]
        target_next_probs = F.softmax(target_next_dist_logits, dim=2)  # [B, A, N]
        # Gather the target distributions corresponding to the best actions chosen by the online net
        target_next_best_dist_probs = target_next_probs[
            torch.arange(batch_size), best_next_actions
        ]  # [B, N]

        # --- Project the target distribution onto the support ---
        # Calculate the projected support atoms Tz = R + gamma^n * z
        Tz = rewards + discounts * self.support.unsqueeze(0) * (1.0 - dones)  # [B, N]
        # Clamp the projected atoms into the support range [Vmin, Vmax]
        Tz = Tz.clamp(self.v_min, self.v_max)
        # Calculate the bin indices (b) and weights for projection
        b = (Tz - self.v_min) / self.delta_z  # [B, N]
        lower_idx = b.floor().long()  # [B, N]
        upper_idx = b.ceil().long()  # [B, N]

        # Handle cases where b is exactly an integer (l == u)
        lower_idx = torch.where(lower_idx == upper_idx, lower_idx - 1, lower_idx)
        upper_idx = torch.where(lower_idx >= upper_idx, lower_idx + 1, upper_idx)
        # Ensure indices are within bounds [0, N_atoms - 1]
        lower_idx = lower_idx.clamp(0, self.num_atoms - 1)
        upper_idx = upper_idx.clamp(0, self.num_atoms - 1)

        # Calculate projection weights (linear interpolation)
        weight_u = b - b.floor()  # [B, N]
        weight_l = 1.0 - weight_u  # [B, N]

        # Distribute the probability mass m(z_j) to the adjacent atoms m_l and m_u
        target_dist = torch.zeros_like(target_next_best_dist_probs)  # [B, N]
        # Use index_add_ for efficient distribution
        target_dist.scatter_add_(1, lower_idx, target_next_best_dist_probs * weight_l)
        target_dist.scatter_add_(1, upper_idx, target_next_best_dist_probs * weight_u)
        # --- End Projection ---

        return target_dist  # Shape: [B, N_atoms]

    def _log_q_values(
        self,
        grids: torch.Tensor,
        shapes: torch.Tensor,
        current_q: Optional[torch.Tensor] = None,
    ):
        """Logs average max Q and optionally Q-values for actions taken."""
        # Calculate average max Q for logging (even if C51 is used)
        with torch.no_grad():
            self.online_net.eval()  # Ensure eval mode
            q_or_dist = self.online_net(grids, shapes)
            if self.use_distributional:
                probabilities = F.softmax(q_or_dist, dim=2)
                q_values = (probabilities * self.support.unsqueeze(0).unsqueeze(0)).sum(
                    dim=2
                )
            else:
                q_values = q_or_dist
            # Calculate the mean of the maximum Q values across the batch
            self._last_avg_max_q = q_values.max(dim=1)[0].mean().item()

        # Store batch Q-values for actions taken if needed for TB logging
        if (
            self.tb_config.LOG_SHAPE_PLACEMENT_Q_VALUES
            and current_q is not None
            and not self.use_distributional
        ):
            # Only store if not C51, as current_q represents Q-values directly
            self._batch_q_values_for_actions_taken = (
                current_q.detach().squeeze().cpu().numpy()
            )
        else:
            self._batch_q_values_for_actions_taken = None

    def _apply_per_weights(
        self, elementwise_loss: torch.Tensor, is_weights: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Applies PER weights to the element-wise loss."""
        if is_weights is not None:
            # Convert numpy weights to tensor
            is_weights_t = torch.tensor(
                is_weights, dtype=torch.float32, device=self.device
            )
            # Ensure weights tensor matches loss tensor dimensions for broadcasting
            if elementwise_loss.ndim > 1 and is_weights_t.ndim == 1:
                is_weights_t = is_weights_t.unsqueeze(1)
            # Compute weighted mean loss
            loss = (is_weights_t * elementwise_loss).mean()
        else:
            # Compute standard mean loss if no weights provided (uniform sampling)
            loss = elementwise_loss.mean()
        return loss

    def get_last_avg_max_q(self) -> float:
        """Returns the average max Q value from the last processed batch."""
        return self._last_avg_max_q

    def get_last_batch_q_values_for_actions(self) -> Optional[np.ndarray]:
        """Returns the Q-values for actions taken in the last processed batch (if logged)."""
        return self._batch_q_values_for_actions_taken
