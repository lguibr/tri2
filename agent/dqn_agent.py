# File: agent/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from agent.networks.noisy_layer import NoisyLinear


class DQNAgent:
    """DQN Agent using Noisy Nets, Dueling, Double DQN, C51, and LR Scheduling."""

    def __init__(
        self, config: ModelConfig, dqn_config: DQNConfig, env_config: EnvConfig
    ):
        print("[DQNAgent] Initializing...")
        self.device = DEVICE
        self.action_dim = env_config.ACTION_DIM
        self.gamma = dqn_config.GAMMA
        self.use_double_dqn = dqn_config.USE_DOUBLE_DQN
        self.gradient_clip_norm = dqn_config.GRADIENT_CLIP_NORM
        self.use_noisy_nets = dqn_config.USE_NOISY_NETS
        self.use_dueling = dqn_config.USE_DUELING
        self.use_distributional = dqn_config.USE_DISTRIBUTIONAL
        self.v_min = dqn_config.V_MIN
        self.v_max = dqn_config.V_MAX
        self.num_atoms = dqn_config.NUM_ATOMS
        self.dqn_config = dqn_config  # Store config

        if self.use_distributional:
            if self.num_atoms <= 1:
                raise ValueError("NUM_ATOMS must be >= 2 for Distributional RL")
            self.support = torch.linspace(
                self.v_min, self.v_max, self.num_atoms, device=self.device
            )
            # Ensure delta_z is not zero if num_atoms is 1 (although prevented above)
            self.delta_z = (self.v_max - self.v_min) / max(1, self.num_atoms - 1)

        self.online_net = create_network(
            env_config.STATE_DIM, self.action_dim, config, dqn_config
        ).to(self.device)
        self.target_net = create_network(
            env_config.STATE_DIM, self.action_dim, config, dqn_config
        ).to(self.device)
        print(
            f"[DQNAgent] Initial online_net device: {next(self.online_net.parameters()).device}"
        )
        print(
            f"[DQNAgent] Initial target_net device: {next(self.target_net.parameters()).device}"
        )
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.online_net.parameters(),
            lr=dqn_config.LEARNING_RATE,
            eps=dqn_config.ADAM_EPS,
            weight_decay=1e-5,
        )

        self.scheduler = None
        if dqn_config.USE_LR_SCHEDULER:
            print(
                f"[DQNAgent] Using CosineAnnealingLR scheduler (T_max={dqn_config.LR_SCHEDULER_T_MAX}, eta_min={dqn_config.LR_SCHEDULER_ETA_MIN})"
            )
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=dqn_config.LR_SCHEDULER_T_MAX,
                eta_min=dqn_config.LR_SCHEDULER_ETA_MIN,
            )

        if not self.use_distributional:
            self.loss_fn = nn.SmoothL1Loss(reduction="none", beta=1.0)
        self._last_avg_max_q: float = 0.0
        self._print_init_info(dqn_config)

    def _print_init_info(self, dqn_config: DQNConfig):
        print(f"[DQNAgent] Using Device: {self.device}")
        print(f"[DQNAgent] Online Network: {type(self.online_net).__name__}")
        print(f"[DQNAgent] Using Double DQN: {self.use_double_dqn}")
        print(f"[DQNAgent] Using Dueling: {self.use_dueling}")
        print(f"[DQNAgent] Using Noisy Nets: {self.use_noisy_nets}")
        print(f"[DQNAgent] Using Distributional (C51): {self.use_distributional}")
        if self.use_distributional:
            print(
                f"  - Atoms: {self.num_atoms}, Vmin: {self.v_min}, Vmax: {self.v_max}"
            )
        print(
            f"[DQNAgent] Using LR Scheduler: {self.dqn_config.USE_LR_SCHEDULER}"
            + (
                f" (T_max={self.dqn_config.LR_SCHEDULER_T_MAX}, eta_min={self.dqn_config.LR_SCHEDULER_ETA_MIN})"
                if self.dqn_config.USE_LR_SCHEDULER
                else ""
            )
        )
        print(
            f"[DQNAgent] Optimizer: AdamW (LR={dqn_config.LEARNING_RATE}, EPS={dqn_config.ADAM_EPS})"
        )
        total_params = sum(
            p.numel() for p in self.online_net.parameters() if p.requires_grad
        )
        print(f"[DQNAgent] Trainable Parameters: {total_params / 1e6:.2f} M")

    @torch.no_grad()
    def select_action(
        self, state: StateType, epsilon: float, valid_actions: List[ActionType]
    ) -> ActionType:
        if not valid_actions:
            return 0

        state_t = torch.tensor(
            ensure_numpy(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        model_device = next(self.online_net.parameters()).device
        if state_t.device != model_device:
            state_t = state_t.to(model_device)

        self.online_net.eval()
        dist_or_q = self.online_net(state_t)

        if self.use_distributional:
            probabilities = F.softmax(dist_or_q, dim=2)
            q_values = (probabilities * self.support).sum(dim=2)
        else:
            q_values = dist_or_q

        q_values_masked = torch.full_like(q_values[0], -float("inf"))
        valid_action_indices = torch.tensor(
            valid_actions, dtype=torch.long, device=q_values.device
        )

        if valid_action_indices.numel() > 0:
            max_valid_idx = torch.max(valid_action_indices)
            if max_valid_idx < q_values.shape[1]:
                q_values_masked[valid_action_indices] = q_values[
                    0, valid_action_indices
                ]
            else:
                print(
                    f"Warning: Max valid action index ({max_valid_idx}) >= action_dim ({q_values.shape[1]}). Choosing random valid action."
                )
                return random.choice(valid_actions)
        else:
            return 0

        best_action = torch.argmax(q_values_masked).item()
        q_val_of_best_action = q_values_masked[best_action].item()
        self._last_avg_max_q = (
            q_val_of_best_action
            if q_val_of_best_action > -float("inf")
            else -float("inf")
        )

        return best_action

    def _np_batch_to_tensor(
        self, batch: Union[NumpyBatch, NumpyNStepBatch], is_n_step: bool
    ) -> Union[TensorBatch, TensorNStepBatch]:
        """Converts numpy batch tuple to tensor tuple on the correct device."""
        if is_n_step:
            s, a, rn, nsn, dn, gamma_n = batch
            tensors = [
                torch.tensor(arr, device=self.device)
                for arr in [s, a, rn, nsn, dn, gamma_n]
            ]
            tensors[0] = tensors[0].float()
            tensors[1] = tensors[1].long().unsqueeze(1)
            tensors[2] = tensors[2].float().unsqueeze(1)
            tensors[3] = tensors[3].float()
            tensors[4] = tensors[4].float().unsqueeze(1)
            tensors[5] = tensors[5].float().unsqueeze(1)
            return tuple(tensors)
        else:
            s, a, r, ns, d = batch
            tensors = [
                torch.tensor(arr, device=self.device) for arr in [s, a, r, ns, d]
            ]
            tensors[0] = tensors[0].float()
            tensors[1] = tensors[1].long().unsqueeze(1)
            tensors[2] = tensors[2].float().unsqueeze(1)
            tensors[3] = tensors[3].float()
            tensors[4] = tensors[4].float().unsqueeze(1)
            return tuple(tensors)

    @torch.no_grad()
    def _get_target_distribution(
        self, batch: Union[TensorBatch, TensorNStepBatch], is_n_step: bool
    ) -> torch.Tensor:
        """Calculates the target distribution for C51 using Double DQN logic."""
        if is_n_step:
            _, _, rewards, next_states, dones, discounts = batch
        else:
            _, _, rewards, next_states, dones = batch[:5]
            discounts = torch.full_like(rewards, self.gamma)

        batch_size = next_states.size(0)

        self.online_net.eval()
        online_next_dist_logits = self.online_net(next_states)
        online_next_probs = F.softmax(online_next_dist_logits, dim=2)
        online_expected_q = (online_next_probs * self.support).sum(dim=2)
        best_next_actions = online_expected_q.argmax(dim=1)

        self.target_net.eval()
        target_next_dist_logits = self.target_net(next_states)
        target_next_probs = F.softmax(target_next_dist_logits, dim=2)
        target_next_best_dist_probs = target_next_probs[
            torch.arange(batch_size), best_next_actions
        ]

        Tz = rewards + discounts * self.support.unsqueeze(0) * (1.0 - dones)
        Tz = Tz.clamp(self.v_min, self.v_max)

        # --- MODIFIED: Fix index_add_ logic ---
        # Calculate indices and weights
        b = (Tz - self.v_min) / self.delta_z
        lower_idx = b.floor().long()
        upper_idx = b.ceil().long()
        lower_idx[(upper_idx > 0) & (lower_idx == upper_idx)] -= 1
        upper_idx[(lower_idx < (self.num_atoms - 1)) & (lower_idx == upper_idx)] += 1
        lower_idx = lower_idx.clamp(0, self.num_atoms - 1)
        upper_idx = upper_idx.clamp(0, self.num_atoms - 1)
        weight_u = b - lower_idx.float()
        weight_l = 1.0 - weight_u

        # Project onto target distribution tensor using a loop (safer than complex indexing)
        target_dist = torch.zeros(batch_size, self.num_atoms, device=self.device)
        for i in range(batch_size):
            # Use index_add_ *per batch item* to avoid race conditions if indices repeat within an item
            target_dist[i].index_add_(
                0, lower_idx[i], target_next_best_dist_probs[i] * weight_l[i]
            )
            target_dist[i].index_add_(
                0, upper_idx[i], target_next_best_dist_probs[i] * weight_u[i]
            )
        # --- END MODIFIED ---

        return target_dist

    def compute_loss(
        self,
        batch: Union[NumpyBatch, NumpyNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes the loss (Cross-Entropy for C51, SmoothL1 otherwise) and TD errors."""
        tensor_batch = self._np_batch_to_tensor(batch, is_n_step)

        if self.use_distributional:
            states, actions, _, _, _, _ = tensor_batch
            batch_size = states.size(0)
            target_distribution = self._get_target_distribution(tensor_batch, is_n_step)
            self.online_net.train()
            current_dist_logits = self.online_net(states)
            action_idx = actions.view(batch_size, 1, 1).expand(-1, -1, self.num_atoms)
            current_dist_logits_for_actions = current_dist_logits.gather(
                1, action_idx
            ).squeeze(1)
            current_log_probs = F.log_softmax(current_dist_logits_for_actions, dim=1)
            elementwise_loss = -(target_distribution * current_log_probs).sum(dim=1)
            td_errors = elementwise_loss.detach()
            with torch.no_grad():
                self.online_net.eval()
                q_vals = (F.softmax(self.online_net(states), dim=2) * self.support).sum(
                    dim=2
                )
                self._last_avg_max_q = q_vals.max(dim=1)[0].mean().item()
        else:
            states, actions, rewards, next_states, dones = tensor_batch[:5]
            discounts = (
                tensor_batch[5]  # Use pre-calculated gamma^n from NStepBatch
                if is_n_step
                else torch.full_like(rewards, self.gamma)  # Use standard gamma
            )
            with torch.no_grad():
                self.online_net.eval()
                best_next_actions = self.online_net(next_states).argmax(
                    dim=1, keepdim=True
                )
                self.target_net.eval()
                target_next_q = self.target_net(next_states).gather(
                    1, best_next_actions
                )
                target_q = rewards + discounts * target_next_q * (1.0 - dones)
            self.online_net.train()
            current_q = self.online_net(states).gather(1, actions)
            elementwise_loss = self.loss_fn(current_q, target_q)
            td_errors = (target_q - current_q).abs().detach()
            with torch.no_grad():
                self.online_net.eval()
                self._last_avg_max_q = (
                    self.online_net(states).max(dim=1)[0].mean().item()
                )

        is_weights_t = (
            torch.tensor(is_weights, dtype=torch.float32, device=self.device)
            if is_weights is not None
            else None
        )
        loss = (
            (is_weights_t * elementwise_loss.squeeze()).mean()
            if is_weights_t is not None
            else elementwise_loss.mean()
        )
        return loss, td_errors.squeeze()

    def update(self, loss: torch.Tensor) -> Optional[float]:
        """Performs optimizer step, gradient clipping, scheduler step, and noise reset."""
        grad_norm = None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.online_net.train()
        if self.gradient_clip_norm > 0:
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.online_net.parameters(), max_norm=self.gradient_clip_norm
                ).item()
            except Exception as clip_err:
                print(f"Warning: Error during gradient clipping: {clip_err}")
                grad_norm = None
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        if self.use_noisy_nets:
            self.online_net.reset_noise()
            self.target_net.reset_noise()
        return grad_norm

    def get_last_avg_max_q(self) -> float:
        """Returns the average max Q value computed during the last loss calculation."""
        return self._last_avg_max_q

    def update_target_network(self):
        """Copies weights from the online network to the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def get_state_dict(self) -> AgentStateDict:
        """Returns the agent's state including networks, optimizer, and scheduler."""
        self.online_net.cpu()
        self.target_net.cpu()
        optim_state_cpu = {}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p in self.optimizer.state:
                    state = self.optimizer.state[p]
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()
        state = {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
        }
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p in self.optimizer.state:
                    state = self.optimizer.state[p]
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
        return state

    def load_state_dict(self, state_dict: AgentStateDict):
        """Loads the agent's state from a state dictionary."""
        print(f"[DQNAgent] Loading state dict. Target device: {self.device}")
        try:
            self.online_net.load_state_dict(
                state_dict["online_net_state_dict"], strict=False
            )
            print("[DQNAgent] online_net state_dict loaded (strict=False).")
        except Exception as e:
            print(f"ERROR loading online_net state_dict: {e}")
            traceback.print_exc()
            raise
        if "target_net_state_dict" in state_dict:
            try:
                self.target_net.load_state_dict(
                    state_dict["target_net_state_dict"], strict=False
                )
                print("[DQNAgent] target_net state_dict loaded (strict=False).")
            except Exception as e:
                print(f"ERROR loading target_net state_dict: {e}. Copying from online.")
                self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            print(
                "Warning: Target net state missing in checkpoint, copying from online."
            )
            self.target_net.load_state_dict(self.online_net.state_dict())
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        print(
            f"[DQNAgent] online_net moved to device: {next(self.online_net.parameters()).device}"
        )
        print(
            f"[DQNAgent] target_net moved to device: {next(self.target_net.parameters()).device}"
        )
        if "optimizer_state_dict" in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.device != self.device:
                            state[k] = v.to(self.device)
                print("[DQNAgent] Optimizer state loaded and moved to device.")
            except Exception as e:
                print(
                    f"Warning: Could not load optimizer state ({e}). Resetting optimizer."
                )
                self.optimizer = optim.AdamW(
                    self.online_net.parameters(),
                    lr=self.dqn_config.LEARNING_RATE,
                    eps=self.dqn_config.ADAM_EPS,
                    weight_decay=1e-5,
                )
        else:
            print(
                "Warning: Optimizer state not found in checkpoint. Resetting optimizer."
            )
            self.optimizer = optim.AdamW(
                self.online_net.parameters(),
                lr=self.dqn_config.LEARNING_RATE,
                eps=self.dqn_config.ADAM_EPS,
                weight_decay=1e-5,
            )
        if (
            self.scheduler
            and "scheduler_state_dict" in state_dict
            and state_dict["scheduler_state_dict"] is not None
        ):
            try:
                self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
                print("[DQNAgent] LR Scheduler state loaded.")
            except Exception as e:
                print(
                    f"Warning: Could not load LR scheduler state ({e}). Scheduler may reset."
                )
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.dqn_config.LR_SCHEDULER_T_MAX,
                    eta_min=self.dqn_config.LR_SCHEDULER_ETA_MIN,
                )
        elif self.scheduler:
            print(
                "Warning: LR Scheduler state not found in checkpoint. Scheduler may reset."
            )
        self.online_net.train()
        self.target_net.eval()
        print("[DQNAgent] load_state_dict complete.")
