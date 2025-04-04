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

# --- MODIFIED: Import specific StateType ---
from environment.game_state import StateType  # Use the Dict type

# --- END MODIFIED ---
from utils.types import (
    # StateType, # Removed, using env's definition
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
        self.env_config = env_config  # Store instance
        self.action_dim = env_config.ACTION_DIM  # Use property
        self.gamma = dqn_config.GAMMA
        self.use_double_dqn = dqn_config.USE_DOUBLE_DQN
        self.gradient_clip_norm = dqn_config.GRADIENT_CLIP_NORM
        self.use_noisy_nets = dqn_config.USE_NOISY_NETS
        self.use_dueling = dqn_config.USE_DUELING
        self.use_distributional = dqn_config.USE_DISTRIBUTIONAL
        self.v_min = dqn_config.V_MIN
        self.v_max = dqn_config.V_MAX
        self.num_atoms = dqn_config.NUM_ATOMS
        self.dqn_config = dqn_config

        if self.use_distributional:
            if self.num_atoms <= 1:
                raise ValueError("NUM_ATOMS must be >= 2 for Distributional RL")
            self.support = torch.linspace(
                self.v_min, self.v_max, self.num_atoms, device=self.device
            )
            self.delta_z = (self.v_max - self.v_min) / max(1, self.num_atoms - 1)

        # --- MODIFIED: Pass EnvConfig instance to factory ---
        self.online_net = create_network(
            env_config=self.env_config,  # Pass instance
            action_dim=self.action_dim,
            model_config=config,
            dqn_config=dqn_config,
        ).to(self.device)
        self.target_net = create_network(
            env_config=self.env_config,  # Pass instance
            action_dim=self.action_dim,
            model_config=config,
            dqn_config=dqn_config,
        ).to(self.device)
        # --- END MODIFIED ---

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
            # Use reduction='none' for PER, apply weights later
            self.loss_fn = nn.SmoothL1Loss(reduction="none", beta=1.0)
        self._last_avg_max_q: float = 0.0
        self._print_init_info(dqn_config)

    def _print_init_info(self, dqn_config: DQNConfig):
        # (No changes needed here)
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
        """Selects action using epsilon-greedy or noisy nets, applying action masking."""
        if not valid_actions:
            # print("Warning: No valid actions provided to select_action. Returning action 0.")
            return 0  # Default action if no valid moves

        # --- MODIFIED: Handle dictionary state ---
        grid_np = ensure_numpy(state["grid"])
        shapes_np = ensure_numpy(state["shapes"])

        grid_t = torch.tensor(
            grid_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        shapes_t = torch.tensor(
            shapes_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        # --- END MODIFIED ---

        model_device = next(self.online_net.parameters()).device
        if grid_t.device != model_device:
            grid_t = grid_t.to(model_device)
        if shapes_t.device != model_device:
            shapes_t = shapes_t.to(model_device)

        # Epsilon-greedy exploration (only if not using noisy nets)
        if not self.use_noisy_nets and random.random() < epsilon:
            return random.choice(valid_actions)

        # --- MODIFIED: Pass separate tensors to network ---
        self.online_net.eval()  # Ensure eval mode for selection
        dist_or_q = self.online_net(grid_t, shapes_t)
        # --- END MODIFIED ---

        if self.use_distributional:
            probabilities = F.softmax(dist_or_q, dim=2)
            q_values = (probabilities * self.support).sum(
                dim=2
            )  # Shape: [1, action_dim]
        else:
            q_values = dist_or_q  # Shape: [1, action_dim]

        # --- Action Masking ---
        q_values_masked = torch.full_like(q_values[0], -float("inf"))  # Start with -inf
        if valid_actions:
            valid_action_indices = torch.tensor(
                valid_actions, dtype=torch.long, device=q_values.device
            )
            # Ensure indices are within bounds before gathering
            if valid_action_indices.max() < q_values.shape[1]:
                q_values_masked[valid_action_indices] = q_values[
                    0, valid_action_indices
                ]
            else:
                print(
                    f"Warning: Max valid action index ({valid_action_indices.max()}) >= action_dim ({q_values.shape[1]}). Choosing random valid action."
                )
                return random.choice(valid_actions)  # Fallback

        # Select best valid action
        best_action = torch.argmax(q_values_masked).item()

        # Store Q-value of the chosen action (for logging/debugging)
        q_val_of_best_action = q_values_masked[best_action].item()
        self._last_avg_max_q = (
            q_val_of_best_action if q_val_of_best_action > -float("inf") else 0.0
        )  # Use 0 if no valid action found

        return best_action

    # --- MODIFIED: _np_batch_to_tensor handles dictionary states ---
    def _np_batch_to_tensor(
        self, batch: Union[NumpyBatch, NumpyNStepBatch], is_n_step: bool
    ) -> Union[TensorBatch, TensorNStepBatch]:
        """Converts numpy batch tuple (where states are dicts) to tensor tuple."""

        # Unpack based on n-step or not
        if is_n_step:
            states_dicts, actions, rewards, next_states_dicts, dones, discounts = batch
        else:
            states_dicts, actions, rewards, next_states_dicts, dones = batch[:5]
            discounts = None  # Will be calculated later if needed

        # Process states and next_states (which are lists of dicts)
        grid_states = np.array([s["grid"] for s in states_dicts], dtype=np.float32)
        shape_states = np.array([s["shapes"] for s in states_dicts], dtype=np.float32)
        grid_next_states = np.array(
            [ns["grid"] for ns in next_states_dicts], dtype=np.float32
        )
        shape_next_states = np.array(
            [ns["shapes"] for ns in next_states_dicts], dtype=np.float32
        )

        # Convert components to tensors
        grid_s_t = torch.tensor(grid_states, device=self.device, dtype=torch.float32)
        shape_s_t = torch.tensor(shape_states, device=self.device, dtype=torch.float32)
        a_t = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        r_t = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(
            1
        )
        grid_ns_t = torch.tensor(
            grid_next_states, device=self.device, dtype=torch.float32
        )
        shape_ns_t = torch.tensor(
            shape_next_states, device=self.device, dtype=torch.float32
        )
        d_t = torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(1)

        # Combine state tensors into tuples for convenience
        states_t = (grid_s_t, shape_s_t)
        next_states_t = (grid_ns_t, shape_ns_t)

        if is_n_step:
            disc_t = torch.tensor(
                discounts, device=self.device, dtype=torch.float32
            ).unsqueeze(1)
            return states_t, a_t, r_t, next_states_t, d_t, disc_t
        else:
            return states_t, a_t, r_t, next_states_t, d_t

    # --- END MODIFIED ---

    @torch.no_grad()
    def _get_target_distribution(
        self, batch: Union[TensorBatch, TensorNStepBatch], is_n_step: bool
    ) -> torch.Tensor:
        """Calculates the target distribution for C51 using Double DQN logic."""
        if is_n_step:
            _, _, rewards, next_states_tuple, dones, discounts = batch
        else:
            _, _, rewards, next_states_tuple, dones = batch[:5]
            discounts = torch.full_like(rewards, self.gamma)

        next_grids, next_shapes = next_states_tuple
        batch_size = next_grids.size(0)

        # Double DQN: Use online net to select best action, target net to evaluate
        self.online_net.eval()
        online_next_dist_logits = self.online_net(
            next_grids, next_shapes
        )  # [B, A, N_atoms]
        online_next_probs = F.softmax(online_next_dist_logits, dim=2)
        online_expected_q = (online_next_probs * self.support).sum(dim=2)  # [B, A]
        # --- MASKING NEEDED HERE ---
        # We need the valid actions mask for the *next* states.
        # This is tricky because the batch doesn't contain this info directly.
        # Option 1: Recompute valid actions (slow).
        # Option 2: Store valid actions mask in replay buffer (increases memory).
        # Option 3: Ignore masking in target calculation (simpler but less accurate).
        # Let's go with Option 3 for now, but acknowledge this limitation.
        # TODO: Revisit masking in target calculation if performance suffers.
        best_next_actions = online_expected_q.argmax(dim=1)  # [B]
        # --- END MASKING NOTE ---

        self.target_net.eval()
        target_next_dist_logits = self.target_net(
            next_grids, next_shapes
        )  # [B, A, N_atoms]
        target_next_probs = F.softmax(target_next_dist_logits, dim=2)  # [B, A, N_atoms]
        # Gather the distributions corresponding to the best actions selected by online net
        target_next_best_dist_probs = target_next_probs[
            torch.arange(batch_size), best_next_actions
        ]  # [B, N_atoms]

        # Project the target distribution
        Tz = rewards + discounts * self.support.unsqueeze(0) * (
            1.0 - dones
        )  # [B, N_atoms]
        Tz = Tz.clamp(self.v_min, self.v_max)

        b = (Tz - self.v_min) / self.delta_z
        lower_idx = b.floor().long()
        upper_idx = b.ceil().long()
        # Handle cases where Tz falls exactly on a support bin edge
        lower_idx[(upper_idx > 0) & (lower_idx == upper_idx)] -= 1
        upper_idx[(lower_idx < (self.num_atoms - 1)) & (lower_idx == upper_idx)] += 1
        lower_idx = lower_idx.clamp(0, self.num_atoms - 1)
        upper_idx = upper_idx.clamp(0, self.num_atoms - 1)

        weight_u = b - lower_idx.float()
        weight_l = 1.0 - weight_u

        target_dist = torch.zeros_like(target_next_best_dist_probs)  # [B, N_atoms]
        # Use index_add_ for projection (safer than scatter_add_)
        target_dist.index_add_(
            1, lower_idx.view(-1), (target_next_best_dist_probs * weight_l).view(-1)
        )
        target_dist.index_add_(
            1, upper_idx.view(-1), (target_next_best_dist_probs * weight_u).view(-1)
        )
        # Reshape indices and values for index_add_
        # This requires careful handling of batch dimension. Let's use a loop for clarity first.
        # target_dist = torch.zeros(batch_size, self.num_atoms, device=self.device)
        # for i in range(batch_size):
        #      target_dist[i].index_add_(0, lower_idx[i], target_next_best_dist_probs[i] * weight_l[i])
        #      target_dist[i].index_add_(0, upper_idx[i], target_next_best_dist_probs[i] * weight_u[i])

        return target_dist

    # --- MODIFIED: compute_loss handles dictionary states ---
    def compute_loss(
        self,
        batch: Union[NumpyBatch, NumpyNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes the loss (Cross-Entropy for C51, SmoothL1 otherwise) and TD errors."""
        tensor_batch = self._np_batch_to_tensor(batch, is_n_step)
        states_tuple, actions, rewards, next_states_tuple, dones = tensor_batch[:5]
        grids, shapes = states_tuple
        next_grids, next_shapes = next_states_tuple

        td_errors = None  # Initialize td_errors

        if self.use_distributional:
            batch_size = grids.size(0)
            target_distribution = self._get_target_distribution(
                tensor_batch, is_n_step
            )  # [B, N_atoms]

            self.online_net.train()  # Ensure train mode
            current_dist_logits = self.online_net(grids, shapes)  # [B, A, N_atoms]
            # Gather the logits for the actions taken
            action_idx = actions.view(batch_size, 1, 1).expand(
                -1, -1, self.num_atoms
            )  # [B, 1, N_atoms]
            current_dist_logits_for_actions = current_dist_logits.gather(
                1, action_idx
            ).squeeze(
                1
            )  # [B, N_atoms]

            # Calculate cross-entropy loss
            current_log_probs = F.log_softmax(current_dist_logits_for_actions, dim=1)
            elementwise_loss = -(target_distribution * current_log_probs).sum(
                dim=1
            )  # [B]
            td_errors = elementwise_loss.detach()  # Use loss as TD error proxy for PER

            # Calculate Q-values for logging _last_avg_max_q
            with torch.no_grad():
                self.online_net.eval()
                q_vals_all_actions = (
                    F.softmax(self.online_net(grids, shapes), dim=2) * self.support
                ).sum(
                    dim=2
                )  # [B, A]
                # TODO: Apply masking here if possible for a more accurate avg_max_q
                self._last_avg_max_q = q_vals_all_actions.max(dim=1)[0].mean().item()

        else:  # Standard DQN (Non-distributional)
            discounts = (
                tensor_batch[5]  # Use pre-calculated gamma^n from NStepBatch
                if is_n_step
                else torch.full_like(rewards, self.gamma)  # Use standard gamma
            )

            # Target Q-value calculation (Double DQN)
            with torch.no_grad():
                self.online_net.eval()
                # Select best actions using online network
                online_next_q = self.online_net(next_grids, next_shapes)  # [B, A]
                # --- MASKING NEEDED HERE ---
                # Again, ideally mask online_next_q before argmax
                # TODO: Revisit masking in target calculation
                best_next_actions = online_next_q.argmax(dim=1, keepdim=True)  # [B, 1]
                # --- END MASKING NOTE ---

                self.target_net.eval()
                # Evaluate selected actions using target network
                target_next_q_values = self.target_net(
                    next_grids, next_shapes
                )  # [B, A]
                target_q_for_best_actions = target_next_q_values.gather(
                    1, best_next_actions
                )  # [B, 1]

                # Calculate TD target
                target_q = rewards + discounts * target_q_for_best_actions * (
                    1.0 - dones
                )  # [B, 1]

            # Current Q-value calculation
            self.online_net.train()  # Ensure train mode
            current_q_all_actions = self.online_net(grids, shapes)  # [B, A]
            current_q = current_q_all_actions.gather(1, actions)  # [B, 1]

            # Calculate element-wise loss (Huber)
            elementwise_loss = self.loss_fn(current_q, target_q)  # [B, 1]
            td_errors = (target_q - current_q).abs().detach()  # [B, 1]

            # Calculate Q-values for logging _last_avg_max_q
            with torch.no_grad():
                self.online_net.eval()
                # TODO: Apply masking here if possible
                self._last_avg_max_q = (
                    self.online_net(grids, shapes).max(dim=1)[0].mean().item()
                )

        # Apply PER weights and calculate final loss
        if is_weights is not None:
            is_weights_t = torch.tensor(
                is_weights, dtype=torch.float32, device=self.device
            ).unsqueeze(
                1
            )  # Ensure shape matches loss [B, 1]
            loss = (is_weights_t * elementwise_loss).mean()
        else:
            loss = elementwise_loss.mean()

        # Ensure td_errors is correctly shaped for PER update (squeeze if needed)
        td_errors_for_per = td_errors.squeeze() if td_errors is not None else None

        return loss, td_errors_for_per

    # --- END MODIFIED ---

    def update(self, loss: torch.Tensor) -> Optional[float]:
        """Performs optimizer step, gradient clipping, scheduler step, and noise reset."""
        grad_norm = None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.online_net.train()  # Ensure train mode before clipping/stepping

        if self.gradient_clip_norm > 0:
            try:
                # Clip gradients of the online network
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.online_net.parameters(),
                    max_norm=self.gradient_clip_norm,
                    error_if_nonfinite=True,  # Added for debugging
                ).item()
            except RuntimeError as clip_err:
                print(f"ERROR: Gradient clipping failed: {clip_err}")
                # Optionally: Log parameters with NaN/inf gradients
                # for p in self.online_net.parameters():
                #     if p.grad is not None and not torch.isfinite(p.grad).all():
                #         print(f"Non-finite gradient found in parameter: {p.shape}")
                return None  # Skip optimizer step if clipping fails
            except Exception as clip_err:
                print(f"Warning: Error during gradient clipping: {clip_err}")
                grad_norm = None  # Continue but report no norm

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
        # Move models to CPU before getting state_dict to avoid GPU memory in saved file
        self.online_net.cpu()
        self.target_net.cpu()

        # Ensure optimizer state is also on CPU
        optim_state_cpu = {}
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p in self.optimizer.state:
                    state = self.optimizer.state[p]
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            # Store a CPU copy in the temp dict
                            optim_state_cpu[id(p), k] = v.cpu()
                            # Temporarily replace tensor in optimizer state with CPU version
                            state[k] = v.cpu()

        state = {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            # Add other state if needed (e.g., training step for scheduler)
        }

        # Restore optimizer state to original device tensors
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p in self.optimizer.state:
                    state_opt = self.optimizer.state[p]
                    for k, v_cpu in state_opt.items():
                        if (
                            isinstance(v_cpu, torch.Tensor)
                            and (id(p), k) in optim_state_cpu
                        ):
                            # Restore original tensor (which should be on self.device)
                            original_tensor = optim_state_cpu[(id(p), k)].to(
                                self.device
                            )
                            state_opt[k] = original_tensor

        # Move models back to the original device
        self.online_net.to(self.device)
        self.target_net.to(self.device)

        return state

    def load_state_dict(self, state_dict: AgentStateDict):
        """Loads the agent's state from a state dictionary."""
        print(f"[DQNAgent] Loading state dict. Target device: {self.device}")
        try:
            # Load network weights (allow missing keys, e.g., if architecture changed)
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

        # Ensure models are on the correct device AFTER loading state dict
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        print(
            f"[DQNAgent] online_net moved to device: {next(self.online_net.parameters()).device}"
        )
        print(
            f"[DQNAgent] target_net moved to device: {next(self.target_net.parameters()).device}"
        )

        # Load optimizer state
        if "optimizer_state_dict" in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                # Move optimizer state tensors to the correct device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.device != self.device:
                            state[k] = v.to(self.device)
                print("[DQNAgent] Optimizer state loaded and moved to device.")
            except Exception as e:
                print(
                    f"Warning: Could not load optimizer state ({e}). Resetting optimizer."
                )
                # Re-initialize optimizer if loading fails
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

        # Load scheduler state
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
                # Re-initialize scheduler
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.dqn_config.LR_SCHEDULER_T_MAX,
                    eta_min=self.dqn_config.LR_SCHEDULER_ETA_MIN,
                )
        elif self.scheduler:
            print(
                "Warning: LR Scheduler state not found in checkpoint. Scheduler may reset."
            )

        # Ensure nets are in correct mode after loading
        self.online_net.train()
        self.target_net.eval()
        print("[DQNAgent] load_state_dict complete.")
