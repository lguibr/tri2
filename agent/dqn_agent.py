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

# --- MODIFIED: Import EnvConfig directly ---
from config import ModelConfig, EnvConfig, DQNConfig, DEVICE, TensorBoardConfig

# --- END MODIFIED ---
from agent.model_factory import create_network
from environment.game_state import StateType  # Use the Dict type from env
from utils.types import (
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
    """DQN Agent implementing two-stage action selection (shape -> placement)."""

    def __init__(
        self, config: ModelConfig, dqn_config: DQNConfig, env_config: EnvConfig
    ):
        print("[DQNAgent] Initializing...")
        self.device = DEVICE
        self.env_config = env_config  # Store instance
        self.action_dim = env_config.ACTION_DIM  # Total output dimension
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.num_shape_slots = env_config.NUM_SHAPE_SLOTS
        self.locations_per_shape = self.rows * self.cols

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
        # --- NEW: Reference TensorBoard config for logging flags ---
        self.tb_config = TensorBoardConfig()
        # --- END NEW ---

        if self.use_distributional:
            if self.num_atoms <= 1:
                raise ValueError("NUM_ATOMS must be >= 2 for Distributional RL")
            self.support = torch.linspace(
                self.v_min, self.v_max, self.num_atoms, device=self.device
            )
            self.delta_z = (self.v_max - self.v_min) / max(1, self.num_atoms - 1)

        self.online_net = create_network(
            env_config=self.env_config,  # Pass instance
            action_dim=self.action_dim,  # Still use full output dim
            model_config=config,
            dqn_config=dqn_config,
        ).to(self.device)
        self.target_net = create_network(
            env_config=self.env_config,  # Pass instance
            action_dim=self.action_dim,
            model_config=config,
            dqn_config=dqn_config,
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

        # --- Internal state for logging ---
        self._last_avg_max_q: float = 0.0
        self._last_chosen_shape_slot: Optional[int] = None
        self._last_shape_slot_max_q_values: Optional[List[float]] = None
        self._last_placement_q_values_for_chosen_shape: Optional[List[float]] = None
        # --- End Internal state ---

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

    # --- MODIFIED: select_action implements two-stage logic ---
    @torch.no_grad()
    def select_action(
        self, state: StateType, epsilon: float, valid_actions_indices: List[ActionType]
    ) -> ActionType:
        """
        Selects action using a two-stage process based on Q-values:
        1. Find the best Q-value achievable for each available shape slot across its valid placements.
        2. Choose the shape slot with the highest maximum Q-value.
        3. Choose the placement corresponding to that maximum Q-value for the chosen shape.
        Applies epsilon-greedy if not using noisy nets.
        Logs intermediate values for debugging/visualization.
        """
        # Reset logging state
        self._last_chosen_shape_slot = None
        self._last_shape_slot_max_q_values = [-float("inf")] * self.num_shape_slots
        self._last_placement_q_values_for_chosen_shape = None

        # Fallback if no valid actions provided by environment
        if not valid_actions_indices:
            # print("Warning: No valid actions provided to select_action. Returning action 0.")
            self._last_avg_max_q = -float("inf")  # Indicate no valid Q found
            return 0  # Default action

        # Epsilon-greedy exploration (only if not using noisy nets)
        if not self.use_noisy_nets and random.random() < epsilon:
            self._last_avg_max_q = -float("inf")  # Exploration, no meaningful Q
            return random.choice(valid_actions_indices)

        # --- Get Q-values from network ---
        grid_np = ensure_numpy(state["grid"])
        shapes_np = ensure_numpy(state["shapes"])
        grid_t = torch.tensor(
            grid_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        shapes_t = torch.tensor(
            shapes_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        model_device = next(self.online_net.parameters()).device
        if grid_t.device != model_device:
            grid_t = grid_t.to(model_device)
        if shapes_t.device != model_device:
            shapes_t = shapes_t.to(model_device)

        self.online_net.eval()
        dist_or_q = self.online_net(
            grid_t, shapes_t
        )  # Shape [1, action_dim] or [1, action_dim, atoms]

        if self.use_distributional:
            probabilities = F.softmax(dist_or_q, dim=2)
            q_values = (probabilities * self.support).sum(
                dim=2
            )  # Shape: [1, action_dim]
        else:
            q_values = dist_or_q  # Shape: [1, action_dim]

        q_values_np = q_values.squeeze(0).cpu().numpy()  # Shape [action_dim]

        # --- Two-Stage Selection Logic ---
        best_overall_q = -float("inf")
        best_shape_slot = -1
        best_placement_idx = -1  # Placement index within 0 to ROWS*COLS-1
        placement_qs_for_best_shape: List[float] = []

        # Group valid actions by shape slot
        valid_actions_by_slot: Dict[int, List[int]] = {
            i: [] for i in range(self.num_shape_slots)
        }
        for action_idx in valid_actions_indices:
            s_idx = action_idx // self.locations_per_shape
            p_idx = action_idx % self.locations_per_shape
            if 0 <= s_idx < self.num_shape_slots:
                valid_actions_by_slot[s_idx].append(p_idx)

        # Find best action per shape slot
        for s_idx in range(self.num_shape_slots):
            valid_placements_for_slot = valid_actions_by_slot[s_idx]
            if not valid_placements_for_slot:
                # self._last_shape_slot_max_q_values remains -inf
                continue  # No valid placements for this shape

            # Get global indices for this slot's valid placements
            global_indices = [
                s_idx * self.locations_per_shape + p_idx
                for p_idx in valid_placements_for_slot
            ]
            q_values_for_valid_placements = q_values_np[global_indices]

            if (
                q_values_for_valid_placements.size == 0
            ):  # Should not happen if valid_placements_for_slot is not empty
                continue

            max_q_for_slot = np.max(q_values_for_valid_placements)
            best_placement_idx_for_slot = valid_placements_for_slot[
                np.argmax(q_values_for_valid_placements)
            ]

            self._last_shape_slot_max_q_values[s_idx] = float(max_q_for_slot)

            # Update overall best if this shape slot is better
            if max_q_for_slot > best_overall_q:
                best_overall_q = max_q_for_slot
                best_shape_slot = s_idx
                best_placement_idx = best_placement_idx_for_slot
                # Store Q values for all possible placements of this *potentially* best shape
                # (for logging) - Extract Qs for all placements (0 to R*C-1) for this slot
                start_g_idx = s_idx * self.locations_per_shape
                end_g_idx = start_g_idx + self.locations_per_shape
                placement_qs_for_best_shape = q_values_np[
                    start_g_idx:end_g_idx
                ].tolist()

        # --- Final Action Selection & Logging ---
        if best_shape_slot != -1:
            # Found a valid action
            final_action = (
                best_shape_slot * self.locations_per_shape + best_placement_idx
            )
            self._last_avg_max_q = float(best_overall_q)
            self._last_chosen_shape_slot = best_shape_slot
            # Store placement Qs only for the finally chosen shape
            self._last_placement_q_values_for_chosen_shape = placement_qs_for_best_shape

            # Sanity check: Ensure the chosen action was in the original valid list
            if final_action not in valid_actions_indices:
                print(
                    f"CRITICAL WARNING: Chosen action {final_action} (Shape:{best_shape_slot}, Place:{best_placement_idx}) not in original valid_actions list! Falling back."
                )
                # Fallback to random valid action to prevent crash
                self._last_avg_max_q = -float("inf")
                self._last_chosen_shape_slot = None
                self._last_placement_q_values_for_chosen_shape = None
                return random.choice(valid_actions_indices)

            return final_action
        else:
            # No valid action found across all shapes (should match initial check)
            # print("Warning: No valid action found during two-stage selection. Returning 0.")
            self._last_avg_max_q = -float("inf")
            return 0  # Fallback

    # --- END MODIFIED ---

    def _np_batch_to_tensor(
        self, batch: Union[NumpyBatch, NumpyNStepBatch], is_n_step: bool
    ) -> Union[TensorBatch, TensorNStepBatch]:
        """Converts numpy batch tuple (where states are dicts) to tensor tuple."""

        if is_n_step:
            states_dicts, actions, rewards, next_states_dicts, dones, discounts = batch
        else:
            states_dicts, actions, rewards, next_states_dicts, dones = batch[:5]
            discounts = None

        # Process states and next_states (lists of dicts)
        # Ensure grid features match the expected 2 channels
        grid_states = np.array([s["grid"] for s in states_dicts], dtype=np.float32)
        shape_states = np.array([s["shapes"] for s in states_dicts], dtype=np.float32)
        grid_next_states = np.array(
            [ns["grid"] for ns in next_states_dicts], dtype=np.float32
        )
        shape_next_states = np.array(
            [ns["shapes"] for ns in next_states_dicts], dtype=np.float32
        )

        # Validate grid state channel count
        expected_channels = self.env_config.GRID_FEATURES_PER_CELL
        if (
            grid_states.shape[1] != expected_channels
            or grid_next_states.shape[1] != expected_channels
        ):
            raise ValueError(
                f"Batch grid state channel mismatch! Expected {expected_channels}, got {grid_states.shape[1]} and {grid_next_states.shape[1]}. Check env config and state generation."
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

        states_t = (grid_s_t, shape_s_t)
        next_states_t = (grid_ns_t, shape_ns_t)

        if is_n_step:
            disc_t = torch.tensor(
                discounts, device=self.device, dtype=torch.float32
            ).unsqueeze(1)
            return states_t, a_t, r_t, next_states_t, d_t, disc_t
        else:
            return states_t, a_t, r_t, next_states_t, d_t

    # --- MODIFIED: Added placeholder for action masking in target ---
    @torch.no_grad()
    def _get_target_distribution(
        self, batch: Union[TensorBatch, TensorNStepBatch], is_n_step: bool
    ) -> torch.Tensor:
        """Calculates the target distribution for C51 using Double DQN logic."""
        # --- THIS FUNCTION IS UNUSED IF USE_DISTRIBUTIONAL is False ---
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

        # Double DQN: Use online net to select best action, target net to evaluate
        self.online_net.eval()
        online_next_dist_logits = self.online_net(
            next_grids, next_shapes
        )  # [B, A, N_atoms]
        online_next_probs = F.softmax(online_next_dist_logits, dim=2)
        online_expected_q = (online_next_probs * self.support).sum(dim=2)  # [B, A]

        # --- MASKING NEEDED HERE ---
        # Ideally, mask online_expected_q with valid actions for next_states
        # For now, we proceed without masking, acknowledging the limitation.
        # TODO: Implement masking if performance requires it.
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
        lower_idx[(upper_idx > 0) & (lower_idx == upper_idx)] -= 1
        upper_idx[(lower_idx < (self.num_atoms - 1)) & (lower_idx == upper_idx)] += 1
        lower_idx = lower_idx.clamp(0, self.num_atoms - 1)
        upper_idx = upper_idx.clamp(0, self.num_atoms - 1)

        weight_u = b - lower_idx.float()
        weight_l = 1.0 - weight_u

        target_dist = torch.zeros_like(target_next_best_dist_probs)  # [B, N_atoms]
        # Using index_add_ for projection (safer) - requires careful indexing
        # Flatten indices and values for batch operation
        batch_indices_l = (
            torch.arange(batch_size, device=self.device) * self.num_atoms
        ).unsqueeze(1) + lower_idx
        batch_indices_u = (
            torch.arange(batch_size, device=self.device) * self.num_atoms
        ).unsqueeze(1) + upper_idx
        values_l = (target_next_best_dist_probs * weight_l).view(-1)
        values_u = (target_next_best_dist_probs * weight_u).view(-1)

        target_dist.view(-1).index_add_(0, batch_indices_l.view(-1), values_l)
        target_dist.view(-1).index_add_(0, batch_indices_u.view(-1), values_u)

        return target_dist

    # --- END MODIFIED ---

    # --- MODIFIED: Log Q-values for chosen actions if enabled ---
    def compute_loss(
        self,
        batch: Union[NumpyBatch, NumpyNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Computes the loss (SmoothL1) and TD errors."""
        if self.use_distributional:
            raise NotImplementedError(
                "Distributional loss calculation needs review with two-stage selection"
            )

        tensor_batch = self._np_batch_to_tensor(batch, is_n_step)
        states_tuple, actions, rewards, next_states_tuple, dones = tensor_batch[:5]
        grids, shapes = states_tuple
        next_grids, next_shapes = next_states_tuple
        batch_size = grids.size(0)  # Get batch size

        td_errors = None

        # --- Standard DQN (Non-distributional) ---
        discounts = (
            tensor_batch[5]  # Use pre-calculated gamma^n from NStepBatch
            if is_n_step
            else torch.full_like(rewards, self.gamma)  # Use standard gamma
        )

        # Target Q-value calculation (Double DQN)
        with torch.no_grad():
            self.online_net.eval()
            online_next_q = self.online_net(next_grids, next_shapes)  # [B, A]
            # --- MASKING NEEDED HERE ---
            # TODO: Implement masking if performance requires it.
            best_next_actions = online_next_q.argmax(dim=1, keepdim=True)  # [B, 1]
            # --- END MASKING NOTE ---

            self.target_net.eval()
            target_next_q_values = self.target_net(next_grids, next_shapes)  # [B, A]
            target_q_for_best_actions = target_next_q_values.gather(
                1, best_next_actions
            )  # [B, 1]

            # Calculate TD target
            target_q = rewards + discounts * target_q_for_best_actions * (
                1.0 - dones
            )  # [B, 1]

        # Current Q-value calculation
        self.online_net.train()
        current_q_all_actions = self.online_net(grids, shapes)  # [B, A]
        current_q = current_q_all_actions.gather(1, actions)  # [B, 1]

        # Calculate element-wise loss (Huber)
        elementwise_loss = self.loss_fn(current_q, target_q)  # [B, 1]
        td_errors = (target_q - current_q).abs().detach()  # [B, 1]

        # --- Logging Q-values (Optional) ---
        # Store batch average max Q for simple logging
        with torch.no_grad():
            self.online_net.eval()
            # TODO: Apply masking here if possible for a more accurate avg_max_q
            self._last_avg_max_q = (
                self.online_net(grids, shapes).max(dim=1)[0].mean().item()
            )

        # --- Store Q-values for actions taken in the batch (for potential histogram) ---
        if self.tb_config.LOG_SHAPE_PLACEMENT_Q_VALUES:
            # Detach and store Q-values for the specific actions taken in this batch
            self._batch_q_values_for_actions_taken = (
                current_q.detach().squeeze().cpu().numpy()
            )
        else:
            self._batch_q_values_for_actions_taken = None
        # --- End Logging Q-values ---

        # Apply PER weights and calculate final loss
        if is_weights is not None:
            is_weights_t = torch.tensor(
                is_weights, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            loss = (is_weights_t * elementwise_loss).mean()
        else:
            loss = elementwise_loss.mean()

        td_errors_for_per = td_errors.squeeze() if td_errors is not None else None

        return loss, td_errors_for_per

    # --- END MODIFIED ---

    def update(self, loss: torch.Tensor) -> Optional[float]:
        """Performs optimizer step, gradient clipping, scheduler step, and noise reset."""
        grad_norm = None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.online_net.train()

        if self.gradient_clip_norm > 0:
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.online_net.parameters(),
                    max_norm=self.gradient_clip_norm,
                    error_if_nonfinite=True,
                ).item()
            except RuntimeError as clip_err:
                print(f"ERROR: Gradient clipping failed: {clip_err}")
                return None
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

    # --- NEW: Methods to retrieve logging info ---
    def get_last_avg_max_q(self) -> float:
        """Returns the average max Q value computed during the last loss calculation."""
        return self._last_avg_max_q

    def get_last_shape_selection_info(
        self,
    ) -> Tuple[Optional[int], Optional[List[float]], Optional[List[float]]]:
        """Returns info logged during the last select_action call."""
        return (
            self._last_chosen_shape_slot,
            self._last_shape_slot_max_q_values,
            self._last_placement_q_values_for_chosen_shape,
        )

    def get_last_batch_q_values_for_actions(self) -> Optional[np.ndarray]:
        """Returns the Q-values for actions taken in the last processed batch."""
        return getattr(self, "_batch_q_values_for_actions_taken", None)

    # --- END NEW ---

    def update_target_network(self):
        """Copies weights from the online network to the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def get_state_dict(self) -> AgentStateDict:
        """Returns the agent's state including networks, optimizer, and scheduler."""
        self.online_net.cpu()
        self.target_net.cpu()

        optim_state_cpu = {}
        if hasattr(self.optimizer, "state") and self.optimizer.state:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p in self.optimizer.state:
                        state = self.optimizer.state[p]
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                optim_state_cpu[id(p), k] = v.cpu()
                                state[k] = v.cpu()

        state = {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
        }

        # Restore optimizer state
        if hasattr(self.optimizer, "state") and self.optimizer.state:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p in self.optimizer.state:
                        state_opt = self.optimizer.state[p]
                        for k, v_cpu in state_opt.items():
                            if (
                                isinstance(v_cpu, torch.Tensor)
                                and (id(p), k) in optim_state_cpu
                            ):
                                original_tensor = optim_state_cpu[(id(p), k)].to(
                                    self.device
                                )
                                state_opt[k] = original_tensor

        self.online_net.to(self.device)
        self.target_net.to(self.device)

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
