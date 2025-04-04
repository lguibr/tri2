# File: agent/action_selector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Optional, Tuple

from config import EnvConfig, DQNConfig, DEVICE
from environment.game_state import StateType
from utils.types import ActionType
from utils.helpers import ensure_numpy


class ActionSelector:
    """Handles the two-stage action selection process for the DQNAgent."""

    def __init__(
        self,
        online_net: nn.Module,
        env_config: EnvConfig,
        dqn_config: DQNConfig,
        device: torch.device,
    ):
        self.online_net = online_net
        self.env_config = env_config
        self.dqn_config = dqn_config
        self.device = device
        self.num_shape_slots = env_config.NUM_SHAPE_SLOTS
        self.locations_per_shape = env_config.ROWS * env_config.COLS
        self.use_distributional = dqn_config.USE_DISTRIBUTIONAL
        self.use_noisy_nets = dqn_config.USE_NOISY_NETS

        if self.use_distributional:
            self.support = torch.linspace(
                dqn_config.V_MIN,
                dqn_config.V_MAX,
                dqn_config.NUM_ATOMS,
                device=self.device,
            )

        self._last_avg_max_q: float = -float("inf")
        self._last_chosen_shape_slot: Optional[int] = None
        self._last_shape_slot_max_q_values: List[float] = [
            -float("inf")
        ] * self.num_shape_slots
        self._last_placement_q_values_for_chosen_shape: Optional[List[float]] = None

    @torch.no_grad()
    def select_action(
        self, state: StateType, epsilon: float, valid_actions_indices: List[ActionType]
    ) -> ActionType:
        """Selects action using a two-stage process based on Q-values."""
        self._reset_log_state()

        if not valid_actions_indices:
            self._last_avg_max_q = -float("inf")
            return 0

        if not self.use_noisy_nets and random.random() < epsilon:
            self._last_avg_max_q = -float("inf")
            return random.choice(valid_actions_indices)

        q_values_np = self._get_q_values_for_state(state)
        return self._select_best_action_from_q_values(
            q_values_np, valid_actions_indices
        )

    def _reset_log_state(self):
        self._last_avg_max_q = -float("inf")
        self._last_chosen_shape_slot = None
        self._last_shape_slot_max_q_values = [-float("inf")] * self.num_shape_slots
        self._last_placement_q_values_for_chosen_shape = None

    def _get_q_values_for_state(self, state: StateType) -> np.ndarray:
        """Calculates Q-values (or expected Q-values for C51) for a given state."""
        grid_np = ensure_numpy(state["grid"])
        shapes_np = ensure_numpy(state["shapes"])
        grid_t = torch.tensor(
            grid_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        shapes_t = torch.tensor(
            shapes_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        model_device = next(self.online_net.parameters()).device
        grid_t = grid_t.to(model_device)
        shapes_t = shapes_t.to(model_device)

        self.online_net.eval()
        dist_or_q = self.online_net(grid_t, shapes_t)

        if self.use_distributional:
            probabilities = F.softmax(dist_or_q, dim=2)
            q_values = (probabilities * self.support).sum(dim=2)
        else:
            q_values = dist_or_q

        return q_values.squeeze(0).cpu().numpy()

    def _select_best_action_from_q_values(
        self, q_values_np: np.ndarray, valid_actions_indices: List[ActionType]
    ) -> ActionType:
        """Implements the two-stage selection logic."""
        best_overall_q = -float("inf")
        best_shape_slot = -1
        best_placement_idx = -1
        placement_qs_for_best_shape: List[float] = []

        valid_actions_by_slot: Dict[int, List[int]] = {
            i: [] for i in range(self.num_shape_slots)
        }
        for action_idx in valid_actions_indices:
            s_idx = action_idx // self.locations_per_shape
            p_idx = action_idx % self.locations_per_shape
            if 0 <= s_idx < self.num_shape_slots:
                valid_actions_by_slot[s_idx].append(p_idx)

        for s_idx in range(self.num_shape_slots):
            valid_placements_for_slot = valid_actions_by_slot[s_idx]
            if not valid_placements_for_slot:
                continue

            global_indices = [
                s_idx * self.locations_per_shape + p_idx
                for p_idx in valid_placements_for_slot
            ]
            q_values_for_valid_placements = q_values_np[global_indices]

            if q_values_for_valid_placements.size == 0:
                continue

            max_q_for_slot = np.max(q_values_for_valid_placements)
            best_placement_idx_for_slot = valid_placements_for_slot[
                np.argmax(q_values_for_valid_placements)
            ]

            self._last_shape_slot_max_q_values[s_idx] = float(max_q_for_slot)

            if max_q_for_slot > best_overall_q:
                best_overall_q = max_q_for_slot
                best_shape_slot = s_idx
                best_placement_idx = best_placement_idx_for_slot
                start_g_idx = s_idx * self.locations_per_shape
                end_g_idx = start_g_idx + self.locations_per_shape
                placement_qs_for_best_shape = q_values_np[
                    start_g_idx:end_g_idx
                ].tolist()

        if best_shape_slot != -1:
            final_action = (
                best_shape_slot * self.locations_per_shape + best_placement_idx
            )
            self._last_avg_max_q = float(best_overall_q)
            self._last_chosen_shape_slot = best_shape_slot
            self._last_placement_q_values_for_chosen_shape = placement_qs_for_best_shape

            if final_action not in valid_actions_indices:
                print(
                    f"CRITICAL WARNING: Chosen action {final_action} not in valid_actions! Falling back."
                )
                self._reset_log_state()
                return random.choice(valid_actions_indices)

            return final_action
        else:
            self._last_avg_max_q = -float("inf")
            return 0

    def get_last_avg_max_q(self) -> float:
        return self._last_avg_max_q

    def get_last_shape_selection_info(
        self,
    ) -> Tuple[Optional[int], Optional[List[float]], Optional[List[float]]]:
        return (
            self._last_chosen_shape_slot,
            self._last_shape_slot_max_q_values,
            self._last_placement_q_values_for_chosen_shape,
        )
