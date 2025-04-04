# File: agent/replay_buffer/prioritized_buffer.py
# (No structural changes, cleanup comments, improved error message)
import random
import numpy as np
from typing import Optional, Tuple, Any, Dict, Union, List
from .base_buffer import ReplayBufferBase
from .sum_tree import SumTree
from utils.types import (
    Transition,
    StateType,
    ActionType,
    NumpyBatch,
    PrioritizedNumpyBatch,
    NumpyNStepBatch,
    PrioritizedNumpyNStepBatch,
)
from utils.helpers import save_object, load_object


class PrioritizedReplayBuffer(ReplayBufferBase):
    """Prioritized Experience Replay (PER) buffer using a SumTree."""

    def __init__(self, capacity: int, alpha: float, epsilon: float):
        super().__init__(capacity)
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Controls prioritization strength (0=uniform, 1=full)
        self.epsilon = (
            epsilon  # Small value added to priorities to ensure non-zero probability
        )
        self.beta = 0.0  # Importance sampling exponent (annealed externally)
        self.max_priority = 1.0  # Initial max priority

    def push(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool,
        **kwargs,  # Accept potential n_step_discount
    ):
        """Adds new experience with maximum priority."""
        n_step_discount = kwargs.get("n_step_discount")
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            n_step_discount=n_step_discount,
        )
        # Add with max priority initially, will be updated after first training step
        self.tree.add(self.max_priority, transition)

    def sample(
        self, batch_size: int
    ) -> Optional[Union[PrioritizedNumpyBatch, PrioritizedNumpyNStepBatch]]:
        """Samples batch using priorities, calculates IS weights."""
        if len(self) < batch_size:
            return None

        batch_data: List[Transition] = []
        indices = np.empty(batch_size, dtype=np.int64)  # Tree indices
        priorities = np.empty(batch_size, dtype=np.float64)
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            s = max(1e-9, s)  # Avoid s=0 issues

            idx, p, data = self.tree.get(s)

            # Retry sampling if data is invalid (should be rare with proper init)
            retries = 0
            max_retries = 5
            while not isinstance(data, Transition) and retries < max_retries:
                # Resample from the entire range if the segment failed
                s = random.uniform(1e-9, self.tree.total())
                idx, p, data = self.tree.get(s)
                retries += 1

            if not isinstance(data, Transition):
                print(
                    f"ERROR: PER sample failed to get valid data after {max_retries} retries (total entries: {len(self)}, tree total: {self.tree.total():.4f}). Skipping batch."
                )
                return None  # Return None if any sample fails

            priorities[i] = p
            batch_data.append(data)
            indices[i] = idx

        sampling_probabilities = priorities / self.tree.total()
        sampling_probabilities = np.maximum(
            sampling_probabilities, 1e-9
        )  # Epsilon for stability

        # Calculate Importance Sampling (IS) weights
        is_weights = np.power(len(self) * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max() + 1e-9  # Normalize weights

        # Check if N-step based on first item
        is_n_step = batch_data[0].n_step_discount is not None

        if is_n_step:
            s, a, rn, nsn, dn, gamma_n = zip(
                *[
                    (
                        t.state,
                        t.action,
                        t.reward,
                        t.next_state,
                        t.done,
                        t.n_step_discount,
                    )
                    for t in batch_data
                ]
            )
            states_np = np.array(s, dtype=np.float32)
            actions_np = np.array(a, dtype=np.int64)
            rewards_np = np.array(rn, dtype=np.float32)
            next_states_np = np.array(nsn, dtype=np.float32)
            dones_np = np.array(dn, dtype=np.float32)
            discounts_np = np.array(gamma_n, dtype=np.float32)
            batch_tuple = (
                states_np,
                actions_np,
                rewards_np,
                next_states_np,
                dones_np,
                discounts_np,
            )
            return batch_tuple, indices, is_weights.astype(np.float32)
        else:
            s, a, r, ns, d = zip(
                *[
                    (t.state, t.action, t.reward, t.next_state, t.done)
                    for t in batch_data
                ]
            )
            states_np = np.array(s, dtype=np.float32)
            actions_np = np.array(a, dtype=np.int64)
            rewards_np = np.array(r, dtype=np.float32)
            next_states_np = np.array(ns, dtype=np.float32)
            dones_np = np.array(d, dtype=np.float32)
            batch_tuple = (states_np, actions_np, rewards_np, next_states_np, dones_np)
            return batch_tuple, indices, is_weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Updates priorities of experiences at given tree indices using TD errors."""
        if len(indices) != len(priorities):
            print(
                f"Error: Mismatch indices ({len(indices)}) vs priorities ({len(priorities)}) in PER update"
            )
            return

        # Use absolute TD error for priority, add epsilon, raise to alpha
        priorities = np.abs(priorities) + self.epsilon
        priorities = np.power(priorities, self.alpha)

        for idx, priority in zip(indices, priorities):
            # Index should be leaf node index from sampling
            if not (self.tree.capacity - 1 <= idx < 2 * self.tree.capacity - 1):
                # This might happen if buffer wraps around. Log silently or with low severity.
                # print(f"Debug: Attempting update on invalid tree index {idx}. Skipping.")
                continue
            self.tree.update(idx, priority)
            self.max_priority = max(
                self.max_priority, priority
            )  # Update max priority seen

    def set_beta(self, beta: float):
        self.beta = beta

    def flush_pending(self):
        pass  # No-op for this buffer

    def __len__(self) -> int:
        return self.tree.n_entries

    def get_state(self) -> Dict[str, Any]:
        """Return state for saving."""
        return {
            "tree_nodes": self.tree.tree.copy(),
            "tree_data": self.tree.data.copy(),  # Actual transition data
            "tree_write_ptr": self.tree.write_ptr,
            "tree_n_entries": self.tree.n_entries,
            "max_priority": self.max_priority,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            # Beta is transient, set by trainer
        }

    def load_state_from_data(self, state: Dict[str, Any]):
        """Load state from dictionary."""
        if "tree_nodes" not in state or "tree_data" not in state:
            print("Error: Invalid PER state format during load. Skipping.")
            return

        loaded_capacity = len(state["tree_data"])
        if loaded_capacity != self.capacity:
            print(
                f"Warning: Loaded PER capacity ({loaded_capacity}) != current buffer capacity ({self.capacity}). Recreating tree structure."
            )
            # Recreate tree with current capacity and load data partially
            self.tree = SumTree(self.capacity)
            num_to_load = min(loaded_capacity, self.capacity)
            # Simple load - just copy data, priorities will reset.
            # A complex load would rebuild tree priorities, harder if capacity changed.
            self.tree.data[:num_to_load] = state["tree_data"][:num_to_load]
            self.tree.write_ptr = (
                state.get("tree_write_ptr", 0) % self.capacity
            )  # Ensure valid ptr
            self.tree.n_entries = min(state.get("tree_n_entries", 0), self.capacity)
            self.tree.tree.fill(0)  # Clear old priorities
            self.max_priority = 1.0  # Reset max priority
            print(
                f"[PrioritizedReplayBuffer] Loaded {self.tree.n_entries} transitions (priorities reset due to capacity mismatch)."
            )

        else:
            # Capacities match, load everything
            self.tree.tree = state["tree_nodes"]
            self.tree.data = state["tree_data"]
            self.tree.write_ptr = state.get("tree_write_ptr", 0)
            self.tree.n_entries = state.get("tree_n_entries", 0)
            self.max_priority = state.get("max_priority", 1.0)
            print(
                f"[PrioritizedReplayBuffer] Loaded {self.tree.n_entries} transitions."
            )

        # Load config params if they exist in save, otherwise keep current config
        self.alpha = state.get("alpha", self.alpha)
        self.epsilon = state.get("epsilon", self.epsilon)

    def save_state(self, filepath: str):
        state = self.get_state()
        save_object(state, filepath)

    def load_state(self, filepath: str):
        state = load_object(filepath)
        self.load_state_from_data(state)
