# File: agent/replay_buffer/prioritized_buffer.py
import random
import numpy as np
from typing import Optional, Tuple, Any, Dict, Union, List
from .base_buffer import ReplayBufferBase
from .sum_tree import SumTree

# --- MODIFIED: Import specific StateType ---
from environment.game_state import StateType  # Use the Dict type

# --- END MODIFIED ---
from utils.types import (
    Transition,
    # StateType, # Removed
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
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = 0.0
        self.max_priority = 1.0

    def push(
        self,
        state: StateType,  # State is Dict
        action: ActionType,
        reward: float,
        next_state: StateType,  # Next state is Dict
        done: bool,
        **kwargs,
    ):
        """Adds new experience with maximum priority."""
        n_step_discount = kwargs.get("n_step_discount")
        transition = Transition(
            state, action, reward, next_state, done, n_step_discount
        )
        # Add with current max priority to ensure new samples get seen
        self.tree.add(self.max_priority, transition)

    # --- MODIFIED: Sample returns dict states ---
    def sample(
        self, batch_size: int
    ) -> Optional[Union[PrioritizedNumpyBatch, PrioritizedNumpyNStepBatch]]:
        """Samples batch using priorities, calculates IS weights."""
        if len(self) < batch_size:
            return None

        batch_data: List[Transition] = []
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            s = max(1e-9, s)  # Avoid zero
            idx, p, data = self.tree.get(s)

            retries = 0
            max_retries = 5
            while not isinstance(data, Transition) and retries < max_retries:
                print(
                    f"Warning: PER sample retrieved invalid data (type: {type(data)}). Retrying..."
                )
                s = random.uniform(1e-9, self.tree.total())
                idx, p, data = self.tree.get(s)
                retries += 1

            if not isinstance(data, Transition):
                print(
                    f"ERROR: PER sample failed after {max_retries} retries. Skipping batch."
                )
                # This indicates a potential issue with the SumTree or data storage
                return None

            priorities[i] = p
            batch_data.append(data)
            indices[i] = idx

        # Calculate Importance Sampling (IS) weights
        sampling_probs = np.maximum(
            priorities / self.tree.total(), 1e-9
        )  # Avoid division by zero
        is_weights = np.power(len(self) * sampling_probs, -self.beta)
        # Normalize weights by max weight for stability
        is_weights = (is_weights / (is_weights.max() + 1e-9)).astype(np.float32)

        # Check if N-step based on first item
        is_n_step = batch_data[0].n_step_discount is not None
        batch_tuple = self._unpack_batch(batch_data, is_n_step)  # Returns dict states

        return batch_tuple, indices, is_weights

    # --- END MODIFIED ---

    # --- MODIFIED: Unpack returns dict states ---
    def _unpack_batch(
        self, batch_data: List[Transition], is_n_step: bool
    ) -> Union[NumpyBatch, NumpyNStepBatch]:
        """Unpacks list of Transitions into numpy arrays, keeping states as dicts."""
        states_dicts = [t.state for t in batch_data]
        actions_np = np.array([t.action for t in batch_data], dtype=np.int64)
        rewards_np = np.array([t.reward for t in batch_data], dtype=np.float32)
        next_states_dicts = [t.next_state for t in batch_data]
        dones_np = np.array([t.done for t in batch_data], dtype=np.float32)

        if is_n_step:
            discounts_np = np.array(
                [t.n_step_discount for t in batch_data], dtype=np.float32
            )
            return (
                states_dicts,
                actions_np,
                rewards_np,
                next_states_dicts,
                dones_np,
                discounts_np,
            )
        else:
            return states_dicts, actions_np, rewards_np, next_states_dicts, dones_np

    # --- END MODIFIED ---

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Updates priorities of experiences at given tree indices."""
        if len(indices) != len(priorities):
            print(
                f"Error: Mismatch indices ({len(indices)}) / priorities ({len(priorities)}) in PER update"
            )
            return

        # Apply alpha transformation to TD errors
        priorities = np.power(np.abs(priorities) + self.epsilon, self.alpha)
        priorities = np.maximum(priorities, 1e-6)  # Ensure positive priorities

        for idx, priority in zip(indices, priorities):
            # Validate index corresponds to a leaf node in the tree structure
            if not (self.tree.capacity - 1 <= idx < 2 * self.tree.capacity - 1):
                # print(f"Warning: Invalid tree index {idx} provided to update_priorities. Skipping.")
                continue
            self.tree.update(idx, priority)
            self.max_priority = max(
                self.max_priority, priority
            )  # Update max priority seen

    def set_beta(self, beta: float):
        self.beta = beta

    def flush_pending(self):
        pass  # No-op for core PER buffer

    def __len__(self) -> int:
        return self.tree.n_entries

    def get_state(self) -> Dict[str, Any]:
        # Ensure data contains serializable items (should be Transitions)
        serializable_data = [
            d if isinstance(d, Transition) else None for d in self.tree.data
        ]
        return {
            "tree_nodes": self.tree.tree.copy(),
            "tree_data": serializable_data,  # Save potentially filtered list
            "tree_write_ptr": self.tree.write_ptr,
            "tree_n_entries": self.tree.n_entries,
            "max_priority": self.max_priority,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            # Beta is transient, no need to save
        }

    def load_state_from_data(self, state: Dict[str, Any]):
        if "tree_nodes" not in state or "tree_data" not in state:
            print("Error: Invalid PER state format during load. Skipping.")
            self.tree = SumTree(self.capacity)  # Reinitialize tree
            self.max_priority = 1.0
            return

        loaded_capacity = len(state["tree_data"])
        if loaded_capacity != self.capacity:
            print(
                f"Warning: Loaded PER capacity ({loaded_capacity}) != current ({self.capacity}). Recreating tree."
            )
            self.tree = SumTree(self.capacity)
            num_to_load = min(loaded_capacity, self.capacity)
            # Load only valid transitions
            valid_data = [
                d for d in state["tree_data"][:num_to_load] if isinstance(d, Transition)
            ]
            self.tree.data[: len(valid_data)] = valid_data
            self.tree.write_ptr = state.get("tree_write_ptr", 0) % self.capacity
            self.tree.n_entries = len(
                valid_data
            )  # Correct n_entries based on valid data
            # Rebuild tree priorities from loaded data (expensive but necessary if capacity changed)
            print("Rebuilding SumTree priorities from loaded data...")
            self.tree.tree.fill(0)  # Clear existing tree sums
            self.max_priority = 1.0  # Reset max priority
            for i in range(self.tree.n_entries):
                # Assign default max priority during rebuild
                self.tree.update(i + self.capacity - 1, self.max_priority)
            print(f"[PER] Rebuilt tree with {self.tree.n_entries} transitions.")

        else:  # Capacities match
            self.tree.tree = state["tree_nodes"]
            # Load data, filtering invalid entries
            valid_data = [
                d for d in state["tree_data"] if isinstance(d, Transition) or d is None
            ]  # Allow None placeholders
            if len(valid_data) != len(state["tree_data"]):
                print(
                    f"Warning: Filtered {len(state['tree_data']) - len(valid_data)} invalid items from PER data during load."
                )
            self.tree.data = np.array(valid_data, dtype=object)  # Ensure numpy array
            self.tree.write_ptr = state.get("tree_write_ptr", 0)
            self.tree.n_entries = state.get("tree_n_entries", 0)
            # Ensure n_entries is consistent with actual data
            actual_entries = sum(1 for d in self.tree.data if isinstance(d, Transition))
            if self.tree.n_entries != actual_entries:
                print(
                    f"Warning: Correcting PER n_entries from {self.tree.n_entries} to {actual_entries}"
                )
                self.tree.n_entries = actual_entries

            self.max_priority = state.get("max_priority", 1.0)
            print(f"[PER] Loaded {self.tree.n_entries} transitions.")

        self.alpha = state.get("alpha", self.alpha)
        self.epsilon = state.get("epsilon", self.epsilon)

    def save_state(self, filepath: str):
        save_object(self.get_state(), filepath)

    def load_state(self, filepath: str):
        state = load_object(filepath)
        self.load_state_from_data(state)
