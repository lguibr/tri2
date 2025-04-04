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
    ActionType,
    NumpyBatch,
    PrioritizedNumpyBatch,  # Added
    NumpyNStepBatch,
    PrioritizedNumpyNStepBatch,  # Added
)
from utils.helpers import save_object, load_object


class PrioritizedReplayBuffer(ReplayBufferBase):
    """Prioritized Experience Replay (PER) buffer using a SumTree."""

    def __init__(self, capacity: int, alpha: float, epsilon: float):
        super().__init__(capacity)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = 0.0  # Initial beta, will be annealed by Trainer
        self.max_priority = 1.0  # Initial max priority

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

    def sample(
        self, batch_size: int
    ) -> Optional[Union[PrioritizedNumpyBatch, PrioritizedNumpyNStepBatch]]:
        """Samples batch using priorities, calculates IS weights."""
        if len(self) < batch_size:
            # print(f"PER Sample Warning: Not enough samples ({len(self)} < {batch_size})")
            return None

        batch_data: List[Transition] = []
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            # Sample uniformly from each segment
            s = random.uniform(segment * i, segment * (i + 1))
            # Ensure s is within valid range, slightly above 0
            s = max(1e-9, min(s, self.tree.total()))

            try:
                idx, p, data = self.tree.get(s)
            except Exception as e:
                print(
                    f"ERROR during SumTree.get(s={s}, total={self.tree.total()}): {e}"
                )
                return None  # Cannot proceed if sampling fails

            retries = 0
            max_retries = 5
            # Handle cases where data might be invalid (e.g., None during fill)
            while not isinstance(data, Transition) and retries < max_retries:
                # print(f"Warning: PER sample retrieved invalid data (type: {type(data)}). Retrying...")
                # Resample from the entire range if invalid data found
                s_retry = random.uniform(1e-9, self.tree.total())
                try:
                    idx, p, data = self.tree.get(s_retry)
                except Exception as e:
                    print(f"ERROR during SumTree.get() retry: {e}")
                    return None
                retries += 1

            if not isinstance(data, Transition):
                print(
                    f"ERROR: PER sample failed after {max_retries} retries. Skipping batch."
                )
                # This could indicate a persistent issue with the SumTree or data storage
                return None

            priorities[i] = p
            batch_data.append(data)
            indices[i] = idx

        # Calculate Importance Sampling (IS) weights
        sampling_probs = np.maximum(
            priorities / self.tree.total(), 1e-9
        )  # Avoid division by zero
        # is_weights = np.power(len(self) * sampling_probs, -self.beta)
        # Use current number of entries (n_entries) instead of capacity (len(self))
        num_entries = max(
            1, len(self)
        )  # Avoid division by zero if buffer is empty somehow
        is_weights = np.power(num_entries * sampling_probs, -self.beta)

        # Normalize weights by max weight for stability (clip weights for safety)
        is_weights = is_weights / (is_weights.max() + 1e-9)
        is_weights = np.clip(is_weights, 1e-6, 100.0).astype(np.float32)

        # Check if N-step based on first item
        # Note: Assumes all items in batch are either N-step or 1-step consistently
        is_n_step = batch_data[0].n_step_discount is not None
        batch_tuple = self._unpack_batch(batch_data, is_n_step)  # Returns dict states

        return batch_tuple, indices, is_weights

    def _unpack_batch(
        self, batch_data: List[Transition], is_n_step: bool
    ) -> Union[NumpyBatch, NumpyNStepBatch]:
        """Unpacks list of Transitions into numpy arrays, keeping states as dicts."""
        # --- Keeps states as list of dictionaries ---
        states_dicts = [t.state for t in batch_data]
        actions_np = np.array([t.action for t in batch_data], dtype=np.int64)
        rewards_np = np.array([t.reward for t in batch_data], dtype=np.float32)
        next_states_dicts = [t.next_state for t in batch_data]
        dones_np = np.array(
            [t.done for t in batch_data], dtype=np.float32
        )  # Use float32 for dones

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
            return (states_dicts, actions_np, rewards_np, next_states_dicts, dones_np)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Updates priorities of experiences at given tree indices."""
        if len(indices) != len(priorities):
            print(
                f"Error: Mismatch indices ({len(indices)}) / priorities ({len(priorities)}) in PER update"
            )
            return

        # Apply alpha transformation to TD errors before updating the tree
        priorities = np.power(np.abs(priorities) + self.epsilon, self.alpha)
        priorities = np.maximum(priorities, 1e-6)  # Ensure positive priorities

        for idx, priority in zip(indices, priorities):
            # Validate index corresponds to a leaf node in the tree structure
            if not (self.tree.capacity - 1 <= idx < 2 * self.tree.capacity - 1):
                # print(f"Warning: Invalid tree index {idx} provided to update_priorities. Skipping.")
                continue
            self.tree.update(idx, priority)
            # Update max priority seen so far
            self.max_priority = max(self.max_priority, priority)

    def set_beta(self, beta: float):
        """Updates the beta exponent for IS weight calculation."""
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
            # Beta is transient, derived from training progress, no need to save
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
            # Populate the data array directly
            self.tree.data[: len(valid_data)] = valid_data
            self.tree.write_ptr = state.get("tree_write_ptr", 0) % self.capacity
            self.tree.n_entries = len(
                valid_data
            )  # Correct n_entries based on valid data loaded
            # Rebuild tree priorities from loaded data (expensive but necessary if capacity changed)
            print("Rebuilding SumTree priorities from loaded data...")
            self.tree.tree.fill(0)  # Clear existing tree sums
            self.max_priority = 1.0  # Reset max priority
            # Re-calculate priorities based on existing data - assume max priority for loaded data
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
            # Ensure n_entries is consistent with actual data after filtering
            actual_entries = sum(1 for d in self.tree.data if isinstance(d, Transition))
            if self.tree.n_entries != actual_entries:
                print(
                    f"Warning: Correcting PER n_entries from {self.tree.n_entries} to {actual_entries}"
                )
                self.tree.n_entries = actual_entries

            self.max_priority = state.get("max_priority", 1.0)
            print(f"[PER] Loaded {self.tree.n_entries} transitions.")

        # Load hyperparameters
        self.alpha = state.get("alpha", self.alpha)
        self.epsilon = state.get("epsilon", self.epsilon)
        # Beta is set by Trainer based on current step, not loaded

    def save_state(self, filepath: str):
        save_object(self.get_state(), filepath)

    def load_state(self, filepath: str):
        state = load_object(filepath)
        self.load_state_from_data(state)
