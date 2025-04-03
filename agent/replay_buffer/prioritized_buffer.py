import random
import numpy as np
from typing import (
    Optional,
    Tuple,
    Any,
    Dict,
    Union,
    List,
)
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
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = 0.0  # Set externally by Trainer
        self.max_priority = 1.0

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

        # <<< MODIFIED >>> sampling loop for efficiency
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            s = max(1e-9, s)  # Avoid issues with s=0

            idx, p, data = self.tree.get(s)

            # Retry sampling if data is invalid (e.g., None during buffer fill)
            retries = 0
            while not isinstance(data, Transition) and retries < 5:
                s = random.uniform(1e-9, self.tree.total())
                idx, p, data = self.tree.get(s)
                retries += 1
            if not isinstance(data, Transition):
                print(
                    f"Error: PER sample failed to get valid data after retries. Skipping sample point."
                )
                # How to handle? Can't easily resize numpy arrays here.
                # Could return None, or return a smaller batch? Let's return None if any sample fails.
                return None

            priorities[i] = p
            batch_data.append(data)
            indices[i] = idx

        sampling_probabilities = priorities / self.tree.total()
        sampling_probabilities = np.maximum(
            sampling_probabilities, 1e-9
        )  # Avoid division by zero

        is_weights = np.power(len(self) * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max() + 1e-9  # Normalize

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
        """Updates priorities of experiences at given tree indices."""
        if len(indices) != len(priorities):
            print(
                f"Error: Mismatch indices ({len(indices)}) vs priorities ({len(priorities)})"
            )
            return

        # Use TD error magnitude for priority, add epsilon, raise to alpha
        priorities = np.abs(priorities) + self.epsilon
        priorities = np.power(priorities, self.alpha)

        for idx, priority in zip(indices, priorities):
            # Check index validity (should be leaf node index)
            if not (self.tree.capacity - 1 <= idx < 2 * self.tree.capacity - 1):
                # print(f"Warning: Attempting update on invalid tree index {idx}. Skipping.")
                continue  # Silently skip invalid indices that might occur near capacity limits?
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def set_beta(self, beta: float):
        self.beta = beta

    def flush_pending(self):
        pass  # No-op

    def __len__(self) -> int:
        return self.tree.n_entries

    # <<< NEW >>> Save/Load state
    def get_state(self) -> Dict[str, Any]:
        """Return state for saving."""
        return {
            "tree_nodes": self.tree.tree.copy(),  # Save sumtree node values
            "tree_data": self.tree.data.copy(),  # Save actual transition data
            "tree_write_ptr": self.tree.write_ptr,
            "tree_n_entries": self.tree.n_entries,
            "max_priority": self.max_priority,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            # Beta is transient, set by trainer based on global step
        }

    def load_state_from_data(self, state: Dict[str, Any]):
        """Load state from dictionary."""
        # Simple validation
        if "tree_nodes" not in state or "tree_data" not in state:
            print("Error: Invalid PER state format during load. Skipping.")
            return

        loaded_capacity = len(state["tree_data"])
        if loaded_capacity != self.capacity:
            print(
                f"Warning: Loaded PER capacity ({loaded_capacity}) != current buffer capacity ({self.capacity}). Adjusting."
            )
            # Recreate tree with loaded capacity? Or keep current and load partially?
            # Let's try to adapt to the loaded data's capacity if possible.
            self.capacity = loaded_capacity
            self.tree = SumTree(self.capacity)  # Recreate tree

        self.tree.tree = state["tree_nodes"]
        self.tree.data = state["tree_data"]
        self.tree.write_ptr = state.get("tree_write_ptr", 0)
        self.tree.n_entries = state.get("tree_n_entries", 0)
        self.max_priority = state.get("max_priority", 1.0)
        self.alpha = state.get("alpha", self.alpha)  # Keep config alpha if not saved?
        self.epsilon = state.get("epsilon", self.epsilon)
        print(f"[PrioritizedReplayBuffer] Loaded {self.tree.n_entries} transitions.")

    def save_state(self, filepath: str):
        """Save buffer state to file."""
        state = self.get_state()
        save_object(state, filepath)

    def load_state(self, filepath: str):
        """Load buffer state from file."""
        state = load_object(filepath)
        self.load_state_from_data(state)
