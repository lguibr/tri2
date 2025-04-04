# File: agent/replay_buffer/prioritized_buffer.py
# File: agent/replay_buffer/prioritized_buffer.py
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
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = 0.0  # Annealed externally
        self.max_priority = 1.0

    def push(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool,
        **kwargs,
    ):
        """Adds new experience with maximum priority."""
        n_step_discount = kwargs.get("n_step_discount")
        transition = Transition(
            state, action, reward, next_state, done, n_step_discount
        )
        self.tree.add(self.max_priority, transition)

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
            s = max(1e-9, s)
            idx, p, data = self.tree.get(s)

            retries = 0
            max_retries = 5
            while (
                not isinstance(data, Transition) and retries < max_retries
            ):  # Retry if invalid data
                s = random.uniform(1e-9, self.tree.total())
                idx, p, data = self.tree.get(s)
                retries += 1
            if not isinstance(data, Transition):
                print(
                    f"ERROR: PER sample failed after {max_retries} retries. Skipping batch."
                )
                return None

            priorities[i] = p
            batch_data.append(data)
            indices[i] = idx

        sampling_probs = np.maximum(priorities / self.tree.total(), 1e-9)
        is_weights = np.power(len(self) * sampling_probs, -self.beta)
        is_weights = (is_weights / (is_weights.max() + 1e-9)).astype(np.float32)

        # Check if N-step based on first item
        is_n_step = batch_data[0].n_step_discount is not None
        batch_tuple = self._unpack_batch(batch_data, is_n_step)
        return batch_tuple, indices, is_weights

    def _unpack_batch(
        self, batch_data: List[Transition], is_n_step: bool
    ) -> Union[NumpyBatch, NumpyNStepBatch]:
        """Unpacks list of Transitions into numpy arrays."""
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
            return (
                np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(rn, dtype=np.float32),
                np.array(nsn, dtype=np.float32),
                np.array(dn, dtype=np.float32),
                np.array(gamma_n, dtype=np.float32),
            )
        else:
            s, a, r, ns, d = zip(
                *[
                    (t.state, t.action, t.reward, t.next_state, t.done)
                    for t in batch_data
                ]
            )
            return (
                np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32),
            )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Updates priorities of experiences at given tree indices."""
        if len(indices) != len(priorities):
            print(f"Error: Mismatch indices/priorities in PER update")
            return

        priorities = np.power(np.abs(priorities) + self.epsilon, self.alpha)
        for idx, priority in zip(indices, priorities):
            if not (self.tree.capacity - 1 <= idx < 2 * self.tree.capacity - 1):
                continue  # Skip invalid index
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def set_beta(self, beta: float):
        self.beta = beta

    def flush_pending(self):
        pass  # No-op

    def __len__(self) -> int:
        return self.tree.n_entries

    def get_state(self) -> Dict[str, Any]:
        return {
            "tree_nodes": self.tree.tree.copy(),
            "tree_data": self.tree.data.copy(),
            "tree_write_ptr": self.tree.write_ptr,
            "tree_n_entries": self.tree.n_entries,
            "max_priority": self.max_priority,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
        }

    def load_state_from_data(self, state: Dict[str, Any]):
        if "tree_nodes" not in state or "tree_data" not in state:
            print("Error: Invalid PER state format. Skipping.")
            return

        loaded_capacity = len(state["tree_data"])
        if loaded_capacity != self.capacity:
            print(
                f"Warning: Loaded PER capacity ({loaded_capacity}) != current ({self.capacity}). Recreating tree."
            )
            self.tree = SumTree(self.capacity)
            num_to_load = min(loaded_capacity, self.capacity)
            self.tree.data[:num_to_load] = state["tree_data"][:num_to_load]
            self.tree.write_ptr = state.get("tree_write_ptr", 0) % self.capacity
            self.tree.n_entries = min(state.get("tree_n_entries", 0), self.capacity)
            self.tree.tree.fill(0)
            self.max_priority = 1.0  # Reset priorities
            print(f"[PER] Loaded {self.tree.n_entries} transitions (priorities reset).")
        else:
            self.tree.tree = state["tree_nodes"]
            self.tree.data = state["tree_data"]
            self.tree.write_ptr = state.get("tree_write_ptr", 0)
            self.tree.n_entries = state.get("tree_n_entries", 0)
            self.max_priority = state.get("max_priority", 1.0)
            print(f"[PER] Loaded {self.tree.n_entries} transitions.")
        self.alpha = state.get("alpha", self.alpha)
        self.epsilon = state.get("epsilon", self.epsilon)

    def save_state(self, filepath: str):
        save_object(self.get_state(), filepath)

    def load_state(self, filepath: str):
        self.load_state_from_data(load_object(filepath))
