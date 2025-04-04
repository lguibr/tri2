# File: agent/replay_buffer/sum_tree.py
# File: agent/replay_buffer/sum_tree.py
import numpy as np


class SumTree:
    """Simple SumTree implementation using numpy arrays for PER."""

    def __init__(self, capacity: int):
        if capacity <= 0 or not isinstance(capacity, int):
            raise ValueError("SumTree capacity must be positive integer")
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)  # Use float64 for sums
        self.data = np.zeros(capacity, dtype=object)  # Holds Transition objects
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index for a given cumulative priority value s."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx  # Leaf node

        # Use tolerance for float comparison
        if s <= self.tree[left] + 1e-8:
            return self._retrieve(left, s)
        else:
            return self._retrieve(
                right, max(0.0, s - self.tree[left])
            )  # Ensure non-negative s

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: object):
        """Add new experience, overwriting oldest if full."""
        priority = max(abs(priority), 1e-6)  # Ensure positive priority
        tree_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx: int, priority: float):
        """Update priority of an experience at a given tree index."""
        priority = max(abs(priority), 1e-6)
        if not (self.capacity - 1 <= tree_idx < 2 * self.capacity - 1):
            return  # Skip invalid index

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        if abs(change) > 1e-9 and tree_idx > 0:
            self._propagate(tree_idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        """Sample an experience based on cumulative priority s. Returns: (tree_idx, priority, data)"""
        if self.total() <= 0 or self.n_entries == 0:
            return 0, 0.0, None  # Handle empty tree

        s = np.clip(s, 1e-9, self.total())  # Clip s to valid range
        idx = self._retrieve(0, s)  # Leaf node index in tree array
        data_idx = idx - self.capacity + 1  # Corresponding index in data array

        # Validate data_idx before access (important if buffer not full)
        if not (0 <= data_idx < self.n_entries):
            # Fallback: return last valid entry if index is out of bounds (rare)
            if self.n_entries > 0:
                last_valid_data_idx = (
                    self.write_ptr - 1 + self.capacity
                ) % self.capacity
                last_valid_tree_idx = last_valid_data_idx + self.capacity - 1
                priority = (
                    self.tree[last_valid_tree_idx]
                    if (
                        self.capacity - 1 <= last_valid_tree_idx < 2 * self.capacity - 1
                    )
                    else 0.0
                )
                return (last_valid_tree_idx, priority, self.data[last_valid_data_idx])
            else:
                return 0, 0.0, None  # Truly empty

        return (idx, self.tree[idx], self.data[data_idx])

    def __len__(self) -> int:
        return self.n_entries
