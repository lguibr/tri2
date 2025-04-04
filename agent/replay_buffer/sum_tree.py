# File: agent/replay_buffer/sum_tree.py
# (Content as provided by user, assuming it's correct and functional)
import numpy as np


class SumTree:
    """Simple SumTree implementation using numpy arrays for PER."""

    def __init__(self, capacity: int):
        if capacity <= 0 or not isinstance(capacity, int):
            raise ValueError("SumTree capacity must be positive integer")
        # Ensure capacity is power of 2 for simpler implementation, or handle general case
        # For simplicity here, we assume any capacity works with the logic.
        # A more robust implementation might pad capacity to the next power of 2.
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

        # Use tolerance for float comparison if needed, but direct comparison often fine
        if s <= self.tree[left] + 1e-8:  # Added tolerance
            return self._retrieve(left, s)
        else:
            # Ensure non-negative s for recursive call
            return self._retrieve(right, max(0.0, s - self.tree[left]))

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: object):
        """Add new experience, overwriting oldest if full."""
        priority = max(abs(priority), 1e-6)  # Ensure positive priority
        tree_idx = self.write_ptr + self.capacity - 1

        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)  # Update priorities after inserting data

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx: int, priority: float):
        """Update priority of an experience at a given tree index."""
        priority = max(abs(priority), 1e-6)  # Ensure positive priority

        # Ensure the index is within the valid range for leaf nodes
        if not (self.capacity - 1 <= tree_idx < 2 * self.capacity - 1):
            # print(f"Warning: Invalid tree index {tree_idx} passed to update.")
            return  # Silently ignore invalid indices or raise error

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate change only if it's significant and not the root
        if abs(change) > 1e-9 and tree_idx > 0:
            self._propagate(tree_idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        """Sample an experience based on cumulative priority s. Returns: (tree_idx, priority, data)"""
        if self.total() <= 0 or self.n_entries == 0:
            # Handle empty tree case gracefully
            return 0, 0.0, None

        # Clip s to be within valid range [epsilon, total_priority]
        s = np.clip(s, 1e-9, self.total())

        idx = self._retrieve(0, s)  # Find leaf node index
        data_idx = (
            idx - self.capacity + 1
        )  # Convert tree leaf index to data array index

        # Validate data_idx before accessing self.data
        if not (0 <= data_idx < self.n_entries):
            # This case might happen if sampling occurs while buffer is filling,
            # or due to floating point inaccuracies. Fallback strategy:
            if self.n_entries > 0:
                # Sample the last valid entry as a fallback
                last_valid_data_idx = (
                    self.write_ptr - 1 + self.capacity
                ) % self.capacity
                last_valid_tree_idx = last_valid_data_idx + self.capacity - 1
                # Ensure fallback index is valid before accessing tree
                priority = (
                    self.tree[last_valid_tree_idx]
                    if (
                        self.capacity - 1 <= last_valid_tree_idx < 2 * self.capacity - 1
                    )
                    else 0.0
                )
                # print(f"Warning: PER get() resulted in invalid data_idx {data_idx} for s={s}. Falling back to index {last_valid_data_idx}.")
                return (last_valid_tree_idx, priority, self.data[last_valid_data_idx])
            else:
                return 0, 0.0, None  # Return nothing if truly empty

        # Return valid data
        return (idx, self.tree[idx], self.data[data_idx])

    def __len__(self) -> int:
        return self.n_entries
