# File: agent/replay_buffer/sum_tree.py
# (No structural changes, cleanup comments, fixed potential float comparison issue)
import numpy as np


class SumTree:
    """
    Simple SumTree implementation using numpy arrays for Prioritized Experience Replay.
    Tree structure: [root] [internal nodes] ... [leaves]
    Array size = 2 * capacity - 1. Leaves start at index capacity - 1.
    """

    def __init__(self, capacity: int):
        if capacity <= 0 or not isinstance(capacity, int):
            raise ValueError("SumTree capacity must be a positive integer")
        self.capacity = capacity
        # Use float64 for priority sums to minimize precision errors
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        # Holds the actual experience data (e.g., Transition objects)
        self.data = np.zeros(capacity, dtype=object)
        self.write_ptr = 0  # Current position to write new data
        self.n_entries = 0  # Number of valid entries currently in the buffer

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

        if left >= len(self.tree):  # Leaf node
            return idx

        # Use a small tolerance for floating point comparison
        if s <= self.tree[left] + 1e-8:
            return self._retrieve(left, s)
        else:
            # Ensure s subtraction doesn't go negative due to fp errors
            s_new = max(0.0, s - self.tree[left])
            return self._retrieve(right, s_new)

    def total(self) -> float:
        """Get the total priority sum (value of the root node)."""
        return self.tree[0]

    def add(self, priority: float, data: object):
        """Add new experience, overwriting oldest if buffer is full."""
        priority = max(abs(priority), 1e-6)  # Ensure positive priority

        tree_idx = self.write_ptr + self.capacity - 1  # Index in the tree array
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)  # Update priority in tree

        self.write_ptr = (
            self.write_ptr + 1
        ) % self.capacity  # Advance ptr with wrap-around

        if self.n_entries < self.capacity:
            self.n_entries += 1  # Increment count until full

    def update(self, tree_idx: int, priority: float):
        """Update priority of an experience at a given tree index."""
        priority = max(abs(priority), 1e-6)  # Ensure positive priority

        # Validate index refers to a leaf node
        if not (self.capacity - 1 <= tree_idx < 2 * self.capacity - 1):
            # print(f"Warning: Invalid tree index {tree_idx} for update. Capacity {self.capacity}. Skipping.")
            return  # Silently skip invalid index updates

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up the tree if it's significant
        if abs(change) > 1e-9 and tree_idx > 0:
            self._propagate(tree_idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        """Sample an experience based on cumulative priority s.
        Returns: (tree_idx, priority, data)
        """
        if self.total() <= 0 or self.n_entries == 0:
            # print("Warning: Sampling from empty or zero-priority SumTree.")
            return 0, 0.0, None  # Return valid types even if empty

        # Clip s to valid range [epsilon, total]
        s = np.clip(s, 1e-9, self.total())

        idx = self._retrieve(0, s)  # Leaf node index in the tree array
        data_idx = idx - self.capacity + 1  # Corresponding index in data array

        # Validate data_idx before access (important if buffer not full)
        if not (0 <= data_idx < self.n_entries):
            # This can happen due to floating point issues near boundaries or
            # if sampling races with adding near capacity.
            # Fallback: return the last valid entry added?
            # print(f"Warning: SumTree get resulted in invalid data index {data_idx} (n_entries={self.n_entries}, total_p={self.total():.4f}, s={s:.4f}). Falling back.")
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
            else:  # Truly empty
                return 0, 0.0, None

        # Return tree_idx, priority from tree, data from data array
        return (idx, self.tree[idx], self.data[data_idx])

    def __len__(self) -> int:
        """Number of valid entries in the buffer."""
        return self.n_entries
