# File: src/utils/sumtree.py
# NEW FILE
import numpy as np
from typing import Tuple

# Import Experience type hint carefully
from .types import Experience  # Use relative import within utils


class SumTree:
    """
    Simple SumTree implementation for efficient prioritized sampling.
    Stores priorities and allows sampling proportional to priority.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree structure: Internal nodes store sum of priorities of children.
        # Leaf nodes store priorities of individual experiences.
        # Size is 2*capacity - 1 (leaves + internal nodes)
        self.tree = np.zeros(2 * capacity - 1)
        # Data storage (parallel array to leaf nodes)
        # Now stores Experience tuples which contain StateType (numpy arrays)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0
        self._max_priority = 1.0  # Track max priority for new entries

    def add(self, priority: float, data: Experience):
        """Adds an experience with a given priority."""
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # Wrap around

        if self.n_entries < self.capacity:
            self.n_entries += 1

        self._max_priority = max(self._max_priority, priority)

    def update(self, tree_idx: int, priority: float):
        """Updates the priority of an experience at a given tree index."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value: float) -> Tuple[int, float, Experience]:
        """Finds the leaf node corresponding to a given value (for sampling)."""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            # If we reach bottom, end the search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a leaf node
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        # Ensure data is returned correctly, handle potential empty slots if needed
        # The caller (ExperienceBuffer.sample) should handle cases where data[data_idx] is invalid
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        """Returns the total priority (root node value)."""
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        """Returns the maximum priority seen so far."""
        return self._max_priority if self.n_entries > 0 else 1.0

    def __len__(self) -> int:
        return self.n_entries
