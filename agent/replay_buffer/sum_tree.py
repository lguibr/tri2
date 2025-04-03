import numpy as np


class SumTree:
    """
    Simple SumTree implementation using numpy arrays.
    Tree structure: [root] [layer 1] [layer 2] ... [leaves]
    Size = 2 * capacity - 1. Leaves start at index capacity - 1.
    Data array holds corresponding experiences.
    """

    def __init__(self, capacity: int):
        if capacity <= 0 or not isinstance(capacity, int):
            raise ValueError("SumTree capacity must be a positive integer")
        # Ensure capacity is power of 2 for simpler indexing? Not strictly required.
        self.capacity = capacity
        self.tree = np.zeros(
            2 * capacity - 1, dtype=np.float64
        )  # Use float64 for precision
        # dtype=object allows storing arbitrary Transition objects
        self.data = np.zeros(capacity, dtype=object)
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

        if left >= len(self.tree):  # Leaf node found
            return idx

        # Handle edge case where tree[left] might be slightly larger due to fp error
        if s <= self.tree[left] + 1e-8:  # Added tolerance
            return self._retrieve(left, s)
        else:
            # Ensure s subtraction doesn't go negative due to fp errors
            s_new = s - self.tree[left]
            # if s_new < 0: s_new = 0 # Clamp if necessary (shouldn't happen often)
            return self._retrieve(right, s_new)

    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0]

    def add(self, priority: float, data: object):
        """Add new experience, overwriting oldest if full."""
        if priority < 0:
            priority = abs(priority) + 1e-6  # Ensure positive

        tree_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)  # Update priority in tree

        self.write_ptr += 1
        if self.write_ptr >= self.capacity:
            self.write_ptr = 0  # Wrap around

        # Only increment n_entries up to capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx: int, priority: float):
        """Update priority of an experience at a given tree index."""
        if priority < 0:
            priority = abs(priority) + 1e-6  # Ensure positive
        if not (self.capacity - 1 <= tree_idx < 2 * self.capacity - 1):
            # print(f"Warning: Invalid tree index {tree_idx} for update. Capacity {self.capacity}. Skipping.")
            return  # Silently skip invalid index updates

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate change only if it's significant enough to avoid fp noise
        # Or always propagate? Always propagate seems safer.
        if abs(change) > 1e-9 and tree_idx > 0:
            self._propagate(tree_idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        """Sample an experience (returns tree_idx, priority, data)."""
        if self.total() <= 0 or self.n_entries == 0:
            # print("Warning: Sampling from empty or zero-priority SumTree.")
            # Need to return valid types even if empty
            return 0, 0.0, None

        # Clip s to valid range [epsilon, total] to avoid issues at boundaries
        s = np.clip(s, 1e-9, self.total())

        idx = self._retrieve(0, s)  # Get leaf node index in the tree array
        data_idx = idx - self.capacity + 1  # Corresponding index in data array

        # Validate data_idx before access
        if not (0 <= data_idx < self.n_entries):
            # This can happen if sampling races with adding near capacity
            # Fallback: sample again or return last valid entry? Return last valid entry.
            # print(f"Warning: SumTree get resulted in invalid data index {data_idx} (n_entries={self.n_entries}). Returning last valid entry.")
            if self.n_entries > 0:
                last_valid_data_idx = (
                    self.write_ptr - 1 + self.capacity
                ) % self.capacity
                last_valid_tree_idx = last_valid_data_idx + self.capacity - 1
                # Ensure last valid priority is read correctly
                priority = (
                    self.tree[last_valid_tree_idx]
                    if self.capacity - 1 <= last_valid_tree_idx < 2 * self.capacity - 1
                    else 0.0
                )
                return (last_valid_tree_idx, priority, self.data[last_valid_data_idx])
            else:  # Truly empty
                return 0, 0.0, None

        # Return tree_idx, priority from tree, data from data array
        return (idx, self.tree[idx], self.data[data_idx])

    def __len__(self) -> int:
        return self.n_entries
