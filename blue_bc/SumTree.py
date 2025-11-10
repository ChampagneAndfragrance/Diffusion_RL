"""A small SumTree implementation used for prioritized experience replay.

This module provides the SumTree class, a binary tree stored in an array
where each parent node is the sum of its two children. It is commonly used
to efficiently sample elements proportional to their priority.

Reference usage: store experience tuples in `data` and their sampling
priorities in `tree`. The `get` method retrieves a leaf given a sample value
in [0, total_priority).
"""

import numpy


class SumTree:
    """Binary indexed sum tree for proportional sampling.

    Attributes:
        capacity (int): Number of leaf nodes (maximum number of stored entries).
        tree (numpy.ndarray): Array representation of the binary tree of size
            `2 * capacity - 1`. The root at index 0 stores the total sum.
        data (numpy.ndarray): Array of stored data objects of length `capacity`.
        write (int): Rolling write index for the circular buffer of `data`.
        n_entries (int): Number of entries written so far (<= capacity).
    """

    write = 0

    def __init__(self, capacity):
        """Create a SumTree with the requested leaf capacity.

        Args:
            capacity (int): Maximum number of elements the tree can hold.
        """
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propagate a change in a leaf node upward to the root.

        The tree stores sums at internal nodes; when a leaf's priority is
        updated by `change`, this method updates all affected parent nodes.

        Args:
            idx (int): Index in `tree` where the change originated.
            change (float): Difference between new and old priority.
        """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Recursively search for a leaf corresponding to sample value s.

        The method walks the tree starting from `idx` (normally 0) and
        descends left or right depending on whether `s` falls into the
        left-subtree cumulative sum. When a leaf is reached, its index is
        returned.

        Args:
            idx (int): Current tree index (start with 0 for the root).
            s (float): Value in [0, total_priority) used to sample a leaf.

        Returns:
            int: Index in `tree` of the selected leaf node.
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Return the total sum of priorities (value at the root).

        Returns:
            float: Sum of all priorities stored in the tree.
        """
        return self.tree[0]

    def add(self, p, data):
        """Add a new data sample with priority p into the tree.

        The method writes `data` into the circular `data` buffer at index
        `self.write` and updates the corresponding leaf node priority.

        Args:
            p (float): Priority value for the new sample.
            data (object): Arbitrary python object to store alongside the priority.
        """
        # Map the write index to the corresponding leaf index in the tree
        idx = self.write + (self.capacity - 1)

        self.data[self.write] = data
        self.update(idx, p)

        # advance circular buffer write pointer
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1


    def update(self, idx, p):
        """Update the priority at tree index `idx` to `p` and propagate change.

        Args:
            idx (int): Index in `tree` to update (leaf index for a data entry).
            p (float): New priority value.
        """
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Get the leaf index, priority and data for a sample value `s`.

        Args:
            s (float): A value in [0, total_priority) used to select a leaf.

        Returns:
            tuple: (tree_index, priority, data_object)
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


