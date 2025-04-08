# File: ui/mcts_visualizer/tree_layout.py
from typing import Dict, Tuple, Optional, List
from mcts.node import MCTSNode
import math


class TreeLayout:
    """Calculates positions for nodes in the MCTS tree for visualization."""

    HORIZONTAL_SPACING = 50
    VERTICAL_SPACING = 80
    SUBTREE_HORIZONTAL_PADDING = 10

    def __init__(self, root_node: MCTSNode, canvas_width: int, canvas_height: int):
        self.root = root_node
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.positions: Dict[MCTSNode, Tuple[int, int]] = {}
        self.subtree_widths: Dict[MCTSNode, int] = {}

    def calculate_layout(self) -> Dict[MCTSNode, Tuple[int, int]]:
        """Calculates and returns the positions for all nodes."""
        self._calculate_subtree_widths(self.root)
        self._calculate_positions(
            self.root, self.canvas_width // 2, 50
        )  # Start root at top-center
        return self.positions

    def _calculate_subtree_widths(self, node: MCTSNode):
        """Recursively calculates the horizontal space needed for each subtree."""
        if not node.children:
            self.subtree_widths[node] = self.HORIZONTAL_SPACING
            return

        total_width = 0
        for child in node.children.values():
            self._calculate_subtree_widths(child)
            total_width += self.subtree_widths[child]

        # Add padding between subtrees
        total_width += max(0, len(node.children) - 1) * self.SUBTREE_HORIZONTAL_PADDING
        # Ensure node itself has minimum spacing
        self.subtree_widths[node] = max(total_width, self.HORIZONTAL_SPACING)

    def _calculate_positions(self, node: MCTSNode, x: int, y: int):
        """Recursively calculates the (x, y) position for each node."""
        self.positions[node] = (x, y)

        if not node.children:
            return

        num_children = len(node.children)
        total_children_width = (
            self.subtree_widths[node] - self.HORIZONTAL_SPACING
        )  # Width excluding node itself
        current_x = x - total_children_width // 2

        child_list = list(node.children.values())  # Consistent order
        for i, child in enumerate(child_list):
            child_subtree_width = self.subtree_widths[child]
            child_x = current_x + child_subtree_width // 2
            child_y = y + self.VERTICAL_SPACING
            self._calculate_positions(child, child_x, child_y)
            current_x += child_subtree_width
            if i < num_children - 1:
                current_x += self.SUBTREE_HORIZONTAL_PADDING  # Add padding
