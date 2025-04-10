# File: src/mcts/core/node.py
from __future__ import annotations  # For type hinting Node within Node
import math
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment import GameState  # Keep GameState import
    from src.utils.types import ActionType


class Node:
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(
        self,
        state: "GameState",
        parent: Optional[Node] = None,
        action_taken: Optional["ActionType"] = None,  # Action that *led* to this state
        prior_probability: float = 0.0,  # P(action_taken | parent.state)
    ):
        self.state = state  # The game state this node represents
        self.parent = parent  # Parent node in the tree
        self.action_taken = action_taken  # Action taken from parent to reach this node

        # Children nodes, keyed by the action taken *from this node's state*
        self.children: Dict["ActionType", Node] = {}

        # MCTS statistics
        self.visit_count: int = 0
        self.total_action_value: float = (
            0.0  # Sum of values from simulations passing through here
        )
        self.prior_probability: float = (
            prior_probability  # Prior prob of the action *leading* to this node
        )

    @property
    def is_expanded(self) -> bool:
        """Checks if the node has been expanded (i.e., children generated)."""
        return bool(self.children)

    @property
    def is_leaf(self) -> bool:
        """Checks if the node is a leaf (not expanded)."""
        return not self.is_expanded

    @property
    def value_estimate(self) -> float:
        """
        Calculates the Q-value (average action value) estimate for this node's state.
        This is the average value observed from simulations starting from this state.
        """
        if self.visit_count == 0:
            # Return 0 or perhaps an initial estimate? AlphaZero uses 0.
            return 0.0
        # Average value = Total value accumulated / Number of visits
        return self.total_action_value / self.visit_count

    def __repr__(self) -> str:
        parent_action = self.parent.action_taken if self.parent else "Root"
        return (
            f"Node(StateStep={self.state.current_step}, "
            f"FromAction={self.action_taken}, Visits={self.visit_count}, "
            f"Value={self.value_estimate:.3f}, Prior={self.prior_probability:.4f}, "
            f"Children={len(self.children)})"
        )
