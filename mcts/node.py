import math
import numpy as np
from typing import Dict, Optional, TYPE_CHECKING, Any, List

# Assuming GameState is hashable or identifiable
from environment.game_state import GameState
from utils.types import ActionType
from config import MCTSConfig  # Import from config package


class MCTSNode:
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(
        self,
        game_state: GameState,
        parent: Optional["MCTSNode"] = None,
        action_taken: Optional[ActionType] = None,
        prior: float = 0.0,
        config: Optional[MCTSConfig] = None,  # Pass config for PUCT_C
    ):
        self.game_state = game_state
        self.parent = parent
        self.action_taken = action_taken

        self.children: Dict[ActionType, "MCTSNode"] = {}
        self.is_expanded: bool = False
        self.is_terminal: bool = game_state.is_over()

        self.visit_count: int = 0
        self.total_action_value: float = 0.0
        self.mean_action_value: float = 0.0
        self.prior: float = prior

        self._config = config if config else MCTSConfig()  # Use default if None

    def get_ucb_score(self) -> float:
        """Calculates the PUCT score for this node (from the perspective of its parent)."""
        if self.parent is None:
            return self.mean_action_value  # Root node score

        exploration_bonus = (
            self._config.PUCT_C
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        q_value = self.mean_action_value
        return q_value + exploration_bonus

    def select_best_child(self) -> "MCTSNode":
        """Selects the child with the highest UCB score."""
        if not self.children:
            raise ValueError("Cannot select best child from a node with no children.")
        # Simple way to handle potential ties: add small random noise or just pick first max
        best_score = -float("inf")
        best_children = []
        for child in self.children.values():
            score = child.get_ucb_score()
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        if not best_children:
            raise RuntimeError("Could not select a best child node.")
        # Randomly pick among the best children in case of ties
        return np.random.choice(best_children)

    def backpropagate(self, value: float):
        """Updates the visit count and action value of this node and its ancestors."""
        node: Optional[MCTSNode] = self
        while node is not None:
            node.visit_count += 1
            node.total_action_value += value
            node.mean_action_value = node.total_action_value / node.visit_count
            node = node.parent
