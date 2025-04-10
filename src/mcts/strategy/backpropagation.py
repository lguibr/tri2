# File: src/mcts/strategy/backpropagation.py
import logging
from typing import TYPE_CHECKING

# Use relative imports within mcts package
if TYPE_CHECKING:
    from ..core.node import Node

logger = logging.getLogger(__name__)


def backpropagate_value(leaf_node: "Node", value: float) -> None:
    """
    Propagates the simulation value back up the tree from the leaf node.
    """
    current_node: "Node" | None = leaf_node
    path_str = []  # For debugging path
    depth = 0

    while current_node is not None:
        q_before = current_node.value_estimate
        current_node.visit_count += 1
        current_node.total_action_value += value
        q_after = current_node.value_estimate
        path_str.append(
            f"N(a={current_node.action_taken},v={current_node.visit_count},q={q_after:.3f})"
        )

        # Log value changes, especially near the root
        if depth <= 2:  # Log details for nodes close to the leaf
            logger.debug(
                f"  Backprop Depth {depth}: Node(Act={current_node.action_taken}, V={current_node.visit_count}), Value={value:.3f}, Q_before={q_before:.3f}, Q_after={q_after:.3f}"
            )

        current_node = current_node.parent
        depth += 1

    # logger.debug(f"Backpropagated value {value:.3f} up path: {' <- '.join(reversed(path_str))}")
