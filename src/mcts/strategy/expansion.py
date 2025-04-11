# File: src/mcts/strategy/expansion.py
import logging
from typing import TYPE_CHECKING, List

# Use relative imports within mcts package
from ..core.node import Node  # Import Node relatively
from ..core.types import (
    ActionPolicyMapping,
    ActionPolicyValueEvaluator,
)  # Import evaluator
from ...utils.types import ActionType  # Core ActionType

logger = logging.getLogger(__name__)


def expand_node_with_policy(node: Node, action_policy: ActionPolicyMapping):
    """
    Expands a node by creating children for valid actions using the
    pre-computed action policy priors from the network.
    Assumes the node is not terminal and not already expanded.
    """
    if node.is_expanded:
        # This might happen if max_depth is reached, log as debug instead of warning
        logger.debug(f"Attempted to expand an already expanded node: {node}")
        return
    if node.state.is_over():
        logger.warning(f"Attempted to expand a terminal node: {node}")
        return

    valid_actions: List[ActionType] = node.state.valid_actions()

    if not valid_actions:
        logger.warning(
            f"Expanding node at step {node.state.current_step} with no valid actions but not terminal?"
        )
        # Mark the node's state as game over if expansion reveals no moves
        node.state.game_over = True
        return

    # Create child nodes for valid actions using the provided policy
    children_created = 0
    for action in valid_actions:
        prior = action_policy.get(action, 0.0)
        if prior < 0.0:
            logger.warning(
                f"Received negative prior ({prior}) for action {action}. Clamping to 0."
            )
            prior = 0.0
        elif prior == 0.0:
            # It's possible for the network to assign zero probability to a valid action
            logger.debug(f"Valid action {action} received prior=0 from network.")

        # Create child node - state is not strictly needed until selection/expansion of the child
        # Let's generate state here for simplicity.
        next_state_copy = node.state.copy()
        try:
            # Step the copied state to represent the child's state
            _, _, _ = next_state_copy.step(action)
        except Exception as e:
            logger.error(
                f"Error stepping state for child node expansion (action {action}): {e}",
                exc_info=True,
            )
            continue  # Skip creating this child if stepping fails

        child = Node(
            state=next_state_copy,
            parent=node,
            action_taken=action,
            prior_probability=prior,
        )
        node.children[action] = child
        children_created += 1

    # logger.debug(f"Expanded node {node} with {children_created} children.")


# --- Function for simplified (non-batched) search - kept for potential debugging ---
def expand_leaf_node(node: Node, network_evaluator: ActionPolicyValueEvaluator):
    """
    Evaluates a leaf node using the network and expands it.
    (Used by the simplified, non-batched search loop).
    """
    if node.is_expanded:
        logger.debug(f"Node already expanded: {node}")
        return
    if node.state.is_over():
        logger.warning(f"Attempted to expand a terminal node: {node}")
        return

    try:
        action_policy, value = network_evaluator.evaluate(node.state)
        expand_node_with_policy(node, action_policy)
        logger.debug(f"Expanded leaf node. Network Value: {value:.3f}")
        # Note: The value is returned by the search loop and backpropagated there.
    except Exception as e:
        logger.error(
            f"Network evaluation failed during leaf expansion: {e}", exc_info=True
        )
        # How to handle? Node remains unexpanded. Search might select it again.
        # Backpropagation in the search loop will use value=0.0 in case of error.