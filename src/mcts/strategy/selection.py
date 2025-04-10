# File: src/mcts/strategy/selection.py
import math
import numpy as np
import logging
from typing import TYPE_CHECKING, Tuple, Optional

# Use relative imports within mcts package
from ..core.node import Node
from ..core.config import MCTSConfig

logger = logging.getLogger(__name__)


def calculate_puct_score(
    child_node: Node,
    parent_visit_count: int,
    config: MCTSConfig,
    log_details: bool = False,
) -> Tuple[float, float, float]:  # Return components for logging
    """Calculates the PUCT score and its components for a child node."""
    q_value = child_node.value_estimate
    prior = child_node.prior_probability
    visits = child_node.visit_count

    if parent_visit_count == 0:
        exploration_term = config.puct_coefficient * prior
    else:
        exploration_term = (
            config.puct_coefficient
            * prior
            * (math.sqrt(parent_visit_count) / (1 + visits))
        )

    score = q_value + exploration_term

    # Use logger.debug for detailed logs
    if log_details:
        logger.debug(
            f"    Action {child_node.action_taken}: Q={q_value:.3f}, P={prior:.4f}, N={visits}, ParentN={parent_visit_count} -> ExpTerm={exploration_term:.4f} -> PUCT={score:.4f}"
        )

    return score, q_value, exploration_term


def add_dirichlet_noise(node: Node, config: MCTSConfig):
    """Adds Dirichlet noise to the prior probabilities of the children of this node."""
    if (
        config.dirichlet_alpha <= 0.0
        or config.dirichlet_epsilon <= 0.0
        or not node.children
        or len(node.children) <= 1
    ):
        return

    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.dirichlet_alpha] * len(actions))
    eps = config.dirichlet_epsilon

    for i, action in enumerate(actions):
        child = node.children[action]
        child.prior_probability = (1 - eps) * child.prior_probability + eps * noise[i]

    logger.debug(
        f"Added Dirichlet noise (alpha={config.dirichlet_alpha}, eps={eps}) to node priors."
    )


def select_child_node(node: Node, config: MCTSConfig) -> Node:
    """Selects the child node with the highest PUCT score. Assumes noise already added if root."""
    if not node.children:
        raise ValueError("Cannot select child from a node with no children.")

    best_score = -float("inf")
    best_child: Optional[Node] = None

    is_root = node.parent is None
    log_limit = 5
    logged_count = 0
    # Log details only when selecting from the root node
    if is_root:
        logger.debug(
            f"Selecting child for Node (Step {node.state.current_step}, Visits {node.visit_count}):"
        )

    children_items = list(node.children.items())

    for action, child in children_items:
        log_this_child = is_root and logged_count < log_limit
        score, q, exp_term = calculate_puct_score(
            child, node.visit_count, config, log_details=log_this_child
        )
        if log_this_child:
            logged_count += 1

        if score > best_score:
            best_score = score
            best_child = child

    if best_child is None:
        logger.error(
            f"Could not select best child for node step {node.state.current_step}. Defaulting to random."
        )
        return np.random.choice(list(node.children.values()))

    if is_root:
        # Log the Q-value of the selected child
        logger.debug(
            f"Selected Child: Action {best_child.action_taken}, Score {best_score:.4f}, Q-value {best_child.value_estimate:.3f}"
        )

    return best_child


def traverse_to_leaf(root_node: Node, config: MCTSConfig) -> Tuple[Node, int]:
    """
    Traverses the tree from root to a leaf node using PUCT selection.
    Returns the leaf node and the depth reached.
    """
    current_node = root_node
    depth = 0
    # logger.debug(f"--- Start Traverse (Root Visits: {root_node.visit_count}) ---")
    while current_node.is_expanded:
        if current_node.state.is_over():
            # logger.debug(f"  Traverse hit terminal node at depth {depth}. Node: {current_node}")
            break
        if config.max_search_depth and depth >= config.max_search_depth:
            logger.debug(f"  Traverse hit max depth {config.max_search_depth}.")
            break

        selected_child = select_child_node(current_node, config)
        # logger.debug(f"  Depth {depth}: Selected Action {selected_child.action_taken} -> Node Visits: {selected_child.visit_count}, Q: {selected_child.value_estimate:.3f}")
        current_node = selected_child
        depth += 1

    # logger.debug(f"--- End Traverse: Reached Leaf Depth {depth}. Node: {current_node} ---")
    return current_node, depth
