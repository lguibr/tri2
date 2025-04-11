# File: src/mcts/strategy/selection.py
import math
import numpy as np
import logging
import random  # Import random for fallback choice
from typing import TYPE_CHECKING, Tuple, Optional

# Use relative imports within mcts package
from ..core.node import Node

# Change: Import MCTSConfig from the central config location
from src.config import MCTSConfig

logger = logging.getLogger(__name__)


def calculate_puct_score(
    child_node: Node,
    parent_visit_count: int,
    config: MCTSConfig,  # Type hint uses the imported config
    log_details: bool = False,  # Keep log_details flag
) -> Tuple[float, float, float]:  # Return components for logging
    """Calculates the PUCT score and its components for a child node."""
    q_value = child_node.value_estimate
    prior = child_node.prior_probability
    visits = child_node.visit_count

    # Handle case where parent_visit_count might be 0 (e.g., root before backprop)
    # Add small epsilon to denominator to prevent division by zero if visits=0
    # Use sqrt(max(1, parent_visit_count)) to handle parent_visit_count=0 gracefully
    exploration_term = (
        config.puct_coefficient
        * prior
        * (math.sqrt(max(1, parent_visit_count)) / (1 + visits))
    )

    score = q_value + exploration_term

    # Logging is handled by the caller (select_child_node) based on logger level

    return score, q_value, exploration_term


def add_dirichlet_noise(
    node: Node, config: MCTSConfig
):  # Type hint uses the imported config
    """Adds Dirichlet noise to the prior probabilities of the children of this node."""
    if (
        config.dirichlet_alpha <= 0.0
        or config.dirichlet_epsilon <= 0.0
        or not node.children
        or len(node.children) <= 1  # No noise needed if only one action
    ):
        return

    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.dirichlet_alpha] * len(actions))
    eps = config.dirichlet_epsilon

    noisy_priors_sum = 0.0
    for i, action in enumerate(actions):
        child = node.children[action]
        original_prior = child.prior_probability  # Store original for logging if needed
        child.prior_probability = (1 - eps) * child.prior_probability + eps * noise[i]
        noisy_priors_sum += child.prior_probability
        # logger.debug(f"  Noise Action {action}: OrigP={original_prior:.4f}, Noise={noise[i]:.4f} -> NewP={child.prior_probability:.4f}")

    # Optional: Re-normalize priors after adding noise (though dirichlet sum is 1, mixing might slightly change sum)
    # if abs(noisy_priors_sum - 1.0) > 1e-6:
    #     logger.debug(f"Re-normalizing priors after noise. Sum was {noisy_priors_sum:.6f}")
    #     for action in actions:
    #         node.children[action].prior_probability /= noisy_priors_sum

    logger.debug(
        f"Added Dirichlet noise (alpha={config.dirichlet_alpha}, eps={eps}) to {len(actions)} root node priors."
    )


def select_child_node(
    node: Node, config: MCTSConfig
) -> Node:  # Type hint uses the imported config
    """Selects the child node with the highest PUCT score. Assumes noise already added if root."""
    if not node.children:
        raise ValueError("Cannot select child from a node with no children.")

    best_score = -float("inf")
    best_child: Optional[Node] = None

    # Enhanced logging for child selection - controlled by logger level
    # Log the parent node's state before iterating children
    log_msg_parent = f"  Selecting child for Node (Visits={node.visit_count}, Children={len(node.children)}, StateStep={node.state.current_step}):"
    logger.debug(log_msg_parent)  # Keep detailed selection start as DEBUG

    children_items = list(node.children.items())

    for action, child in children_items:
        # Always calculate components for potential logging
        score, q, exp_term = calculate_puct_score(
            child,
            node.visit_count,
            config,
            log_details=True,  # Pass True to enable calculation, logging depends on level
        )
        # Log the calculated score for each child if logger level is DEBUG
        log_msg_child = f"    Child Action {action}: Q={q:.3f}, P={child.prior_probability:.4f}, N={child.visit_count}, ParentN={node.visit_count} -> ExpTerm={exp_term:.4f} -> PUCT={score:.4f}"
        logger.debug(log_msg_child)  # Keep per-child PUCT as DEBUG

        if score > best_score:
            best_score = score
            best_child = child

    if best_child is None:
        # This should ideally not happen if there are children.
        # Could occur if all scores are -inf (e.g., invalid priors/values).
        error_msg = f"Could not select best child for node step {node.state.current_step} (all scores -inf?). Defaulting to random."
        logger.error(error_msg)
        # Fallback to random choice among children
        return random.choice(list(node.children.values()))

    # Log the selected child as DEBUG
    log_msg_selected = f"  --> Selected Child: Action {best_child.action_taken}, Score {best_score:.4f}, Q-value {best_child.value_estimate:.3f}"
    logger.debug(log_msg_selected)

    return best_child


def traverse_to_leaf(
    root_node: Node, config: MCTSConfig
) -> Tuple[Node, int]:  # Type hint uses the imported config
    """
    Traverses the tree from root to a leaf node using PUCT selection.
    A leaf is defined as a node that is not expanded OR is terminal.
    Stops also if the maximum search depth has been reached.
    Returns the leaf node and the depth reached. Includes detailed logging.
    """
    current_node = root_node
    depth = 0
    # CHANGE: MCTS traversal start log to DEBUG
    logger.debug(f"--- Start Traverse (Root Visits: {root_node.visit_count}) ---")

    # **MODIFIED LOOP CONDITION**
    while current_node.is_expanded and not current_node.state.is_over():
        # CHANGE: MCTS node consideration log to DEBUG
        log_msg_consider = f"  Depth {depth}: Considering Node: {current_node}"
        logger.debug(log_msg_consider)

        # Check max depth condition inside the loop
        if config.max_search_depth and depth >= config.max_search_depth:
            # CHANGE: MCTS max depth hit log to DEBUG
            log_msg_break = f"  Depth {depth}: Hit MAX DEPTH ({config.max_search_depth}). Breaking traverse."
            logger.debug(log_msg_break)
            break  # Stop traversal if max depth reached

        # If node is expanded and non-terminal, select next child
        log_msg_select = (
            f"  Depth {depth}: Node is expanded and non-terminal. Selecting child..."
        )
        logger.debug(log_msg_select)  # Keep selection intent as DEBUG
        try:
            selected_child = select_child_node(current_node, config)
            current_node = selected_child
            depth += 1
        except Exception as e:
            log_msg_err = f"  Depth {depth}: Error during child selection: {e}. Breaking traverse."
            logger.error(log_msg_err, exc_info=True)
            break  # Stop traversal if selection fails

    # After loop, current_node is either unexpanded, terminal, or at max depth
    if not current_node.is_expanded and not current_node.state.is_over():
        # CHANGE: MCTS leaf reached log to DEBUG
        log_msg_break = f"  Depth {depth}: Node is LEAF (not expanded). Final node."
        logger.debug(log_msg_break)
    elif current_node.state.is_over():
        # CHANGE: MCTS terminal node reached log to DEBUG
        log_msg_break = f"  Depth {depth}: Node is TERMINAL. Final node."
        logger.debug(log_msg_break)
    # Max depth case logged inside loop

    # CHANGE: MCTS traversal end log to DEBUG
    log_msg_end = f"--- End Traverse: Reached Node at Depth {depth}. Final Node: {current_node} ---"
    logger.debug(log_msg_end)
    return current_node, depth
