# File: src/mcts/core/search.py
import logging
from typing import TYPE_CHECKING, List, Tuple, Set

# Use relative imports within the mcts package
from ..strategy import selection, expansion, backpropagation
from .node import Node
from .config import MCTSConfig
from .types import ActionPolicyValueEvaluator, ActionPolicyMapping

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from ...environment import GameState  # Import GameState for type hint

logger = logging.getLogger(__name__)


def run_mcts_simulations(
    root_node: Node,
    config: MCTSConfig,
    network_evaluator: ActionPolicyValueEvaluator,
) -> int:
    """
    Runs the specified number of MCTS simulations from the root node.
    Evaluates nodes individually. Includes more robust logging.

    Returns:
        The maximum tree depth reached during the simulations.
    """
    if root_node.state.is_over():
        logger.warning("MCTS started on a terminal state. No simulations run.")
        return 0

    max_depth_reached = 0
    sim_success_count = 0
    sim_error_count = 0

    # Initial expansion and noise application if root is not expanded
    if not root_node.is_expanded:
        logger.debug("Root node not expanded, performing initial evaluation...")
        try:
            action_policy, root_value = network_evaluator.evaluate(root_node.state)
            expansion.expand_node_with_policy(root_node, action_policy)
            backpropagation.backpropagate_value(root_node, root_value)  # Initial visit
            logger.debug(f"Initial root expansion complete. Value: {root_value:.3f}")
            selection.add_dirichlet_noise(root_node, config)
        except Exception as e:
            logger.error(f"Initial root expansion failed: {e}", exc_info=True)
            return 0
    elif root_node.visit_count == 0:
        logger.warning(
            "Root node expanded but visit_count is 0. Backpropagating current estimate."
        )
        backpropagation.backpropagate_value(root_node, root_node.value_estimate)

    # --- Main Simulation Loop ---
    logger.debug(f"Starting MCTS loop for {config.num_simulations} simulations...")
    for sim_idx in range(config.num_simulations):
        # Log progress every N simulations to avoid flooding
        # if (sim_idx + 1) % 100 == 0:
        #      logger.debug(f"--- Simulation {sim_idx + 1}/{config.num_simulations} ---")
        logger.debug(
            f"--- Simulation {sim_idx + 1}/{config.num_simulations} ---"
        )  # Log every sim start

        leaf_node = None  # Initialize leaf_node for the finally block
        try:
            # 1. Selection
            logger.debug(f"  Sim {sim_idx+1}: Starting selection...")
            leaf_node, depth = selection.traverse_to_leaf(root_node, config)
            max_depth_reached = max(max_depth_reached, depth)
            logger.debug(
                f"  Sim {sim_idx+1}: Selection finished at depth {depth}. Leaf: {leaf_node}"
            )

            # 2. Expansion & Evaluation (if not terminal)
            value = 0.0
            if leaf_node.state.is_over():
                value = leaf_node.state.get_outcome()
                logger.debug(
                    f"  Sim {sim_idx+1}: Leaf is terminal. Outcome: {value:.3f}"
                )
            else:
                logger.debug(f"  Sim {sim_idx+1}: Evaluating/Expanding leaf...")
                if not leaf_node.is_expanded:
                    action_policy, value = network_evaluator.evaluate(leaf_node.state)
                    expansion.expand_node_with_policy(leaf_node, action_policy)
                    logger.debug(
                        f"  Sim {sim_idx+1}: Expanded & Evaluated leaf. Network Value: {value:.3f}"
                    )
                else:
                    value = leaf_node.value_estimate
                    logger.debug(
                        f"  Sim {sim_idx+1}: Leaf already expanded (depth {depth}). Using current value estimate: {value:.3f}"
                    )

            # 3. Backpropagation
            logger.debug(
                f"  Sim {sim_idx+1}: Starting backpropagation with value {value:.4f}..."
            )
            backpropagation.backpropagate_value(leaf_node, value)
            logger.debug(f"  Sim {sim_idx+1}: Backpropagation complete.")
            sim_success_count += 1

        except Exception as e:
            sim_error_count += 1
            logger.error(
                f"Error during MCTS simulation {sim_idx + 1}: {e}", exc_info=True
            )
            # Log the leaf node state if available when error occurred
            if leaf_node:
                logger.error(f"Error occurred at leaf node: {leaf_node}")
            else:
                logger.error("Error occurred before leaf node was determined.")
            continue  # Try next simulation

        # --- End of Simulation Iteration ---
        # logger.debug(f"--- Finished Simulation {sim_idx + 1} ---") # Optional end log

    # --- Loop Finished ---
    final_log_level = logging.INFO if sim_error_count == 0 else logging.WARNING
    logger.log(
        final_log_level,
        f"MCTS loop finished. Ran {sim_success_count}/{config.num_simulations} sims ({sim_error_count} errors). "
        f"Root visits: {root_node.visit_count}. Max depth: {max_depth_reached}",
    )
    expected_visits = 1 + sim_success_count  # 1 for root + 1 per successful sim
    if root_node.visit_count != expected_visits:
        logger.warning(
            f"Root visit count ({root_node.visit_count}) does not match expected ({expected_visits})!"
        )

    return max_depth_reached
