# File: src/mcts/core/search.py
import logging
import time
from typing import TYPE_CHECKING, List, Tuple, Set, Dict

# Use relative imports within the mcts package
from ..strategy import selection, expansion, backpropagation
from .node import Node
from .config import MCTSConfig
from .types import ActionPolicyValueEvaluator, ActionPolicyMapping, PolicyValueOutput

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from ...environment import GameState  # Import GameState for type hint

logger = logging.getLogger(__name__)

# Configuration for batching within MCTS
MCTS_BATCH_SIZE = 8  # How many leaves to collect before evaluating


def run_mcts_simulations(
    root_node: Node,
    config: MCTSConfig,
    network_evaluator: ActionPolicyValueEvaluator,
) -> int:
    """
    Runs the specified number of MCTS simulations from the root node.
    Uses BATCHED evaluation of leaf nodes for potentially improved performance.
    Ensures unique nodes are evaluated per batch.

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
            # Evaluate the root node individually first
            action_policy, root_value = network_evaluator.evaluate(root_node.state)
            expansion.expand_node_with_policy(root_node, action_policy)
            backpropagation.backpropagate_value(root_node, root_value)  # Initial visit
            logger.debug(f"Initial root expansion complete. Value: {root_value:.3f}")
            selection.add_dirichlet_noise(root_node, config)
        except Exception as e:
            logger.error(f"Initial root expansion failed: {e}", exc_info=True)
            return 0  # Cannot proceed if root expansion fails
    elif root_node.visit_count == 0:
        # This case might occur if loaded from checkpoint but not visited yet in this run
        logger.warning(
            "Root node expanded but visit_count is 0. Backpropagating current estimate."
        )
        backpropagation.backpropagate_value(root_node, root_node.value_estimate)

    # --- Main Simulation Loop ---
    # KEEP INFO: Core MCTS loop start log
    logger.info(
        f"Starting MCTS loop for {config.num_simulations} simulations (Batch Size: {MCTS_BATCH_SIZE})..."
    )
    sim_count = 0
    while sim_count < config.num_simulations:
        start_time_batch = time.monotonic()
        leaves_to_evaluate: List[Node] = []
        terminal_leaves: List[Tuple[Node, float]] = (
            []
        )  # Store terminal leaves and their outcomes
        # **MODIFICATION: Track unique nodes added to this batch for evaluation**
        unique_leaves_in_batch: Set[Node] = set()

        # 1. Selection Phase (Collect a batch of leaves)
        # KEEP INFO: MCTS batch start log
        logger.info(
            f"--- MCTS Batch Starting (Sim {sim_count+1} / {config.num_simulations}) ---"
        )
        selection_start_time = time.monotonic()
        num_collected_for_batch = (
            0  # Track how many unique leaves collected for this batch
        )
        while (
            num_collected_for_batch < MCTS_BATCH_SIZE
            and sim_count < config.num_simulations
        ):
            sim_count += 1
            # KEEP INFO: MCTS simulation selection start log
            logger.info(f"  Starting Sim {sim_count} Selection...")
            try:
                leaf_node, depth = selection.traverse_to_leaf(root_node, config)
                max_depth_reached = max(max_depth_reached, depth)

                if leaf_node.state.is_over():
                    outcome = leaf_node.state.get_outcome()
                    terminal_leaves.append((leaf_node, outcome))
                    # KEEP INFO: MCTS terminal leaf log
                    logger.info(
                        f"  Sim {sim_count}: Selected TERMINAL leaf at depth {depth}. Outcome: {outcome:.3f}"
                    )
                elif leaf_node.is_expanded:
                    # KEEP INFO: MCTS expanded leaf hit log
                    logger.info(
                        f"  Sim {sim_count}: Selected EXPANDED leaf at depth {depth}. Value: {leaf_node.value_estimate:.3f}. Backpropagating immediately."
                    )
                    backpropagation.backpropagate_value(
                        leaf_node, leaf_node.value_estimate
                    )
                    sim_success_count += 1
                else:
                    # **MODIFICATION: Check uniqueness before adding to batch**
                    if leaf_node not in unique_leaves_in_batch:
                        leaves_to_evaluate.append(leaf_node)
                        unique_leaves_in_batch.add(leaf_node)
                        num_collected_for_batch += 1
                        # KEEP INFO: MCTS unique leaf found log
                        logger.info(
                            f"  Sim {sim_count}: Selected UNIQUE leaf for EVALUATION at depth {depth}. Node: {leaf_node}. Batch size now: {num_collected_for_batch}"
                        )
                    else:
                        # If node is already in batch, we still need to backpropagate *something*
                        # Using its current estimate is reasonable, similar to hitting an expanded node.
                        # KEEP INFO: MCTS duplicate leaf in batch log
                        logger.info(
                            f"  Sim {sim_count}: Selected leaf ALREADY IN BATCH at depth {depth}. Value: {leaf_node.value_estimate:.3f}. Backpropagating immediately."
                        )
                        backpropagation.backpropagate_value(
                            leaf_node, leaf_node.value_estimate
                        )
                        sim_success_count += 1

            except Exception as e:
                sim_error_count += 1
                logger.error(
                    f"Error during MCTS selection phase (Sim {sim_count}): {e}",
                    exc_info=True,
                )
                # Continue to next simulation

        selection_duration = time.monotonic() - selection_start_time
        # KEEP INFO: MCTS selection phase summary log
        logger.info(
            f"Selection phase finished. Collected {len(leaves_to_evaluate)} unique leaves for NN eval, {len(terminal_leaves)} terminal. Duration: {selection_duration:.4f}s"
        )

        # 2. Batch Evaluation & Expansion
        evaluation_start_time = time.monotonic()
        if leaves_to_evaluate:
            # KEEP INFO: MCTS batch evaluation start log
            logger.info(
                f"  Evaluating batch of {len(leaves_to_evaluate)} unique leaves..."
            )
            try:
                leaf_states = [node.state for node in leaves_to_evaluate]
                batch_results: List[PolicyValueOutput] = (
                    network_evaluator.evaluate_batch(leaf_states)
                )

                if len(batch_results) != len(leaves_to_evaluate):
                    raise ValueError(
                        f"Mismatch between evaluated results ({len(batch_results)}) and leaves provided ({len(leaves_to_evaluate)})"
                    )

                for i, node in enumerate(leaves_to_evaluate):
                    action_policy, value = batch_results[i]
                    # **MODIFICATION: Check if node was already expanded by another sim in the *previous* batch cycle**
                    # This check prevents the "Attempted to expand already expanded node" log if tree reuse is working correctly
                    # across MCTS steps, but expansion might still happen multiple times if the *same* leaf is hit
                    # multiple times *within* the selection phase of a single MCTS step (handled above by unique_leaves_in_batch).
                    if not node.is_expanded:
                        expansion.expand_node_with_policy(node, action_policy)
                    else:
                        # This case should be less common now with the uniqueness check during selection,
                        # but might occur if max_depth was hit previously.
                        logger.debug(
                            f"  Node {node.action_taken} was already expanded before batch eval completed (likely hit max depth previously). Skipping expansion."
                        )

                    node._temp_value_for_backprop = value  # type: ignore
                # KEEP INFO: MCTS batch evaluation summary log
                logger.info(
                    f"  Batch evaluated and expanded {len(leaves_to_evaluate)} unique leaves."
                )

            except Exception as e:
                sim_error_count += len(leaves_to_evaluate)
                logger.error(
                    f"Error during MCTS batch evaluation/expansion: {e}", exc_info=True
                )
                for node in leaves_to_evaluate:
                    node._temp_value_for_backprop = 0.0  # type: ignore

        evaluation_duration = time.monotonic() - evaluation_start_time
        # KEEP INFO: MCTS evaluation phase duration log
        logger.info(
            f"Evaluation/Expansion phase finished. Duration: {evaluation_duration:.4f}s"
        )

        # 3. Backpropagation
        backprop_start_time = time.monotonic()
        # KEEP INFO: MCTS backpropagation start log
        logger.info(
            f"  Backpropagating {len(leaves_to_evaluate)} evaluated leaves and {len(terminal_leaves)} terminal leaves..."
        )
        # Backpropagate values from evaluated leaves (unique ones that were evaluated)
        for node in leaves_to_evaluate:
            value = getattr(node, "_temp_value_for_backprop", 0.0)
            backpropagation.backpropagate_value(node, value)
            sim_success_count += 1
            if hasattr(node, "_temp_value_for_backprop"):
                del node._temp_value_for_backprop  # type: ignore

        # Backpropagate outcomes from terminal leaves
        for node, outcome in terminal_leaves:
            backpropagation.backpropagate_value(node, outcome)
            sim_success_count += 1

        backprop_duration = time.monotonic() - backprop_start_time
        batch_duration = time.monotonic() - start_time_batch
        # KEEP INFO: MCTS backpropagation phase summary log
        logger.info(
            f"Backpropagation phase finished. Duration: {backprop_duration:.4f}s. Total Batch Duration: {batch_duration:.4f}s"
        )
        # --- End of Batch ---

    # --- Loop Finished ---
    final_log_level = logging.INFO  # Always log final summary as INFO
    logger.log(
        final_log_level,
        f"MCTS loop finished. Completed {sim_success_count} backpropagations ({sim_error_count} errors). "
        f"Target Sims: {config.num_simulations}. Root visits: {root_node.visit_count}. Max depth: {max_depth_reached}",
    )

    return max_depth_reached
