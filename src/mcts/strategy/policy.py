# File: src/mcts/strategy/policy.py
import numpy as np
import logging
from typing import TYPE_CHECKING, Optional

# Relative imports within MCTS strategy
from ..core.node import Node

# Import ActionPolicyMapping from local types, ActionType from central utils
from ..core.types import ActionPolicyMapping
from ...utils.types import ActionType  # Correct import path for ActionType

logger = logging.getLogger(__name__)


def select_action_based_on_visits(
    root_node: Node, temperature: float
) -> Optional[ActionType]:
    """Selects an action from the root node based on visit counts and temperature."""
    if not root_node.children:
        logger.warning("Cannot select action: Root node has no children.")
        return None

    actions = list(root_node.children.keys())
    visit_counts = np.array(
        [root_node.children[action].visit_count for action in actions],
        dtype=np.float64,
    )

    if len(actions) == 0:
        logger.warning("Cannot select action: No actions available in children.")
        return None

    if temperature == 0.0:
        # Greedy selection
        max_visits = np.max(visit_counts)
        if max_visits == 0:
            logger.warning(
                "No visits recorded for any child node, selecting uniformly."
            )
            selected_action = np.random.choice(actions)
        else:
            best_action_indices = np.where(visit_counts == max_visits)[0]
            chosen_index = np.random.choice(best_action_indices)
            selected_action = actions[chosen_index]
        # logger.info(f"Greedy action selection: {selected_action}")
    else:
        # Temperature-based probabilistic selection (using log-space for stability)
        log_visits = np.log(np.maximum(visit_counts, 1e-9))
        scaled_log_visits = log_visits / temperature
        scaled_log_visits -= np.max(scaled_log_visits)  # Stability trick
        probabilities = np.exp(scaled_log_visits)
        sum_probs = np.sum(probabilities)

        if sum_probs < 1e-9 or not np.isfinite(sum_probs):
            logger.warning(
                f"Could not normalize visit probabilities (sum={sum_probs}). Selecting uniformly."
            )
            probabilities = np.ones(len(actions)) / len(actions)
        else:
            probabilities /= sum_probs  # Normalize
            probabilities /= np.sum(
                probabilities
            )  # Re-normalize for floating point safety

        try:
            selected_action = np.random.choice(actions, p=probabilities)
            # logger.info(f"Sampled action (temp={temperature:.2f}): {selected_action}")
        except ValueError as e:
            logger.error(
                f"Error during np.random.choice: {e}. Probs: {probabilities}, Sum: {np.sum(probabilities)}"
            )
            selected_action = np.random.choice(actions)  # Fallback

    return selected_action


def get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping:
    """Calculates the policy target distribution based on MCTS visit counts."""
    policy_target: ActionPolicyMapping = {}
    if not root_node.children or root_node.visit_count == 0:
        logger.warning(
            "Cannot compute policy target: Root node has no children or zero visits. Falling back to uniform over valid actions."
        )
        try:
            valid_actions = root_node.state.valid_actions()
            if valid_actions:
                prob = 1.0 / len(valid_actions)
                # Return full policy vector for training if needed (zeroes elsewhere)
                full_target = {
                    a: 0.0 for a in range(root_node.state.env_config.ACTION_DIM)
                }
                for a in valid_actions:
                    full_target[a] = prob
                return full_target
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting valid actions for fallback policy target: {e}")
            return {}

    child_visits = {
        action: child.visit_count for action, child in root_node.children.items()
    }
    actions = list(child_visits.keys())
    visits = np.array(list(child_visits.values()), dtype=np.float64)

    if not actions:
        logger.warning("Cannot compute policy target: No actions found in children.")
        return {}  # Should be caught above, but safety check

    if temperature == 0.0:  # Deterministic target for temp=0
        max_visits = np.max(visits)
        if max_visits == 0:
            prob = 1.0 / len(actions)
            best_actions = actions
        else:
            best_actions = [actions[i] for i, v in enumerate(visits) if v == max_visits]
            prob = 1.0 / len(best_actions)
        policy_target = {
            a: prob if a in best_actions else 0.0
            for a in range(root_node.state.env_config.ACTION_DIM)
        }

    else:  # Proportional target
        visit_probs = visits ** (1.0 / temperature)
        sum_probs = np.sum(visit_probs)

        if sum_probs < 1e-9 or not np.isfinite(sum_probs):
            logger.warning(
                f"Sum of visit probabilities is near zero ({sum_probs}). Using uniform."
            )
            prob = 1.0 / len(actions)
            raw_policy = {action: prob for action in actions}
        else:
            probabilities = visit_probs / sum_probs
            probabilities /= np.sum(probabilities)  # Re-normalize
            raw_policy = {action: probabilities[i] for i, action in enumerate(actions)}

        # Create full policy vector
        policy_target = {a: 0.0 for a in range(root_node.state.env_config.ACTION_DIM)}
        policy_target.update(raw_policy)

    return policy_target
