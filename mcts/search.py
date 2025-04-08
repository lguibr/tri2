# File: mcts/search.py
import numpy as np
import time
import copy
from typing import Dict, Optional, Tuple, Callable, Any
import logging

from environment.game_state import GameState
from utils.types import ActionType, StateType
from .node import MCTSNode
from config import MCTSConfig, EnvConfig


NetworkPredictor = Callable[[StateType], Tuple[Dict[ActionType, float], float]]
logger = logging.getLogger(__name__)


class MCTS:
    """Monte Carlo Tree Search implementation based on AlphaZero principles."""

    def __init__(
        self,
        network_predictor: NetworkPredictor,
        config: Optional[MCTSConfig] = None,
        env_config: Optional[EnvConfig] = None,
    ):
        self.network_predictor = network_predictor
        self.config = config if config else MCTSConfig()
        self.env_config = env_config if env_config else EnvConfig()
        self.log_prefix = "[MCTS]"

    def _select_leaf(self, root_node: MCTSNode) -> Tuple[MCTSNode, int]:
        """Traverses the tree using PUCT until a leaf node is reached. Returns node and depth."""
        node = root_node
        depth = 0
        while node.is_expanded and not node.is_terminal:
            if depth >= self.config.MAX_SEARCH_DEPTH:
                break
            if not node.children:
                break
            node = node.select_best_child()
            depth += 1
        return node, depth

    def _expand_node(self, node: MCTSNode) -> Tuple[Optional[float], float, int]:
        """
        Expands a leaf node: gets NN predictions and creates children.
        Returns (predicted_value, nn_prediction_time, children_created_count).
        """
        nn_prediction_time = 0.0
        children_created_count = 0

        if node.is_expanded or node.is_terminal:
            value = node.mean_action_value if node.visit_count > 0 else 0.0
            return value, nn_prediction_time, children_created_count

        state_features = node.game_state.get_state()
        try:
            start_pred_time = time.monotonic()
            policy_probs_dict, predicted_value = self.network_predictor(state_features)
            nn_prediction_time = time.monotonic() - start_pred_time
            logger.debug(  # Changed to debug
                f"{self.log_prefix} NN Prediction took {nn_prediction_time:.4f}s. Value: {predicted_value:.3f}"
            )
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Error during network prediction: {e}", exc_info=True
            )
            node.is_expanded = True
            return 0.0, nn_prediction_time, children_created_count

        valid_actions = node.game_state.valid_actions()
        if not valid_actions:
            node.is_expanded = True
            node.is_terminal = True
            return predicted_value, nn_prediction_time, children_created_count

        parent_state = node.game_state
        start_expand_time = time.monotonic()
        for action in valid_actions:
            try:
                child_state = copy.deepcopy(parent_state)
                _, done = child_state.step(action)
                prior_prob = policy_probs_dict.get(action, 0.0)
                child_node = MCTSNode(
                    game_state=child_state,
                    parent=node,
                    action_taken=action,
                    prior=prior_prob,
                    config=self.config,
                )
                node.children[action] = child_node
                children_created_count += 1
            except Exception as child_creation_err:
                logger.error(
                    f"{self.log_prefix} Error creating child for action {action}: {child_creation_err}",
                    exc_info=True,
                )
                continue
        expand_duration = time.monotonic() - start_expand_time
        logger.debug(  # Changed to debug
            f"{self.log_prefix} Node expansion ({children_created_count} children) took {expand_duration:.4f}s."
        )

        node.is_expanded = True
        return predicted_value, nn_prediction_time, children_created_count

    def run_simulations(
        self, root_state: GameState, num_simulations: int
    ) -> Tuple[MCTSNode, Dict[str, Any]]:
        """
        Runs the MCTS process for a given number of simulations.
        Returns the root node and a dictionary of simulation statistics.
        """
        root_node = MCTSNode(game_state=root_state, config=self.config)
        sim_start_time = time.monotonic()
        total_nn_prediction_time = 0.0
        nodes_created_this_run = 1  # Start with root node
        total_leaf_depth = 0
        simulations_run = 0

        if not root_node.is_terminal:
            initial_value, nn_time, children_count = self._expand_node(root_node)
            total_nn_prediction_time += nn_time
            nodes_created_this_run += children_count
            if initial_value is not None:
                self._add_dirichlet_noise(root_node)
                root_node.backpropagate(initial_value)

        for sim_num in range(num_simulations):
            simulations_run += 1
            leaf_node, depth = self._select_leaf(root_node)
            total_leaf_depth += depth

            if leaf_node.is_terminal:
                value = leaf_node.game_state.get_outcome()
                nn_time, children_count = 0.0, 0  # No expansion if terminal
            else:
                value, nn_time, children_count = self._expand_node(leaf_node)
                if value is None:
                    logger.warning(
                        f"{self.log_prefix} Expansion returned None for non-terminal node. Using 0."
                    )
                    value = 0.0

            total_nn_prediction_time += nn_time
            nodes_created_this_run += children_count
            leaf_node.backpropagate(value)

        sim_duration = time.monotonic() - sim_start_time
        avg_leaf_depth = (
            total_leaf_depth / simulations_run if simulations_run > 0 else 0
        )
        logger.debug(  # Changed to debug
            f"{self.log_prefix} Finished {simulations_run} simulations in {sim_duration:.4f}s. "
            f"Root visits: {root_node.visit_count}, Nodes created: {nodes_created_this_run}, "
            f"Total NN time: {total_nn_prediction_time:.4f}s, Avg Depth: {avg_leaf_depth:.1f}"
        )

        mcts_stats = {
            "simulations_run": simulations_run,
            "mcts_total_duration": sim_duration,
            "total_nn_prediction_time": total_nn_prediction_time,
            "nodes_created": nodes_created_this_run,
            "avg_leaf_depth": avg_leaf_depth,
            "root_visits": root_node.visit_count,
        }
        return root_node, mcts_stats

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Adds Dirichlet noise to the prior probabilities of the root node's children."""
        if not node.children or self.config.DIRICHLET_ALPHA <= 0:
            return
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA] * len(actions))
        eps = self.config.DIRICHLET_EPSILON
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def get_policy_target(
        self, root_node: MCTSNode, temperature: float
    ) -> Dict[ActionType, float]:
        """Calculates the improved policy distribution based on visit counts."""
        if not root_node.children:
            return {}
        total_visits = sum(child.visit_count for child in root_node.children.values())
        if total_visits == 0:
            num_children = len(root_node.children)
            return (
                {a: 1.0 / num_children for a in root_node.children}
                if num_children > 0
                else {}
            )

        policy_target: Dict[ActionType, float] = {}
        if temperature == 0:
            most_visited_action = max(
                root_node.children.items(), key=lambda item: item[1].visit_count
            )[0]
            for action in root_node.children:
                policy_target[action] = 1.0 if action == most_visited_action else 0.0
        else:
            total_power, powered_counts = 0.0, {}
            for action, child in root_node.children.items():
                visit_count = max(0, child.visit_count)
                try:
                    # Use float64 for intermediate power calculation to avoid overflow
                    powered_count = np.power(
                        np.float64(visit_count), 1.0 / temperature, dtype=np.float64
                    )
                except OverflowError:
                    powered_count = float("inf") if visit_count > 0 else 0.0
                powered_counts[action] = powered_count
                if powered_count != float("inf"):
                    total_power += powered_count

            if total_power == 0 or total_power == float("inf"):
                visited_children = [
                    a for a, c in root_node.children.items() if c.visit_count > 0
                ]
                num_visited = len(visited_children)
                prob = (
                    1.0 / num_visited
                    if num_visited > 0
                    else (1.0 / len(root_node.children) if root_node.children else 1.0)
                )
                for action in root_node.children:
                    policy_target[action] = prob if action in visited_children else 0.0
            else:
                for action, powered_count in powered_counts.items():
                    policy_target[action] = float(
                        powered_count / total_power
                    )  # Convert back to float

        full_policy = np.zeros(self.env_config.ACTION_DIM, dtype=np.float32)
        for action, prob in policy_target.items():
            if 0 <= action < self.env_config.ACTION_DIM:
                full_policy[action] = prob
            else:
                logger.warning(
                    f"{self.log_prefix} MCTS produced invalid action index {action}"
                )

        policy_sum = np.sum(full_policy)
        if policy_sum > 1e-6 and not np.isclose(policy_sum, 1.0):
            full_policy /= policy_sum
        elif policy_sum <= 1e-6 and self.env_config.ACTION_DIM > 0:
            pass

        return {i: float(prob) for i, prob in enumerate(full_policy)}

    def choose_action(self, root_node: MCTSNode, temperature: float) -> ActionType:
        """Chooses an action based on MCTS visit counts and temperature."""
        policy_dict = self.get_policy_target(root_node, temperature)
        valid_actions = root_node.game_state.valid_actions()
        if not policy_dict or not valid_actions:
            if valid_actions:
                logger.warning(
                    f"{self.log_prefix} Policy dict empty/invalid, choosing random valid action."
                )
                return np.random.choice(valid_actions)
            else:
                logger.error(
                    f"{self.log_prefix} MCTS failed: no policy and no valid actions."
                )
                raise RuntimeError("MCTS failed: no policy/valid actions.")

        filtered_policy = {a: p for a, p in policy_dict.items() if a in valid_actions}
        if not filtered_policy:
            logger.warning(
                f"{self.log_prefix} MCTS policy zero for all valid actions. Choosing uniformly."
            )
            return np.random.choice(valid_actions)

        actions = np.array(list(filtered_policy.keys()))
        probabilities = np.array(list(filtered_policy.values()))
        prob_sum = np.sum(probabilities)
        if prob_sum <= 1e-6:
            logger.warning(
                f"{self.log_prefix} Filtered policy sum near zero. Choosing uniformly."
            )
            return np.random.choice(actions)
        probabilities /= prob_sum

        try:
            return np.random.choice(actions, p=probabilities)
        except ValueError as e:
            logger.error(f"{self.log_prefix} Error during np.random.choice: {e}")
            # Fallback: choose uniformly among valid actions with non-zero probability
            non_zero_prob_actions = [a for a, p in zip(actions, probabilities) if p > 0]
            if non_zero_prob_actions:
                return np.random.choice(non_zero_prob_actions)
            else:  # If somehow all probabilities became zero after normalization
                return np.random.choice(actions)
