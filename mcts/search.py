import numpy as np
import time
import copy
from typing import Dict, Optional, Tuple, Callable
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

    def _select_leaf(self, root_node: MCTSNode) -> MCTSNode:
        """Traverses the tree using PUCT until a leaf node is reached."""
        node = root_node
        depth = 0
        while node.is_expanded and not node.is_terminal:
            if depth >= self.config.MAX_SEARCH_DEPTH:
                break
            if not node.children:
                break
            node = node.select_best_child()
            depth += 1
        return node

    def _expand_node(self, node: MCTSNode) -> Optional[float]:
        """Expands a leaf node: gets NN predictions and creates children."""
        if node.is_expanded or node.is_terminal:
            return node.mean_action_value if node.visit_count > 0 else 0.0

        state_features = node.game_state.get_state()
        try:
            start_pred_time = time.monotonic()
            policy_probs_dict, predicted_value = self.network_predictor(state_features)
            pred_duration = time.monotonic() - start_pred_time
            logger.info(
                f"{self.log_prefix} NN Prediction took {pred_duration:.4f}s. Value: {predicted_value:.3f}"
            )
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Error during network prediction: {e}", exc_info=True
            )
            node.is_expanded = True
            return 0.0

        valid_actions = node.game_state.valid_actions()
        if not valid_actions:
            node.is_expanded = True
            node.is_terminal = True
            return predicted_value

        parent_state = node.game_state
        start_expand_time = time.monotonic()
        children_created = 0
        for action in valid_actions:
            try:
                # --- Deepcopy moved INSIDE the loop ---
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
                children_created += 1
            except Exception as child_creation_err:
                logger.error(
                    f"{self.log_prefix} Error creating child for action {action}: {child_creation_err}",
                    exc_info=True,
                )
                continue
        expand_duration = time.monotonic() - start_expand_time
        logger.info(
            f"{self.log_prefix} Node expansion ({children_created} children) took {expand_duration:.4f}s."
        )

        node.is_expanded = True
        return predicted_value

    def run_simulations(self, root_state: GameState, num_simulations: int) -> MCTSNode:
        """Runs the MCTS process for a given number of simulations."""
        root_node = MCTSNode(game_state=root_state, config=self.config)
        sim_start_time = time.monotonic()

        if not root_node.is_terminal:
            initial_value = self._expand_node(root_node)
            if initial_value is not None:
                self._add_dirichlet_noise(root_node)
                root_node.backpropagate(initial_value)
            # else: logger.info(f"{self.log_prefix} Root expansion failed or node is terminal.")
        # else: logger.info(f"{self.log_prefix} Root node is terminal. Skipping initial expansion.")

        for sim_num in range(num_simulations):
            # logger.info(f"{self.log_prefix} --- Simulation {sim_num+1}/{num_simulations} ---")
            leaf_node = self._select_leaf(root_node)
            value = (
                leaf_node.game_state.get_outcome()
                if leaf_node.is_terminal
                else self._expand_node(leaf_node)
            )
            if value is None:
                logger.warning(
                    f"{self.log_prefix} Expansion returned None for non-terminal node. Using 0."
                )
                value = 0.0
            leaf_node.backpropagate(value)

        sim_duration = time.monotonic() - sim_start_time
        logger.info(
            f"{self.log_prefix} Finished {num_simulations} simulations in {sim_duration:.4f}s. Root visits: {root_node.visit_count}"
        )
        return root_node

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
                    powered_count = float(visit_count) ** (1.0 / temperature)
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
                    policy_target[action] = powered_count / total_power

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
            pass  # Keep zeros

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
            return np.random.choice(actions)
