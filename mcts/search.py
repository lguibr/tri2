# File: mcts/search.py
import numpy as np
import time
import copy
from typing import Dict, Optional, Tuple, Callable, Any, List
import logging
import torch  # Added for batching
import threading  # Added for stop_event

from environment.game_state import GameState
from utils.types import ActionType, StateType
from .node import MCTSNode
from config import MCTSConfig, EnvConfig


# Updated signature for batch prediction
NetworkPredictor = Callable[
    [List[StateType]], Tuple[List[Dict[ActionType, float]], List[float]]
]
logger = logging.getLogger(__name__)


class MCTS:
    """Monte Carlo Tree Search implementation based on AlphaZero principles with batching."""

    def __init__(
        self,
        network_predictor: NetworkPredictor,  # Predictor now handles batches
        config: Optional[MCTSConfig] = None,
        env_config: Optional[EnvConfig] = None,
        batch_size: int = 8,  # Default MCTS batch size
        stop_event: Optional[threading.Event] = None,  # Add stop_event
    ):
        self.network_predictor = network_predictor
        self.config = config if config else MCTSConfig()
        self.env_config = env_config if env_config else EnvConfig()
        self.batch_size = batch_size  # Store batch size
        self.stop_event = stop_event  # Store stop_event
        self.log_prefix = "[MCTS]"
        logger.info(
            f"{self.log_prefix} Initialized with NN batch size: {self.batch_size}"
        )

    def _select_leaf(self, root_node: MCTSNode) -> Tuple[MCTSNode, int]:
        """Traverses the tree using PUCT until a leaf node is reached. Returns node and depth."""
        node = root_node
        depth = 0
        while node.is_expanded and not node.is_terminal:
            # Check stop event during selection
            if self.stop_event and self.stop_event.is_set():
                raise InterruptedError("MCTS selection interrupted by stop event.")

            if depth >= self.config.MAX_SEARCH_DEPTH:
                break
            if not node.children:
                break
            node = node.select_best_child()
            depth += 1
        return node, depth

    def _expand_and_backpropagate_batch(
        self, nodes_to_expand: List[MCTSNode]
    ) -> Tuple[float, int]:
        """
        Expands a batch of leaf nodes using batched NN prediction and backpropagates results.
        Returns (total_nn_prediction_time, nodes_created_count).
        """
        if not nodes_to_expand:
            return 0.0, 0

        # Check stop event before NN prediction
        if self.stop_event and self.stop_event.is_set():
            raise InterruptedError("MCTS expansion interrupted by stop event.")

        batch_states = [node.game_state.get_state() for node in nodes_to_expand]
        total_nn_prediction_time = 0.0
        nodes_created_count = 0

        try:
            start_pred_time = time.monotonic()
            policy_probs_list, predicted_values = self.network_predictor(batch_states)
            total_nn_prediction_time = time.monotonic() - start_pred_time
            logger.info(  # Changed to debug
                f"{self.log_prefix} Batched NN Prediction ({len(batch_states)} states) took {total_nn_prediction_time:.4f}s."
            )
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Error during batched network prediction: {e}",
                exc_info=True,
            )
            for node in nodes_to_expand:
                node.is_expanded = True
                node.backpropagate(0.0)
            return total_nn_prediction_time, 0

        # Process results for each node in the batch
        for i, node in enumerate(nodes_to_expand):
            # Check stop event during expansion processing
            if self.stop_event and self.stop_event.is_set():
                raise InterruptedError(
                    "MCTS expansion processing interrupted by stop event."
                )

            if node.is_expanded or node.is_terminal:
                continue

            policy_probs_dict = policy_probs_list[i]
            predicted_value = predicted_values[i]
            children_created_count_node = 0

            valid_actions = node.game_state.valid_actions()
            if not valid_actions:
                node.is_expanded = True
                node.is_terminal = True
                node.backpropagate(predicted_value)
                continue

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
                    if done:
                        child_node.is_terminal = True

                    node.children[action] = child_node
                    children_created_count_node += 1
                except Exception as child_creation_err:
                    logger.error(
                        f"{self.log_prefix} Error creating child for action {action}: {child_creation_err}",
                        exc_info=True,
                    )
                    continue
            expand_duration = time.monotonic() - start_expand_time
            logger.info(  # Changed to debug
                f"{self.log_prefix} Node expansion ({children_created_count_node} children) took {expand_duration:.4f}s."
            )

            node.is_expanded = True
            nodes_created_count += children_created_count_node
            node.backpropagate(predicted_value)

        return total_nn_prediction_time, nodes_created_count

    def run_simulations(
        self, root_state: GameState, num_simulations: int
    ) -> Tuple[MCTSNode, Dict[str, Any]]:
        """
        Runs the MCTS process for a given number of simulations using batching.
        Returns the root node and a dictionary of simulation statistics.
        """
        root_node = MCTSNode(game_state=root_state, config=self.config)
        sim_start_time = time.monotonic()
        total_nn_prediction_time = 0.0
        nodes_created_this_run = 1
        total_leaf_depth = 0
        simulations_run = 0

        if root_node.is_terminal:
            logger.warning(
                f"{self.log_prefix} Root node is terminal. No simulations run."
            )
            return root_node, {
                "simulations_run": 0,
                "mcts_total_duration": 0.0,
                "total_nn_prediction_time": 0.0,
                "nodes_created": 1,
                "avg_leaf_depth": 0.0,
                "root_visits": 0,
            }

        try:  # Wrap simulation loop to catch InterruptedError
            # Initial expansion/prediction
            initial_batch_time, initial_nodes_created = (
                self._expand_and_backpropagate_batch([root_node])
            )
            total_nn_prediction_time += initial_batch_time
            nodes_created_this_run += initial_nodes_created
            simulations_run += 1

            if root_node.is_expanded and not root_node.is_terminal:
                self._add_dirichlet_noise(root_node)

            leaves_to_expand: List[MCTSNode] = []
            for sim_num in range(simulations_run, num_simulations):
                # Check stop event at the start of each simulation
                if self.stop_event and self.stop_event.is_set():
                    logger.info(
                        f"{self.log_prefix} Stop event detected during simulation {sim_num+1}. Stopping MCTS."
                    )
                    break  # Exit simulation loop

                sim_start_step = time.monotonic()
                leaf_node, depth = self._select_leaf(root_node)
                total_leaf_depth += depth

                if leaf_node.is_terminal:
                    value = leaf_node.game_state.get_outcome()
                    leaf_node.backpropagate(value)
                    sim_duration_step = time.monotonic() - sim_start_step
                    logger.info(  # Changed to debug
                        f"{self.log_prefix} Sim {sim_num+1}/{num_simulations} hit terminal node. Backprop: {value:.2f}. Took {sim_duration_step:.5f}s"
                    )
                    continue

                leaves_to_expand.append(leaf_node)

                if (
                    len(leaves_to_expand) >= self.batch_size
                    or sim_num == num_simulations - 1
                ):
                    if leaves_to_expand:
                        batch_nn_time, batch_nodes_created = (
                            self._expand_and_backpropagate_batch(leaves_to_expand)
                        )
                        total_nn_prediction_time += batch_nn_time
                        nodes_created_this_run += batch_nodes_created
                        leaves_to_expand = []

                # sim_duration_step = time.monotonic() - sim_start_step # Less useful now

            # Process any remaining leaves if loop finished early (or normally)
            if leaves_to_expand:
                batch_nn_time, batch_nodes_created = (
                    self._expand_and_backpropagate_batch(leaves_to_expand)
                )
                total_nn_prediction_time += batch_nn_time
                nodes_created_this_run += batch_nodes_created

        except InterruptedError as e:
            logger.warning(f"{self.log_prefix} MCTS run interrupted: {e}")
            # Return current state of root node and stats gathered so far
            pass  # Fall through to return current state

        sim_duration_total = time.monotonic() - sim_start_time
        # Use root_node.visit_count as the effective number of simulations completed
        effective_sims = max(1, root_node.visit_count)
        avg_leaf_depth = total_leaf_depth / effective_sims

        logger.info(  # Changed to debug
            f"{self.log_prefix} Finished {root_node.visit_count} effective simulations in {sim_duration_total:.4f}s. "
            f"Nodes created: {nodes_created_this_run}, "
            f"Total NN time: {total_nn_prediction_time:.4f}s, Avg Depth: {avg_leaf_depth:.1f}"
        )

        mcts_stats = {
            "simulations_run": root_node.visit_count,  # Report effective sims
            "mcts_total_duration": sim_duration_total,
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
        if not actions:
            return
        noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA] * len(actions))
        eps = self.config.DIRICHLET_EPSILON
        for i, action in enumerate(actions):
            child = node.children.get(action)
            if child:
                child.prior = (1 - eps) * child.prior + eps * noise[i]
        logger.info(
            f"{self.log_prefix} Applied Dirichlet noise to root node priors."
        )  # Changed to debug

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
            best_action = -1
            max_visits = -1
            for action, child in root_node.children.items():
                if child.visit_count > max_visits:
                    max_visits = child.visit_count
                    best_action = action
            if best_action != -1:
                for action in root_node.children:
                    policy_target[action] = 1.0 if action == best_action else 0.0
            else:
                num_children = len(root_node.children)
                prob = 1.0 / num_children if num_children > 0 else 0.0
                for action in root_node.children:
                    policy_target[action] = prob

        else:
            total_power, powered_counts = 0.0, {}
            for action, child in root_node.children.items():
                visit_count = max(0, child.visit_count)
                try:
                    powered_count = np.power(
                        np.float64(visit_count), 1.0 / temperature, dtype=np.float64
                    )
                    if np.isinf(powered_count):
                        logger.warning(
                            f"{self.log_prefix} Infinite powered count encountered for action {action}. Clamping."
                        )
                        pass

                except (OverflowError, ValueError):
                    logger.warning(
                        f"{self.log_prefix} Power calculation overflow/error for action {action}. Setting to 0."
                    )
                    powered_count = 0.0

                powered_counts[action] = powered_count
                if not np.isinf(powered_count):
                    total_power += powered_count

            if total_power <= 1e-9 or np.isinf(total_power):
                if np.isinf(total_power):
                    inf_actions = [
                        a for a, pc in powered_counts.items() if np.isinf(pc)
                    ]
                    num_inf = len(inf_actions)
                    prob = 1.0 / num_inf if num_inf > 0 else 0.0
                    for action in root_node.children:
                        policy_target[action] = prob if action in inf_actions else 0.0
                else:
                    visited_children = [
                        a for a, c in root_node.children.items() if c.visit_count > 0
                    ]
                    num_visited = len(visited_children)
                    prob = (
                        1.0 / num_visited
                        if num_visited > 0
                        else (
                            1.0 / len(root_node.children) if root_node.children else 0.0
                        )
                    )
                    for action in root_node.children:
                        policy_target[action] = (
                            prob if action in visited_children else 0.0
                        )
            else:
                for action, powered_count in powered_counts.items():
                    policy_target[action] = float(powered_count / total_power)

        full_policy = np.zeros(self.env_config.ACTION_DIM, dtype=np.float32)
        policy_sum_check = 0.0
        for action, prob in policy_target.items():
            if 0 <= action < self.env_config.ACTION_DIM:
                full_policy[action] = prob
                policy_sum_check += prob
            else:
                logger.warning(
                    f"{self.log_prefix} MCTS produced invalid action index {action} in policy target."
                )

        if not np.isclose(policy_sum_check, 1.0, atol=1e-4):
            logger.warning(
                f"{self.log_prefix} Policy target sum is {policy_sum_check:.4f} before final conversion. Renormalizing."
            )
            current_sum = np.sum(full_policy)
            if current_sum > 1e-6:
                full_policy /= current_sum
            else:
                valid_actions = root_node.game_state.valid_actions()
                num_valid = len(valid_actions)
                if num_valid > 0:
                    prob = 1.0 / num_valid
                    full_policy.fill(0.0)
                    for action in valid_actions:
                        if 0 <= action < self.env_config.ACTION_DIM:
                            full_policy[action] = prob

        return {i: float(prob) for i, prob in enumerate(full_policy)}

    def choose_action(self, root_node: MCTSNode, temperature: float) -> ActionType:
        """Chooses an action based on MCTS visit counts and temperature."""
        policy_dict = self.get_policy_target(root_node, temperature)
        valid_actions_list = root_node.game_state.valid_actions()
        valid_actions_set = set(valid_actions_list)

        if not policy_dict or not valid_actions_list:
            if valid_actions_list:
                logger.warning(
                    f"{self.log_prefix} Policy dict empty/invalid, choosing random valid action."
                )
                return np.random.choice(valid_actions_list)
            else:
                logger.error(
                    f"{self.log_prefix} MCTS failed: no policy and no valid actions."
                )
                return -1

        filtered_actions = []
        filtered_probs = []
        for action, prob in policy_dict.items():
            if action in valid_actions_set and prob > 1e-7:
                filtered_actions.append(action)
                filtered_probs.append(prob)

        if not filtered_actions:
            logger.warning(
                f"{self.log_prefix} MCTS policy effectively zero for all valid actions. Choosing uniformly among valid."
            )
            return np.random.choice(valid_actions_list)

        actions = np.array(filtered_actions)
        probabilities = np.array(filtered_probs, dtype=np.float64)

        prob_sum = np.sum(probabilities)
        if prob_sum <= 1e-7:
            logger.warning(
                f"{self.log_prefix} Filtered policy sum near zero ({prob_sum}). Choosing uniformly among filtered."
            )
            return np.random.choice(actions)

        probabilities /= prob_sum

        try:
            chosen_action = np.random.choice(actions, p=probabilities)
            return int(chosen_action)
        except ValueError as e:
            logger.error(
                f"{self.log_prefix} Error during np.random.choice: {e}. Probabilities sum: {np.sum(probabilities)}. Actions: {actions}. Probs: {probabilities}"
            )
            return np.random.choice(actions)
