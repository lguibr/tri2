# File: mcts/search.py
import numpy as np
import time
import copy
from typing import Dict, Optional, Tuple, Callable, Any, List
import logging
import torch
import threading
import multiprocessing as mp
import ray
import asyncio  # Added asyncio

from environment.game_state import GameState
from utils.types import ActionType, StateType
from .node import MCTSNode
from config import MCTSConfig, EnvConfig

logger = logging.getLogger(__name__)


class MCTS:
    """Monte Carlo Tree Search implementation based on AlphaZero principles with batching."""

    # Increase internal batch size for potentially better GPU util during self-play
    MCTS_NN_BATCH_SIZE = 32  # Increased batch size for NN predictions within MCTS

    def __init__(
        self,
        agent_predictor: ray.actor.ActorHandle,
        config: Optional[MCTSConfig] = None,
        env_config: Optional[EnvConfig] = None,
        batch_size: int = MCTS_NN_BATCH_SIZE,
        # stop_event: Optional[mp.Event] = None, # Stop event removed
    ):
        self.agent_predictor = agent_predictor
        self.config = config if config else MCTSConfig()
        self.env_config = env_config if env_config else EnvConfig()
        self.batch_size = max(1, batch_size)
        # self.stop_event = stop_event # Removing stop_event check from MCTS itself
        self.log_prefix = "[MCTS]"
        logger.info(
            f"{self.log_prefix} Initialized with AgentPredictor actor. NN batch size: {self.batch_size}"
        )

    def _select_leaf(self, root_node: MCTSNode) -> Tuple[MCTSNode, int]:
        """Selects a leaf node using PUCT criteria."""
        node = root_node
        depth = 0
        while node.is_expanded and not node.is_terminal:
            # Removed stop_event check here
            if depth >= self.config.MAX_SEARCH_DEPTH:
                break
            if not node.children:
                break

            try:
                node = node.select_best_child()
                depth += 1
            except ValueError:
                logger.warning(
                    f"{self.log_prefix} Node claims expanded but has no selectable children."
                )
                break

        return node, depth

    async def _expand_and_backpropagate_batch(  # Made async
        self, nodes_to_expand: List[MCTSNode]
    ) -> Tuple[float, int]:
        """
        Expands a batch of leaf nodes using batched NN prediction via Ray actor and backpropagates results.
        Returns (total_nn_prediction_time, nodes_created_count).
        """
        if not nodes_to_expand:
            return 0.0, 0

        # Removed stop_event check here

        batch_states = [node.game_state.get_state() for node in nodes_to_expand]
        total_nn_prediction_time = 0.0
        nodes_created_count = 0
        policy_probs_list = []
        predicted_values = []

        try:
            start_pred_time = time.monotonic()
            # --- Call the AgentPredictor actor ---
            prediction_ref = self.agent_predictor.predict_batch.remote(batch_states)
            # Use await instead of ray.get()
            policy_probs_list, predicted_values = await prediction_ref
            # --- End Actor Call ---
            total_nn_prediction_time = time.monotonic() - start_pred_time
            logger.debug(
                f"{self.log_prefix} Batched NN Prediction ({len(batch_states)} states) via Actor took {total_nn_prediction_time:.4f}s."
            )
        except ray.exceptions.RayActorError as rae:
            logger.error(
                f"{self.log_prefix} RayActorError during prediction: {rae}",
                exc_info=True,
            )
            # Backpropagate 0 if NN fails, mark as expanded to avoid re-selection
            for node in nodes_to_expand:
                # Removed stop_event check here
                if not node.is_expanded:
                    node.is_expanded = True
                    node.backpropagate(0.0)
            return total_nn_prediction_time, 0
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Error during batched network prediction: {e}",
                exc_info=True,
            )
            # Backpropagate 0 if NN fails, mark as expanded to avoid re-selection
            for node in nodes_to_expand:
                # Removed stop_event check here
                if not node.is_expanded:
                    node.is_expanded = True
                    node.backpropagate(0.0)
            return total_nn_prediction_time, 0

        # Removed stop_event check here

        for i, node in enumerate(nodes_to_expand):
            # Removed stop_event check here

            if node.is_expanded or node.is_terminal:
                if node.visit_count == 0:
                    value_to_prop = (
                        predicted_values[i] if i < len(predicted_values) else 0.0
                    )
                    node.backpropagate(value_to_prop)
                continue

            policy_probs_dict = (
                policy_probs_list[i] if i < len(policy_probs_list) else {}
            )
            predicted_value = predicted_values[i] if i < len(predicted_values) else 0.0
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
                # Removed stop_event check here

                try:
                    # Creating child state remains synchronous
                    child_state = GameState()
                    child_state.grid = parent_state.grid.deepcopy_grid()
                    child_state.shapes = [
                        s.copy() if s else None for s in parent_state.shapes
                    ]
                    child_state.game_score = parent_state.game_score
                    child_state.triangles_cleared_this_episode = (
                        parent_state.triangles_cleared_this_episode
                    )
                    child_state.pieces_placed_this_episode = (
                        parent_state.pieces_placed_this_episode
                    )
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

            # Removed stop_event check here

            expand_duration = time.monotonic() - start_expand_time
            node.is_expanded = True
            nodes_created_count += children_created_count_node

            # Removed stop_event check here
            node.backpropagate(predicted_value)

        return total_nn_prediction_time, nodes_created_count

    async def run_simulations(  # Made async
        self, root_state: GameState, num_simulations: int
    ) -> Tuple[MCTSNode, Dict[str, Any]]:
        """
        Runs the MCTS process for a given number of simulations using batching (async).
        Returns the root node and a dictionary of simulation statistics.
        """
        # Removed stop_event check here

        root_node = MCTSNode(game_state=root_state, config=self.config)
        sim_start_time = time.monotonic()
        total_nn_prediction_time = 0.0
        nodes_created_this_run = 1
        total_leaf_depth = 0
        simulations_run_attempted = 0
        simulations_completed_full = 0

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

        try:
            if not root_node.is_expanded:
                # Removed stop_event check here
                # Await the async batch expansion
                initial_batch_time, initial_nodes_created = (
                    await self._expand_and_backpropagate_batch([root_node])
                )
                total_nn_prediction_time += initial_batch_time
                nodes_created_this_run += initial_nodes_created
                simulations_completed_full += 1
                if root_node.is_expanded and not root_node.is_terminal:
                    self._add_dirichlet_noise(root_node)

            leaves_to_expand: List[MCTSNode] = []
            simulations_run_attempted = 1

            for sim_num in range(simulations_run_attempted, num_simulations):
                simulations_run_attempted += 1
                # Removed stop_event check here

                sim_start_step = time.monotonic()
                leaf_node, depth = self._select_leaf(root_node)
                total_leaf_depth += depth

                if leaf_node.is_terminal:
                    value = leaf_node.game_state.get_outcome()
                    leaf_node.backpropagate(value)
                    sim_duration_step = time.monotonic() - sim_start_step
                    simulations_completed_full += 1
                    continue

                leaves_to_expand.append(leaf_node)

                if (
                    len(leaves_to_expand) >= self.batch_size
                    or sim_num == num_simulations - 1
                ):
                    if leaves_to_expand:
                        # Removed stop_event check here

                        # Await the async batch expansion
                        batch_nn_time, batch_nodes_created = (
                            await self._expand_and_backpropagate_batch(leaves_to_expand)
                        )
                        total_nn_prediction_time += batch_nn_time
                        nodes_created_this_run += batch_nodes_created
                        simulations_completed_full += len(leaves_to_expand)
                        leaves_to_expand = []

            # Removed stop_event check here
            if leaves_to_expand:
                logger.info(
                    f"{self.log_prefix} Processing remaining {len(leaves_to_expand)} leaves after loop exit."
                )
                # Await the async batch expansion
                batch_nn_time, batch_nodes_created = (
                    await self._expand_and_backpropagate_batch(leaves_to_expand)
                )
                total_nn_prediction_time += batch_nn_time
                nodes_created_this_run += batch_nodes_created
                simulations_completed_full += len(leaves_to_expand)

        except InterruptedError as e:  # This might not be reachable now
            logger.warning(f"{self.log_prefix} MCTS run interrupted gracefully: {e}")
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Error during MCTS run_simulations: {e}",
                exc_info=True,
            )

        sim_duration_total = time.monotonic() - sim_start_time
        effective_sims = max(1, root_node.visit_count)
        avg_leaf_depth = (
            total_leaf_depth / effective_sims if effective_sims > 0 else 0.0
        )

        logger.info(
            f"{self.log_prefix} Finished {root_node.visit_count} effective simulations ({simulations_run_attempted} attempted) in {sim_duration_total:.4f}s. "
            f"Nodes created: {nodes_created_this_run}, "
            f"Total NN time: {total_nn_prediction_time:.4f}s, Avg Depth: {avg_leaf_depth:.1f}"
        )

        mcts_stats = {
            "simulations_run": root_node.visit_count,
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
        child_actions = [a for a in node.children.keys() if a in node.children]
        if not child_actions:
            return

        num_children = len(child_actions)
        noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA] * num_children)
        eps = self.config.DIRICHLET_EPSILON
        for i, action in enumerate(child_actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]
        logger.debug(f"{self.log_prefix} Applied Dirichlet noise to root node priors.")

    def get_policy_target(
        self, root_node: MCTSNode, temperature: float
    ) -> Dict[ActionType, float]:
        """Calculates the improved policy distribution based on visit counts."""
        if not root_node.children:
            return {}

        existing_children = {a: c for a, c in root_node.children.items() if c}
        if not existing_children:
            return {}

        total_visits = sum(child.visit_count for child in existing_children.values())
        if total_visits == 0:
            num_children = len(existing_children)
            logger.warning(
                f"{self.log_prefix} Root node has 0 total visits across children. Returning uniform policy."
            )
            return (
                {a: 1.0 / num_children for a in existing_children}
                if num_children > 0
                else {}
            )

        policy_target: Dict[ActionType, float] = {}
        if temperature == 0:
            best_action = max(
                existing_children, key=lambda a: existing_children[a].visit_count
            )
            for action in existing_children:
                policy_target[action] = 1.0 if action == best_action else 0.0
        else:
            total_power, powered_counts = 0.0, {}
            max_power_val = np.finfo(np.float64).max / (len(existing_children) + 1)

            for action, child in existing_children.items():
                visit_count = max(0, child.visit_count)
                try:
                    powered_count = np.power(
                        np.float64(visit_count), 1.0 / temperature, dtype=np.float64
                    )
                    if np.isinf(powered_count) or np.isnan(powered_count):
                        logger.warning(
                            f"{self.log_prefix} Infinite/NaN powered count for action {action}. Clamping."
                        )
                        powered_count = max_power_val
                except (OverflowError, ValueError):
                    logger.warning(
                        f"{self.log_prefix} Power calc overflow/error for action {action}. Setting large value."
                    )
                    powered_count = max_power_val

                powered_counts[action] = powered_count
                if not np.isinf(powered_count) and not np.isnan(powered_count):
                    total_power += powered_count

            if total_power <= 1e-9 or np.isinf(total_power) or np.isnan(total_power):
                num_valid_children = len(powered_counts)
                if num_valid_children > 0:
                    prob = 1.0 / num_valid_children
                    for action in existing_children:
                        policy_target[action] = prob
                    logger.warning(
                        f"{self.log_prefix} Total power invalid ({total_power:.2e}), assigned uniform prob {prob:.3f}."
                    )
                else:
                    policy_target = {}
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

        filtered_actions, filtered_probs = [], []
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
                f"{self.log_prefix} Error during np.random.choice: {e}. Prob sum: {np.sum(probabilities)}. Choosing uniformly."
            )
            return np.random.choice(actions)
