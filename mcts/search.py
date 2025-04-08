# File: mcts/search.py
import math
import numpy as np
import time
import copy
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Callable, List, Any

from environment.game_state import GameState
from utils.types import ActionType, StateType
from .node import MCTSNode
from config import MCTSConfig, EnvConfig

# Import the actual network class for type hinting
from agent.alphazero_net import AlphaZeroNet


# Define a type hint for the network prediction function
# It takes a game state dict and returns (policy_probs_dict, value)
NetworkPredictor = Callable[[StateType], Tuple[Dict[ActionType, float], float]]


class MCTS:
    """Monte Carlo Tree Search implementation based on AlphaZero principles."""

    def __init__(
        self,
        network_predictor: NetworkPredictor,  # Expects a function like agent.predict
        config: Optional[MCTSConfig] = None,
        env_config: Optional[EnvConfig] = None,
    ):
        self.network_predictor = network_predictor
        self.config = config if config else MCTSConfig()
        self.env_config = env_config if env_config else EnvConfig()

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
        """
        Expands a leaf node: gets NN predictions and creates children.
        Returns the predicted value from the network for this node's state.
        """
        if node.is_expanded or node.is_terminal:
            return node.mean_action_value if node.visit_count > 0 else 0.0

        # Get state features and call the network predictor function
        state_features = node.game_state.get_state()
        try:
            # This call should execute the NN's forward pass (via agent.predict)
            policy_probs_dict, predicted_value = self.network_predictor(state_features)
        except Exception as e:
            print(f"Error during network prediction in MCTS expand: {e}")
            # Handle error: maybe return a default value or re-raise
            node.is_expanded = True  # Mark as expanded to avoid retrying
            return 0.0  # Return neutral value on prediction error

        valid_actions = node.game_state.valid_actions()

        if not valid_actions:
            node.is_expanded = True
            node.is_terminal = True
            # Return the value predicted for this terminal state
            return predicted_value

        # Create child nodes using the policy priors from the network
        for action in valid_actions:
            # Create a *copy* of the state to simulate the action
            # This is crucial to avoid modifying the parent node's state
            temp_state = copy.deepcopy(node.game_state)
            _, done = temp_state.step(action)  # Simulate the action

            prior_prob = policy_probs_dict.get(action, 0.0)  # Get prior from NN output
            child_node = MCTSNode(
                game_state=temp_state,  # Use the state *after* the action
                parent=node,
                action_taken=action,
                prior=prior_prob,
                config=self.config,
            )
            node.children[action] = child_node

        node.is_expanded = True
        # Return the value predicted by the network for the *expanded node's state*
        return predicted_value

    def run_simulations(self, root_state: GameState, num_simulations: int) -> MCTSNode:
        """Runs the MCTS process for a given number of simulations."""
        root_node = MCTSNode(game_state=root_state, config=self.config)

        # Expand root immediately to get initial policy/value and apply noise
        if not root_node.is_terminal:
            initial_value = self._expand_node(root_node)
            if initial_value is not None:
                self._add_dirichlet_noise(root_node)
                # Backpropagate the initial value estimate for the root itself
                root_node.backpropagate(initial_value)
            else:  # Root might be terminal or expansion failed
                pass  # No noise or initial backprop needed

        for _ in range(num_simulations):
            leaf_node = self._select_leaf(root_node)

            # Use the network's prediction as the simulation result (AlphaZero style)
            if leaf_node.is_terminal:
                # Determine terminal value (e.g., based on game score or fixed values)
                # For simplicity, let's use 0 for now, but a real implementation
                # might use +1 for win, -1 for loss, 0 for draw/timeout.
                # This requires the GameState to provide the outcome.
                value = 0.0  # Placeholder terminal value
            else:
                # Expand the node if it hasn't been, get NN value prediction
                value = self._expand_node(leaf_node)
                if (
                    value is None
                ):  # Should not happen if not terminal, but handle defensively
                    value = 0.0  # Fallback value

            # Backpropagate the estimated value up the tree
            leaf_node.backpropagate(value)

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

        visit_counts: List[Tuple[ActionType, int]] = []
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
            # Choose the action with the highest visit count deterministically
            most_visited_action = max(
                root_node.children.items(), key=lambda item: item[1].visit_count
            )[0]
            for action in root_node.children:
                policy_target[action] = 1.0 if action == most_visited_action else 0.0
        else:
            # Sample proportionally to N^(1/temperature)
            total_power = 0.0
            powered_counts = {}
            for action, child in root_node.children.items():
                # Ensure visit_count is non-negative before exponentiation
                visit_count = max(0, child.visit_count)
                try:
                    powered_count = visit_count ** (1.0 / temperature)
                except OverflowError:
                    # Handle potential overflow if temperature is very small and visits large
                    # Assign a large number, or handle based on relative visits
                    powered_count = float("inf") if visit_count > 0 else 0.0
                powered_counts[action] = powered_count
                if powered_count != float("inf"):
                    total_power += powered_count

            # Handle case where all powered counts are inf or total_power is 0
            if total_power == 0 or total_power == float("inf"):
                # Fallback to uniform distribution among children with visits > 0
                visited_children = [
                    a for a, c in root_node.children.items() if c.visit_count > 0
                ]
                num_visited = len(visited_children)
                if num_visited > 0:
                    prob = 1.0 / num_visited
                    for action in root_node.children:
                        policy_target[action] = (
                            prob if action in visited_children else 0.0
                        )
                else:  # If no children were visited (shouldn't happen if total_visits > 0)
                    num_children = len(root_node.children)
                    prob = 1.0 / num_children if num_children > 0 else 1.0
                    for action in root_node.children:
                        policy_target[action] = prob

            else:
                for action, powered_count in powered_counts.items():
                    policy_target[action] = powered_count / total_power

        # Create full policy vector matching ACTION_DIM (important for training target)
        full_policy = np.zeros(self.env_config.ACTION_DIM, dtype=np.float32)
        for action, prob in policy_target.items():
            if 0 <= action < self.env_config.ACTION_DIM:
                full_policy[action] = prob
            else:
                print(f"Warning: MCTS produced invalid action index {action}")

        # Normalize the full policy vector (optional, but good practice)
        policy_sum = np.sum(full_policy)
        if policy_sum > 1e-6 and not np.isclose(policy_sum, 1.0):
            full_policy /= policy_sum
        elif policy_sum <= 1e-6 and self.env_config.ACTION_DIM > 0:
            # Handle zero sum case - distribute probability uniformly?
            # This might happen if temperature is extremely high or visits are zero
            print(
                f"Warning: MCTS policy target sum is near zero ({policy_sum}). Check visits/temperature."
            )
            # Fallback to uniform? Or keep as zeros? Keeping zeros for now.
            pass

        # Return as dict for compatibility, though array might be better for training
        return {i: float(prob) for i, prob in enumerate(full_policy)}

    def choose_action(self, root_node: MCTSNode, temperature: float) -> ActionType:
        """Chooses an action based on MCTS visit counts and temperature."""
        policy_dict = self.get_policy_target(root_node, temperature)
        if not policy_dict:
            valid_actions = root_node.game_state.valid_actions()
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                raise RuntimeError("MCTS failed: no policy and no valid actions.")

        # Filter policy_dict to only include valid actions for sampling
        valid_actions = root_node.game_state.valid_actions()
        filtered_policy = {a: p for a, p in policy_dict.items() if a in valid_actions}

        if not filtered_policy:  # If no valid actions have non-zero probability
            if valid_actions:
                print(
                    "Warning: MCTS policy has zero probability for all valid actions. Choosing uniformly."
                )
                return np.random.choice(valid_actions)
            else:
                raise RuntimeError("MCTS failed: no valid actions to choose from.")

        actions = np.array(list(filtered_policy.keys()))
        probabilities = np.array(list(filtered_policy.values()))

        # Ensure probabilities sum to 1 after filtering
        prob_sum = np.sum(probabilities)
        if prob_sum <= 1e-6:
            # Fallback to uniform distribution among the filtered valid actions
            print(
                f"Warning: MCTS filtered policy sum is near zero ({prob_sum}). Choosing uniformly among valid."
            )
            return np.random.choice(actions)

        probabilities /= prob_sum  # Normalize

        try:
            chosen_action_index = np.random.choice(len(actions), p=probabilities)
            return actions[chosen_action_index]
        except ValueError as e:
            print(f"Error during np.random.choice: {e}")
            print(f"Actions: {actions}")
            print(f"Probabilities: {probabilities} (Sum: {np.sum(probabilities)})")
            # Fallback to uniform choice among valid actions
            return np.random.choice(actions)
