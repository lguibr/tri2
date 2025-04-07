from collections import deque
from typing import Deque, Dict, Any,
import time


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator."""

    def __init__(self, plot_window: int):
        self.plot_window = plot_window

        # --- Deques for Plotting ---
        # Removed: self.policy_losses
        # Removed: self.value_losses
        # Removed: self.entropies
        # Removed: self.grad_norms
        self.avg_max_qs: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep if Q-values are estimated by NN
        self.episode_scores: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep RL score if used, or repurpose
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_triangles_cleared: Deque[int] = deque(maxlen=plot_window)
        # Removed: self.sps_values
        # Removed: self.update_steps_per_second_values
        # Removed: self.minibatch_update_sps_values
        self.buffer_sizes: Deque[int] = deque(
            maxlen=plot_window
        )  # Might be useful for MCTS buffer/NN training data
        self.beta_values: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep if used (e.g., PER)
        self.best_rl_score_history: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep RL score if used
        self.best_game_score_history: Deque[int] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep for NN training LR
        self.epsilon_values: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep if epsilon-greedy is used in MCTS/NN
        self.cpu_usage: Deque[float] = deque(maxlen=plot_window)
        self.memory_usage: Deque[float] = deque(maxlen=plot_window)
        self.gpu_memory_usage_percent: Deque[float] = deque(maxlen=plot_window)

        # --- Scalar State Variables ---
        self.total_episodes = 0
        self.total_triangles_cleared = 0
        self.current_epsilon: float = 0.0
        self.current_beta: float = 0.0
        self.current_buffer_size: int = 0
        self.current_global_step: int = (
            0  # Might represent training steps or games played now
        )

        self.current_lr: float = 0.0  # Keep for NN
        self.start_time: float = time.time()
        self.training_target_step: int = 0  # Target might be games played or NN steps
        self.current_cpu_usage: float = 0.0
        self.current_memory_usage: float = 0.0
        self.current_gpu_memory_usage_percent: float = 0.0

        # --- Best Value Tracking ---
        self.best_score: float = -float("inf")  # Keep RL score if used
        self.previous_best_score: float = -float("inf")
        self.best_score_step: int = 0
        self.best_game_score: float = -float("inf")
        self.previous_best_game_score: float = -float("inf")
        self.best_game_score_step: int = 0
        self.best_value_loss: float = float("inf")  # Keep for NN value head loss
        self.previous_best_value_loss: float = float("inf")
        self.best_value_loss_step: int = 0
        # Add best policy loss?
        self.best_policy_loss: float = float("inf")
        self.previous_best_policy_loss: float = float("inf")
        self.best_policy_loss_step: int = 0

    def get_deque(self, name: str) -> Deque:
        """Safely gets a deque attribute."""
        return getattr(self, name, deque(maxlen=self.plot_window))

    def get_all_plot_deques(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        deque_names = [
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            # Add NN losses if tracked
            "policy_losses",  # Added back for NN policy head
            "value_losses",  # Kept for NN value head
        ]
        # Filter out names that might not exist if loaded from old state
        return {
            name: self.get_deque(name).copy()
            for name in deque_names
            if hasattr(self, name)
        }

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the storage for saving."""
        state = {}
        # Deques
        deque_names = [
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            # Add NN losses if tracked
            "policy_losses",  # Added back for NN policy head
            "value_losses",  # Kept for NN value head
        ]
        for name in deque_names:
            if hasattr(self, name):  # Check if deque exists before saving
                state[name] = list(self.get_deque(name))

        # Scalar State Variables
        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_epsilon",
            "current_beta",
            "current_buffer_size",
            "current_global_step",
            # Removed SPS scalars
            "current_lr",
            "start_time",
            "training_target_step",
            "current_cpu_usage",
            "current_memory_usage",
            "current_gpu_memory_usage_percent",
        ]
        for key in scalar_keys:
            state[key] = getattr(self, key, 0)

        # Best Value Tracking
        best_value_keys = [
            "best_score",
            "previous_best_score",
            "best_score_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
            "best_policy_loss",  # Added policy loss tracking
            "previous_best_policy_loss",
            "best_policy_loss_step",
        ]
        for key in best_value_keys:
            state[key] = getattr(self, key, 0)

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        self.plot_window = plot_window

        deque_names = [
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            "policy_losses",
            "value_losses",
        ]
        for key in deque_names:
            data_to_load = state_dict.get(key)
            if data_to_load is not None:
                try:
                    if isinstance(data_to_load, (list, tuple)):
                        setattr(self, key, deque(data_to_load, maxlen=self.plot_window))
                    else:
                        print(
                            f"  -> Warning: Invalid type for deque '{key}'. Initializing empty."
                        )
                        setattr(self, key, deque(maxlen=self.plot_window))
                except Exception as e:
                    print(f"  -> Error loading deque '{key}': {e}. Initializing empty.")
                    setattr(self, key, deque(maxlen=self.plot_window))
            else:
                # Ensure deque exists even if not in state_dict
                setattr(self, key, deque(maxlen=self.plot_window))

        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_epsilon",
            "current_beta",
            "current_buffer_size",
            "current_global_step",
            "current_lr",
            "start_time",
            "training_target_step",
            "current_cpu_usage",
            "current_memory_usage",
            "current_gpu_memory_usage_percent",
        ]
        default_values = {
            "start_time": time.time(),
            "training_target_step": 0,
            "current_global_step": 0,
        }
        for key in scalar_keys:
            value_to_load = state_dict.get(key, default_values.get(key, 0))
            setattr(self, key, value_to_load)

        # Best Value Tracking
        best_value_keys = [
            "best_score",
            "previous_best_score",
            "best_score_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
            "best_policy_loss",
            "previous_best_policy_loss",
            "best_policy_loss_step",
        ]
        default_best = {
            "best_score": -float("inf"),
            "previous_best_score": -float("inf"),
            "best_game_score": -float("inf"),
            "previous_best_game_score": -float("inf"),
            "best_value_loss": float("inf"),
            "previous_best_value_loss": float("inf"),
            "best_policy_loss": float("inf"),  # Added policy loss tracking
            "previous_best_policy_loss": float("inf"),
        }
        for key in best_value_keys:
            setattr(self, key, state_dict.get(key, default_best.get(key, 0)))

        # Ensure current_global_step exists after loading
        if not hasattr(self, "current_global_step"):
            self.current_global_step = 0
        # Ensure training_target_step exists after loading
        if not hasattr(self, "training_target_step"):
            self.training_target_step = 0
        # Ensure policy loss tracking exists
        if not hasattr(self, "best_policy_loss"):
            self.best_policy_loss = float("inf")
        if not hasattr(self, "previous_best_policy_loss"):
            self.previous_best_policy_loss = float("inf")
        if not hasattr(self, "best_policy_loss_step"):
            self.best_policy_loss_step = 0
