# File: stats/aggregator_storage.py
# File: stats/aggregator_storage.py
from collections import deque
from typing import Deque, Dict, Any, List, Optional  # Added Optional
import time
import numpy as np


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator."""

    def __init__(self, plot_window: int):
        self.plot_window = plot_window

        # --- Deques for Plotting ---
        self.policy_losses: Deque[float] = deque(maxlen=plot_window)
        self.value_losses: Deque[float] = deque(maxlen=plot_window)
        self.avg_max_qs: Deque[float] = deque(maxlen=plot_window)
        self.episode_scores: Deque[float] = deque(
            maxlen=plot_window
        )  # Game outcome (-1, 0, 1)
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_triangles_cleared: Deque[int] = deque(maxlen=plot_window)
        self.buffer_sizes: Deque[int] = deque(maxlen=plot_window)
        self.beta_values: Deque[float] = deque(maxlen=plot_window)
        # self.best_rl_score_history: Deque[float] = deque(maxlen=plot_window) # Removed RL score history
        self.best_game_score_history: Deque[int] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(maxlen=plot_window)
        self.epsilon_values: Deque[float] = deque(maxlen=plot_window)
        self.cpu_usage: Deque[float] = deque(maxlen=plot_window)
        self.memory_usage: Deque[float] = deque(maxlen=plot_window)
        self.gpu_memory_usage_percent: Deque[float] = deque(maxlen=plot_window)

        # --- Scalar State Variables ---
        self.total_episodes = 0
        self.total_triangles_cleared = 0
        self.current_epsilon: float = 0.0
        self.current_beta: float = 0.0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0
        self.current_lr: float = 0.0
        self.start_time: float = time.time()
        self.training_target_step: int = 0
        self.current_cpu_usage: float = 0.0
        self.current_memory_usage: float = 0.0
        self.current_gpu_memory_usage_percent: float = 0.0

        # --- Best Value Tracking ---
        self.best_score: float = -float(
            "inf"
        )  # Best game outcome (closer to 1 is better)
        self.previous_best_score: float = -float("inf")
        self.best_score_step: int = 0
        self.best_game_score: float = -float("inf")
        self.previous_best_game_score: float = -float("inf")
        self.best_game_score_step: int = 0
        self.best_value_loss: float = float("inf")
        self.previous_best_value_loss: float = float("inf")
        self.best_value_loss_step: int = 0
        self.best_policy_loss: float = float("inf")
        self.previous_best_policy_loss: float = float("inf")
        self.best_policy_loss_step: int = 0

        # --- Best Game State Data ---
        self.best_game_state_data: Optional[Dict[str, Any]] = (
            None  # Store grid data for best game score
        )

    def get_deque(self, name: str) -> Deque:
        """Safely gets a deque attribute."""
        return getattr(self, name, deque(maxlen=self.plot_window))

    def get_all_plot_deques(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        deque_names = [
            "policy_losses",
            "value_losses",
            # "avg_max_qs", # Removed PPO/DQN specific
            "episode_scores",  # Game outcome
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            # "beta_values", # Removed PER specific
            # "best_rl_score_history", # Removed RL score
            "best_game_score_history",
            "lr_values",
            # "epsilon_values", # Removed Epsilon-greedy specific
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
        ]
        return {
            name: self.get_deque(name).copy()
            for name in deque_names
            if hasattr(self, name)
        }

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the storage for saving."""
        state = {}
        deque_names = [
            "policy_losses",
            "value_losses",
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",  # "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
        ]
        for name in deque_names:
            if hasattr(self, name):
                # Ensure deque exists before accessing
                deque_instance = getattr(self, name, None)
                if deque_instance is not None:
                    state[name] = list(deque_instance)

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
        for key in scalar_keys:
            state[key] = getattr(self, key, 0)  # Use default 0 if missing

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
        for key in best_value_keys:
            # Provide appropriate defaults for inf/-inf
            default = 0
            if "loss" in key:
                default = float("inf")
            elif "score" in key:
                default = -float("inf")
            state[key] = getattr(self, key, default)

        # Save best game state data
        state["best_game_state_data"] = self.best_game_state_data

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        self.plot_window = plot_window

        deque_names = [
            "policy_losses",
            "value_losses",
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",  # "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
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
                # Initialize deque if key is missing in saved state
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
            "total_episodes": 0,
            "total_triangles_cleared": 0,
            "current_epsilon": 0.0,
            "current_beta": 0.0,
            "current_buffer_size": 0,
            "current_lr": 0.0,
            "current_cpu_usage": 0.0,
            "current_memory_usage": 0.0,
            "current_gpu_memory_usage_percent": 0.0,
        }
        for key in scalar_keys:
            value_to_load = state_dict.get(key, default_values.get(key))
            setattr(self, key, value_to_load)

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
            "best_policy_loss": float("inf"),
            "previous_best_policy_loss": float("inf"),
            "best_score_step": 0,
            "best_game_score_step": 0,
            "best_value_loss_step": 0,
            "best_policy_loss_step": 0,
        }
        for key in best_value_keys:
            setattr(self, key, state_dict.get(key, default_best.get(key)))

        # Load best game state data
        self.best_game_state_data = state_dict.get("best_game_state_data", None)

        # Ensure critical attributes exist after loading
        if not hasattr(self, "current_global_step"):
            self.current_global_step = 0
        if not hasattr(self, "training_target_step"):
            self.training_target_step = 0
        if not hasattr(self, "best_policy_loss"):
            self.best_policy_loss = float("inf")
        if not hasattr(self, "previous_best_policy_loss"):
            self.previous_best_policy_loss = float("inf")
        if not hasattr(self, "best_policy_loss_step"):
            self.best_policy_loss_step = 0
        if not hasattr(self, "best_game_score"):
            self.best_game_score = -float("inf")
        if not hasattr(self, "best_game_state_data"):
            self.best_game_state_data = None
