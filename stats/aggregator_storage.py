from collections import deque
from typing import Deque, Dict, Any, List, Optional
import time
import numpy as np


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator.
    Refactored for AlphaZero focus. Resource usage removed."""

    def __init__(self, plot_window: int):
        self.plot_window = plot_window

        # --- Deques for Plotting (AlphaZero Relevant) ---
        # Stores recent values for plotting trends. Max length defined by plot_window.
        self.policy_losses: Deque[float] = deque(maxlen=plot_window)
        self.value_losses: Deque[float] = deque(maxlen=plot_window)
        self.episode_outcomes: Deque[float] = deque(
            maxlen=plot_window
        )  # -1 (loss), 0 (draw), 1 (win)
        self.episode_lengths: Deque[int] = deque(
            maxlen=plot_window
        )  # Steps per episode
        self.game_scores: Deque[int] = deque(
            maxlen=plot_window
        )  # Raw game score per episode
        self.episode_triangles_cleared: Deque[int] = deque(maxlen=plot_window)
        self.buffer_sizes: Deque[int] = deque(
            maxlen=plot_window
        )  # Replay buffer size over time
        self.best_game_score_history: Deque[int] = deque(
            maxlen=plot_window
        )  # Tracks the best score found so far
        self.lr_values: Deque[float] = deque(
            maxlen=plot_window
        )  # Learning rate over time

        # --- Scalar State Variables ---
        # Tracks overall progress and current state.
        self.total_episodes: int = 0  # Total completed episodes since start/load
        self.total_triangles_cleared: int = 0  # Cumulative triangles cleared
        self.current_buffer_size: int = 0  # Latest known buffer size
        self.current_global_step: int = 0  # Tracks NN training steps primarily
        self.current_lr: float = 0.0  # Current learning rate
        self.start_time: float = time.time()  # Timestamp of aggregator creation/load
        self.training_target_step: int = (
            0  # Target training step for completion (if any)
        )

        # --- Intermediate Progress Tracking ---
        # Useful for detailed status updates during runs.
        self.current_self_play_game_number: int = (
            0  # Track which game is being played by workers
        )
        self.current_self_play_game_steps: int = (
            0  # Steps within the current self-play game
        )
        self.training_steps_performed: int = (
            0  # Total training steps executed by training worker
        )

        # --- Best Value Tracking (AlphaZero Relevant) ---
        # Stores the best values achieved and the step they occurred at.
        self.best_outcome: float = -float("inf")
        self.previous_best_outcome: float = -float("inf")
        self.best_outcome_step: int = 0
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
        # Stores data needed to visualize the best game state found.
        self.best_game_state_data: Optional[Dict[str, Any]] = None

    def get_deque(self, name: str) -> Deque:
        """Safely gets a deque attribute."""
        return getattr(self, name, deque(maxlen=self.plot_window))

    def get_all_plot_deques(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        deque_names = [
            "policy_losses",
            "value_losses",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "best_game_score_history",
            "lr_values",
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
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "best_game_score_history",
            "lr_values",
        ]
        for name in deque_names:
            if hasattr(self, name):
                deque_instance = getattr(self, name, None)
                if deque_instance is not None:
                    state[name] = list(deque_instance)

        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_buffer_size",
            "current_global_step",
            "current_lr",
            "start_time",
            "training_target_step",
            "current_self_play_game_number",
            "current_self_play_game_steps",
            "training_steps_performed",
        ]
        for key in scalar_keys:
            state[key] = getattr(self, key, 0)

        best_value_keys = [
            "best_outcome",
            "previous_best_outcome",
            "best_outcome_step",
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
            default = (
                0
                if "step" in key
                else (float("inf") if "loss" in key else -float("inf"))
            )
            state[key] = getattr(self, key, default)

        if self.best_game_state_data:
            serializable_data = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.best_game_state_data.items()
            }
            state["best_game_state_data"] = serializable_data
        else:
            state["best_game_state_data"] = None
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        self.plot_window = plot_window
        deque_names = [
            "policy_losses",
            "value_losses",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "best_game_score_history",
            "lr_values",
        ]
        for key in deque_names:
            data = state_dict.get(key)
            if isinstance(data, (list, tuple)):
                setattr(self, key, deque(data, maxlen=self.plot_window))
            else:
                setattr(self, key, deque(maxlen=self.plot_window))

        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_buffer_size",
            "current_global_step",
            "current_lr",
            "start_time",
            "training_target_step",
            "current_self_play_game_number",
            "current_self_play_game_steps",
            "training_steps_performed",
        ]
        defaults = {
            "start_time": time.time(),
            "training_target_step": 0,
            "current_global_step": 0,
            "total_episodes": 0,
            "total_triangles_cleared": 0,
            "current_buffer_size": 0,
            "current_lr": 0.0,
            "current_self_play_game_number": 0,
            "current_self_play_game_steps": 0,
            "training_steps_performed": 0,
        }
        for key in scalar_keys:
            setattr(self, key, state_dict.get(key, defaults.get(key)))

        best_value_keys = [
            "best_outcome",
            "previous_best_outcome",
            "best_outcome_step",
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
        best_defaults = {
            "best_outcome": -float("inf"),
            "previous_best_outcome": -float("inf"),
            "best_game_score": -float("inf"),
            "previous_best_game_score": -float("inf"),
            "best_value_loss": float("inf"),
            "previous_best_value_loss": float("inf"),
            "best_policy_loss": float("inf"),
            "previous_best_policy_loss": float("inf"),
            "best_outcome_step": 0,
            "best_game_score_step": 0,
            "best_value_loss_step": 0,
            "best_policy_loss_step": 0,
        }
        for key in best_value_keys:
            setattr(self, key, state_dict.get(key, best_defaults.get(key)))

        loaded_best_data = state_dict.get("best_game_state_data")
        if loaded_best_data:
            try:
                self.best_game_state_data = {
                    k: (
                        np.array(v)
                        if isinstance(v, list) and v and isinstance(v[0], list)
                        else v
                    )
                    for k, v in loaded_best_data.items()
                }
            except Exception as e:
                print(f"Error converting loaded best_game_state_data: {e}")
                self.best_game_state_data = None
        else:
            self.best_game_state_data = None

        # Ensure critical attributes exist
        for attr, default in [
            ("current_global_step", 0),
            ("best_game_score", -float("inf")),
            ("best_game_state_data", None),
            ("training_steps_performed", 0),
            ("current_self_play_game_number", 0),
            ("current_self_play_game_steps", 0),
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, default)
