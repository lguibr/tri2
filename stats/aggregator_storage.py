# File: stats/aggregator_storage.py
from collections import deque
from typing import Deque, Dict, Any, List, Optional
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator.
    Refactored for AlphaZero focus. Resource usage removed."""

    def __init__(self, plot_window: int):
        self.plot_window = plot_window

        # --- Deques for Plotting ---
        # Training Stats
        self.policy_losses: Deque[float] = deque(maxlen=plot_window)
        self.value_losses: Deque[float] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(maxlen=plot_window)
        # Game Stats
        self.episode_outcomes: Deque[float] = deque(maxlen=plot_window)  # -1, 0, 1
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_triangles_cleared: Deque[int] = deque(maxlen=plot_window)
        self.best_game_score_history: Deque[int] = deque(maxlen=plot_window)
        # MCTS Stats
        self.mcts_simulation_times: Deque[float] = deque(maxlen=plot_window)
        self.mcts_nn_prediction_times: Deque[float] = deque(maxlen=plot_window)
        self.mcts_nodes_explored: Deque[int] = deque(maxlen=plot_window)
        self.mcts_avg_depths: Deque[float] = deque(maxlen=plot_window)
        # System Stats
        self.buffer_sizes: Deque[int] = deque(maxlen=plot_window)
        self.steps_per_second: Deque[float] = deque(
            maxlen=plot_window
        )  # Added steps/sec deque
        self._last_step_time: Optional[float] = None
        self._last_step_count: Optional[int] = None

        # --- Scalar State Variables ---
        self.total_episodes: int = 0
        self.total_triangles_cleared: int = 0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0
        self.current_lr: float = 0.0
        self.start_time: float = time.time()
        self.training_target_step: int = 0

        # --- Intermediate Progress Tracking ---
        self.current_self_play_game_number: int = 0
        self.current_self_play_game_steps: int = 0
        self.training_steps_performed: int = 0

        # --- Best Value Tracking ---
        self.best_outcome: float = -float("inf")  # Less relevant now
        self.previous_best_outcome: float = -float("inf")  # Less relevant now
        self.best_outcome_step: int = 0  # Less relevant now
        self.best_game_score: float = -float("inf")
        self.previous_best_game_score: float = -float("inf")
        self.best_game_score_step: int = 0
        self.best_value_loss: float = float("inf")
        self.previous_best_value_loss: float = float("inf")
        self.best_value_loss_step: int = 0
        self.best_policy_loss: float = float("inf")
        self.previous_best_policy_loss: float = float("inf")
        self.best_policy_loss_step: int = 0
        self.best_mcts_sim_time: float = float("inf")  # Best is lowest time
        self.previous_best_mcts_sim_time: float = float("inf")
        self.best_mcts_sim_time_step: int = 0

        # --- Best Game State Data ---
        self.best_game_state_data: Optional[Dict[str, Any]] = None

    def get_deque(self, name: str) -> Deque:
        """Safely gets a deque attribute."""
        return getattr(self, name, deque(maxlen=self.plot_window))

    def get_all_plot_deques(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        deque_names = [
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "best_game_score_history",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "mcts_nodes_explored",
            "mcts_avg_depths",
            "buffer_sizes",
            "steps_per_second",
        ]
        return {
            name: self.get_deque(name).copy()
            for name in deque_names
            if hasattr(self, name)
        }

    def update_steps_per_second(self, global_step: int):
        """Calculates and updates the steps per second deque."""
        current_time = time.time()
        if self._last_step_time is not None and self._last_step_count is not None:
            time_diff = current_time - self._last_step_time
            step_diff = global_step - self._last_step_count
            if (
                time_diff > 1e-3 and step_diff > 0
            ):  # Avoid division by zero and stale data
                sps = step_diff / time_diff
                self.steps_per_second.append(sps)
            elif (
                step_diff <= 0 and time_diff > 1.0
            ):  # If no steps for a while, record 0
                self.steps_per_second.append(0.0)

        # Update last step time/count for the next calculation
        self._last_step_time = current_time
        self._last_step_count = global_step

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the storage for saving."""
        state = {}
        deque_names = [
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "best_game_score_history",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "mcts_nodes_explored",
            "mcts_avg_depths",
            "buffer_sizes",
            "steps_per_second",
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
            "_last_step_time",
            "_last_step_count",  # Include for sps calculation resume
        ]
        for key in scalar_keys:
            state[key] = getattr(self, key, None if key.startswith("_last") else 0)

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
            "best_mcts_sim_time",
            "previous_best_mcts_sim_time",
            "best_mcts_sim_time_step",
        ]
        for key in best_value_keys:
            default = (
                0
                if "step" in key
                else (
                    float("inf") if ("loss" in key or "time" in key) else -float("inf")
                )
            )
            state[key] = getattr(self, key, default)

        # Serialize best game state data carefully
        if self.best_game_state_data:
            try:
                # Convert GameState object to a serializable format (e.g., its state dict)
                serializable_data = {
                    "score": self.best_game_state_data.get("score"),
                    "step": self.best_game_state_data.get("step"),
                    # Add other relevant scalar info if needed
                }
                # Save the game state's internal state (which should be serializable)
                game_state_obj = self.best_game_state_data.get("game_state")
                if game_state_obj and hasattr(game_state_obj, "get_state"):
                    serializable_data["game_state_dict"] = game_state_obj.get_state()

                state["best_game_state_data"] = serializable_data
            except Exception as e:
                logger.error(f"Error serializing best_game_state_data: {e}")
                state["best_game_state_data"] = None
        else:
            state["best_game_state_data"] = None

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        self.plot_window = plot_window
        deque_names = [
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "best_game_score_history",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "mcts_nodes_explored",
            "mcts_avg_depths",
            "buffer_sizes",
            "steps_per_second",
        ]
        for key in deque_names:
            data = state_dict.get(key)
            if isinstance(data, (list, tuple)):
                setattr(self, key, deque(data, maxlen=self.plot_window))
            else:
                # Initialize empty deque if data is missing or invalid
                setattr(self, key, deque(maxlen=self.plot_window))
                if data is not None:
                    logger.warning(
                        f"Invalid data type for deque '{key}' in loaded state: {type(data)}. Initializing empty deque."
                    )

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
            "_last_step_time",
            "_last_step_count",  # Restore for sps calculation
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
            "_last_step_time": None,
            "_last_step_count": None,
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
            "best_mcts_sim_time",
            "previous_best_mcts_sim_time",
            "best_mcts_sim_time_step",
        ]
        best_defaults = {
            "best_outcome": -float("inf"),
            "previous_best_outcome": -float("inf"),
            "best_outcome_step": 0,
            "best_game_score": -float("inf"),
            "previous_best_game_score": -float("inf"),
            "best_game_score_step": 0,
            "best_value_loss": float("inf"),
            "previous_best_value_loss": float("inf"),
            "best_value_loss_step": 0,
            "best_policy_loss": float("inf"),
            "previous_best_policy_loss": float("inf"),
            "best_policy_loss_step": 0,
            "best_mcts_sim_time": float("inf"),
            "previous_best_mcts_sim_time": float("inf"),
            "best_mcts_sim_time_step": 0,
        }
        for key in best_value_keys:
            setattr(self, key, state_dict.get(key, best_defaults.get(key)))

        # Deserialize best game state data (requires GameState class)
        loaded_best_data = state_dict.get("best_game_state_data")
        if loaded_best_data and isinstance(loaded_best_data, dict):
            try:
                from environment.game_state import GameState  # Local import

                temp_game_state = GameState()
                # We only saved the state dict, not the object itself
                # Reconstructing the exact state might be complex or impossible
                # Store the basic info (score, step) and maybe the state dict for inspection
                self.best_game_state_data = {
                    "score": loaded_best_data.get("score"),
                    "step": loaded_best_data.get("step"),
                    "game_state_dict": loaded_best_data.get(
                        "game_state_dict"
                    ),  # Store the dict
                    # Cannot easily reconstruct the full GameState object here
                }
            except ImportError:
                logger.error(
                    "Could not import GameState during best_game_state_data deserialization."
                )
                self.best_game_state_data = None
            except Exception as e:
                logger.error(f"Error deserializing best_game_state_data: {e}")
                self.best_game_state_data = None
        else:
            self.best_game_state_data = None

        # Ensure critical attributes exist after loading
        for attr, default_factory in [
            ("current_global_step", lambda: 0),
            ("best_game_score", lambda: -float("inf")),
            ("best_game_state_data", lambda: None),
            ("training_steps_performed", lambda: 0),
            ("current_self_play_game_number", lambda: 0),
            ("current_self_play_game_steps", lambda: 0),
            ("best_mcts_sim_time", lambda: float("inf")),
            ("steps_per_second", lambda: deque(maxlen=self.plot_window)),
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, default_factory())
