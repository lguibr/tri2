# File: stats/aggregator_storage.py
from collections import deque
from typing import Deque, Dict, Any, List, Optional
import time
import numpy as np
import logging
import pickle  # Needed for potential complex data in best_game_state_data

logger = logging.getLogger(__name__)


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator.
    Ensures best_game_state_data is stored as a serializable dictionary.
    """

    def __init__(self, plot_window: int):
        self.plot_window = plot_window
        # ... (Deque definitions remain the same) ...
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
        self.steps_per_second: Deque[float] = deque(maxlen=plot_window)
        self._last_step_time: Optional[float] = None
        self._last_step_count: Optional[int] = None

        # --- Scalar State Variables ---
        # ... (remain the same) ...
        self.total_episodes: int = 0
        self.total_triangles_cleared: int = 0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0
        self.current_lr: float = 0.0
        self.start_time: float = time.time()
        self.training_target_step: int = 0

        # --- Intermediate Progress Tracking ---
        # ... (remain the same) ...
        self.current_self_play_game_number: int = 0
        self.current_self_play_game_steps: int = 0
        self.training_steps_performed: int = 0

        # --- Best Value Tracking ---
        # ... (remain the same) ...
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
        self.best_mcts_sim_time: float = float("inf")
        self.previous_best_mcts_sim_time: float = float("inf")
        self.best_mcts_sim_time_step: int = 0

        # --- Best Game State Data ---
        # Now stores the already processed serializable dict
        self.best_game_state_data: Optional[Dict[str, Any]] = None

    # get_deque remains the same
    def get_deque(self, name: str) -> Deque:
        return getattr(self, name, deque(maxlen=self.plot_window))

    # get_all_plot_deques remains the same
    def get_all_plot_deques(self) -> Dict[str, Deque]:
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

    # update_steps_per_second remains the same
    def update_steps_per_second(self, global_step: int):
        current_time = time.time()
        if self._last_step_time is not None and self._last_step_count is not None:
            time_diff = current_time - self._last_step_time
            step_diff = global_step - self._last_step_count
            if time_diff > 1e-3 and step_diff > 0:
                sps = step_diff / time_diff
                self.steps_per_second.append(sps)
            elif step_diff <= 0 and time_diff > 1.0:
                self.steps_per_second.append(0.0)
        self._last_step_time = current_time
        self._last_step_count = global_step

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the storage for saving."""
        state = {}
        # ... (Deque serialization remains the same) ...
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

        # ... (Scalar serialization remains the same) ...
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
            "_last_step_count",
        ]
        for key in scalar_keys:
            state[key] = getattr(self, key, None if key.startswith("_last") else 0)

        # ... (Best value serialization remains the same) ...
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

        # Serialize best game state data directly (it should already be a dict)
        # We might need pickle for numpy arrays within the dict
        if self.best_game_state_data:
            try:
                # Ensure numpy arrays are handled by pickle if torch save fails
                state["best_game_state_data_pkl"] = pickle.dumps(
                    self.best_game_state_data
                )
            except Exception as e:
                logger.error(f"Could not pickle best_game_state_data: {e}")
                state["best_game_state_data_pkl"] = None
        else:
            state["best_game_state_data_pkl"] = None

        # Deprecated direct storage in torch checkpoint:
        # state["best_game_state_data"] = self.best_game_state_data

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        # ... (Deque loading remains the same) ...
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
                setattr(self, key, deque(maxlen=self.plot_window))
                if data is not None:
                    logger.warning(
                        f"Invalid data type for deque '{key}' in loaded state: {type(data)}. Init empty."
                    )

        # ... (Scalar loading remains the same) ...
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
            "_last_step_count",
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

        # ... (Best value loading remains the same) ...
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

        # Deserialize best game state data using pickle
        best_game_data_pkl = state_dict.get("best_game_state_data_pkl")
        if best_game_data_pkl:
            try:
                self.best_game_state_data = pickle.loads(best_game_data_pkl)
                if not isinstance(self.best_game_state_data, dict):
                    logger.warning(
                        "Loaded best_game_state_data is not a dict, resetting."
                    )
                    self.best_game_state_data = None
                else:
                    # Basic validation
                    if (
                        "score" not in self.best_game_state_data
                        or "step" not in self.best_game_state_data
                        or "game_state_dict" not in self.best_game_state_data
                    ):
                        logger.warning(
                            "Loaded best_game_state_data dict missing keys, resetting."
                        )
                        self.best_game_state_data = None
            except Exception as e:
                logger.error(f"Error unpickling best_game_state_data: {e}")
                self.best_game_state_data = None
        else:
            # Fallback to old direct storage method if pkl missing
            loaded_best_data = state_dict.get("best_game_state_data")
            if isinstance(loaded_best_data, dict):
                self.best_game_state_data = loaded_best_data
                # Basic validation
                if (
                    "score" not in self.best_game_state_data
                    or "step" not in self.best_game_state_data
                    or "game_state_dict" not in self.best_game_state_data
                ):
                    logger.warning(
                        "Loaded legacy best_game_state_data dict missing keys, resetting."
                    )
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
