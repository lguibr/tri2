# File: stats/aggregator_storage.py
from collections import deque
from typing import Deque, Dict, Any, List, Optional
import time
import numpy as np
import logging
import pickle

logger = logging.getLogger(__name__)


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator."""

    def __init__(self, plot_window: int):
        self.plot_window = plot_window
        # Training Stats
        self.policy_losses: Deque[float] = deque(maxlen=plot_window)
        self.value_losses: Deque[float] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(maxlen=plot_window)
        # Game Stats
        self.episode_outcomes: Deque[float] = deque(maxlen=plot_window)
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
        self.nodes_per_second: Deque[float] = deque(maxlen=plot_window) # New deque for Nodes/Sec
        self._last_step_time: Optional[float] = None
        self._last_step_count: Optional[int] = None
        self._last_nodes_time: Optional[float] = None # Time for nodes/sec calc
        self._last_nodes_count: Optional[int] = None # Count for nodes/sec calc

        # --- Scalar State Variables ---
        self.total_episodes: int = 0
        self.total_triangles_cleared: int = 0
        self.total_mcts_nodes_explored: int = 0 # New total counter
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
        self.best_game_state_data: Optional[Dict[str, Any]] = None

    def get_deque(self, name: str) -> Deque:
        return getattr(self, name, deque(maxlen=self.plot_window))

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
            "nodes_per_second", # Added new deque
        ]
        return {
            name: self.get_deque(name).copy()
            for name in deque_names
            if hasattr(self, name)
        }

    def update_steps_per_second(self, global_step: int):
        current_time = time.time()
        if self._last_step_time is not None and self._last_step_count is not None:
            time_diff = current_time - self._last_step_time
            step_diff = global_step - self._last_step_count
            if time_diff > 1e-3 and step_diff > 0:
                sps = step_diff / time_diff
                self.steps_per_second.append(sps)
            elif step_diff <= 0 and time_diff > 1.0: # Log 0 if no progress for a while
                self.steps_per_second.append(0.0)
        self._last_step_time = current_time
        self._last_step_count = global_step

    def update_nodes_per_second(self):
        """Calculates and updates the MCTS nodes explored per second."""
        current_time = time.time()
        if self._last_nodes_time is not None and self._last_nodes_count is not None:
            time_diff = current_time - self._last_nodes_time
            nodes_diff = self.total_mcts_nodes_explored - self._last_nodes_count
            if time_diff > 1e-3 and nodes_diff > 0:
                nps = nodes_diff / time_diff
                self.nodes_per_second.append(nps)
            elif nodes_diff <= 0 and time_diff > 1.0: # Log 0 if no progress for a while
                self.nodes_per_second.append(0.0)
        self._last_nodes_time = current_time
        self._last_nodes_count = self.total_mcts_nodes_explored


    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the storage for saving."""
        state = {}
        deque_names = [
            "policy_losses", "value_losses", "lr_values",
            "episode_outcomes", "episode_lengths", "game_scores",
            "episode_triangles_cleared", "best_game_score_history",
            "mcts_simulation_times", "mcts_nn_prediction_times",
            "mcts_nodes_explored", "mcts_avg_depths",
            "buffer_sizes", "steps_per_second", "nodes_per_second", # Added nodes_per_second
        ]
        for name in deque_names:
            if hasattr(self, name):
                deque_instance = getattr(self, name, None)
                if deque_instance is not None:
                    state[name] = list(deque_instance)

        scalar_keys = [
            "total_episodes", "total_triangles_cleared", "total_mcts_nodes_explored", # Added total_mcts_nodes_explored
            "current_buffer_size", "current_global_step", "current_lr",
            "start_time", "training_target_step",
            "current_self_play_game_number", "current_self_play_game_steps",
            "training_steps_performed",
            "_last_step_time", "_last_step_count",
            "_last_nodes_time", "_last_nodes_count", # Added node timing/count state
        ]
        for key in scalar_keys:
            state[key] = getattr(self, key, None if key.startswith("_last") else 0)

        best_value_keys = [
            "best_outcome", "previous_best_outcome", "best_outcome_step",
            "best_game_score", "previous_best_game_score", "best_game_score_step",
            "best_value_loss", "previous_best_value_loss", "best_value_loss_step",
            "best_policy_loss", "previous_best_policy_loss", "best_policy_loss_step",
            "best_mcts_sim_time", "previous_best_mcts_sim_time", "best_mcts_sim_time_step",
        ]
        for key in best_value_keys:
            default = (0 if "step" in key else
                       (float("inf") if ("loss" in key or "time" in key) else -float("inf")))
            state[key] = getattr(self, key, default)

        if self.best_game_state_data:
            try:
                state["best_game_state_data_pkl"] = pickle.dumps(self.best_game_state_data)
            except Exception as e:
                logger.error(f"Could not pickle best_game_state_data: {e}")
                state["best_game_state_data_pkl"] = None
        else:
            state["best_game_state_data_pkl"] = None

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        self.plot_window = plot_window
        deque_names = [
            "policy_losses", "value_losses", "lr_values",
            "episode_outcomes", "episode_lengths", "game_scores",
            "episode_triangles_cleared", "best_game_score_history",
            "mcts_simulation_times", "mcts_nn_prediction_times",
            "mcts_nodes_explored", "mcts_avg_depths",
            "buffer_sizes", "steps_per_second", "nodes_per_second", # Added nodes_per_second
        ]
        for key in deque_names:
            data = state_dict.get(key)
            if isinstance(data, (list, tuple)):
                setattr(self, key, deque(data, maxlen=self.plot_window))
            else:
                setattr(self, key, deque(maxlen=self.plot_window))
                if data is not None:
                    logger.warning(f"Invalid data type for deque '{key}' in loaded state: {type(data)}. Init empty.")

        scalar_keys = [
            "total_episodes", "total_triangles_cleared", "total_mcts_nodes_explored", # Added total_mcts_nodes_explored
            "current_buffer_size", "current_global_step", "current_lr",
            "start_time", "training_target_step",
            "current_self_play_game_number", "current_self_play_game_steps",
            "training_steps_performed",
            "_last_step_time", "_last_step_count",
            "_last_nodes_time", "_last_nodes_count", # Added node timing/count state
        ]
        defaults = {
            "start_time": time.time(), "training_target_step": 0, "current_global_step": 0,
            "total_episodes": 0, "total_triangles_cleared": 0, "total_mcts_nodes_explored": 0,
            "current_buffer_size": 0, "current_lr": 0.0,
            "current_self_play_game_number": 0, "current_self_play_game_steps": 0,
            "training_steps_performed": 0,
            "_last_step_time": None, "_last_step_count": None,
            "_last_nodes_time": None, "_last_nodes_count": None,
        }
        for key in scalar_keys:
            setattr(self, key, state_dict.get(key, defaults.get(key)))

        best_value_keys = [
            "best_outcome", "previous_best_outcome", "best_outcome_step",
            "best_game_score", "previous_best_game_score", "best_game_score_step",
            "best_value_loss", "previous_best_value_loss", "best_value_loss_step",
            "best_policy_loss", "previous_best_policy_loss", "best_policy_loss_step",
            "best_mcts_sim_time", "previous_best_mcts_sim_time", "best_mcts_sim_time_step",
        ]
        best_defaults = {
            "best_outcome": -float("inf"), "previous_best_outcome": -float("inf"), "best_outcome_step": 0,
            "best_game_score": -float("inf"), "previous_best_game_score": -float("inf"), "best_game_score_step": 0,
            "best_value_loss": float("inf"), "previous_best_value_loss": float("inf"), "best_value_loss_step": 0,
            "best_policy_loss": float("inf"), "previous_best_policy_loss": float("inf"), "best_policy_loss_step": 0,
            "best_mcts_sim_time": float("inf"), "previous_best_mcts_sim_time": float("inf"), "best_mcts_sim_time_step": 0,
        }
        for key in best_value_keys:
            setattr(self, key, state_dict.get(key, best_defaults.get(key)))

        best_game_data_pkl = state_dict.get("best_game_state_data_pkl")
        if best_game_data_pkl:
            try:
                self.best_game_state_data = pickle.loads(best_game_data_pkl)
                if not isinstance(self.best_game_state_data, dict):
                    logger.warning("Loaded best_game_state_data is not a dict, resetting.")
                    self.best_game_state_data = None
                else:
                    if ("score" not in self.best_game_state_data or
                        "step" not in self.best_game_state_data or
                        "game_state_dict" not in self.best_game_state_data):
                        logger.warning("Loaded best_game_state_data dict missing keys, resetting.")
                        self.best_game_state_data = None
            except Exception as e:
                logger.error(f"Error unpickling best_game_state_data: {e}")
                self.best_game_state_data = None
        else:
            loaded_best_data = state_dict.get("best_game_state_data")
            if isinstance(loaded_best_data, dict):
                self.best_game_state_data = loaded_best_data
                if ("score" not in self.best_game_state_data or
                    "step" not in self.best_game_state_data or
                    "game_state_dict" not in self.best_game_state_data):
                    logger.warning("Loaded legacy best_game_state_data dict missing keys, resetting.")
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
            ("nodes_per_second", lambda: deque(maxlen=self.plot_window)), # Added nodes_per_second
            ("total_mcts_nodes_explored", lambda: 0), # Added total_mcts_nodes_explored
            ("_last_nodes_time", lambda: None), # Added node timing/count state
            ("_last_nodes_count", lambda: None), # Added node timing/count state
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, default_factory())