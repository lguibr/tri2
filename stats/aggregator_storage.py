from collections import deque
from typing import Deque, Dict, Any, List
import time
import numpy as np


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator."""

    def __init__(self, plot_window: int):
        self.plot_window = plot_window

        # --- Deques for Plotting ---
        self.policy_losses: Deque[float] = deque(maxlen=plot_window)
        self.value_losses: Deque[float] = deque(maxlen=plot_window)
        self.entropies: Deque[float] = deque(maxlen=plot_window)
        self.grad_norms: Deque[float] = deque(maxlen=plot_window)
        self.avg_max_qs: Deque[float] = deque(maxlen=plot_window)
        self.episode_scores: Deque[float] = deque(maxlen=plot_window)
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_triangles_cleared: Deque[int] = deque(maxlen=plot_window)
        self.sps_values: Deque[float] = deque(
            maxlen=plot_window
        )  # Deprecated, keep for loading old checkpoints?
        self.update_steps_per_second_values: Deque[float] = deque(
            maxlen=plot_window
        )  # Overall SPS
        self.minibatch_update_sps_values: Deque[float] = deque(
            maxlen=plot_window
        )  # NEW: SPS per minibatch
        self.buffer_sizes: Deque[int] = deque(maxlen=plot_window)
        self.beta_values: Deque[float] = deque(maxlen=plot_window)
        self.best_rl_score_history: Deque[float] = deque(maxlen=plot_window)
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
        self.current_sps: float = 0.0  # Deprecated
        self.current_update_steps_per_second: float = 0.0  # Overall SPS
        self.current_minibatch_update_sps: float = 0.0  # NEW: SPS per minibatch
        self.current_lr: float = 0.0
        self.start_time: float = time.time()
        self.training_target_step: int = 0  # Initialized to 0, set by CheckpointManager
        self.current_cpu_usage: float = 0.0
        self.current_memory_usage: float = 0.0
        self.current_gpu_memory_usage_percent: float = 0.0

        # --- Best Value Tracking ---
        self.best_score: float = -float("inf")
        self.previous_best_score: float = -float("inf")
        self.best_score_step: int = 0
        self.best_game_score: float = -float("inf")
        self.previous_best_game_score: float = -float("inf")
        self.best_game_score_step: int = 0
        self.best_value_loss: float = float("inf")
        self.previous_best_value_loss: float = float("inf")
        self.best_value_loss_step: int = 0

    def get_deque(self, name: str) -> Deque:
        """Safely gets a deque attribute."""
        return getattr(self, name, deque(maxlen=self.plot_window))

    def get_all_plot_deques(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        deque_names = [
            "policy_losses",
            "value_losses",
            "entropies",
            "grad_norms",
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "update_steps_per_second_values",  # Overall SPS
            "minibatch_update_sps_values",  # NEW: Minibatch SPS
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
        ]
        return {name: self.get_deque(name).copy() for name in deque_names}

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the storage for saving."""
        state = {}
        # Deques
        deque_names = [
            "policy_losses",
            "value_losses",
            "entropies",
            "grad_norms",
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "sps_values",  # Keep for loading old checkpoints
            "update_steps_per_second_values",  # Overall SPS
            "minibatch_update_sps_values",  # NEW: Minibatch SPS
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
        ]
        for name in deque_names:
            state[name] = list(self.get_deque(name))

        # Scalar State Variables
        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_epsilon",
            "current_beta",
            "current_buffer_size",
            "current_global_step",
            "current_sps",  # Keep for loading old checkpoints
            "current_update_steps_per_second",  # Overall SPS
            "current_minibatch_update_sps",  # NEW: Minibatch SPS
            "current_lr",
            "start_time",
            "training_target_step",  # Add target step
            "current_cpu_usage",
            "current_memory_usage",
            "current_gpu_memory_usage_percent",
        ]
        for key in scalar_keys:
            state[key] = getattr(self, key, 0)  # Default to 0 if missing

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
        ]
        for key in best_value_keys:
            state[key] = getattr(self, key, 0)  # Default to 0 if missing

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        self.plot_window = plot_window  # Update plot window size first

        # Deques
        deque_names = [
            "policy_losses",
            "value_losses",
            "entropies",
            "grad_norms",
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "sps_values",  # Keep for loading old checkpoints
            "update_steps_per_second_values",  # Overall SPS
            "minibatch_update_sps_values",  # NEW: Minibatch SPS
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
        ]
        for key in deque_names:
            # Handle potential renaming from old checkpoints
            data_to_load = None
            if key == "update_steps_per_second_values" and key not in state_dict:
                # Try loading from old key "update_sps_values" or "sps_values" if new key missing
                old_key_1 = "update_sps_values"
                old_key_2 = "sps_values"
                if old_key_1 in state_dict:
                    data_to_load = state_dict[old_key_1]
                    print(
                        f"  -> Info: Loading deque '{key}' from old key '{old_key_1}'."
                    )
                elif old_key_2 in state_dict:
                    data_to_load = state_dict[old_key_2]
                    print(
                        f"  -> Info: Loading deque '{key}' from old key '{old_key_2}'."
                    )
            elif key in state_dict:
                data_to_load = state_dict[key]

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

        # Scalar State Variables
        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_epsilon",
            "current_beta",
            "current_buffer_size",
            "current_global_step",
            "current_sps",  # Keep for loading old checkpoints
            "current_update_steps_per_second",  # Overall SPS
            "current_minibatch_update_sps",  # NEW: Minibatch SPS
            "current_lr",
            "start_time",
            "training_target_step",  # Add target step
            "current_cpu_usage",
            "current_memory_usage",
            "current_gpu_memory_usage_percent",
        ]
        default_values = {
            "start_time": time.time(),
            "training_target_step": 0,  # Default target step if not found
            "current_global_step": 0,
        }
        for key in scalar_keys:
            # Handle potential renaming from old checkpoints
            value_to_load = None
            if key == "current_update_steps_per_second" and key not in state_dict:
                # Try loading from old key "current_update_sps" or "current_sps"
                old_key_1 = "current_update_sps"
                old_key_2 = "current_sps"
                if old_key_1 in state_dict:
                    value_to_load = state_dict[old_key_1]
                    print(
                        f"  -> Info: Loading scalar '{key}' from old key '{old_key_1}'."
                    )
                elif old_key_2 in state_dict:
                    value_to_load = state_dict[old_key_2]
                    print(
                        f"  -> Info: Loading scalar '{key}' from old key '{old_key_2}'."
                    )
                else:
                    value_to_load = default_values.get(key, 0)
            # Handle new minibatch SPS scalar
            elif key == "current_minibatch_update_sps" and key not in state_dict:
                value_to_load = default_values.get(key, 0)
            else:
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
        ]
        default_best = {
            "best_score": -float("inf"),
            "previous_best_score": -float("inf"),
            "best_game_score": -float("inf"),
            "previous_best_game_score": -float("inf"),
            "best_value_loss": float("inf"),
            "previous_best_value_loss": float("inf"),
        }
        for key in best_value_keys:
            setattr(self, key, state_dict.get(key, default_best.get(key, 0)))

        # Ensure current_global_step exists after loading
        if not hasattr(self, "current_global_step"):
            self.current_global_step = 0
        # Ensure training_target_step exists after loading
        if not hasattr(self, "training_target_step"):
            self.training_target_step = 0
        # Ensure new scalar exists after loading
        if not hasattr(self, "current_minibatch_update_sps"):
            self.current_minibatch_update_sps = 0.0
