# File: stats/aggregator_logic.py
# File: stats/aggregator_logic.py
from collections import deque
from typing import Deque, Dict, Any, Optional, List
import numpy as np
import time
import copy  # Import copy for deepcopy

from .aggregator_storage import AggregatorStorage

# Import GameState only for type hinting if needed, avoid direct dependency
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.game_state import GameState


class AggregatorLogic:
    """Handles the calculation logic for StatsAggregator."""

    def __init__(self, storage: AggregatorStorage):
        self.storage = storage

    def update_episode_stats(
        self,
        episode_score: float,  # Game outcome (-1, 0, 1)
        episode_length: int,
        episode_num: int,
        current_step: int,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional[
            "GameState"
        ] = None,  # Pass the actual GameState object
    ) -> Dict[str, Any]:
        """Updates storage with episode data and checks for bests."""
        update_info = {"new_best_rl": False, "new_best_game": False}

        self.storage.episode_scores.append(episode_score)
        self.storage.episode_lengths.append(episode_length)
        if game_score is not None:
            self.storage.game_scores.append(game_score)
        if triangles_cleared is not None:
            self.storage.episode_triangles_cleared.append(triangles_cleared)
            self.storage.total_triangles_cleared += triangles_cleared
        self.storage.total_episodes = episode_num

        # Track best game outcome (closer to 1 is better)
        if episode_score > self.storage.best_score:
            self.storage.previous_best_score = self.storage.best_score
            self.storage.best_score = episode_score
            self.storage.best_score_step = current_step
            update_info["new_best_rl"] = True  # Keep flag name for consistency

        # Track best game score and store corresponding state data
        if game_score is not None and game_score > self.storage.best_game_score:
            self.storage.previous_best_game_score = self.storage.best_game_score
            self.storage.best_game_score = float(game_score)
            self.storage.best_game_score_step = current_step
            update_info["new_best_game"] = True
            # Store data needed to render the best state
            if game_state_for_best and hasattr(game_state_for_best, "grid"):
                try:
                    grid = game_state_for_best.grid
                    occupancy = np.array(
                        [[t.is_occupied for t in row] for row in grid.triangles],
                        dtype=bool,
                    )
                    colors = np.array(
                        [[t.color for t in row] for row in grid.triangles], dtype=object
                    )
                    death_cells = np.array(
                        [[t.is_death for t in row] for row in grid.triangles],
                        dtype=bool,
                    )
                    is_up = np.array(
                        [[t.is_up for t in row] for row in grid.triangles], dtype=bool
                    )

                    self.storage.best_game_state_data = {
                        "score": game_score,
                        "occupancy": copy.deepcopy(occupancy),
                        "colors": copy.deepcopy(colors),
                        "death": copy.deepcopy(death_cells),
                        "is_up": copy.deepcopy(is_up),
                        "rows": grid.rows,
                        "cols": grid.cols,
                        "step": current_step,
                    }
                    print(
                        f"[Aggregator] New best game state saved (Score: {game_score} at Step {current_step})"
                    )
                except Exception as e:
                    print(f"[Aggregator] Error saving best game state data: {e}")
                    self.storage.best_game_state_data = None  # Clear on error

        # RL score history removed
        # self.storage.best_rl_score_history.append(self.storage.best_score)
        current_best_game = (
            int(self.storage.best_game_score)
            if self.storage.best_game_score > -float("inf")
            else 0
        )
        self.storage.best_game_score_history.append(current_best_game)

        return update_info

    def update_step_stats(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Updates storage with step data and checks for best loss."""
        g_step = step_data.get("global_step", self.storage.current_global_step)
        if g_step > self.storage.current_global_step:
            self.storage.current_global_step = g_step

        if "training_target_step" in step_data:
            self.storage.training_target_step = step_data["training_target_step"]

        update_info = {
            "new_best_value_loss": False,
            "new_best_policy_loss": False,
        }

        # Append to deques
        # --- NN Policy Loss ---
        if "policy_loss" in step_data and step_data["policy_loss"] is not None:
            current_policy_loss = step_data["policy_loss"]
            if np.isfinite(current_policy_loss):
                self.storage.policy_losses.append(current_policy_loss)
                if current_policy_loss < self.storage.best_policy_loss and g_step > 0:
                    self.storage.previous_best_policy_loss = (
                        self.storage.best_policy_loss
                    )
                    self.storage.best_policy_loss = current_policy_loss
                    self.storage.best_policy_loss_step = g_step
                    update_info["new_best_policy_loss"] = True
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Policy Loss: {current_policy_loss}"
                )
        # --- End NN Policy Loss ---

        # --- NN Value Loss ---
        if "value_loss" in step_data and step_data["value_loss"] is not None:
            current_value_loss = step_data["value_loss"]
            if np.isfinite(current_value_loss):
                self.storage.value_losses.append(current_value_loss)
                if current_value_loss < self.storage.best_value_loss and g_step > 0:
                    self.storage.previous_best_value_loss = self.storage.best_value_loss
                    self.storage.best_value_loss = current_value_loss
                    self.storage.best_value_loss_step = g_step
                    update_info["new_best_value_loss"] = True
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Value Loss: {current_value_loss}"
                )
        # --- End NN Value Loss ---

        # Append other optional metrics
        optional_metrics = [
            # ("avg_max_q", "avg_max_qs"), # Removed PPO/DQN specific
            # ("beta", "beta_values"), # Removed PER specific
            ("buffer_size", "buffer_sizes"),
            ("lr", "lr_values"),
            # ("epsilon", "epsilon_values"), # Removed Epsilon-greedy specific
            ("cpu_usage", "cpu_usage"),
            ("memory_usage", "memory_usage"),
            ("gpu_memory_usage_percent", "gpu_memory_usage_percent"),
        ]
        for data_key, deque_name in optional_metrics:
            if data_key in step_data and step_data[data_key] is not None:
                val = step_data[data_key]
                if np.isfinite(val):
                    if hasattr(self.storage, deque_name):
                        getattr(self.storage, deque_name).append(val)
                    else:
                        print(
                            f"[Aggregator Warning] Deque '{deque_name}' not found in storage."
                        )
                else:
                    print(f"[Aggregator Warning] Received non-finite {data_key}: {val}")

        # Update scalar values
        scalar_updates = {
            # "beta": "current_beta", # Removed PER specific
            "buffer_size": "current_buffer_size",
            "lr": "current_lr",
            # "epsilon": "current_epsilon", # Removed Epsilon-greedy specific
            "cpu_usage": "current_cpu_usage",
            "memory_usage": "current_memory_usage",
            "gpu_memory_usage_percent": "current_gpu_memory_usage_percent",
        }
        for data_key, storage_key in scalar_updates.items():
            if data_key in step_data and step_data[data_key] is not None:
                val = step_data[data_key]
                if np.isfinite(val):
                    setattr(self.storage, storage_key, val)
                else:
                    print(f"[Aggregator Warning] Received non-finite {data_key}: {val}")

        return update_info

    def calculate_summary(
        self, current_global_step: int, summary_avg_window: int
    ) -> Dict[str, Any]:
        """Calculates the summary dictionary based on stored data."""

        def safe_mean(q_name: str, default=0.0) -> float:
            if not hasattr(self.storage, q_name):
                return default
            deque_instance = self.storage.get_deque(q_name)
            window_data = list(deque_instance)[-summary_avg_window:]
            finite_data = [x for x in window_data if np.isfinite(x)]
            return float(np.mean(finite_data)) if finite_data else default

        summary = {
            "avg_score_window": safe_mean("episode_scores"),  # Game outcome avg
            "avg_length_window": safe_mean("episode_lengths"),
            "policy_loss": safe_mean("policy_losses"),
            "value_loss": safe_mean("value_losses"),
            # "avg_max_q_window": safe_mean("avg_max_qs"), # Removed PPO/DQN specific
            "avg_game_score_window": safe_mean("game_scores"),
            "avg_triangles_cleared_window": safe_mean("episode_triangles_cleared"),
            "avg_lr_window": safe_mean("lr_values", default=self.storage.current_lr),
            "avg_cpu_window": safe_mean("cpu_usage"),
            "avg_memory_window": safe_mean("memory_usage"),
            "avg_gpu_memory_window": safe_mean("gpu_memory_usage_percent"),
            "total_episodes": self.storage.total_episodes,
            # "beta": self.storage.current_beta, # Removed PER specific
            "buffer_size": self.storage.current_buffer_size,
            "global_step": current_global_step,
            "current_lr": self.storage.current_lr,
            "best_score": self.storage.best_score,  # Best game outcome
            "previous_best_score": self.storage.previous_best_score,
            "best_score_step": self.storage.best_score_step,
            "best_game_score": self.storage.best_game_score,
            "previous_best_game_score": self.storage.previous_best_game_score,
            "best_game_score_step": self.storage.best_game_score_step,
            "best_value_loss": self.storage.best_value_loss,
            "previous_best_value_loss": self.storage.previous_best_value_loss,
            "best_value_loss_step": self.storage.best_value_loss_step,
            "best_policy_loss": self.storage.best_policy_loss,
            "previous_best_policy_loss": self.storage.previous_best_policy_loss,
            "best_policy_loss_step": self.storage.best_policy_loss_step,
            "num_ep_scores": len(self.storage.episode_scores),
            "num_value_losses": len(self.storage.value_losses),
            "num_policy_losses": len(self.storage.policy_losses),
            "summary_avg_window_size": summary_avg_window,
            "start_time": self.storage.start_time,
            "training_target_step": self.storage.training_target_step,
            "current_cpu_usage": self.storage.current_cpu_usage,
            "current_memory_usage": self.storage.current_memory_usage,
            "current_gpu_memory_usage_percent": self.storage.current_gpu_memory_usage_percent,
            "best_game_state_data": self.storage.best_game_state_data,  # Include best state data
        }
        return summary
