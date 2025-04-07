# File: stats/aggregator_logic.py
from collections import deque
from typing import Deque, Dict, Any, Optional, List
import numpy as np
import time

from .aggregator_storage import AggregatorStorage


class AggregatorLogic:
    """Handles the calculation logic for StatsAggregator."""

    def __init__(self, storage: AggregatorStorage):
        self.storage = storage

    def update_episode_stats(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        current_step: int,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
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

        if episode_score > self.storage.best_score:
            self.storage.previous_best_score = self.storage.best_score
            self.storage.best_score = episode_score
            self.storage.best_score_step = current_step
            update_info["new_best_rl"] = True

        if game_score is not None and game_score > self.storage.best_game_score:
            self.storage.previous_best_game_score = self.storage.best_game_score
            self.storage.best_game_score = float(game_score)
            self.storage.best_game_score_step = current_step
            update_info["new_best_game"] = True

        self.storage.best_rl_score_history.append(self.storage.best_score)
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
            "new_best_loss": False,
            "new_best_policy_loss": False,
        }  # Added policy loss flag

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
                    update_info["new_best_loss"] = (
                        True  # Keep original flag name for value loss
                    )
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Value Loss: {current_value_loss}"
                )
        # --- End NN Value Loss ---

        # Removed Entropy

        # Append other optional metrics
        optional_metrics = [
            # Removed grad_norm, update_steps_per_second, minibatch_update_sps
            ("avg_max_q", "avg_max_qs"),
            ("beta", "beta_values"),
            ("buffer_size", "buffer_sizes"),
            ("lr", "lr_values"),
            ("epsilon", "epsilon_values"),
            ("cpu_usage", "cpu_usage"),
            ("memory_usage", "memory_usage"),
            ("gpu_memory_usage_percent", "gpu_memory_usage_percent"),
        ]
        for data_key, deque_name in optional_metrics:
            if data_key in step_data and step_data[data_key] is not None:
                val = step_data[data_key]
                if np.isfinite(val):
                    # Ensure deque exists before appending
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
            # Removed SPS scalars
            "beta": "current_beta",
            "buffer_size": "current_buffer_size",
            "lr": "current_lr",
            "epsilon": "current_epsilon",
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
            # Check if deque exists before accessing
            if not hasattr(self.storage, q_name):
                return default
            deque_instance = self.storage.get_deque(q_name)
            window_data = list(deque_instance)[-summary_avg_window:]
            finite_data = [x for x in window_data if np.isfinite(x)]
            return float(np.mean(finite_data)) if finite_data else default

        summary = {
            "avg_score_window": safe_mean("episode_scores"),
            "avg_length_window": safe_mean("episode_lengths"),
            "policy_loss": safe_mean("policy_losses"),  # Added policy loss
            "value_loss": safe_mean("value_losses"),
            # Removed entropy, avg_update_sps, avg_minibatch_sps
            "avg_max_q_window": safe_mean("avg_max_qs"),
            "avg_game_score_window": safe_mean("game_scores"),
            "avg_triangles_cleared_window": safe_mean("episode_triangles_cleared"),
            "avg_lr_window": safe_mean("lr_values", default=self.storage.current_lr),
            "avg_cpu_window": safe_mean("cpu_usage"),
            "avg_memory_window": safe_mean("memory_usage"),
            "avg_gpu_memory_window": safe_mean("gpu_memory_usage_percent"),
            "total_episodes": self.storage.total_episodes,
            "beta": self.storage.current_beta,
            "buffer_size": self.storage.current_buffer_size,
            # Removed SPS scalars
            "global_step": current_global_step,
            "current_lr": self.storage.current_lr,
            "best_score": self.storage.best_score,
            "previous_best_score": self.storage.previous_best_score,
            "best_score_step": self.storage.best_score_step,
            "best_game_score": self.storage.best_game_score,
            "previous_best_game_score": self.storage.previous_best_game_score,
            "best_game_score_step": self.storage.best_game_score_step,
            "best_loss": self.storage.best_value_loss,  # Keep as value loss best
            "previous_best_loss": self.storage.previous_best_value_loss,
            "best_loss_step": self.storage.best_value_loss_step,
            "best_policy_loss": self.storage.best_policy_loss,  # Added policy loss best
            "previous_best_policy_loss": self.storage.previous_best_policy_loss,
            "best_policy_loss_step": self.storage.best_policy_loss_step,
            "num_ep_scores": len(self.storage.episode_scores),
            "num_losses": len(
                self.storage.value_losses
            ),  # Maybe rename to num_value_losses?
            "summary_avg_window_size": summary_avg_window,
            "start_time": self.storage.start_time,
            "training_target_step": self.storage.training_target_step,
            "current_cpu_usage": self.storage.current_cpu_usage,
            "current_memory_usage": self.storage.current_memory_usage,
            "current_gpu_memory_usage_percent": self.storage.current_gpu_memory_usage_percent,
        }
        return summary
