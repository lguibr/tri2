# File: stats/aggregator_logic.py
import time
from typing import Deque, Dict, Any, Optional, List, TYPE_CHECKING
import numpy as np
import copy

from .aggregator_storage import AggregatorStorage
from utils.helpers import format_eta

if TYPE_CHECKING:
    from environment.game_state import GameState


class AggregatorLogic:
    """Handles the logic for updating stats and calculating summaries."""

    def __init__(self, storage: AggregatorStorage):
        self.storage = storage

    def _update_deque(self, deque_instance: Deque, value: Any):
        """Appends a value to a deque if it's finite."""
        if isinstance(value, (int, float)) and np.isfinite(value):
            deque_instance.append(value)
        elif isinstance(value, np.number) and np.isfinite(value):
            deque_instance.append(float(value))

    def _calculate_average(self, deque_instance: Deque, window: int) -> float:
        """Calculates the average of the last 'window' elements in a deque."""
        if not deque_instance:
            return 0.0
        count = min(len(deque_instance), window)
        if count == 0:
            return 0.0
        # Use slicing for efficiency
        last_n = list(deque_instance)[-count:]
        return sum(last_n) / count

    def update_episode_stats(
        self,
        episode_outcome: float,
        episode_length: int,
        episode_num: int,
        current_step: int,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional["GameState"] = None,
    ) -> Dict[str, Any]:
        """Updates stats related to a completed episode."""
        update_info = {}
        self.storage.total_episodes += 1
        self._update_deque(self.storage.episode_outcomes, episode_outcome)
        self._update_deque(self.storage.episode_lengths, episode_length)

        if game_score is not None:
            self._update_deque(self.storage.game_scores, game_score)
            if game_score > self.storage.best_game_score:
                self.storage.previous_best_game_score = self.storage.best_game_score
                self.storage.best_game_score = game_score
                self.storage.best_game_score_step = current_step
                update_info["new_best_game"] = True
                self._update_deque(
                    self.storage.best_game_score_history, self.storage.best_game_score
                )
                # Store best game state data
                if game_state_for_best:
                    try:
                        state_dict = game_state_for_best.get_state()
                        self.storage.best_game_state_data = {
                            "score": game_score,
                            "step": current_step,
                            "occupancy": state_dict["grid"][0],  # Occupancy channel
                            "colors": game_state_for_best.grid.get_color_data(),
                            "death": game_state_for_best.grid.get_death_data(),
                            "is_up": state_dict["grid"][1],  # Orientation channel
                            "rows": game_state_for_best.env_config.ROWS,
                            "cols": game_state_for_best.env_config.COLS,
                        }
                    except Exception as e:
                        print(f"Error storing best game state data: {e}")
                        self.storage.best_game_state_data = None
            elif not self.storage.best_game_score_history:  # Add initial best score
                self._update_deque(
                    self.storage.best_game_score_history, self.storage.best_game_score
                )

        if triangles_cleared is not None:
            self._update_deque(
                self.storage.episode_triangles_cleared, triangles_cleared
            )
            self.storage.total_triangles_cleared += triangles_cleared

        # Update intermediate progress tracker
        self.storage.current_self_play_game_number = episode_num + 1
        self.storage.current_self_play_game_steps = 0

        return update_info

    def update_step_stats(
        self, step_data: Dict[str, Any], current_step: int
    ) -> Dict[str, Any]:
        """Updates stats related to a training or environment step."""
        update_info = {}

        # Training Stats
        policy_loss = step_data.get("policy_loss")
        value_loss = step_data.get("value_loss")
        lr = step_data.get("lr")
        training_steps = step_data.get("training_steps_performed")

        if policy_loss is not None:
            self._update_deque(self.storage.policy_losses, policy_loss)
            if policy_loss < self.storage.best_policy_loss:
                self.storage.previous_best_policy_loss = self.storage.best_policy_loss
                self.storage.best_policy_loss = policy_loss
                self.storage.best_policy_loss_step = current_step
                update_info["new_best_policy_loss"] = True

        if value_loss is not None:
            self._update_deque(self.storage.value_losses, value_loss)
            if value_loss < self.storage.best_value_loss:
                self.storage.previous_best_value_loss = self.storage.best_value_loss
                self.storage.best_value_loss = value_loss
                self.storage.best_value_loss_step = current_step
                update_info["new_best_value_loss"] = True

        if lr is not None:
            self._update_deque(self.storage.lr_values, lr)
            self.storage.current_lr = lr

        if training_steps is not None:
            self.storage.training_steps_performed = training_steps

        # MCTS Stats (from SelfPlayWorker)
        mcts_sim_time = step_data.get("mcts_sim_time")
        mcts_nn_time = step_data.get("mcts_nn_time")
        mcts_nodes = step_data.get("mcts_nodes_explored")
        mcts_depth = step_data.get("mcts_avg_depth")

        if mcts_sim_time is not None:
            self._update_deque(self.storage.mcts_simulation_times, mcts_sim_time)
            if mcts_sim_time < self.storage.best_mcts_sim_time:
                self.storage.previous_best_mcts_sim_time = (
                    self.storage.best_mcts_sim_time
                )
                self.storage.best_mcts_sim_time = mcts_sim_time
                self.storage.best_mcts_sim_time_step = current_step
                update_info["new_best_mcts_sim_time"] = True

        if mcts_nn_time is not None:
            self._update_deque(self.storage.mcts_nn_prediction_times, mcts_nn_time)
        if mcts_nodes is not None:
            self._update_deque(self.storage.mcts_nodes_explored, mcts_nodes)
        if mcts_depth is not None:
            self._update_deque(self.storage.mcts_avg_depths, mcts_depth)

        # General Stats
        buffer_size = step_data.get("buffer_size")
        if buffer_size is not None:
            self._update_deque(self.storage.buffer_sizes, buffer_size)
            self.storage.current_buffer_size = buffer_size

        # Intermediate Progress
        game_num = step_data.get("current_self_play_game_number")
        game_step = step_data.get("current_self_play_game_steps")
        if game_num is not None:
            self.storage.current_self_play_game_number = game_num
        if game_step is not None:
            self.storage.current_self_play_game_steps = game_step

        return update_info

    def calculate_summary(
        self, current_global_step: int, avg_window: int
    ) -> Dict[str, Any]:
        """Calculates and returns the summary dictionary."""
        summary = {}
        elapsed_time = time.time() - self.storage.start_time

        # Calculate averages using the specified window
        summary["avg_game_score_window"] = self._calculate_average(
            self.storage.game_scores, avg_window
        )
        summary["avg_episode_length_window"] = self._calculate_average(
            self.storage.episode_lengths, avg_window
        )
        summary["avg_triangles_cleared_window"] = self._calculate_average(
            self.storage.episode_triangles_cleared, avg_window
        )
        summary["avg_policy_loss_window"] = self._calculate_average(
            self.storage.policy_losses, avg_window
        )
        summary["avg_value_loss_window"] = self._calculate_average(
            self.storage.value_losses, avg_window
        )
        # MCTS Averages
        summary["avg_mcts_sim_time_window"] = self._calculate_average(
            self.storage.mcts_simulation_times, avg_window
        )
        summary["avg_mcts_nn_time_window"] = self._calculate_average(
            self.storage.mcts_nn_prediction_times, avg_window
        )
        summary["avg_mcts_nodes_explored_window"] = self._calculate_average(
            self.storage.mcts_nodes_explored, avg_window
        )
        summary["avg_mcts_avg_depth_window"] = self._calculate_average(
            self.storage.mcts_avg_depths, avg_window
        )

        # Add scalar values
        summary["total_episodes"] = self.storage.total_episodes
        summary["total_triangles_cleared"] = self.storage.total_triangles_cleared
        summary["buffer_size"] = self.storage.current_buffer_size
        summary["global_step"] = current_global_step
        summary["current_lr"] = self.storage.current_lr
        summary["elapsed_time_seconds"] = elapsed_time
        summary["start_time"] = self.storage.start_time
        summary["summary_avg_window_size"] = avg_window
        summary["training_target_step"] = self.storage.training_target_step

        # Add intermediate progress
        summary["current_self_play_game_number"] = (
            self.storage.current_self_play_game_number
        )
        summary["current_self_play_game_steps"] = (
            self.storage.current_self_play_game_steps
        )
        summary["training_steps_performed"] = self.storage.training_steps_performed

        # Add best values
        summary["best_outcome"] = self.storage.best_outcome
        summary["best_outcome_step"] = self.storage.best_outcome_step
        summary["best_game_score"] = self.storage.best_game_score
        summary["best_game_score_step"] = self.storage.best_game_score_step
        summary["best_value_loss"] = self.storage.best_value_loss
        summary["best_value_loss_step"] = self.storage.best_value_loss_step
        summary["best_policy_loss"] = self.storage.best_policy_loss
        summary["best_policy_loss_step"] = self.storage.best_policy_loss_step
        summary["best_mcts_sim_time"] = self.storage.best_mcts_sim_time
        summary["best_mcts_sim_time_step"] = self.storage.best_mcts_sim_time_step

        # Add previous bests for comparison in logs
        summary["previous_best_outcome"] = self.storage.previous_best_outcome
        summary["previous_best_game_score"] = self.storage.previous_best_game_score
        summary["previous_best_value_loss"] = self.storage.previous_best_value_loss
        summary["previous_best_policy_loss"] = self.storage.previous_best_policy_loss
        summary["previous_best_mcts_sim_time"] = (
            self.storage.previous_best_mcts_sim_time
        )

        # Calculate ETA if target step is set
        eta_seconds = None
        if self.storage.training_target_step > 0 and current_global_step > 0:
            steps_remaining = self.storage.training_target_step - current_global_step
            if (
                steps_remaining > 0 and elapsed_time > 10
            ):  # Avoid division by zero and early instability
                steps_per_second = current_global_step / elapsed_time
                if steps_per_second > 1e-3:
                    eta_seconds = steps_remaining / steps_per_second
        summary["eta_seconds"] = eta_seconds
        summary["eta_formatted"] = format_eta(eta_seconds)

        # Calculate steps per second
        steps_per_second = 0
        if elapsed_time > 1:
            steps_per_second = current_global_step / elapsed_time
        summary["steps_per_second"] = steps_per_second

        # Use latest deque values for instantaneous metrics if available
        summary["policy_loss"] = (
            self.storage.policy_losses[-1] if self.storage.policy_losses else 0.0
        )
        summary["value_loss"] = (
            self.storage.value_losses[-1] if self.storage.value_losses else 0.0
        )
        summary["mcts_sim_time"] = (
            self.storage.mcts_simulation_times[-1]
            if self.storage.mcts_simulation_times
            else 0.0
        )
        summary["mcts_nn_time"] = (
            self.storage.mcts_nn_prediction_times[-1]
            if self.storage.mcts_nn_prediction_times
            else 0.0
        )
        summary["mcts_nodes_explored"] = (
            self.storage.mcts_nodes_explored[-1]
            if self.storage.mcts_nodes_explored
            else 0
        )
        summary["mcts_avg_depth"] = (
            self.storage.mcts_avg_depths[-1] if self.storage.mcts_avg_depths else 0.0
        )

        return summary
