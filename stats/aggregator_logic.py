# File: stats/aggregator_logic.py
import time
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
import logging
import copy

from .aggregator_storage import AggregatorStorage
from utils.helpers import format_eta

if TYPE_CHECKING:
    from environment.game_state import GameState, StateType  # Added StateType

logger = logging.getLogger(__name__)
from utils.types import StateType, ActionType, AgentStateDict


class AggregatorLogic:
    """Handles the logic for updating stats and calculating summaries."""

    def __init__(self, storage: AggregatorStorage):
        self.storage = storage

    def _calculate_average(self, deque_name: str, window: int) -> float:
        """Calculates the average of the last 'window' items in a deque."""
        dq = self.storage.get_deque(deque_name)
        if not dq:
            return 0.0
        items = list(dq)[-window:]
        if not items:
            return 0.0
        try:
            # Filter out potential None or non-numeric values if necessary
            numeric_items = [
                x for x in items if isinstance(x, (int, float)) and np.isfinite(x)
            ]
            if not numeric_items:
                return 0.0
            return float(np.mean(numeric_items))
        except (TypeError, ValueError) as e:
            logger.warning(f"Error calculating average for {deque_name}: {e}")
            return 0.0

    def update_episode_stats(
        self,
        episode_outcome: float,
        episode_length: int,
        episode_num: int,
        global_step: int,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional[StateType] = None,  # Accept StateType dict
    ) -> Dict[str, Any]:
        """Updates episode-related deques and best values."""
        update_info = {"new_best_game": False}

        self.storage.episode_outcomes.append(episode_outcome)
        self.storage.episode_lengths.append(episode_length)
        self.storage.total_episodes = episode_num  # Use the number passed from worker

        if game_score is not None:
            self.storage.game_scores.append(game_score)
            if game_score > self.storage.best_game_score:
                self.storage.previous_best_game_score = self.storage.best_game_score
                self.storage.best_game_score = float(game_score)
                self.storage.best_game_score_step = global_step
                update_info["new_best_game"] = True
                # Store data needed for rendering the best game state
                if game_state_for_best:
                    self.storage.best_game_state_data = {
                        "score": game_score,
                        "step": global_step,
                        "game_state_dict": copy.deepcopy(
                            game_state_for_best
                        ),  # Store the dict
                    }
                    logger.info(
                        f"Stored new best game state data (Score: {game_score})"
                    )

            # Update history regardless of whether it's the absolute best
            self.storage.best_game_score_history.append(
                int(self.storage.best_game_score)
            )

        if triangles_cleared is not None:
            self.storage.episode_triangles_cleared.append(triangles_cleared)
            self.storage.total_triangles_cleared += triangles_cleared

        return update_info

    def update_step_stats(
        self,
        step_data: Dict[str, Any],
        global_step: int,
        mcts_sim_time: Optional[float] = None,
        mcts_nn_time: Optional[float] = None,
        mcts_nodes_explored: Optional[int] = None,
        mcts_avg_depth: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Updates step-related deques and best values."""
        update_info = {
            "new_best_value_loss": False,
            "new_best_policy_loss": False,
            "new_best_mcts_sim_time": False,
        }

        # Update global step if provided and greater
        if global_step > self.storage.current_global_step:
            self.storage.current_global_step = global_step

        # Calculate and update steps per second
        self.storage.update_steps_per_second(global_step)

        # Training Worker Stats
        policy_loss = step_data.get("policy_loss")
        value_loss = step_data.get("value_loss")
        lr = step_data.get("lr")

        if policy_loss is not None:
            self.storage.policy_losses.append(policy_loss)
            if policy_loss < self.storage.best_policy_loss:
                self.storage.previous_best_policy_loss = self.storage.best_policy_loss
                self.storage.best_policy_loss = policy_loss
                self.storage.best_policy_loss_step = global_step
                update_info["new_best_policy_loss"] = True

        if value_loss is not None:
            self.storage.value_losses.append(value_loss)
            if value_loss < self.storage.best_value_loss:
                self.storage.previous_best_value_loss = self.storage.best_value_loss
                self.storage.best_value_loss = value_loss
                self.storage.best_value_loss_step = global_step
                update_info["new_best_value_loss"] = True

        if lr is not None:
            self.storage.lr_values.append(lr)
            self.storage.current_lr = lr

        # Self-Play Worker Stats (MCTS)
        if mcts_sim_time is not None:
            self.storage.mcts_simulation_times.append(mcts_sim_time)
            if mcts_sim_time < self.storage.best_mcts_sim_time:
                self.storage.previous_best_mcts_sim_time = (
                    self.storage.best_mcts_sim_time
                )
                self.storage.best_mcts_sim_time = mcts_sim_time
                self.storage.best_mcts_sim_time_step = global_step
                update_info["new_best_mcts_sim_time"] = True

        if mcts_nn_time is not None:
            self.storage.mcts_nn_prediction_times.append(mcts_nn_time)
        if mcts_nodes_explored is not None:
            self.storage.mcts_nodes_explored.append(mcts_nodes_explored)
        if mcts_avg_depth is not None:
            self.storage.mcts_avg_depths.append(mcts_avg_depth)

        # System Stats
        buffer_size = step_data.get("buffer_size")
        if buffer_size is not None:
            self.storage.buffer_sizes.append(buffer_size)
            self.storage.current_buffer_size = buffer_size

        # Intermediate Progress Stats
        current_game = step_data.get("current_self_play_game_number")
        current_game_step = step_data.get("current_self_play_game_steps")
        training_steps = step_data.get("training_steps_performed")

        if current_game is not None:
            self.storage.current_self_play_game_number = current_game
        if current_game_step is not None:
            self.storage.current_self_play_game_steps = current_game_step
        if training_steps is not None:
            self.storage.training_steps_performed = training_steps

        return update_info

    def calculate_summary(
        self, current_global_step: int, avg_window: int
    ) -> Dict[str, Any]:
        """Calculates the summary dictionary."""
        summary = {}
        summary["global_step"] = current_global_step
        summary["total_episodes"] = self.storage.total_episodes
        summary["summary_avg_window_size"] = avg_window

        # Calculate averages using the helper
        summary["avg_game_score_window"] = self._calculate_average(
            "game_scores", avg_window
        )
        summary["avg_episode_length_window"] = self._calculate_average(
            "episode_lengths", avg_window
        )
        summary["avg_triangles_cleared_window"] = self._calculate_average(
            "episode_triangles_cleared", avg_window
        )
        summary["policy_loss"] = self._calculate_average("policy_losses", avg_window)
        summary["value_loss"] = self._calculate_average("value_losses", avg_window)
        summary["mcts_simulation_time_avg"] = self._calculate_average(
            "mcts_simulation_times", avg_window
        )
        summary["mcts_nn_prediction_time_avg"] = self._calculate_average(
            "mcts_nn_prediction_times", avg_window
        )
        summary["mcts_nodes_explored_avg"] = self._calculate_average(
            "mcts_nodes_explored", avg_window
        )
        summary["mcts_avg_depth_avg"] = self._calculate_average(
            "mcts_avg_depths", avg_window
        )
        summary["steps_per_second_avg"] = self._calculate_average(
            "steps_per_second", avg_window
        )  # Average SPS

        # Current values
        summary["buffer_size"] = self.storage.current_buffer_size
        summary["current_lr"] = self.storage.current_lr
        summary["start_time"] = self.storage.start_time

        # Best values
        summary["best_game_score"] = self.storage.best_game_score
        summary["best_game_score_step"] = self.storage.best_game_score_step
        summary["best_value_loss"] = self.storage.best_value_loss
        summary["best_value_loss_step"] = self.storage.best_value_loss_step
        summary["best_policy_loss"] = self.storage.best_policy_loss
        summary["best_policy_loss_step"] = self.storage.best_policy_loss_step
        summary["best_mcts_sim_time"] = self.storage.best_mcts_sim_time
        summary["best_mcts_sim_time_step"] = self.storage.best_mcts_sim_time_step

        # Intermediate progress
        summary["current_self_play_game_number"] = (
            self.storage.current_self_play_game_number
        )
        summary["current_self_play_game_steps"] = (
            self.storage.current_self_play_game_steps
        )
        summary["training_steps_performed"] = self.storage.training_steps_performed

        # ETA Calculation
        eta_seconds = None
        if self.storage.training_target_step > 0:
            steps_remaining = self.storage.training_target_step - current_global_step
            avg_sps = summary["steps_per_second_avg"]
            if (
                steps_remaining > 0 and avg_sps > 0.1
            ):  # Only calculate if SPS is meaningful
                eta_seconds = steps_remaining / avg_sps
        summary["eta_seconds"] = eta_seconds
        summary["eta_formatted"] = format_eta(eta_seconds)

        return summary
