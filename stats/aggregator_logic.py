# File: stats/aggregator_logic.py
import time
import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging

from .aggregator_storage import AggregatorStorage

if TYPE_CHECKING:
    from environment.game_state import GameState

logger = logging.getLogger(__name__)


class AggregatorLogic:
    """Handles the logic for updating stats and calculating summaries."""

    def __init__(self, storage: AggregatorStorage):
        self.storage = storage

    def update_episode_stats(
        self,
        episode_outcome: float,
        episode_length: int,
        episode_num: int,
        current_step: int,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional[
            Dict[str, np.ndarray]
        ] = None,  # Expect state dict
    ) -> Dict[str, Any]:
        """Updates deques and best values related to episode completion."""
        update_info = {"new_best_game": False}
        self.storage.total_episodes += 1
        self.storage.episode_outcomes.append(episode_outcome)
        self.storage.episode_lengths.append(episode_length)

        if game_score is not None:
            self.storage.game_scores.append(game_score)
            if game_score > self.storage.best_game_score:
                self.storage.previous_best_game_score = self.storage.best_game_score
                self.storage.best_game_score = game_score
                self.storage.best_game_score_step = current_step
                self.storage.best_game_score_history.append(game_score)
                update_info["new_best_game"] = True
                if game_state_for_best and isinstance(game_state_for_best, dict):
                    self.storage.best_game_state_data = {
                        "score": game_score,
                        "step": current_step,
                        "game_state_dict": game_state_for_best,
                    }
                elif game_state_for_best:
                    logger.warning(
                        f"Received game_state_for_best of type {type(game_state_for_best)}, expected dict. Skipping storage."
                    )
                    self.storage.best_game_state_data = None
                else:
                    self.storage.best_game_state_data = None

            elif self.storage.best_game_score > -float("inf"):
                self.storage.best_game_score_history.append(
                    self.storage.best_game_score
                )

        if triangles_cleared is not None:
            self.storage.episode_triangles_cleared.append(triangles_cleared)
            self.storage.total_triangles_cleared += triangles_cleared

        if episode_outcome > self.storage.best_outcome:
            self.storage.previous_best_outcome = self.storage.best_outcome
            self.storage.best_outcome = episode_outcome
            self.storage.best_outcome_step = current_step
            update_info["new_best_outcome"] = True

        return update_info

    def update_step_stats(
        self,
        step_data: Dict[str, Any],
        g_step: int,
        mcts_sim_time: Optional[float] = None,
        mcts_nn_time: Optional[float] = None,
        mcts_nodes_explored: Optional[int] = None,
        mcts_avg_depth: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Updates deques and best values related to individual steps (training or MCTS)."""
        update_info = {
            "new_best_value_loss": False,
            "new_best_policy_loss": False,
            "new_best_mcts_sim_time": False,
        }

        # --- Log Global Step Update ---
        # Use INFO level as requested
        # logger.info(f"[AggLogic] update_step_stats called. Incoming g_step: {g_step}, Stored step before update: {self.storage.current_global_step}")
        if g_step > self.storage.current_global_step:
            self.storage.current_global_step = g_step
            # logger.info(f"[AggLogic] Updated storage.current_global_step to: {self.storage.current_global_step}")
        # --- End Log ---

        # Training Stats
        policy_loss = step_data.get("policy_loss")
        value_loss = step_data.get("value_loss")
        lr = step_data.get("lr")

        if policy_loss is not None and np.isfinite(policy_loss):
            self.storage.policy_losses.append(policy_loss)
            if policy_loss < self.storage.best_policy_loss:
                self.storage.previous_best_policy_loss = self.storage.best_policy_loss
                self.storage.best_policy_loss = policy_loss
                self.storage.best_policy_loss_step = g_step
                update_info["new_best_policy_loss"] = True

        if value_loss is not None and np.isfinite(value_loss):
            self.storage.value_losses.append(value_loss)
            if value_loss < self.storage.best_value_loss:
                self.storage.previous_best_value_loss = self.storage.best_value_loss
                self.storage.best_value_loss = value_loss
                self.storage.best_value_loss_step = g_step
                update_info["new_best_value_loss"] = True

        if lr is not None and np.isfinite(lr):
            self.storage.lr_values.append(lr)
            self.storage.current_lr = lr

        # MCTS Stats
        if mcts_sim_time is not None and np.isfinite(mcts_sim_time):
            self.storage.mcts_simulation_times.append(mcts_sim_time)
            if mcts_sim_time < self.storage.best_mcts_sim_time:
                self.storage.previous_best_mcts_sim_time = (
                    self.storage.best_mcts_sim_time
                )
                self.storage.best_mcts_sim_time = mcts_sim_time
                self.storage.best_mcts_sim_time_step = g_step
                update_info["new_best_mcts_sim_time"] = True

        if mcts_nn_time is not None and np.isfinite(mcts_nn_time):
            self.storage.mcts_nn_prediction_times.append(mcts_nn_time)
        if mcts_nodes_explored is not None and np.isfinite(mcts_nodes_explored):
            self.storage.mcts_nodes_explored.append(mcts_nodes_explored)
            self.storage.total_mcts_nodes_explored += (
                mcts_nodes_explored  # Accumulate total nodes
            )
        if mcts_avg_depth is not None and np.isfinite(mcts_avg_depth):
            self.storage.mcts_avg_depths.append(mcts_avg_depth)

        # System Stats
        buffer_size = step_data.get("buffer_size")
        if buffer_size is not None:
            self.storage.buffer_sizes.append(buffer_size)
            self.storage.current_buffer_size = buffer_size

        # Update Rates
        self.storage.update_steps_per_second(g_step)
        self.storage.update_nodes_per_second()  # Call new method

        # Intermediate Progress
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

    def _calculate_average(self, deque, window_size):
        """Calculates the average of the last 'window_size' elements in a deque."""
        if not deque:
            return 0.0
        actual_window = min(window_size, len(deque))
        items = list(deque)[-actual_window:]
        return np.mean(items) if items else 0.0

    def calculate_summary(
        self, current_global_step: int, avg_window: int
    ) -> Dict[str, Any]:
        """Calculates the summary dictionary based on stored data."""
        summary = {}
        summary["global_step"] = current_global_step
        summary["total_episodes"] = self.storage.total_episodes
        summary["start_time"] = self.storage.start_time
        summary["runtime_seconds"] = time.time() - self.storage.start_time
        summary["buffer_size"] = self.storage.current_buffer_size
        summary["current_lr"] = self.storage.current_lr
        summary["summary_avg_window_size"] = avg_window

        summary["avg_episode_length"] = self._calculate_average(
            self.storage.episode_lengths, avg_window
        )
        summary["avg_game_score_window"] = self._calculate_average(
            self.storage.game_scores, avg_window
        )
        summary["avg_triangles_cleared"] = self._calculate_average(
            self.storage.episode_triangles_cleared, avg_window
        )
        summary["policy_loss"] = self._calculate_average(
            self.storage.policy_losses, avg_window
        )
        summary["value_loss"] = self._calculate_average(
            self.storage.value_losses, avg_window
        )
        summary["mcts_simulation_time_avg"] = self._calculate_average(
            self.storage.mcts_simulation_times, avg_window
        )
        summary["mcts_nn_prediction_time_avg"] = self._calculate_average(
            self.storage.mcts_nn_prediction_times, avg_window
        )
        summary["mcts_nodes_explored_avg"] = self._calculate_average(
            self.storage.mcts_nodes_explored, avg_window
        )
        summary["steps_per_second_avg"] = self._calculate_average(
            self.storage.steps_per_second, avg_window
        )
        summary["nodes_per_second_avg"] = (
            self._calculate_average(  # Added nodes/sec avg
                self.storage.nodes_per_second, avg_window
            )
        )

        summary["best_game_score"] = self.storage.best_game_score
        summary["best_game_score_step"] = self.storage.best_game_score_step
        summary["best_value_loss"] = self.storage.best_value_loss
        summary["best_value_loss_step"] = self.storage.best_value_loss_step
        summary["best_policy_loss"] = self.storage.best_policy_loss
        summary["best_policy_loss_step"] = self.storage.best_policy_loss_step
        summary["best_mcts_sim_time"] = self.storage.best_mcts_sim_time
        summary["best_mcts_sim_time_step"] = self.storage.best_mcts_sim_time_step

        summary["current_self_play_game_number"] = (
            self.storage.current_self_play_game_number
        )
        summary["current_self_play_game_steps"] = (
            self.storage.current_self_play_game_steps
        )
        summary["training_steps_performed"] = self.storage.training_steps_performed
        summary["training_target_step"] = self.storage.training_target_step

        return summary
