from typing import  Dict, Any, Optional
import numpy as np

from .aggregator_storage import AggregatorStorage

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.game_state import GameState


class AggregatorLogic:
    """Handles the calculation logic for StatsAggregator.
    Refactored for AlphaZero focus and intermediate stats. Resource usage removed."""

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
        game_state_for_best: Optional["GameState"] = None,
    ) -> Dict[str, Any]:
        """Updates storage with episode data and checks for bests."""
        update_info = {"new_best_outcome": False, "new_best_game": False}

        self.storage.episode_outcomes.append(episode_outcome)
        self.storage.episode_lengths.append(episode_length)
        if game_score is not None:
            self.storage.game_scores.append(game_score)
        if triangles_cleared is not None:
            self.storage.episode_triangles_cleared.append(triangles_cleared)
            self.storage.total_triangles_cleared += triangles_cleared
        self.storage.total_episodes = episode_num
        self.storage.current_self_play_game_number = episode_num + 1
        self.storage.current_self_play_game_steps = 0

        if episode_outcome > self.storage.best_outcome:
            self.storage.previous_best_outcome = self.storage.best_outcome
            self.storage.best_outcome = episode_outcome
            self.storage.best_outcome_step = current_step
            update_info["new_best_outcome"] = True

        if game_score is not None and game_score > self.storage.best_game_score:
            self.storage.previous_best_game_score = self.storage.best_game_score
            self.storage.best_game_score = float(game_score)
            self.storage.best_game_score_step = current_step
            update_info["new_best_game"] = True
            if game_state_for_best and hasattr(game_state_for_best, "grid"):
                try:
                    grid = game_state_for_best.grid
                    occupancy = np.array(
                        [[t.is_occupied for t in row] for row in grid.triangles],
                        dtype=bool,
                    )
                    colors = [
                        [t.color if t.color else (0, 0, 0) for t in row]
                        for row in grid.triangles
                    ]
                    death_cells = np.array(
                        [[t.is_death for t in row] for row in grid.triangles],
                        dtype=bool,
                    )
                    is_up = np.array(
                        [[t.is_up for t in row] for row in grid.triangles], dtype=bool
                    )
                    self.storage.best_game_state_data = {
                        "score": game_score,
                        "occupancy": occupancy,
                        "colors": colors,
                        "death": death_cells,
                        "is_up": is_up,
                        "rows": grid.rows,
                        "cols": grid.cols,
                        "step": current_step,
                    }
                    print(
                        f"[Aggregator] New best game state saved (Score: {game_score} at Step {current_step})"
                    )
                except Exception as e:
                    print(f"[Aggregator] Error saving best game state data: {e}")
                    self.storage.best_game_state_data = None

        current_best_game = (
            int(self.storage.best_game_score)
            if self.storage.best_game_score > -float("inf")
            else 0
        )
        self.storage.best_game_score_history.append(current_best_game)

        return update_info

    def update_step_stats(
        self, step_data: Dict[str, Any], g_step: int
    ) -> Dict[str, Any]:
        """Updates storage with step data and checks for best loss."""
        update_info = {"new_best_value_loss": False, "new_best_policy_loss": False}

        if "current_self_play_game_steps" in step_data:
            self.storage.current_self_play_game_steps = step_data[
                "current_self_play_game_steps"
            ]
        if "training_steps_performed" in step_data:
            self.storage.training_steps_performed = step_data[
                "training_steps_performed"
            ]
        if "current_self_play_game_number" in step_data: 
            self.storage.current_self_play_game_number = step_data[
                "current_self_play_game_number"
            ]

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

        # Resource usage metrics removed
        optional_metrics = [
            ("buffer_size", "buffer_sizes"),
            ("lr", "lr_values"),
        ]
        for data_key, deque_name in optional_metrics:
            if data_key in step_data and step_data[data_key] is not None:
                val = step_data[data_key]
                if np.isfinite(val):
                    if hasattr(self.storage, deque_name):
                        getattr(self.storage, deque_name).append(val)
                    else:
                        print(f"[Aggregator Warning] Deque '{deque_name}' not found.")
                else:
                    print(f"[Aggregator Warning] Received non-finite {data_key}: {val}")

        # Resource usage scalars removed
        scalar_updates = {
            "buffer_size": "current_buffer_size",
            "lr": "current_lr",
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
            "avg_outcome_window": safe_mean("episode_outcomes"),
            "avg_length_window": safe_mean("episode_lengths"),
            "policy_loss": safe_mean("policy_losses"),
            "value_loss": safe_mean("value_losses"),
            "avg_game_score_window": safe_mean("game_scores"),
            "avg_triangles_cleared_window": safe_mean("episode_triangles_cleared"),
            "avg_lr_window": safe_mean("lr_values", default=self.storage.current_lr),
            # Resource usage averages removed
            "total_episodes": self.storage.total_episodes,
            "buffer_size": self.storage.current_buffer_size,
            "global_step": current_global_step,
            "current_lr": self.storage.current_lr,
            "best_outcome": self.storage.best_outcome,
            "previous_best_outcome": self.storage.previous_best_outcome,
            "best_outcome_step": self.storage.best_outcome_step,
            "best_game_score": self.storage.best_game_score,
            "previous_best_game_score": self.storage.previous_best_game_score,
            "best_game_score_step": self.storage.best_game_score_step,
            "best_value_loss": self.storage.best_value_loss,
            "previous_best_value_loss": self.storage.previous_best_value_loss,
            "best_value_loss_step": self.storage.best_value_loss_step,
            "best_policy_loss": self.storage.best_policy_loss,
            "previous_best_policy_loss": self.storage.previous_best_policy_loss,
            "best_policy_loss_step": self.storage.best_policy_loss_step,
            "num_ep_outcomes": len(self.storage.episode_outcomes),
            "num_value_losses": len(self.storage.value_losses),
            "num_policy_losses": len(self.storage.policy_losses),
            "summary_avg_window_size": summary_avg_window,
            "start_time": self.storage.start_time,
            # Resource usage current values removed
            "current_self_play_game_number": self.storage.current_self_play_game_number,
            "current_self_play_game_steps": self.storage.current_self_play_game_steps,
            "training_steps_performed": self.storage.training_steps_performed,
        }
        return summary
