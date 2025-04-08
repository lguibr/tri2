# File: stats/aggregator.py
# File: stats/aggregator.py
import time
from collections import deque
from typing import (
    Deque,
    Dict,
    Any,
    Optional,
    List,
    TYPE_CHECKING,
)  # Added TYPE_CHECKING
import numpy as np
import threading

from config import StatsConfig
from .aggregator_storage import AggregatorStorage
from .aggregator_logic import AggregatorLogic

if TYPE_CHECKING:
    from environment.game_state import GameState  # Import for type hinting


class StatsAggregator:
    """
    Handles aggregation and storage of training statistics using deques.
    Calculates rolling averages and tracks best values. Does not perform logging.
    Includes locks for thread safety. Delegates storage and logic to helper classes.
    """

    def __init__(
        self,
        avg_windows: List[int] = StatsConfig.STATS_AVG_WINDOW,
        plot_window: int = StatsConfig.PLOT_DATA_WINDOW,
    ):
        if not avg_windows or not all(
            isinstance(w, int) and w > 0 for w in avg_windows
        ):
            print("Warning: Invalid avg_windows list. Using default [100].")
            self.avg_windows = [100]
        else:
            self.avg_windows = sorted(list(set(avg_windows)))

        if plot_window <= 0:
            plot_window = 10000
        self.plot_window = plot_window
        self.summary_avg_window = self.avg_windows[0]

        self._lock = threading.Lock()
        self.storage = AggregatorStorage(plot_window)
        self.logic = AggregatorLogic(self.storage)

        print(
            f"[StatsAggregator] Initialized. Avg Windows: {self.avg_windows}, Plot Window: {self.plot_window}"
        )

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional["GameState"] = None,  # Added optional GameState
    ) -> Dict[str, Any]:
        with self._lock:
            # Use the passed global_step or the current stored one
            current_step = (
                global_step
                if global_step is not None
                else self.storage.current_global_step
            )
            update_info = self.logic.update_episode_stats(
                episode_score,
                episode_length,
                episode_num,
                current_step,
                game_score,
                triangles_cleared,
                game_state_for_best,  # Pass GameState to logic
            )
            return update_info

    def record_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Records step data, now likely related to NN training steps."""
        with self._lock:
            # Ensure global_step is updated within the storage based on step_data
            g_step = step_data.get("global_step")
            if g_step is not None and g_step > self.storage.current_global_step:
                self.storage.current_global_step = g_step
            elif g_step is None:
                # If no global step provided, maybe increment internally?
                # Or rely on the caller (TrainingWorker) to always provide it.
                # For now, use the stored value if not provided.
                g_step = self.storage.current_global_step

            update_info = self.logic.update_step_stats(
                step_data, g_step
            )  # Pass g_step to logic
            return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            if current_global_step is None:
                current_global_step = self.storage.current_global_step
            summary = self.logic.calculate_summary(
                current_global_step, self.summary_avg_window
            )
            return summary

    def get_plot_data(self) -> Dict[str, Deque]:
        with self._lock:
            return self.storage.get_all_plot_deques()

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            state = self.storage.state_dict()
            state["plot_window"] = self.plot_window
            state["avg_windows"] = self.avg_windows
            return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        with self._lock:
            print("[StatsAggregator] Loading state...")
            self.plot_window = state_dict.get("plot_window", self.plot_window)
            self.avg_windows = state_dict.get("avg_windows", self.avg_windows)
            self.summary_avg_window = self.avg_windows[0] if self.avg_windows else 100

            self.storage.load_state_dict(state_dict, self.plot_window)

            print("[StatsAggregator] State loaded.")
            print(f"  -> Loaded total_episodes: {self.storage.total_episodes}")
            print(f"  -> Loaded best_score: {self.storage.best_score}")
            print(
                f"  -> Loaded start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.storage.start_time))}"
            )
            print(
                f"  -> Loaded training_target_step: {self.storage.training_target_step}"
            )
            print(
                f"  -> Loaded current_global_step: {self.storage.current_global_step}"
            )
            if self.storage.best_game_state_data:
                print(
                    f"  -> Loaded best_game_state_data (Score: {self.storage.best_game_state_data.get('score', 'N/A')})"
                )
            else:
                print("  -> No best_game_state_data found in loaded state.")
