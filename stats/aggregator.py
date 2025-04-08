import time
from typing import (
    Deque,
    Dict,
    Any,
    Optional,
    List,
    TYPE_CHECKING,
)
import threading

from config import StatsConfig
from .aggregator_storage import AggregatorStorage
from .aggregator_logic import AggregatorLogic

if TYPE_CHECKING:
    from environment.game_state import GameState


class StatsAggregator:
    """
    Handles aggregation and storage of training statistics using deques.
    Calculates rolling averages and tracks best values. Does not perform logging.
    Includes locks for thread safety. Delegates storage and logic to helper classes.
    Refactored for clarity and AlphaZero focus.
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
        episode_outcome: float,  # Renamed from episode_score for clarity (-1, 0, 1)
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional["GameState"] = None,
    ) -> Dict[str, Any]:
        """Records episode stats and checks for new bests."""
        with self._lock:
            current_step = (
                global_step
                if global_step is not None
                else self.storage.current_global_step
            )
            update_info = self.logic.update_episode_stats(
                episode_outcome,
                episode_length,
                episode_num,
                current_step,
                game_score,
                triangles_cleared,
                game_state_for_best,
            )
            return update_info

    def record_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Records step data (e.g., NN training step) and checks for new bests."""
        with self._lock:
            g_step = step_data.get("global_step")
            if g_step is not None and g_step > self.storage.current_global_step:
                self.storage.current_global_step = g_step
            elif g_step is None:
                g_step = self.storage.current_global_step

            update_info = self.logic.update_step_stats(step_data, g_step)
            return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        """Calculates and returns the summary dictionary."""
        with self._lock:
            if current_global_step is None:
                current_global_step = self.storage.current_global_step
            summary = self.logic.calculate_summary(
                current_global_step, self.summary_avg_window
            )
            return summary

    def get_plot_data(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        with self._lock:
            return self.storage.get_all_plot_deques()

    def get_best_game_state_data(self) -> Optional[Dict[str, Any]]:
        """Returns the data needed to render the best game state found."""
        with self._lock:
            # Return a copy to prevent modification outside the lock
            return (
                self.storage.best_game_state_data.copy()
                if self.storage.best_game_state_data
                else None
            )

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state for checkpointing."""
        with self._lock:
            state = self.storage.state_dict()
            state["plot_window"] = self.plot_window
            state["avg_windows"] = self.avg_windows
            return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the state from a checkpoint."""
        with self._lock:
            print("[StatsAggregator] Loading state...")
            self.plot_window = state_dict.get("plot_window", self.plot_window)
            self.avg_windows = state_dict.get("avg_windows", self.avg_windows)
            self.summary_avg_window = self.avg_windows[0] if self.avg_windows else 100

            self.storage.load_state_dict(state_dict, self.plot_window)

            print("[StatsAggregator] State loaded.")
            print(f"  -> Loaded total_episodes: {self.storage.total_episodes}")
            print(f"  -> Loaded best_outcome: {self.storage.best_outcome}")
            print(f"  -> Loaded best_game_score: {self.storage.best_game_score}")
            print(
                f"  -> Loaded start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.storage.start_time))}"
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
# File: stats/aggregator.py
import time
from typing import (
    Deque,
    Dict,
    Any,
    Optional,
    List,
    TYPE_CHECKING,
)
import threading

from config import StatsConfig
from .aggregator_storage import AggregatorStorage
from .aggregator_logic import AggregatorLogic

if TYPE_CHECKING:
    from environment.game_state import GameState


class StatsAggregator:
    """
    Handles aggregation and storage of training statistics using deques.
    Calculates rolling averages and tracks best values. Does not perform logging.
    Includes locks for thread safety. Delegates storage and logic to helper classes.
    Refactored for clarity and AlphaZero focus.
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
        episode_outcome: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional["GameState"] = None,
    ) -> Dict[str, Any]:
        """Records episode stats and checks for new bests."""
        with self._lock:
            current_step = (
                global_step
                if global_step is not None
                else self.storage.current_global_step
            )
            update_info = self.logic.update_episode_stats(
                episode_outcome,
                episode_length,
                episode_num,
                current_step,
                game_score,
                triangles_cleared,
                game_state_for_best,
            )
            return update_info

    def record_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Records step data (e.g., NN training step, MCTS step) and checks for new bests.
        Handles both training worker and self-play worker data.
        """
        with self._lock:
            g_step = step_data.get("global_step")
            if g_step is not None and g_step > self.storage.current_global_step:
                self.storage.current_global_step = g_step
            elif g_step is None:
                # Use current step if not provided (e.g., for MCTS stats from self-play)
                g_step = self.storage.current_global_step

            update_info = self.logic.update_step_stats(step_data, g_step)
            return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        """Calculates and returns the summary dictionary."""
        with self._lock:
            if current_global_step is None:
                current_global_step = self.storage.current_global_step
            summary = self.logic.calculate_summary(
                current_global_step, self.summary_avg_window
            )
            return summary

    def get_plot_data(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        with self._lock:
            return self.storage.get_all_plot_deques()

    def get_best_game_state_data(self) -> Optional[Dict[str, Any]]:
        """Returns the data needed to render the best game state found."""
        with self._lock:
            return (
                self.storage.best_game_state_data.copy()
                if self.storage.best_game_state_data
                else None
            )

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state for checkpointing."""
        with self._lock:
            state = self.storage.state_dict()
            state["plot_window"] = self.plot_window
            state["avg_windows"] = self.avg_windows
            return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the state from a checkpoint."""
        with self._lock:
            print("[StatsAggregator] Loading state...")
            self.plot_window = state_dict.get("plot_window", self.plot_window)
            self.avg_windows = state_dict.get("avg_windows", self.avg_windows)
            self.summary_avg_window = self.avg_windows[0] if self.avg_windows else 100

            self.storage.load_state_dict(state_dict, self.plot_window)

            print("[StatsAggregator] State loaded.")
            print(f"  -> Loaded total_episodes: {self.storage.total_episodes}")
            # print(f"  -> Loaded best_outcome: {self.storage.best_outcome}") # Less relevant now
            print(f"  -> Loaded best_game_score: {self.storage.best_game_score}")
            print(
                f"  -> Loaded start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.storage.start_time))}"
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
