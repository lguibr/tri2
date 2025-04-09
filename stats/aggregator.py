import time
from typing import Deque, Dict, Any, Optional, List, TYPE_CHECKING
import threading
import logging
import numpy as np
import ray

from config import StatsConfig
from .aggregator_storage import AggregatorStorage
from .aggregator_logic import AggregatorLogic

if TYPE_CHECKING:
    from environment.game_state import GameState

logger = logging.getLogger(__name__)


# --- Ray Actor Version ---
@ray.remote
class StatsAggregatorActor:
    """
    Ray Actor version of StatsAggregator. Handles aggregation and storage
    of training statistics using deques within the actor process.
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
        self.plot_window = max(1, plot_window)
        self.summary_avg_window = self.avg_windows[0]
        self.storage = AggregatorStorage(self.plot_window)
        self.logic = AggregatorLogic(self.storage)
        logger.info(
            f"[StatsAggregatorActor] Initialized. Avg Windows: {self.avg_windows}, Plot Window: {self.plot_window}"
        )

    def record_episode(
        self,
        episode_outcome: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Records episode stats and checks for new bests."""
        current_step = (
            global_step if global_step is not None else self.storage.current_global_step
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
        """Records step stats."""
        g_step = step_data.get("global_step")
        if g_step is not None and g_step > self.storage.current_global_step:
            self.storage.current_global_step = g_step
        elif g_step is None:
            g_step = self.storage.current_global_step

        update_info = self.logic.update_step_stats(
            step_data,
            g_step,
            mcts_sim_time=step_data.get("mcts_sim_time"),
            mcts_nn_time=step_data.get("mcts_nn_time"),
            mcts_nodes_explored=step_data.get("mcts_nodes_explored"),
            mcts_avg_depth=step_data.get("mcts_avg_depth"),
        )
        return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        """Calculates and returns the summary dictionary."""
        if current_global_step is None:
            current_global_step = self.storage.current_global_step
        summary = self.logic.calculate_summary(
            current_global_step, self.summary_avg_window
        )
        summary["device"] = "Actor"
        try:
            from config.core import TrainConfig

            summary["min_buffer_size"] = TrainConfig.MIN_BUFFER_SIZE_TO_TRAIN
        except ImportError:
            summary["min_buffer_size"] = 0
        return summary

    def get_plot_data(self) -> Dict[str, List]:  # Return List for serialization
        """Returns copies of data deques as lists for plotting."""
        plot_deques = self.storage.get_all_plot_deques()
        return {name: list(dq) for name, dq in plot_deques.items()}

    def get_best_game_state_data(self) -> Optional[Dict[str, Any]]:
        """Returns the serializable data needed to render the best game state found."""
        return self.storage.best_game_state_data

    def state_dict(self) -> Dict[str, Any]:
        """Returns the internal state for saving."""
        state = self.storage.state_dict()
        state["plot_window"] = self.plot_window
        state["avg_windows"] = self.avg_windows
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the internal state from a dictionary."""
        logger.info("[StatsAggregatorActor] Loading state...")
        self.plot_window = state_dict.get("plot_window", self.plot_window)
        self.avg_windows = state_dict.get("avg_windows", self.avg_windows)
        self.summary_avg_window = self.avg_windows[0] if self.avg_windows else 100
        self.storage.load_state_dict(state_dict, self.plot_window)
        logger.info("[StatsAggregatorActor] State loaded.")
        logger.info(f"  -> Loaded total_episodes: {self.storage.total_episodes}")
        logger.info(f"  -> Loaded best_game_score: {self.storage.best_game_score}")
        logger.info(
            f"  -> Loaded start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.storage.start_time))}"
        )
        logger.info(
            f"  -> Loaded current_global_step: {self.storage.current_global_step}"
        )
        if self.storage.best_game_state_data:
            logger.info(
                f"  -> Loaded best_game_state_data (Score: {self.storage.best_game_state_data.get('score', 'N/A')})"
            )
        else:
            logger.info("  -> No best_game_state_data found in loaded state.")

    def get_total_episodes(self) -> int:
        """Returns the total number of episodes recorded."""
        return self.storage.total_episodes

    def get_current_global_step(self) -> int:
        """Returns the current global step."""
        return self.storage.current_global_step

    def set_training_target_step(self, target_step: int):
        """Sets the training target step."""
        self.storage.training_target_step = target_step

    def get_training_target_step(self) -> int:
        """Returns the training target step."""
        return self.storage.training_target_step

    def health_check(self):
        """Basic health check method for Ray."""
        return "OK"
