# File: stats/simple_stats_recorder.py
# File: stats/simple_stats_recorder.py
import time
from collections import deque
from typing import (
    Deque,
    Dict,
    Any,
    Optional,
    Union,
    List,
    TYPE_CHECKING,
)  # Added TYPE_CHECKING
import numpy as np
import torch
import threading

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from config import StatsConfig

if TYPE_CHECKING:
    from environment.game_state import GameState  # Import for type hinting


class SimpleStatsRecorder(StatsRecorderBase):
    """
    Logs aggregated statistics to the console periodically. Thread-safe.
    Delegates data storage and aggregation to a StatsAggregator instance.
    Provides no-op implementations for histogram, image, hparam, graph logging.
    """

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_log_interval: int = StatsConfig.CONSOLE_LOG_FREQ,
    ):
        self.aggregator = aggregator
        self.console_log_interval = (
            max(1, console_log_interval) if console_log_interval > 0 else -1
        )
        self.last_log_time: float = time.time()
        self.start_time: float = time.time()
        self.summary_avg_window = self.aggregator.summary_avg_window
        self.updates_since_last_log = 0

        self._lock = threading.Lock()

        print(
            f"[SimpleStatsRecorder] Initialized. Console Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'} updates/episodes"
        )
        print(
            f"[SimpleStatsRecorder] Console logs will use Avg Window: {self.summary_avg_window}"
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
    ):
        """Records episode stats and prints new bests to console. Thread-safe."""
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
            game_state_for_best,  # Pass GameState down
        )
        current_step = (
            global_step
            if global_step is not None
            else self.aggregator.storage.current_global_step
        )
        step_info = f"at Step ~{current_step/1e6:.1f}M"

        # Print new bests immediately
        if update_info.get("new_best_game"):
            prev_str = (
                f"{self.aggregator.storage.previous_best_game_score:.0f}"
                if self.aggregator.storage.previous_best_game_score > -float("inf")
                else "N/A"
            )
            print(
                f"--- 🎮 New Best Game: {self.aggregator.storage.best_game_score:.0f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_value_loss"):
            prev_str = (
                f"{self.aggregator.storage.previous_best_value_loss:.4f}"
                if self.aggregator.storage.previous_best_value_loss < float("inf")
                else "N/A"
            )
            print(
                f"---📉 New Best V.Loss: {self.aggregator.storage.best_value_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_policy_loss"):
            prev_str = (
                f"{self.aggregator.storage.previous_best_policy_loss:.4f}"
                if self.aggregator.storage.previous_best_policy_loss < float("inf")
                else "N/A"
            )
            print(
                f"---📉 New Best P.Loss: {self.aggregator.storage.best_policy_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )

        log_now = False
        with self._lock:
            self.updates_since_last_log += 1
            if (
                self.console_log_interval > 0
                and self.updates_since_last_log >= self.console_log_interval
            ):
                log_now = True
                self.updates_since_last_log = 0

        if log_now:
            self.log_summary(current_step)

    def record_step(self, step_data: Dict[str, Any]):
        """Records step stats (e.g., NN update) and triggers console logging if interval met. Thread-safe."""
        update_info = self.aggregator.record_step(step_data)
        g_step = step_data.get(
            "global_step", self.aggregator.storage.current_global_step
        )

        if update_info.get("new_best_value_loss"):
            prev_str = (
                f"{self.aggregator.storage.previous_best_value_loss:.4f}"
                if self.aggregator.storage.previous_best_value_loss < float("inf")
                else "N/A"
            )
            step_info = f"at Step ~{g_step/1e6:.1f}M"
            print(
                f"---📉 New Best V.Loss: {self.aggregator.storage.best_value_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_policy_loss"):
            prev_str = (
                f"{self.aggregator.storage.previous_best_policy_loss:.4f}"
                if self.aggregator.storage.previous_best_policy_loss < float("inf")
                else "N/A"
            )
            step_info = f"at Step ~{g_step/1e6:.1f}M"
            print(
                f"---📉 New Best P.Loss: {self.aggregator.storage.best_policy_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )

        log_now = False
        with self._lock:
            if "policy_loss" in step_data or "value_loss" in step_data:
                self.updates_since_last_log += 1
                if (
                    self.console_log_interval > 0
                    and self.updates_since_last_log >= self.console_log_interval
                ):
                    log_now = True
                    self.updates_since_last_log = 0

        if log_now:
            self.log_summary(g_step)

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Gets the summary dictionary from the aggregator (thread-safe)."""
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        """Gets the plot data deques from the aggregator (thread-safe)."""
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        """Logs the current summary statistics to the console."""
        summary = self.get_summary(global_step)
        elapsed_runtime = time.time() - self.aggregator.storage.start_time
        runtime_hrs = elapsed_runtime / 3600

        best_game_score_val = (
            f"{summary['best_game_score']:.0f}"
            if summary["best_game_score"] > -float("inf")
            else "N/A"
        )
        best_v_loss_val = (
            f"{summary['best_value_loss']:.4f}"
            if summary["best_value_loss"] < float("inf")
            else "N/A"
        )
        best_p_loss_val = (
            f"{summary['best_policy_loss']:.4f}"
            if summary["best_policy_loss"] < float("inf")
            else "N/A"
        )
        avg_window_size = summary.get("summary_avg_window_size", "?")

        log_str = (
            f"[{runtime_hrs:.1f}h|Console] Step: {global_step/1e6:<6.2f}M | "
            f"Ep: {summary['total_episodes']:<7} | "
            f"GameScore(Avg{avg_window_size}): {summary['avg_game_score_window']:<6.0f} (Best: {best_game_score_val}) | "
            f"V.Loss(Avg{avg_window_size}): {summary['value_loss']:.4f} (Best: {best_v_loss_val}) | "
            f"P.Loss(Avg{avg_window_size}): {summary['policy_loss']:.4f} (Best: {best_p_loss_val}) | "
            f"LR: {summary['current_lr']:.1e}"
        )
        avg_tris_cleared = summary.get("avg_triangles_cleared_window", 0.0)
        log_str += f" | TrisClr(Avg{avg_window_size}): {avg_tris_cleared:.1f}"

        print(log_str)

        self.last_log_time = time.time()

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        pass

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        pass

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        pass

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[Any] = None
    ):
        pass

    def close(self, is_cleanup: bool = False):
        """Closes the recorder (no action needed for simple console logger)."""
        print(f"[SimpleStatsRecorder] Closed (is_cleanup={is_cleanup}).")
