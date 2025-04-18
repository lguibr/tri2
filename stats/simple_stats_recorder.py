# File: stats/simple_stats_recorder.py
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List
import numpy as np
import torch
from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from config import StatsConfig


class SimpleStatsRecorder(StatsRecorderBase):
    """
    Logs aggregated statistics to the console periodically.
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
        self.last_log_step: int = 0
        self.start_time: float = time.time()
        # --- MODIFIED: Get the window size used for summary ---
        self.summary_avg_window = self.aggregator.summary_avg_window
        # --- END MODIFIED ---
        print(
            f"[SimpleStatsRecorder] Initialized. Console Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'}"
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
        lines_cleared: Optional[int] = None,
    ):
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            lines_cleared,
        )
        current_step = (
            global_step
            if global_step is not None
            else self.aggregator.current_global_step
        )
        step_info = f"at Step ~{current_step/1e6:.1f}M"

        if update_info.get("new_best_rl"):
            prev_str = (
                f"{self.aggregator.previous_best_score:.2f}"
                if self.aggregator.previous_best_score > -float("inf")
                else "N/A"
            )
            print(
                f"\n--- 🏆 New Best RL: {self.aggregator.best_score:.2f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_game"):
            prev_str = (
                f"{self.aggregator.previous_best_game_score:.0f}"
                if self.aggregator.previous_best_game_score > -float("inf")
                else "N/A"
            )
            print(
                f"--- 🎮 New Best Game: {self.aggregator.best_game_score:.0f} {step_info} (Prev: {prev_str}) ---"
            )

    def record_step(self, step_data: Dict[str, Any]):
        _ = self.aggregator.record_step(step_data)
        g_step = step_data.get("global_step", self.aggregator.current_global_step)
        self.log_summary(g_step)

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        if (
            self.console_log_interval <= 0
            or global_step < self.last_log_step + self.console_log_interval
        ):
            return

        summary = self.get_summary(global_step)
        elapsed_runtime = time.time() - self.start_time
        runtime_hrs = elapsed_runtime / 3600

        best_score_val = (
            f"{summary['best_score']:.2f}"
            if summary["best_score"] > -float("inf")
            else "N/A"
        )
        best_loss_val = (
            f"{summary['best_loss']:.4f}"
            if summary["best_loss"] < float("inf")
            else "N/A"
        )

        # --- MODIFIED: Indicate the window size used for averages ---
        avg_window_size = summary.get("summary_avg_window_size", "?")
        log_str = (
            f"[{runtime_hrs:.1f}h|Console] Step: {global_step/1e6:<6.2f}M | "
            f"Ep: {summary['total_episodes']:<7} | SPS: {summary['steps_per_second']:<5.0f} | "
            f"RLScore(Avg{avg_window_size}): {summary['avg_score_window']:<6.2f} (Best: {best_score_val}) | "
            f"Loss(Avg{avg_window_size}): {summary['avg_loss_window']:.4f} (Best: {best_loss_val}) | "
            f"LR: {summary['current_lr']:.1e} | "
            f"Buf: {summary['buffer_size']/1e6:.2f}M"
        )
        # --- END MODIFIED ---
        if summary["beta"] > 0 and summary["beta"] < 1.0:
            log_str += f" | Beta: {summary['beta']:.3f}"

        print(log_str)

        self.last_log_time = time.time()
        self.last_log_step = global_step

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

    def close(self):
        print("[SimpleStatsRecorder] Closed.")
