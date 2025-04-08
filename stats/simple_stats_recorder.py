import time
from typing import Deque, Dict, Any, Optional, Union, List, TYPE_CHECKING
import numpy as np
import torch
import threading
import logging  # Import logging

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from config import StatsConfig, TrainConfig

if TYPE_CHECKING:
    from environment.game_state import GameState

logger = logging.getLogger(__name__)  # Use logger


class SimpleStatsRecorder(StatsRecorderBase):
    """Logs aggregated statistics to the console periodically."""

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_log_interval: int = StatsConfig.CONSOLE_LOG_FREQ,
        train_config: Optional[TrainConfig] = None,
    ):
        self.aggregator = aggregator
        self.console_log_interval = (
            max(1, console_log_interval) if console_log_interval > 0 else -1
        )
        self.train_config = train_config if train_config else TrainConfig()
        self.last_log_time: float = time.time()
        self.summary_avg_window = self.aggregator.summary_avg_window
        self.updates_since_last_log = 0
        self._lock = threading.Lock()
        logger.info(  # Use logger
            f"[SimpleStatsRecorder] Initialized. Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'} updates/episodes. Avg Window: {self.summary_avg_window}"
        )

    def _log_new_best(
        self,
        metric_name: str,
        current_best: float,
        previous_best: float,
        step: int,
        is_loss: bool,
    ):
        """Logs a new best value achieved."""
        if (
            not np.isfinite(current_best)
            or (is_loss and current_best == float("inf"))
            or (not is_loss and current_best == -float("inf"))
        ):
            return
        prev_str = "N/A"
        if np.isfinite(previous_best) and (
            (is_loss and previous_best != float("inf"))
            or (not is_loss and previous_best != -float("inf"))
        ):
            prev_str = f"{previous_best:.4f}" if is_loss else f"{previous_best:.0f}"
        current_str = f"{current_best:.4f}" if is_loss else f"{current_best:.0f}"
        step_info = f"at Step ~{step/1e6:.1f}M" if step > 0 else "at Start"
        prefix = "📉" if is_loss else "🎮"
        # Use logger.info instead of print
        logger.info(
            f"--- {prefix} New Best {metric_name}: {current_str} {step_info} (Prev: {prev_str}) ---"
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
    ):
        """Records episode stats and prints new bests to console."""
        update_info = self.aggregator.record_episode(
            episode_outcome,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
            game_state_for_best,
        )
        current_step = (
            global_step
            if global_step is not None
            else self.aggregator.storage.current_global_step
        )

        if update_info.get("new_best_game"):
            self._log_new_best(
                "Game Score",
                self.aggregator.storage.best_game_score,
                self.aggregator.storage.previous_best_game_score,
                current_step,
                is_loss=False,
            )
        # Note: Best outcome logging removed as it's less informative than score for this game

        # Trigger summary check after recording an episode
        self._check_and_log_summary(current_step)

    def record_step(self, step_data: Dict[str, Any]):
        """Records step stats and triggers console logging if interval met."""
        update_info = self.aggregator.record_step(step_data)
        g_step = step_data.get(
            "global_step", self.aggregator.storage.current_global_step
        )

        if update_info.get("new_best_value_loss"):
            self._log_new_best(
                "V.Loss",
                self.aggregator.storage.best_value_loss,
                self.aggregator.storage.previous_best_value_loss,
                g_step,
                is_loss=True,
            )
        if update_info.get("new_best_policy_loss"):
            self._log_new_best(
                "P.Loss",
                self.aggregator.storage.best_policy_loss,
                self.aggregator.storage.previous_best_policy_loss,
                g_step,
                is_loss=True,
            )

        # Trigger summary check after *any* step record (training or intermediate)
        self._check_and_log_summary(g_step)

    def _check_and_log_summary(self, global_step: int):
        """Checks if the logging interval is met and logs summary."""
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
            self.log_summary(global_step)

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Gets the summary dictionary from the aggregator."""
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        """Gets the plot data deques from the aggregator."""
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        """Logs the current summary statistics to the console."""
        summary = self.get_summary(global_step)
        runtime_hrs = (time.time() - self.aggregator.storage.start_time) / 3600
        best_score = (
            f"{summary['best_game_score']:.0f}"
            if summary["best_game_score"] > -float("inf")
            else "N/A"
        )
        best_v = (
            f"{summary['best_value_loss']:.4f}"
            if summary["best_value_loss"] < float("inf")
            else "N/A"
        )
        best_p = (
            f"{summary['best_policy_loss']:.4f}"
            if summary["best_policy_loss"] < float("inf")
            else "N/A"
        )
        avg_win = summary.get("summary_avg_window_size", "?")
        buf_size = summary.get("buffer_size", 0)
        min_buf = self.train_config.MIN_BUFFER_SIZE_TO_TRAIN
        phase = "Buffering" if buf_size < min_buf and global_step == 0 else "Training"

        # Include current game/step from self-play workers if available
        current_game = summary.get("current_self_play_game_number", 0)
        current_game_step = summary.get("current_self_play_game_steps", 0)
        game_prog_str = (
            f"Game: {current_game} (Step {current_game_step})"
            if current_game > 0
            else ""
        )

        log_items = [
            f"[{runtime_hrs:.1f}h|{phase}]",
            f"Step: {global_step/1e6:<6.2f}M",
            f"Ep: {summary['total_episodes']:<7,}".replace(",", "_"),
            f"Buf: {buf_size:,}/{min_buf:,}".replace(",", "_"),
            f"Score(Avg{avg_win}): {summary['avg_game_score_window']:<6.0f} (Best: {best_score})",
        ]
        if game_prog_str:
            log_items.append(game_prog_str)  # Add game progress

        if global_step > 0:  # Only show loss/LR if training has occurred
            log_items.extend(
                [
                    f"V.Loss(Avg{avg_win}): {summary['value_loss']:.4f} (Best: {best_v})",
                    f"P.Loss(Avg{avg_win}): {summary['policy_loss']:.4f} (Best: {best_p})",
                    f"LR: {summary['current_lr']:.1e}",
                ]
            )
        elif phase == "Buffering":  # Show N/A during buffering
            log_items.append("Loss: N/A (Buffering)")

        # Use logger.info instead of print
        logger.info(" | ".join(log_items))
        self.last_log_time = time.time()

    # --- No-op methods for other recording types ---
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
        # Use logger.info
        logger.info(f"[SimpleStatsRecorder] Closed (is_cleanup={is_cleanup}).")
