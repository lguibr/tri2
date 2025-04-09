# File: stats/simple_stats_recorder.py
import time
from typing import Deque, Dict, Any, Optional, Union, List, TYPE_CHECKING
import numpy as np
import torch
import threading
import logging
import ray # Added Ray

# from .stats_recorder import StatsRecorderBase # Keep Base class import
from .aggregator import StatsAggregatorActor # Import Actor for type hint
from config import StatsConfig, TrainConfig
from utils.helpers import format_eta

if TYPE_CHECKING:
    from environment.game_state import GameState
    StatsAggregatorHandle = ray.actor.ActorHandle # Type hint for handle

# Import base class correctly
from .stats_recorder import StatsRecorderBase

logger = logging.getLogger(__name__)


class SimpleStatsRecorder(StatsRecorderBase):
    """Logs aggregated statistics fetched from StatsAggregatorActor to the console periodically."""

    def __init__(
        self,
        aggregator: "StatsAggregatorHandle", # Expect Actor Handle
        console_log_interval: int = StatsConfig.CONSOLE_LOG_FREQ,
        train_config: Optional[TrainConfig] = None,
    ):
        self.aggregator_handle = aggregator # Store actor handle
        self.console_log_interval = (
            max(1, console_log_interval) if console_log_interval > 0 else -1
        )
        self.train_config = train_config if train_config else TrainConfig()
        self.last_log_time: float = time.time()
        # Get summary window size from config, actor doesn't store it directly this way
        self.summary_avg_window = StatsConfig.STATS_AVG_WINDOW[0] if StatsConfig.STATS_AVG_WINDOW else 100
        self.updates_since_last_log = 0
        self._lock = threading.Lock() # Lock for updates_since_last_log counter
        logger.info(
            f"[SimpleStatsRecorder] Initialized. Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'} updates/episodes. Avg Window: {self.summary_avg_window}"
        )
        # Store last known best values locally to detect changes
        self._last_best_score = -float('inf')
        self._last_best_vloss = float('inf')
        self._last_best_ploss = float('inf')
        self._last_best_mcts_time = float('inf')


    def _log_new_best(
        self,
        metric_name: str,
        current_best: float,
        previous_best: float,
        step: int,
        is_loss: bool,
        is_time: bool = False,
    ):
        """Logs a new best value achieved."""
        # This logic remains the same, uses passed values
        improvement_made = False
        if is_loss:
            if np.isfinite(current_best) and current_best < previous_best: improvement_made = True
        else:
            if np.isfinite(current_best) and current_best > previous_best: improvement_made = True
        if not improvement_made: return

        if is_time:
            format_str = "{:.3f}s"
            prev_str = format_str.format(previous_best) if np.isfinite(previous_best) and previous_best != float("inf") else "N/A"
            current_str = format_str.format(current_best)
            prefix = "⏱️"
        elif is_loss:
            format_str = "{:.4f}"
            prev_str = format_str.format(previous_best) if np.isfinite(previous_best) and previous_best != float("inf") else "N/A"
            current_str = format_str.format(current_best)
            prefix = "📉"
        else:
            format_str = "{:.0f}"
            prev_str = format_str.format(previous_best) if np.isfinite(previous_best) and previous_best != -float("inf") else "N/A"
            current_str = format_str.format(current_best)
            prefix = "🎮"

        step_info = f"at Step ~{step/1e6:.1f}M" if step > 0 else "at Start"
        logger.info(f"--- {prefix} New Best {metric_name}: {current_str} {step_info} (Prev: {prev_str}) ---")

    def record_episode(
        self,
        episode_outcome: float, # These args are now less relevant as data comes from aggregator
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional["GameState"] = None, # This arg is also less relevant
    ):
        """Checks for new bests by fetching summary from aggregator."""
        # This method is now primarily a trigger based on episode completion
        # The actual recording happens via remote calls from workers
        # We just need to check if the interval requires logging a summary
        current_step = global_step # Use passed step if available
        if current_step is None:
             # Fetch step from aggregator if not passed (blocking)
             try:
                  step_ref = self.aggregator_handle.get_current_global_step.remote()
                  current_step = ray.get(step_ref)
             except Exception as e:
                  logger.error(f"Error fetching global step from aggregator: {e}")
                  current_step = 0 # Fallback

        self._check_and_log_summary(current_step)


    def record_step(self, step_data: Dict[str, Any]):
        """Checks for new bests by fetching summary from aggregator."""
        # This method is now primarily a trigger based on step completion
        # The actual recording happens via remote calls from workers
        g_step = step_data.get("global_step")
        if g_step is None:
             # Fetch step from aggregator if not passed (blocking)
             try:
                  step_ref = self.aggregator_handle.get_current_global_step.remote()
                  g_step = ray.get(step_ref)
             except Exception as e:
                  logger.error(f"Error fetching global step from aggregator: {e}")
                  g_step = 0 # Fallback

        self._check_and_log_summary(g_step)


    def _check_and_log_summary(self, global_step: int):
        """Checks if the logging interval is met and logs summary by fetching from actor."""
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
            self.log_summary(global_step) # Fetch data and log

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Gets the summary dictionary from the aggregator actor (blocking)."""
        if not self.aggregator_handle: return {}
        try:
            summary_ref = self.aggregator_handle.get_summary.remote(current_global_step)
            summary = ray.get(summary_ref)
            return summary
        except Exception as e:
            logger.error(f"Error getting summary from StatsAggregatorActor: {e}")
            return {"error": str(e)}

    def get_plot_data(self) -> Dict[str, Deque]:
        """Gets the plot data deques from the aggregator actor (blocking)."""
        # Note: Returns lists, not deques, due to serialization
        if not self.aggregator_handle: return {}
        try:
            plot_data_ref = self.aggregator_handle.get_plot_data.remote()
            plot_data_list_dict = ray.get(plot_data_ref)
            # Convert lists back to deques locally if needed by UI plotter
            # For now, return the dict of lists as received
            return plot_data_list_dict
        except Exception as e:
            logger.error(f"Error getting plot data from StatsAggregatorActor: {e}")
            return {"error": str(e)}

    def log_summary(self, global_step: int):
        """Logs the current summary statistics fetched from the aggregator actor."""
        summary = self.get_summary(global_step)
        if not summary or "error" in summary:
             logger.error(f"Could not log summary, failed to fetch data: {summary.get('error', 'Unknown error')}")
             return

        # --- Check for New Bests ---
        # Compare fetched best values with locally stored last known bests
        new_best_score = summary.get("best_game_score", -float('inf'))
        new_best_vloss = summary.get("best_value_loss", float('inf'))
        new_best_ploss = summary.get("best_policy_loss", float('inf'))
        new_best_mcts_time = summary.get("best_mcts_sim_time", float('inf'))

        if new_best_score > self._last_best_score:
             self._log_new_best("Game Score", new_best_score, self._last_best_score, summary.get("best_game_score_step", 0), is_loss=False)
             self._last_best_score = new_best_score
        if new_best_vloss < self._last_best_vloss:
             self._log_new_best("V.Loss", new_best_vloss, self._last_best_vloss, summary.get("best_value_loss_step", 0), is_loss=True)
             self._last_best_vloss = new_best_vloss
        if new_best_ploss < self._last_best_ploss:
             self._log_new_best("P.Loss", new_best_ploss, self._last_best_ploss, summary.get("best_policy_loss_step", 0), is_loss=True)
             self._last_best_ploss = new_best_ploss
        if new_best_mcts_time < self._last_best_mcts_time:
             self._log_new_best("MCTS Sim Time", new_best_mcts_time, self._last_best_mcts_time, summary.get("best_mcts_sim_time_step", 0), is_loss=True, is_time=True)
             self._last_best_mcts_time = new_best_mcts_time
        # --- End New Best Check ---


        runtime_hrs = (time.time() - summary.get("start_time", time.time())) / 3600
        best_score_str = f"{new_best_score:.0f}" if new_best_score > -float("inf") else "N/A"
        avg_win = summary.get("summary_avg_window_size", self.summary_avg_window)
        buf_size = summary.get("buffer_size", 0)
        min_buf = summary.get("min_buffer_size", self.train_config.MIN_BUFFER_SIZE_TO_TRAIN)
        phase = "Buffering" if buf_size < min_buf and global_step == 0 else "Training"
        steps_sec = summary.get("steps_per_second_avg", 0.0)

        current_game = summary.get("current_self_play_game_number", 0)
        current_game_step = summary.get("current_self_play_game_steps", 0)
        game_prog_str = f"Game: {current_game}({current_game_step})" if current_game > 0 else ""

        log_items = [
            f"[{runtime_hrs:.1f}h|{phase}]",
            f"Step: {global_step/1e6:<6.2f}M ({steps_sec:.1f}/s)",
            f"Ep: {summary.get('total_episodes', 0):<7,}".replace(",", "_"),
            f"Buf: {buf_size:,}/{min_buf:,}".replace(",", "_"),
            f"Score(Avg{avg_win}): {summary.get('avg_game_score_window', 0.0):<6.0f} (Best: {best_score_str})",
        ]

        if global_step > 0 or phase == "Training":
            log_items.extend([
                f"VLoss(Avg{avg_win}): {summary.get('value_loss', 0.0):.4f}",
                f"PLoss(Avg{avg_win}): {summary.get('policy_loss', 0.0):.4f}",
                f"LR: {summary.get('current_lr', 0.0):.1e}",
            ])
        else: log_items.append("Loss: N/A")

        mcts_sim_time_avg = summary.get("mcts_simulation_time_avg", 0.0)
        mcts_nn_time_avg = summary.get("mcts_nn_prediction_time_avg", 0.0)
        mcts_nodes_avg = summary.get("mcts_nodes_explored_avg", 0.0)
        if mcts_sim_time_avg > 0 or mcts_nn_time_avg > 0 or mcts_nodes_avg > 0:
            mcts_str = f"MCTS(Avg{avg_win}): SimT={mcts_sim_time_avg*1000:.1f}ms | NNT={mcts_nn_time_avg*1000:.1f}ms | Nodes={mcts_nodes_avg:.0f}"
            log_items.append(mcts_str)

        if game_prog_str: log_items.append(game_prog_str)

        training_target_step = summary.get("training_target_step", 0)
        if training_target_step > 0 and steps_sec > 0:
            steps_remaining = training_target_step - global_step
            if steps_remaining > 0:
                eta_seconds = steps_remaining / steps_sec
                eta_str = format_eta(eta_seconds)
                log_items.append(f"ETA: {eta_str}")

        logger.info(" | ".join(log_items))
        self.last_log_time = time.time()

    # --- No-op methods for other recording types ---
    def record_histogram(self, tag: str, values: Union[np.ndarray, torch.Tensor, List[float]], global_step: int): pass
    def record_image(self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int): pass
    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]): pass
    def record_graph(self, model: torch.nn.Module, input_to_model: Optional[Any] = None): pass

    def close(self, is_cleanup: bool = False):
        # Ensure final summary is logged if interval logging is enabled
        if self.console_log_interval > 0 and self.updates_since_last_log > 0:
            logger.info("[SimpleStatsRecorder] Logging final summary before closing...")
            # Fetch final step count before logging
            final_step = 0
            if self.aggregator_handle:
                 try:
                      step_ref = self.aggregator_handle.get_current_global_step.remote()
                      final_step = ray.get(step_ref)
                 except Exception: pass # Ignore error on close
            self.log_summary(final_step)
        logger.info(f"[SimpleStatsRecorder] Closed (is_cleanup={is_cleanup}).")