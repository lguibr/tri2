# File: stats/simple_stats_recorder.py
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List
import numpy as np
import torch
from .stats_recorder import StatsRecorderBase  # Import base class


class SimpleStatsRecorder(StatsRecorderBase):
    """
    Records stats in memory using deques for rolling averages.
    Provides no-op implementations for histogram, image, hparam, graph logging.
    Primarily used internally by TensorBoardStatsRecorder for UI/console display,
    or can be used standalone for simple console logging.
    """

    def __init__(self, console_log_interval: int = 50_000, avg_window: int = 500):
        if avg_window <= 0:
            avg_window = 100
        self.console_log_interval = (
            max(1, console_log_interval) if console_log_interval > 0 else -1
        )  # -1 disables console logging
        self.avg_window = avg_window

        # Deques for averaging recent data
        step_reward_window = max(
            avg_window * 10, 1000
        )  # Larger window for step rewards
        self.step_rewards: Deque[float] = deque(maxlen=step_reward_window)
        self.losses: Deque[float] = deque(maxlen=avg_window)
        self.grad_norms: Deque[float] = deque(maxlen=avg_window)
        self.avg_max_qs: Deque[float] = deque(maxlen=avg_window)
        self.episode_scores: Deque[float] = deque(maxlen=avg_window)  # RL Scores
        self.episode_lengths: Deque[int] = deque(maxlen=avg_window)
        self.game_scores: Deque[int] = deque(maxlen=avg_window)
        self.episode_lines_cleared: Deque[int] = deque(maxlen=avg_window)

        # Aggregate stats
        self.total_episodes = 0
        self.best_score = -float("inf")
        self.best_game_score = -float("inf")
        self.total_lines_cleared = 0

        # Current values (updated via record_step)
        self.current_epsilon: float = (
            0.0  # Noisy nets usually mean effective epsilon is 0
        )
        self.current_beta: float = 0.0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0

        # Timing for SPS calculation and console logging
        self.last_log_time: float = time.time()
        self.last_log_step: int = 0

        print(
            f"[SimpleStatsRecorder] Initialized. Avg Window: {self.avg_window}, Console Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'}"
        )

    def record_episode(
        self,
        episode_score: float,  # RL Score
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        lines_cleared: Optional[int] = None,
    ):
        """Records stats for one completed episode."""
        self.episode_scores.append(episode_score)
        self.episode_lengths.append(episode_length)
        if game_score is not None:
            self.game_scores.append(game_score)
        if lines_cleared is not None:
            self.episode_lines_cleared.append(lines_cleared)
            self.total_lines_cleared += lines_cleared

        self.total_episodes = episode_num  # Update total count

        step_info = (
            f"at Step ~{global_step/1e6:.1f}M" if global_step is not None else ""
        )

        # Print updates for new best scores
        if episode_score > self.best_score:
            self.best_score = episode_score
            print(
                f"\n--- New Best RL Score: {self.best_score:.2f} (Ep {episode_num} {step_info}) ---"
            )
        if game_score is not None and game_score > self.best_game_score:
            self.best_game_score = game_score
            print(
                f"--- New Best Game Score: {self.best_game_score} (Ep {episode_num} {step_info}) ---"
            )

    def record_step(self, step_data: Dict[str, Any]):
        """Records data from a single step or training batch."""
        # Store values in deques if present
        if "loss" in step_data:
            self.losses.append(step_data["loss"])
        if "grad_norm" in step_data:
            self.grad_norms.append(step_data["grad_norm"])
        if "step_reward" in step_data:
            self.step_rewards.append(step_data["step_reward"])
        if "avg_max_q" in step_data:
            self.avg_max_qs.append(step_data["avg_max_q"])

        # Update current values if present
        if "epsilon" in step_data:
            self.current_epsilon = step_data["epsilon"]
        if "beta" in step_data:
            self.current_beta = step_data["beta"]
        if "buffer_size" in step_data:
            self.current_buffer_size = step_data["buffer_size"]
        if "global_step" in step_data:
            self.current_global_step = step_data["global_step"]

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        """Calculates and returns a dictionary of summary statistics."""
        if current_global_step is None:
            current_global_step = self.current_global_step

        current_time = time.time()
        elapsed_time = max(1e-6, current_time - self.last_log_time)
        steps_since_last = max(0, current_global_step - self.last_log_step)
        steps_per_sec = steps_since_last / elapsed_time if elapsed_time > 0 else 0.0

        # Calculate averages safely (return 0 if deque is empty)
        avg_score = np.mean(self.episode_scores) if self.episode_scores else 0.0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        avg_grad = np.mean(self.grad_norms) if self.grad_norms else 0.0
        avg_step_reward = np.mean(self.step_rewards) if self.step_rewards else 0.0
        avg_max_q = np.mean(self.avg_max_qs) if self.avg_max_qs else 0.0
        avg_game_score = np.mean(self.game_scores) if self.game_scores else 0.0
        avg_lines_cleared = (
            np.mean(self.episode_lines_cleared) if self.episode_lines_cleared else 0.0
        )

        summary = {
            # Averaged over window
            "avg_score_100": avg_score,
            "avg_length_100": avg_length,
            "avg_loss_100": avg_loss,
            "avg_grad_100": avg_grad,
            "avg_max_q_100": avg_max_q,
            "avg_game_score_100": avg_game_score,
            "avg_lines_cleared_100": avg_lines_cleared,
            "avg_step_reward_1k": avg_step_reward,  # Note different window size here
            # Aggregate / Current
            "total_episodes": self.total_episodes,
            "best_score": self.best_score if self.best_score > -float("inf") else 0.0,
            "best_game_score": (
                self.best_game_score if self.best_game_score > -float("inf") else 0.0
            ),
            "total_lines_cleared": self.total_lines_cleared,
            "epsilon": self.current_epsilon,
            "beta": self.current_beta,
            "buffer_size": self.current_buffer_size,
            "steps_per_second": steps_per_sec,
            "global_step": current_global_step,
            # Counts for context
            "num_ep_scores": len(self.episode_scores),
            "num_ep_lengths": len(self.episode_lengths),
            "num_losses": len(self.losses),
            "num_avg_max_qs": len(self.avg_max_qs),
            "num_game_scores": len(self.game_scores),
            "num_lines_cleared": len(self.episode_lines_cleared),
        }
        return summary

    def log_summary(self, global_step: int):
        """Logs summary to console if interval has passed."""
        # Check if console logging is enabled and interval is met
        if (
            self.console_log_interval <= 0
            or global_step < self.last_log_step + self.console_log_interval
        ):
            return

        summary = self.get_summary(global_step)

        log_str = (
            f"[Stats] Step: {global_step/1e6:<7.2f}M | "
            f"Ep: {summary['total_episodes']:<8} | "
            f"SPS: {summary['steps_per_second']:<6.0f} | "
            f"RLScore({summary['num_ep_scores']}): {summary['avg_score_100']:<6.2f} | "
            f"GameScore({summary['num_game_scores']}): {summary['avg_game_score_100']:<6.1f} | "
            f"Lines({summary['num_lines_cleared']}): {summary['avg_lines_cleared_100']:<5.2f} | "
            f"Len({summary['num_ep_lengths']}): {summary['avg_length_100']:.1f} | "
            f"Loss({summary['num_losses']}): {summary['avg_loss_100']:.4f} | "
            f"AvgMaxQ({summary['num_avg_max_qs']}): {summary['avg_max_q_100']:.3f} | "
            # f"Eps: {summary['epsilon']:.3f} | " # Epsilon is always 0 for noisy
            f"Beta: {summary['beta']:.3f} | "
            f"Buf: {summary['buffer_size']/1e6:.2f}M"
        )
        print(log_str)

        # Update timing info for next SPS calculation
        self.last_log_time = time.time()
        self.last_log_step = global_step

    # --- New Methods (No-op Implementation) ---
    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Placeholder: Simple recorder does not log histograms."""
        pass  # No operation

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Placeholder: Simple recorder does not log images."""
        pass  # No operation

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Placeholder: Simple recorder does not log hyperparameters."""
        pass  # No operation

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[torch.Tensor] = None
    ):
        """Placeholder: Simple recorder does not log the model graph."""
        pass  # No operation

    # --- End New ---

    def close(self):
        # No resources to close for simple in-memory recorder
        print("[SimpleStatsRecorder] Closed.")
