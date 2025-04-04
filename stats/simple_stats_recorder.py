# File: stats/simple_stats_recorder.py
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List, Tuple, Callable
import numpy as np
import torch
from .stats_recorder import StatsRecorderBase
from config import EnvConfig, StatsConfig
import warnings


class SimpleStatsRecorder(StatsRecorderBase):
    """
    Records stats in memory using deques for rolling averages.
    Provides no-op implementations for histogram, image, hparam, graph logging.
    Primarily used internally by TensorBoardStatsRecorder for UI/console display,
    or can be used standalone for simple console logging.
    """

    def __init__(
        self,
        console_log_interval: int = 50_000,
        avg_window: int = StatsConfig.STATS_AVG_WINDOW,
    ):
        if avg_window <= 0:
            avg_window = 100
        self.console_log_interval = (
            max(1, console_log_interval) if console_log_interval > 0 else -1
        )
        self.avg_window = avg_window

        # Data Deques
        step_reward_window = max(avg_window * 10, 1000)
        self.step_rewards: Deque[float] = deque(maxlen=step_reward_window)
        self.losses: Deque[float] = deque(maxlen=avg_window)
        self.grad_norms: Deque[float] = deque(maxlen=avg_window)
        self.avg_max_qs: Deque[float] = deque(maxlen=avg_window)
        self.episode_scores: Deque[float] = deque(maxlen=avg_window)
        self.episode_lengths: Deque[int] = deque(maxlen=avg_window)
        self.game_scores: Deque[int] = deque(maxlen=avg_window)
        self.episode_lines_cleared: Deque[int] = deque(maxlen=avg_window)
        self.sps_values: Deque[float] = deque(maxlen=avg_window)
        self.buffer_sizes: Deque[int] = deque(maxlen=avg_window)
        self.beta_values: Deque[float] = deque(maxlen=avg_window)
        self.best_rl_score_history: Deque[float] = deque(maxlen=avg_window)
        self.best_game_score_history: Deque[int] = deque(maxlen=avg_window)
        self.lr_values: Deque[float] = deque(maxlen=avg_window)
        self.epsilon_values: Deque[float] = deque(maxlen=avg_window)

        # Current State / Best Values
        self.total_episodes = 0
        self.total_lines_cleared = 0
        self.current_epsilon: float = 0.0
        self.current_beta: float = 0.0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0
        self.current_sps: float = 0.0
        self.current_lr: float = 0.0

        # --- MODIFIED: Detailed Best Tracking ---
        self.best_score: float = -float("inf")
        self.previous_best_score: float = -float("inf")
        self.best_score_step: int = 0

        self.best_game_score: int = -float("inf")  # Use float for consistent init
        self.previous_best_game_score: int = -float("inf")
        self.best_game_score_step: int = 0

        self.best_loss: float = float("inf")  # Lower is better for loss
        self.previous_best_loss: float = float("inf")
        self.best_loss_step: int = 0
        # --- END MODIFIED ---

        # Timing / Logging Control
        self.last_log_time: float = time.time()
        self.last_log_step: int = 0
        self.start_time: float = time.time()
        print(
            f"[SimpleStatsRecorder] Initialized. Avg Window: {self.avg_window}, Console Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'}"
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
        # Use provided global_step or the internally tracked one
        current_step = (
            global_step if global_step is not None else self.current_global_step
        )

        self.episode_scores.append(episode_score)
        self.episode_lengths.append(episode_length)
        if game_score is not None:
            self.game_scores.append(game_score)
        if lines_cleared is not None:
            self.episode_lines_cleared.append(lines_cleared)
            self.total_lines_cleared += lines_cleared
        self.total_episodes = episode_num
        step_info = f"at Step ~{current_step/1e6:.1f}M"

        # --- MODIFIED: Update detailed best score tracking ---
        if episode_score > self.best_score:
            self.previous_best_score = self.best_score  # Store old best
            self.best_score = episode_score
            self.best_score_step = current_step  # Record step
            prev_str = (
                f"{self.previous_best_score:.2f}"
                if self.previous_best_score > -float("inf")
                else "N/A"
            )
            print(
                f"\n--- 🏆 New Best RL: {self.best_score:.2f} {step_info} (Prev: {prev_str}) ---"
            )

        if game_score is not None and game_score > self.best_game_score:
            self.previous_best_game_score = self.best_game_score  # Store old best
            self.best_game_score = game_score
            self.best_game_score_step = current_step  # Record step
            prev_str = (
                f"{self.previous_best_game_score:.0f}"
                if self.previous_best_game_score > -float("inf")
                else "N/A"
            )
            print(
                f"--- 🎮 New Best Game: {self.best_game_score} {step_info} (Prev: {prev_str}) ---"
            )
        # --- END MODIFIED ---

        # Update history deques for plotting best scores over time
        current_best_rl = self.best_score if self.best_score > -float("inf") else 0.0
        current_best_game = (
            self.best_game_score if self.best_game_score > -float("inf") else 0
        )
        self.best_rl_score_history.append(current_best_rl)
        self.best_game_score_history.append(current_best_game)

    def record_step(self, step_data: Dict[str, Any]):
        g_step = step_data.get("global_step", self.current_global_step)
        if g_step > self.current_global_step:
            self.current_global_step = g_step

        # Append data to deques if present in step_data
        if "loss" in step_data and step_data["loss"] is not None and g_step > 0:
            current_loss = step_data["loss"]
            self.losses.append(current_loss)
            # --- MODIFIED: Track best loss ---
            if current_loss < self.best_loss:
                self.previous_best_loss = self.best_loss
                self.best_loss = current_loss
                self.best_loss_step = g_step
                prev_str = (
                    f"{self.previous_best_loss:.4f}"
                    if self.previous_best_loss < float("inf")
                    else "N/A"
                )
                # Optional: Print new best loss to console
                # print(f"--- ✨ New Best Loss: {self.best_loss:.4f} at Step ~{g_step/1e6:.1f}M (Prev: {prev_str}) ---")
            # --- END MODIFIED ---

        if (
            "grad_norm" in step_data
            and step_data["grad_norm"] is not None
            and g_step > 0
        ):
            self.grad_norms.append(step_data["grad_norm"])
        if "step_reward" in step_data and step_data["step_reward"] is not None:
            self.step_rewards.append(step_data["step_reward"])
        if (
            "avg_max_q" in step_data
            and step_data["avg_max_q"] is not None
            and g_step > 0
        ):
            self.avg_max_qs.append(step_data["avg_max_q"])
        if "beta" in step_data and step_data["beta"] is not None:
            self.current_beta = step_data["beta"]
            self.beta_values.append(self.current_beta)
        if "buffer_size" in step_data and step_data["buffer_size"] is not None:
            self.current_buffer_size = step_data["buffer_size"]
            self.buffer_sizes.append(self.current_buffer_size)
        if "lr" in step_data and step_data["lr"] is not None:
            self.current_lr = step_data["lr"]
            self.lr_values.append(self.current_lr)
        if "epsilon" in step_data and step_data["epsilon"] is not None:
            self.current_epsilon = step_data["epsilon"]
            self.epsilon_values.append(self.current_epsilon)

        # Calculate SPS
        if "step_time" in step_data and step_data["step_time"] > 1e-6:
            num_steps_in_call = step_data.get("num_steps_processed", 1)
            sps = num_steps_in_call / step_data["step_time"]
            self.sps_values.append(sps)
            self.current_sps = sps

        # Trigger console log periodically
        self.log_summary(g_step)

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        if current_global_step is None:
            current_global_step = self.current_global_step

        # Calculate averages safely
        avg_sps = np.mean(self.sps_values) if self.sps_values else self.current_sps
        avg_score = np.mean(self.episode_scores) if self.episode_scores else 0.0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        avg_max_q = np.mean(self.avg_max_qs) if self.avg_max_qs else 0.0
        avg_game_score = np.mean(self.game_scores) if self.game_scores else 0.0
        avg_lines_cleared = (
            np.mean(self.episode_lines_cleared) if self.episode_lines_cleared else 0.0
        )
        avg_lr = np.mean(self.lr_values) if self.lr_values else self.current_lr

        summary = {
            # Averages
            "avg_score_window": avg_score,
            "avg_length_window": avg_length,
            "avg_loss_window": avg_loss,
            "avg_max_q_window": avg_max_q,
            "avg_game_score_window": avg_game_score,
            "avg_lines_cleared_window": avg_lines_cleared,
            "avg_sps_window": avg_sps,
            "avg_lr_window": avg_lr,
            # Current / Total
            "total_episodes": self.total_episodes,
            "beta": self.current_beta,
            "buffer_size": self.current_buffer_size,
            "steps_per_second": self.current_sps,
            "global_step": current_global_step,
            "current_lr": self.current_lr,
            # --- MODIFIED: Add detailed best tracking ---
            "best_score": self.best_score,
            "previous_best_score": self.previous_best_score,
            "best_score_step": self.best_score_step,
            "best_game_score": self.best_game_score,
            "previous_best_game_score": self.previous_best_game_score,
            "best_game_score_step": self.best_game_score_step,
            "best_loss": self.best_loss,
            "previous_best_loss": self.previous_best_loss,
            "best_loss_step": self.best_loss_step,
            # --- END MODIFIED ---
            # Counts (for debugging/UI)
            "num_ep_scores": len(self.episode_scores),
            "num_losses": len(self.losses),
        }
        return summary

    def log_summary(self, global_step: int):
        """Logs summary statistics to the console periodically."""
        if (
            self.console_log_interval <= 0
            or global_step < self.last_log_step + self.console_log_interval
        ):
            return

        summary = self.get_summary(global_step)
        elapsed_runtime = time.time() - self.start_time
        runtime_hrs = elapsed_runtime / 3600

        best_score_val = (
            summary["best_score"] if summary["best_score"] > -float("inf") else "N/A"
        )
        if isinstance(best_score_val, float):
            best_score_val = f"{best_score_val:.2f}"

        best_loss_val = (
            summary["best_loss"] if summary["best_loss"] < float("inf") else "N/A"
        )
        if isinstance(best_loss_val, float):
            best_loss_val = f"{best_loss_val:.4f}"

        log_str = (
            f"[{runtime_hrs:.1f}h|Stats] Step: {global_step/1e6:<6.2f}M | "
            f"Ep: {summary['total_episodes']:<7} | SPS: {summary['steps_per_second']:<5.0f} | "
            f"RLScore(Avg): {summary['avg_score_window']:<6.2f} (Best: {best_score_val}) | "
            f"Loss(Avg): {summary['avg_loss_window']:.4f} (Best: {best_loss_val}) | "
            f"LR: {summary['current_lr']:.1e} | "
            f"Buf: {summary['buffer_size']/1e6:.2f}M"
        )
        print(log_str)

        self.last_log_time = time.time()
        self.last_log_step = global_step

    def get_plot_data(self) -> Dict[str, Deque]:
        """Returns copies of deques needed for UI plotting."""
        return {
            "episode_scores": self.episode_scores.copy(),
            "episode_lengths": self.episode_lengths.copy(),
            "losses": self.losses.copy(),
            "avg_max_qs": self.avg_max_qs.copy(),
            "game_scores": self.game_scores.copy(),
            "episode_lines_cleared": self.episode_lines_cleared.copy(),
            "sps_values": self.sps_values.copy(),
            "buffer_sizes": self.buffer_sizes.copy(),
            "beta_values": self.beta_values.copy(),
            "best_rl_score_history": self.best_rl_score_history.copy(),
            "best_game_score_history": self.best_game_score_history.copy(),
            "lr_values": self.lr_values.copy(),
            "epsilon_values": self.epsilon_values.copy(),
        }

    # --- No-op methods for compatibility with base class ---
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
        self, model: torch.nn.Module, input_to_model: Optional[torch.Tensor] = None
    ):
        pass

    def close(self):
        """Closes the recorder (no-op for simple recorder)."""
        print("[SimpleStatsRecorder] Closed.")
