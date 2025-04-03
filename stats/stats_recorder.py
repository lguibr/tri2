import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Dict, Any, Optional
import numpy as np


class StatsRecorderBase(ABC):
    """Base class for recording training statistics."""

    @abstractmethod
    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
    ):
        pass

    @abstractmethod
    def record_step(self, step_data: Dict[str, Any]):
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def log_summary(self, global_step: int):
        pass

    @abstractmethod
    def close(self):
        pass


class SimpleStatsRecorder(StatsRecorderBase):
    """Records stats in memory using deques and prints summaries."""

    def __init__(self, console_log_interval: int = 1000, avg_window: int = 100):
        if avg_window <= 0:
            avg_window = 100
        self.console_log_interval = max(1, console_log_interval)  
        self.avg_window = avg_window

        # Deques for averaging step/episode data
        self.step_rewards: Deque[float] = deque(
            maxlen=max(avg_window * 10, 1000)
        )  # Larger window for frequent rewards
        self.losses: Deque[float] = deque(maxlen=avg_window)
        self.grad_norms: Deque[float] = deque(maxlen=avg_window)
        self.episode_scores: Deque[float] = deque(maxlen=avg_window)
        self.episode_lengths: Deque[int] = deque(maxlen=avg_window)

        # Aggregate stats
        self.total_episodes = 0
        self.best_score = -float("inf")

        # Current values (updated frequently)
        self.current_epsilon = 1.0
        self.current_beta = 0.0
        self.current_buffer_size = 0

        # Timing for Steps Per Second (SPS) calculation
        self.last_log_time = time.time()
        self.last_log_step = 0
        self.steps_since_last_log = 0 

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
    ):
        self.episode_scores.append(episode_score)
        self.episode_lengths.append(episode_length)
        self.total_episodes = episode_num
        if episode_score > self.best_score:
            self.best_score = episode_score
            print(
                f"\n--- New Best Score: {self.best_score:.2f} at Ep {episode_num}, Step ~{global_step} ---\n"
            )

    def record_step(self, step_data: Dict[str, Any]):
        # Store values if present in the dictionary
        if "loss" in step_data:
            self.losses.append(step_data["loss"])
        if "grad_norm" in step_data:
            self.grad_norms.append(step_data["grad_norm"])
        if "step_reward" in step_data:
            self.step_rewards.append(step_data["step_reward"])
        if "epsilon" in step_data:
            self.current_epsilon = step_data["epsilon"]
        if "beta" in step_data:
            self.current_beta = step_data["beta"]
        if "buffer_size" in step_data:
            self.current_buffer_size = step_data["buffer_size"]
        self.steps_since_last_log += step_data.get(
            "num_steps", 1
        )  # Assume 1 step if not specified (e.g., from Trainer)

    def get_summary(self) -> Dict[str, Any]:
        """Calculates and returns a dictionary of summary statistics."""
        current_time = time.time()
        elapsed_time = max(
            1e-6, current_time - self.last_log_time
        ) 
        steps_per_sec = self.steps_since_last_log / elapsed_time

        avg_score = np.mean(self.episode_scores) if self.episode_scores else 0.0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        avg_grad = np.mean(self.grad_norms) if self.grad_norms else 0.0
        avg_step_reward = np.mean(self.step_rewards) if self.step_rewards else 0.0

        return {
            "avg_score_100": avg_score,
            "avg_length_100": avg_length,
            "avg_loss_100": avg_loss,
            "avg_grad_100": avg_grad,
            "avg_step_reward_1k": avg_step_reward,
            "total_episodes": self.total_episodes,
            "best_score": self.best_score if self.best_score > -float("inf") else 0.0,
            "epsilon": self.current_epsilon,
            "beta": self.current_beta,
            "buffer_size": self.current_buffer_size,
            "num_ep_scores": len(self.episode_scores),
            "num_ep_lengths": len(self.episode_lengths),  
            "num_losses": len(self.losses),
            "steps_per_second": steps_per_sec,
            # Add global_step here for convenience if needed by DB logger caller
            "global_step": self.last_log_step + self.steps_since_last_log,
        }

    def log_summary(self, global_step: int):
        """Logs summary to console if interval has passed."""
        # Check if enough steps passed based on the *actual* global step
        if (
            global_step == 0
            or global_step < self.last_log_step + self.console_log_interval
        ):
            return

        summary = self.get_summary()  # Calculate stats including SPS

        log_str = (
            f"[Stats] Step: {global_step} | "
            f"Ep: {summary['total_episodes']} | "
            f"SPS: {summary['steps_per_second']:.1f} | "
            f"Score({summary['num_ep_scores']}): {summary['avg_score_100']:.2f} | "
            f"Best: {summary['best_score']:.2f} | "
            f"Len({summary['num_ep_lengths']}): {summary['avg_length_100']:.1f} | " 
            f"Loss({summary['num_losses']}): {summary['avg_loss_100']:.4f} | "
            f"Grad({len(self.grad_norms)}): {summary['avg_grad_100']:.3f} | "
            f"Eps: {summary['epsilon']:.3f} | "
            f"Beta: {summary['beta']:.3f} | "
            f"Buf: {summary['buffer_size']}"
        )
        print(log_str)

        # Reset timing and step count for next interval
        self.last_log_time = time.time()
        self.last_log_step = global_step
        self.steps_since_last_log = 0  # Reset counter

    def close(self):
        """Cleanup for SimpleStatsRecorder (usually nothing needed)."""
        print("[SimpleStatsRecorder] Closed.")
