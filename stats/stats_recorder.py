# File: stats/stats_recorder.py
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
        global_step: Optional[int] = None,  # Added global_step for context
    ):
        """Record stats for a completed episode."""
        pass

    @abstractmethod
    def record_step(self, step_data: Dict[str, Any]):
        """Record stats from a training or environment step."""
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """Return a dictionary containing summary statistics."""
        pass

    @abstractmethod
    def log_summary(self, global_step: int):
        """Log the summary statistics (e.g., print to console)."""
        pass

    @abstractmethod
    def close(self):
        """Perform any necessary cleanup (e.g., close files/connections)."""
        pass


class SimpleStatsRecorder(StatsRecorderBase):
    """Records stats in memory using deques and prints summaries."""

    def __init__(self, console_log_interval: int = 1000, avg_window: int = 100):
        if avg_window <= 0:
            print(f"Warning: avg_window must be > 0. Setting to default 100.")
            avg_window = 100
        self.console_log_interval = max(1, console_log_interval)  # Ensure interval > 0
        self.avg_window = avg_window

        # Deques for averaging recent data
        # Use larger window for frequent step rewards, smaller for episode/loss stats
        step_reward_window = max(avg_window * 10, 1000)
        self.step_rewards: Deque[float] = deque(maxlen=step_reward_window)
        self.losses: Deque[float] = deque(maxlen=avg_window)
        self.grad_norms: Deque[float] = deque(maxlen=avg_window)
        self.avg_max_qs: Deque[float] = deque(
            maxlen=avg_window
        )  # <<< NEW for Avg Max Q
        self.episode_scores: Deque[float] = deque(maxlen=avg_window)
        self.episode_lengths: Deque[int] = deque(maxlen=avg_window)

        # Aggregate stats over the entire run
        self.total_episodes = 0
        self.best_score = -float("inf")

        # Current values (updated frequently, not averaged)
        self.current_epsilon: float = 1.0
        self.current_beta: float = 0.0  # Default beta if PER not used
        self.current_buffer_size: int = 0

        # Timing for Steps Per Second (SPS) calculation
        self.last_log_time: float = time.time()
        self.last_log_step: int = 0
        # self.steps_since_last_log: int = 0 # Removed: Calculate SPS based on global_step delta

        print(
            f"[SimpleStatsRecorder] Initialized. Log Interval: {self.console_log_interval} steps, Avg Window: {self.avg_window}"
        )

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,  # Accept global_step
    ):
        """Records stats for one completed episode."""
        self.episode_scores.append(episode_score)
        self.episode_lengths.append(episode_length)
        self.total_episodes = episode_num  # Update total count

        if episode_score > self.best_score:
            self.best_score = episode_score
            step_info = f"at Step ~{global_step}" if global_step is not None else ""
            print(
                f"\n--- New Best Score: {self.best_score:.2f} (Ep {episode_num} {step_info}) ---\n"
            )

    def record_step(self, step_data: Dict[str, Any]):
        """Records stats from a single training step or environment interaction."""
        # Store values if they are present in the dictionary
        if "loss" in step_data and step_data["loss"] is not None:
            self.losses.append(step_data["loss"])
        if "grad_norm" in step_data and step_data["grad_norm"] is not None:
            self.grad_norms.append(step_data["grad_norm"])
        if "step_reward" in step_data and step_data["step_reward"] is not None:
            self.step_rewards.append(step_data["step_reward"])
        if "epsilon" in step_data and step_data["epsilon"] is not None:
            self.current_epsilon = step_data["epsilon"]
        if "beta" in step_data and step_data["beta"] is not None:
            self.current_beta = step_data["beta"]
        if "buffer_size" in step_data and step_data["buffer_size"] is not None:
            self.current_buffer_size = step_data["buffer_size"]
        if "avg_max_q" in step_data and step_data["avg_max_q"] is not None:  # <<< NEW
            self.avg_max_qs.append(step_data["avg_max_q"])
        # self.steps_since_last_log += step_data.get("num_steps", 1) # Removed

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Calculates and returns a dictionary of summary statistics."""
        current_time = time.time()
        elapsed_time = max(
            1e-6, current_time - self.last_log_time
        )  # Avoid division by zero
        steps_since_last = max(0, current_global_step - self.last_log_step)
        steps_per_sec = steps_since_last / elapsed_time if elapsed_time > 0 else 0.0

        # Calculate averages safely (return 0 if deque is empty)
        avg_score = np.mean(self.episode_scores) if self.episode_scores else 0.0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        avg_grad = np.mean(self.grad_norms) if self.grad_norms else 0.0
        avg_step_reward = np.mean(self.step_rewards) if self.step_rewards else 0.0
        avg_max_q = np.mean(self.avg_max_qs) if self.avg_max_qs else 0.0  # <<< NEW

        summary = {
            # Averaged over window
            "avg_score_100": avg_score,
            "avg_length_100": avg_length,
            "avg_loss_100": avg_loss,
            "avg_grad_100": avg_grad,
            "avg_max_q_100": avg_max_q,  # <<< NEW Key
            "avg_step_reward_1k": avg_step_reward,  # Key reflects typical larger window
            # Current / Aggregate values
            "total_episodes": self.total_episodes,
            "best_score": self.best_score if self.best_score > -float("inf") else 0.0,
            "epsilon": self.current_epsilon,
            "beta": self.current_beta,
            "buffer_size": self.current_buffer_size,
            # Context
            "steps_per_second": steps_per_sec,
            "global_step": current_global_step,  # Pass current step for reference
            # Counts (useful for plotter/debugging)
            "num_ep_scores": len(self.episode_scores),
            "num_ep_lengths": len(self.episode_lengths),
            "num_losses": len(self.losses),
            "num_avg_max_qs": len(self.avg_max_qs),  # <<< NEW
        }
        return summary

    def log_summary(self, global_step: int):
        """Logs summary to console if interval has passed."""
        # Check if enough steps have passed since the last log based on global_step
        if (
            global_step == 0
            or global_step < self.last_log_step + self.console_log_interval
        ):
            return  # Not time to log yet

        summary = self.get_summary(global_step)  # Calculate stats including SPS

        # Format the log string - Adjust precision as needed
        log_str = (
            f"[Stats] Step: {global_step:<8} | "
            f"Ep: {summary['total_episodes']:<6} | "
            f"SPS: {summary['steps_per_second']:<6.1f} | "
            # Avg Score (Score Window Size)
            f"Score({summary['num_ep_scores']}): {summary['avg_score_100']:<6.2f} | "
            # Best Score
            f"Best: {summary['best_score']:.2f} | "
            # Avg Length (Length Window Size)
            f"Len({summary['num_ep_lengths']}): {summary['avg_length_100']:.1f} | "
            # Avg Loss (Loss Window Size)
            f"Loss({summary['num_losses']}): {summary['avg_loss_100']:.4f} | "
            # Avg Max Q (Q Window Size) <<< NEW
            f"AvgMaxQ({summary['num_avg_max_qs']}): {summary['avg_max_q_100']:.3f} | "
            # Grad Norm (Grad Window Size)
            # f"Grad({len(self.grad_norms)}): {summary['avg_grad_100']:.3f} | " # Optional: uncomment if needed
            # Epsilon, Beta, Buffer Size
            f"Eps: {summary['epsilon']:.3f} | "
            f"Beta: {summary['beta']:.3f} | "
            f"Buf: {summary['buffer_size']}"
        )
        print(log_str)

        # Reset timing and step count *after* logging
        self.last_log_time = time.time()
        self.last_log_step = global_step
        # self.steps_since_last_log = 0 # Reset counter - Removed

    def close(self):
        """Cleanup for SimpleStatsRecorder (usually nothing needed)."""
        print("[SimpleStatsRecorder] Closed.")
        # If writing to a file, close it here
