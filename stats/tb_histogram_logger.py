# File: stats/tb_histogram_logger.py
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from typing import Union, List, Optional
import numpy as np


class TBHistogramLogger:
    """Handles logging histograms to TensorBoard based on frequency."""

    def __init__(
        self, writer: Optional[SummaryWriter], lock: threading.Lock, log_interval: int
    ):
        self.writer = writer
        self._lock = lock
        self.log_interval = log_interval  # Interval in terms of updates/rollouts
        self.last_log_step = -1
        self.rollouts_since_last_log = 0  # Track rollouts internally

    def should_log(self, global_step: int) -> bool:
        """Checks if enough rollouts have passed and the global step has advanced."""
        if not self.writer or self.log_interval <= 0:
            return False
        # Check based on internal rollout counter and if step has advanced
        return (
            self.rollouts_since_last_log >= self.log_interval
            and global_step > self.last_log_step
        )

    def log_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Logs a histogram if the interval condition is met."""
        if not self.should_log(global_step):
            # Increment counter even if not logging this specific histogram
            # This assumes record_step increments the counter externally
            # Let's manage the counter internally based on when log_histogram is called
            # self.rollouts_since_last_log += 1 # No, let external call manage this
            return

        with self._lock:
            # Double check condition inside lock
            if global_step > self.last_log_step:
                try:
                    self.writer.add_histogram(tag, values, global_step)
                    self.last_log_step = global_step
                    # Reset counter after successful log
                    # self.rollouts_since_last_log = 0 # Resetting counter is handled in record_step
                except Exception as e:
                    print(f"Error logging histogram '{tag}': {e}")

    def increment_rollout_counter(self):
        """Increments the internal counter, called after each update/rollout."""
        if self.log_interval > 0:
            self.rollouts_since_last_log += 1

    def reset_rollout_counter(self):
        """Resets the counter, called after logging."""
        self.rollouts_since_last_log = 0
