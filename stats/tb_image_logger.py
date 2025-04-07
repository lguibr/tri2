# File: stats/tb_image_logger.py
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from typing import Union, Optional
import numpy as np
import traceback

from .tb_log_utils import format_image_for_tb


class TBImageLogger:
    """Handles logging images to TensorBoard based on frequency."""

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
        return (
            self.rollouts_since_last_log >= self.log_interval
            and global_step > self.last_log_step
        )

    def log_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Logs an image if the interval condition is met."""
        if not self.should_log(global_step):
            # self.rollouts_since_last_log += 1 # No, let external call manage this
            return

        with self._lock:
            # Double check condition inside lock
            if global_step > self.last_log_step:
                try:
                    image_tensor = format_image_for_tb(image)
                    self.writer.add_image(
                        tag, image_tensor, global_step, dataformats="CHW"
                    )
                    self.last_log_step = global_step
                    # self.rollouts_since_last_log = 0 # Resetting counter is handled in record_step
                except Exception as e:
                    print(f"Error logging image '{tag}': {e}")
                    # traceback.print_exc() # Optional: more detail

    def increment_rollout_counter(self):
        """Increments the internal counter, called after each update/rollout."""
        if self.log_interval > 0:
            self.rollouts_since_last_log += 1

    def reset_rollout_counter(self):
        """Resets the counter, called after logging."""
        self.rollouts_since_last_log = 0
