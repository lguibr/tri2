# File: stats/stats_recorder.py
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Dict, Any, Optional, Union  # Added Union
import numpy as np
import torch  # Added torch for Tensor type hints


class StatsRecorderBase(ABC):
    """Base class for recording training statistics."""

    @abstractmethod
    def record_episode(
        self,
        episode_score: float,  # RL Score
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        lines_cleared: Optional[int] = None,
    ):
        """Record stats for a completed episode."""
        pass

    @abstractmethod
    def record_step(self, step_data: Dict[str, Any]):
        """Record stats from a training or environment step (e.g., loss, rewards)."""
        pass

    # --- New Abstract Methods ---
    @abstractmethod
    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Record a histogram of values."""
        pass

    @abstractmethod
    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Record an image."""
        pass

    @abstractmethod
    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Record hyperparameters and final/key metrics."""
        pass

    @abstractmethod
    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[torch.Tensor] = None
    ):
        """Record the model graph."""
        pass

    # --- End New ---

    @abstractmethod
    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Return a dictionary containing summary statistics (usually averaged)."""
        pass

    @abstractmethod
    def log_summary(self, global_step: int):
        """Trigger the logging action (e.g., print to console, write to TensorBoard)."""
        pass

    @abstractmethod
    def close(self):
        """Perform any necessary cleanup (e.g., close files/writers)."""
        pass
