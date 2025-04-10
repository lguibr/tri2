# File: src/rl/core/buffer.py
import random
import logging
from collections import deque
from typing import List, Optional

# Use core types - Experience now contains GameState
from ...utils.types import Experience, ExperienceBatch, PolicyTargetMapping
from ...config import TrainConfig
from ...environment import GameState  # Keep GameState import

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """Simple FIFO Experience Replay Buffer storing (GameState, PolicyTarget, Value)."""

    def __init__(self, config: TrainConfig):
        self.capacity = config.BUFFER_CAPACITY
        self.min_size_to_train = config.MIN_BUFFER_SIZE_TO_TRAIN
        # Store the deque itself - type hint updated to Experience
        self.buffer: deque[Experience] = deque(maxlen=self.capacity)
        logger.info(f"Experience buffer initialized with capacity {self.capacity}.")

    def add(self, experience: Experience):
        """Adds a single experience tuple (GameState, PolicyTarget, Value) to the buffer."""
        self.buffer.append(experience)

    def add_batch(self, experiences: List[Experience]):
        """Adds a batch of experiences to the buffer."""
        self.buffer.extend(experiences)

    def sample(self, batch_size: int) -> Optional[ExperienceBatch]:
        """Samples a batch of experiences uniformly from the buffer."""
        current_size = len(self.buffer)
        if current_size < batch_size or current_size < self.min_size_to_train:
            return None
        batch = random.sample(self.buffer, batch_size)
        return batch  # Type hint ExperienceBatch already matches List[Experience]

    def __len__(self) -> int:
        """Returns the current number of experiences in the buffer."""
        return len(self.buffer)

    def is_ready(self) -> bool:
        """Checks if the buffer has enough samples to start training."""
        return len(self.buffer) >= self.min_size_to_train
