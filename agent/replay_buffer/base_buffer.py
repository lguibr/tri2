# File: agent/replay_buffer/base_buffer.py
# (No changes needed)
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict
import numpy as np
from utils.types import StateType, ActionType


class ReplayBufferBase(ABC):
    """Abstract base class for all replay buffers."""

    def __init__(self, capacity: int):
        self.capacity = capacity

    @abstractmethod
    def push(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool,
        **kwargs  # Allow passing extra info like n_step_discount
    ):
        """Add a new experience to the buffer."""
        pass

    @abstractmethod
    def sample(
        self, batch_size: int
    ) -> Optional[Any]:  # Return type depends on PER/NStep
        """Sample a batch of experiences from the buffer."""
        pass

    @abstractmethod
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for PER (no-op for uniform buffer)."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        pass

    @abstractmethod
    def set_beta(self, beta: float):
        """Set the beta value for PER IS weights (no-op for uniform buffer)."""
        pass

    @abstractmethod
    def flush_pending(self):
        """Process any pending transitions (e.g., for N-step)."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return the buffer's state as a dictionary suitable for saving."""
        pass

    @abstractmethod
    def load_state_from_data(self, state: Dict[str, Any]):
        """Load the buffer's state from a dictionary."""
        pass

    @abstractmethod
    def save_state(self, filepath: str):
        """Save the buffer's state to a file."""
        pass

    @abstractmethod
    def load_state(self, filepath: str):
        """Load the buffer's state from a file."""
        pass
