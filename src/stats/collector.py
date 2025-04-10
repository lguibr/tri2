# File: src/stats/collector.py
import logging
import ray
from collections import deque
from typing import Dict, List, Tuple, Optional, Deque, Any
import numpy as np

# Import type alias from utils
from src.utils.types import StatsCollectorData

# Use print for actor logging
# logger = logging.getLogger(__name__)

@ray.remote
class StatsCollectorActor:
    """Ray actor for collecting time-series statistics."""

    def __init__(self, max_history: Optional[int] = 1000):
        self.max_history = max_history
        self._data: StatsCollectorData = {}
        print(f"[StatsCollectorActor] Initialized with max_history={max_history}.")

    def log(self, metric_name: str, value: float, step: int):
        """Logs a single metric value."""
        if not isinstance(metric_name, str):
            print(f"[StatsCollectorActor] ERROR: Invalid metric_name type: {type(metric_name)}")
            return
        if not np.isfinite(value):
            print(f"[StatsCollectorActor] WARNING: Received non-finite value for metric '{metric_name}': {value}. Skipping log.")
            return
        if metric_name not in self._data:
            self._data[metric_name] = deque(maxlen=self.max_history)
        try:
            self._data[metric_name].append((int(step), float(value)))
        except (ValueError, TypeError) as e:
            print(f"[StatsCollectorActor] ERROR: Could not log metric '{metric_name}'. Invalid step/value: {e}")

    def log_batch(self, metrics: Dict[str, Tuple[float, int]]):
        """Logs a batch of metrics."""
        for name, (value, step) in metrics.items():
            self.log(name, value, step)

    def get_data(self) -> StatsCollectorData:
        """Returns a copy of the collected statistics data."""
        return self._data.copy()

    def get_metric_data(self, metric_name: str) -> Optional[Deque[Tuple[int, float]]]:
        """Returns the data deque for a specific metric."""
        return self._data.get(metric_name)

    def clear(self):
        """Clears all collected statistics."""
        self._data = {}
        print("[StatsCollectorActor] Data cleared.")

    def get_state(self) -> Dict[str, Any]:
        """Returns the internal state for saving (converts deques to lists)."""
        serializable_data = {key: list(dq) for key, dq in self._data.items()}
        state = {'max_history': self.max_history, '_data_list': serializable_data}
        # print(f"[StatsCollectorActor] get_state called. Returning state for {len(serializable_data)} metrics.")
        return state

    def set_state(self, state: Dict[str, Any]):
        """Restores the internal state from saved data (expects lists)."""
        # print("[StatsCollectorActor] set_state called.")
        self.max_history = state.get('max_history', self.max_history)
        loaded_data_list = state.get('_data_list', {})
        self._data = {}
        restored_count = 0
        for key, items_list in loaded_data_list.items():
            # Ensure items_list is actually a list of tuples before creating deque
            if isinstance(items_list, list) and all(isinstance(item, tuple) and len(item) == 2 for item in items_list):
                self._data[key] = deque(items_list, maxlen=self.max_history)
                restored_count += 1
            else:
                print(f"[StatsCollectorActor] WARNING: Skipping restore for metric '{key}'. Invalid data format: {type(items_list)}")

        print(f"[StatsCollectorActor] State restored. Restored {restored_count} metrics. Max history: {self.max_history}")