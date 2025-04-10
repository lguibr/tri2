# File: src/rl/core/visual_state_actor.py
import ray
import time
from typing import Dict, Optional, Any, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from src.environment import GameState  # Import GameState for type hint


@ray.remote
class VisualStateActor:
    """A simple Ray actor to hold the latest game states from workers for visualization."""

    def __init__(self):
        self.worker_states: Dict[int, "GameState"] = {}
        self.global_stats: Dict[str, Any] = {}
        self.last_update_times: Dict[int, float] = {}

    def update_state(self, worker_id: int, game_state: "GameState"):
        """Workers call this to update their latest state."""
        # Store a copy to prevent issues with shared references if needed,
        # but direct reference might be faster if visualizer is read-only.
        # Let's store the reference for now as copy was removed from worker.
        self.worker_states[worker_id] = game_state
        self.last_update_times[worker_id] = time.time()

    def update_global_stats(self, stats: Dict[str, Any]):
        """Orchestrator calls this to update global stats."""
        self.global_stats = stats

    def get_all_states(self) -> Dict[int, Any]:
        """Called by the orchestrator to get states for the visual queue."""
        # Return a copy to avoid race conditions in the orchestrator/visualizer
        combined_states = {wid: state for wid, state in self.worker_states.items()}
        # Add global stats under key -1
        combined_states[-1] = self.global_stats.copy()
        return combined_states

    def get_state(self, worker_id: int) -> Optional["GameState"]:
        """Get state for a specific worker (unused currently)."""
        return self.worker_states.get(worker_id)
