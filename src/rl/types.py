# File: src/rl/types.py
from typing import List
from pydantic import BaseModel, ConfigDict, Field

# Import types needed for the model definition
from src.utils.types import (
    Experience,
)  # Import Experience tuple type (now contains StateType)
from src.environment import GameState  # Import GameState directly

# Pydantic configuration to allow arbitrary types like GameState
arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class SelfPlayResult(BaseModel):
    """Pydantic model for structuring results from a self-play worker."""

    model_config = arbitrary_types_config

    # This list now contains Experience tuples where the first element is StateType
    episode_experiences: List[Experience]
    final_score: float
    episode_steps: int
    final_game_state: GameState  # Use GameState directly here

    # Add aggregated stats from the episode
    total_simulations: int = Field(..., ge=0)
    avg_root_visits: float = Field(..., ge=0)
    avg_tree_depth: float = Field(..., ge=0)


# Crucial step: Explicitly rebuild models to resolve forward references
# This is needed because Experience -> StateType -> np.ndarray
# Pydantic needs help resolving this.
SelfPlayResult.model_rebuild(force=True)
