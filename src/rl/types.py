# File: src/rl/types.py
# NEW FILE
from typing import List
from pydantic import BaseModel, ConfigDict

# Import types needed for the model definition
from src.utils.types import Experience # Import Experience tuple type
from src.environment import GameState # Import GameState directly

# Pydantic configuration to allow arbitrary types like GameState
arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)

class SelfPlayResult(BaseModel):
    """Pydantic model for structuring results from a self-play worker."""
    model_config = arbitrary_types_config

    episode_experiences: List[Experience]
    final_score: float
    episode_steps: int
    final_game_state: GameState # Use GameState directly here

# Crucial step: Explicitly rebuild models to resolve forward references
# This is needed because Experience -> GameState is a forward reference
# that Pydantic needs help resolving in this context.
# Now that GameState is imported directly, this should work.
SelfPlayResult.model_rebuild(force=True)