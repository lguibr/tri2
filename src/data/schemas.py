# File: src/data/schemas.py
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from collections import deque
import numpy as np  # Import numpy

# Import Experience type hint carefully
from src.utils.types import Experience, StateType  # Import StateType as well

# Import GameState OUTSIDE TYPE_CHECKING for Pydantic model_rebuild
# Removed GameState import as it's no longer stored in BufferData

# Pydantic configuration to allow arbitrary types like torch.Tensor and deque
arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class CheckpointData(BaseModel):
    """Pydantic model defining the structure of saved checkpoint data."""

    model_config = arbitrary_types_config

    run_name: str
    global_step: int = Field(..., ge=0)
    episodes_played: int = Field(..., ge=0)
    total_simulations_run: int = Field(..., ge=0)
    model_config_dict: Dict[str, Any]
    env_config_dict: Dict[str, Any]
    model_state_dict: Dict[str, Any]  # Placeholder for torch tensors
    optimizer_state_dict: Dict[str, Any]  # Placeholder for optimizer state
    stats_collector_state: Dict[str, Any]  # Contains lists from deque conversion


class BufferData(BaseModel):
    """Pydantic model defining the structure of saved buffer data."""

    model_config = arbitrary_types_config

    # Store buffer as a list for better serialization compatibility
    # The Experience tuple now contains StateType (which has numpy arrays)
    buffer_list: List[Experience]


# Type hint for the combined loaded state, using the Pydantic models
class LoadedTrainingState(BaseModel):
    """Pydantic model representing the fully loaded state."""

    model_config = arbitrary_types_config

    checkpoint_data: Optional[CheckpointData] = None
    buffer_data: Optional[BufferData] = None


# Crucial step: Explicitly rebuild models to resolve forward references
# This is needed because Experience -> StateType -> np.ndarray
# Pydantic needs help resolving this.
BufferData.model_rebuild(force=True)
LoadedTrainingState.model_rebuild(force=True)
