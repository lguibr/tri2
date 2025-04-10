# File: src/utils/types.py
from typing import Dict, Any, List, Tuple, Mapping, TypedDict, Optional, Deque
import numpy as np
# Removed Pydantic imports as they are no longer needed here for SelfPlayResult
# from pydantic import BaseModel, Field, ConfigDict

# Import GameState for Experience type hint - use TYPE_CHECKING to avoid circular import at runtime
from typing import TYPE_CHECKING
from collections import deque  # Import deque for StatsCollectorData

if TYPE_CHECKING:
    # Import GameState ONLY for type checking
    from src.environment import GameState


# Basic type for representing the dictionary structure of features extracted from a state
# This is the *output* of the feature extractor and the *input* to the NN.
# Kept as TypedDict for performance in NN/feature extraction path.
class StateType(TypedDict):
    grid: np.ndarray  # (C, H, W) float32
    other_features: np.ndarray  # (OtherFeatDim,) float32


# Action representation (integer index)
ActionType = int

# Policy target from MCTS (visit counts distribution)
# Mapping from action index to its probability (normalized visit count)
PolicyTargetMapping = Mapping[ActionType, float]

# Experience tuple stored in buffer
# Now stores the raw GameState object instead of extracted features.
# Kept as Tuple for performance in buffer operations.
# Use forward reference string "GameState" to avoid runtime import
Experience = Tuple["GameState", PolicyTargetMapping, float]

# Batch of experiences for training
ExperienceBatch = List[Experience]

# Output type from the neural network's evaluate method
# (Policy Mapping, Value Estimate)
# Kept as Tuple for performance.
PolicyValueOutput = Tuple[Mapping[ActionType, float], float]

# Type alias for the data structure holding collected statistics
# Maps metric name to a deque of (step, value) tuples
# Kept as Dict[Deque] internally in StatsCollectorActor, type alias is sufficient here.
StatsCollectorData = Dict[str, Deque[Tuple[int, float]]]

# --- Pydantic Models for Data Transfer ---
# SelfPlayResult moved to src/rl/types.py to resolve circular import

# No model_rebuild needed here anymore