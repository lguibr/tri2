# File: src/utils/types.py
from typing import Dict, Any, List, Tuple, Mapping, Optional, Deque

# Change: Import TypedDict from typing_extensions
from typing_extensions import TypedDict
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
# NOW stores the extracted StateType (features) instead of the raw GameState object.
# Kept as Tuple for performance in buffer operations.
Experience = Tuple[StateType, PolicyTargetMapping, float]

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


# --- Prioritized Experience Replay Types ---
# TypedDict for the output of the PER buffer's sample method
class PERBatchSample(TypedDict):
    batch: ExperienceBatch
    indices: np.ndarray  # Indices of the sampled experiences in the buffer
    weights: np.ndarray  # Importance sampling weights for each experience
