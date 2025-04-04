# File: utils/types.py
from typing import NamedTuple, Union, Tuple, List, Dict, Any, Optional
import numpy as np
import torch

# --- MODIFIED: StateType is now Dict ---
# from environment.game_state import StateType # Import from env if defined there
StateType = Dict[str, np.ndarray]  # e.g., {"grid": ndarray, "shapes": ndarray}
# --- END MODIFIED ---

# Type alias for action
ActionType = int


class Transition(NamedTuple):
    state: StateType  # State is now a Dict
    action: ActionType
    reward: float
    next_state: StateType  # Next state is now a Dict
    done: bool
    n_step_discount: Optional[float] = None


# --- Batch Types (Numpy) ---
# States and next_states are now lists of dictionaries
NumpyStateBatch = List[StateType]
NumpyActionBatch = np.ndarray
NumpyRewardBatch = np.ndarray
NumpyDoneBatch = np.ndarray
NumpyDiscountBatch = np.ndarray  # For N-step

# Standard 1-step batch
NumpyBatch = Tuple[
    NumpyStateBatch, NumpyActionBatch, NumpyRewardBatch, NumpyStateBatch, NumpyDoneBatch
]

# N-step batch
NumpyNStepBatch = Tuple[
    NumpyStateBatch,
    NumpyActionBatch,
    NumpyRewardBatch,
    NumpyStateBatch,
    NumpyDoneBatch,
    NumpyDiscountBatch,
]

# Prioritized 1-step batch
PrioritizedNumpyBatch = Tuple[NumpyBatch, np.ndarray, np.ndarray]
# ((s_dicts, a, r, ns_dicts, d), indices, weights)

# Prioritized N-step batch
PrioritizedNumpyNStepBatch = Tuple[NumpyNStepBatch, np.ndarray, np.ndarray]
# ((s_dicts, a, rn, nsn_dicts, dn, gamman), indices, weights)


# --- Batch Types (Tensor) ---
# State tensors are now tuples (grid_tensor, shape_tensor)
TensorStateBatch = Tuple[torch.Tensor, torch.Tensor]
TensorActionBatch = torch.Tensor
TensorRewardBatch = torch.Tensor
TensorDoneBatch = torch.Tensor
TensorDiscountBatch = torch.Tensor  # For N-step
TensorWeightsBatch = torch.Tensor  # For PER

# Standard 1-step batch
TensorBatch = Tuple[
    TensorStateBatch,
    TensorActionBatch,
    TensorRewardBatch,
    TensorStateBatch,
    TensorDoneBatch,
]

# N-step batch
TensorNStepBatch = Tuple[
    TensorStateBatch,
    TensorActionBatch,
    TensorRewardBatch,
    TensorStateBatch,
    TensorDoneBatch,
    TensorDiscountBatch,
]


# --- Agent State ---
AgentStateDict = Dict[str, Any]  # Remains the same
