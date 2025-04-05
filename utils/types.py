# File: utils/types.py
from typing import NamedTuple, Union, Tuple, List, Dict, Any, Optional
import numpy as np
import torch

# --- State represented as a dictionary ---
StateType = Dict[str, np.ndarray]  # e.g., {"grid": ndarray, "shapes": ndarray}

# Type alias for action
ActionType = int


# --- Transition Tuple ---
class Transition(NamedTuple):
    state: StateType
    action: ActionType
    reward: float
    next_state: StateType
    done: bool
    n_step_discount: Optional[float] = None  # Store gamma^n for N-step


# --- Batch Types (Numpy) ---
# State batches are now lists of dictionaries
NumpyStateBatch = List[StateType]
NumpyActionBatch = np.ndarray
NumpyRewardBatch = np.ndarray
NumpyDoneBatch = np.ndarray
NumpyDiscountBatch = np.ndarray  # For N-step discount factor (gamma^n)

# Standard 1-step batch
NumpyBatch = Tuple[
    NumpyStateBatch, NumpyActionBatch, NumpyRewardBatch, NumpyStateBatch, NumpyDoneBatch
]

# N-step batch
NumpyNStepBatch = Tuple[
    NumpyStateBatch,
    NumpyActionBatch,
    NumpyRewardBatch,  # N-step rewards
    NumpyStateBatch,  # State after N steps
    NumpyDoneBatch,  # Done flag after N steps
    NumpyDiscountBatch,  # Effective discount gamma^n
]

# Prioritized 1-step batch
PrioritizedNumpyBatch = Tuple[
    NumpyBatch,  # (states_dicts, actions, rewards, next_states_dicts, dones)
    np.ndarray,  # indices (tree indices for PER update)
    np.ndarray,  # is_weights (importance sampling weights)
]

# Prioritized N-step batch
PrioritizedNumpyNStepBatch = Tuple[
    NumpyNStepBatch,  # (states_dicts, actions, rewards_n, next_states_dicts_n, dones_n, discounts_n)
    np.ndarray,  # indices
    np.ndarray,  # is_weights
]


# --- Batch Types (Tensor) ---
# State tensors are tuples (grid_tensor, shape_tensor) after conversion
TensorStateBatch = Tuple[torch.Tensor, torch.Tensor]
TensorActionBatch = torch.Tensor
TensorRewardBatch = torch.Tensor
TensorDoneBatch = torch.Tensor
TensorDiscountBatch = torch.Tensor  # For N-step discount factor
TensorWeightsBatch = torch.Tensor  # For PER IS weights

# Standard 1-step batch (Tensor)
TensorBatch = Tuple[
    TensorStateBatch,
    TensorActionBatch,
    TensorRewardBatch,
    TensorStateBatch,
    TensorDoneBatch,
]

# N-step batch (Tensor)
TensorNStepBatch = Tuple[
    TensorStateBatch,
    TensorActionBatch,
    TensorRewardBatch,
    TensorStateBatch,
    TensorDoneBatch,
    TensorDiscountBatch,
]


# --- Agent State Dictionary ---
AgentStateDict = Dict[str, Any]  # For saving/loading agent checkpoints
