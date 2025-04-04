# File: utils/types.py
from typing import NamedTuple, Union, Tuple, List, Dict, Any, Optional
import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float  # For N-step buffer, this holds the N-step RL reward
    next_state: np.ndarray  # For N-step buffer, this holds the N-step next state
    done: bool  # For N-step buffer, this holds the N-step done flag
    n_step_discount: Optional[float] = None  # gamma^k for N-step


# Type alias for state
StateType = np.ndarray
# Type alias for action
ActionType = int

# --- Batch Types (Numpy) ---
# Standard 1-step batch
NumpyBatch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (states, actions, rewards, next_states, dones)

# N-step batch (includes discount factor gamma^k)
NumpyNStepBatch = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
# (states, actions, n_step_rewards, n_step_next_states, n_step_dones, n_step_discounts)

# Prioritized 1-step batch
PrioritizedNumpyBatch = Tuple[NumpyBatch, np.ndarray, np.ndarray]
# ((s,a,r,ns,d), indices, weights)

# Prioritized N-step batch
PrioritizedNumpyNStepBatch = Tuple[NumpyNStepBatch, np.ndarray, np.ndarray]
# ((s,a,rn,nsn,dn,gamman), indices, weights)


# --- Batch Types (Tensor) ---
# Standard 1-step batch
TensorBatch = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
# (states, actions, rewards, next_states, dones)

# N-step batch (includes discount factor gamma^k)
TensorNStepBatch = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
# (states, actions, n_step_rewards, n_step_next_states, n_step_dones, n_step_discounts)


# --- Agent State ---
AgentStateDict = Dict[str, Any]
