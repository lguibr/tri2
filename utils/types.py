from typing import NamedTuple, Union, Tuple, List, Dict, Any, Optional
import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float  # For N-step buffer, this holds the N-step reward
    next_state: np.ndarray  # For N-step buffer, this holds the N-step next state
    done: bool  # For N-step buffer, this holds the N-step done flag
    n_step_discount: Optional[float] = None


# Type alias for state
StateType = np.ndarray
# Type alias for action
ActionType = int

NumpyBatch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

NumpyNStepBatch = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]

PrioritizedNumpyBatch = Tuple[
    NumpyBatch, np.ndarray, np.ndarray
]  # ((s,a,r,ns,d), indices, weights)

PrioritizedNumpyNStepBatch = Tuple[
    NumpyNStepBatch, np.ndarray, np.ndarray
]  # ((s,a,rn,nsn,dn,gamman), indices, weights)

TensorBatch = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

TensorNStepBatch = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

AgentStateDict = Dict[str, Any]
