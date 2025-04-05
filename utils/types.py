# File: utils/types.py
from typing import NamedTuple, Union, Tuple, List, Dict, Any, Optional
import numpy as np
import torch

StateType = Dict[str, np.ndarray]
ActionType = int
AgentStateDict = Dict[str, Any]
