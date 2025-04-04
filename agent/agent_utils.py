# File: agent/agent_utils.py
import torch
import numpy as np
from typing import Union, Tuple, List, Dict
from utils.types import (
    StateType,
    NumpyBatch,
    NumpyNStepBatch,
    TensorBatch,
    TensorNStepBatch,
)
from config import EnvConfig


def np_batch_to_tensor(
    batch: Union[NumpyBatch, NumpyNStepBatch],
    is_n_step: bool,
    env_config: EnvConfig,
    device: torch.device,
) -> Union[TensorBatch, TensorNStepBatch]:
    """Converts numpy batch tuple (where states are dicts) to tensor tuple."""

    if is_n_step:
        states_dicts, actions, rewards, next_states_dicts, dones, discounts = batch
    else:
        states_dicts, actions, rewards, next_states_dicts, dones = batch[:5]
        discounts = None

    grid_states = np.array([s["grid"] for s in states_dicts], dtype=np.float32)
    shape_states = np.array([s["shapes"] for s in states_dicts], dtype=np.float32)
    grid_next_states = np.array(
        [ns["grid"] for ns in next_states_dicts], dtype=np.float32
    )
    shape_next_states = np.array(
        [ns["shapes"] for ns in next_states_dicts], dtype=np.float32
    )

    expected_channels = env_config.GRID_FEATURES_PER_CELL
    if (
        grid_states.shape[1] != expected_channels
        or grid_next_states.shape[1] != expected_channels
    ):
        raise ValueError(
            f"Batch grid state channel mismatch! Expected {expected_channels}, got {grid_states.shape[1]} and {grid_next_states.shape[1]}."
        )

    grid_s_t = torch.tensor(grid_states, device=device, dtype=torch.float32)
    shape_s_t = torch.tensor(shape_states, device=device, dtype=torch.float32)
    a_t = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)
    r_t = torch.tensor(rewards, device=device, dtype=torch.float32).unsqueeze(1)
    grid_ns_t = torch.tensor(grid_next_states, device=device, dtype=torch.float32)
    shape_ns_t = torch.tensor(shape_next_states, device=device, dtype=torch.float32)
    d_t = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(1)

    states_t = (grid_s_t, shape_s_t)
    next_states_t = (grid_ns_t, shape_ns_t)

    if is_n_step:
        disc_t = torch.tensor(discounts, device=device, dtype=torch.float32).unsqueeze(
            1
        )
        return states_t, a_t, r_t, next_states_t, d_t, disc_t
    else:
        return states_t, a_t, r_t, next_states_t, d_t
