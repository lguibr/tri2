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

    # --- MODIFIED: Unpack based on is_n_step flag ---
    if is_n_step:
        # Ensure the batch has 6 elements for N-step
        if len(batch) != 6:
            raise ValueError(f"Expected 6 elements in N-step batch, got {len(batch)}")
        states_dicts, actions, rewards, next_states_dicts, dones, discounts = batch
    else:
        # Ensure the batch has 5 elements for 1-step
        if len(batch) != 5:
            raise ValueError(f"Expected 5 elements in 1-step batch, got {len(batch)}")
        states_dicts, actions, rewards, next_states_dicts, dones = batch
        discounts = None  # No discount element in 1-step batch
    # --- END MODIFIED ---

    # --- Input Validation and Conversion (Remains the same) ---
    if not isinstance(states_dicts, list) or not isinstance(next_states_dicts, list):
        raise TypeError("States and next_states must be lists of dictionaries.")
    if not states_dicts or not next_states_dicts:
        raise ValueError("State lists cannot be empty.")

    # Validate first state dictionary structure
    first_state = states_dicts[0]
    if (
        not isinstance(first_state, dict)
        or "grid" not in first_state
        or "shapes" not in first_state
    ):
        raise ValueError("State dictionaries must contain 'grid' and 'shapes' keys.")

    try:
        grid_states = np.array([s["grid"] for s in states_dicts], dtype=np.float32)
        shape_states = np.array([s["shapes"] for s in states_dicts], dtype=np.float32)
        grid_next_states = np.array(
            [ns["grid"] for ns in next_states_dicts], dtype=np.float32
        )
        shape_next_states = np.array(
            [ns["shapes"] for ns in next_states_dicts], dtype=np.float32
        )
    except KeyError as e:
        raise ValueError(
            f"Missing key in state dictionary during batch conversion: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Error converting states to numpy arrays: {e}")

    expected_channels = env_config.GRID_FEATURES_PER_CELL
    if (
        grid_states.shape[1] != expected_channels
        or grid_next_states.shape[1] != expected_channels
    ):
        raise ValueError(
            f"Batch grid state channel mismatch! Expected {expected_channels}, got {grid_states.shape[1]} and {grid_next_states.shape[1]}."
        )

    try:
        grid_s_t = torch.tensor(grid_states, device=device, dtype=torch.float32)
        shape_s_t = torch.tensor(shape_states, device=device, dtype=torch.float32)
        a_t = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)
        r_t = torch.tensor(rewards, device=device, dtype=torch.float32).unsqueeze(1)
        grid_ns_t = torch.tensor(grid_next_states, device=device, dtype=torch.float32)
        shape_ns_t = torch.tensor(shape_next_states, device=device, dtype=torch.float32)
        # Convert dones to float for multiplication in loss calculation
        d_t = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(1)
    except Exception as e:
        raise RuntimeError(f"Error converting numpy arrays to tensors: {e}")

    states_t = (grid_s_t, shape_s_t)
    next_states_t = (grid_ns_t, shape_ns_t)

    # --- MODIFIED: Return based on is_n_step ---
    if is_n_step:
        if discounts is None:
            raise ValueError("Discounts array is missing for N-step batch.")
        try:
            disc_t = torch.tensor(
                discounts, device=device, dtype=torch.float32
            ).unsqueeze(1)
        except Exception as e:
            raise RuntimeError(f"Error converting discounts to tensor: {e}")
        return states_t, a_t, r_t, next_states_t, d_t, disc_t
    else:
        return states_t, a_t, r_t, next_states_t, d_t
    # --- END MODIFIED ---
