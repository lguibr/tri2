import torch
import numpy as np
import random
import os
from typing import Union, Tuple, Optional, Any
import pickle
import cloudpickle


def get_device() -> torch.device:
    """Gets the appropriate torch device (CUDA if available, else CPU)."""
    force_cpu = os.environ.get("FORCE_CPU", "false").lower() == "true"
    if force_cpu:
        print("Forcing CPU device based on environment variable.")
        return torch.device("cpu")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str.upper()}")
    if device_str == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    return torch.device(device_str)


def set_random_seeds(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Set random seeds to {seed}")


def ensure_numpy(
    data: Union[np.ndarray, list, tuple, torch.Tensor],
) -> np.ndarray:
    """
    Ensures the input data is a numpy array with float32 type.
    Handles numpy arrays, lists, tuples, and torch Tensors.
    """
    try:
        if isinstance(data, np.ndarray):
            if data.dtype != np.float32:
                return data.astype(np.float32)
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float32)
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=np.float32)
            if arr.dtype == np.object_:
                raise ValueError(
                    "Cannot convert ragged list/tuple to float32 numpy array."
                )
            return arr
        else:
            raise TypeError(f"Unsupported type for ensure_numpy: {type(data)}")

    except (
        ValueError,
        TypeError,
        RuntimeError,
    ) as e:
        print(
            f"Warning: ensure_numpy failed conversion: {e}. Input type: {type(data)}. Returning empty array."
        )
        raise ValueError(f"ensure_numpy failed: {e}") from e


def save_object(obj: Any, filepath: str):
    """Saves an arbitrary Python object to a file using cloudpickle."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            cloudpickle.dump(obj, f)
    except Exception as e:
        print(f"Error saving object to {filepath}: {e}")
        raise e


def load_object(filepath: str) -> Any:
    """Loads a Python object from a file using cloudpickle."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        with open(filepath, "rb") as f:
            obj = cloudpickle.load(f)
        return obj
    except Exception as e:
        print(f"Error loading object from {filepath}: {e}")
        raise e
