# File: src/utils/helpers.py
import torch
import numpy as np
import random
import os
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_device(device_preference: str = "auto") -> torch.device:
    """Gets the appropriate torch device based on preference and availability."""
    if device_preference == "cuda" and torch.cuda.is_available():
        logger.info("Using CUDA device.")
        return torch.device("cuda")
    # Note: MPS backend check might differ slightly across torch versions
    if (
        device_preference == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        logger.info("Using MPS device.")
        return torch.device("mps")
    if device_preference == "cpu":
        logger.info("Using CPU device.")
        return torch.device("cpu")

    # Auto selection priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        logger.info("Auto-selected CUDA device.")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Auto-selected MPS device.")
        return torch.device("mps")

    logger.info("Auto-selected CPU device.")
    return torch.device("cpu")


def set_random_seeds(seed: int = 42):
    """Sets random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # Add MPS seed setting if needed and available
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     torch.mps.manual_seed(seed) # Check correct function if needed
    logger.info(f"Set random seeds to {seed}")


def format_eta(seconds: Optional[float]) -> str:
    """Formats seconds into a human-readable HH:MM:SS or MM:SS string."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "N/A"
    if seconds > 3600 * 24 * 30:  # Arbitrary limit for very long times
        return ">1 month"

    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"
