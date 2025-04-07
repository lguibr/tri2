# File: training/normalization.py
import numpy as np
from typing import Optional

from utils.running_mean_std import RunningMeanStd


def update_obs_rms(obs_batch: np.ndarray, rms_instance: Optional[RunningMeanStd]):
    """Update the running mean/std for a given observation key if enabled."""
    if rms_instance is not None:
        rms_instance.update(obs_batch)


def normalize_obs(
    obs_batch: np.ndarray, rms_instance: Optional[RunningMeanStd], clip_value: float
) -> np.ndarray:
    """Normalize observations using running mean/std if enabled."""
    if rms_instance is not None:
        normalized_obs = rms_instance.normalize(obs_batch)
        clipped_obs = np.clip(normalized_obs, -clip_value, clip_value)
        return clipped_obs.astype(np.float32)
    else:
        return obs_batch.astype(np.float32)
