# File: config/utils.py
# --- Configuration Utilities ---
import torch
from typing import Dict, Any
from .core import (
    VisConfig,
    EnvConfig,
    RewardConfig,
    DQNConfig,
    TrainConfig,
    BufferConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
)
from .general import DEVICE, RANDOM_SEED, RUN_ID


def get_config_dict() -> Dict[str, Any]:
    """Returns a flat dictionary of all relevant config values for logging."""
    all_configs = {}

    # Helper to flatten classes (excluding methods, dunders, ClassVars, nested Classes)
    def flatten_class(cls, prefix=""):
        d = {}
        for k, v in cls.__dict__.items():
            # Basic check: not dunder, not callable, not a type (nested class)
            if not k.startswith("__") and not callable(v) and not isinstance(v, type):
                # More robust checking might be needed for complex configs
                d[f"{prefix}{k}"] = v
        return d

    all_configs.update(flatten_class(VisConfig, "Vis."))
    all_configs.update(flatten_class(EnvConfig, "Env."))
    all_configs.update(flatten_class(RewardConfig, "Reward."))
    all_configs.update(flatten_class(DQNConfig, "DQN."))
    all_configs.update(flatten_class(TrainConfig, "Train."))
    all_configs.update(flatten_class(BufferConfig, "Buffer."))
    all_configs.update(
        flatten_class(ModelConfig.Network, "Model.Net.")
    )  # Flatten sub-class explicitly
    all_configs.update(flatten_class(StatsConfig, "Stats."))
    all_configs.update(flatten_class(TensorBoardConfig, "TB."))

    # Add general constants
    all_configs["General.DEVICE"] = str(DEVICE)
    all_configs["General.RANDOM_SEED"] = RANDOM_SEED
    all_configs["General.RUN_ID"] = RUN_ID

    # Filter out None values specifically for paths that might not be set
    # This prevents logging None for load paths if they aren't specified
    all_configs = {
        k: v for k, v in all_configs.items() if not (k.endswith("_PATH") and v is None)
    }

    # Convert torch.nn activation functions to string representation for logging
    for key, value in all_configs.items():
        if isinstance(value, type) and issubclass(value, torch.nn.Module):
            all_configs[key] = value.__name__
        # Handle potential list values (e.g., CONV_CHANNELS) - convert to string?
        # Tensorboard hparams prefers simple types (int, float, bool, str)
        if isinstance(value, list):
            all_configs[key] = str(value)

    return all_configs
