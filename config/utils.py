# File: config/utils.py
import torch
from typing import Dict, Any
from .core import (
    VisConfig,
    EnvConfig,
    RewardConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    DemoConfig,
)
from .general import DEVICE, RANDOM_SEED, RUN_ID


def get_config_dict() -> Dict[str, Any]:
    """Returns a flat dictionary of all relevant config values for logging."""
    all_configs = {}

    def flatten_class(cls, prefix=""):
        d = {}
        for k, v in vars(cls).items():
            if (
                not k.startswith("__")
                and not callable(v)
                and not isinstance(v, type)
                and not hasattr(v, "__module__")
            ):
                if isinstance(getattr(cls, k, None), property):
                    try:
                        v = getattr(cls(), k)
                    except Exception:
                        continue
                d[f"{prefix}{k}"] = v
        return d

    all_configs.update(flatten_class(VisConfig, "Vis."))
    all_configs.update(flatten_class(EnvConfig, "Env."))
    all_configs.update(flatten_class(RewardConfig, "Reward."))
    all_configs.update(flatten_class(PPOConfig, "PPO."))
    all_configs.update(flatten_class(RNNConfig, "RNN."))
    all_configs.update(flatten_class(TrainConfig, "Train."))
    all_configs.update(flatten_class(ModelConfig.Network, "Model.Net."))
    all_configs.update(flatten_class(StatsConfig, "Stats."))
    all_configs.update(flatten_class(TensorBoardConfig, "TB."))
    all_configs.update(flatten_class(DemoConfig, "Demo."))

    all_configs["General.DEVICE"] = str(DEVICE)
    all_configs["General.RANDOM_SEED"] = RANDOM_SEED
    all_configs["General.RUN_ID"] = RUN_ID

    all_configs = {
        k: v for k, v in all_configs.items() if not (k.endswith("_PATH") and v is None)
    }

    for key, value in all_configs.items():
        if isinstance(value, type) and issubclass(value, torch.nn.Module):
            all_configs[key] = value.__name__
        elif isinstance(value, (list, tuple)):
            all_configs[key] = str(value)
        if not isinstance(value, (int, float, str, bool)):
            all_configs[key] = str(value)

    return all_configs
