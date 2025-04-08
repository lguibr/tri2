# File: config/utils.py
import torch
from typing import Dict, Any
from .core import (
    VisConfig,
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    DemoConfig,
    TransformerConfig,
    MCTSConfig,  # Import MCTSConfig from core
)

# Import DEVICE and RANDOM_SEED directly, use get_run_id() for the run ID
from .general import DEVICE, RANDOM_SEED, get_run_id


def get_config_dict() -> Dict[str, Any]:
    """Returns a flat dictionary of all relevant config values for logging."""
    all_configs = {}

    def flatten_class(cls, prefix=""):
        d = {}
        instance = None
        try:
            instance = cls()
        except Exception:
            instance = None

        for k, v in vars(cls).items():
            if (
                not k.startswith("__")
                and not callable(v)
                and not isinstance(v, type)
                and not hasattr(v, "__module__")
            ):
                is_property = isinstance(getattr(cls, k, None), property)
                if is_property and instance:
                    try:
                        v = getattr(instance, k)
                    except Exception:
                        continue
                elif is_property and not instance:
                    continue
                d[f"{prefix}{k}"] = v
        return d

    # Flatten core config classes
    all_configs.update(flatten_class(VisConfig, "Vis."))
    all_configs.update(flatten_class(EnvConfig, "Env."))
    all_configs.update(flatten_class(RNNConfig, "RNN."))
    all_configs.update(flatten_class(TrainConfig, "Train."))
    all_configs.update(flatten_class(ModelConfig.Network, "Model.Net."))
    all_configs.update(flatten_class(StatsConfig, "Stats."))
    all_configs.update(flatten_class(TensorBoardConfig, "TB."))
    all_configs.update(flatten_class(DemoConfig, "Demo."))
    all_configs.update(flatten_class(TransformerConfig, "Transformer."))
    all_configs.update(flatten_class(MCTSConfig, "MCTS."))  # Flatten MCTSConfig

    # Add general config values
    all_configs["General.DEVICE"] = str(DEVICE) if DEVICE else "None"
    all_configs["General.RANDOM_SEED"] = RANDOM_SEED
    all_configs["General.RUN_ID"] = get_run_id()

    all_configs = {
        k: v for k, v in all_configs.items() if not (k.endswith("_PATH") and v is None)
    }

    # Convert non-basic types to strings for logging
    for key, value in all_configs.items():
        if isinstance(value, type) and issubclass(value, torch.nn.Module):
            all_configs[key] = value.__name__
        elif isinstance(value, (list, tuple)):
            all_configs[key] = str(value)
        elif not isinstance(value, (int, float, str, bool)):
            all_configs[key] = str(value) if value is not None else "None"

    return all_configs
