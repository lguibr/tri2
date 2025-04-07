# File: config/utils.py
import torch
from typing import Dict, Any
from .core import (
    VisConfig,
    EnvConfig,
    # Removed RewardConfig
    # Removed PPOConfig
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    DemoConfig,
    # Removed ObsNormConfig
    TransformerConfig,
)

# Import DEVICE and RANDOM_SEED directly, use get_run_id() for the run ID
from .general import DEVICE, RANDOM_SEED, get_run_id


def get_config_dict() -> Dict[str, Any]:
    """Returns a flat dictionary of all relevant config values for logging."""
    all_configs = {}

    def flatten_class(cls, prefix=""):
        d = {}
        # Use cls() to instantiate if needed for properties, but be careful
        instance = None
        try:
            instance = cls()
        except Exception:
            instance = None  # Cannot instantiate, rely on class vars

        for k, v in vars(cls).items():
            if (
                not k.startswith("__")
                and not callable(v)
                and not isinstance(v, type)
                and not hasattr(v, "__module__")  # Exclude modules
            ):
                # Check if it's a property descriptor on the class
                is_property = isinstance(getattr(cls, k, None), property)

                if is_property and instance:
                    try:
                        v = getattr(instance, k)  # Get property value from instance
                    except Exception:
                        continue  # Skip if property access fails
                elif is_property and not instance:
                    continue  # Skip properties if instance couldn't be created

                d[f"{prefix}{k}"] = v
        return d

    # Flatten core config classes
    all_configs.update(flatten_class(VisConfig, "Vis."))
    all_configs.update(flatten_class(EnvConfig, "Env."))
    # Removed RewardConfig flattening
    # Removed PPOConfig flattening
    all_configs.update(flatten_class(RNNConfig, "RNN."))
    all_configs.update(flatten_class(TrainConfig, "Train."))
    all_configs.update(flatten_class(ModelConfig.Network, "Model.Net."))
    all_configs.update(flatten_class(StatsConfig, "Stats."))
    all_configs.update(flatten_class(TensorBoardConfig, "TB."))
    all_configs.update(flatten_class(DemoConfig, "Demo."))
    # Removed ObsNormConfig flattening
    all_configs.update(flatten_class(TransformerConfig, "Transformer."))

    # Add general config values
    all_configs["General.DEVICE"] = str(DEVICE) if DEVICE else "None"
    all_configs["General.RANDOM_SEED"] = RANDOM_SEED
    all_configs["General.RUN_ID"] = get_run_id()  # Use the getter function

    # Filter out None paths
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
            # Check for None explicitly before converting to string
            if value is None:
                all_configs[key] = "None"
            else:
                all_configs[key] = str(value)

    return all_configs
