# File: config/__init__.py
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
from .general import (
    DEVICE,
    RANDOM_SEED,
    RUN_ID,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,
    MODEL_SAVE_PATH,
    TOTAL_TRAINING_STEPS,
)
from .utils import get_config_dict
from .validation import print_config_info_and_validate

# Assign RUN_LOG_DIR to TensorBoardConfig after imports
TensorBoardConfig.LOG_DIR = RUN_LOG_DIR

__all__ = [
    # Core Classes
    "VisConfig",
    "EnvConfig",
    "RewardConfig",
    "PPOConfig",
    "RNNConfig",
    "TrainConfig",
    "ModelConfig",
    "StatsConfig",
    "TensorBoardConfig",
    "DemoConfig",
    # General Constants/Paths
    "DEVICE",
    "RANDOM_SEED",
    "RUN_ID",
    "BASE_CHECKPOINT_DIR",
    "BASE_LOG_DIR",
    "RUN_CHECKPOINT_DIR",
    "RUN_LOG_DIR",
    "MODEL_SAVE_PATH",
    "TOTAL_TRAINING_STEPS",
    # Utils/Validation
    "get_config_dict",
    "print_config_info_and_validate",
]
