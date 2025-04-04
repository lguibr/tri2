# File: config/__init__.py
# Expose core config classes and general constants/paths
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
    DemoConfig,  # Add this if you created the class
)
from .general import (
    DEVICE,
    RANDOM_SEED,
    RUN_ID,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,  # Need this path constant
    BUFFER_SAVE_PATH,
    MODEL_SAVE_PATH,
    TOTAL_TRAINING_STEPS,  # Need this constant
)
from .utils import get_config_dict
from .validation import print_config_info_and_validate

# --- MODIFIED: Assign LOG_DIR after imports ---
# This ensures the correct run-specific directory is set in the config class
# before it's potentially used elsewhere (like in the stats recorder init).
TensorBoardConfig.LOG_DIR = RUN_LOG_DIR
# --- END MODIFIED ---

# You can choose to expose everything or just specific items
__all__ = [
    # Core Classes
    "VisConfig",
    "EnvConfig",
    "RewardConfig",
    "DQNConfig",
    "TrainConfig",
    "BufferConfig",
    "ModelConfig",
    "StatsConfig",
    "TensorBoardConfig",
    "DemoConfig",  # Add this
    # General Constants/Paths
    "DEVICE",
    "RANDOM_SEED",
    "RUN_ID",
    "BASE_CHECKPOINT_DIR",
    "BASE_LOG_DIR",
    "RUN_CHECKPOINT_DIR",
    "RUN_LOG_DIR",
    "BUFFER_SAVE_PATH",
    "MODEL_SAVE_PATH",
    "TOTAL_TRAINING_STEPS",
    # Utils/Validation
    "get_config_dict",
    "print_config_info_and_validate",
]
