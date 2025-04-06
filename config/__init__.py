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
    ObsNormConfig,
    TransformerConfig,
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

from .constants import (
    WHITE,
    BLACK,
    LIGHTG,
    GRAY,
    RED,
    DARK_RED,
    BLUE,
    YELLOW,
    GOOGLE_COLORS,
    LINE_CLEAR_FLASH_COLOR,
    LINE_CLEAR_HIGHLIGHT_COLOR,
    GAME_OVER_FLASH_COLOR,
)

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
    "ObsNormConfig",
    "TransformerConfig",
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
    "WHITE",
    "BLACK",
    "LIGHTG",
    "GRAY",
    "RED",
    "DARK_RED",
    "BLUE",
    "YELLOW",
    "GOOGLE_COLORS",
    "LINE_CLEAR_FLASH_COLOR",
    "LINE_CLEAR_HIGHLIGHT_COLOR",
    "GAME_OVER_FLASH_COLOR",
]
