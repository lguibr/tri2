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
    ObsNormConfig,
    TransformerConfig,
)
from .general import (
    DEVICE,  # Keep DEVICE as it's set early
    RANDOM_SEED,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    TOTAL_TRAINING_STEPS,
    # Import getter functions instead of direct constants
    get_run_id,
    get_run_checkpoint_dir,
    get_run_log_dir,
    get_model_save_path,
    get_console_log_dir,
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

# Assign RUN_LOG_DIR to TensorBoardConfig using the getter
# This ensures it uses the potentially resumed run's log directory
TensorBoardConfig.LOG_DIR = get_run_log_dir()

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
    "BASE_CHECKPOINT_DIR",
    "BASE_LOG_DIR",
    "TOTAL_TRAINING_STEPS",
    # Getters for dynamic paths
    "get_run_id",
    "get_run_checkpoint_dir",
    "get_run_log_dir",
    "get_model_save_path",
    "get_console_log_dir",
    # Utils/Validation
    "get_config_dict",
    "print_config_info_and_validate",
    # Constants
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
