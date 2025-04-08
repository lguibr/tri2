# File: config/__init__.py
# config/__init__.py
# This file marks the 'config' directory as a Python package.

# Import core configuration classes to make them available directly under 'config'
from .core import (
    VisConfig,
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    # TensorBoardConfig removed
    DemoConfig,
    TransformerConfig,
    MCTSConfig,
)

# Import general configuration settings and functions
from .general import (
    DEVICE,
    RANDOM_SEED,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    set_device,
    get_run_id,
    set_run_id,
    get_run_checkpoint_dir,
    get_run_log_dir,
    get_console_log_dir,
    get_model_save_path,
)

# Import utility functions
from .utils import get_config_dict

# Import validation function
from .validation import print_config_info_and_validate

# Import constants
from .constants import (
    WHITE,
    BLACK,
    LIGHTG,
    GRAY,
    DARK_GRAY,
    RED,
    DARK_RED,
    BLUE,
    YELLOW,
    GREEN,
    DARK_GREEN,  # Added DARK_GREEN import
    ORANGE,
    PURPLE,
    CYAN,
    GOOGLE_COLORS,
    LINE_CLEAR_FLASH_COLOR,
    LINE_CLEAR_HIGHLIGHT_COLOR,
    GAME_OVER_FLASH_COLOR,
    # MCTS Colors (also available directly)
    MCTS_NODE_WIN_COLOR,
    MCTS_NODE_LOSS_COLOR,
    MCTS_NODE_NEUTRAL_COLOR,
    MCTS_NODE_BORDER_COLOR,
    MCTS_NODE_SELECTED_BORDER_COLOR,
    MCTS_EDGE_COLOR,
    MCTS_EDGE_HIGHLIGHT_COLOR,
    MCTS_INFO_TEXT_COLOR,
    MCTS_NODE_TEXT_COLOR,
    MCTS_NODE_PRIOR_COLOR,
    MCTS_NODE_SCORE_COLOR,
    MCTS_MINI_GRID_BG_COLOR,
    MCTS_MINI_GRID_LINE_COLOR,
    MCTS_MINI_GRID_OCCUPIED_COLOR,
)


# Define __all__ to control what 'from config import *' imports
__all__ = [
    # Core Configs
    "VisConfig",
    "EnvConfig",
    "RNNConfig",
    "TrainConfig",
    "ModelConfig",
    "StatsConfig",
    # "TensorBoardConfig", # Removed
    "DemoConfig",
    "TransformerConfig",
    "MCTSConfig",
    # General Configs
    "DEVICE",
    "RANDOM_SEED",
    "BASE_CHECKPOINT_DIR",
    "BASE_LOG_DIR",
    "set_device",
    "get_run_id",
    "set_run_id",
    "get_run_checkpoint_dir",
    "get_run_log_dir",
    "get_console_log_dir",
    "get_model_save_path",
    # Utils
    "get_config_dict",
    "print_config_info_and_validate",
    # Constants
    "WHITE",
    "BLACK",
    "LIGHTG",
    "GRAY",
    "DARK_GRAY",
    "RED",
    "DARK_RED",
    "BLUE",
    "YELLOW",
    "GREEN",
    "DARK_GREEN",  # Added DARK_GREEN export
    "ORANGE",
    "PURPLE",
    "CYAN",
    "GOOGLE_COLORS",
    "LINE_CLEAR_FLASH_COLOR",
    "LINE_CLEAR_HIGHLIGHT_COLOR",
    "GAME_OVER_FLASH_COLOR",
    # MCTS Colors
    "MCTS_NODE_WIN_COLOR",
    "MCTS_NODE_LOSS_COLOR",
    "MCTS_NODE_NEUTRAL_COLOR",
    "MCTS_NODE_BORDER_COLOR",
    "MCTS_NODE_SELECTED_BORDER_COLOR",
    "MCTS_EDGE_COLOR",
    "MCTS_EDGE_HIGHLIGHT_COLOR",
    "MCTS_INFO_TEXT_COLOR",
    "MCTS_NODE_TEXT_COLOR",
    "MCTS_NODE_PRIOR_COLOR",
    "MCTS_NODE_SCORE_COLOR",
    "MCTS_MINI_GRID_BG_COLOR",
    "MCTS_MINI_GRID_LINE_COLOR",
    "MCTS_MINI_GRID_OCCUPIED_COLOR",
]
