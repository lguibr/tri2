import torch
from typing import List, Tuple, Optional

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


class VisConfig:
    NUM_ENVS_TO_RENDER = 16 
    FPS = 60 
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00
    LEFT_PANEL_RATIO = 0.7
    ENV_SPACING = 2 
    ENV_GRID_PADDING = 2 

    WHITE = WHITE
    BLACK = BLACK
    LIGHTG = LIGHTG
    GRAY = GRAY
    RED = RED
    DARK_RED = DARK_RED
    BLUE = BLUE
    YELLOW = YELLOW
    GOOGLE_COLORS = GOOGLE_COLORS
    LINE_CLEAR_FLASH_COLOR = LINE_CLEAR_FLASH_COLOR
    LINE_CLEAR_HIGHLIGHT_COLOR = LINE_CLEAR_HIGHLIGHT_COLOR
    GAME_OVER_FLASH_COLOR = GAME_OVER_FLASH_COLOR


class EnvConfig:
    NUM_ENVS = 1  # Set to 1, as we removed multi-env collection logic
    ROWS = 8
    COLS = 15
    GRID_FEATURES_PER_CELL = 2
    SHAPE_FEATURES_PER_SHAPE = 5
    NUM_SHAPE_SLOTS = 3
    EXPLICIT_FEATURES_DIM = 10
    CALCULATE_POTENTIAL_OUTCOMES_IN_STATE = False  # Keep False for now

    @property
    def GRID_STATE_SHAPE(self) -> Tuple[int, int, int]:
        return (self.GRID_FEATURES_PER_CELL, self.ROWS, self.COLS)

    @property
    def SHAPE_STATE_DIM(self) -> int:
        return self.NUM_SHAPE_SLOTS * self.SHAPE_FEATURES_PER_SHAPE

    @property
    def SHAPE_AVAILABILITY_DIM(self) -> int:
        return self.NUM_SHAPE_SLOTS

    @property
    def ACTION_DIM(self) -> int:
        # Action space might change for MCTS (e.g., just placement coords)
        # Keep original for now, but likely needs refactoring later.
        return self.NUM_SHAPE_SLOTS * (self.ROWS * self.COLS)



class RNNConfig:  # Keep for potential future NN architectures
    USE_RNN = False  # Default to False
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2


class TransformerConfig:  # Keep for potential future NN architectures
    USE_TRANSFORMER = False  # Default to False
    TRANSFORMER_D_MODEL = 256
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_DIM_FEEDFORWARD = 512
    TRANSFORMER_NUM_LAYERS = 3
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "relu"


class TrainConfig:  # Simplified for general training/checkpointing
    # Checkpointing might be handled differently (e.g., saving NN weights + MCTS stats)
    CHECKPOINT_SAVE_FREQ = (
        50  # Frequency might mean something else now (e.g., training steps, games)
    )
    LOAD_CHECKPOINT_PATH: Optional[str] = None


class ModelConfig:  # Keep structure for the AlphaZero NN
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        CONV_CHANNELS = [64, 128, 256]
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True

        SHAPE_FEATURE_MLP_DIMS = [128, 128]
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU

        COMBINED_FC_DIMS = [1024, 256]
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.0


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [25, 50, 100]
    CONSOLE_LOG_FREQ = 1
    PLOT_DATA_WINDOW = 100_000


class TensorBoardConfig:
    LOG_HISTOGRAMS = False
    HISTOGRAM_LOG_FREQ = 20  # Frequency might mean training steps/games
    LOG_IMAGES = False  # Keep ability to log images (e.g., board states)
    IMAGE_LOG_FREQ = 20  # Frequency might mean training steps/games
    LOG_DIR: Optional[str] = None


class DemoConfig:  # Keep as is
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"
    DEBUG_HELP_TEXT = "[Click]=Toggle Triangle | [R]=Reset Grid | [ESC]=Exit"
