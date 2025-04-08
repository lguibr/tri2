# File: config/core.py
import torch
from typing import List, Tuple, Optional

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
    DARK_GREEN,
    ORANGE,
    PURPLE,
    CYAN,
    GOOGLE_COLORS,
    LINE_CLEAR_FLASH_COLOR,
    LINE_CLEAR_HIGHLIGHT_COLOR,
    GAME_OVER_FLASH_COLOR,
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


class MCTSConfig:
    """Configuration parameters for the Monte Carlo Tree Search."""

    PUCT_C: float = 1.5
    NUM_SIMULATIONS: int = 100
    TEMPERATURE_INITIAL: float = 1.0
    TEMPERATURE_FINAL: float = 0.01
    TEMPERATURE_ANNEAL_STEPS: int = 30
    DIRICHLET_ALPHA: float = 0.3
    DIRICHLET_EPSILON: float = 0.25
    MAX_SEARCH_DEPTH: int = 100


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
    DARK_GRAY = DARK_GRAY
    RED = RED
    DARK_RED = DARK_RED
    BLUE = BLUE
    YELLOW = YELLOW
    GREEN = GREEN
    DARK_GREEN = DARK_GREEN
    ORANGE = ORANGE
    PURPLE = PURPLE
    CYAN = CYAN
    GOOGLE_COLORS = GOOGLE_COLORS
    LINE_CLEAR_FLASH_COLOR = LINE_CLEAR_FLASH_COLOR
    LINE_CLEAR_HIGHLIGHT_COLOR = LINE_CLEAR_HIGHLIGHT_COLOR
    GAME_OVER_FLASH_COLOR = GAME_OVER_FLASH_COLOR
    # MCTS Colors
    MCTS_NODE_WIN_COLOR = MCTS_NODE_WIN_COLOR
    MCTS_NODE_LOSS_COLOR = MCTS_NODE_LOSS_COLOR
    MCTS_NODE_NEUTRAL_COLOR = MCTS_NODE_NEUTRAL_COLOR
    MCTS_NODE_BORDER_COLOR = MCTS_NODE_BORDER_COLOR
    MCTS_NODE_SELECTED_BORDER_COLOR = MCTS_NODE_SELECTED_BORDER_COLOR
    MCTS_EDGE_COLOR = MCTS_EDGE_COLOR
    MCTS_EDGE_HIGHLIGHT_COLOR = MCTS_EDGE_HIGHLIGHT_COLOR
    MCTS_INFO_TEXT_COLOR = MCTS_INFO_TEXT_COLOR
    MCTS_NODE_TEXT_COLOR = MCTS_NODE_TEXT_COLOR
    MCTS_NODE_PRIOR_COLOR = MCTS_NODE_PRIOR_COLOR
    MCTS_NODE_SCORE_COLOR = MCTS_NODE_SCORE_COLOR
    MCTS_MINI_GRID_BG_COLOR = MCTS_MINI_GRID_BG_COLOR
    MCTS_MINI_GRID_LINE_COLOR = MCTS_MINI_GRID_LINE_COLOR
    MCTS_MINI_GRID_OCCUPIED_COLOR = MCTS_MINI_GRID_OCCUPIED_COLOR


class EnvConfig:
    NUM_ENVS = 1  # AlphaZero typically uses 1 env for self-play generation
    ROWS = 8
    COLS = 15
    GRID_FEATURES_PER_CELL = 2
    SHAPE_FEATURES_PER_SHAPE = 5
    NUM_SHAPE_SLOTS = 3
    EXPLICIT_FEATURES_DIM = 10
    CALCULATE_POTENTIAL_OUTCOMES_IN_STATE = False

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
        return self.NUM_SHAPE_SLOTS * (self.ROWS * self.COLS)


class RNNConfig:
    USE_RNN = False
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2


class TransformerConfig:
    USE_TRANSFORMER = False
    TRANSFORMER_D_MODEL = 256
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_DIM_FEEDFORWARD = 512
    TRANSFORMER_NUM_LAYERS = 3
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "relu"


class TrainConfig:
    """Configuration parameters for the Training Worker."""

    CHECKPOINT_SAVE_FREQ = 50  # Save every N training steps/epochs
    LOAD_CHECKPOINT_PATH: Optional[str] = None

    # --- Training Loop Parameters ---
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5  # L2 Regularization
    NUM_TRAINING_STEPS_PER_ITER: int = (
        100  # How many optimizer steps per training iteration
    )
    MIN_BUFFER_SIZE_TO_TRAIN: int = (
        1000  # Minimum experiences in buffer before training starts
    )
    BUFFER_CAPACITY: int = 50_000  # Max size of the experience replay buffer

    # --- Loss Weights (Optional) ---
    POLICY_LOSS_WEIGHT: float = 1.0
    VALUE_LOSS_WEIGHT: float = 1.0


class ModelConfig:
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
    CONSOLE_LOG_FREQ = 1  # Log every N training steps or episodes
    PLOT_DATA_WINDOW = 100_000


class TensorBoardConfig:
    LOG_HISTOGRAMS = False
    HISTOGRAM_LOG_FREQ = 20  # Log every N training steps
    LOG_IMAGES = False
    IMAGE_LOG_FREQ = 20  # Log every N training steps
    LOG_DIR: Optional[str] = None


class DemoConfig:
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"
    DEBUG_HELP_TEXT = "[Click]=Toggle Triangle | [R]=Reset Grid | [ESC]=Exit"
