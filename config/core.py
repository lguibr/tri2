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
    NUM_SIMULATIONS: int = 100  # Increased from toy value
    TEMPERATURE_INITIAL: float = 1.0
    TEMPERATURE_FINAL: float = 0.01  # Colder final temperature for exploitation
    TEMPERATURE_ANNEAL_STEPS: int = 30  # Reasonable annealing steps
    DIRICHLET_ALPHA: float = 0.3
    DIRICHLET_EPSILON: float = 0.25
    MAX_SEARCH_DEPTH: int = 100  # Restore reasonable max depth


class VisConfig:
    NUM_ENVS_TO_RENDER = 16  # Render a reasonable subset
    FPS = 30  # Standard FPS
    SCREEN_WIDTH = 1600  # Larger screen for more info
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00
    LEFT_PANEL_RATIO = 0.5  # Balance panels
    ENV_SPACING = 2
    ENV_GRID_PADDING = 1

    # Colors remain the same
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
    # Keep environment dimensions standard unless specifically testing variations
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


class RNNConfig:  # Keep disabled
    USE_RNN = False
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2


class TransformerConfig:  # Keep disabled
    USE_TRANSFORMER = False
    TRANSFORMER_D_MODEL = 256
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_DIM_FEEDFORWARD = 512
    TRANSFORMER_NUM_LAYERS = 3
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "relu"


class TrainConfig:
    """Configuration parameters for the Training Worker."""

    CHECKPOINT_SAVE_FREQ = 1000  # Save periodically
    LOAD_CHECKPOINT_PATH: Optional[str] = None
    NUM_SELF_PLAY_WORKERS: int = 64  # Number of parallel game simulations
    BATCH_SIZE: int = 512  # Batch size for NN training
    LEARNING_RATE: float = 1e-4  # Learning rate
    WEIGHT_DECAY: float = 1e-5  # Weight decay for regularization
    NUM_TRAINING_STEPS_PER_ITER: int = 100  # Steps per buffer sampling cycle
    MIN_BUFFER_SIZE_TO_TRAIN: int = 20000  # Min experiences before training starts
    BUFFER_CAPACITY: int = 200000  # Max experiences in buffer
    POLICY_LOSS_WEIGHT: float = 1.0
    VALUE_LOSS_WEIGHT: float = 1.0
    USE_LR_SCHEDULER: bool = True
    SCHEDULER_TYPE: str = "CosineAnnealingLR"
    SCHEDULER_T_MAX: int = 100000  # Adjust T_max based on expected total steps
    SCHEDULER_ETA_MIN: float = 1e-6


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        # --- Increased Network Complexity ---
        CONV_CHANNELS = [32, 64, 128]  # More/larger conv layers
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True
        SHAPE_FEATURE_MLP_DIMS = [64]  # Larger shape MLP
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU
        COMBINED_FC_DIMS = [256, 128]  # Larger fusion MLP
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.1  # Enable dropout for regularization


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [100, 500]  # Standard averaging windows
    CONSOLE_LOG_FREQ = 100  # Log less frequently to avoid spam
    PLOT_DATA_WINDOW = 10000  # Keep more points for plotting


class DemoConfig:  # No changes needed for non-toy training
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Click Preview]=Select/Deselect | [Click Grid]=Place | [ESC]=Exit"
    DEBUG_HELP_TEXT = "[Click]=Toggle Triangle | [R]=Reset Grid | [ESC]=Exit"
