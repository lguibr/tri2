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
    # --- Toy Level Settings ---
    NUM_SIMULATIONS: int = 6  # Drastically reduced for speed
    # --- End Toy Level ---
    TEMPERATURE_INITIAL: float = 1.0
    TEMPERATURE_FINAL: float = 0.1  # Slightly higher final temp for testing
    TEMPERATURE_ANNEAL_STEPS: int = 10  # Faster annealing
    DIRICHLET_ALPHA: float = 0.3
    DIRICHLET_EPSILON: float = 0.25
    MAX_SEARCH_DEPTH: int = 6  # Reduced max depth


class VisConfig:
    NUM_ENVS_TO_RENDER = 64  # Render fewer envs if using fewer workers
    FPS = 24  
    SCREEN_WIDTH = 1280  # Smaller screen might be okay for toy test
    SCREEN_HEIGHT = 720
    VISUAL_STEP_DELAY = 0.00
    LEFT_PANEL_RATIO = 0.6
    ENV_SPACING = 2
    ENV_GRID_PADDING = 1  # Smaller padding

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

    # --- Toy Level Settings ---
    CHECKPOINT_SAVE_FREQ = 100  # Save less often during quick tests
    LOAD_CHECKPOINT_PATH: Optional[str] = None
    NUM_SELF_PLAY_WORKERS: int = 64  # Reduced workers
    BATCH_SIZE: int = 16  # Smaller batch size
    LEARNING_RATE: float = 1e-4  # Keep LR, might need adjustment later
    WEIGHT_DECAY: float = 1e-5
    NUM_TRAINING_STEPS_PER_ITER: int = 10  # Fewer training steps per buffer fill
    MIN_BUFFER_SIZE_TO_TRAIN: int = 50  # Start training very quickly
    BUFFER_CAPACITY: int = 500  # Smaller buffer
    POLICY_LOSS_WEIGHT: float = 1.0
    VALUE_LOSS_WEIGHT: float = 1.0
    USE_LR_SCHEDULER: bool = True
    SCHEDULER_TYPE: str = "CosineAnnealingLR"
    SCHEDULER_T_MAX: int = 10000  # Shorter cycle for testing
    SCHEDULER_ETA_MIN: float = 1e-6
    # --- End Toy Level ---


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance
        # --- Toy Level Settings ---
        CONV_CHANNELS = [16, 32]  # Fewer/smaller conv layers
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True  # Keep BatchNorm for stability
        SHAPE_FEATURE_MLP_DIMS = [32]  # Smaller shape MLP
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU
        COMBINED_FC_DIMS = [128, 64]  # Smaller fusion MLP
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True  # Keep BatchNorm
        DROPOUT_FC = 0.0  # Disable dropout for simplicity in toy tests
        # --- End Toy Level ---


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [10, 25]  # Shorter averaging windows
    CONSOLE_LOG_FREQ = 1  # Log every update/episode
    PLOT_DATA_WINDOW = 1000  # Keep fewer points for plotting


class DemoConfig:  # No changes needed for toy training test
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Click Preview]=Select/Deselect | [Click Grid]=Place | [ESC]=Exit"
    DEBUG_HELP_TEXT = "[Click]=Toggle Triangle | [R]=Reset Grid | [ESC]=Exit"
