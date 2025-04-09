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

    PUCT_C: float = 1.25
    NUM_SIMULATIONS: int = 150  # Substantial simulations for better policy
    TEMPERATURE_INITIAL: float = 1.0
    TEMPERATURE_FINAL: float = 0.1
    TEMPERATURE_ANNEAL_STEPS: int = 20  # Anneal over a reasonable number of steps
    DIRICHLET_ALPHA: float = 0.3
    DIRICHLET_EPSILON: float = 0.25
    MAX_SEARCH_DEPTH: int = 40


class VisConfig:
    # Render fewer envs to save performance during serious training
    NUM_ENVS_TO_RENDER = 4
    FPS = 60
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00
    LEFT_PANEL_RATIO = 0.5
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
    # Environment definition remains the same
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

    CHECKPOINT_SAVE_FREQ = 1000  # Save every 10k training steps
    LOAD_CHECKPOINT_PATH: Optional[str] = None
    # --- Worker Configuration ---
    # Adjust based on your CPU cores (e.g., num_cores - 2)
    NUM_SELF_PLAY_WORKERS: int = 12
    # --------------------------
    BATCH_SIZE: int = 256  # Larger batch size for stable gradients (monitor VRAM)
    LEARNING_RATE: float = 3e-4  # Initial learning rate
    WEIGHT_DECAY: float = 1e-4  # Regularization
    NUM_TRAINING_STEPS_PER_ITER: int = 100  # Train more per iteration
    MIN_BUFFER_SIZE_TO_TRAIN: int = 5_000  # Ensure sufficient diverse data
    BUFFER_CAPACITY: int = 100_000  # Larger buffer for more experience diversity
    POLICY_LOSS_WEIGHT: float = 1.0
    VALUE_LOSS_WEIGHT: float = 1.0
    USE_LR_SCHEDULER: bool = True
    SCHEDULER_TYPE: str = "CosineAnnealingLR"
    SCHEDULER_T_MAX: int = 100000  # Corresponds to ~1M training steps cycle
    SCHEDULER_ETA_MIN: float = 1e-6


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        # Deeper and wider network
        CONV_CHANNELS = [128, 256, 256]  # Added a third conv stage
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True
        NUM_RESIDUAL_BLOCKS = 3  # More residual blocks per stage
        SHAPE_FEATURE_MLP_DIMS = [256]  # Larger MLP
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU
        COMBINED_FC_DIMS = [512, 512, 256]  # Deeper/wider FC layers
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.3  # Increased dropout for larger network


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [10,50,100,500,1000, 5000]  # Average over more steps/episodes
    CONSOLE_LOG_FREQ = 1000  # Log every 1000 updates (steps or episodes)
    PLOT_DATA_WINDOW = 50000  # See longer trends


class DemoConfig:
    # Demo config remains the same
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Click Preview]=Select/Deselect | [Click Grid]=Place | [ESC]=Exit"
    DEBUG_HELP_TEXT = "[Click]=Toggle Triangle | [R]=Reset Grid | [ESC]=Exit"
