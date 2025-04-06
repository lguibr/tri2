# File: config/core.py
import torch
from typing import Deque, Dict, Any, List, Type, Tuple, Optional

# --- MODIFIED: Import from constants ---
from .general import TOTAL_TRAINING_STEPS
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

# --- END MODIFIED ---


class VisConfig:
    NUM_ENVS_TO_RENDER = 4
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00
    LEFT_PANEL_WIDTH = int(SCREEN_WIDTH * 0.8)
    ENV_SPACING = 0
    ENV_GRID_PADDING = 0
    FPS = 0  # Set to 0 for max speed, or > 0 to cap FPS

    # --- MODIFIED: Use imported constants ---
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
    # --- END MODIFIED ---


# ... (rest of the file remains the same) ...


class EnvConfig:
    NUM_ENVS = 256
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


class RewardConfig:
    REWARD_PLACE_PER_TRI = 0.01
    REWARD_CLEAR_1 = 1.5
    REWARD_CLEAR_2 = 4.0
    REWARD_CLEAR_3PLUS = 8.0
    REWARD_ALIVE_STEP = 0.001
    PENALTY_INVALID_MOVE = -0.1
    PENALTY_GAME_OVER = -1.5
    PENALTY_MAX_HEIGHT_FACTOR = -0.005
    PENALTY_BUMPINESS_FACTOR = -0.01
    PENALTY_HOLE_PER_HOLE = -0.07
    PENALTY_NEW_HOLE = -0.15
    ENABLE_PBRS = True
    PBRS_HEIGHT_COEF = -0.05
    PBRS_HOLE_COEF = -0.20
    PBRS_BUMPINESS_COEF = -0.02


class PPOConfig:
    LEARNING_RATE = 1e-4
    ADAM_EPS = 1e-5
    NUM_STEPS_PER_ROLLOUT = 128  # Reduced for testing/debugging if needed
    PPO_EPOCHS = 6
    NUM_MINIBATCHES = 64
    CLIP_PARAM = 0.1
    GAMMA = 0.995
    GAE_LAMBDA = 0.95
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5

    # --- MODIFIED: Added LR Scheduler Config ---
    USE_LR_SCHEDULER = True
    LR_SCHEDULE_TYPE = "linear"  # Options: "linear", "cosine"
    LR_LINEAR_END_FRACTION = (
        0.0  # For linear schedule: fraction of initial LR at the end
    )
    LR_COSINE_MIN_FACTOR = 0.01  # For cosine schedule: min LR = initial_lr * min_factor
    # --- END MODIFIED ---

    @property
    def MINIBATCH_SIZE(self) -> int:
        env_config_instance = EnvConfig()
        total_data_per_update = (
            env_config_instance.NUM_ENVS * self.NUM_STEPS_PER_ROLLOUT
        )
        del env_config_instance
        if self.NUM_MINIBATCHES <= 0:
            num_minibatches = 1
        else:
            num_minibatches = self.NUM_MINIBATCHES
        batch_size = total_data_per_update // num_minibatches
        min_recommended_size = 128
        if batch_size < min_recommended_size and batch_size > 0:  # Added check > 0
            print(
                f"Warning: Calculated minibatch size ({batch_size}) is < {min_recommended_size}. Consider adjusting NUM_STEPS_PER_ROLLOUT or NUM_MINIBATCHES."
            )
        elif batch_size <= 0:
            print(
                f"ERROR: Calculated minibatch size is <= 0 ({batch_size}). Check NUM_ENVS({env_config_instance.NUM_ENVS}), NUM_STEPS_PER_ROLLOUT({self.NUM_STEPS_PER_ROLLOUT}), NUM_MINIBATCHES({self.NUM_MINIBATCHES}). Defaulting to 1."
            )
            return 1
        return max(1, batch_size)


class RNNConfig:
    USE_RNN = True
    LSTM_HIDDEN_SIZE = 1024
    LSTM_NUM_LAYERS = 1


class ObsNormConfig:
    ENABLE_OBS_NORMALIZATION = True
    NORMALIZE_GRID = True
    NORMALIZE_SHAPES = True
    NORMALIZE_AVAILABILITY = False
    NORMALIZE_EXPLICIT_FEATURES = True
    OBS_CLIP = 10.0
    EPSILON = 1e-8


class TransformerConfig:
    USE_TRANSFORMER = True
    TRANSFORMER_D_MODEL = 896
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_DIM_FEEDFORWARD = 1024
    TRANSFORMER_NUM_LAYERS = 2
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "relu"


class TrainConfig:
    CHECKPOINT_SAVE_FREQ = 20
    LOAD_CHECKPOINT_PATH: Optional[str] = None


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        CONV_CHANNELS = [96, 192, 192]
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True

        SHAPE_FEATURE_MLP_DIMS = [192]
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU

        _transformer_cfg = TransformerConfig()
        _last_fc_dim = (
            _transformer_cfg.TRANSFORMER_D_MODEL
            if _transformer_cfg.USE_TRANSFORMER
            else 896
        )
        COMBINED_FC_DIMS = [1792, _last_fc_dim]
        del _transformer_cfg
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.0


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [50, 100, 500, 1_000, 5_000, 10_000]
    CONSOLE_LOG_FREQ = 5
    PLOT_DATA_WINDOW = 100_000


class TensorBoardConfig:
    LOG_HISTOGRAMS = True
    HISTOGRAM_LOG_FREQ = 20
    LOG_IMAGES = True
    IMAGE_LOG_FREQ = 50
    LOG_DIR: Optional[str] = None
    LOG_SHAPE_PLACEMENT_Q_VALUES = False


class DemoConfig:
    # --- MODIFIED: Use imported constants ---
    BACKGROUND_COLOR = (10, 10, 20)  # Keep specific demo color here
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"
    # --- END MODIFIED ---
