# File: config/core.py
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
    NUM_ENVS_TO_RENDER = 16  # Updated
    FPS = 0
    SCREEN_WIDTH = 1600  # Initial width, but resizable
    SCREEN_HEIGHT = 900  # Initial height, but resizable
    VISUAL_STEP_DELAY = 0.00
    # Changed LEFT_PANEL_WIDTH to LEFT_PANEL_RATIO
    LEFT_PANEL_RATIO = 0.7
    ENV_SPACING = 0
    ENV_GRID_PADDING = 0

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
    NUM_ENVS = 128
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
    REWARD_PLACE_PER_TRI = 0.0
    REWARD_PER_CLEARED_TRIANGLE = 0.2
    REWARD_ALIVE_STEP = 0.01
    PENALTY_INVALID_MOVE = -0.1
    PENALTY_GAME_OVER = -2

    PENALTY_MAX_HEIGHT_FACTOR = -0.005
    PENALTY_BUMPINESS_FACTOR = -0.01
    PENALTY_HOLE_PER_HOLE = -0.07
    PENALTY_NEW_HOLE = -0.15
    ENABLE_PBRS = True

    PBRS_HEIGHT_COEF = -0.05
    PBRS_HOLE_COEF = -0.20
    PBRS_BUMPINESS_COEF = -0.02


class PPOConfig:
    LEARNING_RATE = 2.5e-4
    ADAM_EPS = 1e-5
    # --- Increased Params ---
    NUM_STEPS_PER_ROLLOUT = 256  # Increased from 128
    PPO_EPOCHS = 4  # Increased from 2
    NUM_MINIBATCHES = 16  # Increased from 8
    # --- End Increased Params ---
    CLIP_PARAM = 0.2
    GAMMA = 0.995
    GAE_LAMBDA = 0.95
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5

    USE_LR_SCHEDULER = True
    LR_SCHEDULE_TYPE = "cosine"
    LR_LINEAR_END_FRACTION = 0.0
    LR_COSINE_MIN_FACTOR = 0.01

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
        min_recommended_size = 4  # Keep low for testing flexibility
        if batch_size < min_recommended_size and batch_size > 0:
            pass
        elif batch_size <= 0:
            local_env_config = EnvConfig()
            print(
                f"ERROR: Calculated minibatch size is <= 0 ({batch_size}). Check NUM_ENVS({local_env_config.NUM_ENVS}), NUM_STEPS_PER_ROLLOUT({self.NUM_STEPS_PER_ROLLOUT}), NUM_MINIBATCHES({self.NUM_MINIBATCHES}). Defaulting to 1."
            )
            del local_env_config
            return 1
        return max(1, batch_size)


class RNNConfig:
    USE_RNN = True
    LSTM_HIDDEN_SIZE = 256  # Updated
    LSTM_NUM_LAYERS = 2  # Updated


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
    TRANSFORMER_D_MODEL = 256  # Updated
    TRANSFORMER_NHEAD = 8  # Updated
    TRANSFORMER_DIM_FEEDFORWARD = 512  # Updated
    TRANSFORMER_NUM_LAYERS = 3  # Updated
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "relu"


class TrainConfig:
    CHECKPOINT_SAVE_FREQ = 50  # Increased from 10 (interpreted as rollouts)
    LOAD_CHECKPOINT_PATH: Optional[str] = None


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        CONV_CHANNELS = [64, 128, 256]  # Updated
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True

        SHAPE_FEATURE_MLP_DIMS = [128, 128]  # Updated
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU

        _transformer_cfg = TransformerConfig()
        _rnn_cfg = RNNConfig()
        _last_fc_dim = 256  # Updated
        if _transformer_cfg.USE_TRANSFORMER:
            _last_fc_dim = _transformer_cfg.TRANSFORMER_D_MODEL  # Should be 256
        elif _rnn_cfg.USE_RNN:
            _last_fc_dim = _rnn_cfg.LSTM_HIDDEN_SIZE  # Should be 256

        COMBINED_FC_DIMS = [
            1024,  # Updated
            _last_fc_dim,  # Should be 256 based on Transformer/LSTM
        ]
        del _transformer_cfg, _rnn_cfg  # Clean up temp instances
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.0


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [
        25,
        50,
        100,
    ]
    CONSOLE_LOG_FREQ = 5
    PLOT_DATA_WINDOW = 100_000  # Increased from 20_000


class TensorBoardConfig:
    LOG_HISTOGRAMS = False
    HISTOGRAM_LOG_FREQ = 20
    LOG_IMAGES = False
    IMAGE_LOG_FREQ = 20
    LOG_DIR: Optional[str] = None
    LOG_SHAPE_PLACEMENT_Q_VALUES = False


class DemoConfig:
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"
    DEBUG_HELP_TEXT = "[Click]=Toggle Triangle | [R]=Reset Grid | [ESC]=Exit"
