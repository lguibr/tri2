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
    NUM_ENVS_TO_RENDER = 4
    FPS = 0
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00
    LEFT_PANEL_WIDTH = int(SCREEN_WIDTH * 0.8)  # Keep large for plot + legend
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
    REWARD_PLACE_PER_TRI = 0.01
    REWARD_PER_CLEARED_TRIANGLE = 0.15  # Adjust value as needed
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
    LEARNING_RATE = 2.5e-4
    ADAM_EPS = 1e-5
    NUM_STEPS_PER_ROLLOUT = 2048  # Reduced for faster updates
    PPO_EPOCHS = 6
    NUM_MINIBATCHES = 512  # Adjusted to give minibatch size 512 (128*2048/512=512)
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
        min_recommended_size = 32
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
    TRANSFORMER_D_MODEL = 1024
    TRANSFORMER_NHEAD = 16
    TRANSFORMER_DIM_FEEDFORWARD = 1024
    TRANSFORMER_NUM_LAYERS = 2
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "relu"


class TrainConfig:
    CHECKPOINT_SAVE_FREQ = 10  # Save every 10 rollouts (more frequent)
    LOAD_CHECKPOINT_PATH: Optional[str] = None


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        CONV_CHANNELS = [128, 256, 512]
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True

        SHAPE_FEATURE_MLP_DIMS = [256]
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU

        _transformer_cfg = TransformerConfig()
        _last_fc_dim = (
            _transformer_cfg.TRANSFORMER_D_MODEL
            if _transformer_cfg.USE_TRANSFORMER
            else 1024
        )
        COMBINED_FC_DIMS = [
            2048,
            _last_fc_dim,
        ]
        del _transformer_cfg
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.0


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [
        50,
        100,
        500,
        1000,
        2000,  # Adjusted for shorter training
    ]
    CONSOLE_LOG_FREQ = 5  # Log every 5 rollouts (more frequent)
    PLOT_DATA_WINDOW = 100_000


class TensorBoardConfig:
    LOG_HISTOGRAMS = False
    HISTOGRAM_LOG_FREQ = 50  # Log every 50 rollouts
    LOG_IMAGES = False
    IMAGE_LOG_FREQ = 50  # Log every 50 rollouts
    LOG_DIR: Optional[str] = None
    LOG_SHAPE_PLACEMENT_Q_VALUES = False


class DemoConfig:
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"
    DEBUG_HELP_TEXT = "[Click]=Toggle Triangle | [R]=Reset Grid | [ESC]=Exit"
