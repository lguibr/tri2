# File: config/core.py
import torch
from typing import Deque, Dict, Any, List, Type, Tuple, Optional

from .general import TOTAL_TRAINING_STEPS


class VisConfig:
    NUM_ENVS_TO_RENDER = 9
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.005
    LEFT_PANEL_WIDTH = int(SCREEN_WIDTH * 0.4)
    ENV_SPACING = 1
    ENV_GRID_PADDING = 1
    FPS = 0

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LIGHTG = (140, 140, 140)
    GRAY = (50, 50, 50)
    RED = (255, 50, 50)
    DARK_RED = (80, 10, 10)
    BLUE = (50, 50, 255)
    YELLOW = (255, 255, 100)
    GOOGLE_COLORS = [(15, 157, 88), (244, 180, 0), (66, 133, 244), (219, 68, 55)]
    LINE_CLEAR_FLASH_COLOR = (180, 180, 220)
    LINE_CLEAR_HIGHLIGHT_COLOR = (255, 255, 0, 180)
    GAME_OVER_FLASH_COLOR = (255, 0, 0)


class EnvConfig:
    NUM_ENVS = 512
    ROWS = 8
    COLS = 15
    GRID_FEATURES_PER_CELL = 2
    SHAPE_FEATURES_PER_SHAPE = 5
    NUM_SHAPE_SLOTS = 3

    @property
    def GRID_STATE_SHAPE(self) -> Tuple[int, int, int]:
        return (self.GRID_FEATURES_PER_CELL, self.ROWS, self.COLS)

    @property
    def SHAPE_STATE_DIM(self) -> int:
        return self.NUM_SHAPE_SLOTS * self.SHAPE_FEATURES_PER_SHAPE

    @property
    def ACTION_DIM(self) -> int:
        return self.NUM_SHAPE_SLOTS * (self.ROWS * self.COLS)


class RewardConfig:
    REWARD_PLACE_PER_TRI = 0.01  # Small positive reward for valid placement
    REWARD_CLEAR_1 = 1.5  # Increased reward
    REWARD_CLEAR_2 = 4.0  # Increased reward
    REWARD_CLEAR_3PLUS = 8.0  # Increased reward
    PENALTY_INVALID_MOVE = -0.1
    PENALTY_GAME_OVER = -1.5  # Slightly stronger penalty
    REWARD_ALIVE_STEP = 0.0  # Neutral step reward, focus on clears/placement

    # --- NEW: Height and Bumpiness Penalties ---
    # Penalize based on the maximum height reached on the board
    PENALTY_MAX_HEIGHT_FACTOR = -0.005  # Scaled by max_height
    # Penalize based on the sum of height differences between adjacent columns
    PENALTY_BUMPINESS_FACTOR = -0.01  # Scaled by total bumpiness


class PPOConfig:
    LEARNING_RATE = 3e-4
    ADAM_EPS = 1e-5
    NUM_STEPS_PER_ROLLOUT = 256
    PPO_EPOCHS = 4
    NUM_MINIBATCHES = 32
    CLIP_PARAM = 0.1
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.015  # Slightly increased for more exploration
    MAX_GRAD_NORM = 0.5
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_END_FRACTION = 0.0

    @property
    def MINIBATCH_SIZE(self) -> int:
        total_data_per_update = EnvConfig.NUM_ENVS * self.NUM_STEPS_PER_ROLLOUT
        batch_size = total_data_per_update // self.NUM_MINIBATCHES
        return max(1, batch_size)


class RNNConfig:
    USE_RNN = True
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 1


class TrainConfig:
    CHECKPOINT_SAVE_FREQ = 50
    LOAD_CHECKPOINT_PATH: str | None = None


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        CONV_CHANNELS = [32, 64, 64]
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True

        SHAPE_FEATURE_MLP_DIMS = [64]
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU

        # Slightly larger fusion MLP might help with RNN
        COMBINED_FC_DIMS = [512, 256]
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.0


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [50, 500, 5000]
    CONSOLE_LOG_FREQ = 10
    PLOT_DATA_WINDOW = 100_000


class TensorBoardConfig:
    LOG_HISTOGRAMS = True
    HISTOGRAM_LOG_FREQ = 20
    LOG_IMAGES = True
    IMAGE_LOG_FREQ = 100
    LOG_DIR: Optional[str] = None


class DemoConfig:
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = VisConfig.BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"
