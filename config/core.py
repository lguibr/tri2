# File: config/core.py
# --- Core Configuration Classes ---
import torch
from typing import Deque, Dict, Any, List, Type, Tuple, Optional

from .general import TOTAL_TRAINING_STEPS


# --- Visualization (Pygame) ---
class VisConfig:
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0
    LEFT_PANEL_WIDTH = SCREEN_WIDTH // 2
    ENV_SPACING = 1
    ENV_GRID_PADDING = 1
    FPS = 0
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LIGHTG = (140, 140, 140)
    GRAY = (50, 50, 50)
    RED = (255, 50, 50)
    BLUE = (50, 50, 255)
    YELLOW = (255, 255, 100)
    GOOGLE_COLORS = [(15, 157, 88), (244, 180, 0), (66, 133, 244), (219, 68, 55)]
    NUM_ENVS_TO_RENDER = 48
    LINE_CLEAR_FLASH_COLOR = (180, 180, 220)
    LINE_CLEAR_HIGHLIGHT_COLOR = (255, 255, 0, 180)


# --- Environment ---
class EnvConfig:
    NUM_ENVS = 256
    ROWS = 8
    COLS = 15
    GRID_FEATURES_PER_CELL = 2  # Occupied, Is_Up
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


# --- Reward Shaping (RL Reward) ---
class RewardConfig:
    REWARD_PLACE_PER_TRI = 0.0  # Base reward for placing a piece (can be 0)
    REWARD_CLEAR_1 = 1.0
    REWARD_CLEAR_2 = 3.0
    REWARD_CLEAR_3PLUS = 6.0
    PENALTY_INVALID_MOVE = -0.1
    PENALTY_HOLE_PER_HOLE = -0.05
    PENALTY_GAME_OVER = -1.0
    # --- MODIFIED: Add small survival reward ---
    REWARD_ALIVE_STEP = 0.001  # Small positive reward for making a valid move


# --- DQN Algorithm ---
class DQNConfig:
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = 5_000
    LEARNING_RATE = 5e-5
    ADAM_EPS = 1e-4
    GRADIENT_CLIP_NORM = 10.0
    USE_DOUBLE_DQN = True
    USE_DUELING = True
    USE_NOISY_NETS = True
    USE_DISTRIBUTIONAL = True
    V_MIN = -10.0
    V_MAX = 10.0
    NUM_ATOMS = 51
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_T_MAX: int = TOTAL_TRAINING_STEPS
    LR_SCHEDULER_ETA_MIN: float = 1e-7


# --- Training Loop ---
class TrainConfig:
    BATCH_SIZE = 64
    LEARN_START_STEP = 5_000
    LEARN_FREQ = 4
    CHECKPOINT_SAVE_FREQ = 20_000
    LOAD_CHECKPOINT_PATH: str | None = None
    LOAD_BUFFER_PATH: str | None = None


# --- Replay Buffer ---
class BufferConfig:
    REPLAY_BUFFER_SIZE = 200_000
    USE_N_STEP = True
    N_STEP = 3
    USE_PER = True
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = TOTAL_TRAINING_STEPS // 2
    PER_EPSILON = 1e-6


# --- Model Architecture ---
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
        COMBINED_FC_DIMS = [256, 128]
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.0


# --- Statistics and Logging ---
class StatsConfig:
    STATS_AVG_WINDOW = 1000
    CONSOLE_LOG_FREQ = 5_000
    PLOT_DATA_WINDOW = 100_000


# --- TensorBoard Logging ---
class TensorBoardConfig:
    LOG_HISTOGRAMS = True
    HISTOGRAM_LOG_FREQ = 10_000
    LOG_IMAGES = True
    IMAGE_LOG_FREQ = 50_000
    LOG_DIR: Optional[str] = None
    LOG_SHAPE_PLACEMENT_Q_VALUES = True
    SHAPE_Q_LOG_FREQ = 5_000


# --- Demo Mode Visuals ---
class DemoConfig:
    BACKGROUND_COLOR = (10, 10, 20)
    PREVIEW_COLOR = (200, 200, 200, 100)
    INVALID_PREVIEW_COLOR = (255, 0, 0, 100)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = VisConfig.YELLOW
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"
