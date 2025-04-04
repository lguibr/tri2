# File: config/core.py
# --- Core Configuration Classes ---
import torch
from typing import Deque, Dict, Any, List, Type

from .general import TOTAL_TRAINING_STEPS


# --- Visualization (Pygame) ---
class VisConfig:
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0  
    LEFT_PANEL_WIDTH = SCREEN_WIDTH // 2
    ENV_SPACING = 6
    FPS = 0  # Set to 0 for uncapped FPS (max speed)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LIGHTG = (140, 140, 140)
    GRAY = (50, 50, 50)
    RED = (255, 50, 50)
    BLUE = (50, 50, 255)
    YELLOW = (255, 255, 100)
    GOOGLE_COLORS = [(15, 157, 88), (244, 180, 0), (66, 133, 244), (219, 68, 55)]
    # --- FASTER CONFIG ---
    NUM_ENVS_TO_RENDER = 48  


# --- Environment ---
class EnvConfig:
    # --- FASTER CONFIG ---
    NUM_ENVS = 256  # Reduce parallel envs significantly
    ROWS = 6  # Smaller grid
    COLS = 10  # Smaller grid
    # --- END FASTER CONFIG ---
    GRID_FEATURES_PER_CELL = 3
    SHAPE_FEATURES_PER_SHAPE = 5
    NUM_SHAPE_SLOTS = 3
    # STATE_DIM and ACTION_DIM will be recalculated based on ROWS/COLS
    STATE_DIM = (ROWS * COLS * GRID_FEATURES_PER_CELL) + (
        NUM_SHAPE_SLOTS * SHAPE_FEATURES_PER_SHAPE
    )
    ACTION_DIM = NUM_SHAPE_SLOTS * (ROWS * COLS)


# --- Reward Shaping (RL Reward) ---
class RewardConfig:  # Keep rewards the same unless debugging reward issues
    REWARD_PLACE_PER_TRI = 0.005
    REWARD_CLEAR_1 = 1.0
    REWARD_CLEAR_2 = 3.0
    REWARD_CLEAR_3PLUS = 6.0
    PENALTY_INVALID_MOVE = -0.5
    PENALTY_HOLE_PER_HOLE = -0.1
    PENALTY_GAME_OVER = -10.0
    REWARD_ALIVE_STEP = 0.001


# --- DQN Algorithm ---
class DQNConfig:
    GAMMA = 0.99
    # --- FASTER CONFIG ---
    TARGET_UPDATE_FREQ = 5_000  # Update target net much more often
    LEARNING_RATE = 5e-5  # Slightly higher LR might speed up initial learning
    # --- END FASTER CONFIG ---
    ADAM_EPS = 1e-4
    GRADIENT_CLIP_NORM = 10.0
    USE_DOUBLE_DQN = True
    # --- FASTER CONFIG ---
    USE_DUELING = True  # Disable Dueling (simpler head)
    USE_NOISY_NETS = True  # Keep Noisy for exploration unless debugging exploration
    USE_DISTRIBUTIONAL = True  # Disable C51 (significant computation reduction)
    # --- END FASTER CONFIG ---
    V_MIN = -15.0  # Ignored if USE_DISTRIBUTIONAL is False
    V_MAX = 15.0  # Ignored if USE_DISTRIBUTIONAL is False
    NUM_ATOMS = 51  # Ignored if USE_DISTRIBUTIONAL is False
    USE_LR_SCHEDULER = True  # Keep scheduler unless debugging LR issues
    LR_SCHEDULER_T_MAX: int = TOTAL_TRAINING_STEPS  # Keep T_max related to total steps
    LR_SCHEDULER_ETA_MIN: float = 1e-7


# --- Training Loop ---
class TrainConfig:
    # --- FASTER CONFIG ---
    BATCH_SIZE = 32  # Smaller batch size for more frequent updates
    LEARN_START_STEP = 1_000  # Start learning much earlier
    LEARN_FREQ = 4  # Learn more frequently relative to env steps
    CHECKPOINT_SAVE_FREQ = 20_000  # Save less often
    # --- END FASTER CONFIG ---
    LOAD_CHECKPOINT_PATH: str | None = None
    LOAD_BUFFER_PATH: str | None = None


# --- Replay Buffer ---
class BufferConfig:
    # --- FASTER CONFIG ---
    REPLAY_BUFFER_SIZE = 100_000  # Smaller buffer uses less RAM, faster init/save/load
    USE_N_STEP = True
    N_STEP = 5  # Smaller N-step lookahead
    USE_PER = False  # Disable PER (faster sampling)
    # --- END FASTER CONFIG ---
    PER_ALPHA = 0.6  # Ignored if USE_PER is False
    PER_BETA_START = 0.4  # Ignored if USE_PER is False
    PER_BETA_FRAMES = 25_000_000  # Ignored if USE_PER is False
    PER_EPSILON = 1e-6  # Ignored if USE_PER is False


# --- Model Architecture ---
class ModelConfig:
    class Network:
        # --- FASTER CONFIG ---
        # NOTE: These dimensions depend on the EnvConfig ROWS/COLS above
        # You might need to adjust pooling/strides if the grid gets too small
        HEIGHT = EnvConfig.ROWS  # Use updated EnvConfig
        WIDTH = EnvConfig.COLS  # Use updated EnvConfig
        CONV_CHANNELS = [32, 64]  # Fewer/smaller CNN layers
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        POOL_KERNEL_SIZE = 2
        POOL_STRIDE = 2  # Ensure pooling doesn't reduce dimensions below 1x1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True  # Disable BatchNorm
        SHAPE_MLP_HIDDEN_DIM = 64  # Smaller shape MLP
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU
        COMBINED_FC_DIMS = [256]  # Smaller/fewer fusion layers
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True  # Disable BatchNorm
        DROPOUT_FC = 0.0  # Disable Dropout
        # --- END FASTER CONFIG ---


# --- Statistics and Logging ---
class StatsConfig:
    STATS_AVG_WINDOW = 100  # Smaller window for faster reflection of changes
    # --- FASTER CONFIG ---
    CONSOLE_LOG_FREQ = 5_000  # Log to console more often to see progress
    # --- END FASTER CONFIG ---


# --- TensorBoard Logging ---
class TensorBoardConfig:
    # --- FASTER CONFIG ---
    LOG_HISTOGRAMS = False  # Disable histograms (major speedup)
    HISTOGRAM_LOG_FREQ = 100_000  # Ignored if LOG_HISTOGRAMS is False
    LOG_IMAGES = False  # Disable image logging
    IMAGE_LOG_FREQ = 500_000  # Ignored if LOG_IMAGES is False
    # --- END FASTER CONFIG ---
