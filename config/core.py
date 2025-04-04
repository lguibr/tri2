# File: config/core.py
# --- Core Configuration Classes ---
import torch
from typing import Deque, Dict, Any, List, Type, Tuple, Optional

# --- MODIFIED: Import TOTAL_TRAINING_STEPS from general ---
from .general import TOTAL_TRAINING_STEPS

# --- END MODIFIED ---


# --- Visualization (Pygame) ---
class VisConfig:
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0
    LEFT_PANEL_WIDTH = SCREEN_WIDTH // 2  # Adjusted default
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
    # --- FASTER CONFIG (Example, adjust as needed) ---
    NUM_ENVS_TO_RENDER = 48


# --- Environment ---
class EnvConfig:
    # --- FASTER CONFIG (Example, adjust as needed) ---
    NUM_ENVS = 256
    ROWS = 6
    COLS = 10
    # --- END FASTER CONFIG ---

    GRID_FEATURES_PER_CELL = 3  # Occupied, Is_Up, Is_Death
    SHAPE_FEATURES_PER_SHAPE = 5  # num_tris, ups, downs, height, width (normalized)
    NUM_SHAPE_SLOTS = 3

    # --- MODIFIED: State/Action dimensions are now properties ---
    # These will be calculated based on the above constants
    @property
    def GRID_STATE_SHAPE(self) -> Tuple[int, int, int]:
        return (self.GRID_FEATURES_PER_CELL, self.ROWS, self.COLS)

    @property
    def SHAPE_STATE_DIM(self) -> int:
        return self.NUM_SHAPE_SLOTS * self.SHAPE_FEATURES_PER_SHAPE

    # Action dimension remains the same calculation
    @property
    def ACTION_DIM(self) -> int:
        return self.NUM_SHAPE_SLOTS * (self.ROWS * self.COLS)

    # --- END MODIFIED ---


# --- Reward Shaping (RL Reward) ---
class RewardConfig:
    # --- REVISED REWARD STRUCTURE ---
    REWARD_PLACE_PER_TRI = 0.0  # No reward just for placing
    REWARD_CLEAR_1 = 1.0
    REWARD_CLEAR_2 = 3.0  # Bonus for multi-line
    REWARD_CLEAR_3PLUS = 6.0  # Larger bonus
    PENALTY_INVALID_MOVE = -0.1  # Small penalty for trying invalid move
    PENALTY_HOLE_PER_HOLE = -0.05  # Slightly reduced penalty for holes
    PENALTY_GAME_OVER = -1.0  # Penalty for ending the game
    REWARD_ALIVE_STEP = 0.0  # No reward just for surviving
    # --- END REVISED ---


# --- DQN Algorithm ---
class DQNConfig:
    GAMMA = 0.99
    # --- FASTER CONFIG (Example, adjust as needed) ---
    TARGET_UPDATE_FREQ = 5_000
    LEARNING_RATE = 5e-5
    # --- END FASTER CONFIG ---
    ADAM_EPS = 1e-4
    GRADIENT_CLIP_NORM = 10.0
    USE_DOUBLE_DQN = True
    # --- FASTER CONFIG (Example, adjust as needed) ---
    USE_DUELING = True
    USE_NOISY_NETS = True
    USE_DISTRIBUTIONAL = False  # Disable C51 for simplicity/speed initially
    # --- END FASTER CONFIG ---
    V_MIN = -10.0  # Adjusted range if C51 is re-enabled
    V_MAX = 10.0
    NUM_ATOMS = 51
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_T_MAX: int = TOTAL_TRAINING_STEPS
    LR_SCHEDULER_ETA_MIN: float = 1e-7


# --- Training Loop ---
class TrainConfig:
    # --- FASTER CONFIG (Example, adjust as needed) ---
    BATCH_SIZE = 64  # Slightly larger batch size might be more stable
    LEARN_START_STEP = 5_000  # Start learning a bit later
    LEARN_FREQ = 4
    CHECKPOINT_SAVE_FREQ = 20_000
    # --- END FASTER CONFIG ---
    LOAD_CHECKPOINT_PATH: str | None = None
    LOAD_BUFFER_PATH: str | None = None


# --- Replay Buffer ---
class BufferConfig:
    # --- FASTER CONFIG & REVISED DEFAULTS ---
    REPLAY_BUFFER_SIZE = 200_000  # Slightly larger buffer
    USE_N_STEP = True
    N_STEP = 3  # Common N-step value
    USE_PER = True  # Enable PER by default, often helps with sparse rewards
    # --- END REVISED ---
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = TOTAL_TRAINING_STEPS // 2  # Anneal beta over half of training
    PER_EPSILON = 1e-6


# --- Model Architecture ---
class ModelConfig:
    class Network:
        # --- REVISED ARCHITECTURE (Example, adjust based on performance) ---
        # Note: These dimensions depend on the EnvConfig ROWS/COLS above
        HEIGHT = EnvConfig.ROWS
        WIDTH = EnvConfig.COLS
        CONV_CHANNELS = [32, 64, 64]  # Slightly deeper CNN
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        # --- MODIFIED: No Pooling to preserve spatial info ---
        # POOL_KERNEL_SIZE = 2
        # POOL_STRIDE = 2
        # --- END MODIFIED ---
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True  # Keep BatchNorm for stability
        # --- MODIFIED: Shape Embedding ---
        # SHAPE_MLP_HIDDEN_DIM = 64 # Replaced by embedding
        SHAPE_EMBEDDING_DIM = 16  # Dimension for shape type embedding
        SHAPE_FEATURE_MLP_DIMS = [64]  # MLP layers after embedding + features
        # --- END MODIFIED ---
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU
        COMBINED_FC_DIMS = [256, 128]  # Deeper fusion MLP
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True  # Keep BatchNorm
        DROPOUT_FC = 0.0
        # --- END REVISED ---


# --- Statistics and Logging ---
class StatsConfig:
    STATS_AVG_WINDOW = 100
    # --- FASTER CONFIG (Example, adjust as needed) ---
    CONSOLE_LOG_FREQ = 5_000
    # --- END FASTER CONFIG ---


# --- TensorBoard Logging ---
class TensorBoardConfig:
    # --- REVISED DEFAULTS ---
    LOG_HISTOGRAMS = True  # Enable histograms by default for better debugging
    HISTOGRAM_LOG_FREQ = 10_000  # Log histograms reasonably often
    LOG_IMAGES = True  # Enable image logging
    IMAGE_LOG_FREQ = 50_000  # Log images less frequently
    # --- END REVISED ---
    LOG_DIR: Optional[str] = None  # Will be set in config/__init__.py
