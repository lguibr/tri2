# File: config.py
# File: config.py
import torch
import os
from utils.helpers import get_device

# --- General ---
DEVICE = get_device()
RANDOM_SEED = 42
BUFFER_SAVE_PATH = os.path.join("checkpoints", "replay_buffer_state.pkl")


# --- Visualization (Pygame) ---
class VisConfig:
    SCREEN_WIDTH = 1480
    SCREEN_HEIGHT = 720
    VISUAL_STEP_DELAY = 0.01  # Even faster for 1024 envs, adjust if needed
    LEFT_PANEL_WIDTH = SCREEN_WIDTH // 3
    ENV_SPACING = 1  # Reduced spacing for more envs
    FPS = 20  # Keep FPS reasonable
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LIGHTG = (140, 140, 140)
    GRAY = (50, 50, 50)
    RED = (255, 50, 50)
    BLUE = (50, 50, 255)
    YELLOW = (255, 255, 100)
    GOOGLE_COLORS = [(15, 157, 88), (244, 180, 0), (66, 133, 244), (219, 68, 55)]
    MAX_PLOT_POINTS = 5000  # Keep plot points reasonable


# --- Environment ---
class EnvConfig:
    NUM_ENVS = 2048  # <<< INCREASED: Number of parallel environments
    ROWS = 8  # Environment grid size (Keep same for now)
    COLS = 15  # Environment grid size (Keep same for now)
    GRID_FEATURES_PER_CELL = 3
    SHAPE_FEATURES_PER_SHAPE = 5
    NUM_SHAPE_SLOTS = 3
    STATE_DIM = (ROWS * COLS * GRID_FEATURES_PER_CELL) + (
        NUM_SHAPE_SLOTS * SHAPE_FEATURES_PER_SHAPE
    )  # 360 + 15 = 375 (Remains same if ROWS/COLS don't change)
    ACTION_DIM = NUM_SHAPE_SLOTS * (ROWS * COLS)  # 3 * 120 = 360 (Remains same)


# --- Reward Shaping ---
class RewardConfig:  # Keep rewards same unless specific tuning needed
    REWARD_PLACE_PER_TRI = 0.005
    REWARD_CLEAR_1 = 1.0
    REWARD_CLEAR_2 = 3.0
    REWARD_CLEAR_3PLUS = 6.0
    PENALTY_INVALID_MOVE = -0.5
    PENALTY_HOLE_PER_HOLE = -0.1
    PENALTY_GAME_OVER = -10.0


# --- DQN Algorithm ---
class DQNConfig:
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = (
        10000  # <<< INCREASED: Update target less frequently relative to steps
    )
    LEARNING_RATE = 2.5e-5  # <<< DECREASED: Smaller LR for longer training
    ADAM_EPS = 1e-4
    GRADIENT_CLIP_NORM = 10.0


# --- Training Loop ---
class TrainConfig:
    BATCH_SIZE = (
        64  # Keep default batch size (can be tuned based on GPU memory/stability)
    )
    LEARN_START_STEP = 100_000  # <<< INCREASED: Wait longer to fill larger buffer
    TOTAL_TRAINING_STEPS = (
        25_000_000  # <<< INCREASED: Significantly longer training run
    )
    LEARN_FREQ = 8  # Keep same: learn every 8*1024 = 8192 global steps
    CHECKPOINT_SAVE_FREQ = 500_000  # <<< INCREASED: Save less often during longer run


# --- Replay Buffer ---
class BufferConfig:
    REPLAY_BUFFER_SIZE = 500_000  # <<< INCREASED: Larger buffer for longer training
    USE_N_STEP = True
    N_STEP = 10
    USE_PER = True
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 5_000_000  # <<< INCREASED: Anneal beta over longer period
    PER_EPSILON = 1e-6
    LOAD_BUFFER = True  # Keep True if resuming, False if starting clean buffer


# --- Exploration (Epsilon Greedy) ---
class ExplorationConfig:
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY_FRAMES = 5_000_000  # <<< INCREASED: Decay epsilon over much longer period


# --- Model Architecture ---
class ModelConfig:
    MODEL_TYPE = "mixed"
    SAVE_PATH = os.path.join("checkpoints", "dqn_agent_state.pth")
    # <<< WARNING: Set LOAD_MODEL=False or delete checkpoint if previous run used different architecture! >>>
    LOAD_MODEL = True  # Set to False if starting fresh model training
    USE_DUELING = True
    USE_DOUBLE_DQN = True

    class Transformer:  # Unused if MODEL_TYPE='mixed'
        HDIM = 256
        HEADS = 8
        LAYERS = 6
        DROPOUT = 0.1

    class Conv2D:  # Unused if MODEL_TYPE='mixed'
        HEIGHT = EnvConfig.ROWS
        WIDTH = EnvConfig.COLS
        CHANNELS = [32, 64, 64]
        FC_DIM = 512

    class LSTM:  # Unused if MODEL_TYPE='mixed'
        HIDDEN_DIM = 128
        NUM_LAYERS = 1
        FC_DIM = 128

    # <<< UPDATED: Stronger Mixed Model Config >>>
    class Mixed:
        HEIGHT = EnvConfig.ROWS
        WIDTH = EnvConfig.COLS
        # Conv Branch
        CONV_CHANNELS = [64, 128, 128]  # <<< INCREASED Channels
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        POOL_KERNEL_SIZE = 2
        POOL_STRIDE = 2
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True
        # Shape MLP Branch
        SHAPE_MLP_HIDDEN_DIM = 64  # <<< INCREASED Hidden Dim
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU
        # Transformer Integration
        HDIM = 256  # <<< INCREASED Transformer hidden dimension (d_model)
        TRANSFORMER_HEADS = 8  # <<< INCREASED Attention heads (must divide HDIM)
        TRANSFORMER_LAYERS = 4  # <<< INCREASED Transformer encoder layers
        TRANSFORMER_DIM_FEEDFORWARD = HDIM * 4  # Scales automatically
        TRANSFORMER_DROPOUT = 0.1
        USE_LEARNED_POS_EMBEDDING = True
        # Final Head
        COMBINED_FC_DIMS = [512]  # <<< INCREASED FC hidden layer size
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.1


# --- Statistics and Logging ---
class StatsConfig:
    LOG_INTERVAL_STEPS = 10_000  # <<< INCREASED: Log less frequently in terms of steps
    SQLITE_DB_PATH = os.path.join("logs", "training_log.db")
    USE_SQLITE_LOGGING = True
    LOG_TRANSITIONS_TO_DB = False


# --- Check Config Consistency ---
if ModelConfig.MODEL_TYPE in ["conv2d", "mixed"]:
    if EnvConfig.GRID_FEATURES_PER_CELL != 3 and ModelConfig.MODEL_TYPE == "mixed":
        print(
            "Warning: Mixed model assumes 3 features per cell for reshaping. Check EnvConfig/GameState."
        )
if ModelConfig.LOAD_MODEL:
    print(
        "*****************************************************************************************"
    )
    print(
        "*** Warning: LOAD_MODEL is True. Ensure saved checkpoint architecture matches current ***"
    )
    print(
        "*** ModelConfig. Delete checkpoint file if starting with a new/modified architecture. ***"
    )
    print(f"*** Checkpoint path: {ModelConfig.SAVE_PATH} ***")
    print(
        "*****************************************************************************************"
    )

print(
    f"Config Loaded: Env=(R={EnvConfig.ROWS}, C={EnvConfig.COLS}), StateDim={EnvConfig.STATE_DIM}, ActionDim={EnvConfig.ACTION_DIM}, Model={ModelConfig.MODEL_TYPE}, Device={DEVICE}"
)
print(
    f"Training Params: NUM_ENVS={EnvConfig.NUM_ENVS}, TOTAL_STEPS={TrainConfig.TOTAL_TRAINING_STEPS}, BUFFER_SIZE={BufferConfig.REPLAY_BUFFER_SIZE}"
)
