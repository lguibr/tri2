# File: config.py
import torch
import os
import time
from utils.helpers import get_device
from typing import Deque, Dict, Any  # Added Dict, Any

# --- General ---
DEVICE = get_device()
RANDOM_SEED = 42
# Create a unique run ID for logging purposes
RUN_ID = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
BASE_CHECKPOINT_DIR = "checkpoints"
BASE_LOG_DIR = "logs"
BUFFER_SAVE_PATH = os.path.join(BASE_CHECKPOINT_DIR, RUN_ID, "replay_buffer_state.pkl")
MODEL_SAVE_PATH = os.path.join(BASE_CHECKPOINT_DIR, RUN_ID, "dqn_agent_state.pth")


# --- TensorBoard Logging ---
class TensorBoardConfig:
    LOG_DIR = os.path.join(BASE_LOG_DIR, "tensorboard", RUN_ID)
    LOG_HISTOGRAMS = True  # Log distributions of Q-values, errors, etc.
    HISTOGRAM_LOG_FREQ = 10_000  # How often (in global steps) to log histograms
    LOG_IMAGES = (
        False  # Log sample environment states periodically (can be slow/large logs)
    )
    IMAGE_LOG_FREQ = 500_000  # How often (in global steps) to log images if enabled


# --- Visualization (Pygame) ---
class VisConfig:
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.001
    LEFT_PANEL_WIDTH = 350
    ENV_SPACING = 1
    FPS = 60
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LIGHTG = (140, 140, 140)
    GRAY = (50, 50, 50)
    RED = (255, 50, 50)
    BLUE = (50, 50, 255)
    YELLOW = (255, 255, 100)
    GOOGLE_COLORS = [(15, 157, 88), (244, 180, 0), (66, 133, 244), (219, 68, 55)]
    NUM_ENVS_TO_RENDER = 64


# --- Environment ---
class EnvConfig:
    NUM_ENVS = 256
    ROWS = 8
    COLS = 15
    GRID_FEATURES_PER_CELL = 3  # Occupied, Is_Up, Is_Death
    SHAPE_FEATURES_PER_SHAPE = 5  # N_Tris, Ups, Downs, Height, Width (Normalized)
    NUM_SHAPE_SLOTS = 3
    STATE_DIM = (ROWS * COLS * GRID_FEATURES_PER_CELL) + (
        NUM_SHAPE_SLOTS * SHAPE_FEATURES_PER_SHAPE
    )
    ACTION_DIM = NUM_SHAPE_SLOTS * (ROWS * COLS)


# --- Reward Shaping (RL Reward) ---
class RewardConfig:
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
    TARGET_UPDATE_FREQ = 50_000
    LEARNING_RATE = 1e-5
    ADAM_EPS = 1e-4
    GRADIENT_CLIP_NORM = 10.0
    USE_NOISY_NETS = True  # Hardcoded: Always use Noisy Nets
    USE_DOUBLE_DQN = True
    USE_DUELING = True


# --- Training Loop ---
class TrainConfig:
    BATCH_SIZE = 64
    LEARN_START_STEP = 20_000
    TOTAL_TRAINING_STEPS = 10_000_000
    LEARN_FREQ = 8
    CHECKPOINT_SAVE_FREQ = 100_000
    LOAD_CHECKPOINT_PATH: str | None = (
        None  # Specify path to load a specific checkpoint
    )
    LOAD_BUFFER_PATH: str | None = None  # Specify path to load a specific buffer


# --- Replay Buffer ---
class BufferConfig:
    REPLAY_BUFFER_SIZE = 1_000_000
    USE_N_STEP = True
    N_STEP = 15
    USE_PER = True
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 25_000_000
    PER_EPSILON = 1e-6


# --- Model Architecture ---
class ModelConfig:
    class Network:
        HEIGHT = EnvConfig.ROWS
        WIDTH = EnvConfig.COLS
        CONV_CHANNELS = [64, 128, 256]
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        POOL_KERNEL_SIZE = 2
        POOL_STRIDE = 2
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True
        SHAPE_MLP_HIDDEN_DIM = 128
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU
        COMBINED_FC_DIMS = [1024, 512]
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.1


# --- Statistics and Logging ---
class StatsConfig:
    STATS_AVG_WINDOW = (
        500  # Window for calculating rolling averages (in-memory UI stats)
    )
    CONSOLE_LOG_FREQ = 50_000  # How often to print stats summary to console


# --- Helper Function for Config Dict ---
def get_config_dict() -> Dict[str, Any]:
    """Returns a flat dictionary of all relevant config values for logging."""
    all_configs = {}

    # Helper to flatten classes (excluding methods, dunders, ClassVars, nested Classes)
    def flatten_class(cls, prefix=""):
        d = {}
        for k, v in cls.__dict__.items():
            # Basic check: not dunder, not callable, not a type (nested class)
            if not k.startswith("__") and not callable(v) and not isinstance(v, type):
                # More robust checking might be needed for complex configs
                d[f"{prefix}{k}"] = v
        return d

    all_configs.update(flatten_class(VisConfig, "Vis."))
    all_configs.update(flatten_class(EnvConfig, "Env."))
    all_configs.update(flatten_class(RewardConfig, "Reward."))
    all_configs.update(flatten_class(DQNConfig, "DQN."))
    all_configs.update(flatten_class(TrainConfig, "Train."))
    all_configs.update(flatten_class(BufferConfig, "Buffer."))
    all_configs.update(
        flatten_class(ModelConfig.Network, "Model.Net.")
    )  # Flatten sub-class explicitly
    all_configs.update(flatten_class(StatsConfig, "Stats."))
    all_configs.update(flatten_class(TensorBoardConfig, "TB."))

    # Add general constants
    all_configs["General.DEVICE"] = str(DEVICE)
    all_configs["General.RANDOM_SEED"] = RANDOM_SEED
    all_configs["General.RUN_ID"] = RUN_ID

    # Filter out None values specifically for paths that might not be set
    # This prevents logging None for load paths if they aren't specified
    all_configs = {
        k: v for k, v in all_configs.items() if not (k.endswith("_PATH") and v is None)
    }

    # Convert torch.nn activation functions to string representation for logging
    for key, value in all_configs.items():
        if isinstance(value, type) and issubclass(value, torch.nn.Module):
            all_configs[key] = value.__name__
        # Handle potential list values (e.g., CONV_CHANNELS) - convert to string?
        # Tensorboard hparams prefers simple types (int, float, bool, str)
        if isinstance(value, list):
            all_configs[key] = str(value)

    return all_configs


# --- Config Consistency Checks & Info ---
print("-" * 70)
print(f"RUN ID: {RUN_ID}")
print(f"Log Directory: {TensorBoardConfig.LOG_DIR}")
print(f"Checkpoint Directory: {os.path.dirname(MODEL_SAVE_PATH)}")
print(f"Device: {DEVICE}")
print(
    f"TB Logging: Histograms={'ON' if TensorBoardConfig.LOG_HISTOGRAMS else 'OFF'}, Images={'ON' if TensorBoardConfig.LOG_IMAGES else 'OFF'}"
)


if EnvConfig.GRID_FEATURES_PER_CELL != 3:
    print(
        "Warning: Network assumes 3 features per cell (Occupied, Is_Up, Is_Death). Check EnvConfig/GameState."
    )

if TrainConfig.LOAD_CHECKPOINT_PATH:
    print("*" * 70)
    print(
        f"*** Warning: Attempting to LOAD CHECKPOINT from: {TrainConfig.LOAD_CHECKPOINT_PATH} ***"
    )
    print("*** Ensure saved checkpoint matches current ModelConfig. ***")
    print("*" * 70)
else:
    print("--- Starting training from scratch (no checkpoint specified to load). ---")

if TrainConfig.LOAD_BUFFER_PATH:
    print("*" * 70)
    print(
        f"*** Warning: Attempting to LOAD BUFFER from: {TrainConfig.LOAD_BUFFER_PATH} ***"
    )
    print("*** Ensure saved buffer matches current BufferConfig (PER, N-Step). ***")
    print("*" * 70)
else:
    print("--- Starting with an empty replay buffer (no buffer specified to load). ---")


print("--- Using Noisy Nets for exploration (Epsilon-greedy settings removed) ---")

print(
    f"Config: Env=(R={EnvConfig.ROWS}, C={EnvConfig.COLS}), StateDim={EnvConfig.STATE_DIM}, ActionDim={EnvConfig.ACTION_DIM}"
)
print(
    f"Network: CNN={ModelConfig.Network.CONV_CHANNELS}, ShapeMLP={ModelConfig.Network.SHAPE_MLP_HIDDEN_DIM}, Fusion={ModelConfig.Network.COMBINED_FC_DIMS}, Dueling={DQNConfig.USE_DUELING}, Noisy={DQNConfig.USE_NOISY_NETS}"
)
print(
    f"Training: NUM_ENVS={EnvConfig.NUM_ENVS}, TOTAL_STEPS={TrainConfig.TOTAL_TRAINING_STEPS/1e6:.1f}M, BUFFER={BufferConfig.REPLAY_BUFFER_SIZE/1e6:.1f}M, BATCH={TrainConfig.BATCH_SIZE}, N_STEP={BufferConfig.N_STEP if BufferConfig.USE_N_STEP else 'N/A'}"
)
print(f"Buffer: PER={BufferConfig.USE_PER}, N-Step={BufferConfig.USE_N_STEP}")
print(
    f"Stats: AVG_WINDOW={StatsConfig.STATS_AVG_WINDOW}, Console Log Freq={StatsConfig.CONSOLE_LOG_FREQ}"
)

if EnvConfig.NUM_ENVS >= 1024:
    print("*" * 70)
    print(f"*** Warning: NUM_ENVS={EnvConfig.NUM_ENVS}. Monitor system resources. ***")
    if DEVICE.type == "mps":
        print(
            "*** Using MPS device. Performance varies. Force CPU via env var if needed. ***"
        )
    print("*" * 70)

print(
    f"--- Rendering {VisConfig.NUM_ENVS_TO_RENDER if VisConfig.NUM_ENVS_TO_RENDER > 0 else 'ALL'} of {EnvConfig.NUM_ENVS} environments ---"
)
print("-" * 70)
