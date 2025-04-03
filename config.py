import torch
import os
from utils.helpers import get_device

DEVICE = get_device()
RANDOM_SEED = 42
BUFFER_SAVE_PATH = os.path.join("checkpoints", "replay_buffer_state.pkl")


class VisConfig:
    SCREEN_WIDTH = 1480 
    SCREEN_HEIGHT = 720
    LEFT_PANEL_WIDTH = 400
    ENV_SPACING = 10
    FPS = 60
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LIGHTG = (140, 140, 140)
    GRAY = (50, 50, 50)
    RED = (255, 50, 50)  
    BLUE = (50, 50, 255) 
    YELLOW = (255, 255, 100)  
    GOOGLE_COLORS = [
        (15, 157, 88),
        (244, 180, 0),
        (66, 133, 244),
        (219, 68, 55),
    ]
    # Plotting
    MAX_PLOT_POINTS = 1000
    SHAPE_PREVIEW_X = 10
    SHAPE_PREVIEW_Y_START = 500 
    SHAPE_PREVIEW_HEIGHT = 100
    SHAPE_PREVIEW_BG = (40, 40, 40)
    SHAPE_CELL_SIZE = 10  


# --- Environment ---
class EnvConfig:
    NUM_ENVS = 20
    ROWS = 8
    COLS = 15
    STATE_DIM = (ROWS * COLS * 3) + (
        3 * 5
    )  # 3 values per cell + 5 values per shape * 3 shapes
    ACTION_DIM = 3 * (ROWS * COLS)  # 3 shapes * rows * cols potential placements


# --- DQN Algorithm ---
class DQNConfig:
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = 2500
    LEARNING_RATE = 5e-5
    ADAM_EPS = 1e-4
    GRADIENT_CLIP_NORM = 10.0


# --- Training Loop ---
class TrainConfig:
    BATCH_SIZE = 64
    LEARN_START_STEP = 10000
    TOTAL_TRAINING_STEPS = 5_000_000
    LEARN_FREQ = 9  
    CHECKPOINT_SAVE_FREQ = 100000  


# --- Replay Buffer ---
class BufferConfig:
    REPLAY_BUFFER_SIZE = 100000
    # N-Step Returns
    USE_N_STEP = True
    N_STEP = 10  
    # Prioritized Replay (PER)
    USE_PER = True
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 1000000
    PER_EPSILON = 1e-6
    LOAD_BUFFER = True  


# --- Exploration (Epsilon Greedy) ---
class ExplorationConfig:
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY_FRAMES = 1000000


# --- Model Architecture ---
class ModelConfig:
    MODEL_TYPE = "transformer"  # "transformer", "conv2d", "lstm", "mixed"
    SAVE_PATH = os.path.join("checkpoints", "dqn_agent_state.pth")
    LOAD_MODEL = True

    USE_DUELING = True
    USE_DOUBLE_DQN = True

    class Transformer:
        HDIM = 256
        HEADS = 8
        LAYERS = 6
        DROPOUT = 0.1

    class Conv2D:
        HEIGHT = EnvConfig.ROWS  
        WIDTH = EnvConfig.COLS  
        CHANNELS = [32, 64, 64]
        FC_DIM = 512

    class LSTM:
        HIDDEN_DIM = 128
        NUM_LAYERS = 1
        FC_DIM = 128

    class Mixed:
        CONV_CHANNELS = [64, 128, 128]
        NUM_FC_LAYERS = 4
        USE_BN = True
        DROPOUT = 0.1
        LSTM_HIDDEN_DIM = 256
        HEIGHT = EnvConfig.ROWS 
        WIDTH = EnvConfig.COLS  
        FC_DIM = 256


# --- Statistics and Logging ---
class StatsConfig:
    LOG_INTERVAL_STEPS = 1000
    SQLITE_DB_PATH = os.path.join("logs", "training_log.db")
    USE_SQLITE_LOGGING = True
    LOG_TRANSITIONS_TO_DB = False  # Caution: Can create very large DB files
