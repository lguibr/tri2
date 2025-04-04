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
    ENV_GRID_PADDING = 1  # Padding inside each environment cell
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
    # --- NEW: Added flash color ---
    LINE_CLEAR_FLASH_COLOR = (180, 180, 220)  # Light purplish-white flash
    # --- END NEW ---


# --- Environment ---
class EnvConfig:
    NUM_ENVS = 256
    # --- MODIFIED DIMENSIONS ---
    ROWS = 8
    COLS = 15
    # --- END MODIFIED ---

    # Features per grid cell: Occupied (1/0), Is_Up (1/0)
    GRID_FEATURES_PER_CELL = 2
    # Features per shape: N_Triangles, N_Up, N_Down, Height, Width (all normalized)
    SHAPE_FEATURES_PER_SHAPE = 5
    NUM_SHAPE_SLOTS = 3  # How many shapes are available to choose from

    @property
    def GRID_STATE_SHAPE(self) -> Tuple[int, int, int]:
        # Shape of the grid input to the CNN: [Channels, Height, Width]
        return (self.GRID_FEATURES_PER_CELL, self.ROWS, self.COLS)

    @property
    def SHAPE_STATE_DIM(self) -> int:
        # Total dimension of the flattened shape features input to the MLP
        return self.NUM_SHAPE_SLOTS * self.SHAPE_FEATURES_PER_SHAPE

    @property
    def ACTION_DIM(self) -> int:
        # Total number of possible discrete actions (Shape_Slot * Grid_Position)
        # The network outputs Q-values for all these potential actions.
        # Invalid actions (based on current state) are masked out during selection.
        return self.NUM_SHAPE_SLOTS * (self.ROWS * self.COLS)


# --- Reward Shaping (RL Reward) ---
class RewardConfig:
    REWARD_PLACE_PER_TRI = 0.0  # Small reward for placing triangles (can be non-zero)
    REWARD_CLEAR_1 = 1.0
    REWARD_CLEAR_2 = 3.0
    REWARD_CLEAR_3PLUS = 6.0
    PENALTY_INVALID_MOVE = -0.1  # Penalty for attempting an impossible placement
    PENALTY_HOLE_PER_HOLE = -0.05  # Penalty per empty cell below an occupied cell
    PENALTY_GAME_OVER = -1.0  # Large penalty for losing the game
    REWARD_ALIVE_STEP = 0.0  # Small reward for surviving each step (can be non-zero)


# --- DQN Algorithm ---
class DQNConfig:
    GAMMA = 0.99  # Discount factor for future rewards
    TARGET_UPDATE_FREQ = (
        5_000  # How often to copy online network weights to target network (steps)
    )
    LEARNING_RATE = 5e-5  # AdamW optimizer learning rate
    ADAM_EPS = 1e-4  # AdamW epsilon for numerical stability
    GRADIENT_CLIP_NORM = 10.0  # Max norm for gradient clipping (0 to disable)
    USE_DOUBLE_DQN = True  # Use Double DQN target calculation
    USE_DUELING = True  # Use Dueling Network Architecture
    USE_NOISY_NETS = (
        True  # Use Noisy Linear layers for exploration (instead of epsilon-greedy)
    )
    # --- Distributional (C51) DQN Settings ---
    USE_DISTRIBUTIONAL = (
        False  # Set to True to use C51 (experimental, requires more tuning)
    )
    V_MIN = -10.0  # Minimum value for distributional support
    V_MAX = 10.0  # Maximum value for distributional support
    NUM_ATOMS = 51  # Number of atoms in the distribution
    # --- Learning Rate Scheduler ---
    USE_LR_SCHEDULER = True  # Use Cosine Annealing LR scheduler
    LR_SCHEDULER_T_MAX: int = TOTAL_TRAINING_STEPS  # Scheduler period (total steps)
    LR_SCHEDULER_ETA_MIN: float = 1e-7  # Minimum learning rate for scheduler


# --- Training Loop ---
class TrainConfig:
    BATCH_SIZE = 64  # Number of experiences sampled per training step
    LEARN_START_STEP = 5_000  # Global steps before learning starts (buffer warmup)
    LEARN_FREQ = 4  # Perform a learning update every N global steps
    CHECKPOINT_SAVE_FREQ = (
        20_000  # Save model and buffer state every N global steps (0 to disable)
    )
    LOAD_CHECKPOINT_PATH: str | None = (
        None  # Path to agent .pth file to load (optional)
    )
    LOAD_BUFFER_PATH: str | None = None  # Path to buffer .pkl file to load (optional)


# --- Replay Buffer ---
class BufferConfig:
    REPLAY_BUFFER_SIZE = 200_000  # Maximum number of transitions in the buffer
    USE_N_STEP = True  # Use N-Step returns
    N_STEP = 3  # Number of steps for N-Step returns
    USE_PER = True  # Use Prioritized Experience Replay
    PER_ALPHA = 0.6  # PER exponent α (controls prioritization intensity)
    PER_BETA_START = 0.4  # Initial PER importance sampling exponent β
    PER_BETA_FRAMES = TOTAL_TRAINING_STEPS // 2  # Steps over which β anneals to 1.0
    PER_EPSILON = 1e-6  # Small value added to priorities to ensure non-zero probability


# --- Model Architecture ---
class ModelConfig:
    class Network:  # Nested class for network-specific parameters
        # Get grid dimensions dynamically from EnvConfig to avoid hardcoding
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance  # Avoid keeping instance here
        # --- CNN Branch ---
        CONV_CHANNELS = [32, 64, 64]  # Output channels for each conv layer
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1  # Use 'same' padding equivalent if stride=1, kernel=3
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True  # Use BatchNorm after conv layers (before activation)
        # --- Shape MLP Branch ---
        # Note: Input dim is calculated automatically (NUM_SHAPE_SLOTS * SHAPE_FEATURES_PER_SHAPE)
        SHAPE_FEATURE_MLP_DIMS = [64]  # Hidden layer dimensions for shape feature MLP
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU
        # --- Fusion MLP Branch ---
        # Note: Input dim is calculated automatically (CNN_out_flat + ShapeMLP_out)
        COMBINED_FC_DIMS = [256, 128]  # Hidden layer dimensions after fusion
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True  # Use BatchNorm in fusion MLP layers
        DROPOUT_FC = 0.0  # Dropout probability in fusion MLP layers (0 to disable)


# --- Statistics and Logging ---
class StatsConfig:
    STATS_AVG_WINDOW = (
        100  # Window size for calculating rolling averages (episodes/steps)
    )
    CONSOLE_LOG_FREQ = (
        5_000  # Log summary to console every N global steps (0 to disable)
    )


# --- TensorBoard Logging ---
class TensorBoardConfig:
    LOG_HISTOGRAMS = True  # Log histograms of weights, biases, Q-values, etc.
    HISTOGRAM_LOG_FREQ = 10_000  # Log histograms every N global steps
    LOG_IMAGES = True  # Log sample environment states as images
    IMAGE_LOG_FREQ = 50_000  # Log images every N global steps
    LOG_DIR: Optional[str] = None  # Set automatically in config/__init__.py
    # --- Specific Histogram Logging ---
    LOG_SHAPE_PLACEMENT_Q_VALUES = (
        True  # Log Q-value distributions for shapes/placements
    )
    SHAPE_Q_LOG_FREQ = 5_000  # How often to log these specific Q-value histograms


# --- NEW: Demo Mode Visuals (Optional) ---
class DemoConfig:
    BACKGROUND_COLOR = (10, 10, 20)  # Slightly different background
    PREVIEW_COLOR = (200, 200, 200, 100)  # Semi-transparent white for placement preview
    INVALID_PREVIEW_COLOR = (255, 0, 0, 100)  # Red preview if invalid
    SELECTED_SHAPE_HIGHLIGHT_COLOR = VisConfig.YELLOW
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"
