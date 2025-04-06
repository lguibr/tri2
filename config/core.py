# File: config/core.py
import torch
from typing import Deque, Dict, Any, List, Type, Tuple, Optional

from .general import TOTAL_TRAINING_STEPS


class VisConfig:
    NUM_ENVS_TO_RENDER = 16
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00
    LEFT_PANEL_WIDTH = int(SCREEN_WIDTH * 0.7)
    ENV_SPACING = 0
    ENV_GRID_PADDING = 0
    FPS = 0  # Set to 0 for max speed, or > 0 to cap FPS

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
    NUM_ENVS = 256
    ROWS = 8
    COLS = 15
    GRID_FEATURES_PER_CELL = 2
    SHAPE_FEATURES_PER_SHAPE = 5
    NUM_SHAPE_SLOTS = 3
    EXPLICIT_FEATURES_DIM = (
        10  # Keep 10 for now, even if potential outcomes are disabled
    )
    # --- MODIFIED: Disable potential outcome calculation for performance ---
    CALCULATE_POTENTIAL_OUTCOMES_IN_STATE = False
    # --- END MODIFIED ---

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
    # Extrinsic Rewards
    REWARD_PLACE_PER_TRI = 0.01
    REWARD_CLEAR_1 = 1.5
    REWARD_CLEAR_2 = 4.0
    REWARD_CLEAR_3PLUS = 8.0
    REWARD_ALIVE_STEP = 0.001  # Small incentive to stay alive
    PENALTY_INVALID_MOVE = -0.1
    PENALTY_GAME_OVER = -1.5

    # State-Based Penalties (Applied after placement)
    PENALTY_MAX_HEIGHT_FACTOR = -0.005  # Penalty per unit of max height
    PENALTY_BUMPINESS_FACTOR = -0.01  # Penalty per unit of bumpiness
    PENALTY_HOLE_PER_HOLE = -0.07  # Penalty per existing hole
    PENALTY_NEW_HOLE = -0.15  # Additional penalty for newly created holes

    # --- NEW: Potential-Based Reward Shaping (PBRS) ---
    # These coefficients define the potential function:
    # Potential = PBRS_HEIGHT_COEF * max_height + PBRS_HOLE_COEF * num_holes + PBRS_BUMPINESS_COEF * bumpiness
    # The reward added is: gamma * Potential(next_state) - Potential(current_state)
    # Use negative coefficients for things we want to minimize.
    ENABLE_PBRS = True  # Toggle PBRS on/off
    PBRS_HEIGHT_COEF = -0.05  # Encourage reducing height
    PBRS_HOLE_COEF = -0.20  # Strongly encourage reducing holes
    PBRS_BUMPINESS_COEF = -0.02  # Encourage reducing bumpiness
    # --- END NEW ---


class PPOConfig:
    # --- MODIFIED: Adjusted PPO Hyperparameters ---
    LEARNING_RATE = 1e-4  # Slightly lower LR often helps with larger batches/rollouts
    ADAM_EPS = 1e-5
    NUM_STEPS_PER_ROLLOUT = 4096  # Longer rollouts for more stable updates
    PPO_EPOCHS = 6  # Fewer epochs with longer rollouts
    NUM_MINIBATCHES = 64  # Adjust to keep MINIBATCH_SIZE reasonable (e.g., >= 128)
    CLIP_PARAM = 0.1  # Slightly tighter clip range can sometimes stabilize
    GAMMA = 0.995  # Discount factor for future rewards
    GAE_LAMBDA = 0.95  # Factor for Generalized Advantage Estimation
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.01  # Start here, may need tuning (especially if adding Noisy Nets)
    MAX_GRAD_NORM = 0.5
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_END_FRACTION = 0.0  # Decay LR to zero
    # --- END MODIFIED ---

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
        # Calculate batch size, ensure it's at least 1
        batch_size = total_data_per_update // num_minibatches
        # --- MODIFIED: Ensure minimum minibatch size ---
        min_recommended_size = 128
        if batch_size < min_recommended_size:
            print(
                f"Warning: Calculated minibatch size ({batch_size}) is < {min_recommended_size}. Consider adjusting NUM_STEPS_PER_ROLLOUT or NUM_MINIBATCHES."
            )
        return max(1, batch_size)
        # --- END MODIFIED ---


class RNNConfig:
    USE_RNN = True  # Keep LSTM unless performance suggests otherwise
    LSTM_HIDDEN_SIZE = 1024  # Keep or potentially increase if needed
    LSTM_NUM_LAYERS = 1


class TrainConfig:
    # --- MODIFIED: Checkpoint frequency relative to longer rollouts ---
    CHECKPOINT_SAVE_FREQ = 20  # Save every 20 rollouts (adjust based on rollout time)
    # --- END MODIFIED ---
    LOAD_CHECKPOINT_PATH: Optional[str] = None


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        # --- MODIFIED: Potentially increase network capacity ---
        CONV_CHANNELS = [
            96,
            192,
            192,
        ]  # Keep or potentially increase (e.g., [128, 256, 256])
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True

        SHAPE_FEATURE_MLP_DIMS = [192]  # Keep or potentially increase
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU

        COMBINED_FC_DIMS = [2048, 1024]
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.0 


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [10, 50, 100, 500, 1_000, 5_000, 10_000]
    # --- MODIFIED: Adjust console log frequency relative to longer rollouts ---
    CONSOLE_LOG_FREQ = 5  # Log every 5 rollouts
    # --- END MODIFIED ---
    PLOT_DATA_WINDOW = 100_000


class TensorBoardConfig:
    LOG_HISTOGRAMS = True
    # --- MODIFIED: Adjust histogram log frequency relative to longer rollouts ---
    HISTOGRAM_LOG_FREQ = 20  # Log histograms every 20 rollouts
    # --- END MODIFIED ---
    LOG_IMAGES = True
    # --- MODIFIED: Adjust image log frequency relative to longer rollouts ---
    IMAGE_LOG_FREQ = 50  # Log images every 50 rollouts
    # --- END MODIFIED ---
    LOG_DIR: Optional[str] = None
    LOG_SHAPE_PLACEMENT_Q_VALUES = False  # Keep False unless debugging Q-values


class DemoConfig:
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = VisConfig.BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"
