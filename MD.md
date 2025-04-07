Okay, let's start by identifying the files that are primarily related to the PPO implementation and can likely be removed entirely in the transition to an AlphaZero approach.

**Files to Delete:**

1.  `agent/ppo_agent.py`
2.  `agent/model_factory.py`
3.  `agent/networks/agent_network.py`
4.  `training/rollout_collector.py`
5.  `training/rollout_storage.py`
6.  `training/collector_logic.py`
7.  `training/collector_state.py`
8.  `training/env_interaction.py`
9.  `training/normalization.py`
10. `training/trainer.py`
11. `workers.py`
12. `rl_components/env_runner.py` (Likely duplicate/old)
13. `rl_components/trainer.py` (Likely duplicate/old)
14. `init/rl_components_ppo.py`

---

Now, let's modify the remaining files to remove PPO dependencies and streamline the codebase.

**Files to Modify:**

1.  `config/core.py`: Remove PPOConfig, RewardConfig (PBRS parts), ObsNormConfig.
2.  `config/general.py`: Remove TOTAL_TRAINING_STEPS.
3.  `config/__init__.py`: Update exports based on removed config classes.
4.  `config/utils.py`: Remove flattening of deleted config classes.
5.  `config/validation.py`: Remove validation/printing of deleted config classes.
6.  `stats/aggregator_storage.py`: Remove deques related to PPO losses, entropy, grad norms, SPS.
7.  `stats/aggregator_logic.py`: Remove logic for updating PPO-specific metrics.
8.  `stats/aggregator.py`: Adapt to removed metrics.
9.  `stats/tb_scalar_logger.py`: Remove logging of PPO-specific scalars.
10. `stats/tensorboard_logger.py`: Adapt calls, remove PPO-specific graph/hparam logic if desired for now.
11. `stats/simple_stats_recorder.py`: Adapt console logging format.
12. `app_init.py`: Remove initialization of deleted components (Agent, Collector, etc.) and related configs. Simplify `initialize_rl_components`.
13. `app_logic.py`: Remove `toggle_training_run`, PPO status updates, PPO checkpoint logic. Adapt cleanup.
14. `app_workers.py`: Remove PPO worker management. Keep the class structure for potential future workers (MCTS/NN).
15. `main_pygame.py`: Remove imports, instantiation, and logic related to deleted components. Remove worker management calls. Adapt main loop.
16. `ui/input_handler.py`: Remove `toggle_training_run_cb` and related button logic.
17. `ui/panels/left_panel_components/button_status_renderer.py`: Remove rendering of Run/Stop button, progress bars, SPS. Adapt status display.
18. `ui/panels/left_panel_components/info_text_renderer.py`: Remove rendering of worker counts, LR, network details specific to PPO actor-critic. Adapt param count display if needed later.
19. `ui/panels/left_panel.py`: Adapt calls to sub-renderers, remove passing of PPO-specific data.
20. `ui/plotter.py`: Remove plots for PPO losses, entropy, SPS, LR.
21. `ui/plot_utils.py`: No changes needed for now, but might adapt trend logic later if metrics change.
22. `ui/renderer.py`: Remove passing of PPO-specific data (workers, update progress) to sub-renderers.
23. `utils/init_checks.py`: Remove PBRS check if RewardConfig is simplified.
24. `utils/__init__.py`: Update exports if needed.
25. `requirements.txt`: Remove `cloudpickle` if checkpointing is removed/changed significantly (keeping it is likely fine).

---

**Updated File Contents:**

File: `config/core.py`
Explanation: Removed PPOConfig, RewardConfig, ObsNormConfig. Simplified TrainConfig.
```python
# File: config/core.py
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
    NUM_ENVS_TO_RENDER = 16  # Keep for potential multi-env visualization later
    FPS = 60 # Set a default FPS for UI responsiveness
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00
    LEFT_PANEL_RATIO = 0.7
    ENV_SPACING = 2 # Added minimal spacing
    ENV_GRID_PADDING = 2 # Added minimal padding

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
    # Keep NUM_ENVS for potential future use, but maybe default to 1 for now?
    # Let's keep it higher for UI layout testing, but core logic will focus on demo_env.
    NUM_ENVS = 1 # Set to 1, as we removed multi-env collection logic
    ROWS = 8
    COLS = 15
    GRID_FEATURES_PER_CELL = 2
    SHAPE_FEATURES_PER_SHAPE = 5
    NUM_SHAPE_SLOTS = 3
    EXPLICIT_FEATURES_DIM = 10
    CALCULATE_POTENTIAL_OUTCOMES_IN_STATE = False # Keep False for now

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
        # Action space might change for MCTS (e.g., just placement coords)
        # Keep original for now, but likely needs refactoring later.
        return self.NUM_SHAPE_SLOTS * (self.ROWS * self.COLS)

# Removed RewardConfig - Reward logic will be internal to the MCTS/search or env step

class RNNConfig: # Keep for potential future NN architectures
    USE_RNN = False # Default to False
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2

class TransformerConfig: # Keep for potential future NN architectures
    USE_TRANSFORMER = False # Default to False
    TRANSFORMER_D_MODEL = 256
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_DIM_FEEDFORWARD = 512
    TRANSFORMER_NUM_LAYERS = 3
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "relu"

class TrainConfig: # Simplified for general training/checkpointing
    # Checkpointing might be handled differently (e.g., saving NN weights + MCTS stats)
    CHECKPOINT_SAVE_FREQ = 50 # Frequency might mean something else now (e.g., training steps, games)
    LOAD_CHECKPOINT_PATH: Optional[str] = None # Keep ability to load specific NN weights

class ModelConfig: # Keep structure for the AlphaZero NN
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        # Keep CNN/MLP structure as a potential base for the NN feature extractor
        CONV_CHANNELS = [64, 128, 256]
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True

        SHAPE_FEATURE_MLP_DIMS = [128, 128]
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU

        # Combined MLP dims might change based on NN architecture
        COMBINED_FC_DIMS = [1024, 256] # Example, adjust later
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.0

class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [25, 50, 100]
    CONSOLE_LOG_FREQ = 1 # Log every episode/game now? Or time based? Set to 1 for frequent updates.
    PLOT_DATA_WINDOW = 100_000 # Keep large window

class TensorBoardConfig:
    LOG_HISTOGRAMS = False # Keep ability to log histograms (e.g., NN weights)
    HISTOGRAM_LOG_FREQ = 20 # Frequency might mean training steps/games
    LOG_IMAGES = False # Keep ability to log images (e.g., board states)
    IMAGE_LOG_FREQ = 20 # Frequency might mean training steps/games
    LOG_DIR: Optional[str] = None
    LOG_SHAPE_PLACEMENT_Q_VALUES = False # Remove PPO-specific Q-value logging

class DemoConfig: # Keep as is
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit" # Keep old text for now
    DEBUG_HELP_TEXT = "[Click]=Toggle Triangle | [R]=Reset Grid | [ESC]=Exit"

```

File: `config/general.py`
Explanation: Removed `TOTAL_TRAINING_STEPS`.
```python
# File: config/general.py
import torch
import os
import time
from typing import Optional

# --- Base Directories ---
BASE_CHECKPOINT_DIR = "checkpoints"
BASE_LOG_DIR = "logs"

# --- Device ---
DEVICE: Optional[torch.device] = None


def set_device(device: torch.device):
    """Sets the global DEVICE variable."""
    global DEVICE
    DEVICE = device
    print(f"[Config] Global DEVICE set to: {DEVICE}")


# --- Random Seed ---
RANDOM_SEED = 42

# --- Run ID and Paths (Dynamically Determined) ---
_current_run_id: Optional[str] = None


def get_run_id() -> str:
    """Gets the current run ID, generating one if not set."""
    global _current_run_id
    if _current_run_id is None:
        _current_run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"[Config] Generated new RUN_ID: {_current_run_id}")
    return _current_run_id


def set_run_id(run_id: str):
    """Sets the run ID, typically when resuming a run."""
    global _current_run_id
    if _current_run_id is not None and _current_run_id != run_id:
        print(
            f"[Config] WARNING: Overwriting existing RUN_ID '{_current_run_id}' with '{run_id}'."
        )
    elif _current_run_id is None:
        print(f"[Config] Setting RUN_ID to resumed ID: {run_id}")
    _current_run_id = run_id


def get_run_checkpoint_dir() -> str:
    """Gets the checkpoint directory for the current run."""
    # Checkpoints will now likely store NN weights, maybe MCTS stats
    return os.path.join(BASE_CHECKPOINT_DIR, get_run_id())


def get_run_log_dir() -> str:
    """Gets the TensorBoard log directory for the current run."""
    # Ensure the base 'tensorboard' subdirectory exists within BASE_LOG_DIR
    tb_base = os.path.join(BASE_LOG_DIR, "tensorboard")
    return os.path.join(tb_base, get_run_id())


def get_console_log_dir() -> str:
    """Gets the directory for console logs for the current run."""
    # Place console logs directly within the run-specific log directory
    return get_run_log_dir()


def get_model_save_path() -> str:
    """Gets the base model save path for the current run (adapt name later)."""
    # Rename from ppo_agent_state to something more generic like 'alphatri_nn.pth'
    return os.path.join(get_run_checkpoint_dir(), "alphatri_nn.pth")

# Removed TOTAL_TRAINING_STEPS

```

File: `config/__init__.py`
Explanation: Removed exports for PPOConfig, RewardConfig, ObsNormConfig, TOTAL_TRAINING_STEPS.
```python
# config/__init__.py
# This file marks the 'config' directory as a Python package.

# Import core configuration classes to make them available directly under 'config'
from .core import (
    VisConfig,
    EnvConfig,
    # Removed RewardConfig
    # Removed PPOConfig
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    DemoConfig,
    # Removed ObsNormConfig
    TransformerConfig,
)

# Import general configuration settings and functions
from .general import (
    DEVICE,
    RANDOM_SEED,
    # Removed TOTAL_TRAINING_STEPS
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    set_device,
    get_run_id,
    set_run_id,
    get_run_checkpoint_dir,
    get_run_log_dir,
    get_console_log_dir,
    get_model_save_path,
)

# Import utility functions
from .utils import get_config_dict

# Import validation function
from .validation import print_config_info_and_validate

# Import constants
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


# Define __all__ to control what 'from config import *' imports
__all__ = [
    # Core Configs
    "VisConfig",
    "EnvConfig",
    # Removed RewardConfig
    # Removed PPOConfig
    "RNNConfig",
    "TrainConfig",
    "ModelConfig",
    "StatsConfig",
    "TensorBoardConfig",
    "DemoConfig",
    # Removed ObsNormConfig
    "TransformerConfig",
    # General Configs
    "DEVICE",
    "RANDOM_SEED",
    # Removed TOTAL_TRAINING_STEPS
    "BASE_CHECKPOINT_DIR",
    "BASE_LOG_DIR",
    "set_device",
    "get_run_id",
    "set_run_id",
    "get_run_checkpoint_dir",
    "get_run_log_dir",
    "get_console_log_dir",
    "get_model_save_path",
    # Utils
    "get_config_dict",
    "print_config_info_and_validate",
    # Constants
    "WHITE",
    "BLACK",
    "LIGHTG",
    "GRAY",
    "RED",
    "DARK_RED",
    "BLUE",
    "YELLOW",
    "GOOGLE_COLORS",
    "LINE_CLEAR_FLASH_COLOR",
    "LINE_CLEAR_HIGHLIGHT_COLOR",
    "GAME_OVER_FLASH_COLOR",
]

```

File: `config/utils.py`
Explanation: Removed flattening of PPOConfig, RewardConfig, ObsNormConfig.
```python
# File: config/utils.py
import torch
from typing import Dict, Any
from .core import (
    VisConfig,
    EnvConfig,
    # Removed RewardConfig
    # Removed PPOConfig
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    DemoConfig,
    # Removed ObsNormConfig
    TransformerConfig,
)

# Import DEVICE and RANDOM_SEED directly, use get_run_id() for the run ID
from .general import DEVICE, RANDOM_SEED, get_run_id


def get_config_dict() -> Dict[str, Any]:
    """Returns a flat dictionary of all relevant config values for logging."""
    all_configs = {}

    def flatten_class(cls, prefix=""):
        d = {}
        # Use cls() to instantiate if needed for properties, but be careful
        instance = None
        try:
            instance = cls()
        except Exception:
            instance = None  # Cannot instantiate, rely on class vars

        for k, v in vars(cls).items():
            if (
                not k.startswith("__")
                and not callable(v)
                and not isinstance(v, type)
                and not hasattr(v, "__module__")  # Exclude modules
            ):
                # Check if it's a property descriptor on the class
                is_property = isinstance(getattr(cls, k, None), property)

                if is_property and instance:
                    try:
                        v = getattr(instance, k)  # Get property value from instance
                    except Exception:
                        continue  # Skip if property access fails
                elif is_property and not instance:
                    continue  # Skip properties if instance couldn't be created

                d[f"{prefix}{k}"] = v
        return d

    # Flatten core config classes
    all_configs.update(flatten_class(VisConfig, "Vis."))
    all_configs.update(flatten_class(EnvConfig, "Env."))
    # Removed RewardConfig flattening
    # Removed PPOConfig flattening
    all_configs.update(flatten_class(RNNConfig, "RNN."))
    all_configs.update(flatten_class(TrainConfig, "Train."))
    all_configs.update(flatten_class(ModelConfig.Network, "Model.Net."))
    all_configs.update(flatten_class(StatsConfig, "Stats."))
    all_configs.update(flatten_class(TensorBoardConfig, "TB."))
    all_configs.update(flatten_class(DemoConfig, "Demo."))
    # Removed ObsNormConfig flattening
    all_configs.update(flatten_class(TransformerConfig, "Transformer."))

    # Add general config values
    all_configs["General.DEVICE"] = str(DEVICE) if DEVICE else "None"
    all_configs["General.RANDOM_SEED"] = RANDOM_SEED
    all_configs["General.RUN_ID"] = get_run_id()  # Use the getter function

    # Filter out None paths
    all_configs = {
        k: v for k, v in all_configs.items() if not (k.endswith("_PATH") and v is None)
    }

    # Convert non-basic types to strings for logging
    for key, value in all_configs.items():
        if isinstance(value, type) and issubclass(value, torch.nn.Module):
            all_configs[key] = value.__name__
        elif isinstance(value, (list, tuple)):
            all_configs[key] = str(value)
        elif not isinstance(value, (int, float, str, bool)):
            # Check for None explicitly before converting to string
            if value is None:
                all_configs[key] = "None"
            else:
                all_configs[key] = str(value)

    return all_configs

```

File: `config/validation.py`
Explanation: Removed printing/validation of PPOConfig, ObsNormConfig, TOTAL_TRAINING_STEPS.
```python
# File: config/validation.py
from .core import (
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    VisConfig,
    TransformerConfig,
)
from .general import (
    DEVICE,
    # Removed TOTAL_TRAINING_STEPS
    get_run_id,
    get_run_log_dir,
    get_run_checkpoint_dir,
    get_model_save_path,
)


def print_config_info_and_validate():
    env_config_instance = EnvConfig()
    rnn_config_instance = RNNConfig()
    transformer_config_instance = TransformerConfig()
    vis_config_instance = VisConfig()
    train_config_instance = TrainConfig()

    run_id = get_run_id()
    run_log_dir = get_run_log_dir()
    run_checkpoint_dir = get_run_checkpoint_dir()

    print("-" * 70)
    print(f"RUN ID: {run_id}")
    print(f"Log Directory: {run_log_dir}")
    print(f"Checkpoint Directory: {run_checkpoint_dir}")
    print(f"Device: {DEVICE}")
    print(
        f"TB Logging: Histograms={'ON' if TensorBoardConfig.LOG_HISTOGRAMS else 'OFF'}, "
        f"Images={'ON' if TensorBoardConfig.LOG_IMAGES else 'OFF'}"
    )

    if train_config_instance.LOAD_CHECKPOINT_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD CHECKPOINT specified: {train_config_instance.LOAD_CHECKPOINT_PATH} ***\n"
            "*** CheckpointManager will attempt to load this path (e.g., NN weights). ***\n" + "*" * 70
        )
    else:
        print(
            "--- No explicit checkpoint path. CheckpointManager will attempt auto-resume if applicable. ---"
        )

    print("--- Training Algorithm: AlphaZero (MCTS + NN) ---") # Updated description

    # Removed PPO specific prints

    print(
        f"--- Using RNN: {rnn_config_instance.USE_RNN}"
        + (
            f" (LSTM Hidden: {rnn_config_instance.LSTM_HIDDEN_SIZE}, Layers: {rnn_config_instance.LSTM_NUM_LAYERS})"
            if rnn_config_instance.USE_RNN
            else ""
        )
        + " ---"
    )
    print(
        f"--- Using Transformer: {transformer_config_instance.USE_TRANSFORMER}"
        + (
            f" (d_model={transformer_config_instance.TRANSFORMER_D_MODEL}, nhead={transformer_config_instance.TRANSFORMER_NHEAD}, layers={transformer_config_instance.TRANSFORMER_NUM_LAYERS})"
            if transformer_config_instance.USE_TRANSFORMER
            else ""
        )
        + " ---"
    )
    # Removed ObsNorm print

    print(
        f"Config: Env=(R={env_config_instance.ROWS}, C={env_config_instance.COLS}), "
        f"GridState={env_config_instance.GRID_STATE_SHAPE}, "
        f"ShapeState={env_config_instance.SHAPE_STATE_DIM}, "
        f"ActionDim={env_config_instance.ACTION_DIM}"
    )
    cnn_str = str(ModelConfig.Network.CONV_CHANNELS).replace(" ", "")
    mlp_str = str(ModelConfig.Network.COMBINED_FC_DIMS).replace(" ", "")
    shape_mlp_cfg_str = str(ModelConfig.Network.SHAPE_FEATURE_MLP_DIMS).replace(" ", "")
    print(f"Network Base: CNN={cnn_str}, ShapeMLP={shape_mlp_cfg_str}, Fusion={mlp_str}") # Adapted description

    print(
        f"Training: NUM_ENVS={env_config_instance.NUM_ENVS}" # Removed total steps
    )
    print(
        f"Stats: AVG_WINDOWS={StatsConfig.STATS_AVG_WINDOW}, Console Log Freq={StatsConfig.CONSOLE_LOG_FREQ} (episodes/updates)" # Adapted freq description
    )

    if env_config_instance.NUM_ENVS >= 1024: # Keep warning, though NUM_ENVS is now 1
        device_type = DEVICE.type if DEVICE else "UNKNOWN"
        print(
            "*" * 70
            + f"\n*** Warning: NUM_ENVS={env_config_instance.NUM_ENVS}. Monitor system resources. ***"
            + (
                "\n*** Using MPS device. Performance varies. Force CPU via env var if needed. ***"
                if device_type == "mps"
                else ""
            )
            + "\n"
            + "*" * 70
        )
    print(
        f"--- Rendering {VisConfig.NUM_ENVS_TO_RENDER if VisConfig.NUM_ENVS_TO_RENDER > 0 else 'ALL'} of {env_config_instance.NUM_ENVS} environments ---"
    )
    print("-" * 70)

```

File: `stats/aggregator_storage.py`
Explanation: Removed deques for policy_loss, value_loss, entropy, grad_norm, sps_values, update_steps_per_second, minibatch_sps. Removed related current scalar values. Kept LR for now, might be useful.
```python
# File: stats/aggregator_storage.py
from collections import deque
from typing import Deque, Dict, Any, List
import time
import numpy as np


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator."""

    def __init__(self, plot_window: int):
        self.plot_window = plot_window

        # --- Deques for Plotting ---
        # Removed: self.policy_losses
        # Removed: self.value_losses
        # Removed: self.entropies
        # Removed: self.grad_norms
        self.avg_max_qs: Deque[float] = deque(maxlen=plot_window) # Keep if Q-values are estimated by NN
        self.episode_scores: Deque[float] = deque(maxlen=plot_window) # Keep RL score if used, or repurpose
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_triangles_cleared: Deque[int] = deque(maxlen=plot_window)
        # Removed: self.sps_values
        # Removed: self.update_steps_per_second_values
        # Removed: self.minibatch_update_sps_values
        self.buffer_sizes: Deque[int] = deque(maxlen=plot_window) # Might be useful for MCTS buffer/NN training data
        self.beta_values: Deque[float] = deque(maxlen=plot_window) # Keep if used (e.g., PER)
        self.best_rl_score_history: Deque[float] = deque(maxlen=plot_window) # Keep RL score if used
        self.best_game_score_history: Deque[int] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(maxlen=plot_window) # Keep for NN training LR
        self.epsilon_values: Deque[float] = deque(maxlen=plot_window) # Keep if epsilon-greedy is used in MCTS/NN
        self.cpu_usage: Deque[float] = deque(maxlen=plot_window)
        self.memory_usage: Deque[float] = deque(maxlen=plot_window)
        self.gpu_memory_usage_percent: Deque[float] = deque(maxlen=plot_window)

        # --- Scalar State Variables ---
        self.total_episodes = 0
        self.total_triangles_cleared = 0
        self.current_epsilon: float = 0.0
        self.current_beta: float = 0.0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0 # Might represent training steps or games played now
        # Removed: self.current_sps
        # Removed: self.current_update_steps_per_second
        # Removed: self.current_minibatch_update_sps
        self.current_lr: float = 0.0 # Keep for NN
        self.start_time: float = time.time()
        self.training_target_step: int = 0 # Target might be games played or NN steps
        self.current_cpu_usage: float = 0.0
        self.current_memory_usage: float = 0.0
        self.current_gpu_memory_usage_percent: float = 0.0

        # --- Best Value Tracking ---
        self.best_score: float = -float("inf") # Keep RL score if used
        self.previous_best_score: float = -float("inf")
        self.best_score_step: int = 0
        self.best_game_score: float = -float("inf")
        self.previous_best_game_score: float = -float("inf")
        self.best_game_score_step: int = 0
        self.best_value_loss: float = float("inf") # Keep for NN value head loss
        self.previous_best_value_loss: float = float("inf")
        self.best_value_loss_step: int = 0
        # Add best policy loss?
        self.best_policy_loss: float = float("inf")
        self.previous_best_policy_loss: float = float("inf")
        self.best_policy_loss_step: int = 0


    def get_deque(self, name: str) -> Deque:
        """Safely gets a deque attribute."""
        return getattr(self, name, deque(maxlen=self.plot_window))

    def get_all_plot_deques(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        deque_names = [
            # Removed PPO losses/entropy/grad_norm/sps
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            # Add NN losses if tracked
            "policy_losses", # Added back for NN policy head
            "value_losses", # Kept for NN value head
        ]
        # Filter out names that might not exist if loaded from old state
        return {name: self.get_deque(name).copy() for name in deque_names if hasattr(self, name)}


    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the storage for saving."""
        state = {}
        # Deques
        deque_names = [
            # Removed PPO losses/entropy/grad_norm/sps
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            # Add NN losses if tracked
            "policy_losses", # Added back for NN policy head
            "value_losses", # Kept for NN value head
        ]
        for name in deque_names:
             if hasattr(self, name): # Check if deque exists before saving
                state[name] = list(self.get_deque(name))

        # Scalar State Variables
        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_epsilon",
            "current_beta",
            "current_buffer_size",
            "current_global_step",
            # Removed SPS scalars
            "current_lr",
            "start_time",
            "training_target_step",
            "current_cpu_usage",
            "current_memory_usage",
            "current_gpu_memory_usage_percent",
        ]
        for key in scalar_keys:
            state[key] = getattr(self, key, 0)

        # Best Value Tracking
        best_value_keys = [
            "best_score",
            "previous_best_score",
            "best_score_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
            "best_policy_loss", # Added policy loss tracking
            "previous_best_policy_loss",
            "best_policy_loss_step",
        ]
        for key in best_value_keys:
            state[key] = getattr(self, key, 0)

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        self.plot_window = plot_window

        # Deques
        deque_names = [
            # Removed PPO losses/entropy/grad_norm/sps
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            # Add NN losses if tracked
            "policy_losses", # Added back for NN policy head
            "value_losses", # Kept for NN value head
        ]
        for key in deque_names:
            data_to_load = state_dict.get(key)
            if data_to_load is not None:
                try:
                    if isinstance(data_to_load, (list, tuple)):
                        setattr(self, key, deque(data_to_load, maxlen=self.plot_window))
                    else:
                        print(f"  -> Warning: Invalid type for deque '{key}'. Initializing empty.")
                        setattr(self, key, deque(maxlen=self.plot_window))
                except Exception as e:
                    print(f"  -> Error loading deque '{key}': {e}. Initializing empty.")
                    setattr(self, key, deque(maxlen=self.plot_window))
            else:
                # Ensure deque exists even if not in state_dict
                setattr(self, key, deque(maxlen=self.plot_window))

        # Scalar State Variables
        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_epsilon",
            "current_beta",
            "current_buffer_size",
            "current_global_step",
            # Removed SPS scalars
            "current_lr",
            "start_time",
            "training_target_step",
            "current_cpu_usage",
            "current_memory_usage",
            "current_gpu_memory_usage_percent",
        ]
        default_values = {
            "start_time": time.time(),
            "training_target_step": 0,
            "current_global_step": 0,
        }
        for key in scalar_keys:
            value_to_load = state_dict.get(key, default_values.get(key, 0))
            setattr(self, key, value_to_load)

        # Best Value Tracking
        best_value_keys = [
            "best_score",
            "previous_best_score",
            "best_score_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
            "best_policy_loss", # Added policy loss tracking
            "previous_best_policy_loss",
            "best_policy_loss_step",
        ]
        default_best = {
            "best_score": -float("inf"),
            "previous_best_score": -float("inf"),
            "best_game_score": -float("inf"),
            "previous_best_game_score": -float("inf"),
            "best_value_loss": float("inf"),
            "previous_best_value_loss": float("inf"),
            "best_policy_loss": float("inf"), # Added policy loss tracking
            "previous_best_policy_loss": float("inf"),
        }
        for key in best_value_keys:
            setattr(self, key, state_dict.get(key, default_best.get(key, 0)))

        # Ensure current_global_step exists after loading
        if not hasattr(self, "current_global_step"):
            self.current_global_step = 0
        # Ensure training_target_step exists after loading
        if not hasattr(self, "training_target_step"):
            self.training_target_step = 0
        # Ensure policy loss tracking exists
        if not hasattr(self, "best_policy_loss"):
            self.best_policy_loss = float("inf")
        if not hasattr(self, "previous_best_policy_loss"):
            self.previous_best_policy_loss = float("inf")
        if not hasattr(self, "best_policy_loss_step"):
            self.best_policy_loss_step = 0

```

File: `stats/aggregator_logic.py`
Explanation: Removed logic for updating PPO losses, entropy, grad_norm, SPS. Added tracking for policy loss.
```python
# File: stats/aggregator_logic.py
from collections import deque
from typing import Deque, Dict, Any, Optional, List
import numpy as np
import time

from .aggregator_storage import AggregatorStorage


class AggregatorLogic:
    """Handles the calculation logic for StatsAggregator."""

    def __init__(self, storage: AggregatorStorage):
        self.storage = storage

    def update_episode_stats(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        current_step: int,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Updates storage with episode data and checks for bests."""
        update_info = {"new_best_rl": False, "new_best_game": False}

        self.storage.episode_scores.append(episode_score)
        self.storage.episode_lengths.append(episode_length)
        if game_score is not None:
            self.storage.game_scores.append(game_score)
        if triangles_cleared is not None:
            self.storage.episode_triangles_cleared.append(triangles_cleared)
            self.storage.total_triangles_cleared += triangles_cleared
        self.storage.total_episodes = episode_num

        if episode_score > self.storage.best_score:
            self.storage.previous_best_score = self.storage.best_score
            self.storage.best_score = episode_score
            self.storage.best_score_step = current_step
            update_info["new_best_rl"] = True

        if game_score is not None and game_score > self.storage.best_game_score:
            self.storage.previous_best_game_score = self.storage.best_game_score
            self.storage.best_game_score = float(game_score)
            self.storage.best_game_score_step = current_step
            update_info["new_best_game"] = True

        self.storage.best_rl_score_history.append(self.storage.best_score)
        current_best_game = (
            int(self.storage.best_game_score)
            if self.storage.best_game_score > -float("inf")
            else 0
        )
        self.storage.best_game_score_history.append(current_best_game)

        return update_info

    def update_step_stats(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Updates storage with step data and checks for best loss."""
        g_step = step_data.get("global_step", self.storage.current_global_step)
        if g_step > self.storage.current_global_step:
            self.storage.current_global_step = g_step

        if "training_target_step" in step_data:
            self.storage.training_target_step = step_data["training_target_step"]

        update_info = {"new_best_loss": False, "new_best_policy_loss": False} # Added policy loss flag

        # Append to deques
        # --- NN Policy Loss ---
        if "policy_loss" in step_data and step_data["policy_loss"] is not None:
            current_policy_loss = step_data["policy_loss"]
            if np.isfinite(current_policy_loss):
                self.storage.policy_losses.append(current_policy_loss)
                if current_policy_loss < self.storage.best_policy_loss and g_step > 0:
                    self.storage.previous_best_policy_loss = self.storage.best_policy_loss
                    self.storage.best_policy_loss = current_policy_loss
                    self.storage.best_policy_loss_step = g_step
                    update_info["new_best_policy_loss"] = True
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Policy Loss: {current_policy_loss}"
                )
        # --- End NN Policy Loss ---

        # --- NN Value Loss ---
        if "value_loss" in step_data and step_data["value_loss"] is not None:
            current_value_loss = step_data["value_loss"]
            if np.isfinite(current_value_loss):
                self.storage.value_losses.append(current_value_loss)
                if current_value_loss < self.storage.best_value_loss and g_step > 0:
                    self.storage.previous_best_value_loss = self.storage.best_value_loss
                    self.storage.best_value_loss = current_value_loss
                    self.storage.best_value_loss_step = g_step
                    update_info["new_best_loss"] = True # Keep original flag name for value loss
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Value Loss: {current_value_loss}"
                )
        # --- End NN Value Loss ---

        # Removed Entropy

        # Append other optional metrics
        optional_metrics = [
            # Removed grad_norm, update_steps_per_second, minibatch_update_sps
            ("avg_max_q", "avg_max_qs"),
            ("beta", "beta_values"),
            ("buffer_size", "buffer_sizes"),
            ("lr", "lr_values"),
            ("epsilon", "epsilon_values"),
            ("cpu_usage", "cpu_usage"),
            ("memory_usage", "memory_usage"),
            ("gpu_memory_usage_percent", "gpu_memory_usage_percent"),
        ]
        for data_key, deque_name in optional_metrics:
            if data_key in step_data and step_data[data_key] is not None:
                val = step_data[data_key]
                if np.isfinite(val):
                    # Ensure deque exists before appending
                    if hasattr(self.storage, deque_name):
                         getattr(self.storage, deque_name).append(val)
                    else:
                         print(f"[Aggregator Warning] Deque '{deque_name}' not found in storage.")
                else:
                    print(f"[Aggregator Warning] Received non-finite {data_key}: {val}")

        # Update scalar values
        scalar_updates = {
            # Removed SPS scalars
            "beta": "current_beta",
            "buffer_size": "current_buffer_size",
            "lr": "current_lr",
            "epsilon": "current_epsilon",
            "cpu_usage": "current_cpu_usage",
            "memory_usage": "current_memory_usage",
            "gpu_memory_usage_percent": "current_gpu_memory_usage_percent",
        }
        for data_key, storage_key in scalar_updates.items():
            if data_key in step_data and step_data[data_key] is not None:
                val = step_data[data_key]
                if np.isfinite(val):
                    setattr(self.storage, storage_key, val)
                else:
                    print(f"[Aggregator Warning] Received non-finite {data_key}: {val}")

        return update_info

    def calculate_summary(
        self, current_global_step: int, summary_avg_window: int
    ) -> Dict[str, Any]:
        """Calculates the summary dictionary based on stored data."""

        def safe_mean(q_name: str, default=0.0) -> float:
            # Check if deque exists before accessing
            if not hasattr(self.storage, q_name):
                return default
            deque_instance = self.storage.get_deque(q_name)
            window_data = list(deque_instance)[-summary_avg_window:]
            finite_data = [x for x in window_data if np.isfinite(x)]
            return float(np.mean(finite_data)) if finite_data else default

        summary = {
            "avg_score_window": safe_mean("episode_scores"),
            "avg_length_window": safe_mean("episode_lengths"),
            "policy_loss": safe_mean("policy_losses"), # Added policy loss
            "value_loss": safe_mean("value_losses"),
            # Removed entropy, avg_update_sps, avg_minibatch_sps
            "avg_max_q_window": safe_mean("avg_max_qs"),
            "avg_game_score_window": safe_mean("game_scores"),
            "avg_triangles_cleared_window": safe_mean("episode_triangles_cleared"),
            "avg_lr_window": safe_mean("lr_values", default=self.storage.current_lr),
            "avg_cpu_window": safe_mean("cpu_usage"),
            "avg_memory_window": safe_mean("memory_usage"),
            "avg_gpu_memory_window": safe_mean("gpu_memory_usage_percent"),
            "total_episodes": self.storage.total_episodes,
            "beta": self.storage.current_beta,
            "buffer_size": self.storage.current_buffer_size,
            # Removed SPS scalars
            "global_step": current_global_step,
            "current_lr": self.storage.current_lr,
            "best_score": self.storage.best_score,
            "previous_best_score": self.storage.previous_best_score,
            "best_score_step": self.storage.best_score_step,
            "best_game_score": self.storage.best_game_score,
            "previous_best_game_score": self.storage.previous_best_game_score,
            "best_game_score_step": self.storage.best_game_score_step,
            "best_loss": self.storage.best_value_loss, # Keep as value loss best
            "previous_best_loss": self.storage.previous_best_value_loss,
            "best_loss_step": self.storage.best_value_loss_step,
            "best_policy_loss": self.storage.best_policy_loss, # Added policy loss best
            "previous_best_policy_loss": self.storage.previous_best_policy_loss,
            "best_policy_loss_step": self.storage.best_policy_loss_step,
            "num_ep_scores": len(self.storage.episode_scores),
            "num_losses": len(self.storage.value_losses), # Maybe rename to num_value_losses?
            "summary_avg_window_size": summary_avg_window,
            "start_time": self.storage.start_time,
            "training_target_step": self.storage.training_target_step,
            "current_cpu_usage": self.storage.current_cpu_usage,
            "current_memory_usage": self.storage.current_memory_usage,
            "current_gpu_memory_usage_percent": self.storage.current_gpu_memory_usage_percent,
        }
        return summary

```

File: `stats/aggregator.py`
Explanation: Adapted to removed metrics. Added tracking for policy loss best.
```python
# File: stats/aggregator.py
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, List
import numpy as np
import threading

from config import StatsConfig
from .aggregator_storage import AggregatorStorage
from .aggregator_logic import AggregatorLogic


class StatsAggregator:
    """
    Handles aggregation and storage of training statistics using deques.
    Calculates rolling averages and tracks best values. Does not perform logging.
    Includes locks for thread safety. Delegates storage and logic to helper classes.
    """

    def __init__(
        self,
        avg_windows: List[int] = StatsConfig.STATS_AVG_WINDOW,
        plot_window: int = StatsConfig.PLOT_DATA_WINDOW,
    ):
        if not avg_windows or not all(
            isinstance(w, int) and w > 0 for w in avg_windows
        ):
            print("Warning: Invalid avg_windows list. Using default [100].")
            self.avg_windows = [100]
        else:
            self.avg_windows = sorted(list(set(avg_windows)))

        if plot_window <= 0:
            plot_window = 10000
        self.plot_window = plot_window
        self.summary_avg_window = self.avg_windows[0]

        self._lock = threading.Lock()
        self.storage = AggregatorStorage(plot_window)
        self.logic = AggregatorLogic(self.storage)

        print(
            f"[StatsAggregator] Initialized. Avg Windows: {self.avg_windows}, Plot Window: {self.plot_window}"
        )

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            current_step = (
                global_step
                if global_step is not None
                else self.storage.current_global_step
            )
            update_info = self.logic.update_episode_stats(
                episode_score,
                episode_length,
                episode_num,
                current_step,
                game_score,
                triangles_cleared,
            )
            return update_info

    def record_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Records step data, now likely related to NN training steps."""
        with self._lock:
            update_info = self.logic.update_step_stats(step_data)
            return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            if current_global_step is None:
                current_global_step = self.storage.current_global_step
            summary = self.logic.calculate_summary(
                current_global_step, self.summary_avg_window
            )
            return summary

    def get_plot_data(self) -> Dict[str, Deque]:
        with self._lock:
            return self.storage.get_all_plot_deques()

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            state = self.storage.state_dict()
            state["plot_window"] = self.plot_window
            state["avg_windows"] = self.avg_windows
            return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        with self._lock:
            print("[StatsAggregator] Loading state...")
            self.plot_window = state_dict.get("plot_window", self.plot_window)
            self.avg_windows = state_dict.get("avg_windows", self.avg_windows)
            self.summary_avg_window = self.avg_windows[0] if self.avg_windows else 100

            self.storage.load_state_dict(state_dict, self.plot_window)

            print("[StatsAggregator] State loaded.")
            print(f"  -> Loaded total_episodes: {self.storage.total_episodes}")
            print(f"  -> Loaded best_score: {self.storage.best_score}")
            print(
                f"  -> Loaded start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.storage.start_time))}"
            )
            print(
                f"  -> Loaded training_target_step: {self.storage.training_target_step}"
            )
            print(
                f"  -> Loaded current_global_step: {self.storage.current_global_step}"
            )

```

File: `stats/tb_scalar_logger.py`
Explanation: Removed logging of PPO losses, entropy, grad_norm, SPS. Added policy loss.
```python
# File: stats/tb_scalar_logger.py
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from typing import Dict, Any, Optional

from .aggregator import StatsAggregator  # For type hinting


class TBScalarLogger:
    """Handles logging scalar values to TensorBoard."""

    def __init__(self, writer: Optional[SummaryWriter], lock: threading.Lock):
        self.writer = writer
        self._lock = lock

    def log_episode_scalars(
        self,
        g_step: int,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        game_score: Optional[int],
        triangles_cleared: Optional[int],
        update_info: Dict[str, Any],
        aggregator: StatsAggregator,  # Pass aggregator for best values
    ):
        """Logs scalars related to a completed episode."""
        if not self.writer:
            return
        with self._lock:
            try:
                self.writer.add_scalar("Episode/Score", episode_score, g_step)
                self.writer.add_scalar("Episode/Length", episode_length, g_step)
                if game_score is not None:
                    self.writer.add_scalar("Episode/Game Score", game_score, g_step)
                if triangles_cleared is not None:
                    self.writer.add_scalar(
                        "Episode/Triangles Cleared", triangles_cleared, g_step
                    )
                self.writer.add_scalar("Progress/Total Episodes", episode_num, g_step)

                # --- Corrected Access ---
                if update_info.get("new_best_rl"):
                    self.writer.add_scalar(
                        "Best Performance/RL Score",
                        aggregator.storage.best_score,
                        g_step,
                    )
                if update_info.get("new_best_game"):
                    self.writer.add_scalar(
                        "Best Performance/Game Score",
                        aggregator.storage.best_game_score,
                        g_step,
                    )
                # --- End Correction ---
            except Exception as e:
                print(f"Error writing episode scalars to TensorBoard: {e}")

    def log_step_scalars(
        self,
        g_step: int,
        step_data: Dict[str, Any],
        update_info: Dict[str, Any],
        aggregator: StatsAggregator,  # Pass aggregator for best values
    ):
        """Logs scalars related to a training/environment step."""
        if not self.writer:
            return
        with self._lock:
            try:
                scalar_map = {
                    "policy_loss": "Loss/Policy Loss", # Keep policy loss for NN
                    "value_loss": "Loss/Value Loss", # Keep value loss for NN
                    # Removed entropy, grad_norm, sps_collection, update_steps_per_second, minibatch_update_sps
                    "avg_max_q": "Debug/Avg Max Q", # Keep if NN estimates Q
                    "beta": "Debug/Beta", # Keep if PER used
                    "buffer_size": "Debug/Buffer Size", # Keep for MCTS/NN buffer
                    "lr": "Train/Learning Rate", # Keep for NN
                    "epsilon": "Train/Epsilon", # Keep if used
                    "update_time": "Performance/Update Time", # Keep for NN update time
                    "step_time": "Performance/Total Step Time", # Keep if relevant
                    "cpu_usage": "Resource/CPU Usage (%)",
                    "memory_usage": "Resource/Memory Usage (%)",
                    "gpu_memory_usage_percent": "Resource/GPU Memory Usage (%)",
                }
                for key, tag in scalar_map.items():
                    if key in step_data and step_data[key] is not None:
                        self.writer.add_scalar(tag, step_data[key], g_step)

                # --- Corrected Access ---
                if update_info.get("new_best_loss"): # Value loss
                    self.writer.add_scalar(
                        "Best Performance/Value Loss",
                        aggregator.storage.best_value_loss,
                        g_step,
                    )
                if update_info.get("new_best_policy_loss"): # Policy loss
                    self.writer.add_scalar(
                        "Best Performance/Policy Loss",
                        aggregator.storage.best_policy_loss,
                        g_step,
                    )
                # --- End Correction ---
            except Exception as e:
                print(f"Error writing step scalars to TensorBoard: {e}")

```

File: `stats/tensorboard_logger.py`
Explanation: Removed PPO-specific graph/hparam logic triggers. Adapted helper calls.
```python
# File: stats/tensorboard_logger.py
import time
import traceback
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import threading

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from .simple_stats_recorder import SimpleStatsRecorder
from config import TensorBoardConfig, EnvConfig, RNNConfig # Keep RNNConfig for potential future use
# Removed ActorCriticNetwork import

# Import helper modules
from .tb_log_utils import format_image_for_tb
from .tb_scalar_logger import TBScalarLogger
from .tb_histogram_logger import TBHistogramLogger
from .tb_image_logger import TBImageLogger
from .tb_hparam_logger import TBHparamLogger


class TensorBoardStatsRecorder(StatsRecorderBase):
    """
    Logs aggregated statistics, histograms, images, and hyperparameters to TensorBoard. Thread-safe.
    Uses a SimpleStatsRecorder for console logging and a StatsAggregator for data handling.
    Delegates specific logging tasks to helper classes.
    """

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_recorder: SimpleStatsRecorder,
        log_dir: str,
        hparam_dict: Optional[Dict[str, Any]] = None,
        model_for_graph: Optional[torch.nn.Module] = None, # Changed type hint
        dummy_input_for_graph: Optional[Any] = None, # Changed type hint
        histogram_log_interval: int = TensorBoardConfig.HISTOGRAM_LOG_FREQ,
        image_log_interval: int = TensorBoardConfig.IMAGE_LOG_FREQ,
        env_config: Optional[EnvConfig] = None,
        rnn_config: Optional[RNNConfig] = None,
    ):
        self.aggregator = aggregator
        self.console_recorder = console_recorder
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None
        self._lock = threading.Lock()

        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"[TensorBoardStatsRecorder] Initialized. Logging to: {self.log_dir}")

            # Initialize helper loggers
            self.scalar_logger = TBScalarLogger(self.writer, self._lock)
            self.histogram_logger = TBHistogramLogger(
                self.writer, self._lock, histogram_log_interval
            )
            self.image_logger = TBImageLogger(
                self.writer, self._lock, image_log_interval
            )
            self.hparam_logger = TBHparamLogger(self.writer, self._lock, hparam_dict)

            if model_for_graph and dummy_input_for_graph:
                self.record_graph(model_for_graph, dummy_input_for_graph)
            else:
                print("[TensorBoardStatsRecorder] Model graph logging skipped.")

            self.hparam_logger.log_initial_hparams()

        except Exception as e:
            print(f"FATAL: Error initializing TensorBoard SummaryWriter: {e}")
            traceback.print_exc()
            self.writer = None
            self.scalar_logger = None
            self.histogram_logger = None
            self.image_logger = None
            self.hparam_logger = None

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
    ):
        """Records episode stats to TensorBoard and delegates to console recorder."""
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
        )
        g_step = (
            global_step
            if global_step is not None
            else getattr(
                self.aggregator.storage, "current_global_step", 0
            )
        )

        if self.scalar_logger:
            self.scalar_logger.log_episode_scalars(
                g_step,
                episode_score,
                episode_length,
                episode_num,
                game_score,
                triangles_cleared,
                update_info,
                self.aggregator,
            )

        self.console_recorder.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
        )

    def record_step(self, step_data: Dict[str, Any]):
        """Records step stats (e.g., NN training step) to TensorBoard and console."""
        update_info = self.aggregator.record_step(step_data)
        g_step = step_data.get(
            "global_step",
            getattr(
                self.aggregator.storage, "current_global_step", 0
            ),
        )

        if self.scalar_logger:
            self.scalar_logger.log_step_scalars(
                g_step, step_data, update_info, self.aggregator
            )

        # Increment histogram/image counters if an update occurred
        # Check for a key indicating an NN update, e.g., 'policy_loss' or 'value_loss'
        if "policy_loss" in step_data or "value_loss" in step_data:
            if self.histogram_logger:
                self.histogram_logger.increment_rollout_counter()
                if self.histogram_logger.should_log(g_step):
                     self.histogram_logger.reset_rollout_counter() # Reset only if logged
            if self.image_logger:
                self.image_logger.increment_rollout_counter()
                if self.image_logger.should_log(g_step):
                     self.image_logger.reset_rollout_counter() # Reset only if logged

        self.console_recorder.record_step(step_data)

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Records a histogram to TensorBoard using the helper."""
        if self.histogram_logger:
            self.histogram_logger.log_histogram(tag, values, global_step)

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Records an image to TensorBoard using the helper."""
        if self.image_logger:
            self.image_logger.log_image(tag, image, global_step)

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Records final hyperparameters and metrics using the helper."""
        if self.hparam_logger:
            self.hparam_logger.log_final_hparams(hparam_dict, metric_dict)

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[Any] = None
    ):
        """Records the model graph to TensorBoard."""
        if not self.writer:
            return
        if input_to_model is None:
            print("Warning: Cannot record graph without dummy input.")
            return
        with self._lock:
            try:
                # Ensure model is on CPU for graph tracing if needed
                original_device = next(iter(model.parameters())).device
                model.cpu()
                # Convert input to CPU if it's a tensor or tuple of tensors
                if isinstance(input_to_model, torch.Tensor):
                    dummy_input_cpu = input_to_model.cpu()
                elif isinstance(input_to_model, tuple):
                    dummy_input_cpu = tuple(
                        i.cpu() if isinstance(i, torch.Tensor) else i
                        for i in input_to_model
                    )
                else:
                    dummy_input_cpu = input_to_model # Assume compatible

                self.writer.add_graph(model, dummy_input_cpu, verbose=False)
                print("[TensorBoardStatsRecorder] Model graph logged.")
                model.to(original_device) # Move model back
            except Exception as e:
                print(f"Error logging model graph: {e}.")
                traceback.print_exc()
                try:
                    model.to(original_device) # Attempt to move back even on error
                except Exception:
                    pass

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        self.console_recorder.log_summary(global_step)

    def close(self, is_cleanup: bool = False):
        """Closes the TensorBoard writer and logs final hparams unless cleaning up."""
        print(f"[TensorBoardStatsRecorder] Close called (is_cleanup={is_cleanup})...")
        if not self.writer:
            print("[TensorBoardStatsRecorder] Writer was not initialized or already closed.")
            self.console_recorder.close(is_cleanup=is_cleanup)
            return

        with self._lock:
            print("[TensorBoardStatsRecorder] Acquired lock for closing.")
            try:
                if not is_cleanup and self.hparam_logger:
                    print("[TensorBoardStatsRecorder] Logging final hparams...")
                    final_step = getattr(self.aggregator.storage, "current_global_step", 0)
                    final_summary = self.get_summary(final_step)
                    self.hparam_logger.log_final_hparams_from_summary(final_summary)
                    print("[TensorBoardStatsRecorder] Final hparams logged.")
                elif is_cleanup:
                    print("[TensorBoardStatsRecorder] Skipping final hparams logging due to cleanup.")

                print("[TensorBoardStatsRecorder] Flushing writer...")
                self.writer.flush()
                print("[TensorBoardStatsRecorder] Writer flushed.")
                print("[TensorBoardStatsRecorder] Closing writer...")
                self.writer.close()
                self.writer = None
                print("[TensorBoardStatsRecorder] Writer closed successfully.")
            except Exception as e:
                print(f"[TensorBoardStatsRecorder] Error during writer close: {e}")
                traceback.print_exc()
            finally:
                print("[TensorBoardStatsRecorder] Releasing lock after closing attempt.")

        self.console_recorder.close(is_cleanup=is_cleanup)
        print("[TensorBoardStatsRecorder] Close method finished.")

```

File: `stats/simple_stats_recorder.py`
Explanation: Adapted console log format to remove PPO-specific metrics (SPS, Loss).
```python
# File: stats/simple_stats_recorder.py
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List
import numpy as np
import torch
import threading

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from config import StatsConfig


class SimpleStatsRecorder(StatsRecorderBase):
    """
    Logs aggregated statistics to the console periodically. Thread-safe.
    Delegates data storage and aggregation to a StatsAggregator instance.
    Provides no-op implementations for histogram, image, hparam, graph logging.
    """

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_log_interval: int = StatsConfig.CONSOLE_LOG_FREQ,
    ):
        self.aggregator = aggregator
        # Console log interval might now represent episodes or training steps
        self.console_log_interval = (
            max(1, console_log_interval) if console_log_interval > 0 else -1
        )
        self.last_log_time: float = time.time()
        self.start_time: float = time.time()
        self.summary_avg_window = self.aggregator.summary_avg_window
        # Counter might track episodes or training steps now
        self.updates_since_last_log = 0 # Renamed from rollouts_since_last_log

        self._lock = threading.Lock()

        print(
            f"[SimpleStatsRecorder] Initialized. Console Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'} updates/episodes"
        )
        print(
            f"[SimpleStatsRecorder] Console logs will use Avg Window: {self.summary_avg_window}"
        )

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
    ):
        """Records episode stats and prints new bests to console. Thread-safe."""
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
        )
        current_step = (
            global_step
            if global_step is not None
            else self.aggregator.storage.current_global_step
        )
        step_info = f"at Step ~{current_step/1e6:.1f}M" # Step might mean NN steps now

        # Print new bests immediately
        if update_info.get("new_best_rl"):
            prev_str = (
                f"{self.aggregator.storage.previous_best_score:.2f}"
                if self.aggregator.storage.previous_best_score > -float("inf")
                else "N/A"
            )
            print(
                f"\n--- 🏆 New Best RL: {self.aggregator.storage.best_score:.2f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_game"):
            prev_str = (
                f"{self.aggregator.storage.previous_best_game_score:.0f}"
                if self.aggregator.storage.previous_best_game_score > -float("inf")
                else "N/A"
            )
            print(
                f"--- 🎮 New Best Game: {self.aggregator.storage.best_game_score:.0f} {step_info} (Prev: {prev_str}) ---"
            )
        # Check for new best NN losses
        if update_info.get("new_best_loss"): # Value loss
            prev_str = (
                f"{self.aggregator.storage.previous_best_value_loss:.4f}"
                if self.aggregator.storage.previous_best_value_loss < float("inf")
                else "N/A"
            )
            print(
                f"---📉 New Best V.Loss: {self.aggregator.storage.best_value_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_policy_loss"): # Policy loss
            prev_str = (
                f"{self.aggregator.storage.previous_best_policy_loss:.4f}"
                if self.aggregator.storage.previous_best_policy_loss < float("inf")
                else "N/A"
            )
            print(
                f"---📉 New Best P.Loss: {self.aggregator.storage.best_policy_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )

        # Trigger console log based on episode count if interval is set
        log_now = False
        with self._lock:
            self.updates_since_last_log += 1
            if (
                self.console_log_interval > 0
                and self.updates_since_last_log >= self.console_log_interval
            ):
                log_now = True
                self.updates_since_last_log = 0

        if log_now:
            self.log_summary(current_step)


    def record_step(self, step_data: Dict[str, Any]):
        """Records step stats (e.g., NN update) and triggers console logging if interval met. Thread-safe."""
        update_info = self.aggregator.record_step(step_data)
        g_step = step_data.get(
            "global_step", self.aggregator.storage.current_global_step
        )

        # Print new best loss immediately if it occurred during this step's update
        if update_info.get("new_best_loss"): # Value loss
            prev_str = (
                f"{self.aggregator.storage.previous_best_value_loss:.4f}"
                if self.aggregator.storage.previous_best_value_loss < float("inf")
                else "N/A"
            )
            step_info = f"at Step ~{g_step/1e6:.1f}M"
            print(
                f"---📉 New Best V.Loss: {self.aggregator.storage.best_value_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_policy_loss"): # Policy loss
            prev_str = (
                f"{self.aggregator.storage.previous_best_policy_loss:.4f}"
                if self.aggregator.storage.previous_best_policy_loss < float("inf")
                else "N/A"
            )
            step_info = f"at Step ~{g_step/1e6:.1f}M"
            print(
                f"---📉 New Best P.Loss: {self.aggregator.storage.best_policy_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )

        # Increment counter and check logging frequency (thread-safe)
        # Logging frequency might now be based on NN updates instead of rollouts
        log_now = False
        with self._lock:
            # Increment counter if an NN update occurred (check for loss keys)
            if "policy_loss" in step_data or "value_loss" in step_data:
                self.updates_since_last_log += 1
                if (
                    self.console_log_interval > 0
                    and self.updates_since_last_log >= self.console_log_interval
                ):
                    log_now = True
                    self.updates_since_last_log = 0

        if log_now:
            self.log_summary(g_step)

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Gets the summary dictionary from the aggregator (thread-safe)."""
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        """Gets the plot data deques from the aggregator (thread-safe)."""
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        """Logs the current summary statistics to the console."""
        summary = self.get_summary(global_step)
        elapsed_runtime = time.time() - self.aggregator.storage.start_time
        runtime_hrs = elapsed_runtime / 3600

        best_score_val = (
            f"{summary['best_score']:.2f}"
            if summary["best_score"] > -float("inf")
            else "N/A"
        )
        best_game_score_val = (
            f"{summary['best_game_score']:.0f}"
            if summary['best_game_score'] > -float('inf') else "N/A"
        )
        best_v_loss_val = (
            f"{summary['best_loss']:.4f}" # Value loss
            if summary["best_loss"] < float("inf")
            else "N/A"
        )
        best_p_loss_val = (
            f"{summary['best_policy_loss']:.4f}" # Policy loss
            if summary["best_policy_loss"] < float("inf")
            else "N/A"
        )
        avg_window_size = summary.get("summary_avg_window_size", "?")

        # Removed SPS
        log_str = (
            f"[{runtime_hrs:.1f}h|Console] Step: {global_step/1e6:<6.2f}M | " # Step might mean NN steps
            f"Ep: {summary['total_episodes']:<7} | "
            # f"RLScore(Avg{avg_window_size}): {summary['avg_score_window']:<6.2f} (Best: {best_score_val}) | " # Keep RL score?
            f"GameScore(Avg{avg_window_size}): {summary['avg_game_score_window']:<6.0f} (Best: {best_game_score_val}) | "
            f"V.Loss(Avg{avg_window_size}): {summary['value_loss']:.4f} (Best: {best_v_loss_val}) | "
            f"P.Loss(Avg{avg_window_size}): {summary['policy_loss']:.4f} (Best: {best_p_loss_val}) | "
            f"LR: {summary['current_lr']:.1e}"
        )
        avg_tris_cleared = summary.get("avg_triangles_cleared_window", 0.0)
        log_str += f" | TrisClr(Avg{avg_window_size}): {avg_tris_cleared:.1f}"

        print(log_str)

        self.last_log_time = time.time()

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        pass

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        pass

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        pass

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[Any] = None
    ):
        pass

    def close(self, is_cleanup: bool = False):
        """Closes the recorder (no action needed for simple console logger)."""
        print(f"[SimpleStatsRecorder] Closed (is_cleanup={is_cleanup}).")

```

File: `app_init.py`
Explanation: Removed PPO agent, collector, obs norm config, reward config. Simplified `initialize_rl_components`. Agent is now optional and represents the AlphaZero NN. Checkpoint manager now only handles agent (NN) and stats.
```python
# File: app_init.py
import pygame
import time
import traceback
import sys
import torch
from typing import TYPE_CHECKING, List, Optional, Dict, Any

from config import (
    VisConfig,
    EnvConfig,
    # Removed PPOConfig
    RNNConfig,
    ModelConfig,
    StatsConfig,
    # Removed RewardConfig
    TensorBoardConfig,
    DemoConfig,
    # Removed ObsNormConfig
    TransformerConfig,
    RANDOM_SEED,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
    DEVICE,
)
from environment.game_state import GameState
# Removed PPOAgent import
# from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler
# Removed init.rl_components_ppo import
from training.checkpoint_manager import CheckpointManager
# Removed RolloutCollector import
from app_state import AppState

if TYPE_CHECKING:
    from main_pygame import MainApp
    from agent.base_agent import BaseAgent # Hypothetical base class for NN agent


class AppInitializer:
    """Handles the initialization of core application components."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Config instances
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        # Removed self.ppo_config
        self.rnn_config = RNNConfig()
        self.train_config = app.train_config_instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        # Removed self.reward_config
        # Removed self.obs_norm_config
        self.transformer_config = TransformerConfig()

        # Components to be initialized
        self.envs: List[GameState] = [] # Keep for potential multi-env display
        self.agent: Optional["BaseAgent"] = None # Agent is now the NN
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        # Removed self.rollout_collector

    def initialize_all(self, is_reinit: bool = False):
        """Initializes all core components."""
        try:
            # GPU Memory Info (Keep)
            if self.app.device.type == "cuda":
                try:
                    self.app.total_gpu_memory_bytes = torch.cuda.get_device_properties(
                        self.app.device
                    ).total_memory
                    print(
                        f"Total GPU Memory: {self.app.total_gpu_memory_bytes / (1024**3):.2f} GB"
                    )
                except Exception as e:
                    print(f"Warning: Could not get total GPU memory: {e}")

            # Renderer and Initial Render (Keep)
            if not is_reinit:
                self.app.renderer = UIRenderer(self.app.screen, self.vis_config)
                # Adapt render_all call later if needed
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=self.app.is_process_running, # is_process_running might mean MCTS/Training now
                    status=self.app.status,
                    stats_summary={},
                    envs=[],
                    num_envs=0,
                    env_config=self.env_config,
                    cleanup_confirmation_active=False,
                    cleanup_message="",
                    last_cleanup_message_time=0,
                    tensorboard_log_dir=None,
                    plot_data={},
                    demo_env=None,
                    update_progress_details={}, # Keep for potential NN training progress
                    agent_param_count=0,
                    # worker_counts={}, # Remove worker counts for now
                )
                pygame.display.flip()
                pygame.time.delay(100)

            # Initialize "RL" components (NN, Stats, Checkpoint Manager)
            self.initialize_rl_components(
                is_reinit=is_reinit, checkpoint_to_load=self.app.checkpoint_to_load
            )

            # Demo Env and Input Handler (Keep)
            if not is_reinit:
                self.initialize_demo_env()
                self.initialize_input_handler()

            # Calculate NN parameter count if agent exists
            if self.agent and hasattr(self.agent, 'network'):
                try:
                    self.agent_param_count = sum(
                        p.numel()
                        for p in self.agent.network.parameters()
                        if p.requires_grad
                    )
                except Exception as e:
                    print(f"Warning: Could not calculate agent parameters: {e}")
                    self.agent_param_count = 0


        except Exception as init_err:
            print(f"FATAL ERROR during component initialization: {init_err}")
            traceback.print_exc()
            if self.app.renderer:
                try:
                    self.app.app_state = AppState.ERROR
                    self.app.status = "Initialization Failed"
                    self.app.renderer._render_error_screen(self.app.status)
                    pygame.display.flip()
                    time.sleep(5)
                except Exception:
                    pass
            pygame.quit()
            sys.exit(1)

    def initialize_rl_components(
        self, is_reinit: bool = False, checkpoint_to_load: Optional[str] = None
    ):
        """Initializes NN Agent, Stats Recorder, Checkpoint Manager."""
        print(f"Initializing AlphaZero components... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            # Initialize Envs (only needed for visualization now)
            # self.envs = initialize_envs(self.env_config.NUM_ENVS, self.env_config) # Removed multi-env init
            self.envs = [] # No parallel envs needed for core logic now

            # --- Initialize Agent (Neural Network) ---
            # Replace with AlphaZero NN initialization later
            # For now, set agent to None or a placeholder
            self.agent = None # Placeholder - Initialize AlphaZero NN here later
            print("Agent (NN) initialization SKIPPED (placeholder).")
            # Example placeholder for future NN init:
            # self.agent = initialize_alphazero_agent(
            #     model_config=self.model_config,
            #     rnn_config=self.rnn_config,
            #     env_config=self.env_config,
            #     transformer_config=self.transformer_config,
            #     device=self.app.device,
            # )
            # --- End Agent Init ---

            # --- Initialize Stats Recorder ---
            # Adapt initialize_stats_recorder if needed (e.g., remove PPO hparams)
            # Need to create a simplified version or adapt existing one
            # For now, assume it's adapted or create a placeholder
            try:
                from init.stats_init import initialize_stats_recorder # Hypothetical new init function
                self.stats_recorder = initialize_stats_recorder(
                    stats_config=self.stats_config,
                    tb_config=self.tensorboard_config,
                    config_dict=self.app.config_dict,
                    # Pass agent=None if NN not ready, or pass the NN agent
                    agent=self.agent,
                    env_config=self.env_config,
                    rnn_config=self.rnn_config, # Keep for potential NN config logging
                    transformer_config=self.transformer_config, # Keep for potential NN config logging
                    is_reinit=is_reinit,
                )
            except ImportError:
                 print("Warning: init.stats_init.initialize_stats_recorder not found. Skipping stats recorder init.")
                 self.stats_recorder = None # Fallback

            if self.stats_recorder is None:
                print("Warning: Stats Recorder initialization failed or skipped.")
                # Decide if this is critical - maybe allow running without stats?
                # raise RuntimeError("Stats Recorder init failed.")
            # --- End Stats Recorder Init ---

            # --- Initialize Checkpoint Manager ---
            # Checkpoint manager now handles NN agent state and stats aggregator state
            self.checkpoint_manager = CheckpointManager(
                # Pass the NN agent (or None if not ready)
                agent=self.agent,
                # Pass stats aggregator if it exists
                stats_aggregator=getattr(self.stats_recorder, 'aggregator', None),
                base_checkpoint_dir=BASE_CHECKPOINT_DIR,
                run_checkpoint_dir=get_run_checkpoint_dir(),
                load_checkpoint_path_config=checkpoint_to_load,
                device=self.app.device,
                # obs_rms_dict=None, # Removed Obs RMS
            )
            # --- End Checkpoint Manager Init ---

            # --- Load Checkpoint ---
            if self.checkpoint_manager.get_checkpoint_path_to_load():
                self.checkpoint_manager.load_checkpoint() # Loads NN state and stats state
                # Get initial step/episode count from loaded stats
                loaded_global_step, initial_episode_count = (
                    self.checkpoint_manager.get_initial_state()
                )
                # Sync episode count if needed (e.g., if MCTS tracks episodes)
                # self.mcts_manager.state.episode_count = initial_episode_count # Example
            # --- End Load Checkpoint ---

            print(f"AlphaZero components initialized in {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"Error during AlphaZero component initialization: {e}")
            traceback.print_exc()
            raise e # Re-raise to be caught by initialize_all

    def initialize_demo_env(self):
        """Initializes the separate environment for demo/debug mode."""
        print("Initializing Demo/Debug Environment...")
        try:
            self.demo_env = GameState()
            self.demo_env.reset()
            print("Demo/Debug environment initialized.")
        except Exception as e:
            print(f"ERROR initializing demo/debug environment: {e}")
            traceback.print_exc()
            self.demo_env = None

    def initialize_input_handler(self):
        """Initializes the Input Handler."""
        if not self.app.renderer:
            print("ERROR: Cannot initialize InputHandler before Renderer.")
            return

        self.app.input_handler = InputHandler(
            screen=self.app.screen,
            renderer=self.app.renderer,
            # Removed toggle_training_run_cb
            request_cleanup_cb=self.app.logic.request_cleanup,
            cancel_cleanup_cb=self.app.logic.cancel_cleanup,
            confirm_cleanup_cb=self.app.logic.confirm_cleanup,
            exit_app_cb=self.app.logic.exit_app,
            start_demo_mode_cb=self.app.logic.start_demo_mode,
            exit_demo_mode_cb=self.app.logic.exit_demo_mode,
            handle_demo_mouse_motion_cb=self.app.logic.handle_demo_mouse_motion,
            handle_demo_mouse_button_down_cb=self.app.logic.handle_demo_mouse_button_down,
            start_debug_mode_cb=self.app.logic.start_debug_mode,
            exit_debug_mode_cb=self.app.logic.exit_debug_mode,
            handle_debug_input_cb=self.app.logic.handle_debug_input,
        )
        if self.app.renderer and self.app.renderer.left_panel:
            self.app.renderer.left_panel.input_handler = self.app.input_handler
            if hasattr(self.app.renderer.left_panel, "button_status_renderer"):
                self.app.renderer.left_panel.button_status_renderer.input_handler_ref = (
                    self.app.input_handler
                )

    def close_stats_recorder(self, is_cleanup: bool = False):
        """Safely closes the stats recorder."""
        print(f"[AppInitializer] close_stats_recorder called (is_cleanup={is_cleanup})...")
        if self.stats_recorder:
            print("[AppInitializer] Stats recorder exists, attempting close...")
            try:
                if hasattr(self.stats_recorder, "close"):
                    self.stats_recorder.close(is_cleanup=is_cleanup)
                    print("[AppInitializer] stats_recorder.close() executed.")
                else:
                    print("[AppInitializer] stats_recorder has no close method.")
            except Exception as log_e:
                print(f"[AppInitializer] Error closing stats recorder on exit: {log_e}")
                traceback.print_exc()
        else:
            print("[AppInitializer] No stats recorder instance to close.")
        print("[AppInitializer] close_stats_recorder finished.")

```

File: `app_logic.py`
Explanation: Removed `toggle_training_run`. Removed PPO-specific status updates and checkpoint saving logic. Adapted cleanup to not rely on PPO workers. Status updates now reflect a simpler state (Ready, Playing, Debugging, Initializing, Error). Checkpoint saving is simplified.
```python
# File: app_logic.py
import pygame
import time
import traceback
import os
import shutil
from typing import TYPE_CHECKING, Tuple

from app_state import AppState

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppLogic:
    """Handles the core application logic and state transitions."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def check_initial_completion_status(self):
        """Checks if training target (e.g., games played) was met upon loading."""
        # This needs adaptation based on how AlphaZero training progress is tracked.
        # For now, assume completion isn't automatically checked this way.
        # if (
        #     self.app.initializer.checkpoint_manager
        #     and self.app.initializer.checkpoint_manager.training_target_step > 0 # Target might be games now
        #     and self.app.initializer.checkpoint_manager.global_step # Step might be games now
        #     >= self.app.initializer.checkpoint_manager.training_target_step
        # ):
        #     self.app.status = "Training Complete" # Or "Target Reached"
        #     print(
        #         f"Training target already reached ({self.app.initializer.checkpoint_manager.global_step:,}/{self.app.initializer.checkpoint_manager.training_target_step:,}). Ready."
        #     )
        pass # Keep simple for now

    def update_status_and_check_completion(self):
        """Updates the status text based on application state."""
        # Removed PPO-specific status logic (Collecting, Updating)
        # Removed completion check based on PPO steps

        # Update status based on AppState
        if self.app.app_state == AppState.MAIN_MENU:
            if self.app.cleanup_confirmation_active:
                self.app.status = "Confirm Cleanup"
            else:
                 # Check if training is "complete" based on new criteria if needed
                 # For now, just set to Ready if not confirming cleanup
                 self.app.status = "Ready"
        elif self.app.app_state == AppState.PLAYING:
            self.app.status = "Playing Demo"
        elif self.app.app_state == AppState.DEBUG:
            self.app.status = "Debugging Grid"
        elif self.app.app_state == AppState.INITIALIZING:
            self.app.status = "Initializing..."
        elif self.app.app_state == AppState.ERROR:
            # Status should already be set by the error handler
            pass

    # Removed toggle_training_run method

    def request_cleanup(self):
        # Removed check for is_process_running
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot request cleanup outside MainMenu.")
            return
        self.app.cleanup_confirmation_active = True
        self.app.status = "Confirm Cleanup" # Update status
        print("Cleanup requested. Confirm action.")

    def start_demo_mode(self):
        # Removed check for is_process_running
        if self.app.initializer.demo_env is None:
            print("Cannot start demo mode: Demo environment failed to initialize.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot start demo mode outside MainMenu.")
            return
        print("Entering Demo Mode...")
        self.try_save_checkpoint() # Save NN weights before switching mode?
        self.app.app_state = AppState.PLAYING
        self.app.status = "Playing Demo"
        self.app.initializer.demo_env.reset()

    def start_debug_mode(self):
        # Removed check for is_process_running
        if self.app.initializer.demo_env is None:
            print("Cannot start debug mode: Demo environment failed to initialize.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot start debug mode outside MainMenu.")
            return
        print("Entering Debug Mode...")
        self.try_save_checkpoint() # Save NN weights before switching mode?
        self.app.app_state = AppState.DEBUG
        self.app.status = "Debugging Grid"
        self.app.initializer.demo_env.reset()

    def exit_debug_mode(self):
        if self.app.app_state == AppState.DEBUG:
            print("Exiting Debug Mode...")
            self.app.app_state = AppState.MAIN_MENU
            # Removed setting is_process_running and pause_event
            self.check_initial_completion_status() # Re-check status
            self.app.status = "Ready" # Set status back to Ready

    def cancel_cleanup(self):
        self.app.cleanup_confirmation_active = False
        self.app.cleanup_message = "Cleanup cancelled."
        self.app.last_cleanup_message_time = time.time()
        self.app.status = "Ready" # Set status back
        print("Cleanup cancelled by user.")

    def confirm_cleanup(self):
        print("Cleanup confirmed by user. Starting process...")
        try:
            self._cleanup_data()
        except Exception as e:
            print(f"FATAL ERROR during cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Cleanup Failed Critically"
            self.app.app_state = AppState.ERROR
        finally:
            self.app.cleanup_confirmation_active = False
            # Status is set within _cleanup_data or error handling
            print(f"Cleanup process finished. State: {self.app.app_state}, Status: {self.app.status}")

    def exit_app(self) -> bool:
        print("Exit requested.")
        self.app.stop_event.set() # Keep stop event for main loop exit
        return False

    def exit_demo_mode(self):
        if self.app.app_state == AppState.PLAYING:
            print("Exiting Demo Mode...")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.deselect_dragged_shape()
            self.app.app_state = AppState.MAIN_MENU
            # Removed setting is_process_running and pause_event
            self.check_initial_completion_status() # Re-check status
            self.app.status = "Ready" # Set status back to Ready

    def handle_demo_mouse_motion(self, mouse_pos: Tuple[int, int]):
        if (
            self.app.app_state != AppState.PLAYING
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if demo_env.demo_dragged_shape_idx is None:
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        demo_env.update_snapped_position(grid_coords)

    def handle_demo_mouse_button_down(self, event: pygame.event.Event):
        if (
            self.app.app_state != AppState.PLAYING
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if event.button != 1:
            return
        mouse_pos = event.pos
        clicked_preview_index = self.app.ui_utils.map_screen_to_preview(mouse_pos)
        if clicked_preview_index is not None:
            if clicked_preview_index == demo_env.demo_dragged_shape_idx:
                demo_env.deselect_dragged_shape()
            else:
                demo_env.select_shape_for_drag(clicked_preview_index)
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        if grid_coords is not None:
            if (
                demo_env.demo_dragged_shape_idx is not None
                and demo_env.demo_snapped_position == grid_coords
            ):
                placed = demo_env.place_dragged_shape()
                if placed and demo_env.is_over():
                    print("[Demo] Game Over! Press ESC to exit.")
            else:
                demo_env.deselect_dragged_shape()
            return
        demo_env.deselect_dragged_shape()

    def handle_debug_input(self, event: pygame.event.Event):
        if (
            self.app.app_state != AppState.DEBUG
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                print("[Debug] Resetting grid...")
                demo_env.reset()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            clicked_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
            if clicked_coords:
                row, col = clicked_coords
                demo_env.toggle_triangle_debug(row, col)

    def _cleanup_data(self):
        """Deletes current run's checkpoint and re-initializes components."""
        from config.general import get_run_checkpoint_dir  # Local import

        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        # Removed pause_event setting and is_process_running
        self.app.app_state = AppState.INITIALIZING
        self.app.status = "Cleaning"
        messages = []

        # Render cleaning status
        if self.app.renderer:
            try:
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=False, # No process running now
                    status=self.app.status,
                    stats_summary={},
                    envs=[],
                    num_envs=0,
                    env_config=self.app.env_config,
                    cleanup_confirmation_active=False,
                    cleanup_message="",
                    last_cleanup_message_time=0,
                    tensorboard_log_dir=None,
                    plot_data={},
                    demo_env=self.app.initializer.demo_env,
                    update_progress_details={},
                    agent_param_count=getattr(self.app.initializer, "agent_param_count", 0),
                    # worker_counts={}, # Removed worker counts
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

        # --- Stop existing worker threads (if any) ---
        # Keep this structure in case new workers (MCTS/NN) are added later
        print("[Cleanup] Stopping existing worker threads (if any)...")
        self.app.worker_manager.stop_worker_threads()
        print("[Cleanup] Existing worker threads stopped.")
        # --- End Stop Workers ---

        # --- Close Stats Recorder ---
        print("[Cleanup] Closing stats recorder...")
        self.app.initializer.close_stats_recorder(is_cleanup=True)
        print("[Cleanup] Stats recorder closed.")
        # --- End Close Stats ---

        # --- Delete Checkpoint Directory ---
        print("[Cleanup] Deleting agent checkpoint file/dir...")
        try:
            save_dir = get_run_checkpoint_dir()
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
                msg = f"Run checkpoint directory deleted: {save_dir}"
            else:
                msg = f"Run checkpoint directory not found: {save_dir}"
            print(f"  - {msg}")
            messages.append(msg)
        except OSError as e:
            msg = f"Error deleting checkpoint dir: {e}"
            print(f"  - {msg}")
            messages.append(msg)
        print("[Cleanup] Checkpoint deletion attempt finished.")
        # --- End Delete Checkpoint ---

        time.sleep(0.1)
        print("[Cleanup] Re-initializing components...")
        try:
            # Re-initialize components (NN, Stats, Checkpoint Manager)
            self.app.initializer.initialize_rl_components(
                is_reinit=True, checkpoint_to_load=None
            )
            print("[Cleanup] Components re-initialized.")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
                print("[Cleanup] Demo env reset.")

            # --- Start new worker threads (if any) ---
            # Keep this structure for future workers
            print("[Cleanup] Starting new worker threads (if any)...")
            self.app.worker_manager.start_worker_threads()
            print("[Cleanup] New worker threads started.")
            # --- End Start Workers ---

            print("[Cleanup] Component re-initialization and worker start successful.")
            messages.append("Components re-initialized.")

            # --- Set state after cleanup ---
            # Removed is_process_running and pause_event
            self.app.status = "Ready"
            self.app.app_state = AppState.MAIN_MENU
            print("[Cleanup] Application state set to MAIN_MENU.")
            # --- End Set State ---

        except Exception as e:
            print(f"FATAL ERROR during re-initialization after cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Re-init Failed"
            self.app.app_state = AppState.ERROR
            messages.append("ERROR RE-INITIALIZING COMPONENTS!")
            if self.app.renderer:
                try:
                    self.app.renderer._render_error_screen(self.app.status)
                except Exception as render_err_final:
                    print(f"Warning: Failed to render error screen: {render_err_final}")

        self.app.cleanup_message = "\n".join(messages)
        self.app.last_cleanup_message_time = time.time()
        print(f"--- CLEANUP DATA COMPLETE (Final State: {self.app.app_state}, Status: {self.app.status}) ---")

    def try_save_checkpoint(self):
        """Saves checkpoint (e.g., NN weights) if in main menu."""
        # Removed check for is_process_running
        if (
            self.app.app_state == AppState.MAIN_MENU
            and self.app.initializer.checkpoint_manager
            and self.app.initializer.stats_recorder # Check if stats exist
            and hasattr(self.app.initializer.stats_recorder, "aggregator")
        ):
            print("Saving checkpoint...")
            try:
                # Get step/episode count from aggregator
                current_step = getattr(
                    self.app.initializer.stats_recorder.aggregator.storage,
                    "current_global_step", 0,
                )
                episode_count = getattr(
                    self.app.initializer.stats_recorder.aggregator.storage,
                    "total_episodes", 0,
                )
                target_step = getattr(
                     self.app.initializer.checkpoint_manager, "training_target_step", 0
                )

                self.app.initializer.checkpoint_manager.save_checkpoint(
                    current_step,
                    episode_count,
                    training_target_step=target_step,
                    is_final=False,
                )
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                traceback.print_exc()

    def save_final_checkpoint(self):
        """Saves the final checkpoint (e.g., NN weights)."""
        if (
            self.app.initializer.checkpoint_manager
            and self.app.initializer.stats_recorder
            and hasattr(self.app.initializer.stats_recorder, "aggregator")
        ):
            save_on_exit = (
                self.app.status != "Cleaning" and self.app.app_state != AppState.ERROR
            )

            if save_on_exit:
                print("Performing final checkpoint save...")
                try:
                    current_step = getattr(
                        self.app.initializer.stats_recorder.aggregator.storage,
                        "current_global_step", 0,
                    )
                    episode_count = getattr(
                        self.app.initializer.stats_recorder.aggregator.storage,
                        "total_episodes", 0,
                    )
                    target_step = getattr(
                         self.app.initializer.checkpoint_manager, "training_target_step", 0
                    )
                    self.app.initializer.checkpoint_manager.save_checkpoint(
                        current_step,
                        episode_count,
                        training_target_step=target_step,
                        is_final=True,
                    )
                except Exception as final_save_err:
                    print(f"Error during final checkpoint save: {final_save_err}")
                    traceback.print_exc()
            else:
                print("Skipping final checkpoint save.")

```

File: `app_workers.py`
Explanation: Removed PPO worker classes (`EnvironmentRunner`, `TrainingWorker`). Kept the manager structure but `start_worker_threads` and `stop_worker_threads` are now placeholders.
```python
# File: app_workers.py
import threading
import queue
import time
import traceback
from typing import TYPE_CHECKING, Optional # Added Optional

# Removed worker imports
# from workers import EnvironmentRunner, TrainingWorker

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppWorkerManager:
    """Manages the creation, starting, and stopping of worker threads."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Remove specific worker thread attributes
        # self.env_runner_thread: Optional[EnvironmentRunner] = None
        # self.training_worker_thread: Optional[TrainingWorker] = None
        # Add placeholders for future workers if needed
        self.mcts_worker_thread: Optional[threading.Thread] = None
        self.nn_training_worker_thread: Optional[threading.Thread] = None
        print("[AppWorkerManager] Initialized (No workers started by default).")


    def start_worker_threads(self):
        """Creates and starts worker threads (MCTS, NN Training - Placeholder)."""
        # --- This needs to be implemented based on AlphaZero architecture ---
        print("[AppWorkerManager] start_worker_threads called (Placeholder - No workers started).")
        # Example structure:
        # if not self.app.initializer.agent or not self.app.initializer.mcts_manager:
        #     print("ERROR: Cannot start workers, core components not initialized.")
        #     self.app.app_state = self.app.app_state.ERROR
        #     self.app.status = "Worker Init Failed"
        #     return
        #
        # print("Starting AlphaZero worker threads...")
        # self.app.stop_event.clear()
        # self.app.pause_event.clear() # Or set based on initial state
        #
        # # MCTS Self-Play Worker(s)
        # self.mcts_worker_thread = MCTSSelfPlayWorker(...)
        # self.mcts_worker_thread.start()
        #
        # # NN Training Worker
        # self.nn_training_worker_thread = NNTrainingWorker(...)
        # self.nn_training_worker_thread.start()
        #
        # print("AlphaZero worker threads started.")
        pass

    def stop_worker_threads(self):
        """Signals worker threads to stop and waits for them to join."""
        if self.app.stop_event.is_set():
             print("[AppWorkerManager] Stop event already set.")
             return

        print("[AppWorkerManager] Stopping worker threads (Placeholder)...")
        self.app.stop_event.set()
        # self.app.pause_event.clear() # Ensure threads aren't stuck paused if pause is used

        join_timeout = 5.0

        # --- Join future worker threads ---
        if self.mcts_worker_thread and self.mcts_worker_thread.is_alive():
            print("[AppWorkerManager] Joining MCTS worker...")
            self.mcts_worker_thread.join(timeout=join_timeout)
            if self.mcts_worker_thread.is_alive():
                print("[AppWorkerManager] MCTS worker thread did not join cleanly.")
            self.mcts_worker_thread = None

        if self.nn_training_worker_thread and self.nn_training_worker_thread.is_alive():
            print("[AppWorkerManager] Joining NN Training worker...")
            self.nn_training_worker_thread.join(timeout=join_timeout)
            if self.nn_training_worker_thread.is_alive():
                print("[AppWorkerManager] NN Training worker thread did not join cleanly.")
            self.nn_training_worker_thread = None
        # --- End Join ---

        # Clear queues if used by workers
        # Example:
        # while not self.app.experience_queue.empty():
        #     try:
        #         self.app.experience_queue.get_nowait()
        #     except queue.Empty:
        #         break

        print("[AppWorkerManager] Worker threads stopped.")

```

File: `main_pygame.py`
Explanation: Removed PPO imports, worker logic, PPO config instantiation. Adapted initialization, main loop (removed PPO status/progress updates), and shutdown. Removed `is_process_running` and `pause_event` as they were tied to PPO workers.
```python
# File: main_pygame.py
import pygame
import sys
import time
import threading
import logging
import argparse
import os
import traceback
from typing import Optional, List, Dict, Any
import torch

# Resource Monitoring Import (Keep)
try:
    import psutil
except ImportError:
    print("Warning: psutil not found. CPU/Memory usage monitoring will be disabled.")
    print("Install it using: pip install psutil")
    psutil = None

# Path Adjustment (Keep)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Config Imports (Updated)
try:
    from config import (
        VisConfig,
        EnvConfig,
        # Removed PPOConfig
        ModelConfig,
        StatsConfig,
        TrainConfig,
        TensorBoardConfig,
        DemoConfig,
        RNNConfig,
        TransformerConfig,
        # Removed ObsNormConfig
        # Removed RewardConfig (or keep if needed for env rewards)
        DEVICE,
        RANDOM_SEED,
        # Removed TOTAL_TRAINING_STEPS
        BASE_CHECKPOINT_DIR,
        BASE_LOG_DIR,
        set_device,
        get_run_id,
        set_run_id,
        get_run_checkpoint_dir,
        get_run_log_dir,
        get_console_log_dir,
        print_config_info_and_validate,
        get_config_dict,
    )
    from utils.helpers import get_device as get_torch_device, set_random_seeds
    from logger import TeeLogger
    from utils.init_checks import run_pre_checks
    from utils.types import AgentStateDict # Keep for potential NN state dict type

except ImportError as e:
    print(f"Error importing configuration classes or utils: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during config/util import: {e}")
    traceback.print_exc()
    sys.exit(1)

# Component Imports (Updated)
try:
    from environment.game_state import GameState
    from ui.renderer import UIRenderer
    # Removed PPOAgent import
    # Removed RolloutStorage, RolloutCollector imports
    # Removed worker imports (EnvironmentRunner, TrainingWorker)

    # Keep stats imports
    from stats import (
        StatsRecorderBase,
        SimpleStatsRecorder,
        TensorBoardStatsRecorder,
        StatsAggregator,
    )
    # Keep CheckpointManager import (will manage NN weights now)
    from training.checkpoint_manager import (
        CheckpointManager,
        find_latest_run_and_checkpoint,
    )
    from app_state import AppState
    from app_init import AppInitializer
    from app_logic import AppLogic
    from app_workers import AppWorkerManager # Keep manager structure
    from app_setup import (
        initialize_pygame,
        initialize_directories,
        load_and_validate_configs,
    )
    from app_ui_utils import AppUIUtils
    from ui.input_handler import InputHandler
    import queue # Keep queue if needed for future workers

except ImportError as e:
    print(f"Error importing application components: {e}")
    traceback.print_exc()
    sys.exit(1)

# Logging Setup (Keep)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MainApp:
    """Main application class orchestrating Pygame UI, game logic, and potentially AlphaZero components."""

    def __init__(self, checkpoint_to_load: Optional[str] = None):
        # --- Configuration (Updated) ---
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        # Removed self.ppo_config
        self.rnn_config = RNNConfig()
        self.train_config_instance = TrainConfig()
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        # Removed self.reward_config
        # Removed self.obs_norm_config
        self.transformer_config = TransformerConfig()
        self.config_dict = get_config_dict()

        # --- Core Components (Keep) ---
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None

        # --- State (Updated) ---
        self.app_state: AppState = AppState.INITIALIZING
        # Removed self.is_process_running
        self.status: str = "Initializing..."
        self.running: bool = True
        self.update_progress_details: Dict[str, Any] = {} # Keep for potential NN training progress

        # --- Threading & Communication (Updated) ---
        self.stop_event = threading.Event() # Keep for main loop exit signal
        # Removed self.pause_event (was tied to PPO workers)
        # Keep queue if needed for future workers, otherwise remove
        # self.experience_queue = queue.Queue(maxsize=10)

        # --- RL Components (Managed by Initializer - Updated) ---
        # Placeholders for NN agent, stats, checkpoint manager
        self.envs: List[GameState] = [] # Only for visualization now
        self.agent: Optional[Any] = None # Placeholder for AlphaZero NN Agent
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        # Removed self.rollout_collector
        self.demo_env: Optional[GameState] = None

        # --- Helper Classes (Keep) ---
        self.device = get_torch_device()
        set_device(self.device)
        self.checkpoint_to_load = checkpoint_to_load
        self.initializer = AppInitializer(self)
        self.logic = AppLogic(self)
        self.worker_manager = AppWorkerManager(self) # Manages future workers
        self.ui_utils = AppUIUtils(self)

        # --- UI State (Keep) ---
        self.cleanup_confirmation_active: bool = False
        self.cleanup_message: str = ""
        self.last_cleanup_message_time: float = 0.0
        self.total_gpu_memory_bytes: Optional[int] = None

        # --- Resource Monitoring (Keep) ---
        self.last_resource_update_time: float = 0.0
        self.resource_update_interval: float = 1.0

    def initialize(self):
        """Initializes Pygame, directories, configs, and core components."""
        logger.info("--- Application Initialization ---")
        self.screen, self.clock = initialize_pygame(self.vis_config)
        initialize_directories()
        # Configs are instantiated in __init__

        set_random_seeds(RANDOM_SEED)
        run_pre_checks()

        # Initialize core components via AppInitializer (now initializes NN, Stats, etc.)
        self.app_state = AppState.INITIALIZING
        self.initializer.initialize_all()

        # Set input handler reference in renderer (Keep)
        if self.renderer and self.initializer.app.input_handler:
            self.renderer.set_input_handler(self.initializer.app.input_handler)

        # Start worker threads (Placeholder - will start MCTS/NN workers later)
        self.worker_manager.start_worker_threads()

        # Check initial completion status (Adapt later if needed)
        self.logic.check_initial_completion_status()
        self.status = "Ready" # Default status after init
        self.app_state = AppState.MAIN_MENU

        logger.info("--- Initialization Complete ---")
        if self.tensorboard_config.LOG_DIR:
            tb_path = os.path.abspath(get_run_log_dir())
            logger.info(f"--- TensorBoard logs: tensorboard --logdir {tb_path} ---")

    def _update_resource_stats(self):
        """Updates CPU, Memory, and GPU usage in the StatsAggregator."""
        current_time = time.time()
        if (
            current_time - self.last_resource_update_time
            < self.resource_update_interval
        ):
            return

        # Check if stats recorder and aggregator exist
        if not self.initializer.stats_recorder or not hasattr(
            self.initializer.stats_recorder, "aggregator"
        ):
            return

        aggregator = self.initializer.stats_recorder.aggregator
        storage = aggregator.storage

        cpu_percent, mem_percent, gpu_mem_percent = 0.0, 0.0, 0.0

        if psutil:
            try:
                cpu_percent = psutil.cpu_percent()
                mem_percent = psutil.virtual_memory().percent
            except Exception as e:
                logger.warning(f"Error getting CPU/Mem usage: {e}")

        if self.device.type == "cuda" and self.total_gpu_memory_bytes:
            try:
                allocated = torch.cuda.memory_allocated(self.device)
                gpu_mem_percent = (allocated / self.total_gpu_memory_bytes) * 100.0
            except Exception as e:
                logger.warning(f"Error getting GPU memory usage: {e}")
                gpu_mem_percent = 0.0

        # Update aggregator storage directly (thread-safe via lock)
        with aggregator._lock:
            storage.current_cpu_usage = cpu_percent
            storage.current_memory_usage = mem_percent
            storage.current_gpu_memory_usage_percent = gpu_mem_percent
            # Append to deques if they exist (check needed after refactor)
            if hasattr(storage, 'cpu_usage'): storage.cpu_usage.append(cpu_percent)
            if hasattr(storage, 'memory_usage'): storage.memory_usage.append(mem_percent)
            if hasattr(storage, 'gpu_memory_usage_percent'): storage.gpu_memory_usage_percent.append(gpu_mem_percent)

        self.last_resource_update_time = current_time

    def run_main_loop(self):
        """The main application loop."""
        logger.info("Starting main application loop...")
        while self.running:
            dt = self.clock.tick(self.vis_config.FPS) / 1000.0

            # Handle Input (Keep)
            if self.input_handler:
                self.running = self.input_handler.handle_input(
                    self.app_state.value, self.cleanup_confirmation_active
                )
                if not self.running:
                    self.stop_event.set()
                    break
            else: # Fallback exit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        self.stop_event.set()
                        break
                if not self.running:
                    break

            # --- Fetch Update Progress Details (Keep for potential NN training) ---
            # This needs to be adapted to get progress from the NN training worker if implemented
            self.update_progress_details = {} # Placeholder
            # Example:
            # if self.worker_manager.nn_training_worker_thread and self.worker_manager.nn_training_worker_thread.is_alive():
            #     try:
            #         self.update_progress_details = self.worker_manager.nn_training_worker_thread.get_update_progress_details()
            #     except Exception as e:
            #         logger.warning(f"Could not get update progress details: {e}")
            #         self.update_progress_details = {}
            # else:
            #     self.update_progress_details = {}
            # --- End Fetch Update Progress Details ---

            # Update Logic & State (Simplified)
            self.logic.update_status_and_check_completion()

            # Update Resource Stats (Keep)
            self._update_resource_stats()

            # Update Demo Env Timers (Keep)
            if self.initializer.demo_env:
                try:
                    # Timers are updated internally by demo_env.step or manually if needed
                    # self.initializer.demo_env._update_timers() # Call if step isn't called automatically
                    pass # Assume timers update within demo logic or step
                except Exception as timer_err:
                    logger.error(f"Error updating demo env timers: {timer_err}", exc_info=False)

            # Render UI (Adapted)
            if self.renderer:
                plot_data = {}
                stats_summary = {}
                tb_log_dir = None
                agent_params = 0
                # Removed worker_counts

                if self.initializer.stats_recorder:
                    plot_data = self.initializer.stats_recorder.get_plot_data()
                    current_step = 0
                    if hasattr(self.initializer.stats_recorder, "aggregator"):
                        current_step = getattr(
                            self.initializer.stats_recorder.aggregator.storage,
                            "current_global_step", 0,
                        )
                    stats_summary = self.initializer.stats_recorder.get_summary(current_step)
                    if isinstance(self.initializer.stats_recorder, TensorBoardStatsRecorder):
                        tb_log_dir = self.initializer.stats_recorder.log_dir

                if self.initializer.agent:
                    agent_params = self.initializer.agent_param_count

                # Adapt render_all call
                self.renderer.render_all(
                    app_state=self.app_state.value,
                    is_process_running=False, # No PPO process running
                    status=self.status,
                    stats_summary=stats_summary,
                    envs=self.initializer.envs, # Pass empty list or visualized envs
                    num_envs=self.env_config.NUM_ENVS, # Pass total configured (for layout)
                    env_config=self.env_config,
                    cleanup_confirmation_active=self.cleanup_confirmation_active,
                    cleanup_message=self.cleanup_message,
                    last_cleanup_message_time=self.last_cleanup_message_time,
                    tensorboard_log_dir=tb_log_dir,
                    plot_data=plot_data,
                    demo_env=self.initializer.demo_env,
                    update_progress_details=self.update_progress_details, # Pass NN progress later
                    agent_param_count=agent_params,
                    # worker_counts={}, # Removed worker counts
                )
            else: # Fallback render
                self.screen.fill((20, 0, 0))
                font = pygame.font.Font(None, 30)
                text_surf = font.render("Renderer Error", True, (255, 50, 50))
                self.screen.blit(text_surf, text_surf.get_rect(center=self.screen.get_rect().center))
                pygame.display.flip()

        logger.info("Main application loop exited.")

    def shutdown(self):
        """Cleans up resources and exits."""
        logger.info("Initiating shutdown sequence...")

        # Signal threads to stop (Keep stop_event)
        logger.info("Setting stop event for worker threads (if any).")
        self.stop_event.set()
        # Removed pause_event clearing

        # Stop and join worker threads (Placeholder)
        logger.info("Stopping worker threads (if any)...")
        self.worker_manager.stop_worker_threads()
        logger.info("Worker threads stopped.")

        # Save final checkpoint (NN weights, stats)
        logger.info("Attempting final checkpoint save...")
        self.logic.save_final_checkpoint()
        logger.info("Final checkpoint save attempt finished.")

        # Close stats recorder (Keep)
        logger.info("Closing stats recorder (before pygame.quit)...")
        self.initializer.close_stats_recorder()
        logger.info("Stats recorder closed.")

        # Quit Pygame (Keep)
        logger.info("Quitting Pygame...")
        pygame.quit()
        logger.info("Pygame quit.")
        logger.info("Shutdown complete.")


# Global variable for TeeLogger instance (Keep)
tee_logger_instance: Optional[TeeLogger] = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaTri Trainer") # Updated description
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        # Updated help text
        help="Path to a specific checkpoint file to load (e.g., NN weights). Overrides auto-resume.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    # Setup Logging (Keep, checkpoint finding logic might adapt)
    if args.load_checkpoint:
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                print(f"Using Run ID from checkpoint path: {get_run_id()}")
            else:
                get_run_id()
        except Exception:
            get_run_id()
    else:
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            print(f"Resuming Run ID: {get_run_id()}")
        else:
            get_run_id()

    # Setup TeeLogger (Keep)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        log_file_dir = get_console_log_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, "console_output.log")
        tee_logger_instance = TeeLogger(log_file_path, sys.stdout)
        sys.stdout = tee_logger_instance
        sys.stderr = tee_logger_instance
    except Exception as e:
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)

    # Set logging level (Keep)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.info(f"Logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {get_run_id()}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")

    # Create and Run Application (Keep structure)
    app = None
    exit_code = 0
    try:
        app = MainApp(checkpoint_to_load=args.load_checkpoint)
        app.initialize()
        app.run_main_loop()
        app.shutdown()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down.")
        if app: app.shutdown()
        else: pygame.quit()
        exit_code = 130
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in main: {e}", exc_info=True)
        if app: app.shutdown()
        else: pygame.quit()
        exit_code = 1
    finally:
        # Restore stdout/stderr (Keep)
        print("[Main Finally] Restoring stdout/stderr and closing logger...")
        if tee_logger_instance:
            try:
                if isinstance(sys.stdout, TeeLogger): sys.stdout.flush()
                if isinstance(sys.stderr, TeeLogger): sys.stderr.flush()
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                tee_logger_instance.close()
                print("[Main Finally] TeeLogger closed and streams restored.")
            except Exception as log_close_err:
                original_stdout.write(f"ERROR closing TeeLogger: {log_close_err}\n")
                traceback.print_exc(file=original_stderr)
        print(f"[Main Finally] Exiting with code {exit_code}.")
        sys.exit(exit_code)

```

File: `ui/input_handler.py`
Explanation: Removed `toggle_training_run_cb` and references to the Run/Stop button logic tied to it.
```python
# File: ui/input_handler.py
import pygame
from typing import Tuple, Callable, Dict, TYPE_CHECKING

# Type Aliases for Callbacks
HandleDemoMouseMotionCallback = Callable[[Tuple[int, int]], None]
HandleDemoMouseButtonDownCallback = Callable[[pygame.event.Event], None]
# Removed ToggleTrainingRunCallback
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]
StartDemoModeCallback = Callable[[], None]
ExitDemoModeCallback = Callable[[], None]
StartDebugModeCallback = Callable[[], None]
ExitDebugModeCallback = Callable[[], None]
HandleDebugInputCallback = Callable[[pygame.event.Event], None]

if TYPE_CHECKING:
    from .renderer import UIRenderer
    from app_state import AppState


class InputHandler:
    """Handles Pygame events and triggers callbacks based on application state."""

    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",
        # Removed toggle_training_run_cb
        request_cleanup_cb: RequestCleanupCallback,
        cancel_cleanup_cb: CancelCleanupCallback,
        confirm_cleanup_cb: ConfirmCleanupCallback,
        exit_app_cb: ExitAppCallback,
        start_demo_mode_cb: StartDemoModeCallback,
        exit_demo_mode_cb: ExitDemoModeCallback,
        handle_demo_mouse_motion_cb: HandleDemoMouseMotionCallback,
        handle_demo_mouse_button_down_cb: HandleDemoMouseButtonDownCallback,
        start_debug_mode_cb: StartDebugModeCallback,
        exit_debug_mode_cb: ExitDebugModeCallback,
        handle_debug_input_cb: HandleDebugInputCallback,
    ):
        self.screen = screen
        self.renderer = renderer
        # Removed self.toggle_training_run_cb
        self.request_cleanup_cb = request_cleanup_cb
        self.cancel_cleanup_cb = cancel_cleanup_cb
        self.confirm_cleanup_cb = confirm_cleanup_cb
        self.exit_app_cb = exit_app_cb
        self.start_demo_mode_cb = start_demo_mode_cb
        self.exit_demo_mode_cb = exit_demo_mode_cb
        self.handle_demo_mouse_motion_cb = handle_demo_mouse_motion_cb
        self.handle_demo_mouse_button_down_cb = handle_demo_mouse_button_down_cb
        self.start_debug_mode_cb = start_debug_mode_cb
        self.exit_debug_mode_cb = exit_debug_mode_cb
        self.handle_debug_input_cb = handle_debug_input_cb

        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

        # Button rects (Run button removed)
        # self.run_btn_rect = pygame.Rect(0, 0, 0, 0) # Removed
        self.cleanup_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.demo_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.debug_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_yes_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_no_rect = pygame.Rect(0, 0, 0, 0)
        self._update_button_rects()

    def _update_button_rects(self):
        """Calculates button rects based on initial layout assumptions."""
        button_height = 40
        button_y_pos = 10
        # Removed run_button_width
        cleanup_button_width = 160
        demo_button_width = 120
        debug_button_width = 120
        button_spacing = 10

        # Start directly with cleanup button
        current_x = button_spacing
        # self.run_btn_rect = pygame.Rect(button_spacing, button_y_pos, run_button_width, button_height) # Removed
        # current_x = self.run_btn_rect.right + button_spacing # Removed
        self.cleanup_btn_rect = pygame.Rect(
            current_x, button_y_pos, cleanup_button_width, button_height
        )
        current_x = self.cleanup_btn_rect.right + button_spacing
        self.demo_btn_rect = pygame.Rect(
            current_x, button_y_pos, demo_button_width, button_height
        )
        current_x = self.demo_btn_rect.right + button_spacing
        self.debug_btn_rect = pygame.Rect(
            current_x, button_y_pos, debug_button_width, button_height
        )

        sw, sh = self.screen.get_size()
        self.confirm_yes_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_no_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

    def handle_input(
        self, app_state_str: str, cleanup_confirmation_active: bool
    ) -> bool:
        """Processes Pygame events. Returns True to continue running, False to exit."""
        from app_state import AppState

        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            mouse_pos = (0, 0)

        sw, sh = self.screen.get_size()
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

        if (
            app_state_str == AppState.PLAYING.value
            and self.renderer
            and self.renderer.demo_renderer
        ):
            self.shape_preview_rects = self.renderer.demo_renderer.get_shape_preview_rects()
        else:
            self.shape_preview_rects.clear()

        # Removed hover check

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return self.exit_app_cb()

            if event.type == pygame.VIDEORESIZE:
                try:
                    new_w, new_h = max(320, event.w), max(240, event.h)
                    self.screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)
                    self._update_ui_screen_references(self.screen)
                    self._update_button_rects()
                    if hasattr(self.renderer, "force_redraw"):
                        self.renderer.force_redraw()
                    print(f"Window resized: {new_w}x{new_h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")
                continue

            if cleanup_confirmation_active:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.cancel_cleanup_cb()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.confirm_yes_rect.collidepoint(mouse_pos):
                        self.confirm_cleanup_cb()
                    elif self.confirm_no_rect.collidepoint(mouse_pos):
                        self.cancel_cleanup_cb()
                continue

            elif app_state_str == AppState.PLAYING.value:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_demo_mode_cb()
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_demo_mouse_motion_cb(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_demo_mouse_button_down_cb(event)

            elif app_state_str == AppState.DEBUG.value:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_debug_mode_cb()
                    else:
                        self.handle_debug_input_cb(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_debug_input_cb(event)

            elif app_state_str == AppState.MAIN_MENU.value:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return self.exit_app_cb()
                    # Removed 'P' key binding for toggle run
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Removed run_btn_rect check
                    if self.cleanup_btn_rect.collidepoint(mouse_pos):
                        self.request_cleanup_cb()
                    elif self.demo_btn_rect.collidepoint(mouse_pos):
                        self.start_demo_mode_cb()
                    elif self.debug_btn_rect.collidepoint(mouse_pos):
                        self.start_debug_mode_cb()

            elif app_state_str == AppState.ERROR.value:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return self.exit_app_cb()

        return True

    def _update_ui_screen_references(self, new_screen: pygame.Surface):
        """Updates the screen reference in the renderer and its sub-components."""
        components_to_update = [
            self.renderer,
            getattr(self.renderer, "left_panel", None),
            getattr(self.renderer, "game_area", None),
            getattr(self.renderer, "overlays", None),
            # Removed tooltips
            getattr(self.renderer, "demo_renderer", None),
            getattr(getattr(self.renderer, "demo_renderer", None), "grid_renderer", None),
            getattr(getattr(self.renderer, "demo_renderer", None), "preview_renderer", None),
            getattr(getattr(self.renderer, "demo_renderer", None), "hud_renderer", None),
        ]
        for component in components_to_update:
            if component and hasattr(component, "screen"):
                component.screen = new_screen

```

File: `ui/panels/left_panel_components/button_status_renderer.py`
Explanation: Removed Run/Stop button rendering and logic. Removed progress bars and SPS display. Simplified status display.
```python
# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
import time
import math
from typing import Dict, Tuple, Any, Optional

from config import WHITE, YELLOW, RED, GOOGLE_COLORS, LIGHTG
from utils.helpers import format_eta
from ui.input_handler import InputHandler


class ButtonStatusRenderer:
    """Renders the top buttons (excluding Run/Stop), and compact status block."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("status", pygame.font.Font(None, 28))
        self.status_label_font = fonts.get("notification_label", pygame.font.Font(None, 16))
        # Removed self.progress_font
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.input_handler_ref: Optional[InputHandler] = None

    def _draw_button(
        self,
        rect: pygame.Rect,
        text: str,
        color: Tuple[int, int, int],
        enabled: bool = True,
    ):
        """Helper to draw a single button, optionally grayed out."""
        final_color = (
            color if enabled else tuple(max(30, c // 2) for c in color[:3])
        )
        pygame.draw.rect(self.screen, final_color, rect, border_radius=5)
        text_color = WHITE if enabled else (150, 150, 150)
        if self.ui_font:
            label_surface = self.ui_font.render(text, True, text_color)
            self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        else:
            pygame.draw.line(self.screen, RED, rect.topleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, RED, rect.topright, rect.bottomleft, 2)

    def _render_compact_status(
        self, y_start: int, panel_width: int, status: str, stats_summary: Dict[str, Any]
    ) -> int:
        """Renders the compact status block below buttons."""
        x_margin, current_y = 10, y_start
        line_height_status = self.status_font.get_linesize()
        line_height_label = self.status_label_font.get_linesize()

        status_text = f"Status: {status}"
        status_color = YELLOW
        if "Error" in status: status_color = RED
        elif "Ready" in status: status_color = WHITE
        elif "Debugging" in status: status_color = (200, 100, 200)
        elif "Playing" in status: status_color = (100, 150, 200)
        elif "Initializing" in status: status_color = LIGHTG
        elif "Cleaning" in status: status_color = (200, 100, 100)
        elif "Confirm" in status: status_color = (200, 150, 50)

        status_surface = self.status_font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(status_surface, status_rect)
        current_y += line_height_status

        # Display basic info like total episodes/games played
        global_step = stats_summary.get("global_step", 0) # Step might mean games or NN steps
        total_episodes = stats_summary.get("total_episodes", 0)

        global_step_str = f"{global_step:,}".replace(",", "_")
        eps_str = f"~{total_episodes} Eps" # Or Games

        line2_text = f"{global_step_str} Steps | {eps_str}" # Simplified info line
        line2_surface = self.status_label_font.render(line2_text, True, LIGHTG)
        line2_rect = line2_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(line2_surface, line2_rect)

        current_y += line_height_label + 2
        return current_y

    # Removed _render_single_progress_bar
    # Removed _render_detailed_progress_bars

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_process_running: bool, # Keep for potential future use (e.g., MCTS running)
        status: str,
        stats_summary: Dict[str, Any],
        update_progress_details: Dict[str, Any], # Keep for potential NN progress
    ) -> int:
        """Renders buttons (excluding Run/Stop) and status. Returns next_y."""
        from app_state import AppState

        next_y = y_start

        # Get button rects from InputHandler
        # Removed run_btn_rect
        cleanup_btn_rect = (
            self.input_handler_ref.cleanup_btn_rect
            if self.input_handler_ref
            else pygame.Rect(10, y_start, 160, 40)
        )
        demo_btn_rect = (
            self.input_handler_ref.demo_btn_rect
            if self.input_handler_ref
            else pygame.Rect(cleanup_btn_rect.right + 10, y_start, 120, 40)
        )
        debug_btn_rect = (
            self.input_handler_ref.debug_btn_rect
            if self.input_handler_ref
            else pygame.Rect(demo_btn_rect.right + 10, y_start, 120, 40)
        )

        # Removed Run/Stop button rendering

        # Render other buttons
        buttons_enabled = app_state == AppState.MAIN_MENU.value
        self._draw_button(
            cleanup_btn_rect, "Cleanup This Run", (100, 40, 40), enabled=buttons_enabled
        )
        self._draw_button(
            demo_btn_rect, "Play Demo", (40, 100, 40), enabled=buttons_enabled
        )
        self._draw_button(
            debug_btn_rect, "Debug Mode", (100, 40, 100), enabled=buttons_enabled
        )
        # Set next_y below the buttons
        button_bottom = max(cleanup_btn_rect.bottom, demo_btn_rect.bottom, debug_btn_rect.bottom)
        next_y = button_bottom + 10

        # Render Status Block
        status_block_y = next_y
        # Removed check for "Updating Agent" status to render progress bars
        # Always render the compact status block now
        next_y = self._render_compact_status(
            status_block_y, panel_width, status, stats_summary
        )

        # Render NN training progress bar if applicable (using update_progress_details)
        # Example placeholder:
        # if update_progress_details and update_progress_details.get('phase') == 'Training NN':
        #     next_y = self._render_detailed_progress_bars(...) # Adapt this call

        return next_y

```

File: `ui/panels/left_panel_components/info_text_renderer.py`
Explanation: Removed worker counts, LR display (can be added back if NN LR is tracked), and PPO-specific network details. Simplified network description.
```python
# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple

from config import RNNConfig, TransformerConfig, ModelConfig, WHITE, LIGHTG, GRAY

class InfoTextRenderer:
    """Renders essential non-plotted information text."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.rnn_config = RNNConfig()
        self.transformer_config = TransformerConfig()
        self.model_config_net = ModelConfig.Network()
        self.resource_font = fonts.get("logdir", pygame.font.Font(None, 16))
        self.stats_summary_cache: Dict[str, Any] = {}

    def _get_network_description(self) -> str:
        """Builds a description string based on network components."""
        # Adapt based on AlphaZero NN architecture later
        return "AlphaZero Neural Network" # Placeholder

    def _get_network_details(self) -> str:
        """Builds a detailed string of network configuration."""
        # Adapt based on AlphaZero NN architecture later
        details = []
        # Example: Add details about ResNet blocks, policy/value heads if known
        # cnn_str = str(self.model_config_net.CONV_CHANNELS).replace(" ", "")
        # details.append(f"CNN Base: {cnn_str}")
        return "Details TBD" # Placeholder

    def _get_live_resource_usage(self) -> Dict[str, str]:
        """Fetches live CPU, Memory, and GPU Memory usage from cached summary."""
        from config.general import DEVICE

        usage = {"CPU": "N/A", "Mem": "N/A", "GPU Mem": "N/A"}
        cpu_val = self.stats_summary_cache.get("current_cpu_usage")
        mem_val = self.stats_summary_cache.get("current_memory_usage")
        gpu_val = self.stats_summary_cache.get("current_gpu_memory_usage_percent")

        if cpu_val is not None: usage["CPU"] = f"{cpu_val:.1f}%"
        if mem_val is not None: usage["Mem"] = f"{mem_val:.1f}%" # Fixed quote

        device_type = DEVICE.type if DEVICE else "cpu"
        if gpu_val is not None:
            usage["GPU Mem"] = f"{gpu_val:.1f}%" if device_type == "cuda" else "N/A (CPU)"
        elif device_type != "cuda":
            usage["GPU Mem"] = "N/A (CPU)"
        return usage

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],
        panel_width: int,
        agent_param_count: int, # Keep for NN param count
        worker_counts: Dict[str, int], # Keep structure, but content will change
    ) -> int:
        """Renders the info text block. Returns next_y."""
        from config.general import DEVICE

        self.stats_summary_cache = stats_summary
        ui_font, detail_font, resource_font = (
            self.fonts.get("ui"), self.fonts.get("logdir"), self.resource_font
        )
        if not ui_font or not detail_font or not resource_font: return y_start

        line_height_ui, line_height_detail, line_height_resource = (
            ui_font.get_linesize(), detail_font.get_linesize(), resource_font.get_linesize()
        )
        device_type_str = DEVICE.type.upper() if DEVICE else "CPU"
        network_desc, network_details = self._get_network_description(), self._get_network_details()
        param_str = f"{agent_param_count / 1e6:.2f} M" if agent_param_count > 0 else "N/A"
        start_time_unix = stats_summary.get("start_time", 0.0)
        start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_unix)) if start_time_unix > 0 else "N/A"
        # Removed worker_str and lr_str (can be added back if needed)

        info_lines = [
            ("Device", device_type_str),
            ("Network", network_desc),
            ("Params", param_str),
            # ("LR", lr_str), # Removed LR for now
            # ("Workers", worker_str), # Removed Workers for now
            ("Run Started", start_time_str),
        ]
        last_y, x_pos_key, x_pos_val_offset, current_y = y_start, 10, 5, y_start + 5

        for idx, (key, value_str) in enumerate(info_lines):
            line_y = current_y + idx * line_height_ui
            try:
                key_surf = ui_font.render(f"{key}:", True, LIGHTG)
                key_rect = key_surf.get_rect(topleft=(x_pos_key, line_y))
                self.screen.blit(key_surf, key_rect)
                value_surf = ui_font.render(f"{value_str}", True, WHITE)
                value_rect = value_surf.get_rect(topleft=(key_rect.right + x_pos_val_offset, line_y))
                clip_width = max(0, panel_width - value_rect.left - 10)
                blit_area = pygame.Rect(0, 0, clip_width, value_rect.height) if value_rect.width > clip_width else None
                self.screen.blit(value_surf, value_rect, area=blit_area)
                last_y = key_rect.union(value_rect).bottom
            except Exception as e:
                print(f"Error rendering stat line '{key}': {e}")
                last_y = line_y + line_height_ui

        current_y = last_y + 2
        try:
            detail_surf = detail_font.render(network_details, True, WHITE)
            detail_rect = detail_surf.get_rect(topleft=(x_pos_key, current_y))
            clip_width_detail = max(0, panel_width - detail_rect.left - 10)
            blit_area_detail = pygame.Rect(0, 0, clip_width_detail, detail_rect.height) if detail_rect.width > clip_width_detail else None
            self.screen.blit(detail_surf, detail_rect, area=blit_area_detail)
            last_y = detail_rect.bottom
        except Exception as e:
            print(f"Error rendering network details: {e}")
            last_y = current_y + line_height_detail

        current_y = last_y + 4
        resource_usage = self._get_live_resource_usage()
        resource_str = f"Live Usage | CPU: {resource_usage['CPU']} | Mem: {resource_usage['Mem']} | GPU Mem: {resource_usage['GPU Mem']}"
        try:
            resource_surf = resource_font.render(resource_str, True, WHITE)
            resource_rect = resource_surf.get_rect(topleft=(x_pos_key, current_y))
            clip_width_resource = max(0, panel_width - resource_rect.left - 10)
            blit_area_resource = pygame.Rect(0, 0, clip_width_resource, resource_rect.height) if resource_rect.width > clip_width_resource else None
            self.screen.blit(resource_surf, resource_rect, area=blit_area_resource)
            last_y = resource_rect.bottom
        except Exception as e:
            print(f"Error rendering resource usage: {e}")
            last_y = current_y + line_height_resource

        return last_y

```

File: `ui/panels/left_panel.py`
Explanation: Removed passing of `is_process_running`, `update_progress_details`, `worker_counts` to sub-renderers where they are no longer needed. Adapted plot rendering condition.
```python
# File: ui/panels/left_panel.py
import pygame
from typing import Dict, Any, Optional, Deque

from config import (
    VisConfig,
    RNNConfig,
    TransformerConfig,
    ModelConfig,
    # Removed TOTAL_TRAINING_STEPS
)
from config.general import DEVICE

from ui.plotter import Plotter
from ui.input_handler import InputHandler
from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    TBStatusRenderer,
    PlotAreaRenderer,
)
from app_state import AppState


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.input_handler: Optional[InputHandler] = None

        self.button_status_renderer = ButtonStatusRenderer(self.screen, self.fonts)
        self.info_text_renderer = InfoTextRenderer(self.screen, self.fonts)
        self.tb_status_renderer = TBStatusRenderer(self.screen, self.fonts)
        self.plot_area_renderer = PlotAreaRenderer(
            self.screen, self.fonts, self.plotter
        )
        self.rnn_config = RNNConfig()
        self.transformer_config = TransformerConfig()
        self.model_config_net = ModelConfig.Network()

    def _init_fonts(self):
        """Initializes fonts used in the left panel."""
        fonts = {}
        font_configs = {
            "ui": 24,
            "status": 28,
            "logdir": 16,
            "plot_placeholder": 20,
            "notification_label": 16,
            "plot_title_values": 8,
            "progress_bar": 14,
            "notification": 18,
        }
        for key, size in font_configs.items():
            try:
                fonts[key] = pygame.font.SysFont(None, size)
            except Exception:
                fonts[key] = pygame.font.Font(None, size)
            if fonts[key] is None:
                print(f"ERROR: Font '{key}' failed to load.")
        return fonts

    def render(
        self,
        panel_width: int,
        is_process_running: bool, # Keep for potential future use (MCTS/NN running)
        status: str,
        stats_summary: Dict[str, Any],
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        app_state: str,
        update_progress_details: Dict[str, Any], # Keep for potential NN progress
        agent_param_count: int,
        worker_counts: Dict[str, int], # Keep structure, content will change
    ):
        """Renders the entire left panel within the given width."""
        current_height = self.screen.get_height()
        lp_rect = pygame.Rect(0, 0, panel_width, current_height)

        # Simplified status mapping
        status_color_map = {
            "Ready": (30, 30, 30),
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Debugging Grid": (40, 30, 40),
            "Initializing": (40, 40, 40),
            # Add potential future states
            "Running MCTS": (30, 40, 30),
            "Training NN": (30, 30, 50),
        }
        base_status = status.split(" (")[0] if "(" in status else status
        bg_color = status_color_map.get(base_status, (30, 30, 30))

        pygame.draw.rect(self.screen, bg_color, lp_rect)
        current_y = 10

        # Render Buttons and Status (simplified)
        next_y = self.button_status_renderer.render(
            y_start=current_y,
            panel_width=panel_width,
            app_state=app_state,
            is_process_running=is_process_running, # Pass for potential future use
            status=status,
            stats_summary=stats_summary,
            update_progress_details=update_progress_details, # Pass for potential NN progress
        )
        current_y = next_y

        # Render Info Text (simplified)
        next_y = self.info_text_renderer.render(
            current_y + 5,
            stats_summary,
            panel_width,
            agent_param_count,
            worker_counts, # Pass potentially adapted worker counts
        )
        current_y = next_y

        # Render TB Status (unchanged)
        next_y = self.tb_status_renderer.render(
            current_y + 10, tensorboard_log_dir, panel_width
        )
        current_y = next_y

        # Render Plots (unchanged condition, but plots themselves are simplified)
        if app_state == AppState.MAIN_MENU.value:
            self.plot_area_renderer.render(
                y_start=current_y + 5,
                panel_width=panel_width,
                screen_height=current_height,
                plot_data=plot_data,
                status=status, # Pass status for placeholder text
            )

```

File: `ui/plotter.py`
Explanation: Removed plots for PPO losses, entropy, SPS, LR. Kept game score, RL score (if used), episode length, tris cleared, resource usage. Added placeholder for NN losses.
```python
# File: ui/plotter.py
import pygame
from typing import Dict, Optional, Deque
from collections import deque
import matplotlib
import time
import warnings
from io import BytesIO
import traceback

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, StatsConfig
from .plot_utils import (
    render_single_plot,
    normalize_color_for_matplotlib,
)


class Plotter:
    """Handles creation and caching of the multi-plot Matplotlib surface."""

    def __init__(self):
        self.plot_surface: Optional[pygame.Surface] = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = 0.2
        self.rolling_window_sizes = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW

        self.colors = {
            "rl_score": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[0]),
            "game_score": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[1]),
            "policy_loss": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[3]), # Keep for NN policy loss
            "value_loss": normalize_color_for_matplotlib(VisConfig.BLUE), # Keep for NN value loss
            # Removed entropy
            "len": normalize_color_for_matplotlib(VisConfig.BLUE),
            # Removed minibatch_sps
            "tris_cleared": normalize_color_for_matplotlib(VisConfig.YELLOW),
            # Removed lr (can add back if needed for NN)
            "cpu": normalize_color_for_matplotlib((255, 165, 0)),
            "memory": normalize_color_for_matplotlib((0, 191, 255)),
            "gpu_mem": normalize_color_for_matplotlib((123, 104, 238)),
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
        }

    def create_plot_surface(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:

        if target_width <= 10 or target_height <= 10 or not plot_data:
            return None

        # Updated data keys
        data_keys = [
            "game_scores",
            "episode_triangles_cleared",
            "episode_scores", # Keep RL score? Or remove? Keep for now.
            "episode_lengths",
            "policy_losses", # Added NN policy loss
            "value_losses", # Kept NN value loss
            # Removed: minibatch_update_sps_values, lr_values, entropies
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            # Add placeholders for other potential plots
            "placeholder1", "placeholder2", "placeholder3"
        ]
        data_lists = {key: list(plot_data.get(key, deque())) for key in data_keys}

        has_any_data = any(len(d) > 0 for d in data_lists.values())
        if not has_any_data:
            return None

        fig = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                dpi = 90
                fig_width_in = max(1, target_width / dpi)
                fig_height_in = max(1, target_height / dpi)

                # Keep 4x3 layout for now, fill unused plots with placeholders
                fig, axes = plt.subplots(4, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False)
                fig.subplots_adjust(hspace=0.18, wspace=0.10, left=0.08, right=0.98, bottom=0.05, top=0.95)
                axes_flat = axes.flatten()

                # Row 1: Performance Metrics
                render_single_plot(axes_flat[0], data_lists["game_scores"], "Game Score", self.colors["game_score"], self.rolling_window_sizes, placeholder_text="Game Score")
                render_single_plot(axes_flat[1], data_lists["episode_triangles_cleared"], "Tris Cleared / Ep", self.colors["tris_cleared"], self.rolling_window_sizes, placeholder_text="Triangles Cleared")
                render_single_plot(axes_flat[2], data_lists["episode_scores"], "RL Score", self.colors["rl_score"], self.rolling_window_sizes, placeholder_text="RL Score")

                # Row 2: Training Dynamics / NN Losses
                render_single_plot(axes_flat[3], data_lists["episode_lengths"], "Ep Length", self.colors["len"], self.rolling_window_sizes, placeholder_text="Episode Length")
                render_single_plot(axes_flat[4], data_lists["policy_losses"], "Policy Loss", self.colors["policy_loss"], self.rolling_window_sizes, placeholder_text="Policy Loss") # NN Policy Loss
                render_single_plot(axes_flat[5], data_lists["value_losses"], "Value Loss", self.colors["value_loss"], self.rolling_window_sizes, placeholder_text="Value Loss") # NN Value Loss

                # Row 3: Resource Usage
                render_single_plot(axes_flat[6], data_lists["cpu_usage"], "CPU Usage (%)", self.colors["cpu"], self.rolling_window_sizes, placeholder_text="CPU %")
                render_single_plot(axes_flat[7], data_lists["memory_usage"], "Memory Usage (%)", self.colors["memory"], self.rolling_window_sizes, placeholder_text="Mem %")
                render_single_plot(axes_flat[8], data_lists["gpu_memory_usage_percent"], "GPU Memory (%)", self.colors["gpu_mem"], self.rolling_window_sizes, placeholder_text="GPU Mem %")

                # Row 4: Placeholders / Future Plots
                render_single_plot(axes_flat[9], data_lists["placeholder1"], "Future Plot 1", self.colors["placeholder"], [], placeholder_text="Future Plot 1")
                render_single_plot(axes_flat[10], data_lists["placeholder2"], "Future Plot 2", self.colors["placeholder"], [], placeholder_text="Future Plot 2")
                render_single_plot(axes_flat[11], data_lists["placeholder3"], "Future Plot 3", self.colors["placeholder"], [], placeholder_text="Future Plot 3")


                # Remove x-axis labels/ticks for inner plots
                for i, ax in enumerate(axes_flat):
                    if i < 9: # Adjust based on layout (4x3 -> 9 inner plots)
                        ax.set_xticklabels([])
                        ax.set_xlabel("")
                    ax.tick_params(axis="x", rotation=0)

                buf = BytesIO()
                fig.savefig(buf, format="png", transparent=False, facecolor=plt.rcParams["figure.facecolor"])
                buf.seek(0)
                plot_img_surface = pygame.image.load(buf).convert()
                buf.close()

                current_size = plot_img_surface.get_size()
                if current_size != (target_width, target_height):
                    plot_img_surface = pygame.transform.smoothscale(plot_img_surface, (target_width, target_height))

                return plot_img_surface

        except Exception as e:
            print(f"Error creating plot surface: {e}")
            traceback.print_exc()
            return None
        finally:
            if fig is not None: plt.close(fig)

    def get_cached_or_updated_plot(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        current_time = time.time()
        has_data = any(d for d in plot_data.values())
        needs_update_time = current_time - self.last_plot_update_time > self.plot_update_interval
        size_changed = self.plot_surface and self.plot_surface.get_size() != (target_width, target_height)
        first_plot_needed = has_data and self.plot_surface is None
        can_create_plot = target_width > 50 and target_height > 50

        if can_create_plot and (needs_update_time or size_changed or first_plot_needed):
            if has_data:
                new_plot_surface = self.create_plot_surface(plot_data, target_width, target_height)
                if new_plot_surface: self.plot_surface = new_plot_surface
                self.last_plot_update_time = current_time
            elif not has_data:
                self.plot_surface = None

        return self.plot_surface

```

File: `ui/renderer.py`
Explanation: Removed passing of `is_process_running`, `update_progress_details`, `worker_counts` where they are no longer used by sub-renderers.
```python
# File: ui/renderer.py
import pygame
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque

from config import VisConfig, EnvConfig, DemoConfig
from environment.game_state import GameState
from .panels import LeftPanelRenderer, GameAreaRenderer
from .overlays import OverlayRenderer
from .plotter import Plotter
from .demo_renderer import DemoRenderer
from .input_handler import InputHandler
from app_state import AppState


class UIRenderer:
    """Orchestrates rendering of all UI components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = Plotter()
        self.left_panel = LeftPanelRenderer(screen, vis_config, self.plotter)
        self.game_area = GameAreaRenderer(screen, vis_config)
        self.overlays = OverlayRenderer(screen, vis_config)
        self.demo_config = DemoConfig()
        self.demo_renderer = DemoRenderer(screen, vis_config, self.demo_config, self.game_area)
        self.last_plot_update_time = 0

    def set_input_handler(self, input_handler: InputHandler):
        """Sets the InputHandler reference after it's initialized."""
        self.left_panel.input_handler = input_handler
        if hasattr(self.left_panel, "button_status_renderer"):
            self.left_panel.button_status_renderer.input_handler_ref = input_handler

    def check_hover(self, mouse_pos: Tuple[int, int], app_state_str: str):
        """Placeholder for hover checks if needed later."""
        pass

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0

    def render_all(
        self,
        app_state: str,
        is_process_running: bool, # Keep for potential MCTS/NN status
        status: str,
        stats_summary: Dict[str, Any],
        envs: List[GameState], # Keep for visualization
        num_envs: int, # Keep for visualization layout
        env_config: EnvConfig,
        cleanup_confirmation_active: bool,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        demo_env: Optional[GameState] = None,
        update_progress_details: Dict[str, Any] = {}, # Keep for potential NN progress
        agent_param_count: int = 0, # Keep for NN param count
        worker_counts: Dict[str, int] = {}, # Keep structure for potential future workers
    ):
        """Renders UI based on the application state."""
        try:
            current_app_state = AppState(app_state) if app_state in AppState._value2member_map_ else AppState.UNKNOWN

            if current_app_state == AppState.MAIN_MENU:
                self._render_main_menu(
                    is_process_running=is_process_running, # Pass to left panel
                    status=status,
                    stats_summary=stats_summary,
                    envs=envs,
                    num_envs=num_envs,
                    env_config=env_config,
                    cleanup_message=cleanup_message,
                    last_cleanup_message_time=last_cleanup_message_time,
                    tensorboard_log_dir=tensorboard_log_dir,
                    plot_data=plot_data,
                    update_progress_details=update_progress_details, # Pass to left panel
                    app_state=app_state,
                    agent_param_count=agent_param_count,
                    worker_counts=worker_counts, # Pass to left panel
                )
            elif current_app_state == AppState.PLAYING:
                if demo_env: self.demo_renderer.render(demo_env, env_config, is_debug=False)
                else: self._render_simple_message("Demo Env Error!", VisConfig.RED)
            elif current_app_state == AppState.DEBUG:
                if demo_env: self.demo_renderer.render(demo_env, env_config, is_debug=True)
                else: self._render_simple_message("Debug Env Error!", VisConfig.RED)
            elif current_app_state == AppState.INITIALIZING:
                self._render_initializing_screen(status)
            elif current_app_state == AppState.ERROR:
                self._render_error_screen(status)

            if cleanup_confirmation_active and current_app_state != AppState.ERROR:
                self.overlays.render_cleanup_confirmation()
            elif not cleanup_confirmation_active:
                self.overlays.render_status_message(cleanup_message, last_cleanup_message_time)

            pygame.display.flip()

        except pygame.error as e:
            print(f"Pygame rendering error in render_all: {e}")
        except Exception as e:
            print(f"Unexpected critical rendering error in render_all: {e}")
            traceback.print_exc()
            try:
                self._render_simple_message("Critical Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception: pass

    def _render_main_menu(
        self,
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        update_progress_details: Dict[str, Any],
        app_state: str,
        agent_param_count: int,
        worker_counts: Dict[str, int],
    ):
        """Renders the main dashboard view."""
        self.screen.fill(VisConfig.BLACK)
        current_width, current_height = self.screen.get_size()
        left_panel_ratio = max(0.1, min(0.9, self.vis_config.LEFT_PANEL_RATIO))
        lp_width = int(current_width * left_panel_ratio)
        ga_width = current_width - lp_width
        min_lp_width = 300
        if lp_width < min_lp_width and current_width > min_lp_width:
            lp_width = min_lp_width
            ga_width = max(0, current_width - lp_width)
        elif current_width <= min_lp_width:
            lp_width = current_width
            ga_width = 0

        self.left_panel.render(
            panel_width=lp_width,
            is_process_running=is_process_running, # Pass for potential future use
            status=status,
            stats_summary=stats_summary,
            tensorboard_log_dir=tensorboard_log_dir,
            plot_data=plot_data,
            app_state=app_state,
            update_progress_details=update_progress_details, # Pass for potential NN progress
            agent_param_count=agent_param_count,
            worker_counts=worker_counts, # Pass for potential future use
        )
        # Render game area only if width is sufficient
        if ga_width > 0:
            self.game_area.render(
                envs=envs, # Pass envs for visualization
                num_envs=num_envs, # Pass num_envs for layout
                env_config=env_config,
                panel_width=ga_width,
                panel_x_offset=lp_width,
            )

    def _render_initializing_screen(self, status_message: str = "Initializing..."):
        self._render_simple_message(status_message, VisConfig.WHITE)

    def _render_error_screen(self, status_message: str):
        try:
            self.screen.fill((40, 0, 0))
            font_title = pygame.font.SysFont(None, 70)
            font_msg = pygame.font.SysFont(None, 30)
            title_surf = font_title.render("APPLICATION ERROR", True, VisConfig.RED)
            title_rect = title_surf.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 3))
            msg_surf = font_msg.render(f"Status: {status_message}", True, VisConfig.YELLOW)
            msg_rect = msg_surf.get_rect(center=(self.screen.get_width() // 2, title_rect.bottom + 30))
            exit_surf = font_msg.render("Press ESC or close window to exit.", True, VisConfig.WHITE)
            exit_rect = exit_surf.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() * 0.8))
            self.screen.blit(title_surf, title_rect)
            self.screen.blit(msg_surf, msg_rect)
            self.screen.blit(exit_surf, exit_rect)
        except Exception as e:
            print(f"Error rendering error screen: {e}")
            self._render_simple_message(f"Error State: {status_message}", VisConfig.RED)

    def _render_simple_message(self, message: str, color: Tuple[int, int, int]):
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)
            text_surf = font.render(message, True, color)
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
        except Exception as e:
            print(f"Error rendering simple message '{message}': {e}")

```

File: `utils/init_checks.py`
Explanation: Removed PBRS check as RewardConfig was removed/simplified.
```python
# File: utils/init_checks.py
import sys
import traceback
import numpy as np

from environment.game_state import GameState


def run_pre_checks() -> bool:
    """Performs basic checks on GameState and configuration compatibility."""
    try:
        from config import EnvConfig
    except ImportError as e:
        print(f"FATAL ERROR: Could not import EnvConfig during pre-check: {e}")
        print("This might indicate an issue with the config package structure or an ongoing import cycle.")
        sys.exit(1)

    print("--- Pre-Run Checks ---")
    try:
        print("Checking GameState and Configuration Compatibility...")
        env_config_instance = EnvConfig()

        gs_test = GameState()
        gs_test.reset()
        s_test_dict = gs_test.get_state()

        if not isinstance(s_test_dict, dict):
            raise TypeError(f"GameState.get_state() should return a dict, but got {type(s_test_dict)}")
        print("GameState state type check PASSED (returned dict).")

        if "grid" not in s_test_dict: raise KeyError("State dictionary missing 'grid' key.")
        grid_state = s_test_dict["grid"]
        expected_grid_shape = env_config_instance.GRID_STATE_SHAPE
        if not isinstance(grid_state, np.ndarray): raise TypeError(f"State 'grid' component should be numpy array, but got {type(grid_state)}")
        if grid_state.shape != expected_grid_shape: raise ValueError(f"State 'grid' shape mismatch! GameState:{grid_state.shape}, EnvConfig:{expected_grid_shape}")
        print(f"GameState 'grid' state shape check PASSED (Shape: {grid_state.shape}).")

        if "shapes" not in s_test_dict: raise KeyError("State dictionary missing 'shapes' key.")
        shape_state = s_test_dict["shapes"]
        expected_shape_shape = (env_config_instance.SHAPE_STATE_DIM,)
        if not isinstance(shape_state, np.ndarray): raise TypeError(f"State 'shapes' component should be numpy array, but got {type(shape_state)}")
        if shape_state.shape != expected_shape_shape: raise ValueError(f"State 'shapes' feature shape mismatch! GameState:{shape_state.shape}, EnvConfig:{expected_shape_shape}")
        print(f"GameState 'shapes' feature shape check PASSED (Shape: {shape_state.shape}).")

        if "shape_availability" not in s_test_dict: raise KeyError("State dictionary missing 'shape_availability' key.")
        availability_state = s_test_dict["shape_availability"]
        expected_availability_shape = (env_config_instance.SHAPE_AVAILABILITY_DIM,)
        if not isinstance(availability_state, np.ndarray): raise TypeError(f"State 'shape_availability' component should be numpy array, but got {type(availability_state)}")
        if availability_state.shape != expected_availability_shape: raise ValueError(f"State 'shape_availability' shape mismatch! GameState:{availability_state.shape}, EnvConfig:{expected_availability_shape}")
        print(f"GameState 'shape_availability' state shape check PASSED (Shape: {availability_state.shape}).")

        if "explicit_features" not in s_test_dict: raise KeyError("State dictionary missing 'explicit_features' key.")
        explicit_features_state = s_test_dict["explicit_features"]
        expected_explicit_features_shape = (env_config_instance.EXPLICIT_FEATURES_DIM,)
        if not isinstance(explicit_features_state, np.ndarray): raise TypeError(f"State 'explicit_features' component should be numpy array, but got {type(explicit_features_state)}")
        if explicit_features_state.shape != expected_explicit_features_shape: raise ValueError(f"State 'explicit_features' shape mismatch! GameState:{explicit_features_state.shape}, EnvConfig:{expected_explicit_features_shape}")
        print(f"GameState 'explicit_features' state shape check PASSED (Shape: {explicit_features_state.shape}).")

        if env_config_instance.CALCULATE_POTENTIAL_OUTCOMES_IN_STATE:
            print("Potential outcome calculation is ENABLED in EnvConfig.")
        else:
            print("Potential outcome calculation is DISABLED in EnvConfig.")

        # Removed PBRS check

        _ = gs_test.valid_actions()
        print("GameState valid_actions check PASSED.")
        if not hasattr(gs_test, "game_score"): raise AttributeError("GameState missing 'game_score' attribute!")
        print("GameState 'game_score' attribute check PASSED.")
        if not hasattr(gs_test, "triangles_cleared_this_episode"): raise AttributeError("GameState missing 'triangles_cleared_this_episode' attribute!")
        print("GameState 'triangles_cleared_this_episode' attribute check PASSED.")

        del gs_test
        print("--- Pre-Run Checks Complete ---")
        return True
    except (NameError, ImportError) as e:
        print(f"FATAL ERROR: Import/Name error during pre-check: {e}")
    except (ValueError, AttributeError, TypeError, KeyError) as e:
        print(f"FATAL ERROR during pre-run checks: {e}")
    except Exception as e:
        print(f"FATAL ERROR during GameState pre-check: {e}")
        traceback.print_exc()
    sys.exit(1)

```

File: `utils/__init__.py`
Explanation: No changes needed here based on other modifications.
```python
# File: utils/__init__.py
from .helpers import (
    get_device,
    set_random_seeds,
    ensure_numpy,
    save_object,
    load_object,
    format_eta, # Added format_eta
)
from .init_checks import run_pre_checks
from .types import StateType, ActionType, AgentStateDict
from .running_mean_std import RunningMeanStd


__all__ = [
    "get_device",
    "set_random_seeds",
    "ensure_numpy",
    "save_object",
    "load_object",
    "format_eta", # Added format_eta
    "run_pre_checks",
    "StateType",
    "ActionType",
    "AgentStateDict",
    "RunningMeanStd",
]

```

File: `requirements.txt`
Explanation: Removed `cloudpickle` as checkpointing might change, but kept other core dependencies.
```python
pygame>=2.1.0
numpy>=1.20.0
torch>=1.10.0
tensorboard
matplotlib
psutil
# cloudpickle # Removed, can be added back if complex object saving is needed
```