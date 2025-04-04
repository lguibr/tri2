Okay, let's refactor the code to remove `wandb`, integrate TensorBoard for local logging, and clean up the codebase.

**Summary of Changes:**

1.  **Removed `wandb`:** All `wandb` imports, configurations, initialization, logging calls, and UI elements are removed.
2.  **Integrated TensorBoard:**
    *   Added `tensorboard` to `requirements.txt`.
    *   Created `stats/tensorboard_logger.py` with `TensorBoardStatsRecorder`.
    *   Modified `main_pygame.py` to initialize and use `TensorBoardStatsRecorder`. It now generates a unique log directory for each run based on a timestamp.
    *   Modified `config.py` to add a basic `TensorBoardConfig`.
    *   Updated `ui/renderer.py` to display the TensorBoard log directory instead of the WandB status/link.
3.  **Code Cleanup:**
    *   Removed verbose development comments (`<<< NEW >>>`, `(Same as before)`, etc.).
    *   Removed redundant comments explaining obvious code.
    *   Kept comments explaining configuration choices or complex logic.
    *   Simplified configuration checks and print statements in `config.py`.
    *   Removed unused `.resumed.txt` file.
    *   Slightly streamlined config class structure in `config.py`.

**Refactored Code:**

```python
# File: config.py
import torch
import os
import time
from utils.helpers import get_device
from typing import Deque

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
    NUM_ENVS_TO_RENDER = 16

# --- Environment ---
class EnvConfig:
    NUM_ENVS = 1024
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
    USE_NOISY_NETS = True # Hardcoded: Always use Noisy Nets
    USE_DOUBLE_DQN = True
    USE_DUELING = True

# --- Training Loop ---
class TrainConfig:
    BATCH_SIZE = 64
    LEARN_START_STEP = 200_000
    TOTAL_TRAINING_STEPS = 100_000_000
    LEARN_FREQ = 8
    CHECKPOINT_SAVE_FREQ = 1_000_000
    LOAD_CHECKPOINT_PATH: str | None = None # Specify path to load a specific checkpoint, e.g., "checkpoints/run_xxxxxxxx_xxxxxx/dqn_agent_state.pth"
    LOAD_BUFFER_PATH: str | None = None    # Specify path to load a specific buffer, e.g., "checkpoints/run_xxxxxxxx_xxxxxx/replay_buffer_state.pkl"

# --- Replay Buffer ---
class BufferConfig:
    REPLAY_BUFFER_SIZE = 2_000_000
    USE_N_STEP = True
    N_STEP = 15
    USE_PER = True
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 25_000_000
    PER_EPSILON = 1e-6
    # LOAD_BUFFER handled by TrainConfig.LOAD_BUFFER_PATH

# --- Model Architecture ---
class ModelConfig:
    # SAVE_PATH handled by top-level MODEL_SAVE_PATH
    # LOAD_MODEL handled by TrainConfig.LOAD_CHECKPOINT_PATH
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
    STATS_AVG_WINDOW = 500  # Window for calculating rolling averages (in-memory UI stats)
    CONSOLE_LOG_FREQ = 50_000 # How often to print stats summary to console

# --- Config Consistency Checks & Info ---
print("-" * 70)
print(f"RUN ID: {RUN_ID}")
print(f"Log Directory: {TensorBoardConfig.LOG_DIR}")
print(f"Checkpoint Directory: {os.path.dirname(MODEL_SAVE_PATH)}")
print(f"Device: {DEVICE}")

if EnvConfig.GRID_FEATURES_PER_CELL != 3:
    print("Warning: Network assumes 3 features per cell (Occupied, Is_Up, Is_Death). Check EnvConfig/GameState.")

if TrainConfig.LOAD_CHECKPOINT_PATH:
    print("*" * 70)
    print(f"*** Warning: Attempting to LOAD CHECKPOINT from: {TrainConfig.LOAD_CHECKPOINT_PATH} ***")
    print("*** Ensure saved checkpoint matches current ModelConfig. ***")
    print("*" * 70)
else:
    print("--- Starting training from scratch (no checkpoint specified to load). ---")

if TrainConfig.LOAD_BUFFER_PATH:
    print("*" * 70)
    print(f"*** Warning: Attempting to LOAD BUFFER from: {TrainConfig.LOAD_BUFFER_PATH} ***")
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
        print("*** Using MPS device. Performance varies. Force CPU via env var if needed. ***")
    print("*" * 70)

print(f"--- Rendering {VisConfig.NUM_ENVS_TO_RENDER if VisConfig.NUM_ENVS_TO_RENDER > 0 else 'ALL'} of {EnvConfig.NUM_ENVS} environments ---")
print("-" * 70)
```

```python
# File: requirements.txt
pygame>=2.1.0
numpy>=1.20.0
torch>=1.10.0 # Or specify version compatible with your CUDA/MPS if needed
tensorboard
cloudpickle
```

```python
# File: main_pygame.py
import sys
import pygame
import numpy as np
import os
import time
import traceback
from typing import List, Tuple, Optional, Dict, Any

# Import configurations
from config import (
    VisConfig,
    EnvConfig,
    DQNConfig,
    TrainConfig,
    BufferConfig,
    ModelConfig,
    StatsConfig,
    RewardConfig,
    TensorBoardConfig, # Added TensorBoardConfig
    DEVICE,
    RANDOM_SEED,
    BUFFER_SAVE_PATH, # Default save path for current run
    MODEL_SAVE_PATH,   # Default save path for current run
)

# Import core components
try:
    from environment.game_state import GameState
except ImportError as e:
    print(f"Error importing environment: {e}")
    sys.exit(1)
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from agent.replay_buffer.buffer_utils import create_replay_buffer
from training.trainer import Trainer
from stats.stats_recorder import StatsRecorderBase # Base class
from stats.simple_stats_recorder import SimpleStatsRecorder # Keep for UI display
from stats.tensorboard_logger import TensorBoardStatsRecorder # Use TensorBoard
from ui.renderer import UIRenderer
from utils.helpers import set_random_seeds, ensure_numpy

class MainApp:
    def __init__(self):
        print("Initializing Pygame Application...")
        set_random_seeds(RANDOM_SEED)
        pygame.init()
        pygame.font.init()

        # Store configs
        self.vis_config = VisConfig
        self.env_config = EnvConfig
        self.reward_config = RewardConfig
        self.dqn_config = DQNConfig
        self.train_config = TrainConfig
        self.buffer_config = BufferConfig
        self.model_config = ModelConfig
        self.stats_config = StatsConfig
        self.tensorboard_config = TensorBoardConfig
        self.num_envs = self.env_config.NUM_ENVS

        # --- Ensure log/checkpoint directories exist for this run ---
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(BUFFER_SAVE_PATH), exist_ok=True)
        os.makedirs(self.tensorboard_config.LOG_DIR, exist_ok=True)
        print(f"TensorBoard logs will be saved to: {self.tensorboard_config.LOG_DIR}")
        print(f"Checkpoints/Buffer will be saved to: {os.path.dirname(MODEL_SAVE_PATH)}")

        # Pygame setup
        self.screen = pygame.display.set_mode(
            (self.vis_config.SCREEN_WIDTH, self.vis_config.SCREEN_HEIGHT),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption("TriCrack DQN - TensorBoard")
        self.clock = pygame.time.Clock()

        # App state
        self.is_training = False
        self.cleanup_confirmation_active = False
        self.last_cleanup_message_time = 0.0
        self.cleanup_message = ""
        self.status = "Paused"

        # Init RL components
        print("Initializing RL Components...")
        self._initialize_rl_components()

        # Init Renderer
        self.renderer = UIRenderer(self.screen, self.vis_config)

        print("Initialization Complete. Ready to start.")
        print("--- To view logs, run in terminal: ---")
        print(f"--- tensorboard --logdir {os.path.abspath(BASE_LOG_DIR)} ---")


    def _initialize_envs(self) -> List[GameState]:
        print(f"Initializing {self.num_envs} game environments...")
        try:
            envs = [GameState() for _ in range(self.num_envs)]
            s_test = envs[0].reset()
            s_np = ensure_numpy(s_test)
            if s_np.shape[0] != self.env_config.STATE_DIM:
                raise ValueError(
                    f"FATAL: State dim mismatch! Env:{s_np.shape[0]}, Cfg:{self.env_config.STATE_DIM}"
                )
            _ = envs[0].valid_actions()
            _, _ = envs[0].step(0)
            print(f"Successfully initialized {self.num_envs} environments.")
            return envs
        except Exception as e:
            print(f"FATAL ERROR during env init: {e}")
            traceback.print_exc()
            pygame.quit()
            sys.exit(1)

    def _initialize_stats_recorder(self) -> StatsRecorderBase:
        """Creates the TensorBoard statistics recorder."""
        # Close previous recorder if exists
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"Warn: Error closing prev stats recorder: {e}")

        avg_window = self.stats_config.STATS_AVG_WINDOW
        console_log_freq = self.stats_config.CONSOLE_LOG_FREQ

        print(f"Using TensorBoard Logger (Log Dir: {self.tensorboard_config.LOG_DIR})")
        try:
            # TensorBoard logger handles its own logging frequency via SummaryWriter buffering
            # We also pass the console log freq for its own printing
            return TensorBoardStatsRecorder(
                log_dir=self.tensorboard_config.LOG_DIR,
                console_log_interval=console_log_freq,
                avg_window=avg_window,
            )
        except Exception as e:
            print(f"FATAL: Error initializing TensorBoardStatsRecorder: {e}. Exiting.")
            traceback.print_exc()
            pygame.quit()
            sys.exit(1)
            # Fallback to simple recorder if needed, but TensorBoard is preferred
            # print("Falling back to SimpleStatsRecorder.")
            # return SimpleStatsRecorder(
            #     console_log_interval=console_log_freq,
            #     avg_window=avg_window
            # )

    def _initialize_rl_components(self):
        print("Initializing/Re-initializing RL components...")
        self.envs: List[GameState] = self._initialize_envs()
        self.agent: DQNAgent = DQNAgent(
            config=self.model_config,
            dqn_config=self.dqn_config,
            env_config=self.env_config,
        )
        self.buffer: ReplayBufferBase = create_replay_buffer(
            config=self.buffer_config,
            dqn_config=self.dqn_config,
        )
        # Note: stats_recorder now becomes TensorBoardStatsRecorder
        self.stats_recorder: StatsRecorderBase = self._initialize_stats_recorder()
        self.trainer: Trainer = Trainer(
            envs=self.envs,
            agent=self.agent,
            buffer=self.buffer,
            stats_recorder=self.stats_recorder,
            env_config=self.env_config,
            dqn_config=self.dqn_config,
            train_config=self.train_config,
            buffer_config=self.buffer_config,
            model_config=self.model_config,
            model_save_path=MODEL_SAVE_PATH,      # Pass specific save path for this run
            buffer_save_path=BUFFER_SAVE_PATH,     # Pass specific save path for this run
            load_checkpoint_path=self.train_config.LOAD_CHECKPOINT_PATH, # Optional path to load from
            load_buffer_path=self.train_config.LOAD_BUFFER_PATH # Optional path to load from
        )
        print("RL components initialization finished.")

    def _cleanup_data(self):
        """Stops training, deletes checkpoints, buffer FOR THE CURRENT RUN, and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.is_training = False
        self.status = "Cleaning"
        self.cleanup_confirmation_active = False
        messages = []

        # 1. Trainer Cleanup (saves final state if desired, flushes buffer)
        if hasattr(self, "trainer") and self.trainer:
            print("Running trainer cleanup (saving final state)...")
            try:
                # Force save=True during cleanup? Or rely on standard save logic? Let's save.
                self.trainer.cleanup(save_final=True)
            except Exception as e:
                print(f"Error during trainer cleanup: {e}")
        else:
            print("Trainer object not found, skipping trainer cleanup.")

        # 2. Delete CURRENT Agent Checkpoint
        ckpt_path = MODEL_SAVE_PATH
        try:
            if os.path.isfile(ckpt_path):
                os.remove(ckpt_path)
                messages.append(f"Agent ckpt deleted: {os.path.basename(ckpt_path)}")
            else:
                messages.append("Agent ckpt not found (current run).")
        except OSError as e:
            messages.append(f"Error deleting agent ckpt: {e}")

        # 3. Delete CURRENT Buffer State
        buffer_path = BUFFER_SAVE_PATH
        try:
            if os.path.isfile(buffer_path):
                os.remove(buffer_path)
                messages.append(f"Buffer state deleted: {os.path.basename(buffer_path)}")
            else:
                messages.append("Buffer state not found (current run).")
        except OSError as e:
            messages.append(f"Error deleting buffer: {e}")

        # 4. Close current TensorBoard writer (but don't delete logs)
        if hasattr(self.stats_recorder, 'close'):
            try:
                self.stats_recorder.close()
                messages.append("TensorBoard writer closed.")
            except Exception as e:
                messages.append(f"Error closing TB writer: {e}")

        # 5. Re-initialize RL components
        print("Re-initializing RL components after cleanup...")
        # Re-init will create new logger instance for the *same* run ID - maybe generate NEW run ID?
        # Let's stick to re-initializing within the same run structure for simplicity.
        # The user can manually delete the run folder if they want a truly fresh start.
        self._initialize_rl_components()

        # Re-initialize renderer
        self.renderer = UIRenderer(self.screen, self.vis_config)

        self.cleanup_message = "\n".join(messages)
        self.last_cleanup_message_time = time.time()
        self.status = "Paused"
        print("--- CLEANUP DATA COMPLETE (Current Run Checkpoints/Buffer Removed) ---")

    def _handle_input(self) -> bool:
        mouse_pos = pygame.mouse.get_pos()
        sw, sh = self.screen.get_size()
        train_btn_rect = pygame.Rect(10, 10, 100, 40)
        cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 160, 40) # Wider button
        confirm_yes_rect = pygame.Rect(sw // 2 - 110, sh // 2 + 30, 100, 40)
        confirm_no_rect = pygame.Rect(sw // 2 + 10, sh // 2 + 30, 100, 40)
        self.renderer.check_hover(mouse_pos) # Check for tooltip hover

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                try:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self.renderer.screen = self.screen
                    print(f"Window resized: {event.w}x{event.h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.cleanup_confirmation_active:
                        self.cleanup_confirmation_active = False
                        self.cleanup_message = "Cleanup cancelled."
                        self.last_cleanup_message_time = time.time()
                    else:
                        return False # Exit app on ESC if not confirming
                elif event.key == pygame.K_p and not self.cleanup_confirmation_active:
                    self.is_training = not self.is_training
                    print(f"Training {'STARTED' if self.is_training else 'PAUSED'} (P key)")
                    self._try_save_checkpoint() # Save on pause

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
                if self.cleanup_confirmation_active:
                    if confirm_yes_rect.collidepoint(mouse_pos):
                        self._cleanup_data()
                    elif confirm_no_rect.collidepoint(mouse_pos):
                        self.cleanup_confirmation_active = False
                        self.cleanup_message = "Cleanup cancelled."
                        self.last_cleanup_message_time = time.time()
                else:
                    if train_btn_rect.collidepoint(mouse_pos):
                        self.is_training = not self.is_training
                        print(f"Training {'STARTED' if self.is_training else 'PAUSED'} (Button)")
                        self._try_save_checkpoint() # Save on pause
                    elif cleanup_btn_rect.collidepoint(mouse_pos):
                        self.is_training = False # Pause training before showing confirmation
                        self.cleanup_confirmation_active = True
                        print("Cleanup requested.")
        return True

    def _try_save_checkpoint(self):
        """Saves checkpoint if trainer exists and has the method."""
        if not self.is_training and hasattr(self.trainer, "_save_checkpoint"):
            print("Saving checkpoint on pause...")
            try:
                # Pass is_final=False, maybe add logic for 'reason' if needed
                self.trainer._save_checkpoint(is_final=False)
            except Exception as e:
                print(f"Error saving checkpoint on pause: {e}")

    def _update(self):
        """Performs training step and updates status."""
        if self.is_training:
            if self.trainer.global_step < self.train_config.LEARN_START_STEP:
                self.status = "Buffering"
            else:
                self.status = "Training"
        elif self.cleanup_confirmation_active:
             self.status = "Confirm Cleanup"
        else:
             self.status = "Paused" # Handles initial state and explicit pauses

        if not self.is_training: # Covers Paused, Confirm Cleanup, Cleaning, Error
            return # Don't step trainer if not actively training

        try:
            step_start_time = time.time()
            # Trainer calls stats_recorder internally (which is now TensorBoardStatsRecorder)
            self.trainer.step()
            step_duration = time.time() - step_start_time

            # Optional delay
            if self.vis_config.VISUAL_STEP_DELAY > 0:
                time.sleep(max(0, self.vis_config.VISUAL_STEP_DELAY - step_duration))

        except Exception as e:
            print(f"\n--- ERROR DURING TRAINING UPDATE (Step: {getattr(self.trainer, 'global_step', 'N/A')}) ---")
            traceback.print_exc()
            print(f"--- Pausing training due to error. Check logs. ---")
            self.is_training = False
            self.status = "Error" # Update status


    def _render(self):
        """Delegates rendering to the UIRenderer."""
        # Get summary stats primarily for UI display (TensorBoard has more detailed logs)
        # Use the simple recorder part of the TensorBoard logger for UI stats
        if isinstance(self.stats_recorder, TensorBoardStatsRecorder):
             stats_summary = self.stats_recorder.get_summary(self.trainer.global_step)
        else: # Should not happen based on init, but good fallback
             stats_summary = {}

        buffer_capacity = getattr(self.buffer, "capacity", 0)

        self.renderer.render_all(
            is_training=self.is_training,
            status=self.status,
            stats_summary=stats_summary,
            buffer_capacity=buffer_capacity,
            envs=self.envs,
            num_envs=self.num_envs,
            env_config=self.env_config,
            cleanup_confirmation_active=self.cleanup_confirmation_active,
            cleanup_message=self.cleanup_message,
            last_cleanup_message_time=self.last_cleanup_message_time,
            # Pass TensorBoard log dir instead of WandB URL
            tensorboard_log_dir=self.tensorboard_config.LOG_DIR,
        )
        # Check if status message time expired
        if time.time() - self.last_cleanup_message_time >= 5.0:
            self.cleanup_message = ""

    def run(self):
        print("Starting main application loop...")
        running = True
        try:
            while running:
                running = self._handle_input()
                if not running:
                    break

                self._update()
                self._render()

                self.clock.tick(
                    self.vis_config.FPS if self.vis_config.FPS > 0 else 0
                )
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Exiting gracefully...")
            running = False # Ensure loop terminates
        except Exception as e:
            print("\n--- UNHANDLED EXCEPTION IN MAIN LOOP ---")
            traceback.print_exc()
            print("--- EXITING ---")
            running = False # Ensure loop terminates
        finally:
            print("Exiting application...")
            if hasattr(self, "trainer") and self.trainer:
                print("Performing final trainer cleanup...")
                # Save final state unless an error occurred? Let cleanup handle it.
                self.trainer.cleanup(save_final=True)
            if hasattr(self, "stats_recorder") and self.stats_recorder:
                 # Close TensorBoard writer
                self.stats_recorder.close()

            pygame.quit()
            print("Application exited.")


def run_pre_checks():
    print("--- Pre-Run Checks ---")
    try:
        print("Checking GameState and Configuration Compatibility...")
        gs_test = GameState()
        gs_test.reset()
        s_test = gs_test.get_state()
        if len(s_test) != EnvConfig.STATE_DIM:
            raise ValueError(
                f"State Dim Mismatch! GameState:{len(s_test)}, EnvConfig:{EnvConfig.STATE_DIM}"
            )
        print(f"GameState state dimension check PASSED (Length: {len(s_test)}).")
        _ = gs_test.valid_actions()
        print("GameState valid_actions check PASSED.")
        if not hasattr(gs_test, "game_score"):
            raise AttributeError("GameState missing 'game_score' attribute!")
        print("GameState 'game_score' attribute check PASSED.")
        if not hasattr(gs_test, "lines_cleared_this_episode"):
            raise AttributeError("GameState missing 'lines_cleared_this_episode' attribute!")
        print("GameState 'lines_cleared_this_episode' attribute check PASSED.")
        del gs_test
        print("--- Pre-Run Checks Complete ---")
        return True
    except (NameError, ImportError) as e:
        print(f"FATAL ERROR: Import/Name error: {e}")
    except (ValueError, AttributeError) as e:
        print(f"FATAL ERROR during pre-run checks: {e}")
    except Exception as e:
        print(f"FATAL ERROR during GameState pre-check: {e}")
        traceback.print_exc()
    sys.exit(1)


if __name__ == "__main__":
    # Base directories are created in config now, ensure they exist if needed early
    os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    # Other directories might be implicitly created by imports or savers
    os.makedirs("ui", exist_ok=True) # Keep UI dir
    os.makedirs("stats", exist_ok=True) # Keep stats dir

    if run_pre_checks():
        app = MainApp()
        app.run()

```

```python
# File: ui/renderer.py
import pygame
import math
import time
import os # Import os for path manipulations
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

from config import (
    VisConfig,
    EnvConfig,
    ModelConfig,
    DQNConfig,
    DEVICE,
    BufferConfig,
    StatsConfig,
    TrainConfig,
    # WandbConfig removed
)
from environment.game_state import GameState
from environment.shape import Shape
from environment.triangle import Triangle


# Updated Tooltips (Removed WandB, Added TensorBoard)
TOOLTIP_TEXTS = {
    "Status": "Current state: Paused, Buffering (collecting initial data), Training, Confirm Cleanup, or Error.",
    "Global Steps": "Total environment steps across all parallel environments.",
    "Total Episodes": "Total completed episodes across all environments.",
    "Steps/Sec": f"Average global steps processed per second (rolling average over ~{StatsConfig.STATS_AVG_WINDOW} logs).",
    "Avg RL Score": f"Average RL reward sum per episode (last {StatsConfig.STATS_AVG_WINDOW} eps).",
    "Best RL Score": "Highest RL reward sum in a single episode this run.",
    "Avg Game Score": f"Average game score per episode (last {StatsConfig.STATS_AVG_WINDOW} eps).",
    "Best Game Score": "Highest game score in a single episode this run.",
    "Avg Length": f"Average steps per episode (last {StatsConfig.STATS_AVG_WINDOW} eps).",
    "Avg Lines Clr": f"Average lines cleared per episode (last {StatsConfig.STATS_AVG_WINDOW} eps).",
    "Avg Loss": f"Average DQN loss (last {StatsConfig.STATS_AVG_WINDOW} training steps).",
    "Avg Max Q": f"Average max predicted Q-value (last {StatsConfig.STATS_AVG_WINDOW} training batches).",
    "PER Beta": f"PER Importance Sampling exponent (anneals {BufferConfig.PER_BETA_START:.1f} -> 1.0). {BufferConfig.PER_BETA_FRAMES/1e6:.1f}M steps.",
    "Buffer": f"Replay buffer fill status ({BufferConfig.REPLAY_BUFFER_SIZE / 1e6:.1f}M capacity).",
    "Train Button": "Click to Start/Pause the training process (or press 'P').",
    "Cleanup Button": "Click to delete saved agent & buffer for the CURRENT run, then restart components.",
    "Device": f"Computation device ({DEVICE.type.upper()}).",
    "Network": f"Agent Network Architecture (CNN+MLP Fusion). Noisy={DQNConfig.USE_NOISY_NETS}, Dueling={DQNConfig.USE_DUELING}",
    "TensorBoard Status": "Status of TensorBoard logging. Log directory path is shown.",
}


class UIRenderer:
    """Handles rendering the Pygame UI, including stats and game environments."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config

        # Fonts
        try:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_env_score = pygame.font.SysFont(None, 18)
            self.font_env_overlay = pygame.font.SysFont(None, 36)
            self.font_tooltip = pygame.font.SysFont(None, 18)
            self.font_status = pygame.font.SysFont(None, 28)
            self.font_logdir = pygame.font.SysFont(None, 16) # Smaller font for log dir path
        except Exception as e:
            print(f"Warning: Error initializing SysFont: {e}. Using default font.")
            self.font_ui = pygame.font.Font(None, 24)
            self.font_env_score = pygame.font.Font(None, 18)
            self.font_env_overlay = pygame.font.Font(None, 36)
            self.font_tooltip = pygame.font.Font(None, 18)
            self.font_status = pygame.font.Font(None, 28)
            self.font_logdir = pygame.font.Font(None, 16)

        self.stat_rects: Dict[str, pygame.Rect] = {}
        self.hovered_stat_key: Optional[str] = None
        # self.wandb_link_rect removed

    def _render_left_panel(
        self,
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        tensorboard_log_dir: Optional[str], # Path to TensorBoard log dir
    ):
        """Renders the left information panel."""
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(250, self.vis_config.LEFT_PANEL_WIDTH))
        lp_rect = pygame.Rect(0, 0, lp_width, current_height)

        status_color_map = {
            "Paused": (30, 30, 30), "Buffering": (30, 40, 30),
            "Training": (40, 30, 30), "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30), "Error": (60, 0, 0),
        }
        bg_color = status_color_map.get(status, (30, 30, 30))
        pygame.draw.rect(self.screen, bg_color, lp_rect)

        # Buttons
        train_btn_rect = pygame.Rect(10, 10, 100, 40)
        pygame.draw.rect(self.screen, (70, 70, 70), train_btn_rect, border_radius=5)
        btn_text = "Pause" if is_training else "Train"
        lbl_surf = self.font_ui.render(btn_text, True, VisConfig.WHITE)
        self.screen.blit(lbl_surf, lbl_surf.get_rect(center=train_btn_rect.center))

        cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 160, 40) # Wider
        pygame.draw.rect(self.screen, (100, 40, 40), cleanup_btn_rect, border_radius=5)
        cleanup_lbl_surf = self.font_ui.render("Cleanup This Run", True, VisConfig.WHITE) # Updated text
        self.screen.blit(cleanup_lbl_surf, cleanup_lbl_surf.get_rect(center=cleanup_btn_rect.center))

        # Status Text
        status_surf = self.font_status.render(f"Status: {status}", True, VisConfig.YELLOW)
        status_rect = status_surf.get_rect(topleft=(10, train_btn_rect.bottom + 10))
        self.screen.blit(status_surf, status_rect)

        # Info Text & Tooltips
        self.stat_rects.clear()
        self.stat_rects["Train Button"] = train_btn_rect
        self.stat_rects["Cleanup Button"] = cleanup_btn_rect
        self.stat_rects["Status"] = status_rect
        # self.wandb_link_rect removed

        buffer_size = stats_summary.get("buffer_size", 0)
        buffer_perc = (buffer_size / buffer_capacity * 100) if buffer_capacity > 0 else 0.0

        info_lines_data = [
            ("Global Steps", f"{stats_summary.get('global_step', 0)/1e6:.2f}M / {TrainConfig.TOTAL_TRAINING_STEPS/1e6:.1f}M"),
            ("Total Episodes", f"{stats_summary.get('total_episodes', 0)}"),
            ("Steps/Sec", f"{stats_summary.get('steps_per_second', 0.0):.1f}"),
            ("Avg RL Score", f"({stats_summary.get('num_ep_scores', 0)}): {stats_summary.get('avg_score_100', 0.0):.2f}"),
            ("Best RL Score", f"{stats_summary.get('best_score', 0.0):.2f}"),
            ("Avg Game Score", f"({stats_summary.get('num_game_scores', 0)}): {stats_summary.get('avg_game_score_100', 0.0):.1f}"),
            ("Best Game Score", f"{stats_summary.get('best_game_score', 0.0):.1f}"),
            ("Avg Length", f"({stats_summary.get('num_ep_lengths', 0)}): {stats_summary.get('avg_length_100', 0.0):.1f}"),
            ("Avg Lines Clr", f"({stats_summary.get('num_lines_cleared', 0)}): {stats_summary.get('avg_lines_cleared_100', 0.0):.2f}"),
            ("Avg Loss", f"({stats_summary.get('num_losses', 0)}): {stats_summary.get('avg_loss_100', 0.0):.4f}"),
            ("Avg Max Q", f"({stats_summary.get('num_avg_max_qs', 0)}): {stats_summary.get('avg_max_q_100', 0.0):.3f}"),
            ("PER Beta", (f"{stats_summary.get('beta', 0.0):.3f}" if BufferConfig.USE_PER else "N/A")),
            ("Buffer", f"{buffer_size/1e6:.2f}M / {buffer_capacity/1e6:.1f}M ({buffer_perc:.1f}%)"),
            ("Device", f"{DEVICE.type.upper()}"),
            ("Network", f"CNN+MLP Fusion"),
        ]

        text_y_start = status_rect.bottom + 10
        line_height = self.font_ui.get_linesize()

        for idx, (key, value_str) in enumerate(info_lines_data):
            line_text = f"{key}: {value_str}"
            line_surf = self.font_ui.render(line_text, True, VisConfig.WHITE)
            line_rect = line_surf.get_rect(topleft=(10, text_y_start + idx * line_height))
            line_rect.width = min(line_rect.width, lp_width - 20) # Limit rect width
            self.screen.blit(line_surf, line_rect)
            self.stat_rects[key] = line_rect

        # --- TensorBoard Status ---
        tb_y_start = text_y_start + len(info_lines_data) * line_height + 10
        tb_status_text = "TensorBoard: Logging Active"
        tb_status_color = VisConfig.GOOGLE_COLORS[0] # Green

        tb_surf = self.font_ui.render(tb_status_text, True, tb_status_color)
        tb_rect = tb_surf.get_rect(topleft=(10, tb_y_start))
        self.screen.blit(tb_surf, tb_rect)
        self.stat_rects["TensorBoard Status"] = tb_rect # Add tooltip hover

        # Display Log Directory Path (make it shorter)
        if tensorboard_log_dir:
            try:
                # Show relative path if possible, or just the last couple of dirs
                rel_log_dir = os.path.relpath(tensorboard_log_dir)
                if len(rel_log_dir) > 50: # Heuristic for too long path
                     parts = tensorboard_log_dir.split(os.sep)[-3:] # Show last 3 parts
                     rel_log_dir = os.path.join("...", *parts)
            except ValueError: # Happens if on different drives (Windows)
                rel_log_dir = tensorboard_log_dir # Show full path

            dir_surf = self.font_logdir.render(f"Log Dir: {rel_log_dir}", True, VisConfig.LIGHTG)
            dir_rect = dir_surf.get_rect(topleft=(10, tb_rect.bottom + 2))
            self.screen.blit(dir_surf, dir_rect)
            # Make the directory text area also trigger the tooltip
            combined_tb_rect = tb_rect.union(dir_rect)
            self.stat_rects["TensorBoard Status"] = combined_tb_rect


    def _render_shape_preview(self, surf: pygame.Surface, shape: Shape, cell_size: int):
        if not shape or not shape.triangles: return
        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h_cells = max_r - min_r + 1
        shape_w_cells = max_c - min_c + 1
        total_w_pixels = shape_w_cells * (cell_size * 0.75) + (cell_size * 0.25)
        total_h_pixels = shape_h_cells * cell_size
        offset_x = (surf.get_width() - total_w_pixels) / 2 - min_c * (cell_size * 0.75)
        offset_y = (surf.get_height() - total_h_pixels) / 2 - min_r * cell_size
        for dr, dc, up in shape.triangles:
            tri = Triangle(row=dr, col=dc, is_up=up)
            pts = tri.get_points(ox=offset_x, oy=offset_y, cw=cell_size, ch=cell_size)
            pygame.draw.polygon(surf, shape.color, pts)

    def _render_env(self, surf: pygame.Surface, env: GameState, cell_w: int, cell_h: int):
        try:
            bg_color = VisConfig.YELLOW if env.is_blinking() else \
                       (30, 30, 100) if env.is_frozen() and not env.is_over() else \
                       (20, 20, 20)
            surf.fill(bg_color)

            # Render Grid Triangles
            if hasattr(env, 'grid') and hasattr(env.grid, 'triangles'):
                for r in range(env.grid.rows):
                    for c in range(env.grid.cols):
                        t = env.grid.triangles[r][c]
                        if not hasattr(t, 'get_points'): continue
                        try:
                            pts = t.get_points(ox=0, oy=0, cw=cell_w, ch=cell_h)
                            color = VisConfig.GRAY
                            if t.is_death: color = VisConfig.BLACK
                            elif t.is_occupied: color = t.color if t.color else VisConfig.RED
                            pygame.draw.polygon(surf, color, pts)
                        except Exception as e_render:
                            print(f"Error rendering tri ({r},{c}): {e_render}")
            else:
                 pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
                 err_txt = self.font_env_overlay.render("Invalid Grid", True, VisConfig.RED)
                 surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))

            # Render Scores
            rl_score_val = env.score
            game_score_val = env.game_score
            score_surf = self.font_env_score.render(
                f"GS: {game_score_val} | R: {rl_score_val:.1f}", True, VisConfig.WHITE, (0,0,0,180)
            )
            surf.blit(score_surf, (4, 4))

            # Render Overlays
            if env.is_over():
                overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                overlay.fill((100, 0, 0, 180))
                surf.blit(overlay, (0, 0))
                over_text = self.font_env_overlay.render("GAME OVER", True, VisConfig.WHITE)
                surf.blit(over_text, over_text.get_rect(center=surf.get_rect().center))
            elif env.is_frozen() and not env.is_blinking():
                freeze_text = self.font_env_overlay.render("Frozen", True, VisConfig.WHITE)
                surf.blit(freeze_text, freeze_text.get_rect(center=(surf.get_width()//2, surf.get_height()-15)))

        except AttributeError as e:
            pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
            err_txt = self.font_env_overlay.render(f"Attr Err: {e}", True, VisConfig.RED, VisConfig.BLACK)
            surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))
        except Exception as e:
            print(f"Unexpected Render Error in _render_env: {e}")
            pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
            traceback.print_exc()

    def _render_game_area(self, envs: List[GameState], num_envs: int, env_config: EnvConfig):
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(250, self.vis_config.LEFT_PANEL_WIDTH))
        ga_rect = pygame.Rect(lp_width, 0, current_width - lp_width, current_height)

        if num_envs <= 0 or ga_rect.width <= 0 or ga_rect.height <= 0: return

        render_limit = self.vis_config.NUM_ENVS_TO_RENDER
        num_envs_to_render = num_envs if render_limit <= 0 else min(num_envs, render_limit)
        if num_envs_to_render <= 0: return

        aspect_ratio = ga_rect.width / ga_rect.height
        cols_env = max(1, int(math.sqrt(num_envs_to_render * aspect_ratio)))
        rows_env = math.ceil(num_envs_to_render / cols_env)

        total_spacing_w = (cols_env + 1) * self.vis_config.ENV_SPACING
        total_spacing_h = (rows_env + 1) * self.vis_config.ENV_SPACING
        cell_w = (ga_rect.width - total_spacing_w) // cols_env if cols_env > 0 else 0
        cell_h = (ga_rect.height - total_spacing_h) // rows_env if rows_env > 0 else 0

        if cell_w > 10 and cell_h > 10:
            env_idx = 0
            for r in range(rows_env):
                for c in range(cols_env):
                    if env_idx >= num_envs_to_render: break
                    env_x = ga_rect.x + self.vis_config.ENV_SPACING * (c + 1) + c * cell_w
                    env_y = ga_rect.y + self.vis_config.ENV_SPACING * (r + 1) + r * cell_h
                    env_rect = pygame.Rect(env_x, env_y, cell_w, cell_h)
                    try:
                        sub_surf = self.screen.subsurface(env_rect)
                        tri_cell_w = cell_w / (env_config.COLS * 0.75 + 0.25)
                        tri_cell_h = cell_h / env_config.ROWS
                        self._render_env(sub_surf, envs[env_idx], int(tri_cell_w), int(tri_cell_h))
                        # Shape Previews
                        available_shapes = envs[env_idx].get_shapes()
                        if available_shapes:
                            preview_dim = max(10, min(cell_w // 6, cell_h // 6, 25))
                            preview_spacing = 4
                            total_preview_width = len(available_shapes) * preview_dim + max(0, len(available_shapes)-1) * preview_spacing
                            start_x = sub_surf.get_width() - total_preview_width - preview_spacing
                            start_y = preview_spacing
                            for i, shape in enumerate(available_shapes):
                                preview_x = start_x + i * (preview_dim + preview_spacing)
                                temp_shape_surf = pygame.Surface((preview_dim, preview_dim), pygame.SRCALPHA)
                                temp_shape_surf.fill((0,0,0,0))
                                preview_cell_size = max(2, preview_dim // 4)
                                self._render_shape_preview(temp_shape_surf, shape, preview_cell_size)
                                sub_surf.blit(temp_shape_surf, (preview_x, start_y))
                    except ValueError:
                        pygame.draw.rect(self.screen, (0, 0, 50), env_rect, 1) # Subsurface error
                    except Exception as e_render_env:
                        print(f"Error rendering env {env_idx}: {e_render_env}")
                        pygame.draw.rect(self.screen, (50, 0, 50), env_rect, 1)
                    env_idx += 1
        else:
            err_surf = self.font_ui.render(f"Envs Too Small ({cell_w}x{cell_h})", True, VisConfig.GRAY)
            self.screen.blit(err_surf, err_surf.get_rect(center=ga_rect.center))

        # Display text if not all envs are rendered
        if num_envs_to_render < num_envs:
            info_surf = self.font_ui.render(f"Rendering {num_envs_to_render}/{num_envs} Envs", True, VisConfig.YELLOW, VisConfig.BLACK)
            self.screen.blit(info_surf, info_surf.get_rect(bottomright=(ga_rect.right - 5, ga_rect.bottom - 5)))


    def _render_cleanup_confirmation(self):
        current_width, current_height = self.screen.get_size()
        overlay = pygame.Surface((current_width, current_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        center_x, center_y = current_width // 2, current_height // 2
        prompt_l1 = self.font_env_overlay.render("DELETE CURRENT RUN DATA?", True, VisConfig.RED)
        self.screen.blit(prompt_l1, prompt_l1.get_rect(center=(center_x, center_y - 60)))
        prompt_l2 = self.font_ui.render("(Agent Checkpoint & Buffer State)", True, VisConfig.WHITE) # Simplified
        self.screen.blit(prompt_l2, prompt_l2.get_rect(center=(center_x, center_y - 25)))
        prompt_l3 = self.font_ui.render("This action cannot be undone!", True, VisConfig.YELLOW)
        self.screen.blit(prompt_l3, prompt_l3.get_rect(center=(center_x, center_y)))
        confirm_yes_rect = pygame.Rect(center_x - 110, center_y + 30, 100, 40)
        confirm_no_rect = pygame.Rect(center_x + 10, center_y + 30, 100, 40)
        pygame.draw.rect(self.screen, (0, 150, 0), confirm_yes_rect, border_radius=5)
        pygame.draw.rect(self.screen, (150, 0, 0), confirm_no_rect, border_radius=5)
        yes_text = self.font_ui.render("YES", True, VisConfig.WHITE)
        no_text = self.font_ui.render("NO", True, VisConfig.WHITE)
        self.screen.blit(yes_text, yes_text.get_rect(center=confirm_yes_rect.center))
        self.screen.blit(no_text, no_text.get_rect(center=confirm_no_rect.center))

    def _render_status_message(self, message: str, last_message_time: float):
        if message and (time.time() - last_message_time < 5.0):
            current_width, current_height = self.screen.get_size()
            lines = message.split("\n")
            max_width = 0
            msg_surfs = []
            for line in lines:
                msg_surf = self.font_ui.render(line, True, VisConfig.YELLOW, VisConfig.BLACK)
                msg_surfs.append(msg_surf)
                max_width = max(max_width, msg_surf.get_width())

            total_height = sum(s.get_height() for s in msg_surfs) + max(0, len(lines) - 1) * 2
            bg_rect = pygame.Rect(0, 0, max_width + 10, total_height + 10)
            bg_rect.midbottom = (current_width // 2, current_height - 10)
            pygame.draw.rect(self.screen, VisConfig.BLACK, bg_rect, border_radius=3)
            current_y = bg_rect.top + 5
            for msg_surf in msg_surfs:
                msg_rect = msg_surf.get_rect(midtop=(bg_rect.centerx, current_y))
                self.screen.blit(msg_surf, msg_rect)
                current_y += msg_surf.get_height() + 2
            return True
        return False

    def _render_tooltip(self):
        if self.hovered_stat_key and self.hovered_stat_key in TOOLTIP_TEXTS:
            tooltip_text = TOOLTIP_TEXTS[self.hovered_stat_key]
            mouse_pos = pygame.mouse.get_pos()
            lines = []
            max_width = 300
            words = tooltip_text.split(" ")
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                test_surf = self.font_tooltip.render(test_line, True, VisConfig.BLACK)
                if test_surf.get_width() <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)

            line_surfs = [self.font_tooltip.render(line, True, VisConfig.BLACK) for line in lines]
            total_height = sum(s.get_height() for s in line_surfs)
            max_line_width = max(s.get_width() for s in line_surfs)

            padding = 5
            tooltip_rect = pygame.Rect(mouse_pos[0] + 15, mouse_pos[1] + 10,
                                        max_line_width + padding * 2, total_height + padding * 2)
            tooltip_rect.clamp_ip(self.screen.get_rect())

            pygame.draw.rect(self.screen, VisConfig.YELLOW, tooltip_rect, border_radius=3)
            pygame.draw.rect(self.screen, VisConfig.BLACK, tooltip_rect, 1, border_radius=3)

            current_y = tooltip_rect.y + padding
            for surf in line_surfs:
                self.screen.blit(surf, (tooltip_rect.x + padding, current_y))
                current_y += surf.get_height()

    def check_hover(self, mouse_pos: Tuple[int, int]):
        self.hovered_stat_key = None
        # Iterate in reverse order so tooltips for elements drawn last appear first
        for key, rect in reversed(self.stat_rects.items()):
            if rect.collidepoint(mouse_pos):
                self.hovered_stat_key = key
                return # Found one, stop checking

    def render_all(
        self,
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_confirmation_active: bool,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str], # Use TB log dir
    ):
        """Renders all UI components."""
        try:
            self.screen.fill(VisConfig.BLACK)
            self._render_left_panel(
                is_training, status, stats_summary, buffer_capacity, tensorboard_log_dir
            )
            self._render_game_area(envs, num_envs, env_config)

            if cleanup_confirmation_active:
                self._render_cleanup_confirmation()

            message_active = self._render_status_message(
                cleanup_message, last_cleanup_message_time
            )

            if not cleanup_confirmation_active and not message_active:
                self._render_tooltip()

            pygame.display.flip()
        except pygame.error as e:
            print(f"Pygame rendering error: {e}")
        except Exception as e:
            print(f"Unexpected critical rendering error: {e}")
            import traceback
            traceback.print_exc()

```

```python
# File: training/trainer.py
import time
import torch
import numpy as np
import os
import pickle
import random
import traceback
from typing import List, Optional, Union, Tuple
from collections import deque

from config import (
    EnvConfig, DQNConfig, TrainConfig, BufferConfig, ModelConfig, DEVICE
)
from environment.game_state import GameState
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from agent.replay_buffer.buffer_utils import create_replay_buffer
from stats.stats_recorder import StatsRecorderBase # Base class
from utils.helpers import ensure_numpy
from utils.types import ActionType, StateType

class Trainer:
    """Orchestrates the DQN training process."""

    def __init__(
        self,
        envs: List[GameState],
        agent: DQNAgent,
        buffer: ReplayBufferBase,
        stats_recorder: StatsRecorderBase,
        env_config: EnvConfig,
        dqn_config: DQNConfig,
        train_config: TrainConfig,
        buffer_config: BufferConfig,
        model_config: ModelConfig,
        model_save_path: str,      # Path to save model for THIS run
        buffer_save_path: str,     # Path to save buffer for THIS run
        load_checkpoint_path: Optional[str] = None, # Optional path to LOAD model state
        load_buffer_path: Optional[str] = None,     # Optional path to LOAD buffer state
    ):
        print("[Trainer] Initializing...")
        self.envs = envs
        self.agent = agent
        self.buffer = buffer
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.device = DEVICE

        # Store configs
        self.env_config = env_config
        self.dqn_config = dqn_config
        self.train_config = train_config
        self.buffer_config = buffer_config
        self.model_config = model_config # Keep for reference, not direct loading/saving paths

        # Specific paths for this run
        self.model_save_path = model_save_path
        self.buffer_save_path = buffer_save_path

        # State / Trackers
        self.global_step = 0
        self.episode_count = 0
        try:
            self.current_states: List[StateType] = [ensure_numpy(env.reset()) for env in self.envs]
        except Exception as e:
            print(f"FATAL ERROR during initial reset: {e}")
            raise e
        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_game_scores = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_lines_cleared = np.zeros(self.num_envs, dtype=np.int32)

        # --- Loading State ---
        # Load Agent/Optimizer State
        if load_checkpoint_path:
            self._load_checkpoint(load_checkpoint_path)
        else:
            print("[Trainer] No checkpoint specified to load, starting agent from scratch.")
            self._reset_trainer_state() # Ensure step/ep counts are zero

        # Load Buffer State
        if load_buffer_path:
            self._load_buffer_state(load_buffer_path)
        else:
            print("[Trainer] No buffer specified to load, starting buffer empty.")
            # Ensure buffer is fresh if not loading
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)


        # Initial PER Beta update & logging
        initial_beta = self._update_beta()
        self.stats_recorder.record_step(
            {
                "buffer_size": len(self.buffer),
                "epsilon": 0.0, # Epsilon is 0 with Noisy Nets
                "beta": initial_beta,
                "global_step": self.global_step, # Log initial step count
                "episode_count": self.episode_count # Log initial ep count
            }
        )

        print(
            f"[Trainer] Init complete. Start Step={self.global_step}, Ep={self.episode_count}, Buf={len(self.buffer)}, Beta={initial_beta:.4f}"
        )


    def _load_checkpoint(self, path_to_load: str):
        if not os.path.isfile(path_to_load):
            print(f"[Trainer] LOAD WARNING: Checkpoint file not found at {path_to_load}. Starting agent from scratch.")
            self._reset_trainer_state()
            return

        print(f"[Trainer] Loading agent checkpoint from: {path_to_load}")
        try:
            checkpoint = torch.load(path_to_load, map_location=self.device)
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            self.global_step = checkpoint.get("global_step", 0)
            self.episode_count = checkpoint.get("episode_count", 0)
            print(f"[Trainer] Checkpoint loaded. Resuming from step {self.global_step}, ep {self.episode_count}")
        except FileNotFoundError: # Should be caught by isfile, but belt-and-suspenders
            print(f"[Trainer] Checkpoint file disappeared? ({path_to_load}). Starting fresh.")
            self._reset_trainer_state()
        except KeyError as e:
            print(f"[Trainer] Checkpoint missing key '{e}'. Incompatible format? Starting fresh.")
            self._reset_trainer_state()
        except Exception as e:
            print(f"[Trainer] CRITICAL ERROR loading checkpoint: {e}. Starting fresh.")
            traceback.print_exc()
            self._reset_trainer_state()


    def _reset_trainer_state(self):
        """Resets step and episode counters."""
        self.global_step = 0
        self.episode_count = 0

    def _load_buffer_state(self, path_to_load: str):
        if not os.path.isfile(path_to_load):
            print(f"[Trainer] LOAD WARNING: Buffer file not found at {path_to_load}. Starting empty buffer.")
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
            return

        print(f"[Trainer] Attempting to load buffer state from: {path_to_load}")
        try:
            if hasattr(self.buffer, "load_state"):
                self.buffer.load_state(path_to_load)
                print(f"[Trainer] Buffer state loaded. Size: {len(self.buffer)}")
            else:
                print("[Trainer] Warning: Buffer object has no 'load_state' method. Cannot load.")
        except FileNotFoundError:
            print(f"[Trainer] Buffer file disappeared? ({path_to_load}). Starting empty.")
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
        except (EOFError, pickle.UnpicklingError, ImportError, AttributeError, ValueError) as e:
            print(f"[Trainer] ERROR loading buffer (incompatible/corrupt?): {e}. Starting empty.")
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
        except Exception as e:
            print(f"[Trainer] CRITICAL ERROR loading buffer: {e}. Starting empty.")
            traceback.print_exc()
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)


    def _save_checkpoint(self, is_final=False):
        """Saves agent state and buffer state to the paths defined for THIS run."""
        prefix = "FINAL" if is_final else f"step_{self.global_step}"
        save_dir = os.path.dirname(self.model_save_path)
        os.makedirs(save_dir, exist_ok=True) # Ensure directory exists

        # Save Agent State
        print(f"[Trainer] Saving agent checkpoint ({prefix}) to: {self.model_save_path}")
        try:
            save_data = {
                "global_step": self.global_step,
                "episode_count": self.episode_count,
                "agent_state_dict": self.agent.get_state_dict(),
                # Add config hashes/identifiers here? Could be useful for compatibility checks
            }
            torch.save(save_data, self.model_save_path)
            print(f"[Trainer] Agent checkpoint ({prefix}) saved.")
        except Exception as e:
            print(f"[Trainer] ERROR saving agent checkpoint ({prefix}): {e}")
            traceback.print_exc()


        # Save Buffer State
        print(f"[Trainer] Saving buffer state ({prefix}) to: {self.buffer_save_path} (Size: {len(self.buffer)})")
        try:
            if hasattr(self.buffer, "save_state"):
                self.buffer.save_state(self.buffer_save_path)
                print(f"[Trainer] Buffer state ({prefix}) saved.")
            else:
                print("[Trainer] Warning: Buffer does not support save_state.")
        except Exception as e:
            print(f"[Trainer] ERROR saving buffer state ({prefix}): {e}")
            traceback.print_exc()


    def _update_beta(self) -> float:
        """Updates PER beta based on global steps and returns current beta."""
        if not self.buffer_config.USE_PER:
            beta = 1.0 # Beta is irrelevant if not using PER
        else:
            start = self.buffer_config.PER_BETA_START
            end = 1.0
            anneal_frames = self.buffer_config.PER_BETA_FRAMES
            if anneal_frames <= 0:
                beta = end
            else:
                fraction = min(1.0, float(self.global_step) / anneal_frames)
                beta = start + fraction * (end - start)
            # Update beta in the buffer object (necessary for PER sampling)
            if hasattr(self.buffer, "set_beta"):
                self.buffer.set_beta(beta)
            else:
                 print("Warning: PER is enabled but buffer has no set_beta method.")
        # Record beta value for logging (stats_recorder handles the actual logging action)
        self.stats_recorder.record_step({"beta": beta, "global_step": self.global_step})
        return beta


    def _collect_experience(self):
        """Performs one step in each parallel env, stores transition, handles resets."""
        actions: List[ActionType] = [-1] * self.num_envs # Placeholder

        # --- 1. Select Actions ---
        # No epsilon needed for Noisy Nets
        for i in range(self.num_envs):
            # Reset env if done (handle potential errors)
            if self.envs[i].is_over():
                try:
                    self.current_states[i] = ensure_numpy(self.envs[i].reset())
                    self.current_episode_scores[i] = 0.0
                    self.current_episode_lengths[i] = 0
                    self.current_episode_game_scores[i] = 0
                    self.current_episode_lines_cleared[i] = 0
                except Exception as e:
                    print(f"ERROR: Env {i} failed reset: {e}")
                    self.current_states[i] = np.zeros(self.env_config.STATE_DIM, dtype=np.float32) # Dummy state

            # Get valid actions for the current state
            valid_actions = self.envs[i].valid_actions()

            # Choose action using agent
            if not valid_actions:
                actions[i] = 0 # Default action if no valid moves (should only be if already over)
            else:
                try:
                    # Epsilon is ignored by agent.select_action when using Noisy Nets
                    actions[i] = self.agent.select_action(self.current_states[i], 0.0, valid_actions)
                except Exception as e:
                    print(f"ERROR: Agent select_action env {i}: {e}")
                    actions[i] = random.choice(valid_actions) # Fallback

        # --- 2. Step Environments & Store Transitions ---
        next_states_list: List[StateType] = [np.zeros_like(self.current_states[0]) for _ in range(self.num_envs)]
        rewards_list = np.zeros(self.num_envs, dtype=np.float32)
        dones_list = np.zeros(self.num_envs, dtype=bool)

        for i in range(self.num_envs):
            env = self.envs[i]
            current_state = self.current_states[i]
            action = actions[i]

            # Execute action in environment
            try:
                reward, done = env.step(action)
                next_state = ensure_numpy(env.get_state())
            except Exception as e:
                print(f"ERROR: Env {i} step failed (Action: {action}): {e}")
                reward = self.reward_config.PENALTY_GAME_OVER
                done = True
                next_state = current_state # Reuse current state on error
                if hasattr(env, 'game_over'): env.game_over = True # Ensure env knows it's over

            # Store results for this step
            rewards_list[i] = reward
            dones_list[i] = done
            next_states_list[i] = next_state # Store the obtained next state

            # Store transition in replay buffer (buffer handles N-step logic)
            try:
                self.buffer.push(current_state, action, reward, next_state, done)
            except Exception as e:
                print(f"ERROR: Buffer push env {i}: {e}")


            # --- 3. Update Trackers ---
            self.current_episode_scores[i] += reward
            self.current_episode_lengths[i] += 1
            self.current_episode_game_scores[i] = env.game_score
            self.current_episode_lines_cleared[i] = env.lines_cleared_this_episode

            # Record step reward for detailed stats (optional)
            # self.stats_recorder.record_step({"step_reward": reward, "global_step": self.global_step + i + 1})

            # --- 4. Handle Episode End ---
            if done:
                self.episode_count += 1
                # Use tracked values for final episode summary
                final_rl_score = self.current_episode_scores[i]
                final_length = self.current_episode_lengths[i]
                final_game_score = self.current_episode_game_scores[i]
                final_lines_cleared = self.current_episode_lines_cleared[i]

                # Record episode summary stats (pass global step for association)
                self.stats_recorder.record_episode(
                    episode_score=final_rl_score,
                    episode_length=final_length,
                    episode_num=self.episode_count,
                    global_step=self.global_step + self.num_envs, # Step count *after* this batch
                    game_score=final_game_score,
                    lines_cleared=final_lines_cleared,
                )
                # Actual env reset happens at the start of the next iteration loop

        # --- 5. Update Current States ---
        # This must happen *after* all transitions for the step have been pushed
        self.current_states = next_states_list

        # Increment global step counter
        self.global_step += self.num_envs

        # Record current buffer size and potentially other step stats
        self.stats_recorder.record_step({
            "buffer_size": len(self.buffer),
            "global_step": self.global_step
        })


    def _train_batch(self):
        if (len(self.buffer) < self.train_config.BATCH_SIZE or
            self.global_step < self.train_config.LEARN_START_STEP):
            return

        beta = self._update_beta() # Update PER beta before sampling
        is_n_step = self.buffer_config.USE_N_STEP and self.buffer_config.N_STEP > 1

        # Sample from buffer
        indices, is_weights_np, batch_np_tuple = None, None, None
        try:
            if self.buffer_config.USE_PER:
                sample_result = self.buffer.sample(self.train_config.BATCH_SIZE)
                if sample_result is None:
                    print("Warn: PER Buffer sample returned None.")
                    return
                batch_np_tuple, indices, is_weights_np = sample_result
            else:
                batch_np_tuple = self.buffer.sample(self.train_config.BATCH_SIZE)
                if batch_np_tuple is None:
                     print("Warn: Uniform Buffer sample returned None.")
                     return

        except Exception as e:
            print(f"ERROR sampling buffer: {e}")
            traceback.print_exc()
            return

        # Compute loss
        try:
            loss, td_errors = self.agent.compute_loss(batch_np_tuple, is_n_step, is_weights_np)
        except Exception as e:
            print(f"ERROR computing loss: {e}")
            traceback.print_exc()
            return # Skip update if loss fails

        # Update agent
        try:
            grad_norm = self.agent.update(loss)
        except Exception as e:
            print(f"ERROR updating agent: {e}")
            traceback.print_exc()
            return # Skip priority update if agent update fails

        # Update priorities in PER buffer
        if self.buffer_config.USE_PER and indices is not None and td_errors is not None:
            try:
                td_errors_np = td_errors.squeeze().cpu().numpy()
                self.buffer.update_priorities(indices, td_errors_np)
            except Exception as e:
                print(f"ERROR updating priorities: {e}")
                traceback.print_exc() # Log priority update errors

        # Record training step statistics
        self.stats_recorder.record_step({
            "loss": loss.item(),
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
            "avg_max_q": self.agent.get_last_avg_max_q(),
            "global_step": self.global_step # Associate with current step
        })


    def step(self):
        """Performs one iteration: experience collection & potentially training."""
        step_start_time = time.time()

        # --- Experience Collection ---
        self._collect_experience()

        # --- Learning ---
        # Learn based on global steps collected, check frequency
        if (self.global_step >= self.train_config.LEARN_START_STEP and
            self.global_step % (self.train_config.LEARN_FREQ * self.num_envs) < self.num_envs): # Learn once per LEARN_FREQ env steps *across all envs*
             if len(self.buffer) >= self.train_config.BATCH_SIZE:
                 self._train_batch()

        # --- Target Network Update ---
        # Update based on global step count crossing the frequency boundary
        target_freq = self.dqn_config.TARGET_UPDATE_FREQ
        if target_freq > 0:
            steps_before = self.global_step - self.num_envs
            if steps_before // target_freq < self.global_step // target_freq:
                if self.global_step > 0: # Avoid update at step 0
                    print(f"[Trainer] Updating target network at step {self.global_step}")
                    self.agent.update_target_network()

        # --- Checkpointing ---
        self.maybe_save_checkpoint()

        # Record step time
        step_duration = time.time() - step_start_time
        self.stats_recorder.record_step({"step_time": step_duration, "global_step": self.global_step})


    def maybe_save_checkpoint(self):
        """Saves checkpoint if frequency is met."""
        save_freq = self.train_config.CHECKPOINT_SAVE_FREQ
        if save_freq <= 0: return

        # Save if the global step counter crossed a save frequency boundary
        steps_before = self.global_step - self.num_envs # Steps before this multi-env step
        if steps_before // save_freq < self.global_step // save_freq:
            if self.global_step > 0: # Avoid save at step 0
                self._save_checkpoint(is_final=False)


    def train_loop(self):
        """Main training loop (can be called externally or run standalone)."""
        print("[Trainer] Starting training loop...")
        try:
            while self.global_step < self.train_config.TOTAL_TRAINING_STEPS:
                self.step()
                # Logging (console, TensorBoard) is handled by the stats_recorder called within step()
        except KeyboardInterrupt:
            print("\n[Trainer] Training loop interrupted by user.")
        except Exception as e:
            print(f"\n[Trainer] CRITICAL ERROR in training loop: {e}")
            traceback.print_exc()
        finally:
            print("[Trainer] Training loop finished or terminated.")
            self.cleanup(save_final=True) # Ensure cleanup happens

    def cleanup(self, save_final: bool = True):
        """Cleans up resources: saves final state, flushes buffer, closes logger."""
        print("[Trainer] Cleaning up resources...")

        # 1. Flush N-step buffer if used
        if hasattr(self.buffer, 'flush_pending'):
            print("[Trainer] Flushing pending N-step transitions...")
            try:
                self.buffer.flush_pending()
            except Exception as e:
                print(f"ERROR during buffer flush: {e}")

        # 2. Save final agent and buffer state (optional)
        if save_final:
            print("[Trainer] Saving final checkpoint...")
            try:
                self._save_checkpoint(is_final=True)
            except Exception as e:
                print(f"ERROR during final save: {e}")
        else:
            print("[Trainer] Skipping final save.")

        # 3. Close stats recorder (e.g., TensorBoard SummaryWriter)
        if hasattr(self.stats_recorder, 'close'):
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"ERROR closing stats recorder: {e}")

        print("[Trainer] Cleanup complete.")
```

```python
# File: utils/types.py
# (No changes needed here, it's already clean and framework-agnostic)
from typing import NamedTuple, Union, Tuple, List, Dict, Any, Optional
import numpy as np
import torch

class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float          # For N-step buffer, holds N-step RL reward
    next_state: np.ndarray # For N-step buffer, holds N-step next state
    done: bool             # For N-step buffer, holds N-step done flag
    n_step_discount: Optional[float] = None # gamma^k for N-step

# Type aliases
StateType = np.ndarray
ActionType = int

# --- Batch Types (Numpy) ---
NumpyBatch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (states, actions, rewards, next_states, dones)

NumpyNStepBatch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (states, actions, n_step_rewards, n_step_next_states, n_step_dones, n_step_discounts)

PrioritizedNumpyBatch = Tuple[NumpyBatch, np.ndarray, np.ndarray]
# ((s,a,r,ns,d), indices, weights)

PrioritizedNumpyNStepBatch = Tuple[NumpyNStepBatch, np.ndarray, np.ndarray]
# ((s,a,rn,nsn,dn,gamman), indices, weights)

# --- Batch Types (Tensor) ---
TensorBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
# (states, actions, rewards, next_states, dones)

TensorNStepBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
# (states, actions, n_step_rewards, n_step_next_states, n_step_dones, n_step_discounts)

# --- Agent State ---
AgentStateDict = Dict[str, Any]
```

```python
# File: utils/helpers.py
import torch
import numpy as np
import random
import os
import pickle
import cloudpickle
from typing import Union, Any

def get_device() -> torch.device:
    """Gets the appropriate torch device (MPS, CUDA, or CPU)."""
    force_cpu = os.environ.get("FORCE_CPU", "false").lower() == "true"
    if force_cpu:
        print("Forcing CPU device based on environment variable.")
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        device_str = "mps"
    elif torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"

    print(f"Using device: {device_str.upper()}")
    if device_str == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    elif device_str == "mps":
        print("MPS device found on MacOS.")
    return torch.device(device_str)

def set_random_seeds(seed: int = 42):
    """Sets random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Note: Setting deterministic algorithms can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Set random seeds to {seed}")

def ensure_numpy(data: Union[np.ndarray, list, tuple, torch.Tensor]) -> np.ndarray:
    """Ensures the input data is a numpy array with float32 type."""
    try:
        if isinstance(data, np.ndarray):
            if data.dtype != np.float32:
                return data.astype(np.float32)
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float32)
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=np.float32)
            if arr.dtype == np.object_: # Indicates ragged array
                raise ValueError("Cannot convert ragged list/tuple to float32 numpy array.")
            return arr
        else:
            # Attempt conversion for single numbers or other types
            return np.array([data], dtype=np.float32)
    except (ValueError, TypeError, RuntimeError) as e:
        print(f"CRITICAL ERROR in ensure_numpy conversion: {e}. Input type: {type(data)}. Data (partial): {str(data)[:100]}")
        raise ValueError(f"ensure_numpy failed: {e}") from e


def save_object(obj: Any, filepath: str):
    """Saves an arbitrary Python object to a file using cloudpickle."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            cloudpickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error saving object to {filepath}: {e}")
        raise e # Re-raise after logging

def load_object(filepath: str) -> Any:
    """Loads a Python object from a file using cloudpickle."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found for loading: {filepath}")
    try:
        with open(filepath, "rb") as f:
            obj = cloudpickle.load(f)
        return obj
    except Exception as e:
        print(f"Error loading object from {filepath}: {e}")
        raise e # Re-raise after logging
```

```python
# File: agent/dqn_agent.py
# (Largely unchanged structurally, Noisy Nets were already core)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import traceback
from typing import Tuple, List, Dict, Any, Optional, Union

from config import EnvConfig, ModelConfig, DQNConfig, DEVICE
from agent.model_factory import create_network
from utils.types import (
    StateType, ActionType, NumpyBatch, NumpyNStepBatch,
    AgentStateDict, TensorBatch, TensorNStepBatch
)
from utils.helpers import ensure_numpy
from agent.networks.noisy_layer import NoisyLinear # Keep for type checking info

class DQNAgent:
    """DQN Agent using Noisy Nets for exploration."""

    def __init__(
        self,
        config: ModelConfig,
        dqn_config: DQNConfig,
        env_config: EnvConfig,
    ):
        print("[DQNAgent] Initializing...")
        self.device = DEVICE
        self.action_dim = env_config.ACTION_DIM
        self.gamma = dqn_config.GAMMA
        self.use_double_dqn = dqn_config.USE_DOUBLE_DQN
        self.gradient_clip_norm = dqn_config.GRADIENT_CLIP_NORM
        self.use_noisy_nets = dqn_config.USE_NOISY_NETS
        self.use_dueling = dqn_config.USE_DUELING

        if not self.use_noisy_nets:
            # This shouldn't happen based on config, but warn if it does
            print("WARNING: DQNConfig.USE_NOISY_NETS is False. Agent expects True.")

        self.online_net = create_network(
            env_config.STATE_DIM, env_config.ACTION_DIM, config, dqn_config
        ).to(self.device)
        self.target_net = create_network(
            env_config.STATE_DIM, env_config.ACTION_DIM, config, dqn_config
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() # Target net always in eval mode

        self.optimizer = optim.AdamW(
            self.online_net.parameters(),
            lr=dqn_config.LEARNING_RATE,
            eps=dqn_config.ADAM_EPS,
            weight_decay=1e-5, # Example weight decay
        )
        # Huber loss is generally robust for Q-learning
        self.loss_fn = nn.SmoothL1Loss(reduction='none', beta=1.0) # Use 'none' for PER weights

        self._last_avg_max_q: float = 0.0

        print(f"[DQNAgent] Online Network: {type(self.online_net).__name__}")
        print(f"[DQNAgent] Using Double DQN: {self.use_double_dqn}")
        print(f"[DQNAgent] Using Dueling: {self.use_dueling}")
        print(f"[DQNAgent] Using Noisy Nets: {self.use_noisy_nets}")
        print(f"[DQNAgent] Optimizer: AdamW (LR={dqn_config.LEARNING_RATE}, EPS={dqn_config.ADAM_EPS})")
        total_params = sum(p.numel() for p in self.online_net.parameters() if p.requires_grad)
        print(f"[DQNAgent] Trainable Parameters: {total_params / 1e6:.2f} M")


    @torch.no_grad()
    def select_action(
        self,
        state: StateType,
        epsilon: float, # Epsilon is unused but kept for potential API compatibility
        valid_actions: List[ActionType]
    ) -> ActionType:
        """Selects action using the noisy online network (greedy w.r.t mean weights)."""
        if not valid_actions:
            # print("Warning: select_action called with no valid actions.")
            return 0 # Return a default action index

        state_np = ensure_numpy(state)
        state_t = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Use online network in eval mode for action selection.
        # NoisyLinear layers use mean weights in eval mode.
        self.online_net.eval()
        q_values = self.online_net(state_t)[0] # Q-values for the single state

        # Mask invalid actions
        q_values_masked = torch.full_like(q_values, -float('inf'))
        valid_action_indices = torch.tensor(valid_actions, dtype=torch.long, device=self.device)
        q_values_masked[valid_action_indices] = q_values[valid_action_indices]

        best_action = torch.argmax(q_values_masked).item()

        # Note: online_net is set back to train() mode within compute_loss/update methods

        return best_action


    def _np_batch_to_tensor(self, batch: Union[NumpyBatch, NumpyNStepBatch], is_n_step: bool) -> Union[TensorBatch, TensorNStepBatch]:
        """Converts a numpy batch (1-step or N-step) to tensors on the correct device."""
        if is_n_step:
            states, actions, rewards, next_states, dones, discounts = batch
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
            discounts_t = torch.tensor(discounts, dtype=torch.float32, device=self.device).unsqueeze(1)
            return states_t, actions_t, rewards_t, next_states_t, dones_t, discounts_t
        else:
            states, actions, rewards, next_states, dones = batch
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
            return states_t, actions_t, rewards_t, next_states_t, dones_t


    def compute_loss(
        self,
        batch: Union[NumpyBatch, NumpyNStepBatch],
        is_n_step: bool,
        is_weights: Optional[np.ndarray] = None # PER Importance Sampling weights
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes DQN loss (Huber Loss), handles N-step and PER weights."""

        # Convert Batch to Tensors
        tensor_batch = self._np_batch_to_tensor(batch, is_n_step)
        if is_n_step:
            states, actions, rewards, next_states, dones, discounts = tensor_batch
        else:
            states, actions, rewards, next_states, dones = tensor_batch
            discounts = torch.full_like(rewards, self.gamma, device=self.device) # gamma^1

        is_weights_t = None
        if is_weights is not None:
            is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Calculate Target Q-values (using Double DQN logic)
        with torch.no_grad():
            # Target net is already in eval mode
            # Select best actions for next states using the *online* network (eval mode for consistency)
            self.online_net.eval()
            online_next_q = self.online_net(next_states)
            best_next_actions = online_next_q.argmax(dim=1, keepdim=True)

            # Get Q-values for these best actions using the *target* network
            target_next_q_values = self.target_net(next_states).gather(1, best_next_actions)

            # Calculate the TD target: R + gamma^N * Q_target(s', a') * (1 - done)
            target_q = rewards + discounts * target_next_q_values * (1.0 - dones)

        # Calculate Current Q-values (train mode for gradients and noise)
        self.online_net.train() # Ensure train mode for Noisy Nets and gradients
        current_q = self.online_net(states).gather(1, actions)

        # Calculate Loss
        td_error = target_q - current_q
        elementwise_loss = self.loss_fn(current_q, target_q)

        # Apply PER weights
        loss = (is_weights_t * elementwise_loss).mean() if is_weights_t is not None else elementwise_loss.mean()

        # Update Stats (average max Q for logging - use eval for consistency)
        with torch.no_grad():
            self.online_net.eval()
            self._last_avg_max_q = self.online_net(states).max(dim=1)[0].mean().item()
            self.online_net.train() # Switch back immediately if needed elsewhere? Update handles it.

        return loss, td_error.abs().detach() # Return abs TD error for PER


    def update(self, loss: torch.Tensor) -> Optional[float]:
        """Performs one optimization step and returns gradient norm."""
        grad_norm = None
        try:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient Clipping
            if self.gradient_clip_norm > 0:
                # Ensure online_net is in train mode before clipping/stepping
                self.online_net.train()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.online_net.parameters(), max_norm=self.gradient_clip_norm
                ).item()

            self.optimizer.step()
        except Exception as e:
             print(f"ERROR during agent update/optimizer step: {e}")
             traceback.print_exc()
             # Return None or re-raise? Returning None indicates failure.
             return None

        return grad_norm

    def get_last_avg_max_q(self) -> float:
        """Returns the average max Q value computed during the last loss calculation."""
        return self._last_avg_max_q

    def update_target_network(self):
        """Copies weights from the online network to the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def get_state_dict(self) -> AgentStateDict:
        """Returns the agent's state for saving."""
        return {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: AgentStateDict):
        """Loads the agent's state from a dictionary."""
        self.online_net.load_state_dict(state_dict["online_net_state_dict"])

        if "target_net_state_dict" in state_dict:
            self.target_net.load_state_dict(state_dict["target_net_state_dict"])
        else:
            print("Warning: Target network state missing in checkpoint, copying from online.")
            self.target_net.load_state_dict(self.online_net.state_dict())

        try:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        except ValueError as e:
            print(f"Warning: Optimizer state mismatch ({e}). Resetting optimizer state.")
            # Reset optimizer if loading fails (e.g., model change)
            self.optimizer = optim.AdamW(
                self.online_net.parameters(),
                lr=DQNConfig.LEARNING_RATE, # Use current config LR
                eps=DQNConfig.ADAM_EPS,
                weight_decay=1e-5
            )
        except Exception as e:
            print(f"Warning: Error loading optimizer state: {e}. Resetting optimizer state.")
            self.optimizer = optim.AdamW(
                 self.online_net.parameters(), lr=DQNConfig.LEARNING_RATE, eps=DQNConfig.ADAM_EPS, weight_decay=1e-5
            )


        # Ensure networks are in correct mode after loading
        self.online_net.train()
        self.target_net.eval()

```

```python
# File: agent/model_factory.py
# (Largely unchanged, just cleaner print)
import torch.nn as nn
from config import ModelConfig, EnvConfig, DQNConfig
from typing import Type

from agent.networks.agent_network import AgentNetwork

def create_network(
    state_dim: int,
    action_dim: int,
    model_config: ModelConfig,
    dqn_config: DQNConfig,
) -> nn.Module:
    """Creates the AgentNetwork based on configuration."""

    print(
        f"[ModelFactory] Creating AgentNetwork (Dueling: {dqn_config.USE_DUELING}, NoisyNets Heads: {dqn_config.USE_NOISY_NETS})"
    )

    # Pass the specific sub-config ModelConfig.Network
    return AgentNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        config=model_config.Network, # Pass the Network sub-config
        env_config=EnvConfig,        # AgentNetwork needs EnvConfig
        dueling=dqn_config.USE_DUELING,
        use_noisy=dqn_config.USE_NOISY_NETS,
    )

```

```python
# File: agent/replay_buffer/uniform_buffer.py
# (No structural changes, cleanup comments)
import random
import numpy as np
from collections import deque
from typing import Deque, Tuple, Optional, Any, Dict, Union
from .base_buffer import ReplayBufferBase
from utils.types import (
    Transition, StateType, ActionType, NumpyBatch, NumpyNStepBatch
)
from utils.helpers import save_object, load_object

class UniformReplayBuffer(ReplayBufferBase):
    """Standard uniform experience replay buffer."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool,
        **kwargs, # Accept potential n_step_discount from NStepWrapper
    ):
        n_step_discount = kwargs.get('n_step_discount')
        transition = Transition(state=state, action=action, reward=reward,
                                next_state=next_state, done=done, n_step_discount=n_step_discount)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Optional[Union[NumpyBatch, NumpyNStepBatch]]:
        if len(self.buffer) < batch_size:
            return None

        batch_transitions = random.sample(self.buffer, batch_size)
        is_n_step = batch_transitions[0].n_step_discount is not None

        if is_n_step:
            s, a, rn, nsn, dn, gamma_n = zip(*[
                (t.state, t.action, t.reward, t.next_state, t.done, t.n_step_discount)
                for t in batch_transitions
            ])
            states_np = np.array(s, dtype=np.float32)
            actions_np = np.array(a, dtype=np.int64)
            rewards_np = np.array(rn, dtype=np.float32)
            next_states_np = np.array(nsn, dtype=np.float32)
            dones_np = np.array(dn, dtype=np.float32)
            discounts_np = np.array(gamma_n, dtype=np.float32)
            return states_np, actions_np, rewards_np, next_states_np, dones_np, discounts_np
        else:
            s, a, r, ns, d = zip(*[
                (t.state, t.action, t.reward, t.next_state, t.done)
                for t in batch_transitions
            ])
            states_np = np.array(s, dtype=np.float32)
            actions_np = np.array(a, dtype=np.int64)
            rewards_np = np.array(r, dtype=np.float32)
            next_states_np = np.array(ns, dtype=np.float32)
            dones_np = np.array(d, dtype=np.float32)
            return states_np, actions_np, rewards_np, next_states_np, dones_np

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        pass # No-op for uniform buffer

    def set_beta(self, beta: float):
        pass # No-op for uniform buffer

    def flush_pending(self):
        pass # No-op for uniform buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def get_state(self) -> Dict[str, Any]:
        # Convert deque to list for robust serialization
        return {"buffer": list(self.buffer)}

    def load_state_from_data(self, state: Dict[str, Any]):
        saved_buffer_list = state.get("buffer", [])
        self.buffer = deque(saved_buffer_list, maxlen=self.capacity)
        print(f"[UniformReplayBuffer] Loaded {len(self.buffer)} transitions.")

    def save_state(self, filepath: str):
        state = self.get_state()
        save_object(state, filepath)

    def load_state(self, filepath: str):
        state = load_object(filepath)
        self.load_state_from_data(state)
```

```python
# File: agent/replay_buffer/buffer_utils.py
# (No structural changes, cleaner print statements)
from config import BufferConfig, DQNConfig
from .base_buffer import ReplayBufferBase
from .uniform_buffer import UniformReplayBuffer
from .prioritized_buffer import PrioritizedReplayBuffer
from .nstep_buffer import NStepBufferWrapper

def create_replay_buffer(
    config: BufferConfig, dqn_config: DQNConfig
) -> ReplayBufferBase:
    """Factory function to create the replay buffer based on configuration."""

    print("[BufferFactory] Creating replay buffer...")
    print(f"  Type: {'Prioritized' if config.USE_PER else 'Uniform'}")
    print(f"  Capacity: {config.REPLAY_BUFFER_SIZE / 1e6:.1f}M")
    if config.USE_PER:
         print(f"  PER alpha={config.PER_ALPHA}, eps={config.PER_EPSILON}")

    if config.USE_PER:
        core_buffer = PrioritizedReplayBuffer(
            capacity=config.REPLAY_BUFFER_SIZE,
            alpha=config.PER_ALPHA,
            epsilon=config.PER_EPSILON,
        )
    else:
        core_buffer = UniformReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)

    if config.USE_N_STEP and config.N_STEP > 1:
        print(f"  N-Step Wrapper: Enabled (N={config.N_STEP}, gamma={dqn_config.GAMMA})")
        final_buffer = NStepBufferWrapper(
            wrapped_buffer=core_buffer,
            n_step=config.N_STEP,
            gamma=dqn_config.GAMMA,
        )
    else:
         print(f"  N-Step Wrapper: Disabled")
         final_buffer = core_buffer

    print(f"[BufferFactory] Final buffer type: {type(final_buffer).__name__}")
    return final_buffer
```

```python
# File: agent/replay_buffer/prioritized_buffer.py
# (No structural changes, cleanup comments, improved error message)
import random
import numpy as np
from typing import Optional, Tuple, Any, Dict, Union, List
from .base_buffer import ReplayBufferBase
from .sum_tree import SumTree
from utils.types import (
    Transition, StateType, ActionType, NumpyBatch, PrioritizedNumpyBatch,
    NumpyNStepBatch, PrioritizedNumpyNStepBatch
)
from utils.helpers import save_object, load_object

class PrioritizedReplayBuffer(ReplayBufferBase):
    """Prioritized Experience Replay (PER) buffer using a SumTree."""

    def __init__(self, capacity: int, alpha: float, epsilon: float):
        super().__init__(capacity)
        self.tree = SumTree(capacity)
        self.alpha = alpha      # Controls prioritization strength (0=uniform, 1=full)
        self.epsilon = epsilon  # Small value added to priorities to ensure non-zero probability
        self.beta = 0.0         # Importance sampling exponent (annealed externally)
        self.max_priority = 1.0 # Initial max priority

    def push(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool,
        **kwargs, # Accept potential n_step_discount
    ):
        """Adds new experience with maximum priority."""
        n_step_discount = kwargs.get('n_step_discount')
        transition = Transition(state=state, action=action, reward=reward,
                                next_state=next_state, done=done, n_step_discount=n_step_discount)
        # Add with max priority initially, will be updated after first training step
        self.tree.add(self.max_priority, transition)

    def sample(
        self, batch_size: int
    ) -> Optional[Union[PrioritizedNumpyBatch, PrioritizedNumpyNStepBatch]]:
        """Samples batch using priorities, calculates IS weights."""
        if len(self) < batch_size:
            return None

        batch_data: List[Transition] = []
        indices = np.empty(batch_size, dtype=np.int64) # Tree indices
        priorities = np.empty(batch_size, dtype=np.float64)
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            s = max(1e-9, s) # Avoid s=0 issues

            idx, p, data = self.tree.get(s)

            # Retry sampling if data is invalid (should be rare with proper init)
            retries = 0
            max_retries = 5
            while not isinstance(data, Transition) and retries < max_retries:
                # Resample from the entire range if the segment failed
                s = random.uniform(1e-9, self.tree.total())
                idx, p, data = self.tree.get(s)
                retries += 1

            if not isinstance(data, Transition):
                print(f"ERROR: PER sample failed to get valid data after {max_retries} retries (total entries: {len(self)}, tree total: {self.tree.total():.4f}). Skipping batch.")
                return None # Return None if any sample fails

            priorities[i] = p
            batch_data.append(data)
            indices[i] = idx

        sampling_probabilities = priorities / self.tree.total()
        sampling_probabilities = np.maximum(sampling_probabilities, 1e-9) # Epsilon for stability

        # Calculate Importance Sampling (IS) weights
        is_weights = np.power(len(self) * sampling_probabilities, -self.beta)
        is_weights /= (is_weights.max() + 1e-9) # Normalize weights

        # Check if N-step based on first item
        is_n_step = batch_data[0].n_step_discount is not None

        if is_n_step:
            s, a, rn, nsn, dn, gamma_n = zip(*[
                (t.state, t.action, t.reward, t.next_state, t.done, t.n_step_discount)
                for t in batch_data
            ])
            states_np = np.array(s, dtype=np.float32)
            actions_np = np.array(a, dtype=np.int64)
            rewards_np = np.array(rn, dtype=np.float32)
            next_states_np = np.array(nsn, dtype=np.float32)
            dones_np = np.array(dn, dtype=np.float32)
            discounts_np = np.array(gamma_n, dtype=np.float32)
            batch_tuple = (states_np, actions_np, rewards_np, next_states_np, dones_np, discounts_np)
            return batch_tuple, indices, is_weights.astype(np.float32)
        else:
            s, a, r, ns, d = zip(*[
                (t.state, t.action, t.reward, t.next_state, t.done)
                for t in batch_data
            ])
            states_np = np.array(s, dtype=np.float32)
            actions_np = np.array(a, dtype=np.int64)
            rewards_np = np.array(r, dtype=np.float32)
            next_states_np = np.array(ns, dtype=np.float32)
            dones_np = np.array(d, dtype=np.float32)
            batch_tuple = (states_np, actions_np, rewards_np, next_states_np, dones_np)
            return batch_tuple, indices, is_weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Updates priorities of experiences at given tree indices using TD errors."""
        if len(indices) != len(priorities):
            print(f"Error: Mismatch indices ({len(indices)}) vs priorities ({len(priorities)}) in PER update")
            return

        # Use absolute TD error for priority, add epsilon, raise to alpha
        priorities = np.abs(priorities) + self.epsilon
        priorities = np.power(priorities, self.alpha)

        for idx, priority in zip(indices, priorities):
            # Index should be leaf node index from sampling
            if not (self.tree.capacity - 1 <= idx < 2 * self.tree.capacity - 1):
                # This might happen if buffer wraps around. Log silently or with low severity.
                # print(f"Debug: Attempting update on invalid tree index {idx}. Skipping.")
                continue
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority) # Update max priority seen

    def set_beta(self, beta: float):
        self.beta = beta

    def flush_pending(self):
        pass # No-op for this buffer

    def __len__(self) -> int:
        return self.tree.n_entries

    def get_state(self) -> Dict[str, Any]:
        """Return state for saving."""
        return {
            "tree_nodes": self.tree.tree.copy(),
            "tree_data": self.tree.data.copy(), # Actual transition data
            "tree_write_ptr": self.tree.write_ptr,
            "tree_n_entries": self.tree.n_entries,
            "max_priority": self.max_priority,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            # Beta is transient, set by trainer
        }

    def load_state_from_data(self, state: Dict[str, Any]):
        """Load state from dictionary."""
        if "tree_nodes" not in state or "tree_data" not in state:
            print("Error: Invalid PER state format during load. Skipping.")
            return

        loaded_capacity = len(state["tree_data"])
        if loaded_capacity != self.capacity:
            print(f"Warning: Loaded PER capacity ({loaded_capacity}) != current buffer capacity ({self.capacity}). Recreating tree structure.")
            # Recreate tree with current capacity and load data partially
            self.tree = SumTree(self.capacity)
            num_to_load = min(loaded_capacity, self.capacity)
            # Simple load - just copy data, priorities will reset.
            # A complex load would rebuild tree priorities, harder if capacity changed.
            self.tree.data[:num_to_load] = state["tree_data"][:num_to_load]
            self.tree.write_ptr = state.get("tree_write_ptr", 0) % self.capacity # Ensure valid ptr
            self.tree.n_entries = min(state.get("tree_n_entries", 0), self.capacity)
            self.tree.tree.fill(0) # Clear old priorities
            self.max_priority = 1.0 # Reset max priority
            print(f"[PrioritizedReplayBuffer] Loaded {self.tree.n_entries} transitions (priorities reset due to capacity mismatch).")

        else:
            # Capacities match, load everything
            self.tree.tree = state["tree_nodes"]
            self.tree.data = state["tree_data"]
            self.tree.write_ptr = state.get("tree_write_ptr", 0)
            self.tree.n_entries = state.get("tree_n_entries", 0)
            self.max_priority = state.get("max_priority", 1.0)
            print(f"[PrioritizedReplayBuffer] Loaded {self.tree.n_entries} transitions.")

        # Load config params if they exist in save, otherwise keep current config
        self.alpha = state.get("alpha", self.alpha)
        self.epsilon = state.get("epsilon", self.epsilon)

    def save_state(self, filepath: str):
        state = self.get_state()
        save_object(state, filepath)

    def load_state(self, filepath: str):
        state = load_object(filepath)
        self.load_state_from_data(state)

```

```python
# File: agent/replay_buffer/base_buffer.py
# (No changes needed)
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict
import numpy as np
from utils.types import StateType, ActionType

class ReplayBufferBase(ABC):
    """Abstract base class for all replay buffers."""

    def __init__(self, capacity: int):
        self.capacity = capacity

    @abstractmethod
    def push(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool,
        **kwargs # Allow passing extra info like n_step_discount
    ):
        """Add a new experience to the buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Optional[Any]: # Return type depends on PER/NStep
        """Sample a batch of experiences from the buffer."""
        pass

    @abstractmethod
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for PER (no-op for uniform buffer)."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        pass

    @abstractmethod
    def set_beta(self, beta: float):
        """Set the beta value for PER IS weights (no-op for uniform buffer)."""
        pass

    @abstractmethod
    def flush_pending(self):
        """Process any pending transitions (e.g., for N-step)."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return the buffer's state as a dictionary suitable for saving."""
        pass

    @abstractmethod
    def load_state_from_data(self, state: Dict[str, Any]):
        """Load the buffer's state from a dictionary."""
        pass

    @abstractmethod
    def save_state(self, filepath: str):
        """Save the buffer's state to a file."""
        pass

    @abstractmethod
    def load_state(self, filepath: str):
        """Load the buffer's state from a file."""
        pass
```

```python
# File: agent/replay_buffer/nstep_buffer.py
# (No structural changes, cleanup comments)
from collections import deque
import numpy as np
from typing import Deque, Tuple, Optional, Any, Dict, List
from .base_buffer import ReplayBufferBase
from utils.types import Transition, StateType, ActionType
from utils.helpers import save_object, load_object

class NStepBufferWrapper(ReplayBufferBase):
    """
    Wraps another buffer to implement N-step returns.
    Calculates N-step transitions and pushes them to the wrapped buffer.
    """

    def __init__(self, wrapped_buffer: ReplayBufferBase, n_step: int, gamma: float):
        super().__init__(wrapped_buffer.capacity) # Capacity managed by wrapped buffer
        if n_step <= 0: raise ValueError("N-step must be positive")
        self.wrapped_buffer = wrapped_buffer
        self.n_step = n_step
        self.gamma = gamma
        # Temporary deque for raw (s, a, r, ns, d) tuples
        self.n_step_deque: Deque[Tuple[StateType, ActionType, float, StateType, bool]] = deque(maxlen=n_step)

    def _calculate_n_step_transition(self, current_deque_list: List[Tuple]) -> Optional[Transition]:
        """Calculates the N-step return from a list copy of the deque."""
        if not current_deque_list: return None

        n_step_reward = 0.0
        discount_accum = 1.0
        effective_n = len(current_deque_list)

        state_0, action_0 = current_deque_list[0][0], current_deque_list[0][1]

        for i in range(effective_n):
            s, a, r, ns, d = current_deque_list[i]
            n_step_reward += discount_accum * r

            if d: # Episode terminated within these N steps
                n_step_next_state = ns # Terminal state
                n_step_done = True
                n_step_discount = self.gamma**(i + 1) # Discount for Q(s_terminal) is effectively 0 later, but store factor
                return Transition(state=state_0, action=action_0, reward=n_step_reward,
                                next_state=n_step_next_state, done=n_step_done, n_step_discount=n_step_discount)

            discount_accum *= self.gamma

        # Loop completed without terminal state
        n_step_next_state = current_deque_list[-1][3] # next_state from Nth transition
        n_step_done = current_deque_list[-1][4]       # done flag from Nth transition
        n_step_discount = self.gamma**effective_n     # Discount factor for Q(s_N) is gamma^N

        return Transition(state=state_0, action=action_0, reward=n_step_reward,
                          next_state=n_step_next_state, done=n_step_done, n_step_discount=n_step_discount)

    def push(self, state: StateType, action: ActionType, reward: float, next_state: StateType, done: bool):
        """Adds raw transition, processes N-step if possible, pushes to wrapped buffer."""
        current_transition = (state, action, reward, next_state, done)
        self.n_step_deque.append(current_transition)

        if len(self.n_step_deque) < self.n_step:
            if done: self._flush_on_done() # Process partial if episode ends early
            return # Wait for more steps

        # Deque has N items, calculate N-step transition from the oldest start
        n_step_transition = self._calculate_n_step_transition(list(self.n_step_deque))

        if n_step_transition:
            self.wrapped_buffer.push(
                state=n_step_transition.state, action=n_step_transition.action, reward=n_step_transition.reward,
                next_state=n_step_transition.next_state, done=n_step_transition.done,
                n_step_discount=n_step_transition.n_step_discount # Pass calculated discount
            )

        if done: # If the *newly added* transition was terminal, flush remaining starts
            self._flush_on_done()


    def _flush_on_done(self):
        """Processes remaining partial transitions when an episode ends."""
        # The deque contains transitions leading up to 'done'.
        # We need to process transitions starting *after* the initial state
        # of the first transition processed in the last `push` call.
        temp_deque = list(self.n_step_deque)
        while len(temp_deque) > 1: # Stop when only the 'done' transition remains
            temp_deque.pop(0) # Remove the already processed starting state
            n_step_transition = self._calculate_n_step_transition(temp_deque)
            if n_step_transition:
                self.wrapped_buffer.push(
                    state=n_step_transition.state, action=n_step_transition.action, reward=n_step_transition.reward,
                    next_state=n_step_transition.next_state, done=n_step_transition.done,
                    n_step_discount=n_step_transition.n_step_discount
                )
        self.n_step_deque.clear() # Clear after flushing


    def sample(self, batch_size: int) -> Any:
        return self.wrapped_buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.wrapped_buffer.update_priorities(indices, priorities)

    def set_beta(self, beta: float):
        if hasattr(self.wrapped_buffer, 'set_beta'):
            self.wrapped_buffer.set_beta(beta)

    def flush_pending(self):
        """Processes and pushes any remaining transitions before exit/save."""
        print(f"[NStepWrapper] Flushing {len(self.n_step_deque)} pending transitions on cleanup.")
        temp_deque = list(self.n_step_deque)
        while len(temp_deque) > 0:
            n_step_transition = self._calculate_n_step_transition(temp_deque)
            if n_step_transition:
                 self.wrapped_buffer.push(
                    state=n_step_transition.state, action=n_step_transition.action, reward=n_step_transition.reward,
                    next_state=n_step_transition.next_state, done=n_step_transition.done,
                    n_step_discount=n_step_transition.n_step_discount
                 )
            temp_deque.pop(0)
        self.n_step_deque.clear()

        if hasattr(self.wrapped_buffer, 'flush_pending'):
            self.wrapped_buffer.flush_pending()

    def __len__(self) -> int:
        # Length is the number of processed N-step transitions in wrapped buffer
        return len(self.wrapped_buffer)

    def get_state(self) -> Dict[str, Any]:
        """Return state for saving."""
        return {
            "n_step_deque": list(self.n_step_deque), # Pending raw transitions
            "wrapped_buffer_state": self.wrapped_buffer.get_state(),
        }

    def load_state_from_data(self, state: Dict[str, Any]):
        """Load state from dictionary."""
        saved_deque_list = state.get("n_step_deque", [])
        self.n_step_deque = deque(saved_deque_list, maxlen=self.n_step)
        print(f"[NStepWrapper] Loaded {len(self.n_step_deque)} pending transitions.")

        wrapped_state = state.get("wrapped_buffer_state")
        if wrapped_state is not None:
            self.wrapped_buffer.load_state_from_data(wrapped_state)
        else:
            print("[NStepWrapper] Warning: No wrapped buffer state found in saved data.")

    def save_state(self, filepath: str):
        state = self.get_state()
        save_object(state, filepath)

    def load_state(self, filepath: str):
        try:
            state = load_object(filepath)
            self.load_state_from_data(state)
        except FileNotFoundError:
            print(f"[NStepWrapper] Load failed: File not found at {filepath}. Starting empty.")
        except Exception as e:
            print(f"[NStepWrapper] Load failed: {e}. Starting empty.")

```

```python
# File: agent/replay_buffer/sum_tree.py
# (No structural changes, cleanup comments, fixed potential float comparison issue)
import numpy as np

class SumTree:
    """
    Simple SumTree implementation using numpy arrays for Prioritized Experience Replay.
    Tree structure: [root] [internal nodes] ... [leaves]
    Array size = 2 * capacity - 1. Leaves start at index capacity - 1.
    """

    def __init__(self, capacity: int):
        if capacity <= 0 or not isinstance(capacity, int):
            raise ValueError("SumTree capacity must be a positive integer")
        self.capacity = capacity
        # Use float64 for priority sums to minimize precision errors
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        # Holds the actual experience data (e.g., Transition objects)
        self.data = np.zeros(capacity, dtype=object)
        self.write_ptr = 0  # Current position to write new data
        self.n_entries = 0  # Number of valid entries currently in the buffer

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index for a given cumulative priority value s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree): # Leaf node
            return idx

        # Use a small tolerance for floating point comparison
        if s <= self.tree[left] + 1e-8:
            return self._retrieve(left, s)
        else:
            # Ensure s subtraction doesn't go negative due to fp errors
            s_new = max(0.0, s - self.tree[left])
            return self._retrieve(right, s_new)

    def total(self) -> float:
        """Get the total priority sum (value of the root node)."""
        return self.tree[0]

    def add(self, priority: float, data: object):
        """Add new experience, overwriting oldest if buffer is full."""
        priority = max(abs(priority), 1e-6) # Ensure positive priority

        tree_idx = self.write_ptr + self.capacity - 1 # Index in the tree array
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority) # Update priority in tree

        self.write_ptr = (self.write_ptr + 1) % self.capacity # Advance ptr with wrap-around

        if self.n_entries < self.capacity:
            self.n_entries += 1 # Increment count until full

    def update(self, tree_idx: int, priority: float):
        """Update priority of an experience at a given tree index."""
        priority = max(abs(priority), 1e-6) # Ensure positive priority

        # Validate index refers to a leaf node
        if not (self.capacity - 1 <= tree_idx < 2 * self.capacity - 1):
            # print(f"Warning: Invalid tree index {tree_idx} for update. Capacity {self.capacity}. Skipping.")
            return # Silently skip invalid index updates

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up the tree if it's significant
        if abs(change) > 1e-9 and tree_idx > 0:
             self._propagate(tree_idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        """Sample an experience based on cumulative priority s.
           Returns: (tree_idx, priority, data)
        """
        if self.total() <= 0 or self.n_entries == 0:
            # print("Warning: Sampling from empty or zero-priority SumTree.")
            return 0, 0.0, None # Return valid types even if empty

        # Clip s to valid range [epsilon, total]
        s = np.clip(s, 1e-9, self.total())

        idx = self._retrieve(0, s)      # Leaf node index in the tree array
        data_idx = idx - self.capacity + 1 # Corresponding index in data array

        # Validate data_idx before access (important if buffer not full)
        if not (0 <= data_idx < self.n_entries):
            # This can happen due to floating point issues near boundaries or
            # if sampling races with adding near capacity.
            # Fallback: return the last valid entry added?
            # print(f"Warning: SumTree get resulted in invalid data index {data_idx} (n_entries={self.n_entries}, total_p={self.total():.4f}, s={s:.4f}). Falling back.")
            if self.n_entries > 0:
                last_valid_data_idx = (self.write_ptr - 1 + self.capacity) % self.capacity
                last_valid_tree_idx = last_valid_data_idx + self.capacity - 1
                priority = self.tree[last_valid_tree_idx] if (self.capacity - 1 <= last_valid_tree_idx < 2 * self.capacity - 1) else 0.0
                return (last_valid_tree_idx, priority, self.data[last_valid_data_idx])
            else: # Truly empty
                return 0, 0.0, None

        # Return tree_idx, priority from tree, data from data array
        return (idx, self.tree[idx], self.data[data_idx])

    def __len__(self) -> int:
        """Number of valid entries in the buffer."""
        return self.n_entries

```

```python
# File: agent/networks/noisy_layer.py
# (No changes needed, already clean)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for Noisy Network (Factorised Gaussian Noise).
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable weights and biases (mean parameters)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learnable noise parameters (standard deviation parameters)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Non-learnable noise buffers
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise() # Initial noise generation

    def reset_parameters(self):
        """Initialize mean and std parameters."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)

        # Initialize sigma parameters (std dev)
        nn.init.constant_(self.weight_sigma, self.std_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Generate new noise samples using Factorised Gaussian noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Outer product for weight noise, direct sample for bias noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate noise tensor with sign-sqrt transformation."""
        x = torch.randn(size, device=self.weight_mu.device) # Noise on same device
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters if training, mean parameters otherwise."""
        if self.training:
            # Sample noise is implicitly used via weight_epsilon, bias_epsilon
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            # Reset noise *after* use in forward pass for next iteration?
            # Or reset in train() method? Resetting in train() is common.
        else:
            # Use mean parameters during evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def train(self, mode: bool = True):
        """Override train mode to reset noise when entering training."""
        if self.training is False and mode is True: # If switching from eval to train
             self.reset_noise()
        super().train(mode)

```

```python
# File: agent/networks/agent_network.py
# (No structural changes, cleanup comments)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from config import ModelConfig, EnvConfig
from typing import Tuple, List, Type

from .noisy_layer import NoisyLinear

class AgentNetwork(nn.Module):
    """
    Agent Network: CNN (Grid) + MLP (Shape) -> Fused MLP -> Dueling Heads (Noisy optional).
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: ModelConfig.Network, # The specific network sub-config
        env_config: EnvConfig,
        dueling: bool,
        use_noisy: bool, # Use NoisyLinear in final heads?
    ):
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        self.env_config = env_config
        self.config = config
        self.use_noisy = use_noisy

        # Calculate expected feature dimensions from config
        self.grid_h = env_config.ROWS
        self.grid_w = env_config.COLS
        self.grid_feat_per_cell = env_config.GRID_FEATURES_PER_CELL
        self.expected_grid_flat_dim = self.grid_h * self.grid_w * self.grid_feat_per_cell

        self.num_shape_slots = env_config.NUM_SHAPE_SLOTS
        self.shape_feat_per_shape = env_config.SHAPE_FEATURES_PER_SHAPE
        self.expected_shape_flat_dim = self.num_shape_slots * self.shape_feat_per_shape

        self.expected_total_dim = self.expected_grid_flat_dim + self.expected_shape_flat_dim
        if state_dim != self.expected_total_dim:
            raise ValueError(
                f"AgentNetwork init: State dimension mismatch! "
                f"Input state_dim ({state_dim}) != calculated expected_total_dim ({self.expected_total_dim}). "
                f"Grid={self.expected_grid_flat_dim}, Shape={self.expected_shape_flat_dim}"
            )

        print(f"[AgentNetwork] Initializing (Noisy Heads: {self.use_noisy}):")
        print(f"  Input Dim: {state_dim} (Grid: {self.expected_grid_flat_dim}, Shape: {self.expected_shape_flat_dim})")

        # --- 1. CNN Branch (Grid Features) ---
        conv_layers: List[nn.Module] = []
        current_channels = self.grid_feat_per_cell
        h, w = self.grid_h, self.grid_w
        for i, out_channels in enumerate(config.CONV_CHANNELS):
            conv_layers.append(
                nn.Conv2d(
                    current_channels, out_channels,
                    kernel_size=config.CONV_KERNEL_SIZE, stride=config.CONV_STRIDE, padding=config.CONV_PADDING,
                    bias=not config.USE_BATCHNORM_CONV,
                )
            )
            if config.USE_BATCHNORM_CONV: conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(config.CONV_ACTIVATION())
            conv_layers.append(nn.MaxPool2d(kernel_size=config.POOL_KERNEL_SIZE, stride=config.POOL_STRIDE))
            current_channels = out_channels
            # Calculate output size after conv and pool (for info print)
            h = (h + 2*config.CONV_PADDING - config.CONV_KERNEL_SIZE)//config.CONV_STRIDE + 1
            w = (w + 2*config.CONV_PADDING - config.CONV_KERNEL_SIZE)//config.CONV_STRIDE + 1
            h = (h - config.POOL_KERNEL_SIZE)//config.POOL_STRIDE + 1
            w = (w - config.POOL_KERNEL_SIZE)//config.POOL_STRIDE + 1

        self.conv_base = nn.Sequential(*conv_layers)
        self.conv_out_size = self._get_conv_out_size((self.grid_feat_per_cell, self.grid_h, self.grid_w))
        print(f"  CNN Output Dim (HxWxC): ({h}x{w}x{current_channels}) -> Flattened: {self.conv_out_size}")

        # --- 2. Shape Feature Branch (MLP) ---
        shape_mlp_layers: List[nn.Module] = []
        shape_mlp_layers.append(nn.Linear(self.expected_shape_flat_dim, config.SHAPE_MLP_HIDDEN_DIM))
        shape_mlp_layers.append(config.SHAPE_MLP_ACTIVATION())
        self.shape_mlp = nn.Sequential(*shape_mlp_layers)
        shape_mlp_out_dim = config.SHAPE_MLP_HIDDEN_DIM
        print(f"  Shape MLP Output Dim: {shape_mlp_out_dim}")

        # --- 3. Combined Feature Fusion (MLP) ---
        combined_features_dim = self.conv_out_size + shape_mlp_out_dim
        print(f"  Combined Features Dim (CNN_flat + Shape_MLP): {combined_features_dim}")

        fusion_layers: List[nn.Module] = []
        current_fusion_dim = combined_features_dim
        fusion_linear_layer_class = nn.Linear # Use standard Linear for fusion part
        for i, hidden_dim in enumerate(config.COMBINED_FC_DIMS):
            fusion_layers.append(fusion_linear_layer_class(current_fusion_dim, hidden_dim))
            if config.USE_BATCHNORM_FC: fusion_layers.append(nn.BatchNorm1d(hidden_dim))
            fusion_layers.append(config.COMBINED_ACTIVATION())
            if config.DROPOUT_FC > 0: fusion_layers.append(nn.Dropout(config.DROPOUT_FC))
            current_fusion_dim = hidden_dim

        self.fusion_mlp = nn.Sequential(*fusion_layers)
        head_input_dim = current_fusion_dim
        print(f"  Fusion MLP Output Dim (Input to Heads): {head_input_dim}")

        # --- 4. Final Output Head(s) ---
        head_linear_layer_class = NoisyLinear if self.use_noisy else nn.Linear

        if self.dueling:
            self.value_head = nn.Sequential(head_linear_layer_class(head_input_dim, 1))
            self.advantage_head = nn.Sequential(head_linear_layer_class(head_input_dim, action_dim))
            print(f"  Using Dueling Heads ({head_linear_layer_class.__name__})")
        else:
            self.output_head = nn.Sequential(head_linear_layer_class(head_input_dim, action_dim))
            print(f"  Using Single Output Head ({head_linear_layer_class.__name__})")


    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        """Helper to calculate the flattened output size of the conv base."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape) # Batch size 1
            output = self.conv_base(dummy_input)
            return int(np.prod(output.size()[1:])) # Flattened size (C*H*W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.expected_total_dim:
            raise ValueError(f"AgentNetwork forward: Invalid input shape {x.shape}. Expected [B, {self.expected_total_dim}].")
        batch_size = x.size(0)

        # Split input features
        grid_features_flat = x[:, :self.expected_grid_flat_dim]
        shape_features_flat = x[:, self.expected_grid_flat_dim:]

        # Process Grid Features (CNN)
        grid_features_reshaped = grid_features_flat.view(batch_size, self.grid_feat_per_cell, self.grid_h, self.grid_w)
        conv_output = self.conv_base(grid_features_reshaped)
        conv_output_flat = conv_output.view(batch_size, -1)

        # Process Shape Features (MLP)
        shape_output = self.shape_mlp(shape_features_flat)

        # Feature Fusion
        combined_features = torch.cat((conv_output_flat, shape_output), dim=1)
        fused_output = self.fusion_mlp(combined_features)

        # Output Heads
        if self.dueling:
            value = self.value_head(fused_output)
            advantage = self.advantage_head(fused_output)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True)) # Combine V + (A - mean(A))
        else:
            q_values = self.output_head(fused_output)

        return q_values

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers within the network."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

```

```python
# File: environment/game_state.py
# (No structural changes, minor cleanup)
import time
import numpy as np
from typing import List, Optional, Tuple
from collections import deque
from typing import Deque

from .grid import Grid
from .shape import Shape
from config import EnvConfig, RewardConfig

class GameState:
    def __init__(self):
        self.grid = Grid()
        self.shapes: List[Optional[Shape]] = [Shape() for _ in range(EnvConfig.NUM_SHAPE_SLOTS)]
        self.score = 0.0  # Cumulative RL reward for this episode
        self.game_score = 0 # Game-specific score for this episode
        self.lines_cleared_this_episode = 0 # Lines cleared this episode
        self.blink_time = 0.0 # Visual effect timer
        self.last_time = time.time() # For calculating dt
        self.freeze_time = 0.0 # Timer preventing actions after line clear
        self.game_over = False
        self._last_action_valid = True # Track if last placement was valid
        self.rewards = RewardConfig # Access reward values from config

    def reset(self) -> np.ndarray:
        self.grid = Grid()
        self.shapes = [Shape() for _ in range(EnvConfig.NUM_SHAPE_SLOTS)]
        self.score = 0.0
        self.game_score = 0
        self.lines_cleared_this_episode = 0
        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.game_over = False
        self._last_action_valid = True
        self.last_time = time.time()
        return self.get_state()

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices."""
        if self.game_over or self.freeze_time > 0:
            return []
        acts = []
        locations_per_shape = self.grid.rows * self.grid.cols
        for i, sh in enumerate(self.shapes):
            if not sh: continue # Skip empty slots
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    if self.grid.can_place(sh, r, c):
                        action_index = i * locations_per_shape + (r * self.grid.cols + c)
                        acts.append(action_index)
        return acts

    def is_over(self) -> bool:
        return self.game_over

    def is_frozen(self) -> bool:
        return self.freeze_time > 0

    def decode_act(self, a: int) -> Tuple[int, int, int]:
        """Decodes action index into (shape_slot_index, row, col)."""
        locations_per_shape = self.grid.rows * self.grid.cols
        s_idx = a // locations_per_shape
        pos_idx = a % locations_per_shape
        rr = pos_idx // self.grid.cols
        cc = pos_idx % self.grid.cols
        return s_idx, rr, cc

    def _update_timers(self):
        """Updates internal timers based on elapsed time."""
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        self.freeze_time = max(0, self.freeze_time - dt)
        self.blink_time = max(0, self.blink_time - dt)

    def _handle_invalid_placement(self) -> float:
        """Handles logic for an invalid placement attempt."""
        self._last_action_valid = False
        reward = self.rewards.PENALTY_INVALID_MOVE
        # Check if game should end because no more valid moves exist *at all*
        if not self.valid_actions():
            self.game_over = True
            self.freeze_time = 1.0 # Freeze on game over visual
            reward += self.rewards.PENALTY_GAME_OVER
        return reward

    def _handle_valid_placement(self, shp: Shape, s_idx: int, rr: int, cc: int) -> float:
        """Handles logic for a valid placement."""
        self._last_action_valid = True
        reward = 0.0
        reward += self.rewards.REWARD_PLACE_PER_TRI * len(shp.triangles)
        self.game_score += len(shp.triangles) # Game score based on tris placed

        self.grid.place(shp, rr, cc)
        self.shapes[s_idx] = None # Remove placed shape

        # Clear lines and get reward/score
        lines_cleared, triangles_in_cleared_lines = self.grid.clear_filled_rows()
        self.lines_cleared_this_episode += lines_cleared

        if lines_cleared == 1: reward += self.rewards.REWARD_CLEAR_1
        elif lines_cleared == 2: reward += self.rewards.REWARD_CLEAR_2
        elif lines_cleared >= 3: reward += self.rewards.REWARD_CLEAR_3PLUS

        if triangles_in_cleared_lines > 0:
            self.game_score += triangles_in_cleared_lines * 2 # Bonus game score for clearing
            self.blink_time = 0.5 # Visual effect
            self.freeze_time = 0.5 # Short pause after clear

        # Hole penalty
        num_holes = self.grid.count_holes()
        reward += num_holes * self.rewards.PENALTY_HOLE_PER_HOLE

        