# File: main_pygame.py
import sys
import pygame
import numpy as np
import os
import time
import traceback
import torch  # Import torch for dummy tensor
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
    TensorBoardConfig,
    DEVICE,
    RANDOM_SEED,
    BUFFER_SAVE_PATH,
    MODEL_SAVE_PATH,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    get_config_dict,  # Import helper
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
from stats.stats_recorder import StatsRecorderBase  # Base class

# from stats.simple_stats_recorder import SimpleStatsRecorder # Not directly used now
from stats.tensorboard_logger import TensorBoardStatsRecorder  # Use TensorBoard
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

        # --- Get flatten config dict for logging ---
        self.config_dict = get_config_dict()

        # --- Ensure log/checkpoint directories exist for this run ---
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(BUFFER_SAVE_PATH), exist_ok=True)
        os.makedirs(self.tensorboard_config.LOG_DIR, exist_ok=True)
        print(f"TensorBoard logs will be saved to: {self.tensorboard_config.LOG_DIR}")
        print(
            f"Checkpoints/Buffer will be saved to: {os.path.dirname(MODEL_SAVE_PATH)}"
        )

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

        # --- Init RL components (split for graph logging) ---
        self._initialize_rl_components()  # This now orchestrates the split init

        # Init Renderer
        self.renderer = UIRenderer(self.screen, self.vis_config)

        print("Initialization Complete. Ready to start.")
        print("--- To view logs, run in terminal: ---")
        print(f"--- tensorboard --logdir {os.path.abspath(BASE_LOG_DIR)} ---")

    def _initialize_envs(self) -> List[GameState]:
        # (No changes from previous version)
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

    # --- Split RL component initialization ---
    def _initialize_rl_components_pre_stats(self):
        """Initializes envs, agent, buffer before stats recorder."""
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

    def _initialize_stats_recorder(self):
        """Creates the TensorBoard recorder, logs graph and hparams."""
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"Warn: Error closing prev stats recorder: {e}")

        avg_window = self.stats_config.STATS_AVG_WINDOW
        console_log_freq = self.stats_config.CONSOLE_LOG_FREQ

        # --- Create dummy input for graph logging ---
        dummy_input = None
        if self.agent and self.agent.online_net:
            try:
                dummy_state = np.zeros((1, self.env_config.STATE_DIM), dtype=np.float32)
                dummy_input = torch.tensor(dummy_state, device=DEVICE)
            except Exception as e:
                print(f"Warning: Failed to create dummy input for graph logging: {e}")
        # --- End dummy input ---

        print(f"Using TensorBoard Logger (Log Dir: {self.tensorboard_config.LOG_DIR})")
        try:
            # Pass necessary info for graph and hparams logging
            self.stats_recorder = TensorBoardStatsRecorder(
                log_dir=self.tensorboard_config.LOG_DIR,
                hparam_dict=self.config_dict,  # Pass flattened config
                model_for_graph=self.agent.online_net if self.agent else None,
                dummy_input_for_graph=dummy_input,
                console_log_interval=console_log_freq,
                avg_window=avg_window,
            )
        except Exception as e:
            print(f"FATAL: Error initializing TensorBoardStatsRecorder: {e}. Exiting.")
            traceback.print_exc()
            pygame.quit()
            sys.exit(1)

    def _initialize_trainer(self):
        """Initializes the Trainer, passing the already created components."""
        # Ensure all components are ready before creating Trainer
        if not all(
            hasattr(self, attr)
            for attr in ["envs", "agent", "buffer", "stats_recorder"]
        ):
            print("FATAL: Missing components before initializing Trainer.")
            pygame.quit()
            sys.exit(1)

        self.trainer: Trainer = Trainer(
            envs=self.envs,
            agent=self.agent,
            buffer=self.buffer,
            stats_recorder=self.stats_recorder,  # Pass the created recorder
            env_config=self.env_config,
            dqn_config=self.dqn_config,
            train_config=self.train_config,
            buffer_config=self.buffer_config,
            model_config=self.model_config,
            model_save_path=MODEL_SAVE_PATH,
            buffer_save_path=BUFFER_SAVE_PATH,
            load_checkpoint_path=self.train_config.LOAD_CHECKPOINT_PATH,
            load_buffer_path=self.train_config.LOAD_BUFFER_PATH,
        )
        print("Trainer initialization finished.")

    def _initialize_rl_components(self):
        """Orchestrates the initialization of RL components."""
        print("Initializing Env, Agent, Buffer...")
        self._initialize_rl_components_pre_stats()
        print("Initializing Stats Recorder (logging graph/hparams)...")
        self._initialize_stats_recorder()
        print("Initializing Trainer...")
        self._initialize_trainer()
        print("RL components initialization complete.")

    def _cleanup_data(self):
        """Stops training, deletes checkpoints, buffer FOR THE CURRENT RUN, and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.is_training = False
        self.status = "Cleaning"
        self.cleanup_confirmation_active = False
        messages = []

        # 1. Trainer Cleanup (saves final state, flushes buffer, closes logger)
        if hasattr(self, "trainer") and self.trainer:
            print("Running trainer cleanup (saving final state)...")
            try:
                self.trainer.cleanup(save_final=True)  # Ensure logger is closed here
            except Exception as e:
                print(f"Error during trainer cleanup: {e}")
        else:  # Close logger manually if trainer doesn't exist
            if hasattr(self, "stats_recorder") and self.stats_recorder:
                self.stats_recorder.close()
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
                messages.append(
                    f"Buffer state deleted: {os.path.basename(buffer_path)}"
                )
            else:
                messages.append("Buffer state not found (current run).")
        except OSError as e:
            messages.append(f"Error deleting buffer: {e}")

        # 4. Re-initialize RL components
        print("Re-initializing RL components after cleanup...")
        # This will create a new logger instance for the *same* run ID/directory.
        # This allows continuing logging in the same TB run after cleanup.
        self._initialize_rl_components()

        # Re-initialize renderer
        self.renderer = UIRenderer(self.screen, self.vis_config)

        self.cleanup_message = "\n".join(messages)
        self.last_cleanup_message_time = time.time()
        self.status = "Paused"
        print("--- CLEANUP DATA COMPLETE (Current Run Checkpoints/Buffer Removed) ---")

    def _handle_input(self) -> bool:
        # (No changes from previous version)
        mouse_pos = pygame.mouse.get_pos()
        sw, sh = self.screen.get_size()
        train_btn_rect = pygame.Rect(10, 10, 100, 40)
        cleanup_btn_rect = pygame.Rect(
            train_btn_rect.right + 10, 10, 160, 40
        )  # Wider button
        confirm_yes_rect = pygame.Rect(sw // 2 - 110, sh // 2 + 30, 100, 40)
        confirm_no_rect = pygame.Rect(sw // 2 + 10, sh // 2 + 30, 100, 40)
        self.renderer.check_hover(mouse_pos)  # Check for tooltip hover

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                try:
                    self.screen = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE
                    )
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
                        return False  # Exit app
                elif event.key == pygame.K_p and not self.cleanup_confirmation_active:
                    self.is_training = not self.is_training
                    print(
                        f"Training {'STARTED' if self.is_training else 'PAUSED'} (P key)"
                    )
                    self._try_save_checkpoint()  # Save on pause

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
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
                        print(
                            f"Training {'STARTED' if self.is_training else 'PAUSED'} (Button)"
                        )
                        self._try_save_checkpoint()  # Save on pause
                    elif cleanup_btn_rect.collidepoint(mouse_pos):
                        self.is_training = False  # Pause before confirmation
                        self.cleanup_confirmation_active = True
                        print("Cleanup requested.")
        return True

    def _try_save_checkpoint(self):
        """Saves checkpoint if trainer exists and is paused."""
        if not self.is_training and hasattr(self.trainer, "_save_checkpoint"):
            print("Saving checkpoint on pause...")
            try:
                self.trainer._save_checkpoint(is_final=False)
            except Exception as e:
                print(f"Error saving checkpoint on pause: {e}")

    def _update(self):
        """Performs training step and updates status."""
        # Determine status
        if self.cleanup_confirmation_active:
            self.status = "Confirm Cleanup"
        elif not self.is_training and self.status != "Error":
            self.status = "Paused"  # Keep Error status if paused due to error
        elif self.trainer.global_step < self.train_config.LEARN_START_STEP:
            self.status = "Buffering"
        else:
            self.status = "Training"  # Only set to Training if actively running

        # Only step trainer if in Training or Buffering state
        if self.status not in ["Training", "Buffering"]:
            return

        try:
            step_start_time = time.time()
            self.trainer.step()  # Trainer calls stats_recorder internally
            step_duration = time.time() - step_start_time
            # Optional delay
            if self.vis_config.VISUAL_STEP_DELAY > 0:
                time.sleep(max(0, self.vis_config.VISUAL_STEP_DELAY - step_duration))
        except Exception as e:
            print(
                f"\n--- ERROR DURING TRAINING UPDATE (Step: {getattr(self.trainer, 'global_step', 'N/A')}) ---"
            )
            traceback.print_exc()
            print(f"--- Pausing training due to error. Check logs. ---")
            self.is_training = False
            self.status = "Error"  # Set specific error status

    def _render(self):
        """Delegates rendering to the UIRenderer."""
        stats_summary = {}
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            # Get summary using the simple_recorder part for UI stats
            if isinstance(self.stats_recorder, TensorBoardStatsRecorder):
                stats_summary = self.stats_recorder.get_summary(
                    getattr(self.trainer, "global_step", 0)
                )

        buffer_capacity = (
            getattr(self.buffer, "capacity", 0) if hasattr(self, "buffer") else 0
        )

        self.renderer.render_all(
            is_training=self.is_training,
            status=self.status,
            stats_summary=stats_summary,
            buffer_capacity=buffer_capacity,
            envs=(
                self.envs if hasattr(self, "envs") else []
            ),  # Handle case where envs might not be init yet on error
            num_envs=self.num_envs,
            env_config=self.env_config,
            cleanup_confirmation_active=self.cleanup_confirmation_active,
            cleanup_message=self.cleanup_message,
            last_cleanup_message_time=self.last_cleanup_message_time,
            tensorboard_log_dir=self.tensorboard_config.LOG_DIR,
        )
        # Check if status message time expired
        if time.time() - self.last_cleanup_message_time >= 5.0:
            self.cleanup_message = ""

    def run(self):
        # (No changes from previous version)
        print("Starting main application loop...")
        running = True
        try:
            while running:
                running = self._handle_input()
                if not running:
                    break
                self._update()
                self._render()
                self.clock.tick(self.vis_config.FPS if self.vis_config.FPS > 0 else 0)
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Exiting gracefully...")
        except Exception as e:
            print("\n--- UNHANDLED EXCEPTION IN MAIN LOOP ---")
            traceback.print_exc()
            print("--- EXITING ---")
        finally:
            print("Exiting application...")
            if hasattr(self, "trainer") and self.trainer:
                print("Performing final trainer cleanup...")
                self.trainer.cleanup(save_final=True)  # Save final state
            elif hasattr(self, "stats_recorder") and self.stats_recorder:
                # Close logger if trainer cleanup didn't happen
                self.stats_recorder.close()
            pygame.quit()
            print("Application exited.")


def run_pre_checks():
    # (No changes from previous version)
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
            raise AttributeError(
                "GameState missing 'lines_cleared_this_episode' attribute!"
            )
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
    # Ensure base directories exist
    os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    os.makedirs("ui", exist_ok=True)  # Keep UI dir
    os.makedirs("stats", exist_ok=True)  # Keep stats dir

    if run_pre_checks():
        app = MainApp()
        app.run()
