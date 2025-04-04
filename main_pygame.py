# File: main_pygame.py
import sys
import pygame
import numpy as np
import os
import time
import traceback
import torch
from typing import List, Tuple, Optional, Dict, Any, Deque, TextIO


# --- Logger Class (Unchanged) ---
class TeeLogger:
    def __init__(self, filepath: str, original_stream: TextIO):
        self.terminal = original_stream
        try:
            self.log_file = open(filepath, "w", encoding="utf-8", buffering=1)
            print(f"[TeeLogger] Logging console output to: {filepath}")
        except Exception as e:
            self.terminal.write(
                f"FATAL ERROR: Could not open log file {filepath}: {e}\n"
            )
            self.log_file = None

    def write(self, message: str):
        self.terminal.write(message)
        if self.log_file:
            try:
                self.log_file.write(message)
            except Exception:
                pass  # Ignore errors writing to log file

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass  # Ignore errors flushing log file

    def close(self):
        if self.log_file:
            try:
                self.log_file.close()
                self.log_file = None
            except Exception as e:
                self.terminal.write(f"Warning: Error closing log file: {e}\n")


# --- End Logger Class ---


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
    RUN_LOG_DIR,
    get_config_dict,
    print_config_info_and_validate,
)

# Import core components & helpers
from environment.game_state import GameState
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from training.trainer import Trainer
from stats.stats_recorder import StatsRecorderBase
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler
from utils.helpers import set_random_seeds, ensure_numpy
from utils.init_checks import run_pre_checks
from init.rl_components import (
    initialize_envs,
    initialize_agent_buffer,
    initialize_stats_recorder,
    initialize_trainer,
)


class MainApp:
    def __init__(self):
        print("Initializing Pygame Application...")
        set_random_seeds(RANDOM_SEED)
        pygame.init()
        pygame.font.init()

        # Store configs
        self.vis_config = VisConfig
        self.env_config = EnvConfig
        self.dqn_config = DQNConfig
        self.train_config = TrainConfig
        self.buffer_config = BufferConfig
        self.model_config = ModelConfig
        self.stats_config = StatsConfig
        self.tensorboard_config = TensorBoardConfig

        self.num_envs = self.env_config.NUM_ENVS
        self.config_dict = get_config_dict()

        # Ensure directories exist
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(BUFFER_SAVE_PATH), exist_ok=True)
        os.makedirs(self.tensorboard_config.LOG_DIR, exist_ok=True)
        print_config_info_and_validate()

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
        self.status = "Paused"  # Initial status

        # Init Renderer FIRST
        self.renderer = UIRenderer(self.screen, self.vis_config)

        # --- MODIFIED: Remove notification_callback passing ---
        # Init RL components using helpers
        self._initialize_rl_components()
        # --- END MODIFIED ---

        # Init Input Handler (pass callbacks)
        self.input_handler = InputHandler(
            self.screen,
            self.renderer,
            self._toggle_training,
            self._request_cleanup,
            self._cancel_cleanup,
            self._confirm_cleanup,
            self._exit_app,
        )

        print("Initialization Complete. Ready to start.")
        print(f"--- tensorboard --logdir {os.path.abspath(BASE_LOG_DIR)} ---")

    # --- MODIFIED: Remove notification_callback parameter ---
    def _initialize_rl_components(self):
        """Orchestrates the initialization of RL components using helpers."""
        try:
            self.envs = initialize_envs(self.num_envs, self.env_config)
            self.agent, self.buffer = initialize_agent_buffer(
                self.model_config, self.dqn_config, self.env_config, self.buffer_config
            )
            # Stats recorder init no longer needs callback
            self.stats_recorder = initialize_stats_recorder(
                self.stats_config,
                self.tensorboard_config,
                self.config_dict,
                self.agent,
                self.env_config,
                # notification_callback=... # Removed
            )
            # Trainer init no longer needs callback
            self.trainer = initialize_trainer(
                self.envs,
                self.agent,
                self.buffer,
                self.stats_recorder,
                self.env_config,
                self.dqn_config,
                self.train_config,
                self.buffer_config,
                self.model_config,
                # notification_callback=... # Removed
            )
        except Exception as e:
            print(f"FATAL ERROR during RL component initialization: {e}")
            traceback.print_exc()
            pygame.quit()
            sys.exit(1)

    # --- END MODIFIED ---

    # --- Input Handler Callbacks (Unchanged) ---
    def _toggle_training(self):
        self.is_training = not self.is_training
        print(f"Training {'STARTED' if self.is_training else 'PAUSED'}")
        if not self.is_training:
            self._try_save_checkpoint()

    def _request_cleanup(self):
        was_training = self.is_training
        self.is_training = False
        if was_training:
            self._try_save_checkpoint()  # Save before potentially deleting
        self.cleanup_confirmation_active = True
        print("Cleanup requested. Training paused. Confirm action.")

    def _cancel_cleanup(self):
        self.cleanup_confirmation_active = False
        self.cleanup_message = "Cleanup cancelled."
        self.last_cleanup_message_time = time.time()
        print("Cleanup cancelled by user.")

    def _confirm_cleanup(self):
        self._cleanup_data()

    def _exit_app(self) -> bool:
        """Callback for input handler to signal application exit."""
        return False  # Signal to stop the main loop

    # --- Other Methods (Cleanup, Save, Update) ---
    def _cleanup_data(self):
        """Deletes current run's checkpoint/buffer and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.is_training = False
        self.status = "Cleaning"
        self.cleanup_confirmation_active = False
        messages = []

        # 1. Cleanup Trainer and Stats Recorder first
        if hasattr(self, "trainer") and self.trainer:
            print("Running trainer cleanup...")
            try:
                # Pass save_final=False to prevent saving during cleanup
                self.trainer.cleanup(save_final=False)
                self.trainer = None  # Release trainer object
                # Stats recorder is closed by trainer.cleanup, release ref
                if hasattr(self, "stats_recorder"):
                    self.stats_recorder = None
            except Exception as e:
                print(f"Error during trainer cleanup: {e}")
                # Attempt to close recorder manually if trainer cleanup failed
                if hasattr(self, "stats_recorder") and self.stats_recorder:
                    try:
                        self.stats_recorder.close()
                    except Exception as log_e:
                        print(
                            f"Error closing stats recorder during failed cleanup: {log_e}"
                        )
                    self.stats_recorder = None
        elif hasattr(self, "stats_recorder") and self.stats_recorder:
            # If trainer didn't exist but recorder did, close recorder
            try:
                self.stats_recorder.close()
            except Exception as log_e:
                print(f"Error closing stats recorder: {log_e}")
            self.stats_recorder = None

        # 2. Delete Checkpoint and Buffer Files
        for path, desc in [
            (MODEL_SAVE_PATH, "Agent ckpt"),
            (BUFFER_SAVE_PATH, "Buffer state"),
        ]:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    msg = f"{desc} deleted: {os.path.basename(path)}"
                    print(msg)
                    messages.append(msg)
                else:
                    msg = f"{desc} not found (current run)."
                    print(msg)
                    # messages.append(msg) # Don't show "not found" in UI message
            except OSError as e:
                msg = f"Error deleting {desc}: {e}"
                print(msg)
                messages.append(msg)

        time.sleep(0.1)  # Short pause

        # 3. Re-initialize RL components
        print("Re-initializing RL components after cleanup...")
        try:
            # --- MODIFIED: Remove notification_callback passing ---
            self._initialize_rl_components()
            # --- END MODIFIED ---
            # Re-initialize renderer to clear any old state (like toasts if they existed)
            self.renderer = UIRenderer(self.screen, self.vis_config)
            print("RL components re-initialized.")
            messages.append("RL components re-initialized.")
        except Exception as e:
            print(f"FATAL ERROR during RL component re-initialization: {e}")
            traceback.print_exc()
            self.status = "Error"
            messages.append("ERROR RE-INITIALIZING RL COMPONENTS!")

        # 4. Update UI message
        self.cleanup_message = "\n".join(messages)
        self.last_cleanup_message_time = time.time()
        if self.status != "Error":
            self.status = "Paused"  # Set status back to Paused if re-init successful
        print("--- CLEANUP DATA COMPLETE ---")

    def _try_save_checkpoint(self):
        """Saves checkpoint if not training and trainer exists."""
        if not self.is_training and hasattr(self, "trainer") and self.trainer:
            print("Saving checkpoint on pause...")
            try:
                # Force save ensures it saves even if save interval not reached
                self.trainer.maybe_save_checkpoint(force_save=True)
            except Exception as e:
                print(f"Error saving checkpoint on pause: {e}")
                traceback.print_exc()  # Show details on save error

    def _update(self):
        """Updates the application state and performs training steps."""
        # Update status string based on current state
        if self.cleanup_confirmation_active:
            self.status = "Confirm Cleanup"
        elif not self.is_training and self.status != "Error":
            self.status = "Paused"
        elif not hasattr(self, "trainer") or self.trainer is None:
            if self.status != "Error":  # Avoid overwriting existing error state
                self.status = "Error"
                print("Error: Trainer object not found during update.")
        elif self.trainer.global_step < self.train_config.LEARN_START_STEP:
            self.status = "Buffering"
        elif self.is_training:
            self.status = "Training"

        # Only perform training steps if in Training or Buffering state
        if self.status not in ["Training", "Buffering"]:
            return

        # Double-check trainer exists before stepping
        if not hasattr(self, "trainer") or self.trainer is None:
            print("Error: Trainer became unavailable during _update.")
            self.status = "Error"
            self.is_training = False
            return

        # Perform one trainer step
        try:
            step_start_time = time.time()
            self.trainer.step()
            step_duration = time.time() - step_start_time

            # Optional delay for visualization
            if self.vis_config.VISUAL_STEP_DELAY > 0:
                time.sleep(max(0, self.vis_config.VISUAL_STEP_DELAY - step_duration))

        except Exception as e:
            print(
                f"\n--- ERROR DURING TRAINING UPDATE (Step: {getattr(self.trainer, 'global_step', 'N/A')}) ---"
            )
            traceback.print_exc()
            print(f"--- Pausing training due to error. ---")
            self.is_training = False
            self.status = "Error"

    def _render(self):
        """Renders the entire UI."""
        stats_summary = {}
        plot_data: Dict[str, Deque] = {}

        # Get latest stats from the recorder
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            current_step = getattr(self.trainer, "global_step", 0)
            # Use hasattr for safety, as recorder might be None during cleanup/error
            if hasattr(self.stats_recorder, "get_summary"):
                stats_summary = self.stats_recorder.get_summary(current_step)
            if hasattr(self.stats_recorder, "get_plot_data"):
                plot_data = self.stats_recorder.get_plot_data()
        elif self.status == "Error":
            # Provide minimal stats if in error state
            stats_summary = {"global_step": getattr(self.trainer, "global_step", 0)}
            plot_data = {}

        # Get buffer capacity for display
        buffer_capacity = (
            getattr(self.buffer, "capacity", 0) if hasattr(self, "buffer") else 0
        )

        # Ensure renderer exists
        if not hasattr(self, "renderer") or self.renderer is None:
            print("Error: Renderer not initialized in _render.")
            return

        # Call the main render function
        self.renderer.render_all(
            is_training=self.is_training,
            status=self.status,
            stats_summary=stats_summary,
            buffer_capacity=buffer_capacity,
            envs=(self.envs if hasattr(self, "envs") else []),
            num_envs=self.num_envs,
            env_config=self.env_config,
            cleanup_confirmation_active=self.cleanup_confirmation_active,
            cleanup_message=self.cleanup_message,
            last_cleanup_message_time=self.last_cleanup_message_time,
            tensorboard_log_dir=self.tensorboard_config.LOG_DIR,
            plot_data=plot_data,
        )

        # Clear cleanup message after a delay
        if time.time() - self.last_cleanup_message_time >= 5.0:
            self.cleanup_message = ""

    def run(self):
        """Main application loop."""
        print("Starting main application loop...")
        running = True
        try:
            while running:
                # 1. Handle User Input
                running = self.input_handler.handle_input(
                    self.cleanup_confirmation_active
                )
                if not running:
                    break  # Exit signal received

                # 2. Update State and Train
                try:
                    self._update()
                except Exception as update_err:
                    # Catch errors specifically within the update logic
                    print(f"\n--- UNHANDLED ERROR IN UPDATE LOOP ---")
                    traceback.print_exc()
                    print(f"--- Setting status to Error ---")
                    self.status = "Error"
                    self.is_training = False  # Stop training on error

                # 3. Render UI
                try:
                    self._render()
                except Exception as render_err:
                    # Catch errors specifically within the render logic
                    print(f"\n--- UNHANDLED ERROR IN RENDER LOOP ---")
                    traceback.print_exc()
                    # Don't necessarily stop training on render error, but log it
                    # self.status = "Error" # Optional: Set error status on render fail

                # 4. Control Frame Rate
                self.clock.tick(self.vis_config.FPS if self.vis_config.FPS > 0 else 0)

        except KeyboardInterrupt:
            print("\nCtrl+C detected. Exiting gracefully...")
        except Exception as e:
            # Catch any other unexpected errors in the main loop
            print("\n--- UNHANDLED EXCEPTION IN MAIN LOOP ---")
            traceback.print_exc()
            print("--- EXITING ---")
        finally:
            # --- Cleanup ---
            print("Exiting application...")
            # Ensure trainer cleanup runs to save final state if possible
            if hasattr(self, "trainer") and self.trainer:
                print("Performing final trainer cleanup...")
                try:
                    # Save final checkpoint unless cleanup already happened
                    save_on_exit = self.status != "Cleaning"
                    self.trainer.cleanup(save_final=save_on_exit)
                except Exception as final_cleanup_err:
                    print(f"Error during final trainer cleanup: {final_cleanup_err}")
            elif hasattr(self, "stats_recorder") and self.stats_recorder:
                # Close recorder if trainer cleanup didn't happen/failed
                try:
                    self.stats_recorder.close()
                except Exception as log_e:
                    print(f"Error closing stats recorder on exit: {log_e}")

            pygame.quit()
            print("Application exited.")


if __name__ == "__main__":
    # Setup Dirs
    os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    # Ensure other necessary dirs exist (optional, depends on imports)
    # os.makedirs("ui", exist_ok=True)
    # os.makedirs("stats", exist_ok=True)

    # Setup Logging to file and console
    log_filepath = os.path.join(RUN_LOG_DIR, "console_output.log")
    os.makedirs(RUN_LOG_DIR, exist_ok=True)  # Ensure run-specific log dir exists
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger = TeeLogger(log_filepath, original_stdout)
    sys.stdout = logger
    sys.stderr = logger  # Redirect stderr as well

    try:
        # Perform pre-checks before starting the app
        if run_pre_checks():
            app = MainApp()
            app.run()
    except SystemExit:
        print("Exiting due to SystemExit (likely from pre-checks or init error).")
    except Exception as main_err:
        # Catch errors during App initialization or run() call
        print("\n--- UNHANDLED EXCEPTION DURING APP INITIALIZATION OR RUN ---")
        traceback.print_exc()
        print("--- EXITING DUE TO ERROR ---")
    finally:
        # Restore standard output streams and close logger
        if "logger" in locals() and logger:
            logger.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Print final message to the actual console
        print(f"Console logging restored. Full log saved to: {log_filepath}")
