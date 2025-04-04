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
from utils.helpers import set_random_seeds  # Removed ensure_numpy
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

        # --- MODIFIED: Instantiate config classes ---
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()  # Instantiate
        self.dqn_config = DQNConfig()
        self.train_config = TrainConfig()
        self.buffer_config = BufferConfig()
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        # --- END MODIFIED ---

        # --- MODIFIED: Access NUM_ENVS from instance ---
        self.num_envs = self.env_config.NUM_ENVS
        # --- END MODIFIED ---
        self.config_dict = (
            get_config_dict()
        )  # get_config_dict needs update if it accesses class attrs

        # Ensure directories exist
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(BUFFER_SAVE_PATH), exist_ok=True)
        # --- MODIFIED: Access LOG_DIR from instance ---
        os.makedirs(self.tensorboard_config.LOG_DIR, exist_ok=True)
        # --- END MODIFIED ---
        print_config_info_and_validate()  # This should now work

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

        # Init Renderer FIRST
        self.renderer = UIRenderer(self.screen, self.vis_config)

        # Init RL components using helpers
        self._initialize_rl_components()

        # Init Input Handler
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

    def _initialize_rl_components(self):
        """Orchestrates the initialization of RL components using helpers."""
        try:
            # --- MODIFIED: Pass config instances ---
            self.envs = initialize_envs(self.num_envs, self.env_config)  # Pass instance
            self.agent, self.buffer = initialize_agent_buffer(
                self.model_config,
                self.dqn_config,
                self.env_config,
                self.buffer_config,  # Pass instances
            )
            self.stats_recorder = initialize_stats_recorder(
                self.stats_config,
                self.tensorboard_config,
                self.config_dict,
                self.agent,
                self.env_config,  # Pass instance
            )
            self.trainer = initialize_trainer(
                self.envs,
                self.agent,
                self.buffer,
                self.stats_recorder,
                self.env_config,  # Pass instance
                self.dqn_config,
                self.train_config,
                self.buffer_config,
                self.model_config,
            )
            # --- END MODIFIED ---
        except Exception as e:
            print(f"FATAL ERROR during RL component initialization: {e}")
            traceback.print_exc()
            pygame.quit()
            sys.exit(1)

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
            self._try_save_checkpoint()
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
        return False

    # --- Other Methods (Cleanup, Save, Update) ---
    def _cleanup_data(self):
        """Deletes current run's checkpoint/buffer and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.is_training = False
        self.status = "Cleaning"
        self.cleanup_confirmation_active = False
        messages = []

        if hasattr(self, "trainer") and self.trainer:
            print("Running trainer cleanup...")
            try:
                self.trainer.cleanup(save_final=False)
                self.trainer = None
                if hasattr(self, "stats_recorder"):
                    self.stats_recorder = None
            except Exception as e:
                print(f"Error during trainer cleanup: {e}")
                if hasattr(self, "stats_recorder") and self.stats_recorder:
                    try:
                        self.stats_recorder.close()
                    except Exception as log_e:
                        print(
                            f"Error closing stats recorder during failed cleanup: {log_e}"
                        )
                    self.stats_recorder = None
        elif hasattr(self, "stats_recorder") and self.stats_recorder:
            try:
                self.stats_recorder.close()
            except Exception as log_e:
                print(f"Error closing stats recorder: {log_e}")
            self.stats_recorder = None

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
            except OSError as e:
                msg = f"Error deleting {desc}: {e}"
                print(msg)
                messages.append(msg)

        time.sleep(0.1)

        print("Re-initializing RL components after cleanup...")
        try:
            self._initialize_rl_components()  # Re-initializes with instances
            self.renderer = UIRenderer(self.screen, self.vis_config)  # Re-init renderer
            print("RL components re-initialized.")
            messages.append("RL components re-initialized.")
        except Exception as e:
            print(f"FATAL ERROR during RL component re-initialization: {e}")
            traceback.print_exc()
            self.status = "Error"
            messages.append("ERROR RE-INITIALIZING RL COMPONENTS!")

        self.cleanup_message = "\n".join(messages)
        self.last_cleanup_message_time = time.time()
        if self.status != "Error":
            self.status = "Paused"
        print("--- CLEANUP DATA COMPLETE ---")

    def _try_save_checkpoint(self):
        """Saves checkpoint if not training and trainer exists."""
        if not self.is_training and hasattr(self, "trainer") and self.trainer:
            print("Saving checkpoint on pause...")
            try:
                self.trainer.maybe_save_checkpoint(force_save=True)
            except Exception as e:
                print(f"Error saving checkpoint on pause: {e}")
                traceback.print_exc()

    def _update(self):
        """Updates the application state and performs training steps."""
        if self.cleanup_confirmation_active:
            self.status = "Confirm Cleanup"
        elif not self.is_training and self.status != "Error":
            self.status = "Paused"
        elif not hasattr(self, "trainer") or self.trainer is None:
            if self.status != "Error":
                self.status = "Error"
                print("Error: Trainer object not found during update.")
        # --- MODIFIED: Access LEARN_START_STEP from instance ---
        elif self.trainer.global_step < self.train_config.LEARN_START_STEP:
            # --- END MODIFIED ---
            self.status = "Buffering"
        elif self.is_training:
            self.status = "Training"

        if self.status not in ["Training", "Buffering"]:
            return

        if not hasattr(self, "trainer") or self.trainer is None:
            print("Error: Trainer became unavailable during _update.")
            self.status = "Error"
            self.is_training = False
            return

        try:
            step_start_time = time.time()
            self.trainer.step()
            step_duration = time.time() - step_start_time
            # --- MODIFIED: Access VISUAL_STEP_DELAY from instance ---
            if self.vis_config.VISUAL_STEP_DELAY > 0:
                time.sleep(max(0, self.vis_config.VISUAL_STEP_DELAY - step_duration))
            # --- END MODIFIED ---
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

        if hasattr(self, "stats_recorder") and self.stats_recorder:
            current_step = getattr(self.trainer, "global_step", 0)
            if hasattr(self.stats_recorder, "get_summary"):
                stats_summary = self.stats_recorder.get_summary(current_step)
            if hasattr(self.stats_recorder, "get_plot_data"):
                plot_data = self.stats_recorder.get_plot_data()
        elif self.status == "Error":
            stats_summary = {"global_step": getattr(self.trainer, "global_step", 0)}
            plot_data = {}

        buffer_capacity = (
            getattr(self.buffer, "capacity", 0) if hasattr(self, "buffer") else 0
        )

        if not hasattr(self, "renderer") or self.renderer is None:
            print("Error: Renderer not initialized in _render.")
            return

        # --- MODIFIED: Pass config instances ---
        self.renderer.render_all(
            is_training=self.is_training,
            status=self.status,
            stats_summary=stats_summary,
            buffer_capacity=buffer_capacity,
            envs=(self.envs if hasattr(self, "envs") else []),
            num_envs=self.num_envs,
            env_config=self.env_config,  # Pass instance
            cleanup_confirmation_active=self.cleanup_confirmation_active,
            cleanup_message=self.cleanup_message,
            last_cleanup_message_time=self.last_cleanup_message_time,
            tensorboard_log_dir=self.tensorboard_config.LOG_DIR,  # Access instance attr
            plot_data=plot_data,
        )
        # --- END MODIFIED ---

        if time.time() - self.last_cleanup_message_time >= 5.0:
            self.cleanup_message = ""

    def run(self):
        """Main application loop."""
        print("Starting main application loop...")
        running = True
        try:
            while running:
                running = self.input_handler.handle_input(
                    self.cleanup_confirmation_active
                )
                if not running:
                    break

                try:
                    self._update()
                except Exception as update_err:
                    print(f"\n--- UNHANDLED ERROR IN UPDATE LOOP ---")
                    traceback.print_exc()
                    print(f"--- Setting status to Error ---")
                    self.status = "Error"
                    self.is_training = False

                try:
                    self._render()
                except Exception as render_err:
                    print(f"\n--- UNHANDLED ERROR IN RENDER LOOP ---")
                    traceback.print_exc()

                # --- MODIFIED: Access FPS from instance ---
                self.clock.tick(self.vis_config.FPS if self.vis_config.FPS > 0 else 0)
                # --- END MODIFIED ---

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
                try:
                    save_on_exit = self.status != "Cleaning"
                    self.trainer.cleanup(save_final=save_on_exit)
                except Exception as final_cleanup_err:
                    print(f"Error during final trainer cleanup: {final_cleanup_err}")
            elif hasattr(self, "stats_recorder") and self.stats_recorder:
                try:
                    self.stats_recorder.close()
                except Exception as log_e:
                    print(f"Error closing stats recorder on exit: {log_e}")

            pygame.quit()
            print("Application exited.")


if __name__ == "__main__":
    os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    log_filepath = os.path.join(RUN_LOG_DIR, "console_output.log")
    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger = TeeLogger(log_filepath, original_stdout)
    sys.stdout = logger
    sys.stderr = logger

    try:
        if run_pre_checks():
            app = MainApp()
            app.run()
    except SystemExit:
        print("Exiting due to SystemExit (likely from pre-checks or init error).")
    except Exception as main_err:
        print("\n--- UNHANDLED EXCEPTION DURING APP INITIALIZATION OR RUN ---")
        traceback.print_exc()
        print("--- EXITING DUE TO ERROR ---")
    finally:
        if "logger" in locals() and logger:
            logger.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Console logging restored. Full log saved to: {log_filepath}")
