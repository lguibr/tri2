# File: main_pygame.py
import sys
import pygame
import numpy as np
import os
import time
import traceback
import torch
from typing import List, Tuple, Optional, Dict, Any, Deque

# --- Project Imports ---
# Setup & Logging
from logger import TeeLogger
from app_setup import (
    initialize_pygame,
    initialize_directories,
    load_and_validate_configs,
)

# Configs (Import necessary instances/constants)
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
    DemoConfig,
    DEVICE,
    RANDOM_SEED,
    BUFFER_SAVE_PATH,
    MODEL_SAVE_PATH,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    RUN_LOG_DIR,
)

# Core Components & Helpers
from environment.game_state import GameState, StateType
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from training.trainer import Trainer
from stats.stats_recorder import StatsRecorderBase
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler
from utils.helpers import set_random_seeds
from utils.init_checks import run_pre_checks
from init.rl_components import (
    initialize_envs,
    initialize_agent_buffer,
    initialize_stats_recorder,
    initialize_trainer,
)


class MainApp:
    """Main application class orchestrating the Pygame UI and RL training."""

    def __init__(self):
        print("Initializing Application...")
        set_random_seeds(RANDOM_SEED)

        # --- Configuration Loading ---
        # Instantiate config classes needed by the app directly
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        self.dqn_config = DQNConfig()
        self.train_config = TrainConfig()
        self.buffer_config = BufferConfig()
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        self.reward_config = RewardConfig()  # Although not directly used, good practice

        # Load combined config dict and validate
        self.config_dict = load_and_validate_configs()
        self.num_envs = self.env_config.NUM_ENVS

        # --- Setup Pygame and Directories ---
        initialize_directories()
        self.screen, self.clock = initialize_pygame(self.vis_config)

        # --- App State ---
        self.app_state = "Initializing"
        self.is_training = False
        self.cleanup_confirmation_active = False
        self.last_cleanup_message_time = 0.0
        self.cleanup_message = ""
        self.status = "Initializing Components"

        # --- Initialize Components ---
        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None
        self.envs: List[GameState] = []
        self.agent: Optional[DQNAgent] = None
        self.buffer: Optional[ReplayBufferBase] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.trainer: Optional[Trainer] = None
        self.demo_env: Optional[GameState] = None

        self._initialize_core_components()

        # Transition to Main Menu after successful initialization
        self.app_state = "MainMenu"
        self.status = "Paused"
        print("Initialization Complete. Ready.")
        print(f"--- tensorboard --logdir {os.path.abspath(BASE_LOG_DIR)} ---")

    def _initialize_core_components(self):
        """Initializes Renderer, RL components, Demo Env, and Input Handler."""
        try:
            # Init Renderer FIRST for immediate feedback
            self.renderer = UIRenderer(self.screen, self.vis_config)
            self.renderer.render_all(  # Render initializing screen
                app_state=self.app_state,
                is_training=False,
                status=self.status,
                stats_summary={},
                buffer_capacity=0,
                envs=[],
                num_envs=0,
                env_config=self.env_config,
                cleanup_confirmation_active=False,
                cleanup_message="",
                last_cleanup_message_time=0,
                tensorboard_log_dir=None,
                plot_data={},
                demo_env=None,
            )
            pygame.time.delay(100)  # Brief pause to show initializing screen

            # Init RL components
            self._initialize_rl_components()

            # Init Demo Env
            self._initialize_demo_env()

            # Init Input Handler (pass self methods as callbacks)
            self.input_handler = InputHandler(
                self.screen,
                self.renderer,
                self._toggle_training,
                self._request_cleanup,
                self._cancel_cleanup,
                self._confirm_cleanup,
                self._exit_app,
                self._start_demo_mode,
                self._exit_demo_mode,
                self._handle_demo_input,
            )
        except Exception as init_err:
            print(f"FATAL ERROR during component initialization: {init_err}")
            traceback.print_exc()
            # Attempt to show error on screen if renderer exists
            if self.renderer:
                try:
                    self.app_state = "Error"
                    self.status = "Initialization Failed"
                    self.renderer._render_error_screen(self.status)
                    pygame.display.flip()
                    time.sleep(5)  # Show error for a bit
                except Exception:
                    pass  # Ignore errors during error rendering
            pygame.quit()
            sys.exit(1)

    def _initialize_rl_components(self):
        """Initializes RL components using helper functions."""
        print("Initializing RL components...")
        start_time = time.time()
        try:
            self.envs = initialize_envs(self.num_envs, self.env_config)
            self.agent, self.buffer = initialize_agent_buffer(
                self.model_config, self.dqn_config, self.env_config, self.buffer_config
            )
            if self.buffer is None:
                raise RuntimeError("Buffer initialization failed unexpectedly.")

            self.stats_recorder = initialize_stats_recorder(
                self.stats_config,
                self.tensorboard_config,
                self.config_dict,
                self.agent,
                self.env_config,
            )
            if self.stats_recorder is None:
                raise RuntimeError("Stats Recorder initialization failed unexpectedly.")

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
            )
            print(f"RL components initialized in {time.time() - start_time:.2f}s")
        except Exception as e:
            # Let the caller handle the fatal error exit
            print(f"Error during RL component initialization: {e}")
            raise e  # Re-raise to be caught by _initialize_core_components

    def _initialize_demo_env(self):
        """Initializes the separate environment for demo mode."""
        print("Initializing Demo Environment...")
        try:
            self.demo_env = GameState()
            self.demo_env.reset()
            print("Demo environment initialized.")
        except Exception as e:
            print(f"ERROR initializing demo environment: {e}")
            traceback.print_exc()
            self.demo_env = None  # Continue without demo mode
            print("Warning: Demo mode may be unavailable.")

    # --- Input Handler Callbacks (Remain in MainApp) ---
    def _toggle_training(self):
        if self.app_state != "MainMenu":
            return
        self.is_training = not self.is_training
        print(f"Training {'STARTED' if self.is_training else 'PAUSED'}")
        if not self.is_training:
            self._try_save_checkpoint()

    def _request_cleanup(self):
        if self.app_state != "MainMenu":
            return
        was_training = self.is_training
        self.is_training = False
        if was_training:
            self._try_save_checkpoint()
        self.cleanup_confirmation_active = True
        print("Cleanup requested. Confirm action.")

    def _cancel_cleanup(self):
        self.cleanup_confirmation_active = False
        self.cleanup_message = "Cleanup cancelled."
        self.last_cleanup_message_time = time.time()
        print("Cleanup cancelled by user.")

    def _confirm_cleanup(self):
        print("Cleanup confirmed by user. Starting process...")
        try:
            self._cleanup_data()
        except Exception as e:
            print(f"FATAL ERROR during cleanup: {e}")
            traceback.print_exc()
            self.status = "Error: Cleanup Failed Critically"
            self.app_state = "Error"
        finally:
            self.cleanup_confirmation_active = False  # Ensure flag is reset
            print(
                f"Cleanup process finished. State: {self.app_state}, Status: {self.status}"
            )

    def _exit_app(self) -> bool:
        print("Exit requested.")
        return False  # Signal exit to main loop

    def _start_demo_mode(self):
        if self.demo_env is None:
            print("Cannot start demo mode: Demo environment failed to initialize.")
            return
        if self.app_state == "MainMenu":
            print("Entering Demo Mode...")
            self.is_training = False
            self._try_save_checkpoint()
            self.app_state = "Playing"
            self.status = "Playing Demo"
            self.demo_env.reset()

    def _exit_demo_mode(self):
        if self.app_state == "Playing":
            print("Exiting Demo Mode...")
            self.app_state = "MainMenu"
            self.status = "Paused"

    def _handle_demo_input(self, event: pygame.event.Event):
        """Handles keyboard input during demo mode."""
        if self.app_state != "Playing" or self.demo_env is None:
            return
        if self.demo_env.is_frozen() or self.demo_env.is_over():
            return

        if event.type == pygame.KEYDOWN:
            action_taken = False
            if event.key == pygame.K_LEFT:
                self.demo_env.move_target(0, -1)
                action_taken = True
            elif event.key == pygame.K_RIGHT:
                self.demo_env.move_target(0, 1)
                action_taken = True
            elif event.key == pygame.K_UP:
                self.demo_env.move_target(-1, 0)
                action_taken = True
            elif event.key == pygame.K_DOWN:
                self.demo_env.move_target(1, 0)
                action_taken = True
            elif event.key == pygame.K_q:
                self.demo_env.cycle_shape(-1)
                action_taken = True
            elif event.key == pygame.K_e:
                self.demo_env.cycle_shape(1)
                action_taken = True
            elif event.key == pygame.K_SPACE:
                action_index = self.demo_env.get_action_for_current_selection()
                if action_index is not None:
                    state_before = self.demo_env.get_state()
                    reward, done = self.demo_env.step(action_index)
                    next_state_after = self.demo_env.get_state()
                    if self.buffer:
                        try:
                            self.buffer.push(
                                state_before,
                                action_index,
                                reward,
                                next_state_after,
                                done,
                            )
                            if (
                                len(self.buffer) % 50 == 0
                                and len(self.buffer)
                                <= self.train_config.LEARN_START_STEP
                            ):
                                print(
                                    f"[Demo] Added experience. Buffer: {len(self.buffer)}/{self.buffer.capacity}"
                                )
                        except Exception as buf_e:
                            print(f"Error pushing demo experience to buffer: {buf_e}")
                    action_taken = True
                else:
                    action_taken = (
                        True  # Still counts as an action (attempted placement)
                    )

            if self.demo_env.is_over():
                print("[Demo] Game Over! Press ESC to exit.")

    # --- Core Logic Methods (Remain in MainApp) ---
    def _cleanup_data(self):
        """Deletes current run's checkpoint/buffer and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.app_state = "Initializing"
        self.is_training = False
        self.status = "Cleaning"
        messages = []

        # Render initializing screen during cleanup
        if self.renderer:
            try:
                self.renderer.render_all(
                    app_state=self.app_state,
                    is_training=False,
                    status=self.status,
                    stats_summary={},
                    buffer_capacity=0,
                    envs=[],
                    num_envs=0,
                    env_config=self.env_config,
                    cleanup_confirmation_active=False,
                    cleanup_message="",
                    last_cleanup_message_time=0,
                    tensorboard_log_dir=None,
                    plot_data={},
                    demo_env=self.demo_env,
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

        # Close trainer/stats FIRST
        if self.trainer:
            print("[Cleanup] Running trainer cleanup...")
            try:
                self.trainer.cleanup(save_final=False)
            except Exception as e:
                print(f"Error during trainer cleanup: {e}")
        if self.stats_recorder:
            print("[Cleanup] Closing stats recorder...")
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"Error closing stats recorder: {e}")

        # Delete files
        print("[Cleanup] Deleting files...")
        for path, desc in [
            (MODEL_SAVE_PATH, "Agent ckpt"),
            (BUFFER_SAVE_PATH, "Buffer state"),
        ]:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    msg = f"{desc} deleted: {os.path.basename(path)}"
                else:
                    msg = f"{desc} not found (current run)."
                print(f"  - {msg}")
                messages.append(msg)
            except OSError as e:
                msg = f"Error deleting {desc}: {e}"
                print(f"  - {msg}")
                messages.append(msg)

        time.sleep(0.1)

        # Re-initialize RL components
        print("[Cleanup] Re-initializing RL components...")
        try:
            self._initialize_rl_components()
            if self.demo_env:
                self.demo_env.reset()
            print("[Cleanup] RL components re-initialized successfully.")
            messages.append("RL components re-initialized.")
            self.status = "Paused"
            self.app_state = "MainMenu"
        except Exception as e:
            print(f"FATAL ERROR during RL re-initialization after cleanup: {e}")
            traceback.print_exc()
            self.status = "Error: Re-init Failed"
            self.app_state = "Error"
            messages.append("ERROR RE-INITIALIZING RL COMPONENTS!")
            if self.renderer:
                try:
                    self.renderer._render_error_screen(self.status)
                except Exception as render_err_final:
                    print(f"Warning: Failed to render error screen: {render_err_final}")

        self.cleanup_message = "\n".join(messages)
        self.last_cleanup_message_time = time.time()
        print(
            f"--- CLEANUP DATA COMPLETE (Final State: {self.app_state}, Status: {self.status}) ---"
        )

    def _try_save_checkpoint(self):
        """Saves checkpoint if not training and trainer exists."""
        if self.app_state == "MainMenu" and not self.is_training and self.trainer:
            print("Saving checkpoint on pause...")
            try:
                self.trainer.maybe_save_checkpoint(force_save=True)
            except Exception as e:
                print(f"Error saving checkpoint on pause: {e}")

    def _update(self):
        """Updates the application state and performs training steps."""
        should_step_trainer = False

        if self.app_state == "MainMenu":
            if self.cleanup_confirmation_active:
                self.status = "Confirm Cleanup"
            elif not self.is_training and self.status != "Error":
                self.status = "Paused"
            elif not self.trainer:
                if self.status != "Error":
                    self.status = "Error: Trainer Missing"
            elif self.is_training:
                current_fill = len(self.buffer) if self.buffer else 0
                current_step = self.trainer.global_step if self.trainer else 0
                needed = self.train_config.LEARN_START_STEP
                is_buffer_ready = current_fill >= needed and current_step >= needed

                self.status = (
                    f"Buffering ({current_fill / max(1, needed) * 100:.0f}%)"
                    if not is_buffer_ready
                    else "Training"
                )
                should_step_trainer = True
            else:  # Not training
                if not self.status.startswith("Buffering") and self.status != "Error":
                    self.status = "Paused"

            if should_step_trainer:
                if not self.trainer:
                    print("Error: Trainer became unavailable during _update.")
                    self.status = "Error: Trainer Lost"
                    self.is_training = False
                else:
                    try:
                        step_start_time = time.time()
                        self.trainer.step()
                        step_duration = time.time() - step_start_time
                        if self.vis_config.VISUAL_STEP_DELAY > 0:
                            time.sleep(
                                max(
                                    0, self.vis_config.VISUAL_STEP_DELAY - step_duration
                                )
                            )
                    except Exception as e:
                        print(
                            f"\n--- ERROR DURING TRAINING UPDATE (Step: {getattr(self.trainer, 'global_step', 'N/A')}) ---"
                        )
                        traceback.print_exc()
                        print("--- Pausing training due to error. ---")
                        self.is_training = False
                        self.status = "Error: Training Step Failed"
                        self.app_state = "Error"

        elif self.app_state == "Playing":
            if self.demo_env and hasattr(self.demo_env, "_update_timers"):
                self.demo_env._update_timers()
            self.status = "Playing Demo"
        # Other states (Initializing, Error) have status set elsewhere

    def _render(self):
        """Renders the UI based on the current application state."""
        stats_summary = {}
        plot_data: Dict[str, Deque] = {}
        buffer_capacity = 0

        if self.stats_recorder:
            current_step = getattr(self.trainer, "global_step", 0)
            try:
                stats_summary = self.stats_recorder.get_summary(current_step)
            except Exception as e:
                print(f"Error getting stats summary: {e}")
                stats_summary = {"global_step": current_step}
            try:
                plot_data = self.stats_recorder.get_plot_data()
            except Exception as e:
                print(f"Error getting plot data: {e}")
                plot_data = {}
        elif self.app_state == "Error":
            stats_summary = {"global_step": getattr(self.trainer, "global_step", 0)}

        if self.buffer:
            try:
                buffer_capacity = getattr(self.buffer, "capacity", 0)
            except Exception:
                buffer_capacity = 0

        if not self.renderer:
            print("Error: Renderer not initialized in _render.")
            try:  # Basic error render
                self.screen.fill((0, 0, 0))
                font = pygame.font.SysFont(None, 50)
                surf = font.render("Renderer Error!", True, (255, 0, 0))
                self.screen.blit(
                    surf, surf.get_rect(center=self.screen.get_rect().center)
                )
                pygame.display.flip()
            except Exception:
                pass
            return

        try:
            self.renderer.render_all(
                app_state=self.app_state,
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
                tensorboard_log_dir=(
                    self.tensorboard_config.LOG_DIR
                    if self.tensorboard_config.LOG_DIR
                    else None
                ),
                plot_data=plot_data,
                demo_env=self.demo_env,
            )
        except Exception as render_all_err:
            print(f"CRITICAL ERROR in renderer.render_all: {render_all_err}")
            traceback.print_exc()
            try:
                self.app_state = "Error"
                self.status = "Render Error"
                self.renderer._render_error_screen(self.status)
            except Exception as e:
                print(f"Error rendering error screen: {e}")

        # Clear transient message
        if time.time() - self.last_cleanup_message_time >= 5.0:
            self.cleanup_message = ""

    def _perform_cleanup(self):
        """Handles final cleanup of resources."""
        print("Exiting application...")
        if self.trainer:
            print("Performing final trainer cleanup...")
            try:
                save_on_exit = self.status != "Cleaning" and self.app_state != "Error"
                self.trainer.cleanup(save_final=save_on_exit)
            except Exception as final_cleanup_err:
                print(f"Error during final trainer cleanup: {final_cleanup_err}")
        elif self.stats_recorder:  # Close stats recorder even if trainer failed/missing
            print("Closing stats recorder...")
            try:
                self.stats_recorder.close()
            except Exception as log_e:
                print(f"Error closing stats recorder on exit: {log_e}")

        pygame.quit()
        print("Application exited.")

    def run(self):
        """Main application loop."""
        print("Starting main application loop...")
        running = True
        try:
            while running:
                start_frame_time = time.perf_counter()

                # Handle Input
                if self.input_handler:
                    try:
                        running = self.input_handler.handle_input(
                            self.app_state, self.cleanup_confirmation_active
                        )
                    except Exception as input_err:
                        print(
                            f"\n--- UNHANDLED ERROR IN INPUT LOOP ({self.app_state}) ---"
                        )
                        traceback.print_exc()
                        running = False  # Exit on unhandled input error
                else:  # Basic exit if handler failed
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_ESCAPE
                        ):
                            if self.app_state == "Playing":
                                self._exit_demo_mode()
                            elif not self.cleanup_confirmation_active:
                                running = False
                    if not running:
                        break

                if not running:
                    break

                # Update State
                try:
                    self._update()
                except Exception as update_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN UPDATE LOOP ({self.app_state}) ---"
                    )
                    traceback.print_exc()
                    self.status = "Error: Update Loop Failed"
                    self.app_state = "Error"
                    self.is_training = False

                # Render Frame
                try:
                    self._render()
                except Exception as render_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN RENDER LOOP ({self.app_state}) ---"
                    )
                    traceback.print_exc()
                    self.status = "Error: Render Loop Failed"
                    self.app_state = "Error"

                # Frame Rate Limiting
                frame_time = time.perf_counter() - start_frame_time
                target_frame_time = (
                    1.0 / self.vis_config.FPS if self.vis_config.FPS > 0 else 0
                )
                sleep_time = max(0, target_frame_time - frame_time)
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nCtrl+C detected. Exiting gracefully...")
        except Exception as e:
            print(f"\n--- UNHANDLED EXCEPTION IN MAIN LOOP ({self.app_state}) ---")
            traceback.print_exc()
            print("--- EXITING ---")
        finally:
            self._perform_cleanup()


# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure base directories exist before logger setup
    os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    os.makedirs(RUN_LOG_DIR, exist_ok=True)  # Ensure run-specific log dir exists

    log_filepath = os.path.join(RUN_LOG_DIR, "console_output.log")

    # Setup logging
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger = TeeLogger(log_filepath, original_stdout)
    sys.stdout = logger
    sys.stderr = logger

    app_instance = None
    exit_code = 0

    try:
        if run_pre_checks():
            app_instance = MainApp()
            app_instance.run()  # run() now handles cleanup internally
    except SystemExit as exit_err:
        print(f"Exiting due to SystemExit (Code: {getattr(exit_err, 'code', 'N/A')}).")
        exit_code = (
            getattr(exit_err, "code", 1)
            if isinstance(getattr(exit_err, "code", 1), int)
            else 1
        )
    except Exception as main_err:
        print("\n--- UNHANDLED EXCEPTION DURING APP INITIALIZATION OR RUN ---")
        traceback.print_exc()
        print("--- EXITING DUE TO ERROR ---")
        exit_code = 1
        # Ensure cleanup is attempted even if run() didn't finish
        if app_instance and hasattr(app_instance, "_perform_cleanup"):
            print("Attempting cleanup after main exception...")
            try:
                app_instance._perform_cleanup()
            except Exception as cleanup_err:
                print(f"Error during cleanup after main exception: {cleanup_err}")
    finally:
        # Restore logging
        if logger:
            final_app_state = getattr(app_instance, "app_state", "UNKNOWN")
            print(
                f"Restoring console output (Final App State: {final_app_state}). Full log saved to: {log_filepath}"
            )
            logger.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Console logging restored. Full log should be in: {log_filepath}")
        sys.exit(exit_code)  # Exit with appropriate code
