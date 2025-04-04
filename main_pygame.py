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
            log_dir = os.path.dirname(filepath)
            if log_dir:  # Ensure directory exists only if filepath includes one
                os.makedirs(log_dir, exist_ok=True)
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
    DemoConfig,  # NEW: Import DemoConfig
    DEVICE,
    RANDOM_SEED,
    BUFFER_SAVE_PATH,  # Imported (full path)
    MODEL_SAVE_PATH,  # Imported (full path)
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    RUN_LOG_DIR,  # Imported
    RUN_CHECKPOINT_DIR,  # <<<--- ADD THIS IMPORT
    get_config_dict,
    print_config_info_and_validate,
)

# Import core components & helpers
from environment.game_state import GameState, StateType  # Import StateType
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
    def __init__(self):
        print("Initializing Pygame Application...")
        set_random_seeds(RANDOM_SEED)
        pygame.init()
        pygame.font.init()

        # --- Instantiate config classes ---
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        self.dqn_config = DQNConfig()
        self.train_config = TrainConfig()
        self.buffer_config = BufferConfig()
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()  # NEW
        self.reward_config = (
            RewardConfig()
        )  # Need instance for demo step penalty access
        # --- END MODIFIED ---

        self.num_envs = self.env_config.NUM_ENVS
        self.config_dict = get_config_dict()

        # --- Ensure directories exist using imported paths ---
        os.makedirs(RUN_CHECKPOINT_DIR, exist_ok=True)  # Use imported constant
        os.makedirs(RUN_LOG_DIR, exist_ok=True)  # Use imported constant
        print_config_info_and_validate()
        # --- END MODIFIED ---

        # Pygame setup
        self.screen = pygame.display.set_mode(
            (self.vis_config.SCREEN_WIDTH, self.vis_config.SCREEN_HEIGHT),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption("TriCrack DQN - TensorBoard & Demo")
        self.clock = pygame.time.Clock()

        # --- MODIFIED: App state ---
        self.app_state = "Initializing"  # Start in initializing state
        self.is_training = False
        self.cleanup_confirmation_active = False
        self.last_cleanup_message_time = 0.0
        self.cleanup_message = ""
        self.status = "Initializing"  # Overall RL status
        # --- END MODIFIED ---

        # --- Init components sequentially ---
        self.renderer = None  # Init later
        self.input_handler = None  # Init later
        self.envs: List[GameState] = []
        self.agent: Optional[DQNAgent] = None
        self.buffer: Optional[ReplayBufferBase] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.trainer: Optional[Trainer] = None
        self.demo_env: Optional[GameState] = None  # NEW: Demo environment instance

        # --- Start Initialization ---
        # Render initializing screen immediately
        try:
            temp_renderer = UIRenderer(
                self.screen, self.vis_config
            )  # Temporary for init screen
            temp_renderer.render_all(
                app_state=self.app_state,  # Pass state
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
            pygame.time.delay(100)  # Short delay
        except Exception as init_render_e:
            print(f"Error during initial render: {init_render_e}")
            # Proceed anyway, hope the main renderer works

        # Init Renderer FIRST (official one)
        self.renderer = UIRenderer(self.screen, self.vis_config)

        # Init RL components using helpers
        self._initialize_rl_components()  # This can take time

        # --- NEW: Initialize Demo Env ---
        self._initialize_demo_env()

        # Init Input Handler (after components are ready)
        self.input_handler = InputHandler(
            self.screen,
            self.renderer,
            self._toggle_training,
            self._request_cleanup,
            self._cancel_cleanup,
            self._confirm_cleanup,
            self._exit_app,
            # --- NEW: Pass demo callbacks ---
            self._start_demo_mode,
            self._exit_demo_mode,
            self._handle_demo_input,
            # --- END NEW ---
        )

        # --- Transition to Main Menu ---
        self.app_state = "MainMenu"
        self.status = "Paused"  # Initial status in MainMenu
        print("Initialization Complete. Ready.")
        print(f"--- tensorboard --logdir {os.path.abspath(BASE_LOG_DIR)} ---")

    def _initialize_rl_components(self):
        """Orchestrates the initialization of RL components using helpers."""
        print("Initializing RL components...")
        start_time = time.time()
        try:
            # --- MODIFIED: Remove global and path redefinition, just ensure dir exists ---
            # global BUFFER_SAVE_PATH, MODEL_SAVE_PATH # REMOVED
            # BUFFER_SAVE_PATH = os.path.join(...) # REMOVED
            # MODEL_SAVE_PATH = os.path.join(...) # REMOVED
            # print(f"  Buffer Save Path: {BUFFER_SAVE_PATH}") # REMOVED (paths printed by trainer now)
            # print(f"  Model Save Path: {MODEL_SAVE_PATH}") # REMOVED
            # Ensure the checkpoint directory exists (already done in __init__)
            # os.makedirs(RUN_CHECKPOINT_DIR, exist_ok=True) # Ensure dir exists
            # --- END MODIFIED ---

            self.envs = initialize_envs(self.num_envs, self.env_config)
            self.agent, self.buffer = initialize_agent_buffer(
                self.model_config, self.dqn_config, self.env_config, self.buffer_config
            )
            # Make sure buffer is assigned before stats recorder potentially uses it
            if self.buffer is None:
                raise RuntimeError("Buffer initialization failed unexpectedly.")

            self.stats_recorder = initialize_stats_recorder(
                self.stats_config,
                self.tensorboard_config,
                self.config_dict,
                self.agent,
                self.env_config,
            )
            # Make sure stats recorder is assigned before trainer potentially uses it
            if self.stats_recorder is None:
                raise RuntimeError("Stats Recorder initialization failed unexpectedly.")

            # Trainer now receives the correct, globally defined MODEL_SAVE_PATH and BUFFER_SAVE_PATH
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
            print(f"FATAL ERROR during RL component initialization: {e}")
            traceback.print_exc()
            pygame.quit()
            sys.exit(1)

    # --- NEW: Initialize Demo Env ---
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
            self.demo_env = None  # Ensure it's None on failure
            # Decide if this is fatal or just disables demo mode
            print("Warning: Demo mode may be unavailable due to initialization error.")

    # --- END NEW ---

    # --- Input Handler Callbacks ---
    def _toggle_training(self):
        # Only allow toggling from MainMenu
        if self.app_state != "MainMenu":
            return
        self.is_training = not self.is_training
        print(f"Training {'STARTED' if self.is_training else 'PAUSED'}")
        if not self.is_training:
            self._try_save_checkpoint()

    def _request_cleanup(self):
        # Allow cleanup request from MainMenu
        if self.app_state != "MainMenu":
            return
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
        # Perform cleanup, stay in MainMenu afterwards
        self._cleanup_data()
        # Ensure status is Paused after cleanup if successful
        if self.status != "Error":
            self.status = "Paused"
        self.app_state = "MainMenu"  # Ensure we are back here

    def _exit_app(self) -> bool:
        return False  # Signal exit

    # --- NEW: Demo Mode Callbacks ---
    def _start_demo_mode(self):
        if self.demo_env is None:
            print("Cannot start demo mode: Demo environment failed to initialize.")
            return
        if self.app_state == "MainMenu":
            print("Entering Demo Mode...")
            self.is_training = False  # Ensure training is paused
            self._try_save_checkpoint()
            self.app_state = "Playing"
            self.status = "Playing Demo"
            # Reset the demo env for a fresh start each time
            self.demo_env.reset()

    def _exit_demo_mode(self):
        if self.app_state == "Playing":
            print("Exiting Demo Mode...")
            self.app_state = "MainMenu"
            self.status = "Paused"  # Return to paused state

    def _handle_demo_input(self, event: pygame.event.Event):
        """Handles keyboard input during demo mode."""
        if self.app_state != "Playing" or self.demo_env is None:
            return  # Exit if not playing or no demo env

        # Do not process input if game is frozen or over, except ESC handled in main loop
        if self.demo_env.is_frozen() or self.demo_env.is_over():
            # Allow ESC to exit even if game over, handled in InputHandler
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
            elif event.key == pygame.K_q:  # Cycle shape left
                self.demo_env.cycle_shape(-1)
                action_taken = True
            elif event.key == pygame.K_e:  # Cycle shape right
                self.demo_env.cycle_shape(1)
                action_taken = True
            elif event.key == pygame.K_SPACE:
                # Attempt to place the selected shape
                action_index = self.demo_env.get_action_for_current_selection()
                if action_index is not None:
                    # --- Collect experience BEFORE stepping ---
                    state_before_step = self.demo_env.get_state()
                    # Perform the step
                    reward, done = self.demo_env.step(action_index)
                    # Get the state *after* the step
                    next_state_after_step = self.demo_env.get_state()
                    # --- Push to MAIN buffer ---
                    if self.buffer is not None:
                        try:
                            # Use the captured states and action index
                            self.buffer.push(
                                state_before_step,
                                action_index,
                                reward,
                                next_state_after_step,
                                done,
                            )
                            # Optional: Log buffer size increase for demo feedback
                            if (
                                len(self.buffer) % 20 == 0
                                and len(self.buffer)
                                <= self.train_config.LEARN_START_STEP
                            ):
                                print(
                                    f"Demo play added experience. Buffer size: {len(self.buffer)} / {self.buffer.capacity}"
                                )
                        except Exception as buf_e:
                            print(f"Error pushing demo experience to buffer: {buf_e}")
                            traceback.print_exc()
                    else:
                        print(
                            "Warning: Replay buffer not available to store demo experience."
                        )
                    action_taken = True
                else:
                    # Invalid placement visual feedback? (Handled by preview color)
                    print("Demo: Invalid placement.")
                    # Input was processed, even if invalid (prevents double input)
                    action_taken = True

            # Optional: Add logic for game over reset in demo?
            if self.demo_env.is_over():
                print("Demo: Game Over! Press ESC to exit.")
                # Maybe add a small delay then allow ESC to exit?
                # The input handler already allows ESC to exit demo mode.

    # --- END NEW ---

    # --- Other Methods (Cleanup, Save) ---
    def _cleanup_data(self):
        """Deletes current run's checkpoint/buffer and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        original_state = self.app_state
        self.app_state = "Initializing"  # Show initializing screen during cleanup
        self.is_training = False
        self.status = "Cleaning"
        self.cleanup_confirmation_active = False
        messages = []

        # Render initializing screen
        if self.renderer:
            self.renderer.render_all(
                app_state=self.app_state,  # Pass state
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
            )  # Pass demo env too
            pygame.time.delay(100)
        else:
            print("Warning: Renderer not available during cleanup start.")

        # Close trainer/stats FIRST
        if hasattr(self, "trainer") and self.trainer:
            print("Running trainer cleanup...")
            try:
                self.trainer.cleanup(save_final=False)
                # Don't set trainer to None yet, _initialize_rl_components will replace it
            except Exception as e:
                print(f"Error during trainer cleanup: {e}")
                traceback.print_exc()  # Print stack trace for trainer errors

        # Ensure stats recorder is closed even if trainer cleanup failed
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            print("Closing stats recorder...")
            try:
                self.stats_recorder.close()
                # Don't set stats_recorder to None yet
            except Exception as log_e:
                print(f"Error closing stats recorder during cleanup: {log_e}")
                traceback.print_exc()  # Print stack trace for stats errors

        # Delete files - Use the globally defined paths directly
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
            # Re-init RL components (agent, buffer, trainer, stats)
            # This will create new instances and assign them to self.agent, etc.
            self._initialize_rl_components()
            # Demo env remains as is, maybe reset it?
            if self.demo_env:
                self.demo_env.reset()

            print("RL components re-initialized.")
            messages.append("RL components re-initialized.")
            self.status = "Paused"  # Ready to go in main menu
            self.app_state = "MainMenu"
        except Exception as e:
            print(f"FATAL ERROR during RL component re-initialization: {e}")
            traceback.print_exc()
            self.status = "Error"
            self.app_state = "Error"  # Transition to Error state
            messages.append("ERROR RE-INITIALIZING RL COMPONENTS!")
            # Attempt to render error screen if possible
            if self.renderer:
                try:
                    self.renderer._render_error_screen(self.status)
                except:
                    pass  # Ignore render errors here

        self.cleanup_message = "\n".join(messages)
        self.last_cleanup_message_time = time.time()

        print("--- CLEANUP DATA COMPLETE ---")

    def _try_save_checkpoint(self):
        """Saves checkpoint if not training and trainer exists."""
        # Only save if in main menu and paused
        if (
            self.app_state == "MainMenu"
            and not self.is_training
            and hasattr(self, "trainer")
            and self.trainer
        ):
            print("Saving checkpoint on pause...")
            try:
                self.trainer.maybe_save_checkpoint(force_save=True)
            except Exception as e:
                print(f"Error saving checkpoint on pause: {e}")
                traceback.print_exc()

    def _update(self):
        """Updates the application state and performs training steps (only in MainMenu)."""

        # --- Update logic specific to MainMenu state ---
        if self.app_state == "MainMenu":
            if self.cleanup_confirmation_active:
                self.status = "Confirm Cleanup"
            elif not self.is_training and self.status != "Error":
                self.status = "Paused"
            elif not hasattr(self, "trainer") or self.trainer is None:
                if self.status != "Error":
                    self.status = "Error: Trainer Missing"  # More specific error
                    print("Error: Trainer object not found during update.")
            # Check buffer size AFTER checking trainer existence
            elif (
                self.trainer is not None
                and self.buffer is not None
                and (
                    len(self.buffer) < self.train_config.LEARN_START_STEP
                    or self.trainer.global_step < self.train_config.LEARN_START_STEP
                )
            ):
                # Update status based on buffer fill progress vs learn start step
                current_fill = len(self.buffer) if self.buffer else 0
                current_step = self.trainer.global_step if self.trainer else 0
                needed = self.train_config.LEARN_START_STEP
                # Avoid division by zero if needed is 0
                percent_steps = (current_step / needed * 100) if needed > 0 else 100
                percent_buffer = (current_fill / needed * 100) if needed > 0 else 100
                fill_percent = min(
                    percent_steps, percent_buffer
                )  # Use the minimum progress
                self.status = f"Buffering ({fill_percent:.0f}%)"
            elif self.is_training:
                self.status = "Training"
            else:  # Should be paused if not training and buffer is full enough
                # Check if we were previously buffering to avoid flickering
                if not self.status.startswith("Buffering"):
                    self.status = "Paused"

            # Only step trainer if actively training
            if self.status == "Training":
                if not hasattr(self, "trainer") or self.trainer is None:
                    print("Error: Trainer became unavailable during _update.")
                    self.status = "Error: Trainer Lost"
                    self.is_training = False
                    return

                try:
                    step_start_time = time.time()
                    self.trainer.step()  # Trainer steps the *vectorized* envs
                    step_duration = time.time() - step_start_time
                    if self.vis_config.VISUAL_STEP_DELAY > 0:
                        time.sleep(
                            max(0, self.vis_config.VISUAL_STEP_DELAY - step_duration)
                        )
                except Exception as e:
                    print(
                        f"\n--- ERROR DURING TRAINING UPDATE (Step: {getattr(self.trainer, 'global_step', 'N/A')}) ---"
                    )
                    traceback.print_exc()
                    print(f"--- Pausing training due to error. ---")
                    self.is_training = False
                    self.status = "Error: Training Step Failed"
                    self.app_state = "Error"  # Go to error state

        # --- Update logic specific to Playing state ---
        elif self.app_state == "Playing":
            # Update demo env timers if needed (e.g., blink/freeze)
            if self.demo_env:
                self.demo_env._update_timers()  # Needs to be called periodically
            self.status = "Playing Demo"  # Keep status consistent

        # --- Update logic for other states (e.g., Initializing) ---
        elif self.app_state == "Initializing":
            self.status = "Initializing"
            # No game/trainer updates needed here

        elif self.app_state == "Error":
            # Status might have been set more specifically by the error source
            # self.status = "Error"
            pass  # No updates needed in error state

    def _render(self):
        """Renders the UI based on the current application state."""
        stats_summary = {}
        plot_data: Dict[str, Deque] = {}
        buffer_capacity = 0

        # Gather stats data primarily for MainMenu, but pass empty if not available
        # Check if stats_recorder exists and has the necessary methods
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            current_step = getattr(self.trainer, "global_step", 0)
            if hasattr(self.stats_recorder, "get_summary"):
                try:
                    stats_summary = self.stats_recorder.get_summary(current_step)
                except Exception as e:
                    print(f"Error getting stats summary: {e}")
                    stats_summary = {
                        "global_step": current_step
                    }  # Provide minimal summary on error
            else:
                stats_summary = {
                    "global_step": current_step
                }  # Minimal summary if method missing

            if hasattr(self.stats_recorder, "get_plot_data"):
                try:
                    plot_data = self.stats_recorder.get_plot_data()
                except Exception as e:
                    print(f"Error getting plot data: {e}")
                    plot_data = {}
            else:
                plot_data = {}

        elif (
            self.app_state == "Error"
        ):  # Also get step count if possible in error state
            stats_summary = {"global_step": getattr(self.trainer, "global_step", 0)}

        # Get buffer capacity if buffer exists
        if hasattr(self, "buffer") and self.buffer:
            # Check if buffer has capacity attribute, handle potential errors
            try:
                buffer_capacity = getattr(self.buffer, "capacity", 0)
            except Exception:
                buffer_capacity = 0  # Default if error accessing capacity

        if not hasattr(self, "renderer") or self.renderer is None:
            print("Error: Renderer not initialized in _render.")
            # Attempt to draw a basic error message directly?
            try:
                self.screen.fill((0, 0, 0))
                font = pygame.font.SysFont(None, 50)
                surf = font.render("Renderer Error!", True, (255, 0, 0))
                self.screen.blit(
                    surf, surf.get_rect(center=self.screen.get_rect().center)
                )
                pygame.display.flip()
            except Exception:
                pass  # If Pygame itself is broken, not much we can do
            return

        # Call the main render function, passing the state
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
                demo_env=self.demo_env,  # Pass demo env
            )
        except Exception as render_all_err:
            print(f"CRITICAL ERROR in renderer.render_all: {render_all_err}")
            traceback.print_exc()
            # Attempt to render error screen
            try:
                self.app_state = "Error"
                self.status = "Render Error"
                self.renderer._render_error_screen(self.status)
            except:
                pass  # Ignore errors during error rendering

        # Clear transient message after timeout (applies to cleanup message)
        if time.time() - self.last_cleanup_message_time >= 5.0:
            self.cleanup_message = ""

    def run(self):
        """Main application loop."""
        print("Starting main application loop...")
        running = True
        try:
            while running:
                start_frame_time = time.perf_counter()

                # --- Pass app_state to input handler ---
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
                        # Potentially go to error state? Or just log and continue?
                        # Let's log and continue for now unless it becomes persistent.
                else:  # Handle basic exit events if input_handler failed
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_ESCAPE
                        ):
                            # Allow ESC to exit even if input handler is broken
                            # Exit app directly from most states, exit demo mode from Playing state
                            if self.app_state == "Playing":
                                self._exit_demo_mode()  # Try to cleanly exit demo
                            elif not self.cleanup_confirmation_active:
                                running = False

                if not running:
                    break

                try:
                    self._update()  # Update logic is now state-dependent
                except Exception as update_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN UPDATE LOOP ({self.app_state}) ---"
                    )
                    traceback.print_exc()
                    print(f"--- Setting status to Error ---")
                    self.status = "Error: Update Loop Failed"
                    self.app_state = "Error"  # Transition to Error state
                    self.is_training = False

                try:
                    # Render based on state
                    if self.app_state == "Error":
                        if self.renderer:
                            self.renderer._render_error_screen(self.status)
                        # Skip other rendering if in error state
                    else:
                        self._render()  # Render logic is now state-dependent

                except Exception as render_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN RENDER LOOP ({self.app_state}) ---"
                    )
                    traceback.print_exc()
                    self.status = "Error: Render Loop Failed"
                    self.app_state = "Error"  # Transition to Error state

                # Frame rate limiting
                frame_time = time.perf_counter() - start_frame_time
                target_frame_time = (
                    1.0 / self.vis_config.FPS if self.vis_config.FPS > 0 else 0
                )
                sleep_time = max(0, target_frame_time - frame_time)
                if sleep_time > 0.001:  # Avoid tiny sleeps
                    time.sleep(sleep_time)
                # Alternative: self.clock.tick(self.vis_config.FPS if self.vis_config.FPS > 0 else 0)

        except KeyboardInterrupt:
            print("\nCtrl+C detected. Exiting gracefully...")
        except Exception as e:
            print(f"\n--- UNHANDLED EXCEPTION IN MAIN LOOP ({self.app_state}) ---")
            traceback.print_exc()
            print("--- EXITING ---")
        finally:
            print("Exiting application...")
            # --- Ensure trainer cleanup happens if it exists ---
            if hasattr(self, "trainer") and self.trainer:
                print("Performing final trainer cleanup...")
                try:
                    # Don't save if cleanup was in progress or error state?
                    save_on_exit = (
                        self.status != "Cleaning" and self.app_state != "Error"
                    )
                    self.trainer.cleanup(save_final=save_on_exit)
                except Exception as final_cleanup_err:
                    print(f"Error during final trainer cleanup: {final_cleanup_err}")
                    traceback.print_exc()
            # --- Close stats recorder if it exists and wasn't closed by trainer ---
            elif hasattr(self, "stats_recorder") and self.stats_recorder:
                print("Closing stats recorder...")
                try:
                    self.stats_recorder.close()
                except Exception as log_e:
                    print(f"Error closing stats recorder on exit: {log_e}")
                    traceback.print_exc()

            pygame.quit()
            print("Application exited.")


if __name__ == "__main__":
    # Ensure base directories exist *before* setting up logger
    os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    # Ensure run-specific log dir exists *before* logger init
    os.makedirs(RUN_LOG_DIR, exist_ok=True)

    log_filepath = os.path.join(RUN_LOG_DIR, "console_output.log")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger = TeeLogger(log_filepath, original_stdout)
    sys.stdout = logger
    sys.stderr = logger  # Redirect stderr as well

    app_instance = None  # Hold the app instance for potential final logging

    try:
        if run_pre_checks():
            app_instance = MainApp()
            app_instance.run()
    except SystemExit as exit_err:
        print(
            f"Exiting due to SystemExit (Code: {getattr(exit_err, 'code', 'N/A')}, likely from pre-checks or init error)."
        )
    except Exception as main_err:
        print("\n--- UNHANDLED EXCEPTION DURING APP INITIALIZATION OR RUN ---")
        traceback.print_exc()
        print("--- EXITING DUE TO ERROR ---")
    finally:
        # Ensure logging is restored even if app init failed
        if "logger" in locals() and logger:
            # Try to get the final app state if available
            final_app_state = getattr(app_instance, "app_state", "UNKNOWN")
            print(
                f"Restoring console output (Final App State: {final_app_state}). Full log saved to: {log_filepath}"
            )
            logger.close()  # Close the log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Now print the final message to the actual console
        print(f"Console logging restored. Full log should be in: {log_filepath}")
