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
    DemoConfig,
    DEVICE,
    RANDOM_SEED,
    BUFFER_SAVE_PATH,
    MODEL_SAVE_PATH,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    RUN_LOG_DIR,
    RUN_CHECKPOINT_DIR,
    get_config_dict,
    print_config_info_and_validate,
)

# Import core components & helpers
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
        self.demo_config = DemoConfig()
        self.reward_config = RewardConfig()
        # --- END MODIFIED ---

        self.num_envs = self.env_config.NUM_ENVS
        self.config_dict = get_config_dict()

        # --- Ensure directories exist using imported paths ---
        os.makedirs(RUN_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(RUN_LOG_DIR, exist_ok=True)
        print_config_info_and_validate()
        # --- END MODIFIED ---

        # Pygame setup
        self.screen = pygame.display.set_mode(
            (self.vis_config.SCREEN_WIDTH, self.vis_config.SCREEN_HEIGHT),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption("TriCrack DQN - TensorBoard & Demo")
        self.clock = pygame.time.Clock()

        # --- App state ---
        self.app_state = "Initializing"
        self.is_training = False
        self.cleanup_confirmation_active = False # *** This flag controls the overlay ***
        self.last_cleanup_message_time = 0.0
        self.cleanup_message = ""
        self.status = "Initializing"

        # --- Init components sequentially ---
        self.renderer = None
        self.input_handler = None
        self.envs: List[GameState] = []
        self.agent: Optional[DQNAgent] = None
        self.buffer: Optional[ReplayBufferBase] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.trainer: Optional[Trainer] = None
        self.demo_env: Optional[GameState] = None

        # --- Start Initialization ---
        # Render initializing screen immediately
        try:
            temp_renderer = UIRenderer(self.screen, self.vis_config)
            temp_renderer.render_all(
                app_state=self.app_state,
                is_training=False,
                status=self.status,
                stats_summary={},
                buffer_capacity=0,
                envs=[],
                num_envs=0,
                env_config=self.env_config,
                cleanup_confirmation_active=False, # Initially false
                cleanup_message="",
                last_cleanup_message_time=0,
                tensorboard_log_dir=None,
                plot_data={},
                demo_env=None,
            )
            pygame.time.delay(100)
        except Exception as init_render_e:
            print(f"Error during initial render: {init_render_e}") # LOG

        # Init Renderer FIRST
        self.renderer = UIRenderer(self.screen, self.vis_config)

        # Init RL components using helpers
        self._initialize_rl_components()

        # Init Demo Env
        self._initialize_demo_env()

        # Init Input Handler
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

        # Transition to Main Menu
        self.app_state = "MainMenu"
        self.status = "Paused"
        print("Initialization Complete. Ready.") # LOG
        print(f"--- tensorboard --logdir {os.path.abspath(BASE_LOG_DIR)} ---") # LOG

    def _initialize_rl_components(self):
        """Orchestrates the initialization of RL components using helpers."""
        print("Initializing RL components...") # LOG
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
            print(f"RL components initialized in {time.time() - start_time:.2f}s") # LOG
        except Exception as e:
            print(f"FATAL ERROR during RL component initialization: {e}") # LOG
            traceback.print_exc()
            pygame.quit()
            sys.exit(1)

    def _initialize_demo_env(self):
        """Initializes the separate environment for demo mode."""
        print("Initializing Demo Environment...") # LOG
        try:
            self.demo_env = GameState()
            self.demo_env.reset()
            print("Demo environment initialized.") # LOG
        except Exception as e:
            print(f"ERROR initializing demo environment: {e}") # LOG
            traceback.print_exc()
            self.demo_env = None
            print("Warning: Demo mode may be unavailable due to initialization error.") # LOG

    # --- Input Handler Callbacks ---
    def _toggle_training(self):
        if self.app_state != "MainMenu":
            print(
                f"[MainApp::_toggle_training] Ignored, state is {self.app_state}"
            ) # LOG
            return
        self.is_training = not self.is_training
        print(
            f"[MainApp::_toggle_training] Training {'STARTED' if self.is_training else 'PAUSED'}"
        ) # LOG
        if not self.is_training:
            self._try_save_checkpoint()

    def _request_cleanup(self):
        print("[MainApp::_request_cleanup] Entered.") # LOG
        if self.app_state != "MainMenu":
            print(
                f"[MainApp::_request_cleanup] Not in MainMenu (State: {self.app_state}), returning."
            ) # LOG
            return
        was_training = self.is_training
        self.is_training = False  # Pause training
        if was_training:
            self._try_save_checkpoint()  # Save if was training

        # *** KEY CHANGE: Set the flag to activate the overlay ***
        self.cleanup_confirmation_active = True
        print(
            f"[MainApp::_request_cleanup] Set cleanup_confirmation_active = {self.cleanup_confirmation_active}. Training paused. Confirm action."
        ) # LOG

    def _cancel_cleanup(self):
        print("[MainApp::_cancel_cleanup] Entered.") # LOG
        # *** KEY CHANGE: Set flag FIRST to stop overlay rendering immediately ***
        self.cleanup_confirmation_active = False
        self.cleanup_message = "Cleanup cancelled."
        self.last_cleanup_message_time = time.time()
        print("[MainApp::_cancel_cleanup] Cleanup cancelled by user.") # LOG

    def _confirm_cleanup(self):
        print(
            "[MainApp::_confirm_cleanup] Cleanup confirmed by user. Starting process..."
        ) # LOG
        # Perform cleanup. _cleanup_data handles setting status/state appropriately.
        try:
            self._cleanup_data()  # This handles state transitions internally now
        except Exception as e:
            print(f"FATAL ERROR during _confirm_cleanup -> _cleanup_data call: {e}") # LOG
            traceback.print_exc()
            self.status = "Error: Cleanup Failed Critically"
            self.app_state = "Error"
            # Ensure confirmation flag is false even on critical error here
            self.cleanup_confirmation_active = False
        finally:
            # *** KEY CHANGE: Ensure the confirmation flag is ALWAYS false after attempting cleanup ***
            self.cleanup_confirmation_active = False
            print(
                f"[MainApp::_confirm_cleanup] Cleanup process finished. Final app state: {self.app_state}, Status: {self.status}"
            ) # LOG

    def _exit_app(self) -> bool:
        print("[MainApp::_exit_app] Exit requested.") # LOG
        return False  # Signal exit

    # --- Demo Mode Callbacks ---
    def _start_demo_mode(self):
        print("[MainApp::_start_demo_mode] Entered.") # LOG
        if self.demo_env is None:
            print(
                "[MainApp::_start_demo_mode] Cannot start demo mode: Demo environment failed to initialize."
            ) # LOG
            return
        if self.app_state == "MainMenu":
            print("[MainApp::_start_demo_mode] Entering Demo Mode...") # LOG
            self.is_training = False
            self._try_save_checkpoint()
            self.app_state = "Playing"
            self.status = "Playing Demo"
            self.demo_env.reset()
        else:
            print(
                f"[MainApp::_start_demo_mode] Ignored, state is {self.app_state}"
            ) # LOG

    def _exit_demo_mode(self):
        print("[MainApp::_exit_demo_mode] Entered.") # LOG
        if self.app_state == "Playing":
            print("[MainApp::_exit_demo_mode] Exiting Demo Mode...") # LOG
            self.app_state = "MainMenu"
            self.status = "Paused"
        else:
            print(
                f"[MainApp::_exit_demo_mode] Ignored, state is {self.app_state}"
            ) # LOG

    def _handle_demo_input(self, event: pygame.event.Event):
        """Handles keyboard input during demo mode."""
        if self.app_state != "Playing" or self.demo_env is None:
            return

        if self.demo_env.is_frozen() or self.demo_env.is_over():
            return

        if event.type == pygame.KEYDOWN:
            # print(f"[MainApp::_handle_demo_input] Key down: {event.key}") # LOG (Optional, can be spammy)
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
                    state_before_step = self.demo_env.get_state()
                    reward, done = self.demo_env.step(action_index)
                    next_state_after_step = self.demo_env.get_state()
                    if self.buffer is not None:
                        try:
                            self.buffer.push(
                                state_before_step,
                                action_index,
                                reward,
                                next_state_after_step,
                                done,
                            )
                            if (
                                len(self.buffer) % 50 == 0
                                and len(self.buffer)
                                <= self.train_config.LEARN_START_STEP
                            ):
                                print(
                                    f"[Demo] Added experience. Buffer: {len(self.buffer)}/{self.buffer.capacity}"
                                ) # LOG
                        except Exception as buf_e:
                            print(f"Error pushing demo experience to buffer: {buf_e}") # LOG
                            traceback.print_exc()
                    else:
                        print("Warning: Replay buffer not available.") # LOG
                    action_taken = True
                else:
                    # print("[Demo] Invalid placement.") # LOG (Optional)
                    action_taken = True

            if self.demo_env.is_over():
                print("[Demo] Game Over! Press ESC to exit.") # LOG

    # --- Other Methods (Cleanup, Save) ---
    def _cleanup_data(self):
        """Deletes current run's checkpoint/buffer and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---") # LOG
        # *** KEY CHANGE: Set initial state for cleanup ***
        self.app_state = "Initializing" # Show initializing screen during cleanup
        self.is_training = False
        self.status = "Cleaning"
        # Confirmation flag handled by caller (_confirm_cleanup)

        messages = []

        # *** KEY CHANGE: Render initializing screen immediately ***
        if self.renderer:
            try: # Add try-except for rendering during cleanup
                self.renderer.render_all(
                    app_state=self.app_state, # Use "Initializing" state
                    is_training=False,
                    status=self.status, # Use "Cleaning" status
                    stats_summary={},
                    buffer_capacity=0,
                    envs=[],
                    num_envs=0,
                    env_config=self.env_config,
                    cleanup_confirmation_active=False, # Explicitly false during cleanup itself
                    cleanup_message="",
                    last_cleanup_message_time=0,
                    tensorboard_log_dir=None,
                    plot_data={},
                    demo_env=self.demo_env, # Pass demo env if available
                )
                pygame.display.flip() # Ensure it draws
                pygame.time.delay(100) # Small delay to show the screen
            except Exception as render_err:
                print(
                    f"Warning: Error rendering during cleanup start: {render_err}"
                ) # LOG
        else:
            print("Warning: Renderer not available during cleanup start.") # LOG

        # Close trainer/stats FIRST
        if hasattr(self, "trainer") and self.trainer:
            print("[Cleanup] Running trainer cleanup...") # LOG
            try:
                self.trainer.cleanup(save_final=False) # Don't save during cleanup
            except Exception as e:
                print(f"Error during trainer cleanup: {e}") # LOG
                traceback.print_exc()

        if hasattr(self, "stats_recorder") and self.stats_recorder:
            print("[Cleanup] Closing stats recorder...") # LOG
            try:
                self.stats_recorder.close()
            except Exception as log_e:
                print(f"Error closing stats recorder during cleanup: {log_e}") # LOG
                traceback.print_exc()

        # Delete files
        print("[Cleanup] Deleting files...") # LOG
        for path, desc in [
            (MODEL_SAVE_PATH, "Agent ckpt"),
            (BUFFER_SAVE_PATH, "Buffer state"),
        ]:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    msg = f"{desc} deleted: {os.path.basename(path)}"
                    print(f"  - {msg}") # LOG
                    messages.append(msg)
                else:
                    msg = f"{desc} not found (current run)."
                    print(f"  - {msg}") # LOG
            except OSError as e:
                msg = f"Error deleting {desc}: {e}"
                print(f"  - {msg}") # LOG
                messages.append(msg)

        time.sleep(0.1) # Short delay after file deletion

        print("[Cleanup] Re-initializing RL components...") # LOG
        try:
            # *** KEY CHANGE: Re-init RL components (can fail) ***
            self._initialize_rl_components()

            # Reset Demo env after successful re-init
            if self.demo_env:
                self.demo_env.reset()

            print("[Cleanup] RL components re-initialized successfully.") # LOG
            messages.append("RL components re-initialized.")
            # *** KEY CHANGE: Set state on SUCCESS ***
            self.status = "Paused"
            self.app_state = "MainMenu"

        except Exception as e:
            print(
                f"FATAL ERROR during RL component re-initialization after cleanup: {e}"
            ) # LOG
            traceback.print_exc()
            # *** KEY CHANGE: Set state on FAILURE ***
            self.status = "Error: Re-init Failed"
            self.app_state = "Error"
            messages.append("ERROR RE-INITIALIZING RL COMPONENTS!")

            # Attempt to render error screen
            if self.renderer:
                try:
                    self.renderer._render_error_screen(self.status)
                except Exception as render_err_final:
                    print(
                        f"Warning: Failed to render error screen after cleanup failure: {render_err_final}"
                    ) # LOG

        # Set message regardless of success/failure
        self.cleanup_message = "\n".join(messages)
        self.last_cleanup_message_time = time.time()

        print(
            f"--- CLEANUP DATA COMPLETE (Final State: {self.app_state}, Status: {self.status}) ---"
        ) # LOG

    def _try_save_checkpoint(self):
        """Saves checkpoint if not training and trainer exists."""
        if (
            self.app_state == "MainMenu"
            and not self.is_training
            and hasattr(self, "trainer")
            and self.trainer
        ):
            print("[MainApp] Saving checkpoint on pause...") # LOG
            try:
                self.trainer.maybe_save_checkpoint(force_save=True)
            except Exception as e:
                print(f"Error saving checkpoint on pause: {e}") # LOG
                traceback.print_exc()

    def _update(self):
        """Updates the application state and performs training steps (potentially)."""
        # print(f"[DEBUG Update] State: {self.app_state}, Training: {self.is_training}, Status: {self.status}, Buf: {len(self.buffer) if self.buffer else 'N/A'}, Step: {self.trainer.global_step if self.trainer else 'N/A'}") # DEBUG LOG
        should_step_trainer = False  # Flag to control trainer stepping

        # --- Update logic specific to MainMenu state ---
        if self.app_state == "MainMenu":
            # *** KEY CHANGE: Check confirmation flag FIRST ***
            if self.cleanup_confirmation_active:
                self.status = "Confirm Cleanup"
                # Do NOT step trainer or change status further
            elif not self.is_training and self.status != "Error":
                self.status = "Paused"
            elif not hasattr(self, "trainer") or self.trainer is None:
                if self.status != "Error":
                    self.status = "Error: Trainer Missing"
                    print("Error: Trainer object not found during update.") # LOG
            # --- Determine status based on training flag and buffer ---
            elif self.is_training:
                current_fill = len(self.buffer) if self.buffer else 0
                current_step = self.trainer.global_step if self.trainer else 0
                needed = self.train_config.LEARN_START_STEP
                is_buffer_ready_for_learning = (
                    current_fill >= needed and current_step >= needed
                )

                if not is_buffer_ready_for_learning:
                    percent_buffer = current_fill / max(1, needed) * 100
                    self.status = f"Buffering ({percent_buffer:.0f}%)"
                else:
                    self.status = "Training"

                # *** KEY CHANGE: Always schedule trainer step if is_training is True ***
                should_step_trainer = True
            else:  # Not training
                # Avoid overwriting "Buffering" or "Error" status when paused
                if not self.status.startswith("Buffering") and self.status != "Error":
                    self.status = "Paused"

            # --- Execute trainer step if scheduled ---
            if should_step_trainer:
                # print("[DEBUG Update] Should step trainer.") # DEBUG LOG
                if not hasattr(self, "trainer") or self.trainer is None:
                    print("Error: Trainer became unavailable during _update.") # LOG
                    self.status = "Error: Trainer Lost"
                    self.is_training = False
                else:
                    # print("[DEBUG Update] Calling trainer.step()") # DEBUG LOG
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
                        ) # LOG
                        traceback.print_exc()
                        print(f"--- Pausing training due to error. ---") # LOG
                        self.is_training = False
                        self.status = "Error: Training Step Failed"
                        self.app_state = "Error" # Transition to error state

        # --- Update logic specific to Playing state ---
        elif self.app_state == "Playing":
            if self.demo_env:
                if hasattr(self.demo_env, "_update_timers") and callable(
                    getattr(self.demo_env, "_update_timers")
                ):
                    self.demo_env._update_timers()
                else:
                    print("Warning: demo_env missing _update_timers method.") # LOG
            self.status = "Playing Demo"

        # --- Update logic for other states ---
        elif self.app_state == "Initializing":
            # Status is set during the cleanup/init process itself
            pass
        elif self.app_state == "Error":
            # Status is set when the error occurs
            pass

    def _render(self):
        """Renders the UI based on the current application state."""
        stats_summary = {}
        plot_data: Dict[str, Deque] = {}
        buffer_capacity = 0

        # Gather stats data
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            current_step = getattr(self.trainer, "global_step", 0)
            try:
                stats_summary = self.stats_recorder.get_summary(current_step)
            except Exception as e:
                print(f"Error getting stats summary: {e}") # LOG
                stats_summary = {"global_step": current_step}
            try:
                plot_data = self.stats_recorder.get_plot_data()
            except Exception as e:
                print(f"Error getting plot data: {e}") # LOG
                plot_data = {}
        elif self.app_state == "Error":
            stats_summary = {"global_step": getattr(self.trainer, "global_step", 0)}

        # Get buffer capacity
        if hasattr(self, "buffer") and self.buffer:
            try:
                buffer_capacity = getattr(self.buffer, "capacity", 0)
            except Exception:
                buffer_capacity = 0

        if not hasattr(self, "renderer") or self.renderer is None:
            print("Error: Renderer not initialized in _render.") # LOG
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

        # LOGGING: Check flag value before rendering
        # print(f"[MainApp::_render] Calling render_all. cleanup_confirmation_active = {self.cleanup_confirmation_active}, app_state = {self.app_state}") # LOG (Moved to inside render for better context)

        # Call the main render function
        try:
            # *** KEY CHANGE: Pass the cleanup_confirmation_active flag ***
            self.renderer.render_all(
                app_state=self.app_state,
                is_training=self.is_training,
                status=self.status,
                stats_summary=stats_summary,
                buffer_capacity=buffer_capacity,
                envs=(self.envs if hasattr(self, "envs") else []),
                num_envs=self.num_envs,
                env_config=self.env_config,
                cleanup_confirmation_active=self.cleanup_confirmation_active, # Pass the flag
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
            print(f"CRITICAL ERROR in renderer.render_all: {render_all_err}") # LOG
            traceback.print_exc()
            try:
                self.app_state = "Error"
                self.status = "Render Error"
                self.renderer._render_error_screen(self.status)
            except Exception as e:
                print(f"Error rendering error screen: {e}") # LOG

        # Clear transient message
        if time.time() - self.last_cleanup_message_time >= 5.0:
            self.cleanup_message = ""

    def run(self):
        """Main application loop."""
        print("Starting main application loop...") # LOG
        running = True
        try:
            while running:
                start_frame_time = time.perf_counter()

                # Handle Input
                if self.input_handler:
                    try:
                        # *** KEY CHANGE: Pass cleanup flag to input handler ***
                        running = self.input_handler.handle_input(
                            self.app_state, self.cleanup_confirmation_active
                        )
                    except Exception as input_err:
                        print(
                            f"\n--- UNHANDLED ERROR IN INPUT LOOP ({self.app_state}) ---"
                        ) # LOG
                        traceback.print_exc()
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
                            elif not self.cleanup_confirmation_active: # Only exit if overlay not active
                                running = False

                if not running:
                    break

                # Update State
                try:
                    self._update()
                except Exception as update_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN UPDATE LOOP ({self.app_state}) ---"
                    ) # LOG
                    traceback.print_exc()
                    print(f"--- Setting status to Error ---") # LOG
                    self.status = "Error: Update Loop Failed"
                    self.app_state = "Error"
                    self.is_training = False

                # Render Frame
                try:
                    # *** KEY CHANGE: Rendering is now handled entirely by _render ***
                    # which calls renderer.render_all, which handles states and overlays
                    self._render()
                    # No need for separate error screen rendering here anymore
                except Exception as render_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN RENDER LOOP ({self.app_state}) ---"
                    ) # LOG
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
            print("\nCtrl+C detected. Exiting gracefully...") # LOG
        except Exception as e:
            print(f"\n--- UNHANDLED EXCEPTION IN MAIN LOOP ({self.app_state}) ---") # LOG
            traceback.print_exc()
            print("--- EXITING ---") # LOG
        finally:
            print("Exiting application...") # LOG
            # Final Cleanup
            if hasattr(self, "trainer") and self.trainer:
                print("Performing final trainer cleanup...") # LOG
                try:
                    # *** KEY CHANGE: Don't save if cleanup was in progress or error state ***
                    save_on_exit = (
                        self.status != "Cleaning" and self.app_state != "Error"
                    )
                    self.trainer.cleanup(save_final=save_on_exit)
                except Exception as final_cleanup_err:
                    print(f"Error during final trainer cleanup: {final_cleanup_err}") # LOG
                    traceback.print_exc()
            elif hasattr(self, "stats_recorder") and self.stats_recorder:
                print("Closing stats recorder...") # LOG
                try:
                    self.stats_recorder.close()
                except Exception as log_e:
                    print(f"Error closing stats recorder on exit: {log_e}") # LOG
                    traceback.print_exc()

            pygame.quit()
            print("Application exited.") # LOG


if __name__ == "__main__":
    # Ensure base directories exist
    os.makedirs(BASE_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    os.makedirs(RUN_LOG_DIR, exist_ok=True)

    log_filepath = os.path.join(RUN_LOG_DIR, "console_output.log")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger = TeeLogger(log_filepath, original_stdout)
    sys.stdout = logger
    sys.stderr = logger

    app_instance = None

    try:
        if run_pre_checks():
            app_instance = MainApp()
            app_instance.run()
    except SystemExit as exit_err:
        print(
            f"Exiting due to SystemExit (Code: {getattr(exit_err, 'code', 'N/A')})."
        ) # LOG
    except Exception as main_err:
        print("\n--- UNHANDLED EXCEPTION DURING APP INITIALIZATION OR RUN ---") # LOG
        traceback.print_exc()
        print("--- EXITING DUE TO ERROR ---") # LOG
    finally:
        # Restore logging
        if "logger" in locals() and logger:
            final_app_state = getattr(app_instance, "app_state", "UNKNOWN")
            print(
                f"Restoring console output (Final App State: {final_app_state}). Full log saved to: {log_filepath}"
            ) # LOG
            logger.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(
            f"Console logging restored. Full log should be in: {log_filepath}"
        ) # Final message to actual console