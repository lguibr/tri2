File: app_setup.py
# File: app_setup.py
import os
import pygame
from typing import Tuple, Dict, Any

from config import (
    VisConfig,
    EnvConfig,
    RewardConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    DemoConfig,
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,
    get_config_dict,
    print_config_info_and_validate,
)


def initialize_pygame(
    vis_config: VisConfig,
) -> Tuple[pygame.Surface, pygame.time.Clock]:
    """Initializes Pygame, sets up the screen and clock."""
    print("Initializing Pygame...")
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (vis_config.SCREEN_WIDTH, vis_config.SCREEN_HEIGHT), pygame.RESIZABLE
    )
    pygame.display.set_caption("TriCrack PPO")
    clock = pygame.time.Clock()
    print("Pygame initialized.")
    return screen, clock


def initialize_directories():
    """Creates necessary runtime directories."""
    os.makedirs(RUN_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    print(f"Ensured directories exist: {RUN_CHECKPOINT_DIR}, {RUN_LOG_DIR}")


def load_and_validate_configs() -> Dict[str, Any]:
    """Loads all config classes and returns the combined config dictionary."""
    config_dict = get_config_dict()
    print_config_info_and_validate()
    return config_dict


File: check_gpu.py
import torch

print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is NOT available to PyTorch.")
    # You can add checks for drivers here if needed, but PyTorch check is primary
    try:
        import subprocess
        print("\nAttempting to run nvidia-smi...")
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout)
        else:
            print(f"nvidia-smi command failed or not found (return code {result.returncode}). Ensure NVIDIA drivers are installed.")
            print(f"stderr: {result.stderr}")
    except FileNotFoundError:
         print("nvidia-smi command not found. Ensure NVIDIA drivers are installed and in PATH.")
    except Exception as e:
         print(f"Error running nvidia-smi: {e}")

File: logger.py
# File: logger.py
import os
import sys
from typing import TextIO


class TeeLogger:
    """Redirects stdout/stderr to both the console and a log file."""

    def __init__(self, filepath: str, original_stream: TextIO):
        self.terminal = original_stream
        self.log_file: Optional[TextIO] = None
        try:
            log_dir = os.path.dirname(filepath)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            # Use buffering=1 for line buffering
            self.log_file = open(filepath, "w", encoding="utf-8", buffering=1)
            print(f"[TeeLogger] Logging console output to: {filepath}")
        except Exception as e:
            self.terminal.write(
                f"FATAL ERROR: Could not open log file {filepath}: {e}\n"
            )
            # Continue without file logging if opening fails

    def write(self, message: str):
        self.terminal.write(message)
        if self.log_file:
            try:
                self.log_file.write(message)
            except Exception:
                # Silently ignore errors writing to log file to avoid loops
                pass

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass  # Silently ignore errors flushing log file

    def close(self):
        if self.log_file:
            try:
                self.log_file.close()
                self.log_file = None
            except Exception as e:
                self.terminal.write(f"Warning: Error closing log file: {e}\n")

    def __del__(self):
        # Ensure file is closed if logger object is garbage collected
        self.close()


File: main_pygame.py
# File: main_pygame.py
import sys
import pygame
import numpy as np
import os
import time
import traceback
import torch
from typing import List, Tuple, Optional, Dict, Any, Deque

from logger import TeeLogger
from app_setup import (
    initialize_pygame,
    initialize_directories,
    load_and_validate_configs,
)

from config import (
    VisConfig,
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    RewardConfig,
    TensorBoardConfig,
    DemoConfig,
    DEVICE,
    RANDOM_SEED,
    MODEL_SAVE_PATH,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    RUN_LOG_DIR,
)

from environment.game_state import GameState, StateType
from agent.ppo_agent import PPOAgent
from training.trainer import Trainer
from stats.stats_recorder import StatsRecorderBase
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler
from utils.helpers import set_random_seeds
from utils.init_checks import run_pre_checks

from init.rl_components_ppo import (
    initialize_envs,
    initialize_agent,
    initialize_stats_recorder,
    initialize_trainer,
)


class MainApp:
    """Main application class orchestrating the Pygame UI and RL training."""

    def __init__(self):
        print("Initializing Application...")
        set_random_seeds(RANDOM_SEED)

        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        self.ppo_config = PPOConfig()
        self.rnn_config = RNNConfig()
        self.train_config = TrainConfig()
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        self.reward_config = RewardConfig()

        self.config_dict = load_and_validate_configs()
        self.num_envs = self.env_config.NUM_ENVS

        initialize_directories()
        self.screen, self.clock = initialize_pygame(self.vis_config)

        self.app_state = "Initializing"
        self.is_training_running = False
        self.cleanup_confirmation_active = False
        self.last_cleanup_message_time = 0.0
        self.cleanup_message = ""
        self.status = "Initializing Components"
        self.update_progress: float = 0.0  # For agent update progress bar

        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None
        self.envs: List[GameState] = []
        self.agent: Optional[PPOAgent] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.trainer: Optional[Trainer] = None
        self.demo_env: Optional[GameState] = None

        self._initialize_core_components(is_reinit=False)

        self.app_state = "MainMenu"
        self.status = "Ready"
        print("Initialization Complete. Ready.")
        print(f"--- tensorboard --logdir {os.path.abspath(BASE_LOG_DIR)} ---")

    def _initialize_core_components(self, is_reinit: bool = False):
        """Initializes Renderer, RL components, Demo Env, and Input Handler."""
        try:
            if not is_reinit:
                self.renderer = UIRenderer(self.screen, self.vis_config)
                # Initial render call (pass update_progress=0.0)
                self.renderer.render_all(
                    app_state=self.app_state,
                    is_training_running=self.is_training_running,
                    status=self.status,
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
                    update_progress=0.0,  # Pass initial progress
                )
                pygame.time.delay(100)

            self._initialize_rl_components(is_reinit=is_reinit)

            if not is_reinit:
                self._initialize_demo_env()
                self.input_handler = InputHandler(
                    screen=self.screen,
                    renderer=self.renderer,
                    toggle_training_run_cb=self._toggle_training_run,
                    request_cleanup_cb=self._request_cleanup,
                    cancel_cleanup_cb=self._cancel_cleanup,
                    confirm_cleanup_cb=self._confirm_cleanup,
                    exit_app_cb=self._exit_app,
                    start_demo_mode_cb=self._start_demo_mode,
                    exit_demo_mode_cb=self._exit_demo_mode,
                    handle_demo_input_cb=self._handle_demo_input,
                )
        except Exception as init_err:
            print(f"FATAL ERROR during component initialization: {init_err}")
            traceback.print_exc()
            if self.renderer:
                try:
                    self.app_state = "Error"
                    self.status = "Initialization Failed"
                    self.renderer._render_error_screen(self.status)
                    pygame.display.flip()
                    time.sleep(5)
                except Exception:
                    pass
            pygame.quit()
            sys.exit(1)

    def _initialize_rl_components(self, is_reinit: bool = False):
        """Initializes RL components using helper functions (now for PPO)."""
        print(f"Initializing RL components (PPO)... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            self.envs = initialize_envs(self.num_envs, self.env_config)
            self.agent = initialize_agent(
                self.model_config, self.ppo_config, self.rnn_config, self.env_config
            )

            self.stats_recorder = initialize_stats_recorder(
                stats_config=self.stats_config,
                tb_config=self.tensorboard_config,
                config_dict=self.config_dict,
                agent=self.agent,
                env_config=self.env_config,
                rnn_config=self.rnn_config,
                is_reinit=is_reinit,
            )
            if self.stats_recorder is None:
                raise RuntimeError("Stats Recorder initialization failed unexpectedly.")

            self.trainer = initialize_trainer(
                envs=self.envs,
                agent=self.agent,
                stats_recorder=self.stats_recorder,
                env_config=self.env_config,
                ppo_config=self.ppo_config,
                rnn_config=self.rnn_config,
                train_config=self.train_config,
                model_config=self.model_config,
            )
            print(f"RL components initialized in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Error during RL component initialization: {e}")
            raise e

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
            self.demo_env = None
            print("Warning: Demo mode may be unavailable.")

    # --- Input Handler Callbacks ---
    def _toggle_training_run(self):
        """Starts or stops the PPO training run."""
        if self.app_state != "MainMenu":
            return
        self.is_training_running = not self.is_training_running
        print(f"PPO Run {'STARTED' if self.is_training_running else 'STOPPED'}")
        if not self.is_training_running:
            self._try_save_checkpoint()

    def _request_cleanup(self):
        if self.app_state != "MainMenu":
            return
        was_running = self.is_training_running
        self.is_training_running = False
        if was_running:
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
            self.cleanup_confirmation_active = False
            print(
                f"Cleanup process finished. State: {self.app_state}, Status: {self.status}"
            )

    def _exit_app(self) -> bool:
        print("Exit requested.")
        return False

    def _start_demo_mode(self):
        if self.demo_env is None:
            print("Cannot start demo mode: Demo environment failed to initialize.")
            return
        if self.app_state == "MainMenu":
            print("Entering Demo Mode...")
            self.is_training_running = False
            self._try_save_checkpoint()
            self.app_state = "Playing"
            self.status = "Playing Demo"
            self.demo_env.reset()

    def _exit_demo_mode(self):
        if self.app_state == "Playing":
            print("Exiting Demo Mode...")
            self.app_state = "MainMenu"
            self.status = "Ready"

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
                    reward, done = self.demo_env.step(action_index)
                    action_taken = True
                else:
                    action_taken = True

            if self.demo_env.is_over():
                print("[Demo] Game Over! Press ESC to exit.")

    # --- Core Logic Methods ---
    def _cleanup_data(self):
        """Deletes current run's checkpoint and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.app_state = "Initializing"
        self.is_training_running = False
        self.status = "Cleaning"
        messages = []

        if self.renderer:
            try:
                # Pass update_progress=0.0 during cleanup render
                self.renderer.render_all(
                    app_state=self.app_state,
                    is_training_running=False,
                    status=self.status,
                    stats_summary={},
                    envs=[],
                    num_envs=0,
                    env_config=self.env_config,
                    cleanup_confirmation_active=False,
                    cleanup_message="",
                    last_cleanup_message_time=0,
                    tensorboard_log_dir=None,
                    plot_data={},
                    demo_env=self.demo_env,
                    update_progress=0.0,  # Pass progress
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

        if self.trainer:
            print("[Cleanup] Running trainer cleanup...")
            try:
                self.trainer.cleanup(save_final=False)
            except Exception as e:
                print(f"Error during trainer cleanup: {e}")
        if self.stats_recorder:
            print("[Cleanup] Closing stats recorder...")
            try:
                # Close method might be on the specific recorder instance
                if hasattr(self.stats_recorder, "close"):
                    self.stats_recorder.close()
            except Exception as e:
                print(f"Error closing stats recorder: {e}")

        print("[Cleanup] Deleting agent checkpoint file...")
        try:
            if os.path.isfile(MODEL_SAVE_PATH):
                os.remove(MODEL_SAVE_PATH)
                msg = f"Agent ckpt deleted: {os.path.basename(MODEL_SAVE_PATH)}"
            else:
                msg = f"Agent ckpt not found (current run)."
            print(f"  - {msg}")
            messages.append(msg)
        except OSError as e:
            msg = f"Error deleting agent ckpt: {e}"
            print(f"  - {msg}")
            messages.append(msg)

        time.sleep(0.1)

        print("[Cleanup] Re-initializing RL components...")
        try:
            self._initialize_rl_components(is_reinit=True)
            if self.demo_env:
                self.demo_env.reset()
            print("[Cleanup] RL components re-initialized successfully.")
            messages.append("RL components re-initialized.")
            self.status = "Ready"
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
        """Saves checkpoint if run is stopped and trainer exists."""
        if (
            self.app_state == "MainMenu"
            and not self.is_training_running
            and self.trainer
        ):
            print("Saving checkpoint on stop...")
            try:
                self.trainer.maybe_save_checkpoint(force_save=True)
            except Exception as e:
                print(f"Error saving checkpoint on stop: {e}")

    def _update(self):
        """Updates the application state and performs training steps (PPO)."""
        should_perform_training_iteration = False
        self.update_progress = 0.0  # Reset progress each frame

        if self.app_state == "MainMenu":
            if self.cleanup_confirmation_active:
                self.status = "Confirm Cleanup"
            elif not self.is_training_running and self.status != "Error":
                self.status = "Ready"
            elif not self.trainer:
                if self.status != "Error":
                    self.status = "Error: Trainer Missing"
            elif self.is_training_running:
                # Get detailed status from trainer
                trainer_phase = self.trainer.get_current_phase()
                if trainer_phase == "Collecting":
                    self.status = "Collecting Experience"
                elif trainer_phase == "Updating":
                    self.status = "Updating Agent"
                    self.update_progress = self.trainer.get_update_progress()
                else:
                    self.status = "Training (Unknown Phase)"  # Fallback

                should_perform_training_iteration = True
            else:  # Not running
                if self.status != "Error":
                    self.status = "Ready"

            if should_perform_training_iteration:
                if not self.trainer:
                    print("Error: Trainer became unavailable during _update.")
                    self.status = "Error: Trainer Lost"
                    self.is_training_running = False
                else:
                    try:
                        self.trainer.perform_training_iteration()
                    except Exception as e:
                        print(
                            f"\n--- ERROR DURING TRAINING ITERATION (Step: {getattr(self.trainer, 'global_step', 'N/A')}) ---"
                        )
                        traceback.print_exc()
                        print("--- Stopping training due to error. ---")
                        self.is_training_running = False
                        self.status = "Error: Training Iteration Failed"
                        self.app_state = "Error"

        elif self.app_state == "Playing":
            if self.demo_env and hasattr(self.demo_env, "_update_timers"):
                self.demo_env._update_timers()
            self.status = "Playing Demo"
        elif self.app_state == "Initializing":
            self.status = "Initializing..."
        elif self.app_state == "Error":
            pass  # Status already set

    def _render(self):
        """Renders the UI based on the current application state."""
        stats_summary = {}
        plot_data: Dict[str, Deque] = {}

        if self.app_state != "Initializing":
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

        if not self.renderer:
            print("Error: Renderer not initialized in _render.")
            try:
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
            # Pass update_progress to renderer
            self.renderer.render_all(
                app_state=self.app_state,
                is_training_running=self.is_training_running,
                status=self.status,
                stats_summary=stats_summary,
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
                update_progress=self.update_progress,  # Pass progress
            )
        except Exception as render_all_err:
            print(f"CRITICAL ERROR in renderer.render_all: {render_all_err}")
            traceback.print_exc()
            try:
                self.app_state = "Error"
                self.status = "Render Error"
                self.renderer._render_error_screen(self.status)
                pygame.display.flip()
            except Exception as e:
                print(f"Error rendering error screen: {e}")

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
        elif self.stats_recorder:
            print("Closing stats recorder...")
            try:
                if hasattr(self.stats_recorder, "close"):
                    self.stats_recorder.close()
            except Exception as log_e:
                print(f"Error closing stats recorder on exit: {log_e}")

        pygame.quit()
        print("Application exited.")

    def run(self):
        """Main application loop."""
        print("Starting main application loop...")
        running_flag = True
        try:
            while running_flag:
                start_frame_time = time.perf_counter()

                if self.input_handler:
                    try:
                        running_flag = self.input_handler.handle_input(
                            self.app_state, self.cleanup_confirmation_active
                        )
                    except Exception as input_err:
                        print(
                            f"\n--- UNHANDLED ERROR IN INPUT LOOP ({self.app_state}) ---"
                        )
                        traceback.print_exc()
                        running_flag = False
                else:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running_flag = False
                        if (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_ESCAPE
                        ):
                            if self.app_state == "Playing":
                                self._exit_demo_mode()
                            elif not self.cleanup_confirmation_active:
                                running_flag = False
                    if not running_flag:
                        break

                if not running_flag:
                    break

                try:
                    self._update()
                except Exception as update_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN UPDATE LOOP ({self.app_state}) ---"
                    )
                    traceback.print_exc()
                    self.status = "Error: Update Loop Failed"
                    self.app_state = "Error"
                    self.is_training_running = False

                try:
                    self._render()
                except Exception as render_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN RENDER LOOP ({self.app_state}) ---"
                    )
                    traceback.print_exc()
                    self.status = "Error: Render Loop Failed"
                    self.app_state = "Error"

                # Adjust sleep based on whether agent update is happening
                is_updating = self.status == "Updating Agent"
                if not self.is_training_running or not is_updating:
                    # Sleep longer if not training or just collecting
                    time.sleep(0.01)
                # No explicit sleep during agent update to maximize processing speed

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
    exit_code = 0

    try:
        if run_pre_checks():
            app_instance = MainApp()
            app_instance.run()
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
        if app_instance and hasattr(app_instance, "_perform_cleanup"):
            print("Attempting cleanup after main exception...")
            try:
                app_instance._perform_cleanup()
            except Exception as cleanup_err:
                print(f"Error during cleanup after main exception: {cleanup_err}")
    finally:
        if logger:
            final_app_state = getattr(app_instance, "app_state", "UNKNOWN")
            print(
                f"Restoring console output (Final App State: {final_app_state}). Full log saved to: {log_filepath}"
            )
            logger.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Console logging restored. Full log should be in: {log_filepath}")
        sys.exit(exit_code)


File: requirements.txt
pygame>=2.1.0
numpy>=1.20.0
torch>=1.10.0
tensorboard
cloudpickle
matplotlib

File: agent\model_factory.py
# File: agent/model_factory.py
import torch.nn as nn
from config import ModelConfig, EnvConfig, PPOConfig, RNNConfig
from typing import Type

from agent.networks.agent_network import ActorCriticNetwork


def create_network(
    env_config: EnvConfig,
    action_dim: int,
    model_config: ModelConfig,
    rnn_config: RNNConfig,
) -> nn.Module:
    """Creates the ActorCriticNetwork based on configuration."""
    print(f"[ModelFactory] Creating ActorCriticNetwork (RNN: {rnn_config.USE_RNN})")
    return ActorCriticNetwork(
        env_config=env_config,
        action_dim=action_dim,
        model_config=model_config.Network,
        rnn_config=rnn_config,
    )


File: agent\ppo_agent.py
# File: agent/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import traceback
from typing import Tuple, List, Dict, Any, Optional, Union

from config import (
    ModelConfig,
    EnvConfig,
    PPOConfig,
    RNNConfig,
    DEVICE,
    TensorBoardConfig,
    TOTAL_TRAINING_STEPS,
)
from environment.game_state import StateType
from utils.types import ActionType, AgentStateDict
from agent.model_factory import create_network
from agent.networks.agent_network import ActorCriticNetwork


class PPOAgent:
    """PPO Agent orchestrating network, action selection, and updates."""

    def __init__(
        self,
        model_config: ModelConfig,
        ppo_config: PPOConfig,
        rnn_config: RNNConfig,
        env_config: EnvConfig,
    ):
        self.device = DEVICE
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.tb_config = TensorBoardConfig()
        self.action_dim = env_config.ACTION_DIM
        self.update_progress: float = 0.0  # Track update progress

        self.network = create_network(
            env_config=self.env_config,
            action_dim=self.action_dim,
            model_config=model_config,
            rnn_config=self.rnn_config,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=ppo_config.LEARNING_RATE,
            eps=ppo_config.ADAM_EPS,
        )

        self._print_init_info()

    def _print_init_info(self):
        print(f"[PPOAgent] Using Device: {self.device}")
        print(f"[PPOAgent] Network: {type(self.network).__name__}")
        print(f"[PPOAgent] Using RNN: {self.rnn_config.USE_RNN}")
        total_params = sum(
            p.numel() for p in self.network.parameters() if p.requires_grad
        )
        print(f"[PPOAgent] Trainable Parameters: {total_params / 1e6:.2f} M")

    @torch.no_grad()
    def select_action(
        self,
        state: StateType,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
        valid_actions_indices: Optional[List[ActionType]] = None,
    ) -> Tuple[ActionType, float, float, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        self.network.eval()

        grid_np = state["grid"]
        shapes_np = state["shapes"]
        availability_np = state["shape_availability"]
        explicit_features_np = state["explicit_features"]

        grid_t = torch.from_numpy(grid_np).float().unsqueeze(0).to(self.device)
        shapes_t = torch.from_numpy(shapes_np).float().unsqueeze(0).to(self.device)
        availability_t = (
            torch.from_numpy(availability_np).float().unsqueeze(0).to(self.device)
        )
        explicit_features_t = (
            torch.from_numpy(explicit_features_np).float().unsqueeze(0).to(self.device)
        )

        if self.rnn_config.USE_RNN:
            grid_t = grid_t.unsqueeze(1)
            shapes_t = shapes_t.unsqueeze(1)
            availability_t = availability_t.unsqueeze(1)
            explicit_features_t = explicit_features_t.unsqueeze(1)
            if hidden_state:
                hidden_state = (
                    hidden_state[0][:, 0:1, :].contiguous(),
                    hidden_state[1][:, 0:1, :].contiguous(),
                )

        policy_logits, value, next_hidden_state = self.network(
            grid_t, shapes_t, availability_t, explicit_features_t, hidden_state
        )

        if self.rnn_config.USE_RNN:
            policy_logits = policy_logits.squeeze(1)
            value = value.squeeze(1)

        policy_logits = torch.nan_to_num(policy_logits.squeeze(0), nan=-1e9)

        if valid_actions_indices is not None:
            mask = torch.full_like(policy_logits, -float("inf"))
            valid_indices_in_bounds = [
                idx
                for idx in valid_actions_indices
                if 0 <= idx < policy_logits.shape[0]
            ]
            if valid_indices_in_bounds:
                mask[valid_indices_in_bounds] = 0
                policy_logits += mask

        if torch.all(policy_logits == -float("inf")):
            action = 0
            action_log_prob = torch.tensor(-1e9, device=self.device)
        else:
            distribution = Categorical(logits=policy_logits)
            action_tensor = (
                distribution.mode if deterministic else distribution.sample()
            )
            action_log_prob = distribution.log_prob(action_tensor)
            action = action_tensor.item()

            if (
                valid_actions_indices is not None
                and action not in valid_actions_indices
            ):
                if valid_indices_in_bounds:
                    action = np.random.choice(valid_indices_in_bounds)
                    action_log_prob = distribution.log_prob(
                        torch.tensor(action, device=self.device)
                    )
                else:
                    action = 0
                    action_log_prob = torch.tensor(-1e9, device=self.device)

        return action, action_log_prob.item(), value.squeeze().item(), next_hidden_state

    @torch.no_grad()
    def select_action_batch(
        self,
        grid_batch: torch.Tensor,
        shape_batch: torch.Tensor,
        availability_batch: torch.Tensor,
        explicit_features_batch: torch.Tensor,
        hidden_state_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
        valid_actions_lists: List[Optional[List[ActionType]]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        self.network.eval()
        batch_size = grid_batch.shape[0]

        grid_batch = grid_batch.to(self.device)
        shape_batch = shape_batch.to(self.device)
        availability_batch = availability_batch.to(self.device)
        explicit_features_batch = explicit_features_batch.to(self.device)
        if hidden_state_batch:
            hidden_state_batch = (
                hidden_state_batch[0].to(self.device),
                hidden_state_batch[1].to(self.device),
            )

        if self.rnn_config.USE_RNN:
            grid_batch = grid_batch.unsqueeze(1)
            shape_batch = shape_batch.unsqueeze(1)
            availability_batch = availability_batch.unsqueeze(1)
            explicit_features_batch = explicit_features_batch.unsqueeze(1)

        policy_logits, value, next_hidden_batch = self.network(
            grid_batch,
            shape_batch,
            availability_batch,
            explicit_features_batch,
            hidden_state_batch,
        )

        if self.rnn_config.USE_RNN:
            policy_logits = policy_logits.squeeze(1)
            value = value.squeeze(1)

        policy_logits = torch.nan_to_num(policy_logits, nan=-1e9)

        mask = torch.full_like(policy_logits, -float("inf"))
        any_valid = False
        for i in range(batch_size):
            valid_actions = valid_actions_lists[i]
            if valid_actions:
                valid_indices_in_bounds = [
                    idx for idx in valid_actions if 0 <= idx < self.action_dim
                ]
                if valid_indices_in_bounds:
                    mask[i, valid_indices_in_bounds] = 0
                    any_valid = True

        if any_valid:
            policy_logits += mask

        all_masked_rows = torch.all(policy_logits == -float("inf"), dim=1)
        policy_logits[all_masked_rows] = 0.0

        distribution = Categorical(logits=policy_logits)
        actions_tensor = distribution.sample()
        action_log_probs = distribution.log_prob(actions_tensor)

        actions_tensor[all_masked_rows] = 0
        action_log_probs[all_masked_rows] = -1e9

        value = value.squeeze(-1) if value.ndim > 1 else value

        return actions_tensor, action_log_probs, value, next_hidden_batch

    def evaluate_actions(
        self,
        grid_tensor: torch.Tensor,
        shape_feature_tensor: torch.Tensor,
        shape_availability_tensor: torch.Tensor,
        explicit_features_tensor: torch.Tensor,
        actions: torch.Tensor,
        initial_lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dones_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.network.train()
        is_sequence = self.rnn_config.USE_RNN and grid_tensor.ndim == 5

        if is_sequence:
            batch_size = grid_tensor.shape[0]
            seq_len = grid_tensor.shape[1]

            if initial_lstm_state is None:
                current_hidden_state = self.network.get_initial_hidden_state(batch_size)
            else:
                current_hidden_state = (
                    initial_lstm_state[0].to(self.device),
                    initial_lstm_state[1].to(self.device),
                )

            policy_logits, value, _ = self.network(
                grid_tensor,
                shape_feature_tensor,
                shape_availability_tensor,
                explicit_features_tensor,
                current_hidden_state,
            )

            policy_logits = policy_logits.view(batch_size * seq_len, -1)
            value = value.view(batch_size * seq_len, -1)

        else:
            policy_logits, value, _ = self.network(
                grid_tensor,
                shape_feature_tensor,
                shape_availability_tensor,
                explicit_features_tensor,
                hidden_state=None,
            )

        policy_logits = torch.nan_to_num(policy_logits, nan=-1e9)
        distribution = Categorical(logits=policy_logits)

        action_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        value = value.squeeze(-1)

        return action_log_probs, value, entropy

    def update(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        self.network.train()
        self.update_progress = 0.0

        obs_grid_flat = rollout_data["obs_grid"]
        obs_shapes_flat = rollout_data["obs_shapes"]
        obs_availability_flat = rollout_data["obs_availability"]
        obs_explicit_features_flat = rollout_data["obs_explicit_features"]
        actions_flat = rollout_data["actions"]
        old_log_probs_flat = rollout_data["log_probs"]
        returns_flat = rollout_data["returns"]
        advantages_flat = rollout_data["advantages"]

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        num_samples = actions_flat.shape[0]
        batch_size = self.ppo_config.MINIBATCH_SIZE
        indices = np.arange(num_samples)
        total_minibatches = (num_samples + batch_size - 1) // batch_size
        total_update_steps = self.ppo_config.PPO_EPOCHS * total_minibatches

        for epoch in range(self.ppo_config.PPO_EPOCHS):
            np.random.shuffle(indices)
            for i, start_idx in enumerate(range(0, num_samples, batch_size)):
                end_idx = start_idx + batch_size
                minibatch_indices = indices[start_idx:end_idx]
                minibatch_size_actual = len(minibatch_indices)

                mb_obs_grid = obs_grid_flat[minibatch_indices]
                mb_obs_shapes = obs_shapes_flat[minibatch_indices]
                mb_obs_availability = obs_availability_flat[minibatch_indices]
                mb_obs_explicit_features = obs_explicit_features_flat[minibatch_indices]
                mb_actions = actions_flat[minibatch_indices]
                mb_old_log_probs = old_log_probs_flat[minibatch_indices]
                mb_returns = returns_flat[minibatch_indices]
                mb_advantages = advantages_flat[minibatch_indices]

                if self.rnn_config.USE_RNN:
                    eval_grid = mb_obs_grid.unsqueeze(1)
                    eval_shapes = mb_obs_shapes.unsqueeze(1)
                    eval_availability = mb_obs_availability.unsqueeze(1)
                    eval_explicit_features = mb_obs_explicit_features.unsqueeze(1)
                    eval_actions_seq = mb_actions

                    mb_initial_hidden = self.network.get_initial_hidden_state(
                        minibatch_size_actual
                    )

                    new_log_probs, predicted_values, entropy = self.evaluate_actions(
                        eval_grid,
                        eval_shapes,
                        eval_availability,
                        eval_explicit_features,
                        eval_actions_seq,
                        mb_initial_hidden,
                        dones_tensor=None,
                    )

                else:
                    new_log_probs, predicted_values, entropy = self.evaluate_actions(
                        mb_obs_grid,
                        mb_obs_shapes,
                        mb_obs_availability,
                        mb_obs_explicit_features,
                        mb_actions,
                        initial_lstm_state=None,
                        dones_tensor=None,
                    )

                logratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(logratio)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.ppo_config.CLIP_PARAM,
                        1.0 + self.ppo_config.CLIP_PARAM,
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(predicted_values, mb_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.ppo_config.VALUE_LOSS_COEF * value_loss
                    + self.ppo_config.ENTROPY_COEF * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.ppo_config.MAX_GRAD_NORM > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.ppo_config.MAX_GRAD_NORM
                    )
                else:
                    grad_norm = None

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

                current_update_step = epoch * total_minibatches + (i + 1)
                self.update_progress = current_update_step / total_update_steps

        avg_policy_loss = total_policy_loss / max(1, num_updates)
        avg_value_loss = total_value_loss / max(1, num_updates)
        avg_entropy = total_entropy / max(1, num_updates)

        # --- REMOVED FINAL SUMMARY PRINT ---
        # print(
        #     f"[PPOAgent Update Summary] Avg P Loss: {avg_policy_loss:.4f}, Avg V Loss: {avg_value_loss:.4f}, Avg Entropy: {avg_entropy:.4f}"
        # )
        # --- END REMOVED ---
        self.update_progress = 1.0

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
        }

    def get_state_dict(self) -> AgentStateDict:
        original_device = next(self.network.parameters()).device
        self.network.cpu()
        state = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        self.network.to(original_device)
        return state

    def load_state_dict(self, state_dict: AgentStateDict):
        print(f"[PPOAgent] Loading state dict. Target device: {self.device}")
        try:
            self.network.load_state_dict(state_dict["network_state_dict"])
            self.network.to(self.device)
            print("[PPOAgent] Network state loaded.")

            if "optimizer_state_dict" in state_dict:
                try:
                    self.optimizer = optim.AdamW(
                        self.network.parameters(),
                        lr=self.ppo_config.LEARNING_RATE,
                        eps=self.ppo_config.ADAM_EPS,
                    )
                    self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    print("[PPOAgent] Optimizer state loaded and moved to device.")
                except Exception as e:
                    print(
                        f"Warning: Could not load optimizer state ({e}). Re-initializing optimizer."
                    )
                    self.optimizer = optim.AdamW(
                        self.network.parameters(),
                        lr=self.ppo_config.LEARNING_RATE,
                        eps=self.ppo_config.ADAM_EPS,
                    )
            else:
                print(
                    "[PPOAgent] Optimizer state not found. Re-initializing optimizer."
                )
                self.optimizer = optim.AdamW(
                    self.network.parameters(),
                    lr=self.ppo_config.LEARNING_RATE,
                    eps=self.ppo_config.ADAM_EPS,
                )

            print("[PPOAgent] load_state_dict complete.")

        except Exception as e:
            print(f"CRITICAL ERROR during PPOAgent.load_state_dict: {e}")
            traceback.print_exc()

    def get_initial_hidden_state(
        self, num_envs: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.rnn_config.USE_RNN:
            return None
        return self.network.get_initial_hidden_state(num_envs)

    def get_update_progress(self) -> float:
        """Returns the progress of the current agent update phase (0.0 to 1.0)."""
        return self.update_progress


File: agent\__init__.py
# File: agent/__init__.py
from .ppo_agent import PPOAgent
from .model_factory import create_network
from agent.networks.agent_network import ActorCriticNetwork

__all__ = [
    "PPOAgent",
    "create_network",
    "ActorCriticNetwork",
]


File: agent\networks\agent_network.py
# File: agent/networks/agent_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from config import ModelConfig, EnvConfig, PPOConfig, RNNConfig, DEVICE
from typing import Tuple, List, Type, Optional


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network: CNN+MLP -> Fusion -> Optional LSTM -> Actor/Critic Heads.
    Handles both single step (eval) and sequence (RNN training) inputs.
    Now includes shape availability and explicit features.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        action_dim: int,
        model_config: ModelConfig.Network,
        rnn_config: RNNConfig,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.env_config = env_config
        self.config = model_config
        self.rnn_config = rnn_config
        self.device = DEVICE

        print(f"[ActorCriticNetwork] Target device set to: {self.device}")
        print(f"[ActorCriticNetwork] Using RNN: {self.rnn_config.USE_RNN}")

        self.grid_c, self.grid_h, self.grid_w = self.env_config.GRID_STATE_SHAPE
        self.shape_feat_dim = self.env_config.SHAPE_STATE_DIM
        self.shape_availability_dim = self.env_config.SHAPE_AVAILABILITY_DIM
        # --- UPDATED: Read the potentially larger explicit features dimension ---
        self.explicit_features_dim = self.env_config.EXPLICIT_FEATURES_DIM
        # --- END UPDATED ---
        self.num_shape_slots = self.env_config.NUM_SHAPE_SLOTS
        self.shape_feat_per_slot = self.env_config.SHAPE_FEATURES_PER_SHAPE

        print(f"[ActorCriticNetwork] Initializing:")
        print(f"  Input Grid Shape: [B, {self.grid_c}, {self.grid_h}, {self.grid_w}]")
        print(f"  Input Shape Features Dim: {self.shape_feat_dim}")
        print(f"  Input Shape Availability Dim: {self.shape_availability_dim}")
        # --- UPDATED: Log the correct explicit features dim ---
        print(f"  Input Explicit Features Dim: {self.explicit_features_dim}")
        # --- END UPDATED ---

        self.conv_base, conv_out_h, conv_out_w, conv_out_c = self._build_cnn_branch()
        self.conv_out_size = self._get_conv_out_size(
            (self.grid_c, self.grid_h, self.grid_w)
        )
        print(
            f"  CNN Output Dim (HxWxC): ({conv_out_h}x{conv_out_w}x{conv_out_c}) -> Flat: {self.conv_out_size}"
        )

        self.shape_mlp, self.shape_mlp_out_dim = self._build_shape_mlp_branch()
        print(f"  Shape Feature MLP Output Dim: {self.shape_mlp_out_dim}")

        # --- MODIFIED: Calculate combined dim including availability and the *new* explicit features dim ---
        combined_features_dim = (
            self.conv_out_size
            + self.shape_mlp_out_dim
            + self.shape_availability_dim
            + self.explicit_features_dim  # Use the updated dimension
        )
        # --- END MODIFIED ---
        print(
            f"  Combined Features Dim (CNN + Shape MLP + Availability + Explicit): {combined_features_dim}"
        )

        self.fusion_mlp, self.fusion_output_dim = self._build_fusion_mlp_branch(
            combined_features_dim  # Pass the updated combined dimension
        )
        print(f"  Fusion MLP Output Dim: {self.fusion_output_dim}")

        self.lstm_layer = None
        self.lstm_hidden_size = 0
        if self.rnn_config.USE_RNN:
            self.lstm_hidden_size = self.rnn_config.LSTM_HIDDEN_SIZE
            self.lstm_layer = nn.LSTM(
                input_size=self.fusion_output_dim,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.rnn_config.LSTM_NUM_LAYERS,
                batch_first=True,
            ).to(self.device)
            print(f"  LSTM Layer Added (Hidden Size: {self.lstm_hidden_size})")
            head_input_dim = self.lstm_hidden_size
        else:
            head_input_dim = self.fusion_output_dim

        self.actor_head = nn.Linear(head_input_dim, self.action_dim).to(self.device)
        self.critic_head = nn.Linear(head_input_dim, 1).to(self.device)
        print(f"  Actor Head Output Dim: {self.action_dim}")
        print(f"  Critic Head Output Dim: 1")

        self._init_head_weights()

    def _init_head_weights(self):
        print("  Initializing Actor/Critic heads using Xavier Uniform.")
        actor_gain = nn.init.calculate_gain("linear")
        critic_gain = nn.init.calculate_gain("linear")
        nn.init.xavier_uniform_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0)
        nn.init.xavier_uniform_(self.critic_head.weight, gain=critic_gain)
        nn.init.constant_(self.critic_head.bias, 0)

    def _build_cnn_branch(self) -> Tuple[nn.Sequential, int, int, int]:
        conv_layers: List[nn.Module] = []
        current_channels = self.grid_c
        h, w = self.grid_h, self.grid_w
        cfg = self.config
        for i, out_channels in enumerate(cfg.CONV_CHANNELS):
            conv_layer = nn.Conv2d(
                current_channels,
                out_channels,
                kernel_size=cfg.CONV_KERNEL_SIZE,
                stride=cfg.CONV_STRIDE,
                padding=cfg.CONV_PADDING,
                bias=not cfg.USE_BATCHNORM_CONV,
            ).to(self.device)
            conv_layers.append(conv_layer)
            if cfg.USE_BATCHNORM_CONV:
                conv_layers.append(nn.BatchNorm2d(out_channels).to(self.device))
            conv_layers.append(cfg.CONV_ACTIVATION())
            current_channels = out_channels
            h = (h + 2 * cfg.CONV_PADDING - cfg.CONV_KERNEL_SIZE) // cfg.CONV_STRIDE + 1
            w = (w + 2 * cfg.CONV_PADDING - cfg.CONV_KERNEL_SIZE) // cfg.CONV_STRIDE + 1
        return nn.Sequential(*conv_layers), h, w, current_channels

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape, device=self.device)
            self.conv_base.eval()
            output = self.conv_base(dummy_input)
            self.conv_base.train()
            return int(np.prod(output.size()[1:]))

    def _build_shape_mlp_branch(self) -> Tuple[nn.Sequential, int]:
        shape_mlp_layers: List[nn.Module] = []
        current_dim = self.env_config.SHAPE_STATE_DIM
        cfg = self.config
        for hidden_dim in cfg.SHAPE_FEATURE_MLP_DIMS:
            lin_layer = nn.Linear(current_dim, hidden_dim).to(self.device)
            shape_mlp_layers.append(lin_layer)
            shape_mlp_layers.append(cfg.SHAPE_MLP_ACTIVATION())
            current_dim = hidden_dim
        if not cfg.SHAPE_FEATURE_MLP_DIMS:
            return nn.Identity(), current_dim
        return nn.Sequential(*shape_mlp_layers), current_dim

    def _build_fusion_mlp_branch(self, input_dim: int) -> Tuple[nn.Sequential, int]:
        fusion_layers: List[nn.Module] = []
        current_fusion_dim = input_dim
        cfg = self.config
        for i, hidden_dim in enumerate(cfg.COMBINED_FC_DIMS):
            linear_layer = nn.Linear(
                current_fusion_dim, hidden_dim, bias=not cfg.USE_BATCHNORM_FC
            ).to(self.device)
            fusion_layers.append(linear_layer)
            if cfg.USE_BATCHNORM_FC:
                fusion_layers.append(nn.BatchNorm1d(hidden_dim).to(self.device))
            fusion_layers.append(cfg.COMBINED_ACTIVATION())
            if cfg.DROPOUT_FC > 0:
                fusion_layers.append(nn.Dropout(cfg.DROPOUT_FC).to(self.device))
            current_fusion_dim = hidden_dim
        return nn.Sequential(*fusion_layers), current_fusion_dim

    def forward(
        self,
        grid_tensor: torch.Tensor,
        shape_feature_tensor: torch.Tensor,
        shape_availability_tensor: torch.Tensor,
        # --- Input signature remains the same ---
        explicit_features_tensor: torch.Tensor,
        # --- End Input ---
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        model_device = next(self.parameters()).device
        grid_tensor = grid_tensor.to(model_device)
        shape_feature_tensor = shape_feature_tensor.to(model_device)
        shape_availability_tensor = shape_availability_tensor.to(model_device)
        # --- Move explicit features tensor to device ---
        explicit_features_tensor = explicit_features_tensor.to(model_device)
        # --- End Move ---
        if hidden_state:
            hidden_state = (
                hidden_state[0].to(model_device),
                hidden_state[1].to(model_device),
            )

        is_sequence = self.rnn_config.USE_RNN and grid_tensor.ndim == 5
        initial_batch_size = grid_tensor.shape[0]
        seq_len = grid_tensor.shape[1] if is_sequence else 1
        num_samples = initial_batch_size * seq_len

        # Reshape inputs to [B*T or B, ...]
        grid_input_flat = grid_tensor.reshape(
            num_samples, *self.env_config.GRID_STATE_SHAPE
        )
        shape_feature_input_flat = shape_feature_tensor.reshape(
            num_samples, self.env_config.SHAPE_STATE_DIM
        )
        shape_availability_input_flat = shape_availability_tensor.reshape(
            num_samples, self.env_config.SHAPE_AVAILABILITY_DIM
        )
        # --- Reshape explicit features tensor (using the updated dimension from config) ---
        explicit_features_input_flat = explicit_features_tensor.reshape(
            num_samples, self.env_config.EXPLICIT_FEATURES_DIM
        )
        # --- End Reshape ---

        conv_output = self.conv_base(grid_input_flat)
        conv_output_flat = conv_output.view(num_samples, -1)
        shape_mlp_output = self.shape_mlp(shape_feature_input_flat)

        # --- Combine all feature streams (logic remains the same, dimensions handled by reshape) ---
        combined_features = torch.cat(
            (
                conv_output_flat,
                shape_mlp_output,
                shape_availability_input_flat,
                explicit_features_input_flat,  # Add explicit features
            ),
            dim=1,
        )
        # --- End Combine ---

        fused_output = self.fusion_mlp(combined_features)

        next_hidden_state = hidden_state
        if self.rnn_config.USE_RNN and self.lstm_layer is not None:
            lstm_input = fused_output.view(
                initial_batch_size, seq_len, self.fusion_output_dim
            )
            lstm_output, next_hidden_state = self.lstm_layer(lstm_input, hidden_state)
            head_input = lstm_output.contiguous().view(num_samples, -1)
        else:
            head_input = fused_output

        policy_logits = self.actor_head(head_input)
        value = self.critic_head(head_input)

        # Reshape back if input was sequence
        if is_sequence:
            policy_logits = policy_logits.view(initial_batch_size, seq_len, -1)
            value = value.view(initial_batch_size, seq_len, -1)

        return policy_logits, value, next_hidden_state

    def get_initial_hidden_state(
        self, batch_size: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.rnn_config.USE_RNN or self.lstm_layer is None:
            return None
        model_device = next(self.parameters()).device
        num_layers = self.rnn_config.LSTM_NUM_LAYERS
        hidden_size = self.rnn_config.LSTM_HIDDEN_SIZE
        h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=model_device)
        c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=model_device)
        return (h_0, c_0)


File: agent\networks\__init__.py


File: config\core.py
# File: config/core.py
import torch
from typing import Deque, Dict, Any, List, Type, Tuple, Optional

from .general import TOTAL_TRAINING_STEPS


class VisConfig:
    NUM_ENVS_TO_RENDER = 64
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00
    LEFT_PANEL_WIDTH = int(SCREEN_WIDTH * 0.7)
    ENV_SPACING = 0
    ENV_GRID_PADDING = 0
    FPS = 0

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
    EXPLICIT_FEATURES_DIM = 10
    CALCULATE_POTENTIAL_OUTCOMES_IN_STATE = False

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
    REWARD_PLACE_PER_TRI = 0.01
    REWARD_CLEAR_1 = 1.5
    REWARD_CLEAR_2 = 4.0
    REWARD_CLEAR_3PLUS = 8.0
    REWARD_ALIVE_STEP = 0.001
    PENALTY_INVALID_MOVE = -0.1
    PENALTY_GAME_OVER = -1.5
    PENALTY_MAX_HEIGHT_FACTOR = -0.005
    PENALTY_BUMPINESS_FACTOR = -0.01
    PENALTY_HOLE_PER_HOLE = -0.07
    PENALTY_NEW_HOLE = -0.15


class PPOConfig:
    LEARNING_RATE = 2e-4
    ADAM_EPS = 1e-5
    NUM_STEPS_PER_ROLLOUT = 1024
    PPO_EPOCHS = 12
    NUM_MINIBATCHES = 64
    CLIP_PARAM = 0.2
    GAMMA = 0.995
    GAE_LAMBDA = 0.95
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_END_FRACTION = 0.0

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
        batch_size = total_data_per_update // num_minibatches
        return max(1, batch_size)


class RNNConfig:
    USE_RNN = True
    LSTM_HIDDEN_SIZE = 1024
    LSTM_NUM_LAYERS = 1


class TrainConfig:
    # --- MODIFIED: Reduced checkpoint frequency ---
    CHECKPOINT_SAVE_FREQ = 5  # Save every 5 rollouts
    # --- END MODIFIED ---
    LOAD_CHECKPOINT_PATH: Optional[str] = None


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        CONV_CHANNELS = [96, 192, 192]
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True

        SHAPE_FEATURE_MLP_DIMS = [192]
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU

        COMBINED_FC_DIMS = [1792, 896]
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.0


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [50, 100, 500, 1_000, 5_000, 10_000]
    # --- MODIFIED: Reduced console log frequency ---
    CONSOLE_LOG_FREQ = 2  # Log every 2 rollouts
    # --- END MODIFIED ---
    PLOT_DATA_WINDOW = 100_000


class TensorBoardConfig:
    LOG_HISTOGRAMS = True
    # --- MODIFIED: Reduced histogram log frequency ---
    HISTOGRAM_LOG_FREQ = 5  # Log histograms every 5 rollouts
    # --- END MODIFIED ---
    LOG_IMAGES = True
    # --- MODIFIED: Reduced image log frequency ---
    IMAGE_LOG_FREQ = 10  # Log images every 10 rollouts
    # --- END MODIFIED ---
    LOG_DIR: Optional[str] = None
    LOG_SHAPE_PLACEMENT_Q_VALUES = False


class DemoConfig:
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = VisConfig.BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"


File: config\general.py
# File: config/general.py
# File: config/general.py
import torch
import os
import time
from utils.helpers import get_device

DEVICE = get_device()
RANDOM_SEED = 42
RUN_ID = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
BASE_CHECKPOINT_DIR = "checkpoints"
BASE_LOG_DIR = "logs"

# --- MODIFIED: Reduced total steps ---
TOTAL_TRAINING_STEPS = 10_000_000  # 10 Million steps
# --- END MODIFIED ---

RUN_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, RUN_ID)
RUN_LOG_DIR = os.path.join(BASE_LOG_DIR, "tensorboard", RUN_ID)

MODEL_SAVE_PATH = os.path.join(RUN_CHECKPOINT_DIR, "ppo_agent_state.pth")


File: config\utils.py
import torch
from typing import Dict, Any
from .core import (
    VisConfig,
    EnvConfig,
    RewardConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    DemoConfig,
)
from .general import DEVICE, RANDOM_SEED, RUN_ID


def get_config_dict() -> Dict[str, Any]:
    """Returns a flat dictionary of all relevant config values for logging."""
    all_configs = {}

    def flatten_class(cls, prefix=""):
        d = {}
        for k, v in vars(cls).items():
            if (
                not k.startswith("__")
                and not callable(v)
                and not isinstance(v, type)
                and not hasattr(v, "__module__")
            ):
                if isinstance(getattr(cls, k, None), property):
                    try:
                        v = getattr(cls(), k)
                    except Exception:
                        continue
                d[f"{prefix}{k}"] = v
        return d

    all_configs.update(flatten_class(VisConfig, "Vis."))
    all_configs.update(flatten_class(EnvConfig, "Env."))
    all_configs.update(flatten_class(RewardConfig, "Reward."))
    all_configs.update(flatten_class(PPOConfig, "PPO."))
    all_configs.update(flatten_class(RNNConfig, "RNN."))
    all_configs.update(flatten_class(TrainConfig, "Train."))
    all_configs.update(flatten_class(ModelConfig.Network, "Model.Net."))
    all_configs.update(flatten_class(StatsConfig, "Stats."))
    all_configs.update(flatten_class(TensorBoardConfig, "TB."))
    all_configs.update(flatten_class(DemoConfig, "Demo."))

    all_configs["General.DEVICE"] = str(DEVICE)
    all_configs["General.RANDOM_SEED"] = RANDOM_SEED
    all_configs["General.RUN_ID"] = RUN_ID

    all_configs = {
        k: v for k, v in all_configs.items() if not (k.endswith("_PATH") and v is None)
    }

    for key, value in all_configs.items():
        if isinstance(value, type) and issubclass(value, torch.nn.Module):
            all_configs[key] = value.__name__
        elif isinstance(value, (list, tuple)):
            all_configs[key] = str(value)
        if not isinstance(value, (int, float, str, bool)):
            all_configs[key] = str(value)

    return all_configs


File: config\validation.py
import os, torch
from .core import (
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    VisConfig,
    DemoConfig,
)
from .general import (
    RUN_ID,
    DEVICE,
    MODEL_SAVE_PATH,
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,
    TOTAL_TRAINING_STEPS,
)


def print_config_info_and_validate():
    env_config_instance = EnvConfig()
    ppo_config_instance = PPOConfig()
    rnn_config_instance = RNNConfig()

    print("-" * 70)
    print(f"RUN ID: {RUN_ID}")
    print(f"Log Directory: {RUN_LOG_DIR}")
    print(f"Checkpoint Directory: {RUN_CHECKPOINT_DIR}")
    print(f"Device: {DEVICE}")
    print(
        f"TB Logging: Histograms={'ON' if TensorBoardConfig.LOG_HISTOGRAMS else 'OFF'}, "
        f"Images={'ON' if TensorBoardConfig.LOG_IMAGES else 'OFF'}"
    )

    if TrainConfig.LOAD_CHECKPOINT_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD CHECKPOINT from: {TrainConfig.LOAD_CHECKPOINT_PATH} ***\n"
            "*** Ensure ckpt matches current Model/PPO/RNN Config. ***\n" + "*" * 70
        )
    else:
        print("--- Starting training from scratch (no checkpoint specified). ---")

    print(f"--- Using PPO Algorithm ---")
    print(f"    Rollout Steps: {ppo_config_instance.NUM_STEPS_PER_ROLLOUT}")
    print(f"    PPO Epochs: {ppo_config_instance.PPO_EPOCHS}")
    print(
        f"    Minibatches: {ppo_config_instance.NUM_MINIBATCHES} (Size: {ppo_config_instance.MINIBATCH_SIZE})"
    )
    print(f"    Clip Param: {ppo_config_instance.CLIP_PARAM}")
    print(f"    GAE Lambda: {ppo_config_instance.GAE_LAMBDA}")
    print(
        f"    Value Coef: {ppo_config_instance.VALUE_LOSS_COEF}, Entropy Coef: {ppo_config_instance.ENTROPY_COEF}"
    )
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
        f"--- Using LR Scheduler: {ppo_config_instance.USE_LR_SCHEDULER}"
        + (
            f" (Linear Decay to {ppo_config_instance.LR_SCHEDULER_END_FRACTION * 100}%)"
            if ppo_config_instance.USE_LR_SCHEDULER
            else ""
        )
        + " ---"
    )

    print(
        f"Config: Env=(R={env_config_instance.ROWS}, C={env_config_instance.COLS}), "
        f"GridState={env_config_instance.GRID_STATE_SHAPE}, "
        f"ShapeState={env_config_instance.SHAPE_STATE_DIM}, "
        f"ActionDim={env_config_instance.ACTION_DIM}"
    )
    cnn_str = str(ModelConfig.Network.CONV_CHANNELS).replace(" ", "")
    mlp_str = str(ModelConfig.Network.COMBINED_FC_DIMS).replace(" ", "")
    shape_mlp_cfg_str = str(ModelConfig.Network.SHAPE_FEATURE_MLP_DIMS).replace(" ", "")
    print(f"Network: CNN={cnn_str}, ShapeMLP={shape_mlp_cfg_str}, Fusion={mlp_str}")

    print(
        f"Training: NUM_ENVS={env_config_instance.NUM_ENVS}, TOTAL_STEPS={TOTAL_TRAINING_STEPS/1e6:.1f}M"
    )
    print(
        f"Stats: AVG_WINDOWS={StatsConfig.STATS_AVG_WINDOW}, Console Log Freq={StatsConfig.CONSOLE_LOG_FREQ} (rollouts)"
    )

    if env_config_instance.NUM_ENVS >= 1024:
        print(
            "*" * 70
            + f"\n*** Warning: NUM_ENVS={env_config_instance.NUM_ENVS}. Monitor system resources. ***"
            + (
                "\n*** Using MPS device. Performance varies. Force CPU via env var if needed. ***"
                if DEVICE.type == "mps"
                else ""
            )
            + "\n"
            + "*" * 70
        )
    print(
        f"--- Rendering {VisConfig.NUM_ENVS_TO_RENDER if VisConfig.NUM_ENVS_TO_RENDER > 0 else 'ALL'} of {env_config_instance.NUM_ENVS} environments ---"
    )
    print("-" * 70)


File: config\__init__.py
# File: config/__init__.py
from .core import (
    VisConfig,
    EnvConfig,
    RewardConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    DemoConfig,
)
from .general import (
    DEVICE,
    RANDOM_SEED,
    RUN_ID,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,
    MODEL_SAVE_PATH,
    TOTAL_TRAINING_STEPS,
)
from .utils import get_config_dict
from .validation import print_config_info_and_validate

# Assign RUN_LOG_DIR to TensorBoardConfig after imports
TensorBoardConfig.LOG_DIR = RUN_LOG_DIR

__all__ = [
    # Core Classes
    "VisConfig",
    "EnvConfig",
    "RewardConfig",
    "PPOConfig",
    "RNNConfig",
    "TrainConfig",
    "ModelConfig",
    "StatsConfig",
    "TensorBoardConfig",
    "DemoConfig",
    # General Constants/Paths
    "DEVICE",
    "RANDOM_SEED",
    "RUN_ID",
    "BASE_CHECKPOINT_DIR",
    "BASE_LOG_DIR",
    "RUN_CHECKPOINT_DIR",
    "RUN_LOG_DIR",
    "MODEL_SAVE_PATH",
    "TOTAL_TRAINING_STEPS",
    # Utils/Validation
    "get_config_dict",
    "print_config_info_and_validate",
]


File: environment\game_state.py
# File: environment/game_state.py
import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from collections import deque
from typing import Deque
import copy

from .grid import Grid
from .shape import Shape
from config import EnvConfig, RewardConfig

StateType = Dict[str, np.ndarray]


class GameState:
    def __init__(self):
        self.env_config = EnvConfig()
        self.rewards = RewardConfig()
        self.grid = Grid(self.env_config)
        self.shapes: List[Optional[Shape]] = []
        self.score = 0.0
        self.game_score = 0
        self.lines_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0

        self.blink_time = 0.0
        self.last_time = time.time()
        self.freeze_time = 0.0
        self.line_clear_flash_time = 0.0
        self.line_clear_highlight_time: float = 0.0
        self.game_over_flash_time: float = 0.0
        self.cleared_triangles_coords: List[Tuple[int, int]] = []

        self.game_over = False
        self._last_action_valid = True

        self.demo_selected_shape_idx: int = 0
        self.demo_target_row: int = self.env_config.ROWS // 2
        self.demo_target_col: int = self.env_config.COLS // 2

        self.reset()

    def reset(self) -> StateType:
        self.grid = Grid(self.env_config)
        self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]
        self.score = 0.0
        self.game_score = 0
        self.lines_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0

        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.line_clear_flash_time = 0.0
        self.line_clear_highlight_time = 0.0
        self.game_over_flash_time = 0.0
        self.cleared_triangles_coords = []

        self.game_over = False
        self._last_action_valid = True
        self.last_time = time.time()

        self.demo_selected_shape_idx = 0
        self.demo_target_row = self.env_config.ROWS // 2
        self.demo_target_col = self.env_config.COLS // 2
        return self.get_state()

    def valid_actions(self) -> List[int]:
        if self.game_over or self.freeze_time > 0:
            return []
        valid_action_indices: List[int] = []
        locations_per_shape = self.grid.rows * self.grid.cols
        for shape_slot_index, current_shape in enumerate(self.shapes):
            if not current_shape:
                continue
            for target_row in range(self.grid.rows):
                for target_col in range(self.grid.cols):
                    if self.grid.can_place(current_shape, target_row, target_col):
                        action_index = shape_slot_index * locations_per_shape + (
                            target_row * self.grid.cols + target_col
                        )
                        valid_action_indices.append(action_index)
        return valid_action_indices

    def _check_fundamental_game_over(self) -> bool:
        for current_shape in self.shapes:
            if not current_shape:
                continue
            for target_row in range(self.grid.rows):
                for target_col in range(self.grid.cols):
                    if self.grid.can_place(current_shape, target_row, target_col):
                        return False
        return True

    def is_over(self) -> bool:
        return self.game_over

    def is_frozen(self) -> bool:
        return self.freeze_time > 0

    def is_line_clearing(self) -> bool:
        return self.line_clear_flash_time > 0

    def is_highlighting_cleared(self) -> bool:
        return self.line_clear_highlight_time > 0

    def is_game_over_flashing(self) -> bool:
        return self.game_over_flash_time > 0

    def is_blinking(self) -> bool:
        return self.blink_time > 0

    def get_cleared_triangle_coords(self) -> List[Tuple[int, int]]:
        return self.cleared_triangles_coords

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        locations_per_shape = self.grid.rows * self.grid.cols
        shape_slot_index = action_index // locations_per_shape
        position_index = action_index % locations_per_shape
        target_row = position_index // self.grid.cols
        target_col = position_index % self.grid.cols
        return (shape_slot_index, target_row, target_col)

    def _update_timers(self):
        now = time.time()
        delta_time = now - self.last_time
        self.last_time = now
        self.freeze_time = max(0, self.freeze_time - delta_time)
        self.blink_time = max(0, self.blink_time - delta_time)
        self.line_clear_flash_time = max(0, self.line_clear_flash_time - delta_time)
        self.line_clear_highlight_time = max(
            0, self.line_clear_highlight_time - delta_time
        )
        self.game_over_flash_time = max(0, self.game_over_flash_time - delta_time)
        if self.line_clear_highlight_time <= 0 and self.cleared_triangles_coords:
            self.cleared_triangles_coords = []

    def _calculate_placement_reward(self, placed_shape: Shape) -> float:
        return self.rewards.REWARD_PLACE_PER_TRI * len(placed_shape.triangles)

    def _calculate_line_clear_reward(self, lines_cleared: int) -> float:
        if lines_cleared == 1:
            return self.rewards.REWARD_CLEAR_1
        elif lines_cleared == 2:
            return self.rewards.REWARD_CLEAR_2
        elif lines_cleared >= 3:
            return self.rewards.REWARD_CLEAR_3PLUS
        else:
            return 0.0

    def _calculate_state_penalty(self) -> float:
        penalty = 0.0
        max_height = self.grid.get_max_height()
        bumpiness = self.grid.get_bumpiness()
        num_holes = self.grid.count_holes()

        penalty += max_height * self.rewards.PENALTY_MAX_HEIGHT_FACTOR
        penalty += bumpiness * self.rewards.PENALTY_BUMPINESS_FACTOR
        penalty += num_holes * self.rewards.PENALTY_HOLE_PER_HOLE
        return penalty

    def _handle_invalid_placement(self) -> float:
        self._last_action_valid = False
        reward = self.rewards.PENALTY_INVALID_MOVE
        return reward

    def _handle_game_over_state_change(self) -> float:
        if self.game_over:
            return 0.0
        self.game_over = True
        if self.freeze_time <= 0:
            self.freeze_time = 1.0
        self.game_over_flash_time = 0.6
        return self.rewards.PENALTY_GAME_OVER

    def _handle_valid_placement(
        self,
        shape_to_place: Shape,
        shape_slot_index: int,
        target_row: int,
        target_col: int,
    ) -> float:
        self._last_action_valid = True
        step_reward = 0.0

        step_reward += self._calculate_placement_reward(shape_to_place)
        holes_before = self.grid.count_holes()

        self.grid.place(shape_to_place, target_row, target_col)
        self.shapes[shape_slot_index] = None
        self.game_score += len(shape_to_place.triangles)
        self.pieces_placed_this_episode += 1

        lines_cleared, triangles_cleared, cleared_coords = self.grid.clear_filled_rows()
        self.lines_cleared_this_episode += lines_cleared
        step_reward += self._calculate_line_clear_reward(lines_cleared)

        if triangles_cleared > 0:
            self.game_score += triangles_cleared * 2
            self.blink_time = 0.5
            self.freeze_time = 0.5
            self.line_clear_flash_time = 0.3
            self.line_clear_highlight_time = 0.5
            self.cleared_triangles_coords = cleared_coords

        holes_after = self.grid.count_holes()
        new_holes_created = max(0, holes_after - holes_before)

        step_reward += self._calculate_state_penalty()
        step_reward += new_holes_created * self.rewards.PENALTY_NEW_HOLE

        if all(s is None for s in self.shapes):
            self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]

        if self._check_fundamental_game_over():
            step_reward += self._handle_game_over_state_change()

        self._update_demo_selection_after_placement(shape_slot_index)

        return step_reward

    def step(self, action_index: int) -> Tuple[float, bool]:
        self._update_timers()

        if self.game_over:
            return (0.0, True)

        if self.is_frozen():
            return (self.rewards.REWARD_ALIVE_STEP, False)

        shape_slot_index, target_row, target_col = self.decode_action(action_index)

        shape_to_place = (
            self.shapes[shape_slot_index]
            if 0 <= shape_slot_index < len(self.shapes)
            else None
        )
        is_valid_placement = shape_to_place is not None and self.grid.can_place(
            shape_to_place, target_row, target_col
        )

        if is_valid_placement:
            step_reward = self._handle_valid_placement(
                shape_to_place, shape_slot_index, target_row, target_col
            )
        else:
            step_reward = self._handle_invalid_placement()
            if self._check_fundamental_game_over():
                step_reward += self._handle_game_over_state_change()

        if not self.game_over:
            step_reward += self.rewards.REWARD_ALIVE_STEP

        self.score += step_reward
        return (step_reward, self.game_over)

    def _calculate_potential_placement_outcomes(self) -> Dict[str, float]:
        """
        Simulates all valid placements to find summary outcome statistics.
        Returns a dictionary with keys: 'max_lines', 'min_holes', 'min_height', 'min_bump'.
        """
        valid_actions = self.valid_actions()
        if not valid_actions:
            return {
                "max_lines": 0.0,
                "min_holes": 0.0,
                "min_height": float(self.grid.get_max_height()),
                "min_bump": float(self.grid.get_bumpiness()),
            }

        max_lines_cleared = 0
        min_new_holes = float("inf")
        min_resulting_height = float("inf")
        min_resulting_bumpiness = float("inf")

        initial_holes = self.grid.count_holes()

        for action_index in valid_actions:
            shape_slot_index, target_row, target_col = self.decode_action(action_index)
            shape_to_place = self.shapes[shape_slot_index]

            if shape_to_place is None:
                continue

            temp_grid = copy.deepcopy(self.grid)
            temp_grid.place(shape_to_place, target_row, target_col)
            lines_cleared, _, _ = temp_grid.clear_filled_rows()
            holes_after = temp_grid.count_holes()
            height_after = temp_grid.get_max_height()
            bumpiness_after = temp_grid.get_bumpiness()

            new_holes_created = max(0, holes_after - initial_holes)

            max_lines_cleared = max(max_lines_cleared, lines_cleared)
            min_new_holes = min(min_new_holes, new_holes_created)
            min_resulting_height = min(min_resulting_height, height_after)
            min_resulting_bumpiness = min(min_resulting_bumpiness, bumpiness_after)

        if min_new_holes == float("inf"):
            min_new_holes = 0.0
        if min_resulting_height == float("inf"):
            min_resulting_height = float(self.grid.get_max_height())
        if min_resulting_bumpiness == float("inf"):
            min_resulting_bumpiness = float(self.grid.get_bumpiness())

        return {
            "max_lines": float(max_lines_cleared),
            "min_holes": float(min_new_holes),
            "min_height": float(min_resulting_height),
            "min_bump": float(min_resulting_bumpiness),
        }

    def get_state(self) -> StateType:
        grid_state = self.grid.get_feature_matrix()

        shape_features_per = self.env_config.SHAPE_FEATURES_PER_SHAPE
        num_shapes_expected = self.env_config.NUM_SHAPE_SLOTS
        shape_feature_matrix = np.zeros(
            (num_shapes_expected, shape_features_per), dtype=np.float32
        )

        max_tris_norm = 6.0
        max_h_norm = float(self.grid.rows)
        max_w_norm = float(self.grid.cols)

        for i in range(num_shapes_expected):
            s = self.shapes[i] if i < len(self.shapes) else None
            if s:
                tri_list = s.triangles
                n_tris = len(tri_list)
                ups = sum(1 for (_, _, is_up) in tri_list if is_up)
                downs = n_tris - ups
                min_r, min_c, max_r, max_c = s.bbox()
                height = max_r - min_r + 1
                width = max_c - min_c + 1

                shape_feature_matrix[i, 0] = np.clip(
                    float(n_tris) / max_tris_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 1] = np.clip(
                    float(ups) / max_tris_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 2] = np.clip(
                    float(downs) / max_tris_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 3] = np.clip(
                    float(height) / max_h_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 4] = np.clip(
                    float(width) / max_w_norm, 0.0, 1.0
                )

        shape_availability_dim = self.env_config.SHAPE_AVAILABILITY_DIM
        shape_availability_vector = np.zeros(shape_availability_dim, dtype=np.float32)
        for i in range(min(num_shapes_expected, shape_availability_dim)):
            if i < len(self.shapes) and self.shapes[i] is not None:
                shape_availability_vector[i] = 1.0

        explicit_features_dim = self.env_config.EXPLICIT_FEATURES_DIM
        explicit_features_vector = np.zeros(explicit_features_dim, dtype=np.float32)

        num_holes = self.grid.count_holes()
        col_heights = self.grid.get_column_heights()
        avg_height = np.mean(col_heights) if col_heights else 0
        max_height = max(col_heights) if col_heights else 0
        bumpiness = self.grid.get_bumpiness()

        max_possible_holes = self.env_config.ROWS * self.env_config.COLS
        max_possible_bumpiness = self.env_config.ROWS * (self.env_config.COLS - 1)

        explicit_features_vector[0] = np.clip(
            num_holes / max(1, max_possible_holes), 0.0, 1.0
        )
        explicit_features_vector[1] = np.clip(
            avg_height / self.env_config.ROWS, 0.0, 1.0
        )
        explicit_features_vector[2] = np.clip(
            max_height / self.env_config.ROWS, 0.0, 1.0
        )
        explicit_features_vector[3] = np.clip(
            bumpiness / max(1, max_possible_bumpiness), 0.0, 1.0
        )
        explicit_features_vector[4] = np.clip(
            self.lines_cleared_this_episode / 100.0, 0.0, 1.0
        )
        explicit_features_vector[5] = np.clip(
            self.pieces_placed_this_episode / 500.0, 0.0, 1.0
        )

        # --- MODIFIED: Conditionally calculate potential outcomes ---
        if self.env_config.CALCULATE_POTENTIAL_OUTCOMES_IN_STATE:
            potential_outcomes = self._calculate_potential_placement_outcomes()
            max_possible_lines = self.env_config.ROWS
            max_possible_new_holes = max_possible_holes

            explicit_features_vector[6] = np.clip(
                potential_outcomes["max_lines"] / max(1, max_possible_lines), 0.0, 1.0
            )
            explicit_features_vector[7] = np.clip(
                potential_outcomes["min_holes"] / max(1, max_possible_new_holes),
                0.0,
                1.0,
            )
            explicit_features_vector[8] = np.clip(
                potential_outcomes["min_height"] / self.env_config.ROWS, 0.0, 1.0
            )
            explicit_features_vector[9] = np.clip(
                potential_outcomes["min_bump"] / max(1, max_possible_bumpiness),
                0.0,
                1.0,
            )
        else:
            # Set placeholder values (e.g., 0) if not calculated
            explicit_features_vector[6:10] = 0.0
        # --- END MODIFIED ---

        state_dict: StateType = {
            "grid": grid_state.astype(np.float32),
            "shapes": shape_feature_matrix.astype(np.float32),
            "shape_availability": shape_availability_vector.astype(np.float32),
            "explicit_features": explicit_features_vector.astype(np.float32),
        }
        return state_dict

    def get_shapes(self) -> List[Optional[Shape]]:
        return self.shapes

    def _update_demo_selection_after_placement(self, placed_slot_index: int):
        num_slots = self.env_config.NUM_SHAPE_SLOTS
        if num_slots <= 0:
            return

        next_idx = (placed_slot_index + 1) % num_slots
        for _ in range(num_slots):
            if 0 <= next_idx < len(self.shapes) and self.shapes[next_idx] is not None:
                self.demo_selected_shape_idx = next_idx
                return
            next_idx = (next_idx + 1) % num_slots

        if all(s is None for s in self.shapes):
            first_available = next(
                (i for i, s in enumerate(self.shapes) if s is not None), 0
            )
            self.demo_selected_shape_idx = first_available

    def cycle_shape(self, direction: int):
        if self.game_over or self.freeze_time > 0:
            return
        num_slots = self.env_config.NUM_SHAPE_SLOTS
        if num_slots <= 0:
            return

        available_indices = [
            i for i, s in enumerate(self.shapes) if s is not None and 0 <= i < num_slots
        ]
        if not available_indices:
            return

        try:
            current_list_idx = available_indices.index(self.demo_selected_shape_idx)
        except ValueError:
            current_list_idx = 0
            if self.demo_selected_shape_idx not in available_indices:
                self.demo_selected_shape_idx = available_indices[0]

        new_list_idx = (current_list_idx + direction) % len(available_indices)
        self.demo_selected_shape_idx = available_indices[new_list_idx]

    def move_target(self, delta_row: int, delta_col: int):
        if self.game_over or self.freeze_time > 0:
            return
        self.demo_target_row = np.clip(
            self.demo_target_row + delta_row, 0, self.grid.rows - 1
        )
        self.demo_target_col = np.clip(
            self.demo_target_col + delta_col, 0, self.grid.cols - 1
        )

    def get_action_for_current_selection(self) -> Optional[int]:
        if self.game_over or self.freeze_time > 0:
            return None

        shape_slot_index = self.demo_selected_shape_idx
        current_shape = (
            self.shapes[shape_slot_index]
            if 0 <= shape_slot_index < len(self.shapes)
            else None
        )
        if current_shape is None:
            return None

        target_row, target_col = self.demo_target_row, self.demo_target_col

        if self.grid.can_place(current_shape, target_row, target_col):
            locations_per_shape = self.grid.rows * self.grid.cols
            action_index = shape_slot_index * locations_per_shape + (
                target_row * self.grid.cols + target_col
            )
            return action_index
        else:
            return None

    def get_current_selection_info(self) -> Tuple[Optional[Shape], int, int]:
        shape_slot_index = self.demo_selected_shape_idx
        current_shape = (
            self.shapes[shape_slot_index]
            if 0 <= shape_slot_index < len(self.shapes)
            else None
        )
        return current_shape, self.demo_target_row, self.demo_target_col


File: environment\grid.py
# File: environment/grid.py
import numpy as np
from typing import List, Tuple, Optional
from .triangle import Triangle
from .shape import Shape
from config import EnvConfig


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig):
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.triangles: List[List[Triangle]] = []
        self._create()

    def _create(self) -> None:
        # Determine playable columns based on row index (example pattern)
        # This specific pattern defines a hexagon-like board within the grid bounds
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]  # Example for 8 rows

        if len(cols_per_row) != self.rows:
            raise ValueError(
                f"Grid._create error: Length of cols_per_row ({len(cols_per_row)}) must match EnvConfig.ROWS ({self.rows})"
            )
        if max(cols_per_row) > self.cols:
            raise ValueError(
                f"Grid._create error: Max playable columns ({max(cols_per_row)}) exceeds EnvConfig.COLS ({self.cols})"
            )

        self.triangles = []
        for r in range(self.rows):
            rowt = []
            base_playable_cols = cols_per_row[r]

            # Calculate padding for death cells based on total cols and playable cols
            if base_playable_cols <= 0:
                initial_death_cols_left = self.cols  # All death if 0 playable
            elif base_playable_cols >= self.cols:
                initial_death_cols_left = 0  # No death if playable >= total
            else:
                initial_death_cols_left = (self.cols - base_playable_cols) // 2

            # Calculate the column index where death cells start on the right
            initial_first_death_col_right = initial_death_cols_left + base_playable_cols

            # --- Adjustment for Specific Hex Grid Pattern ---
            # This adjustment slightly shifts the death zones inward for the hex pattern
            # If you want a simple rectangle, remove this adjustment
            adjusted_death_cols_left = initial_death_cols_left + 1
            adjusted_first_death_col_right = initial_first_death_col_right - 1
            # --- End Adjustment ---

            for c in range(self.cols):
                # Determine if the cell is a death cell based on adjusted boundaries
                is_death_cell = (
                    (c < adjusted_death_cols_left)
                    or (
                        c >= adjusted_first_death_col_right
                        and adjusted_first_death_col_right > adjusted_death_cols_left
                    )
                    or (base_playable_cols <= 2)  # Treat very narrow rows as death
                )

                # Determine triangle orientation based on row and column index
                is_up_cell = (r + c) % 2 == 0
                tri = Triangle(r, c, is_up=is_up_cell, is_death=is_death_cell)
                rowt.append(tri)
            self.triangles.append(rowt)

    def valid(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(self, shp: Shape, rr: int, cc: int) -> bool:
        for dr, dc, up in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if not self.valid(nr, nc):
                return False
            # Check bounds for self.triangles access
            if not (
                0 <= nr < len(self.triangles) and 0 <= nc < len(self.triangles[nr])
            ):
                return False  # Should not happen if self.valid passed, but safety check
            tri = self.triangles[nr][nc]
            if tri.is_death or tri.is_occupied or (tri.is_up != up):
                return False
        return True

    def place(self, shp: Shape, rr: int, cc: int) -> None:
        for dr, dc, _ in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if self.valid(nr, nc):
                # Check bounds again before accessing
                if not (
                    0 <= nr < len(self.triangles) and 0 <= nc < len(self.triangles[nr])
                ):
                    continue
                tri = self.triangles[nr][nc]
                # Only place if the target cell is valid (not death, not occupied)
                if not tri.is_death and not tri.is_occupied:
                    tri.is_occupied = True
                    tri.color = shp.color  # Assign shape color

    def clear_filled_rows(self) -> Tuple[int, int, List[Tuple[int, int]]]:
        lines_cleared = 0
        triangles_cleared = 0
        rows_to_clear_indices = []
        cleared_triangles_coords: List[Tuple[int, int]] = []

        # Identify full rows
        for r in range(self.rows):
            if not (0 <= r < len(self.triangles)):
                continue  # Bounds check
            rowt = self.triangles[r]
            is_row_full = True
            num_placeable_triangles_in_row = 0
            for t in rowt:
                if not t.is_death:
                    num_placeable_triangles_in_row += 1
                    if not t.is_occupied:
                        is_row_full = False
                        break  # Can stop checking this row

            # A row is considered full if all non-death cells are occupied
            if is_row_full and num_placeable_triangles_in_row > 0:
                rows_to_clear_indices.append(r)
                lines_cleared += 1

        # Clear the identified rows
        for r_idx in rows_to_clear_indices:
            if not (0 <= r_idx < len(self.triangles)):
                continue  # Bounds check
            for t in self.triangles[r_idx]:
                if not t.is_death and t.is_occupied:
                    triangles_cleared += 1
                    t.is_occupied = False
                    t.color = None
                    cleared_triangles_coords.append(
                        (r_idx, t.col)
                    )  # Store coords for visualization

        # (No gravity/dropping logic is implemented here)

        return lines_cleared, triangles_cleared, cleared_triangles_coords

    def get_column_heights(self) -> List[int]:
        """Calculates the height of occupied cells in each column."""
        heights = [0] * self.cols
        for c in range(self.cols):
            for r in range(self.rows - 1, -1, -1):
                # Check bounds before accessing triangles
                if 0 <= r < len(self.triangles) and 0 <= c < len(self.triangles[r]):
                    tri = self.triangles[r][c]
                    # Check if the cell is occupied and not a death cell
                    if tri.is_occupied and not tri.is_death:
                        heights[c] = r + 1  # Height is row index + 1
                        break  # Found highest occupied cell in this column
        return heights

    def get_max_height(self) -> int:
        """Calculates the maximum height across all columns."""
        heights = self.get_column_heights()
        return max(heights) if heights else 0

    def get_bumpiness(self) -> int:
        """Calculates the sum of absolute height differences between adjacent columns."""
        heights = self.get_column_heights()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def count_holes(self) -> int:
        """Counts the number of empty, non-death cells below an occupied cell in the same column."""
        holes = 0
        for c in range(self.cols):
            occupied_above = False
            for r in range(self.rows):  # Iterate from top to bottom
                # Check bounds before accessing triangles
                if not (
                    0 <= r < len(self.triangles) and 0 <= c < len(self.triangles[r])
                ):
                    continue  # Skip if out of bounds
                tri = self.triangles[r][c]

                # Skip death cells entirely, they don't count as holes or blockers
                if tri.is_death:
                    # If we hit a death cell below the highest block, reset occupied_above?
                    # Or just skip? Let's just skip for simplicity. Holes are non-death cells.
                    continue

                if tri.is_occupied:
                    occupied_above = (
                        True  # Mark that we've seen an occupied cell in this column
                    )
                elif not tri.is_occupied and occupied_above:
                    # This is an empty, non-death cell below an occupied cell in the same column
                    holes += 1
        return holes

    def get_feature_matrix(self) -> np.ndarray:
        """Creates a 2-channel feature matrix: [occupied, is_up]."""
        # Channel 0: Occupied (1.0) or Empty (0.0) - only for non-death cells
        # Channel 1: Orientation (1.0 if Up, 0.0 if Down) - only for non-death cells
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                # Check bounds before accessing triangles
                if not (
                    0 <= r < len(self.triangles) and 0 <= c < len(self.triangles[r])
                ):
                    continue  # Skip if out of bounds
                t = self.triangles[r][c]
                # Only populate features for non-death cells
                if not t.is_death:
                    grid_state[0, r, c] = 1.0 if t.is_occupied else 0.0
                    grid_state[1, r, c] = 1.0 if t.is_up else 0.0
                    # Optionally add more channels here (e.g., cell age, color?)
        return grid_state


File: environment\shape.py
# File: environment/shape.py
# (No changes needed)
import random
from typing import List, Tuple
from config import EnvConfig, VisConfig  # Needs VisConfig only for colors

GOOGLE_COLORS = VisConfig.GOOGLE_COLORS  # Use colors from VisConfig


class Shape:
    """Represents a polyomino-like shape made of triangles."""

    def __init__(self) -> None:
        # List of (relative_row, relative_col, is_up) tuples defining the shape
        self.triangles: List[Tuple[int, int, bool]] = []
        self.color: Tuple[int, int, int] = random.choice(GOOGLE_COLORS)
        self._generate()  # Generate the shape structure

    def _generate(self) -> None:
        """Generates a random shape by adding adjacent triangles."""
        n = random.randint(1, 5)  # Number of triangles in the shape
        first_up = random.choice([True, False])  # Orientation of the root triangle
        self.triangles.append((0, 0, first_up))  # Add the root triangle at (0,0)

        # Add remaining triangles adjacent to existing ones
        for _ in range(n - 1):
            # Find valid neighbors of the *last added* triangle
            lr, lc, lu = self.triangles[-1]
            nbrs = self._find_valid_neighbors(lr, lc, lu)
            if nbrs:
                self.triangles.append(random.choice(nbrs))
            # else: Could break early if no valid neighbors found, shape < n

    def _find_valid_neighbors(
        self, r: int, c: int, up: bool
    ) -> List[Tuple[int, int, bool]]:
        """Finds potential neighbor triangles that are not already part of the shape."""
        if up:  # Neighbors of an UP triangle are DOWN triangles
            ns = [(r, c - 1, False), (r, c + 1, False), (r + 1, c, False)]
        else:  # Neighbors of a DOWN triangle are UP triangles
            ns = [(r, c - 1, True), (r, c + 1, True), (r - 1, c, True)]
        # Return only neighbors that are not already in self.triangles
        return [n for n in ns if n not in self.triangles]

    def bbox(self) -> Tuple[int, int, int, int]:
        """Calculates the bounding box (min_r, min_c, max_r, max_c) of the shape."""
        if not self.triangles:
            return (0, 0, 0, 0)
        rr = [t[0] for t in self.triangles]
        cc = [t[1] for t in self.triangles]
        return (min(rr), min(cc), max(rr), max(cc))


File: environment\triangle.py
# File: environment/triangle.py
# (No changes needed)
from typing import Tuple, Optional, List


class Triangle:
    """Represents a single triangular cell on the grid."""

    def __init__(self, row: int, col: int, is_up: bool, is_death: bool = False):
        self.row = row
        self.col = col
        self.is_up = is_up  # True if pointing up, False if pointing down
        self.is_death = is_death  # True if part of the unplayable border
        self.is_occupied = is_death  # Occupied if it's a death cell initially
        self.color: Optional[Tuple[int, int, int]] = (
            None  # Color if occupied by a shape
        )

    def get_points(
        self, ox: int, oy: int, cw: int, ch: int
    ) -> List[Tuple[float, float]]:
        """Calculates the vertex points for drawing the triangle."""
        x = ox + self.col * (
            cw * 0.75
        )  # Horizontal position based on column and overlap
        y = oy + self.row * ch  # Vertical position based on row
        if self.is_up:
            # Points for an upward-pointing triangle
            return [(x, y + ch), (x + cw, y + ch), (x + cw / 2, y)]
        else:
            # Points for a downward-pointing triangle
            return [(x, y), (x + cw, y), (x + cw / 2, y + ch)]


File: environment\__init__.py


File: init\rl_components_ppo.py
# File: init/rl_components_ppo.py
import sys
import traceback
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Callable

from config import (
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    RewardConfig,
    TensorBoardConfig,
    DEVICE,
    MODEL_SAVE_PATH,
    get_config_dict,
)

try:
    from environment.game_state import GameState, StateType
except ImportError as e:
    print(f"Error importing environment: {e}")
    sys.exit(1)

from agent.ppo_agent import PPOAgent
from training.trainer import Trainer
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregator
from stats.simple_stats_recorder import SimpleStatsRecorder
from stats.tensorboard_logger import TensorBoardStatsRecorder


def initialize_envs(num_envs: int, env_config: EnvConfig) -> List[GameState]:
    """Initializes the specified number of game environments."""
    print(f"Initializing {num_envs} game environments...")
    try:
        envs = [GameState() for _ in range(num_envs)]
        # Basic validation on the first environment
        s_test_dict = envs[0].reset()

        if not isinstance(s_test_dict, dict):
            raise TypeError("Env reset did not return a dictionary state.")

        # --- Check Grid ---
        if "grid" not in s_test_dict:
            raise KeyError("State dict missing 'grid'")
        grid_state = s_test_dict["grid"]
        expected_grid_shape = env_config.GRID_STATE_SHAPE
        if (
            not isinstance(grid_state, np.ndarray)
            or grid_state.shape != expected_grid_shape
        ):
            raise ValueError(
                f"Initial grid state shape mismatch! Env:{grid_state.shape}, Cfg:{expected_grid_shape}"
            )
        print(f"Initial 'grid' state shape check PASSED: {grid_state.shape}")

        # --- Check Shapes Features ---
        if "shapes" not in s_test_dict:
            raise KeyError("State dict missing 'shapes'")
        shape_state = s_test_dict["shapes"]
        expected_shape_feature_shape = (
            env_config.NUM_SHAPE_SLOTS,
            env_config.SHAPE_FEATURES_PER_SHAPE,
        )
        if (
            not isinstance(shape_state, np.ndarray)
            or shape_state.shape != expected_shape_feature_shape
        ):
            raise ValueError(
                f"Initial shape feature shape mismatch! Env:{shape_state.shape}, Cfg:{expected_shape_feature_shape}"
            )
        print(f"Initial 'shapes' feature shape check PASSED: {shape_state.shape}")

        # --- Check Shape Availability ---
        if "shape_availability" not in s_test_dict:
            raise KeyError("State dict missing 'shape_availability'")
        availability_state = s_test_dict["shape_availability"]
        expected_availability_shape = (env_config.SHAPE_AVAILABILITY_DIM,)
        if (
            not isinstance(availability_state, np.ndarray)
            or availability_state.shape != expected_availability_shape
        ):
            raise ValueError(
                f"Initial shape availability shape mismatch! Env:{availability_state.shape}, Cfg:{expected_availability_shape}"
            )
        print(
            f"Initial 'shape_availability' state shape check PASSED: {availability_state.shape}"
        )

        # --- UPDATED: Check Explicit Features (with new dimension) ---
        if "explicit_features" not in s_test_dict:
            raise KeyError("State dict missing 'explicit_features'")
        explicit_features_state = s_test_dict["explicit_features"]
        expected_explicit_features_shape = (env_config.EXPLICIT_FEATURES_DIM,)
        if (
            not isinstance(explicit_features_state, np.ndarray)
            or explicit_features_state.shape != expected_explicit_features_shape
        ):
            raise ValueError(
                f"Initial explicit features shape mismatch! Env:{explicit_features_state.shape}, Cfg:{expected_explicit_features_shape}"
            )
        print(
            f"Initial 'explicit_features' state shape check PASSED: {explicit_features_state.shape}"
        )
        # --- END UPDATED ---

        # Test step with a valid action if available
        valid_acts_init = envs[0].valid_actions()
        if valid_acts_init:
            _, _ = envs[0].step(valid_acts_init[0])
        else:
            print(
                "Warning: No valid actions available after initial reset for testing step()."
            )

        print(f"Successfully initialized {num_envs} environments.")
        return envs
    except Exception as e:
        print(f"FATAL ERROR during env init: {e}")
        traceback.print_exc()
        raise e


def initialize_agent(
    model_config: ModelConfig,
    ppo_config: PPOConfig,
    rnn_config: RNNConfig,
    env_config: EnvConfig,
) -> PPOAgent:
    """Initializes the PPO Agent."""
    print("Initializing PPO Agent...")
    agent = PPOAgent(
        model_config=model_config,
        ppo_config=ppo_config,
        rnn_config=rnn_config,
        env_config=env_config,
    )
    print("PPO Agent initialized.")
    return agent


def initialize_stats_recorder(
    stats_config: StatsConfig,
    tb_config: TensorBoardConfig,
    config_dict: Dict[str, Any],
    agent: Optional[PPOAgent],
    env_config: EnvConfig,
    rnn_config: RNNConfig,
    is_reinit: bool = False,
) -> StatsRecorderBase:
    """Initializes the statistics recording components."""
    print(f"Initializing Statistics Components... Re-init: {is_reinit}")
    stats_aggregator = StatsAggregator(
        avg_windows=stats_config.STATS_AVG_WINDOW,
        plot_window=stats_config.PLOT_DATA_WINDOW,
    )
    console_recorder = SimpleStatsRecorder(
        aggregator=stats_aggregator,
        console_log_interval=stats_config.CONSOLE_LOG_FREQ,
    )

    model_for_graph_cpu = None
    dummy_input_tuple = None

    if not is_reinit and agent and agent.network:
        print("[Stats Init] Preparing model copy and dummy input for graph...")
        try:
            # Prepare dummy input on CPU
            expected_grid_shape = env_config.GRID_STATE_SHAPE
            expected_shape_feat_dim = env_config.SHAPE_STATE_DIM
            expected_availability_dim = env_config.SHAPE_AVAILABILITY_DIM
            # --- UPDATED: Expected explicit features dim ---
            expected_explicit_feat_dim = env_config.EXPLICIT_FEATURES_DIM
            # --- END UPDATED ---

            dummy_grid_np = np.zeros(expected_grid_shape, dtype=np.float32)
            dummy_shapes_np = np.zeros(expected_shape_feat_dim, dtype=np.float32)
            dummy_availability_np = np.zeros(
                expected_availability_dim, dtype=np.float32
            )
            # --- UPDATED: Dummy explicit features ---
            dummy_explicit_features_np = np.zeros(
                expected_explicit_feat_dim, dtype=np.float32
            )
            # --- END UPDATED ---

            batch_dim = 1
            seq_dim = 1 if rnn_config.USE_RNN else 0

            def shape_with_batch_seq(base_shape):
                dims = ([batch_dim, seq_dim] if seq_dim else [batch_dim]) + list(
                    base_shape
                )
                if not base_shape:  # Handle scalar base shape
                    dims = [batch_dim, seq_dim] if seq_dim else [batch_dim]
                return tuple(dim for dim in dims if dim > 0)  # Filter out seq_dim if 0

            grid_dims = shape_with_batch_seq(expected_grid_shape)
            shape_dims = shape_with_batch_seq((expected_shape_feat_dim,))
            availability_dims = shape_with_batch_seq((expected_availability_dim,))
            # --- UPDATED: Explicit feature dims ---
            explicit_feat_dims = shape_with_batch_seq((expected_explicit_feat_dim,))
            # --- END UPDATED ---

            dummy_grid_cpu = torch.tensor(dummy_grid_np).reshape(grid_dims).to("cpu")
            dummy_shapes_cpu = (
                torch.tensor(dummy_shapes_np).reshape(shape_dims).to("cpu")
            )
            dummy_availability_cpu = (
                torch.tensor(dummy_availability_np).reshape(availability_dims).to("cpu")
            )
            # --- UPDATED: Explicit feature tensor ---
            dummy_explicit_features_cpu = (
                torch.tensor(dummy_explicit_features_np)
                .reshape(explicit_feat_dims)
                .to("cpu")
            )
            # --- END UPDATED ---

            # Create a copy of the network on CPU for graph tracing
            model_for_graph_cpu = type(agent.network)(
                env_config=env_config,
                action_dim=env_config.ACTION_DIM,
                model_config=agent.network.config,  # Use loaded agent's network config
                rnn_config=rnn_config,
            ).to("cpu")

            model_for_graph_cpu.load_state_dict(agent.network.state_dict())
            model_for_graph_cpu.eval()

            # --- MODIFIED: Prepare dummy input tuple (grid, shapes, availability, explicit_features) ---
            dummy_input_tuple = (
                dummy_grid_cpu,
                dummy_shapes_cpu,
                dummy_availability_cpu,
                dummy_explicit_features_cpu,  # Add explicit features
            )
            # --- END MODIFIED ---

            print("[Stats Init] Prepared model copy and dummy input on CPU for graph.")
        except Exception as e:
            print(f"Warning: Failed to prepare model/input for graph logging: {e}")
            traceback.print_exc()
            model_for_graph_cpu, dummy_input_tuple = None, None
    elif is_reinit:
        print("[Stats Init] Skipping graph model preparation during re-initialization.")

    print(f"Using TensorBoard Logger (Log Dir: {tb_config.LOG_DIR})")
    try:
        tb_recorder = TensorBoardStatsRecorder(
            aggregator=stats_aggregator,
            console_recorder=console_recorder,
            log_dir=tb_config.LOG_DIR,
            hparam_dict=config_dict if not is_reinit else None,
            model_for_graph=model_for_graph_cpu,
            dummy_input_for_graph=dummy_input_tuple,  # Pass the new tuple
            histogram_log_interval=(
                tb_config.HISTOGRAM_LOG_FREQ if tb_config.LOG_HISTOGRAMS else -1
            ),
            image_log_interval=(
                tb_config.IMAGE_LOG_FREQ if tb_config.LOG_IMAGES else -1
            ),
            env_config=env_config,
            rnn_config=rnn_config,
        )
        print("Statistics Components initialized successfully.")
        return tb_recorder
    except Exception as e:
        print(f"FATAL: Error initializing TensorBoardStatsRecorder: {e}. Exiting.")
        traceback.print_exc()
        raise e


def initialize_trainer(
    envs: List[GameState],
    agent: PPOAgent,
    stats_recorder: StatsRecorderBase,
    env_config: EnvConfig,
    ppo_config: PPOConfig,
    rnn_config: RNNConfig,
    train_config: TrainConfig,
    model_config: ModelConfig,
) -> Trainer:
    """Initializes the PPO Trainer."""
    print("Initializing PPO Trainer...")
    trainer = Trainer(
        envs=envs,
        agent=agent,
        stats_recorder=stats_recorder,
        env_config=env_config,
        ppo_config=ppo_config,
        rnn_config=rnn_config,
        train_config=train_config,
        model_config=model_config,
        model_save_path=MODEL_SAVE_PATH,
        load_checkpoint_path=train_config.LOAD_CHECKPOINT_PATH,
    )
    print("PPO Trainer initialization finished.")
    return trainer


File: init\__init__.py
# File: init/__init__.py
from .rl_components_ppo import (
    initialize_envs,
    initialize_agent,
    initialize_stats_recorder,
    initialize_trainer,
)

__all__ = [
    "initialize_envs",
    "initialize_agent",
    "initialize_stats_recorder",
    "initialize_trainer",
]


File: stats\aggregator.py
# File: stats/aggregator.py
# No changes needed here, the debug prints were already removed in the previous step.
# Keep the file as it was after the previous modification.
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, List
import numpy as np
from config import StatsConfig


class StatsAggregator:
    """
    Handles aggregation and storage of training statistics using deques.
    Calculates rolling averages and tracks best values. Does not perform logging.
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

        self.policy_losses: Deque[float] = deque(maxlen=plot_window)
        self.value_losses: Deque[float] = deque(maxlen=plot_window)
        self.entropies: Deque[float] = deque(maxlen=plot_window)
        self.grad_norms: Deque[float] = deque(maxlen=plot_window)
        self.avg_max_qs: Deque[float] = deque(maxlen=plot_window)
        self.episode_scores: Deque[float] = deque(maxlen=plot_window)
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_lines_cleared: Deque[int] = deque(maxlen=plot_window)
        self.sps_values: Deque[float] = deque(maxlen=plot_window)
        self.buffer_sizes: Deque[int] = deque(maxlen=plot_window)
        self.beta_values: Deque[float] = deque(maxlen=plot_window)
        self.best_rl_score_history: Deque[float] = deque(maxlen=plot_window)
        self.best_game_score_history: Deque[int] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(maxlen=plot_window)
        self.epsilon_values: Deque[float] = deque(maxlen=plot_window)

        self.total_episodes = 0
        self.total_lines_cleared = 0
        self.current_epsilon: float = 0.0
        self.current_beta: float = 0.0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0
        self.current_sps: float = 0.0
        self.current_lr: float = 0.0

        self.best_score: float = -float("inf")
        self.previous_best_score: float = -float("inf")
        self.best_score_step: int = 0

        self.best_game_score: float = -float("inf")
        self.previous_best_game_score: float = -float("inf")
        self.best_game_score_step: int = 0

        self.best_value_loss: float = float("inf")
        self.previous_best_value_loss: float = float("inf")
        self.best_value_loss_step: int = 0

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
        lines_cleared: Optional[int] = None,
    ) -> Dict[str, Any]:
        current_step = (
            global_step if global_step is not None else self.current_global_step
        )
        update_info = {"new_best_rl": False, "new_best_game": False}

        self.episode_scores.append(episode_score)
        self.episode_lengths.append(episode_length)
        if game_score is not None:
            self.game_scores.append(game_score)
        if lines_cleared is not None:
            self.episode_lines_cleared.append(lines_cleared)
            self.total_lines_cleared += lines_cleared
        self.total_episodes = episode_num

        if episode_score > self.best_score:
            self.previous_best_score = self.best_score
            self.best_score = episode_score
            self.best_score_step = current_step
            update_info["new_best_rl"] = True

        if game_score is not None and game_score > self.best_game_score:
            self.previous_best_game_score = self.best_game_score
            self.best_game_score = float(game_score)
            self.best_game_score_step = current_step
            update_info["new_best_game"] = True

        self.best_rl_score_history.append(self.best_score)
        current_best_game = (
            int(self.best_game_score) if self.best_game_score > -float("inf") else 0
        )
        self.best_game_score_history.append(current_best_game)

        return update_info

    def record_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        g_step = step_data.get("global_step", self.current_global_step)
        if g_step > self.current_global_step:
            self.current_global_step = g_step

        update_info = {"new_best_loss": False}

        if "policy_loss" in step_data and step_data["policy_loss"] is not None:
            loss_val = step_data["policy_loss"]
            if np.isfinite(loss_val):
                self.policy_losses.append(loss_val)
                # print(f"[Aggregator Debug] Appended Policy Loss...") # Removed
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Policy Loss: {loss_val}"
                )

        if "value_loss" in step_data and step_data["value_loss"] is not None:
            current_value_loss = step_data["value_loss"]
            if np.isfinite(current_value_loss):
                self.value_losses.append(current_value_loss)
                # print(f"[Aggregator Debug] Appended Value Loss...") # Removed
                if current_value_loss < self.best_value_loss and g_step > 0:
                    self.previous_best_value_loss = self.best_value_loss
                    self.best_value_loss = current_value_loss
                    self.best_value_loss_step = g_step
                    update_info["new_best_loss"] = True
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Value Loss: {current_value_loss}"
                )

        if "entropy" in step_data and step_data["entropy"] is not None:
            entropy_val = step_data["entropy"]
            if np.isfinite(entropy_val):
                self.entropies.append(entropy_val)
                # print(f"[Aggregator Debug] Appended Entropy...") # Removed
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Entropy: {entropy_val}"
                )

        if "grad_norm" in step_data and step_data["grad_norm"] is not None:
            self.grad_norms.append(step_data["grad_norm"])
        if "avg_max_q" in step_data and step_data["avg_max_q"] is not None:
            self.avg_max_qs.append(step_data["avg_max_q"])
        if "beta" in step_data and step_data["beta"] is not None:
            self.current_beta = step_data["beta"]
            self.beta_values.append(self.current_beta)
        if "buffer_size" in step_data and step_data["buffer_size"] is not None:
            self.current_buffer_size = step_data["buffer_size"]
            self.buffer_sizes.append(self.current_buffer_size)
        if "lr" in step_data and step_data["lr"] is not None:
            self.current_lr = step_data["lr"]
            self.lr_values.append(self.current_lr)
        if "epsilon" in step_data and step_data["epsilon"] is not None:
            self.current_epsilon = step_data["epsilon"]
            self.epsilon_values.append(self.current_epsilon)

        if "step_time" in step_data and step_data["step_time"] > 1e-9:
            num_steps = step_data.get("num_steps_processed", 1)
            sps = num_steps / step_data["step_time"]
            self.sps_values.append(sps)
            self.current_sps = sps

        return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        if current_global_step is None:
            current_global_step = self.current_global_step

        summary_window = self.summary_avg_window

        def safe_mean(q: Deque, default=0.0) -> float:
            window_data = list(q)[-summary_window:]
            finite_data = [x for x in window_data if np.isfinite(x)]
            return float(np.mean(finite_data)) if finite_data else default

        summary = {
            "avg_score_window": safe_mean(self.episode_scores),
            "avg_length_window": safe_mean(self.episode_lengths),
            "policy_loss": safe_mean(self.policy_losses),
            "value_loss": safe_mean(self.value_losses),
            "entropy": safe_mean(self.entropies),
            "avg_max_q_window": safe_mean(self.avg_max_qs),
            "avg_game_score_window": safe_mean(self.game_scores),
            "avg_lines_cleared_window": safe_mean(self.episode_lines_cleared),
            "avg_sps_window": safe_mean(self.sps_values, default=self.current_sps),
            "avg_lr_window": safe_mean(self.lr_values, default=self.current_lr),
            "total_episodes": self.total_episodes,
            "beta": self.current_beta,
            "buffer_size": self.current_buffer_size,
            "steps_per_second": self.current_sps,
            "global_step": current_global_step,
            "current_lr": self.current_lr,
            "best_score": self.best_score,
            "previous_best_score": self.previous_best_score,
            "best_score_step": self.best_score_step,
            "best_game_score": self.best_game_score,
            "previous_best_game_score": self.previous_best_game_score,
            "best_game_score_step": self.best_game_score_step,
            "best_loss": self.best_value_loss,
            "previous_best_loss": self.previous_best_value_loss,
            "best_loss_step": self.best_value_loss_step,
            "num_ep_scores": len(self.episode_scores),
            "num_losses": len(
                self.value_losses
            ),  # Keep track of how many loss updates happened
            "summary_avg_window_size": summary_window,
        }
        return summary

    def get_plot_data(self) -> Dict[str, Deque]:
        return {
            "episode_scores": self.episode_scores.copy(),
            "episode_lengths": self.episode_lengths.copy(),
            "policy_loss": self.policy_losses.copy(),
            "value_loss": self.value_losses.copy(),
            "entropy": self.entropies.copy(),
            "avg_max_qs": self.avg_max_qs.copy(),
            "game_scores": self.game_scores.copy(),
            "episode_lines_cleared": self.episode_lines_cleared.copy(),
            "sps_values": self.sps_values.copy(),
            "buffer_sizes": self.buffer_sizes.copy(),
            "beta_values": self.beta_values.copy(),
            "best_rl_score_history": self.best_rl_score_history.copy(),
            "best_game_score_history": self.best_game_score_history.copy(),
            "lr_values": self.lr_values.copy(),
            "epsilon_values": self.epsilon_values.copy(),
        }


File: stats\simple_stats_recorder.py
# File: stats/simple_stats_recorder.py
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List
import numpy as np
import torch
from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from config import StatsConfig


class SimpleStatsRecorder(StatsRecorderBase):
    """
    Logs aggregated statistics to the console periodically.
    Delegates data storage and aggregation to a StatsAggregator instance.
    Provides no-op implementations for histogram, image, hparam, graph logging.
    """

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_log_interval: int = StatsConfig.CONSOLE_LOG_FREQ,
    ):
        self.aggregator = aggregator
        self.console_log_interval = (
            max(1, console_log_interval) if console_log_interval > 0 else -1
        )
        self.last_log_time: float = time.time()
        self.last_log_step: int = 0
        self.start_time: float = time.time()
        self.summary_avg_window = self.aggregator.summary_avg_window
        self.rollouts_since_last_log = 0  # Track rollouts for logging frequency
        print(
            f"[SimpleStatsRecorder] Initialized. Console Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'} rollouts"
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
        lines_cleared: Optional[int] = None,
    ):
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            lines_cleared,
        )
        current_step = (
            global_step
            if global_step is not None
            else self.aggregator.current_global_step
        )
        step_info = f"at Step ~{current_step/1e6:.1f}M"

        if update_info.get("new_best_rl"):
            prev_str = (
                f"{self.aggregator.previous_best_score:.2f}"
                if self.aggregator.previous_best_score > -float("inf")
                else "N/A"
            )
            print(
                f"\n--- 🏆 New Best RL: {self.aggregator.best_score:.2f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_game"):
            prev_str = (
                f"{self.aggregator.previous_best_game_score:.0f}"
                if self.aggregator.previous_best_game_score > -float("inf")
                else "N/A"
            )
            print(
                f"--- 🎮 New Best Game: {self.aggregator.best_game_score:.0f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_loss"):
            prev_str = (
                f"{self.aggregator.previous_best_value_loss:.4f}"
                if self.aggregator.previous_best_value_loss < float("inf")
                else "N/A"
            )
            print(
                f"---📉 New Best Loss: {self.aggregator.best_value_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )

    def record_step(self, step_data: Dict[str, Any]):
        update_info = self.aggregator.record_step(step_data)
        g_step = step_data.get("global_step", self.aggregator.current_global_step)

        if update_info.get("new_best_loss"):
            prev_str = (
                f"{self.aggregator.previous_best_value_loss:.4f}"
                if self.aggregator.previous_best_value_loss < float("inf")
                else "N/A"
            )
            step_info = f"at Step ~{g_step/1e6:.1f}M"
            print(
                f"---📉 New Best Loss: {self.aggregator.best_value_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )

        # Check if an agent update occurred (indicated by presence of loss keys)
        if "policy_loss" in step_data or "value_loss" in step_data:
            self.rollouts_since_last_log += 1
            self.log_summary(g_step)  # Attempt to log after each agent update

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        # Log based on number of rollouts (agent updates) since last log
        if (
            self.console_log_interval <= 0
            or self.rollouts_since_last_log < self.console_log_interval
        ):
            return

        summary = self.get_summary(global_step)
        elapsed_runtime = time.time() - self.start_time
        runtime_hrs = elapsed_runtime / 3600

        best_score_val = (
            f"{summary['best_score']:.2f}"
            if summary["best_score"] > -float("inf")
            else "N/A"
        )
        best_loss_val = (
            f"{summary['best_loss']:.4f}"
            if summary["best_loss"] < float("inf")
            else "N/A"
        )

        # --- MODIFIED: Use 'value_loss' key for the average loss display ---
        avg_window_size = summary.get("summary_avg_window_size", "?")
        log_str = (
            f"[{runtime_hrs:.1f}h|Console] Step: {global_step/1e6:<6.2f}M | "
            f"Ep: {summary['total_episodes']:<7} | SPS: {summary['steps_per_second']:<5.0f} | "
            f"RLScore(Avg{avg_window_size}): {summary['avg_score_window']:<6.2f} (Best: {best_score_val}) | "
            f"Loss(Avg{avg_window_size}): {summary['value_loss']:.4f} (Best: {best_loss_val}) | "  # Corrected key
            f"LR: {summary['current_lr']:.1e}"
        )
        # --- END MODIFIED ---

        epsilon = summary.get("epsilon")
        if epsilon is not None and epsilon < 1.0:
            log_str += f" | Eps: {epsilon:.3f}"

        beta = summary.get("beta")
        if beta is not None and beta > 0.0 and beta < 1.0:
            log_str += f" | Beta: {beta:.3f}"

        buffer_size = summary.get("buffer_size")
        if buffer_size is not None and buffer_size > 0:
            log_str += f" | Buf: {buffer_size/1e6:.2f}M"

        print(log_str)

        self.last_log_time = time.time()
        self.last_log_step = global_step
        self.rollouts_since_last_log = 0  # Reset counter after logging

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

    def close(self):
        print("[SimpleStatsRecorder] Closed.")


File: stats\stats_recorder.py
# File: stats/stats_recorder.py
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Dict, Any, Optional, Union
import numpy as np
import torch


class StatsRecorderBase(ABC):
    """Base class for recording training statistics."""

    @abstractmethod
    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        lines_cleared: Optional[int] = None,
    ):
        """Record stats for a completed episode."""
        pass

    @abstractmethod
    def record_step(self, step_data: Dict[str, Any]):
        """Record stats from a training or environment step."""
        pass

    @abstractmethod
    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Record a histogram of values."""
        pass

    @abstractmethod
    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Record an image."""
        pass

    @abstractmethod
    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Record hyperparameters and final/key metrics."""
        pass

    @abstractmethod
    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[Any] = None
    ):
        """Record the model graph."""
        pass

    @abstractmethod
    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Return a dictionary containing summary statistics."""
        pass

    @abstractmethod
    def get_plot_data(self) -> Dict[str, Deque]:
        """Return copies of data deques for plotting."""
        pass

    @abstractmethod
    def log_summary(self, global_step: int):
        """Trigger the logging action (e.g., print to console)."""
        pass

    @abstractmethod
    def close(self):
        """Perform any necessary cleanup."""
        pass


File: stats\tensorboard_logger.py
# File: stats/tensorboard_logger.py
import time
import traceback
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io
import PIL.Image

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from .simple_stats_recorder import SimpleStatsRecorder
from config import (
    TensorBoardConfig,
    EnvConfig,
    RNNConfig,
    VisConfig,
    StatsConfig,
)
from agent.networks.agent_network import ActorCriticNetwork


class TensorBoardStatsRecorder(StatsRecorderBase):
    """
    Logs aggregated statistics, histograms, images, and hyperparameters to TensorBoard.
    Uses a SimpleStatsRecorder for console logging and a StatsAggregator for data handling.
    """

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_recorder: SimpleStatsRecorder,
        log_dir: str,
        hparam_dict: Optional[Dict[str, Any]] = None,
        model_for_graph: Optional[ActorCriticNetwork] = None,
        dummy_input_for_graph: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        histogram_log_interval: int = TensorBoardConfig.HISTOGRAM_LOG_FREQ,
        image_log_interval: int = TensorBoardConfig.IMAGE_LOG_FREQ,
        env_config: Optional[EnvConfig] = None,
        rnn_config: Optional[RNNConfig] = None,
    ):
        self.aggregator = aggregator
        self.console_recorder = console_recorder
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.hparam_dict = hparam_dict if hparam_dict else {}
        self.histogram_log_interval = (
            max(1, histogram_log_interval) if histogram_log_interval > 0 else -1
        )
        self.image_log_interval = (
            max(1, image_log_interval) if image_log_interval > 0 else -1
        )
        self.last_histogram_log_step = -1
        self.last_image_log_step = -1
        self.env_config = env_config
        self.rnn_config = rnn_config
        self.vis_config = VisConfig()
        self.summary_avg_window = self.aggregator.summary_avg_window

        print(f"[TensorBoardStatsRecorder] Initialized. Logging to: {self.log_dir}")
        print(f"  Histogram Log Interval: {self.histogram_log_interval}")
        print(f"  Image Log Interval: {self.image_log_interval}")
        print(f"  Summary Avg Window: {self.summary_avg_window}")

        if model_for_graph and dummy_input_for_graph:
            self.record_graph(model_for_graph, dummy_input_for_graph)
        else:
            print(
                "[TensorBoardStatsRecorder] Model graph logging skipped (model or dummy input not provided)."
            )

        if self.hparam_dict:
            self._log_hparams_initial()

    def _log_hparams_initial(self):
        """Logs hyperparameters at the beginning of the run."""
        try:
            initial_metrics = {
                "hparam/initial_best_rl_score": -float("inf"),
                "hparam/initial_best_game_score": -float("inf"),
                "hparam/initial_best_loss": float("inf"),
            }
            filtered_hparams = {
                k: v
                for k, v in self.hparam_dict.items()
                if isinstance(v, (int, float, str, bool, torch.Tensor))
            }
            self.writer.add_hparams(filtered_hparams, initial_metrics, run_name=".")
            print("[TensorBoardStatsRecorder] Hyperparameters logged.")
        except Exception as e:
            print(f"Error logging initial hyperparameters: {e}")

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        lines_cleared: Optional[int] = None,
    ):
        # --- MODIFICATION: Call aggregator first to get update_info ---
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            lines_cleared,
        )
        # --- END MODIFICATION ---

        g_step = (
            global_step
            if global_step is not None
            else self.aggregator.current_global_step
        )

        # Log scalar values for episode stats
        self.writer.add_scalar("Episode/Score", episode_score, g_step)
        self.writer.add_scalar("Episode/Length", episode_length, g_step)
        if game_score is not None:
            self.writer.add_scalar("Episode/Game Score", game_score, g_step)
        if lines_cleared is not None:
            self.writer.add_scalar("Episode/Lines Cleared", lines_cleared, g_step)
        self.writer.add_scalar("Progress/Total Episodes", episode_num, g_step)

        # Log best scores if they were updated (using update_info from aggregator)
        if update_info.get("new_best_rl"):
            self.writer.add_scalar(
                "Best Performance/RL Score", self.aggregator.best_score, g_step
            )
            # Optional: Keep console print here or rely on console_recorder's print
            # print(f"--- 🏆 TB Logged New Best RL: {self.aggregator.best_score:.2f} at Step ~{g_step/1e6:.1f}M (Prev: {self.aggregator.previous_best_score:.2f}) ---")

        if update_info.get("new_best_game"):
            self.writer.add_scalar(
                "Best Performance/Game Score", self.aggregator.best_game_score, g_step
            )
            # Optional: Keep console print here or rely on console_recorder's print
            # print(f"--- 🎮 TB Logged New Best Game: {self.aggregator.best_game_score:.0f} at Step ~{g_step/1e6:.1f}M (Prev: {self.aggregator.previous_best_game_score:.0f}) ---")

        # --- MODIFICATION: Delegate to console recorder AFTER aggregator ---
        # Pass the original arguments to the console recorder
        self.console_recorder.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,  # Pass original global_step
            game_score,
            lines_cleared,
        )
        # --- END MODIFICATION ---

    def record_step(self, step_data: Dict[str, Any]):
        # --- MODIFICATION: Call aggregator first to get update_info ---
        update_info = self.aggregator.record_step(step_data)
        # --- END MODIFICATION ---

        g_step = step_data.get("global_step", self.aggregator.current_global_step)

        # Log scalar values from step_data
        if "policy_loss" in step_data:
            self.writer.add_scalar("Loss/Policy Loss", step_data["policy_loss"], g_step)
        if "value_loss" in step_data:
            self.writer.add_scalar("Loss/Value Loss", step_data["value_loss"], g_step)
        if "entropy" in step_data:
            self.writer.add_scalar("Loss/Entropy", step_data["entropy"], g_step)
        if "grad_norm" in step_data:
            self.writer.add_scalar("Debug/Grad Norm", step_data["grad_norm"], g_step)
        if "avg_max_q" in step_data:
            self.writer.add_scalar("Debug/Avg Max Q", step_data["avg_max_q"], g_step)
        if "beta" in step_data:
            self.writer.add_scalar("Debug/Beta", step_data["beta"], g_step)
        if "buffer_size" in step_data:
            self.writer.add_scalar(
                "Debug/Buffer Size", step_data["buffer_size"], g_step
            )
        if "lr" in step_data:
            self.writer.add_scalar("Train/Learning Rate", step_data["lr"], g_step)
        if "epsilon" in step_data:
            self.writer.add_scalar("Train/Epsilon", step_data["epsilon"], g_step)
        if "sps_collection" in step_data:
            self.writer.add_scalar(
                "Performance/SPS Collection", step_data["sps_collection"], g_step
            )
        if "update_time" in step_data:
            self.writer.add_scalar(
                "Performance/Update Time", step_data["update_time"], g_step
            )
        if "step_time" in step_data:
            self.writer.add_scalar(
                "Performance/Total Step Time", step_data["step_time"], g_step
            )
            if step_data["step_time"] > 1e-9:
                num_steps = step_data.get("num_steps_processed", 1)
                sps_total = num_steps / step_data["step_time"]
                self.writer.add_scalar("Performance/SPS Total", sps_total, g_step)

        # Log best loss if updated (using update_info from aggregator)
        if update_info.get("new_best_loss"):
            self.writer.add_scalar(
                "Best Performance/Loss", self.aggregator.best_value_loss, g_step
            )
            # Optional: Keep console print here or rely on console_recorder's print
            # print(f"---📉 TB Logged New Best Loss: {self.aggregator.best_value_loss:.4f} at Step ~{g_step/1e6:.1f}M (Prev: {self.aggregator.previous_best_value_loss:.4f}) ---")

        # --- MODIFICATION: Delegate to console recorder AFTER aggregator ---
        # Pass the original step_data dictionary
        self.console_recorder.record_step(step_data)
        # --- END MODIFICATION ---

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        if self.histogram_log_interval <= 0:
            return
        # Log only at specified intervals based on global_step
        # Use modulo for periodic logging relative to start
        if (
            global_step
            // (
                self.aggregator.num_envs
                * self.aggregator.ppo_config.NUM_STEPS_PER_ROLLOUT
            )
            % self.histogram_log_interval
            == 0
        ):
            # Check if we already logged for this update cycle step
            if global_step > self.last_histogram_log_step:
                try:
                    self.writer.add_histogram(tag, values, global_step)
                    self.last_histogram_log_step = global_step
                except Exception as e:
                    print(f"Error logging histogram '{tag}': {e}")

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        if self.image_log_interval <= 0:
            return
        # Log only at specified intervals based on global_step
        if (
            global_step
            // (
                self.aggregator.num_envs
                * self.aggregator.ppo_config.NUM_STEPS_PER_ROLLOUT
            )
            % self.image_log_interval
            == 0
        ):
            if global_step > self.last_image_log_step:
                try:
                    # Ensure image has channel-first format (C, H, W) or (N, C, H, W)
                    if isinstance(image, np.ndarray):
                        if image.ndim == 3 and image.shape[-1] in [
                            1,
                            3,
                            4,
                        ]:  # HWC -> CHW
                            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
                        elif image.ndim == 2:  # HW -> CHW (add channel dim)
                            image_tensor = torch.from_numpy(image).unsqueeze(0)
                        else:  # Assume CHW or NCHW
                            image_tensor = torch.from_numpy(image)
                    elif isinstance(image, torch.Tensor):
                        if image.ndim == 3 and image.shape[0] not in [
                            1,
                            3,
                            4,
                        ]:  # HWC? -> CHW
                            if image.shape[-1] in [1, 3, 4]:
                                image_tensor = image.permute(2, 0, 1)
                            else:  # Assume CHW
                                image_tensor = image
                        elif image.ndim == 2:  # HW -> CHW
                            image_tensor = image.unsqueeze(0)
                        else:  # Assume CHW or NCHW
                            image_tensor = image
                    else:
                        print(
                            f"Warning: Unsupported image type for tag '{tag}': {type(image)}"
                        )
                        return

                    self.writer.add_image(
                        tag, image_tensor, global_step, dataformats="CHW"
                    )
                    self.last_image_log_step = global_step
                except Exception as e:
                    print(f"Error logging image '{tag}': {e}")

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Records final hyperparameters and metrics."""
        try:
            filtered_hparams = {
                k: v
                for k, v in hparam_dict.items()
                if isinstance(v, (int, float, str, bool, torch.Tensor))
            }
            filtered_metrics = {
                k: v for k, v in metric_dict.items() if isinstance(v, (int, float))
            }
            self.writer.add_hparams(filtered_hparams, filtered_metrics, run_name=".")
            print("[TensorBoardStatsRecorder] Final hparams and metrics logged.")
        except Exception as e:
            print(f"Error logging final hyperparameters/metrics: {e}")

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[Any] = None
    ):
        """Records the model graph."""
        if input_to_model is None:
            print("Warning: Cannot record graph without dummy input.")
            return
        try:
            model.cpu()
            if isinstance(input_to_model, tuple):
                dummy_input_cpu = tuple(
                    t.cpu() for t in input_to_model if isinstance(t, torch.Tensor)
                )
            elif isinstance(input_to_model, torch.Tensor):
                dummy_input_cpu = input_to_model.cpu()
            else:
                dummy_input_cpu = input_to_model

            self.writer.add_graph(model, dummy_input_cpu, verbose=False)
            print("[TensorBoardStatsRecorder] Model graph logged.")
            # Move model back to original device
            if hasattr(model, "device"):  # Check if model has device attr
                model.to(model.device)
            elif self.env_config:  # Fallback to general device
                model.to(DEVICE)

        except Exception as e:
            print(f"Error logging model graph: {e}. Graph logging can be tricky.")

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        # Delegate to console logger
        self.console_recorder.log_summary(global_step)

    def close(self):
        """Closes the TensorBoard writer and logs final hparams."""
        print("[TensorBoardStatsRecorder] Closing writer...")
        try:
            final_summary = self.get_summary(self.aggregator.current_global_step)
            final_metrics = {
                "hparam/final_best_rl_score": final_summary.get(
                    "best_score", -float("inf")
                ),
                "hparam/final_best_game_score": final_summary.get(
                    "best_game_score", -float("inf")
                ),
                "hparam/final_best_loss": final_summary.get("best_loss", float("inf")),
                "hparam/final_total_episodes": final_summary.get("total_episodes", 0),
            }
            # Ensure hparam_dict exists before logging
            if self.hparam_dict:
                self.record_hparams(self.hparam_dict, final_metrics)
            else:
                print(
                    "[TensorBoardStatsRecorder] Skipping final hparam logging (hparam_dict not set)."
                )

            self.writer.flush()
            self.writer.close()
            print("[TensorBoardStatsRecorder] Writer closed.")
        except Exception as e:
            print(f"Error during TensorBoard writer close: {e}")
        # Close console recorder as well
        self.console_recorder.close()


File: stats\__init__.py
# File: stats/__init__.py
from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from .simple_stats_recorder import SimpleStatsRecorder
from .tensorboard_logger import TensorBoardStatsRecorder

__all__ = [
    "StatsRecorderBase",
    "StatsAggregator",
    "SimpleStatsRecorder",
    "TensorBoardStatsRecorder",
]


File: training\checkpoint_manager.py
# File: training/checkpoint_manager.py
import os
import torch
import traceback
from typing import Optional, Dict, Any, Tuple

from agent.ppo_agent import PPOAgent


class CheckpointManager:
    """Handles loading and saving of PPO agent states."""

    def __init__(
        self,
        agent: PPOAgent,
        model_save_path: str,
        load_checkpoint_path: Optional[str],
        device: torch.device,
    ):
        self.agent = agent
        self.model_save_path = model_save_path
        self.device = device

        self.global_step = 0
        self.episode_count = 0

        if load_checkpoint_path:
            self.load_agent_checkpoint(load_checkpoint_path)
        else:
            print(
                "[CheckpointManager-PPO] No agent checkpoint specified, starting fresh."
            )

    def load_agent_checkpoint(self, path_to_load: str):
        if not os.path.isfile(path_to_load):
            print(
                f"[CheckpointManager-PPO] LOAD WARNING: Agent ckpt not found: {path_to_load}"
            )
            return
        print(f"[CheckpointManager-PPO] Loading agent checkpoint from: {path_to_load}")
        try:
            checkpoint = torch.load(path_to_load, map_location=self.device)
            self.agent.load_state_dict(checkpoint)

            self.global_step = checkpoint.get("global_step", 0)
            self.episode_count = checkpoint.get("episode_count", 0)
            print(
                f"  -> Resuming from Step: {self.global_step}, Ep: {self.episode_count}"
            )
        except KeyError as e:
            print(
                f"  -> ERROR loading agent checkpoint: Missing key '{e}'. Check compatibility."
            )
            self.global_step = 0
            self.episode_count = 0
        except Exception as e:
            print(f"  -> ERROR loading agent checkpoint ('{e}'). Check compatibility.")
            traceback.print_exc()
            self.global_step = 0
            self.episode_count = 0

    def save_checkpoint(
        self, global_step: int, episode_count: int, is_final: bool = False
    ):
        prefix = "FINAL" if is_final else f"step_{global_step}"
        save_dir = os.path.dirname(self.model_save_path)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[CheckpointManager-PPO] Saving agent checkpoint ({prefix})...")
        try:
            agent_save_data = self.agent.get_state_dict()
            agent_save_data["global_step"] = global_step
            agent_save_data["episode_count"] = episode_count
            torch.save(agent_save_data, self.model_save_path)
            print(
                f"  -> Agent checkpoint saved: {os.path.basename(self.model_save_path)}"
            )
        except Exception as e:
            print(f"  -> ERROR saving agent checkpoint: {e}")
            traceback.print_exc()

    def get_initial_state(self) -> Tuple[int, int]:
        return self.global_step, self.episode_count


File: training\rollout_collector.py
# File: training/rollout_collector.py
import time
import torch
import numpy as np
import random
import traceback
from typing import List, Dict, Any, Tuple, Optional

from config import (
    EnvConfig,
    RewardConfig,
    TensorBoardConfig,
    PPOConfig,
    RNNConfig,
    DEVICE,
)
from environment.game_state import GameState, StateType
from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from utils.types import ActionType
from .rollout_storage import RolloutStorage


class RolloutCollector:
    """
    Handles interaction with parallel environments to collect rollouts for PPO.
    Includes staggered start of interaction to desynchronize environments.
    """

    def __init__(
        self,
        envs: List[GameState],
        agent: PPOAgent,
        stats_recorder: StatsRecorderBase,
        env_config: EnvConfig,
        ppo_config: PPOConfig,
        rnn_config: RNNConfig,
        reward_config: RewardConfig,
        tb_config: TensorBoardConfig,
    ):
        self.envs = envs
        self.agent = agent
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.reward_config = reward_config
        self.tb_config = tb_config
        self.device = DEVICE

        self.rollout_storage = RolloutStorage(
            ppo_config.NUM_STEPS_PER_ROLLOUT,
            self.num_envs,
            self.env_config,
            self.rnn_config,
            self.device,
        )

        # CPU Buffers for current step's observations and dones
        self.current_obs_grid_cpu = np.zeros(
            (self.num_envs, *self.env_config.GRID_STATE_SHAPE), dtype=np.float32
        )
        self.current_obs_shapes_cpu = np.zeros(
            (self.num_envs, self.env_config.SHAPE_STATE_DIM), dtype=np.float32
        )
        self.current_obs_availability_cpu = np.zeros(
            (self.num_envs, self.env_config.SHAPE_AVAILABILITY_DIM), dtype=np.float32
        )
        # --- UPDATED: Use correct dimension for explicit features ---
        self.current_obs_explicit_features_cpu = np.zeros(
            (self.num_envs, self.env_config.EXPLICIT_FEATURES_DIM), dtype=np.float32
        )
        # --- END UPDATED ---
        self.current_dones_cpu = np.zeros(self.num_envs, dtype=bool)

        # Episode trackers
        self.current_episode_scores = np.zeros(self.num_envs, dtype=np.float32)
        self.current_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_game_scores = np.zeros(self.num_envs, dtype=np.int32)
        self.current_episode_lines_cleared = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_count = 0

        # RNN state
        self.current_lstm_state_device: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
            None
        )
        if self.rnn_config.USE_RNN:
            self.current_lstm_state_device = self.agent.get_initial_hidden_state(
                self.num_envs
            )

        # --- NEW: Staggered Start ---
        # Assign a start step delay for each environment (simple sequential stagger)
        # Ensures interaction starts spread out over the first rollout steps.
        self.env_start_step_delay = (
            np.arange(self.num_envs) % self.ppo_config.NUM_STEPS_PER_ROLLOUT
        )
        print(
            f"[RolloutCollector] Interaction start steps staggered up to step {np.max(self.env_start_step_delay)}."
        )
        # --- END NEW ---

        # Reset environments and populate initial observations
        self._reset_all_envs()
        self._copy_initial_observations_to_storage()

        print(f"[RolloutCollector] Initialized for {self.num_envs} environments.")

    def _reset_env(self, env_index: int) -> StateType:
        """Resets a single environment and returns its initial state dict."""
        try:
            state_dict = self.envs[env_index].reset()
            self.current_episode_scores[env_index] = 0.0
            self.current_episode_lengths[env_index] = 0
            self.current_episode_game_scores[env_index] = 0
            self.current_episode_lines_cleared[env_index] = 0
            return state_dict
        except KeyError as e:
            print(
                f"FATAL ERROR: Env {env_index} reset missing key '{e}'. Check GameState.reset()"
            )
            raise e
        except Exception as e:
            print(f"ERROR resetting env {env_index}: {e}")
            dummy_state: StateType = {
                "grid": np.zeros(self.env_config.GRID_STATE_SHAPE, dtype=np.float32),
                "shapes": np.zeros(
                    (
                        self.env_config.NUM_SHAPE_SLOTS,
                        self.env_config.SHAPE_FEATURES_PER_SHAPE,
                    ),
                    dtype=np.float32,
                ),
                "shape_availability": np.zeros(
                    self.env_config.SHAPE_AVAILABILITY_DIM, dtype=np.float32
                ),
                # --- UPDATED: Use correct dimension for dummy explicit features ---
                "explicit_features": np.zeros(
                    self.env_config.EXPLICIT_FEATURES_DIM, dtype=np.float32
                ),
                # --- END UPDATED ---
            }
            self.current_dones_cpu[env_index] = True
            return dummy_state

    def _update_obs_from_state_dict(self, env_index: int, state_dict: StateType):
        """Updates the CPU observation buffers for a given environment index."""
        self.current_obs_grid_cpu[env_index] = state_dict["grid"]
        self.current_obs_shapes_cpu[env_index] = state_dict["shapes"].reshape(-1)[
            : self.env_config.SHAPE_STATE_DIM
        ]
        self.current_obs_availability_cpu[env_index] = state_dict["shape_availability"]
        # --- UPDATED: Copy explicit features ---
        self.current_obs_explicit_features_cpu[env_index] = state_dict[
            "explicit_features"
        ]
        # --- END UPDATED ---

    def _reset_all_envs(self):
        """Resets all environments and updates initial observations."""
        for i in range(self.num_envs):
            initial_state = self._reset_env(i)
            self._update_obs_from_state_dict(i, initial_state)
            self.current_dones_cpu[i] = False
        if self.rnn_config.USE_RNN:
            self.current_lstm_state_device = self.agent.get_initial_hidden_state(
                self.num_envs
            )

    def _copy_initial_observations_to_storage(self):
        """Copies the initial observations from CPU buffers to the RolloutStorage."""
        initial_obs_grid_t = torch.from_numpy(self.current_obs_grid_cpu).to(
            self.rollout_storage.device
        )
        initial_obs_shapes_t = torch.from_numpy(self.current_obs_shapes_cpu).to(
            self.rollout_storage.device
        )
        initial_obs_availability_t = torch.from_numpy(
            self.current_obs_availability_cpu
        ).to(self.rollout_storage.device)
        # --- UPDATED: Copy explicit features ---
        initial_obs_explicit_features_t = torch.from_numpy(
            self.current_obs_explicit_features_cpu
        ).to(self.rollout_storage.device)
        # --- END UPDATED ---
        initial_dones_t = (
            torch.from_numpy(self.current_dones_cpu)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )

        self.rollout_storage.obs_grid[0].copy_(initial_obs_grid_t)
        self.rollout_storage.obs_shapes[0].copy_(initial_obs_shapes_t)
        self.rollout_storage.obs_availability[0].copy_(initial_obs_availability_t)
        # --- UPDATED: Copy explicit features to storage ---
        self.rollout_storage.obs_explicit_features[0].copy_(
            initial_obs_explicit_features_t
        )
        # --- END UPDATED ---
        self.rollout_storage.dones[0].copy_(initial_dones_t)

        if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
            if (
                self.rollout_storage.hidden_states is not None
                and self.rollout_storage.cell_states is not None
            ):
                self.rollout_storage.hidden_states[0].copy_(
                    self.current_lstm_state_device[0]
                )
                self.rollout_storage.cell_states[0].copy_(
                    self.current_lstm_state_device[1]
                )

    def _record_episode_stats(
        self, env_index: int, final_reward_adjustment: float, current_global_step: int
    ):
        """Helper function to record stats for a finished episode."""
        # Only record stats if the environment has actually started interacting
        if current_global_step >= self.env_start_step_delay[env_index]:
            self.episode_count += 1
            final_episode_score = (
                self.current_episode_scores[env_index] + final_reward_adjustment
            )
            final_episode_length = self.current_episode_lengths[env_index]
            final_game_score = self.current_episode_game_scores[env_index]
            final_lines_cleared = self.current_episode_lines_cleared[env_index]

            approx_global_step_for_log = current_global_step + env_index + 1

            self.stats_recorder.record_episode(
                episode_score=final_episode_score,
                episode_length=final_episode_length,
                episode_num=self.episode_count,
                global_step=approx_global_step_for_log,
                game_score=final_game_score,
                lines_cleared=final_lines_cleared,
            )

    def _reset_rnn_state_for_env(self, env_index: int):
        """Resets the RNN hidden state for a specific environment index."""
        if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
            reset_h, reset_c = self.agent.get_initial_hidden_state(1)
            if reset_h is not None and reset_c is not None:
                reset_h = reset_h.to(self.current_lstm_state_device[0].device)
                reset_c = reset_c.to(self.current_lstm_state_device[1].device)
                self.current_lstm_state_device[0][
                    :, env_index : env_index + 1, :
                ] = reset_h
                self.current_lstm_state_device[1][
                    :, env_index : env_index + 1, :
                ] = reset_c

    def collect_one_step(self, current_global_step: int) -> int:
        """Collects one step of experience from all environments using batching."""
        step_start_time = time.time()
        current_rollout_step = (
            self.rollout_storage.step
        )  # Get current step within the rollout

        # --- 1. Identify active, frozen, waiting, and truly done environments ---
        active_env_indices: List[int] = []
        frozen_env_indices: List[int] = []
        waiting_env_indices: List[int] = (
            []
        )  # NEW: Envs that haven't reached start delay
        envs_done_pre_action: List[int] = (
            []
        )  # Truly done (no moves, not frozen/waiting)
        valid_actions_list: List[Optional[List[int]]] = [None] * self.num_envs

        for i in range(self.num_envs):
            self.envs[i]._update_timers()  # Update timers first

            if self.current_dones_cpu[i]:
                # Already done from previous step, will be reset later
                continue

            # --- NEW: Check if waiting to start ---
            if current_rollout_step < self.env_start_step_delay[i]:
                waiting_env_indices.append(i)
                continue  # Skip checks below if waiting
            # --- END NEW ---

            if self.envs[i].is_frozen():
                frozen_env_indices.append(i)
                continue  # Skip action selection

            valid_actions = self.envs[i].valid_actions()
            if not valid_actions:
                envs_done_pre_action.append(i)
            else:
                valid_actions_list[i] = valid_actions
                active_env_indices.append(i)

        # --- 2. Select actions ONLY for active environments ---
        actions_np = np.zeros(self.num_envs, dtype=np.int64)
        log_probs_np = np.zeros(self.num_envs, dtype=np.float32)
        values_np = np.zeros(self.num_envs, dtype=np.float32)
        next_lstm_state_device = self.current_lstm_state_device

        if active_env_indices:  # Only run agent if there are active envs
            active_indices_tensor = torch.tensor(active_env_indices, dtype=torch.long)

            batch_obs_grid_cpu = self.current_obs_grid_cpu[active_env_indices]
            batch_obs_shapes_cpu = self.current_obs_shapes_cpu[active_env_indices]
            batch_obs_availability_cpu = self.current_obs_availability_cpu[
                active_env_indices
            ]
            # --- UPDATED: Get explicit features for active envs ---
            batch_obs_explicit_features_cpu = self.current_obs_explicit_features_cpu[
                active_env_indices
            ]
            # --- END UPDATED ---
            batch_valid_actions = [valid_actions_list[i] for i in active_env_indices]

            batch_hidden_state_device = None
            if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
                h_n, c_n = self.current_lstm_state_device
                batch_hidden_state_device = (
                    h_n[:, active_indices_tensor, :].contiguous(),
                    c_n[:, active_indices_tensor, :].contiguous(),
                )

            batch_obs_grid_t = torch.from_numpy(batch_obs_grid_cpu).to(
                self.agent.device
            )
            batch_obs_shapes_t = torch.from_numpy(batch_obs_shapes_cpu).to(
                self.agent.device
            )
            batch_obs_availability_t = torch.from_numpy(batch_obs_availability_cpu).to(
                self.agent.device
            )
            # --- UPDATED: Convert explicit features to tensor ---
            batch_obs_explicit_features_t = torch.from_numpy(
                batch_obs_explicit_features_cpu
            ).to(self.agent.device)
            # --- END UPDATED ---

            with torch.no_grad():
                (
                    batch_actions_t,
                    batch_log_probs_t,
                    batch_values_t,
                    batch_next_lstm_state_device,
                ) = self.agent.select_action_batch(
                    batch_obs_grid_t,
                    batch_obs_shapes_t,
                    batch_obs_availability_t,
                    # --- UPDATED: Pass explicit features tensor ---
                    batch_obs_explicit_features_t,
                    # --- END UPDATED ---
                    batch_hidden_state_device,
                    batch_valid_actions,
                )

            actions_np[active_env_indices] = batch_actions_t.cpu().numpy()
            log_probs_np[active_env_indices] = batch_log_probs_t.cpu().numpy()
            values_np[active_env_indices] = batch_values_t.cpu().numpy()

            if self.rnn_config.USE_RNN and batch_next_lstm_state_device is not None:
                next_h = self.current_lstm_state_device[0].clone()
                next_c = self.current_lstm_state_device[1].clone()
                next_h[:, active_indices_tensor, :] = batch_next_lstm_state_device[0]
                next_c[:, active_indices_tensor, :] = batch_next_lstm_state_device[1]
                next_lstm_state_device = (next_h, next_c)

        # --- 3. Step environments, handle resets, and update observations ---
        next_obs_grid_cpu = np.copy(self.current_obs_grid_cpu)
        next_obs_shapes_cpu = np.copy(self.current_obs_shapes_cpu)
        next_obs_availability_cpu = np.copy(self.current_obs_availability_cpu)
        # --- UPDATED: Copy explicit features ---
        next_obs_explicit_features_cpu = np.copy(self.current_obs_explicit_features_cpu)
        # --- END UPDATED ---
        step_rewards_np = np.zeros(self.num_envs, dtype=np.float32)
        step_dones_np = np.copy(self.current_dones_cpu)

        for i in range(self.num_envs):
            is_done_this_step = False
            final_reward_adj = 0.0

            if self.current_dones_cpu[i]:
                # --- Was already done, reset ---
                new_state_dict = self._reset_env(i)
                self._update_obs_from_state_dict(i, new_state_dict)
                self._reset_rnn_state_for_env(i)
                step_dones_np[i] = False
                is_done_this_step = True

            # --- NEW: Handle waiting environments ---
            elif i in waiting_env_indices:
                step_rewards_np[i] = 0.0  # No reward while waiting
                step_dones_np[i] = False  # Not done
                is_done_this_step = False
                # Observations and LSTM state remain the same (initial state)
            # --- END NEW ---

            elif i in frozen_env_indices:
                # --- Was frozen, do not step ---
                step_rewards_np[i] = self.reward_config.REWARD_ALIVE_STEP
                step_dones_np[i] = False
                is_done_this_step = False
                # Observations and LSTM state remain the same

            elif i in envs_done_pre_action:
                # --- Became done (no valid moves), reset ---
                final_reward_adj = self.reward_config.PENALTY_GAME_OVER
                log_probs_np[i] = -1e9
                values_np[i] = 0.0
                self.current_episode_lengths[i] += 1

                self._record_episode_stats(i, final_reward_adj, current_global_step)
                new_state_dict = self._reset_env(i)
                self._update_obs_from_state_dict(i, new_state_dict)
                self._reset_rnn_state_for_env(i)
                step_dones_np[i] = True
                is_done_this_step = True

            else:
                # --- Environment is active, perform step ---
                action_to_take = actions_np[i]
                try:
                    reward, done = self.envs[i].step(action_to_take)
                    step_rewards_np[i] = reward
                    step_dones_np[i] = done

                    # Update episode trackers only if the env has started interacting
                    if current_rollout_step >= self.env_start_step_delay[i]:
                        self.current_episode_scores[i] += reward
                        self.current_episode_lengths[i] += 1
                        self.current_episode_game_scores[i] = self.envs[i].game_score
                        self.current_episode_lines_cleared[i] = self.envs[
                            i
                        ].lines_cleared_this_episode

                    if done:
                        self._record_episode_stats(i, 0.0, current_global_step)
                        new_state_dict = self._reset_env(i)
                        self._update_obs_from_state_dict(i, new_state_dict)
                        self._reset_rnn_state_for_env(i)
                        is_done_this_step = True
                    else:
                        next_state_dict = self.envs[i].get_state()
                        self._update_obs_from_state_dict(i, next_state_dict)
                        is_done_this_step = False

                except Exception as e:
                    print(f"ERROR: Env {i} step failed (Action: {action_to_take}): {e}")
                    traceback.print_exc()
                    step_rewards_np[i] = self.reward_config.PENALTY_GAME_OVER
                    step_dones_np[i] = True
                    # Update length even on error if it was interacting
                    if current_rollout_step >= self.env_start_step_delay[i]:
                        self.current_episode_lengths[i] += 1

                    self._record_episode_stats(i, 0.0, current_global_step)
                    new_state_dict = self._reset_env(i)
                    self._update_obs_from_state_dict(i, new_state_dict)
                    self._reset_rnn_state_for_env(i)
                    is_done_this_step = True

            # Update the observation buffers for the *next* step (S_{t+1})
            next_obs_grid_cpu[i] = self.current_obs_grid_cpu[i]
            next_obs_shapes_cpu[i] = self.current_obs_shapes_cpu[i]
            next_obs_availability_cpu[i] = self.current_obs_availability_cpu[i]
            # --- UPDATED: Update next explicit features ---
            next_obs_explicit_features_cpu[i] = self.current_obs_explicit_features_cpu[
                i
            ]
            # --- END UPDATED ---

        # --- 4. Store results in RolloutStorage ---
        obs_grid_t = torch.from_numpy(self.current_obs_grid_cpu).to(
            self.rollout_storage.device
        )
        obs_shapes_t = torch.from_numpy(self.current_obs_shapes_cpu).to(
            self.rollout_storage.device
        )
        obs_availability_t = torch.from_numpy(self.current_obs_availability_cpu).to(
            self.rollout_storage.device
        )
        # --- UPDATED: Convert explicit features to tensor ---
        obs_explicit_features_t = torch.from_numpy(
            self.current_obs_explicit_features_cpu
        ).to(self.rollout_storage.device)
        # --- END UPDATED ---
        actions_t = (
            torch.from_numpy(actions_np)
            .long()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )
        log_probs_t = (
            torch.from_numpy(log_probs_np)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )
        values_t = (
            torch.from_numpy(values_np)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )
        rewards_t = (
            torch.from_numpy(step_rewards_np)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )
        dones_t_for_storage = (
            torch.from_numpy(step_dones_np)
            .float()
            .unsqueeze(1)
            .to(self.rollout_storage.device)
        )

        lstm_state_to_store = None
        if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
            lstm_state_to_store = (
                self.current_lstm_state_device[0].to(self.rollout_storage.device),
                self.current_lstm_state_device[1].to(self.rollout_storage.device),
            )

        self.rollout_storage.insert(
            obs_grid_t,
            obs_shapes_t,
            obs_availability_t,
            # --- UPDATED: Pass explicit features to storage ---
            obs_explicit_features_t,
            # --- END UPDATED ---
            actions_t,
            log_probs_t,
            values_t,
            rewards_t,
            dones_t_for_storage,
            lstm_state_to_store,
        )

        # --- 5. Update collector's current state for the *next* iteration ---
        self.current_obs_grid_cpu = next_obs_grid_cpu
        self.current_obs_shapes_cpu = next_obs_shapes_cpu
        self.current_obs_availability_cpu = next_obs_availability_cpu
        # --- UPDATED: Update current explicit features ---
        self.current_obs_explicit_features_cpu = next_obs_explicit_features_cpu
        # --- END UPDATED ---
        self.current_dones_cpu = step_dones_np
        self.current_lstm_state_device = next_lstm_state_device

        # --- 6. Record performance ---
        collection_time = time.time() - step_start_time
        sps = self.num_envs / max(1e-9, collection_time)
        self.stats_recorder.record_step(
            {"sps_collection": sps, "rollout_collection_time": collection_time}
        )

        return self.num_envs

    def compute_advantages_for_storage(self):
        """Computes GAE advantages using the data in RolloutStorage."""
        with torch.no_grad():
            final_obs_grid_t = torch.from_numpy(self.current_obs_grid_cpu).to(
                self.agent.device
            )
            final_obs_shapes_t = torch.from_numpy(self.current_obs_shapes_cpu).to(
                self.agent.device
            )
            final_obs_availability_t = torch.from_numpy(
                self.current_obs_availability_cpu
            ).to(self.agent.device)
            # --- UPDATED: Get final explicit features ---
            final_obs_explicit_features_t = torch.from_numpy(
                self.current_obs_explicit_features_cpu
            ).to(self.agent.device)
            # --- END UPDATED ---

            final_lstm_state = None
            if self.rnn_config.USE_RNN and self.current_lstm_state_device is not None:
                final_lstm_state = (
                    self.current_lstm_state_device[0].to(self.agent.device),
                    self.current_lstm_state_device[1].to(self.agent.device),
                )

            if self.rnn_config.USE_RNN:
                final_obs_grid_t = final_obs_grid_t.unsqueeze(1)
                final_obs_shapes_t = final_obs_shapes_t.unsqueeze(1)
                final_obs_availability_t = final_obs_availability_t.unsqueeze(1)
                # --- UPDATED: Add sequence dim if RNN ---
                final_obs_explicit_features_t = final_obs_explicit_features_t.unsqueeze(
                    1
                )
                # --- END UPDATED ---

            _, next_value, _ = self.agent.network(
                final_obs_grid_t,
                final_obs_shapes_t,
                final_obs_availability_t,
                # --- UPDATED: Pass final explicit features ---
                final_obs_explicit_features_t,
                # --- END UPDATED ---
                final_lstm_state,
            )

            if self.rnn_config.USE_RNN:
                next_value = next_value.squeeze(1)

            if next_value.ndim == 1:
                next_value = next_value.unsqueeze(-1)

            final_dones = (
                torch.from_numpy(self.current_dones_cpu)
                .float()
                .unsqueeze(1)
                .to(self.device)
            )

        self.rollout_storage.compute_returns_and_advantages(
            next_value, final_dones, self.ppo_config.GAMMA, self.ppo_config.GAE_LAMBDA
        )

    def get_episode_count(self) -> int:
        return self.episode_count


File: training\rollout_storage.py
# File: training/rollout_storage.py
import torch
from typing import Optional, Tuple, Dict, List, Any
import numpy as np

from config import EnvConfig, PPOConfig, RNNConfig, DEVICE


class RolloutStorage:
    """
    Stores rollout data collected from parallel environments for PPO.
    Now includes storage for shape availability and explicit features.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        env_config: EnvConfig,
        rnn_config: RNNConfig,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.env_config = env_config
        self.rnn_config = rnn_config
        self.device = device

        grid_c, grid_h, grid_w = self.env_config.GRID_STATE_SHAPE
        shape_feat_dim = self.env_config.SHAPE_STATE_DIM
        shape_availability_dim = self.env_config.SHAPE_AVAILABILITY_DIM
        # --- UPDATED: Get the potentially larger explicit features dim ---
        explicit_features_dim = self.env_config.EXPLICIT_FEATURES_DIM
        # --- END UPDATED ---

        # --- Standard PPO Data ---
        self.obs_grid = torch.zeros(
            num_steps + 1, num_envs, grid_c, grid_h, grid_w, device=self.device
        )
        self.obs_shapes = torch.zeros(
            num_steps + 1, num_envs, shape_feat_dim, device=self.device
        )
        self.obs_availability = torch.zeros(
            num_steps + 1, num_envs, shape_availability_dim, device=self.device
        )
        # --- UPDATED: Storage for explicit features with correct dimension ---
        self.obs_explicit_features = torch.zeros(
            num_steps + 1, num_envs, explicit_features_dim, device=self.device
        )
        # --- END UPDATED ---
        self.actions = torch.zeros(num_steps, num_envs, 1, device=self.device).long()
        self.log_probs = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_steps + 1, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_steps + 1, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_steps, num_envs, 1, device=self.device)

        # --- RNN Specific Data ---
        self.hidden_states = None
        self.cell_states = None
        if self.rnn_config.USE_RNN:
            lstm_hidden_size = self.rnn_config.LSTM_HIDDEN_SIZE
            num_layers = self.rnn_config.LSTM_NUM_LAYERS
            self.hidden_states = torch.zeros(
                num_steps + 1,
                num_layers,
                num_envs,
                lstm_hidden_size,
                device=self.device,
            )
            self.cell_states = torch.zeros(
                num_steps + 1,
                num_layers,
                num_envs,
                lstm_hidden_size,
                device=self.device,
            )

        self.step = 0

    def to(self, device: torch.device):
        """Move storage tensors to the specified device."""
        if self.device == device:
            return
        self.obs_grid = self.obs_grid.to(device)
        self.obs_shapes = self.obs_shapes.to(device)
        self.obs_availability = self.obs_availability.to(device)
        # --- UPDATED: Move explicit features ---
        self.obs_explicit_features = self.obs_explicit_features.to(device)
        # --- END UPDATED ---
        self.rewards = self.rewards.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.log_probs = self.log_probs.to(device)
        self.actions = self.actions.to(device)
        self.dones = self.dones.to(device)
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.to(device)
        if self.cell_states is not None:
            self.cell_states = self.cell_states.to(device)
        self.device = device
        print(f"[RolloutStorage] Moved tensors to {device}")

    def insert(
        self,
        obs_grid: torch.Tensor,
        obs_shapes: torch.Tensor,
        obs_availability: torch.Tensor,
        # --- UPDATED: Add explicit features ---
        obs_explicit_features: torch.Tensor,
        # --- END UPDATED ---
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Insert one step of data. Assumes input tensors are already on self.device."""
        if self.step >= self.num_steps:
            raise IndexError(
                f"RolloutStorage step index {self.step} out of bounds (max {self.num_steps-1})"
            )

        current_step_index = self.step

        self.obs_grid[current_step_index].copy_(obs_grid)
        self.obs_shapes[current_step_index].copy_(obs_shapes)
        self.obs_availability[current_step_index].copy_(obs_availability)
        # --- UPDATED: Copy explicit features ---
        self.obs_explicit_features[current_step_index].copy_(obs_explicit_features)
        # --- END UPDATED ---
        self.actions[current_step_index].copy_(action)
        self.log_probs[current_step_index].copy_(log_prob)
        self.values[current_step_index].copy_(value)
        self.rewards[current_step_index].copy_(reward)
        self.dones[current_step_index].copy_(done)

        if self.rnn_config.USE_RNN and lstm_state is not None:
            if self.hidden_states is not None and self.cell_states is not None:
                self.hidden_states[current_step_index].copy_(lstm_state[0])
                self.cell_states[current_step_index].copy_(lstm_state[1])
            else:
                print(
                    "Warning: LSTM state provided but storage tensors not initialized."
                )

        # Store the *next* observation/done state at step+1 index
        # These will be overwritten by the next insert or used in after_update/compute_returns
        next_step_index = current_step_index + 1
        if next_step_index <= self.num_steps:  # Prevent index out of bounds
            self.obs_grid[next_step_index].copy_(
                obs_grid
            )  # These are placeholders for the *next* actual obs
            self.obs_shapes[next_step_index].copy_(obs_shapes)
            self.obs_availability[next_step_index].copy_(obs_availability)
            # --- UPDATED: Copy next explicit features ---
            self.obs_explicit_features[next_step_index].copy_(obs_explicit_features)
            # --- END UPDATED ---
            self.dones[next_step_index].copy_(
                done
            )  # Store the done state corresponding to the obs at current_step_index
            if self.rnn_config.USE_RNN and lstm_state is not None:
                # The LSTM state stored at step+1 should correspond to the state *after* processing obs at step
                if self.hidden_states is not None:
                    self.hidden_states[next_step_index].copy_(
                        lstm_state[0]
                    )  # Store the *next* hidden state
                if self.cell_states is not None:
                    self.cell_states[next_step_index].copy_(
                        lstm_state[1]
                    )  # Store the *next* cell state

        self.step += 1  # Increment step *after* storing

    def after_update(self):
        """Reset storage after PPO update, keeping the last observation and state."""
        last_step_index = self.num_steps
        # Copy the actual *last* observation (which was stored at index num_steps) to index 0
        self.obs_grid[0].copy_(self.obs_grid[last_step_index])
        self.obs_shapes[0].copy_(self.obs_shapes[last_step_index])
        self.obs_availability[0].copy_(self.obs_availability[last_step_index])
        # --- UPDATED: Copy last explicit features ---
        self.obs_explicit_features[0].copy_(self.obs_explicit_features[last_step_index])
        # --- END UPDATED ---
        self.dones[0].copy_(self.dones[last_step_index])

        if self.rnn_config.USE_RNN:
            if self.hidden_states is not None:
                self.hidden_states[0].copy_(self.hidden_states[last_step_index])
            if self.cell_states is not None:
                self.cell_states[0].copy_(self.cell_states[last_step_index])
        self.step = 0

    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,
        final_dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """Computes returns and GAE advantages. Assumes inputs are on self.device."""
        if self.step != self.num_steps:
            print(
                f"Warning: Computing returns before storage is full (step={self.step}, num_steps={self.num_steps})"
            )

        effective_num_steps = self.step  # Use actual steps filled if not full
        last_step_index = effective_num_steps

        # The value of the state *after* the last step in the rollout
        self.values[last_step_index] = next_value.to(self.device)
        # The done state *after* the last step in the rollout
        self.dones[last_step_index] = final_dones.to(self.device)

        gae = 0.0
        for step in reversed(range(effective_num_steps)):
            # delta = R_t + gamma * V(s_{t+1}) * (1-done_{t+1}) - V(s_t)
            # Note: self.dones[step + 1] is the done flag *after* taking action at step `step`
            delta = (
                self.rewards[step]
                + gamma * self.values[step + 1] * (1.0 - self.dones[step + 1])
                - self.values[step]
            )
            # gae_t = delta_t + gamma * lambda * gae_{t+1} * (1-done_{t+1})
            gae = delta + gamma * gae_lambda * gae * (1.0 - self.dones[step + 1])
            # return_t = gae_t + V(s_t)
            self.returns[step] = gae + self.values[step]

    def get_data_for_update(self) -> Dict[str, Any]:
        """
        Returns collected data prepared for PPO update iterations.
        Data is returned as flattened tensors [N = T*B, ...].
        """
        effective_num_steps = self.step
        if effective_num_steps == 0:
            return {}

        # Advantages are calculated based on returns and values up to the effective step count
        advantages = (
            self.returns[:effective_num_steps] - self.values[:effective_num_steps]
        )

        num_samples = effective_num_steps * self.num_envs

        def _flatten(tensor: torch.Tensor, steps: int) -> torch.Tensor:
            return tensor[:steps].reshape(steps * self.num_envs, *tensor.shape[2:])

        initial_lstm_state = None
        if (
            self.rnn_config.USE_RNN
            and self.hidden_states is not None
            and self.cell_states is not None
        ):
            # State at the very beginning of the rollout (index 0)
            initial_lstm_state = (self.hidden_states[0], self.cell_states[0])

        data = {
            "obs_grid": _flatten(self.obs_grid, effective_num_steps),
            "obs_shapes": _flatten(self.obs_shapes, effective_num_steps),
            "obs_availability": _flatten(self.obs_availability, effective_num_steps),
            # --- UPDATED: Flatten explicit features ---
            "obs_explicit_features": _flatten(
                self.obs_explicit_features, effective_num_steps
            ),
            # --- END UPDATED ---
            "actions": _flatten(self.actions, effective_num_steps).squeeze(-1),
            "log_probs": _flatten(self.log_probs, effective_num_steps).squeeze(-1),
            "values": _flatten(self.values, effective_num_steps).squeeze(
                -1
            ),  # Values V(s_0) to V(s_{T-1})
            "returns": _flatten(self.returns, effective_num_steps).squeeze(
                -1
            ),  # Returns GAE(s_0) to GAE(s_{T-1})
            "advantages": _flatten(advantages, effective_num_steps).squeeze(-1),
            "initial_lstm_state": initial_lstm_state,  # Pass the initial LSTM state for sequence evaluation
            # Dones corresponding to obs t=0 to t=T-1 (i.e., d_0 to d_{T-1})
            # Shape [B, T] for potential RNN sequence processing needs
            "dones": self.dones[:effective_num_steps].permute(1, 0, 2).squeeze(-1),
        }
        return data


File: training\trainer.py
# File: training/trainer.py
import time
import torch
import numpy as np
import traceback
import random
from typing import List, Optional, Dict, Any, Union

from config import (
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    DEVICE,
    TensorBoardConfig,
    VisConfig,
    RewardConfig,
    TOTAL_TRAINING_STEPS,
)
from environment.game_state import GameState, StateType
from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from utils.helpers import ensure_numpy
from .rollout_storage import RolloutStorage
from .rollout_collector import RolloutCollector
from .checkpoint_manager import CheckpointManager
from .training_utils import get_env_image_as_numpy


class Trainer:
    """Orchestrates the PPO training process."""

    def __init__(
        self,
        envs: List[GameState],
        agent: PPOAgent,
        stats_recorder: StatsRecorderBase,
        env_config: EnvConfig,
        ppo_config: PPOConfig,
        rnn_config: RNNConfig,
        train_config: TrainConfig,
        model_config: ModelConfig,
        model_save_path: str,
        load_checkpoint_path: Optional[str] = None,
    ):
        print("[Trainer-PPO] Initializing...")
        self.envs = envs
        self.agent = agent
        self.stats_recorder = stats_recorder
        self.num_envs = env_config.NUM_ENVS
        self.device = DEVICE
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.train_config = train_config
        self.model_config = model_config
        self.reward_config = RewardConfig()
        self.tb_config = TensorBoardConfig()
        self.vis_config = VisConfig()

        self.checkpoint_manager = CheckpointManager(
            agent=self.agent,
            model_save_path=model_save_path,
            load_checkpoint_path=load_checkpoint_path,
            device=self.device,
        )
        self.global_step, initial_episode_count = (
            self.checkpoint_manager.get_initial_state()
        )

        self.rollout_collector = RolloutCollector(
            envs=self.envs,
            agent=self.agent,
            stats_recorder=self.stats_recorder,
            env_config=self.env_config,
            ppo_config=self.ppo_config,
            rnn_config=self.rnn_config,
            reward_config=self.reward_config,
            tb_config=self.tb_config,
        )
        self.rollout_collector.episode_count = initial_episode_count

        self.last_image_log_step = -1
        self.last_checkpoint_step = 0
        self.rollouts_completed_since_last_checkpoint = 0

        self.steps_collected_this_rollout = 0
        self.rollout_storage = self.rollout_collector.rollout_storage

        self.current_phase = "Collecting"  # Track current phase

        self._log_initial_state()
        print("[Trainer-PPO] Initialization complete.")

    def get_current_phase(self) -> str:
        """Returns the current phase: 'Collecting' or 'Updating'."""
        return self.current_phase

    def get_update_progress(self) -> float:
        """Returns the agent's update progress if in 'Updating' phase."""
        if self.current_phase == "Updating":
            return self.agent.get_update_progress()
        return 0.0

    def _log_initial_state(self):
        initial_lr = self._get_current_lr()
        self.stats_recorder.record_step(
            {
                "lr": initial_lr,
                "global_step": self.global_step,
                "episode_count": self.rollout_collector.get_episode_count(),
            }
        )
        print(
            f"  -> Start Step={self.global_step}, Ep={self.rollout_collector.get_episode_count()}, LR={initial_lr:.1e}"
        )

    def _get_current_lr(self) -> float:
        if hasattr(self.agent, "optimizer") and self.agent.optimizer.param_groups:
            return self.agent.optimizer.param_groups[0]["lr"]
        else:
            return self.ppo_config.LEARNING_RATE

    def _update_learning_rate(self):
        if not self.ppo_config.USE_LR_SCHEDULER:
            return
        total_steps = max(1, TOTAL_TRAINING_STEPS)
        frac = 1.0 - (self.global_step / total_steps)
        frac = max(self.ppo_config.LR_SCHEDULER_END_FRACTION, frac)
        new_lr = self.ppo_config.LEARNING_RATE * frac
        if hasattr(self.agent, "optimizer"):
            for param_group in self.agent.optimizer.param_groups:
                param_group["lr"] = new_lr

    def perform_training_iteration(self):
        """Performs one step of environment interaction and potentially an agent update."""
        step_start_time = time.time()

        # Ensure phase is 'Collecting' before starting collection
        if self.current_phase != "Collecting":
            self.current_phase = "Collecting"

        steps_collected_this_iter = self.rollout_collector.collect_one_step(
            self.global_step
        )
        self.global_step += steps_collected_this_iter
        self.steps_collected_this_rollout += 1

        update_metrics = {}

        if self.steps_collected_this_rollout >= self.ppo_config.NUM_STEPS_PER_ROLLOUT:
            # print(f"[Trainer Debug] Rollout complete...") # Removed
            self.current_phase = "Updating"  # Set phase before update
            update_start_time = time.time()

            self.rollout_collector.compute_advantages_for_storage()

            self.rollout_storage.to(self.agent.device)
            update_data = self.rollout_storage.get_data_for_update()

            # print(f"[Trainer Debug] update_data keys: ...") # Removed detailed print
            # for key, value in update_data.items(): ... # Removed detailed print

            if update_data:
                try:
                    update_metrics = self.agent.update(update_data)
                    # print(f"[Trainer Debug] update_metrics received...") # Removed
                except Exception as agent_update_err:
                    print(f"CRITICAL ERROR during agent.update: {agent_update_err}")
                    traceback.print_exc()
                    update_metrics = {}
            else:
                print(
                    "[Trainer Warning] No data retrieved from rollout storage for update. Skipping agent update."
                )
                update_metrics = {}

            self.rollout_storage.after_update()
            self.steps_collected_this_rollout = 0
            self.rollouts_completed_since_last_checkpoint += 1
            self.current_phase = "Collecting"  # Reset phase after update

            update_duration = time.time() - update_start_time

            self._update_learning_rate()
            self.maybe_save_checkpoint()
            self._maybe_log_image()

            step_record_data_update = {
                "update_time": update_duration,
                "lr": self._get_current_lr(),
                "global_step": self.global_step,
            }
            if isinstance(update_metrics, dict):
                step_record_data_update.update(update_metrics)
            else:
                print(
                    f"[Trainer Warning] agent.update did not return a dictionary. Received: {type(update_metrics)}"
                )

            # print(f"[Trainer Debug] Data being sent to stats_recorder...") # Removed
            self.stats_recorder.record_step(step_record_data_update)

        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        step_record_data_timing = {
            "step_time": step_duration,
            "num_steps_processed": steps_collected_this_iter,
            "global_step": self.global_step,
            "lr": self._get_current_lr(),
        }

        if not update_metrics:  # Only log step time if no update happened
            self.stats_recorder.record_step(step_record_data_timing)

    def maybe_save_checkpoint(self, force_save=False):
        """Saves agent state based on frequency or if forced."""
        save_freq_rollouts = self.train_config.CHECKPOINT_SAVE_FREQ
        should_save_freq = (
            save_freq_rollouts > 0
            and self.rollouts_completed_since_last_checkpoint >= save_freq_rollouts
        )

        if force_save or should_save_freq:
            print(
                f"[Trainer] Saving checkpoint. Force: {force_save}, FreqMet: {should_save_freq}, Rollouts Since Last: {self.rollouts_completed_since_last_checkpoint}"
            )
            self.checkpoint_manager.save_checkpoint(
                self.global_step, self.rollout_collector.get_episode_count()
            )
            self.rollouts_completed_since_last_checkpoint = 0
            self.last_checkpoint_step = self.global_step

    def _maybe_log_image(self):
        """Logs a sample environment state image to TensorBoard periodically."""
        if not self.tb_config.LOG_IMAGES or self.tb_config.IMAGE_LOG_FREQ <= 0:
            return

        image_log_freq_rollouts = self.tb_config.IMAGE_LOG_FREQ
        # Log based on rollouts completed *since last checkpoint*
        # Check if the *current* rollout number (1-based) is a multiple of the freq
        current_rollout_num_since_chkpt = self.rollouts_completed_since_last_checkpoint
        if (
            current_rollout_num_since_chkpt > 0
            and current_rollout_num_since_chkpt % image_log_freq_rollouts == 0
        ):
            if self.global_step > self.last_image_log_step:
                print(f"[Trainer] Logging image at step {self.global_step}")
                try:
                    env_idx = random.randrange(self.num_envs)
                    img_array = get_env_image_as_numpy(
                        self.envs[env_idx], self.env_config, self.vis_config
                    )
                    if img_array is not None:
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                        self.stats_recorder.record_image(
                            f"Environment/Sample State Env {env_idx}",
                            img_tensor,
                            self.global_step,
                        )
                        self.last_image_log_step = self.global_step
                except Exception as e:
                    print(f"Error logging environment image: {e}")
                    traceback.print_exc()

    def train_loop(self):
        """Main training loop until max steps."""
        print("[Trainer-PPO] Starting training loop...")
        try:
            while self.global_step < TOTAL_TRAINING_STEPS:
                self.perform_training_iteration()
        except KeyboardInterrupt:
            print("\n[Trainer-PPO] Training loop interrupted by user (Ctrl+C).")
        except Exception as e:
            print(f"\n[Trainer-PPO] CRITICAL ERROR in training loop: {e}")
            traceback.print_exc()
        finally:
            print("[Trainer-PPO] Training loop finished or terminated.")
            self.cleanup(save_final=True)

    def cleanup(self, save_final: bool = True):
        """Performs cleanup actions like saving final state and closing logger."""
        print("[Trainer-PPO] Cleaning up resources...")
        if save_final:
            print("[Trainer-PPO] Saving final checkpoint...")
            self.checkpoint_manager.save_checkpoint(
                self.global_step,
                self.rollout_collector.get_episode_count(),
                is_final=True,
            )
        else:
            print("[Trainer-PPO] Skipping final save as requested.")

        if hasattr(self.stats_recorder, "close"):
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"Error closing stats recorder: {e}")

        print("[Trainer-PPO] Cleanup complete.")


File: training\training_utils.py
# File: training/training_utils.py
import pygame
import numpy as np
from typing import Optional
from config import EnvConfig, VisConfig


def get_env_image_as_numpy(
    env, env_config: EnvConfig, vis_config: VisConfig
) -> Optional[np.ndarray]:
    """Renders a single environment state to a NumPy array for logging."""
    img_h = 300
    aspect_ratio = (env_config.COLS * 0.75 + 0.25) / max(1, env_config.ROWS)
    img_w = int(img_h * aspect_ratio)
    if img_w <= 0 or img_h <= 0:
        return None
    try:
        temp_surf = pygame.Surface((img_w, img_h))
        cell_w_px = img_w / (env_config.COLS * 0.75 + 0.25)
        cell_h_px = img_h / max(1, env_config.ROWS)
        temp_surf.fill(vis_config.BLACK)
        if hasattr(env, "grid") and hasattr(env.grid, "triangles"):
            for r in range(env.grid.rows):
                for c in range(env.grid.cols):
                    if r < len(env.grid.triangles) and c < len(env.grid.triangles[r]):
                        t = env.grid.triangles[r][c]
                        if t.is_death:
                            continue
                        pts = t.get_points(
                            ox=0, oy=0, cw=int(cell_w_px), ch=int(cell_h_px)
                        )
                        color = vis_config.GRAY
                        if t.is_occupied:
                            color = t.color if t.color else vis_config.RED
                        pygame.draw.polygon(temp_surf, color, pts)
        img_array = pygame.surfarray.array3d(temp_surf)
        return np.transpose(img_array, (1, 0, 2))
    except Exception as e:
        print(f"Error generating environment image for TB: {e}")
        return None


File: training\__init__.py
# File: training/__init__.py
from .trainer import Trainer
from .rollout_collector import RolloutCollector
from .rollout_storage import RolloutStorage
from .checkpoint_manager import CheckpointManager

__all__ = [
    "Trainer",
    "RolloutCollector",
    "RolloutStorage",
    "CheckpointManager",
]


File: ui\demo_renderer.py
# File: ui/demo_renderer.py
import pygame
import math
import traceback
from typing import Optional, Tuple

from config import VisConfig, EnvConfig, DemoConfig
from environment.game_state import GameState
from .panels.game_area import GameAreaRenderer


class DemoRenderer:
    """Handles rendering specifically for the interactive Demo Mode."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer
        self._init_demo_fonts()
        self.overlay_font = self.game_area_renderer.fonts.get("env_overlay")
        if not self.overlay_font:
            print("Warning: DemoRenderer could not get overlay font. Using default.")
            self.overlay_font = pygame.font.Font(None, 36)
        # --- NEW: Define invalid placement color ---
        self.invalid_placement_color = (0, 0, 0, 150)  # Transparent Gray
        # --- END NEW ---

    def _init_demo_fonts(self):
        try:
            self.demo_hud_font = pygame.font.SysFont(
                None, self.demo_config.HUD_FONT_SIZE
            )
            self.demo_help_font = pygame.font.SysFont(
                None, self.demo_config.HELP_FONT_SIZE
            )
            if not hasattr(
                self.game_area_renderer, "fonts"
            ) or not self.game_area_renderer.fonts.get("ui"):
                self.game_area_renderer._init_fonts()
        except Exception as e:
            print(f"Warning: SysFont error for demo fonts: {e}. Using default.")
            self.demo_hud_font = pygame.font.Font(None, self.demo_config.HUD_FONT_SIZE)
            self.demo_help_font = pygame.font.Font(
                None, self.demo_config.HELP_FONT_SIZE
            )

    def render(self, demo_env: GameState, env_config: EnvConfig):
        if not demo_env:
            print("Error: DemoRenderer called with demo_env=None")
            return

        bg_color = self._determine_background_color(demo_env)
        self.screen.fill(bg_color)

        sw, sh = self.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30
        max_game_h = sh - 2 * padding - hud_height - help_height
        max_game_w = sw - 2 * padding

        if max_game_h <= 0 or max_game_w <= 0:
            self._render_too_small_message(
                "Demo Area Too Small", self.screen.get_rect()
            )
            return

        game_rect, clipped_game_rect = self._calculate_game_area_rect(
            sw, sh, padding, hud_height, help_height, env_config
        )

        if clipped_game_rect.width > 10 and clipped_game_rect.height > 10:
            self._render_game_area(demo_env, env_config, clipped_game_rect, bg_color)
        else:
            self._render_too_small_message("Demo Area Too Small", clipped_game_rect)

        self._render_shape_previews_area(demo_env, sw, clipped_game_rect, padding)
        self._render_hud(demo_env, sw, game_rect.bottom + 10)
        self._render_help_text(sw, sh)

    def _determine_background_color(self, demo_env: GameState) -> Tuple[int, int, int]:
        if demo_env.is_line_clearing():
            return VisConfig.LINE_CLEAR_FLASH_COLOR
        elif demo_env.is_game_over_flashing():
            return VisConfig.GAME_OVER_FLASH_COLOR
        elif demo_env.is_over():
            return VisConfig.DARK_RED
        elif demo_env.is_frozen():
            return (30, 30, 100)
        else:
            return self.demo_config.BACKGROUND_COLOR

    def _calculate_game_area_rect(
        self,
        screen_width: int,
        screen_height: int,
        padding: int,
        hud_height: int,
        help_height: int,
        env_config: EnvConfig,
    ) -> Tuple[pygame.Rect, pygame.Rect]:
        max_game_h = screen_height - 2 * padding - hud_height - help_height
        max_game_w = screen_width - 2 * padding
        aspect_ratio = (env_config.COLS * 0.75 + 0.25) / max(1, env_config.ROWS)
        game_w = max_game_w
        game_h = game_w / aspect_ratio if aspect_ratio > 0 else max_game_h
        if game_h > max_game_h:
            game_h = max_game_h
            game_w = game_h * aspect_ratio
        game_w = math.floor(min(game_w, max_game_w))
        game_h = math.floor(min(game_h, max_game_h))
        game_x = (screen_width - game_w) // 2
        game_y = padding
        game_rect = pygame.Rect(game_x, game_y, game_w, game_h)
        clipped_game_rect = game_rect.clip(self.screen.get_rect())
        return game_rect, clipped_game_rect

    def _render_game_area(
        self,
        demo_env: GameState,
        env_config: EnvConfig,
        clipped_game_rect: pygame.Rect,
        bg_color: Tuple[int, int, int],
    ):
        try:
            game_surf = self.screen.subsurface(clipped_game_rect)
            game_surf.fill(bg_color)

            self.game_area_renderer._render_single_env_grid(
                game_surf, demo_env, env_config
            )

            preview_tri_cell_w, preview_tri_cell_h = self._calculate_demo_triangle_size(
                clipped_game_rect.width, clipped_game_rect.height, env_config
            )
            if preview_tri_cell_w > 0 and preview_tri_cell_h > 0:
                grid_ox, grid_oy = self._calculate_grid_offset(
                    clipped_game_rect.width, clipped_game_rect.height, env_config
                )
                self._render_placement_preview(
                    game_surf,
                    demo_env,
                    preview_tri_cell_w,
                    preview_tri_cell_h,
                    grid_ox,
                    grid_oy,
                )

            if demo_env.is_over():
                self._render_demo_overlay_text(game_surf, "GAME OVER", VisConfig.RED)
            elif demo_env.is_line_clearing():
                self._render_demo_overlay_text(game_surf, "Line Clear!", VisConfig.BLUE)

        except ValueError as e:
            print(f"Error subsurface demo game ({clipped_game_rect}): {e}")
            pygame.draw.rect(self.screen, VisConfig.RED, clipped_game_rect, 1)
        except Exception as render_e:
            print(f"Error rendering demo game area: {render_e}")
            traceback.print_exc()
            pygame.draw.rect(self.screen, VisConfig.RED, clipped_game_rect, 1)

    def _render_shape_previews_area(
        self,
        demo_env: GameState,
        screen_width: int,
        clipped_game_rect: pygame.Rect,
        padding: int,
    ):
        preview_area_w = min(150, screen_width - clipped_game_rect.right - padding // 2)
        if preview_area_w > 20:
            preview_area_rect = pygame.Rect(
                clipped_game_rect.right + padding // 2,
                clipped_game_rect.top,
                preview_area_w,
                clipped_game_rect.height,
            )
            clipped_preview_area_rect = preview_area_rect.clip(self.screen.get_rect())
            if (
                clipped_preview_area_rect.width > 0
                and clipped_preview_area_rect.height > 0
            ):
                try:
                    preview_area_surf = self.screen.subsurface(
                        clipped_preview_area_rect
                    )
                    self._render_demo_shape_previews(preview_area_surf, demo_env)
                except ValueError as e:
                    print(f"Error subsurface demo shape preview area: {e}")
                    pygame.draw.rect(
                        self.screen, VisConfig.RED, clipped_preview_area_rect, 1
                    )
                except Exception as e:
                    print(f"Error rendering demo shape previews: {e}")
                    traceback.print_exc()

    def _render_hud(self, demo_env: GameState, screen_width: int, hud_y: int):
        score_text = f"Score: {demo_env.game_score} | Lines: {demo_env.lines_cleared_this_episode}"
        try:
            score_surf = self.demo_hud_font.render(score_text, True, VisConfig.WHITE)
            score_rect = score_surf.get_rect(midtop=(screen_width // 2, hud_y))
            self.screen.blit(score_surf, score_rect)
        except Exception as e:
            print(f"HUD render error: {e}")

    def _render_help_text(self, screen_width: int, screen_height: int):
        try:
            help_surf = self.demo_help_font.render(
                self.demo_config.HELP_TEXT, True, VisConfig.LIGHTG
            )
            help_rect = help_surf.get_rect(
                centerx=screen_width // 2, bottom=screen_height - 10
            )
            self.screen.blit(help_surf, help_rect)
        except Exception as e:
            print(f"Help render error: {e}")

    def _render_demo_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        try:
            if not self.overlay_font:
                print("Error: Overlay font not available for demo overlay.")
                return
            text_surf = self.overlay_font.render(
                text,
                True,
                VisConfig.WHITE,
                (color[0] // 2, color[1] // 2, color[2] // 2, 220),
            )
            text_rect = text_surf.get_rect(center=surf.get_rect().center)
            surf.blit(text_surf, text_rect)
        except Exception as e:
            print(f"Error rendering demo overlay text '{text}': {e}")

    def _calculate_demo_triangle_size(
        self, surf_w: int, surf_h: int, env_config: EnvConfig
    ) -> Tuple[int, int]:
        padding = self.vis_config.ENV_GRID_PADDING
        drawable_w = max(1, surf_w - 2 * padding)
        drawable_h = max(1, surf_h - 2 * padding)
        grid_rows = env_config.ROWS
        grid_cols_eff_width = env_config.COLS * 0.75 + 0.25
        if grid_rows <= 0 or grid_cols_eff_width <= 0:
            return 0, 0
        scale_w = drawable_w / grid_cols_eff_width
        scale_h = drawable_h / grid_rows
        final_scale = min(scale_w, scale_h)
        if final_scale <= 0:
            return 0, 0
        tri_cell_size = max(1, int(final_scale))
        return tri_cell_size, tri_cell_size

    def _calculate_grid_offset(
        self, surf_w: int, surf_h: int, env_config: EnvConfig
    ) -> Tuple[float, float]:
        padding = self.vis_config.ENV_GRID_PADDING
        drawable_w = max(1, surf_w - 2 * padding)
        drawable_h = max(1, surf_h - 2 * padding)
        grid_rows = env_config.ROWS
        grid_cols_eff_width = env_config.COLS * 0.75 + 0.25
        if grid_rows <= 0 or grid_cols_eff_width <= 0:
            return float(padding), float(padding)
        scale_w = drawable_w / grid_cols_eff_width
        scale_h = drawable_h / grid_rows
        final_scale = min(scale_w, scale_h)
        final_grid_pixel_w = max(1, grid_cols_eff_width * final_scale)
        final_grid_pixel_h = max(1, grid_rows * final_scale)
        grid_ox = padding + (drawable_w - final_grid_pixel_w) / 2
        grid_oy = padding + (drawable_h - final_grid_pixel_h) / 2
        return grid_ox, grid_oy

    def _render_placement_preview(
        self,
        surf: pygame.Surface,
        env: GameState,
        cell_w: int,
        cell_h: int,
        offset_x: float,
        offset_y: float,
    ):
        if cell_w <= 0 or cell_h <= 0:
            return
        shp, rr, cc = env.get_current_selection_info()
        if shp is None:
            return

        is_valid = env.grid.can_place(shp, rr, cc)

        preview_alpha = 150
        if is_valid:
            shape_rgb = shp.color
            preview_color_to_use = (
                shape_rgb[0],
                shape_rgb[1],
                shape_rgb[2],
                preview_alpha,
            )
        else:
            # --- MODIFIED: Use the defined gray color for invalid placement ---
            preview_color_to_use = self.invalid_placement_color
            # --- END MODIFIED ---

        temp_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))

        for dr, dc, up in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if (
                env.grid.valid(nr, nc)
                and 0 <= nr < len(env.grid.triangles)
                and 0 <= nc < len(env.grid.triangles[nr])
                and not env.grid.triangles[nr][nc].is_death
            ):
                temp_tri = env.grid.triangles[nr][nc]
                try:
                    pts = temp_tri.get_points(
                        ox=offset_x, oy=offset_y, cw=cell_w, ch=cell_h
                    )
                    pygame.draw.polygon(temp_surface, preview_color_to_use, pts)
                except Exception as e:
                    pass

        surf.blit(temp_surface, (0, 0))

    def _render_demo_shape_previews(self, surf: pygame.Surface, env: GameState):
        surf.fill((25, 25, 25))
        all_slots = env.shapes
        selected_shape_obj = (
            all_slots[env.demo_selected_shape_idx]
            if 0 <= env.demo_selected_shape_idx < len(all_slots)
            else None
        )
        num_slots = env.env_config.NUM_SHAPE_SLOTS
        surf_w, surf_h = surf.get_size()
        preview_padding = 5

        if num_slots <= 0:
            return

        preview_h = max(20, (surf_h - (num_slots + 1) * preview_padding) / num_slots)
        preview_w = max(20, surf_w - 2 * preview_padding)
        current_preview_y = preview_padding

        for i in range(num_slots):
            shp = all_slots[i] if i < len(all_slots) else None
            preview_rect = pygame.Rect(
                preview_padding, current_preview_y, preview_w, preview_h
            )
            clipped_preview_rect = preview_rect.clip(surf.get_rect())

            if clipped_preview_rect.width <= 0 or clipped_preview_rect.height <= 0:
                current_preview_y += preview_h + preview_padding
                continue

            bg_color = (40, 40, 40)
            border_color = VisConfig.GRAY
            border_width = 1
            if shp is not None and shp == selected_shape_obj:
                border_color = self.demo_config.SELECTED_SHAPE_HIGHLIGHT_COLOR
                border_width = 2

            pygame.draw.rect(surf, bg_color, clipped_preview_rect, border_radius=3)
            pygame.draw.rect(
                surf, border_color, clipped_preview_rect, border_width, border_radius=3
            )

            if shp is not None:
                self._render_single_shape_in_preview_box(
                    surf, shp, preview_rect, clipped_preview_rect
                )

            current_preview_y += preview_h + preview_padding

    def _render_single_shape_in_preview_box(
        self,
        surf: pygame.Surface,
        shp,
        preview_rect: pygame.Rect,
        clipped_preview_rect: pygame.Rect,
    ):
        try:
            inner_padding = 2
            shape_render_area_rect = pygame.Rect(
                inner_padding,
                inner_padding,
                clipped_preview_rect.width - 2 * inner_padding,
                clipped_preview_rect.height - 2 * inner_padding,
            )
            if shape_render_area_rect.width > 0 and shape_render_area_rect.height > 0:
                sub_surf_x = preview_rect.left + shape_render_area_rect.left
                sub_surf_y = preview_rect.top + shape_render_area_rect.top
                shape_sub_surf = surf.subsurface(
                    sub_surf_x,
                    sub_surf_y,
                    shape_render_area_rect.width,
                    shape_render_area_rect.height,
                )
                min_r, min_c, max_r, max_c = shp.bbox()
                shape_h = max(1, max_r - min_r + 1)
                shape_w_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)

                scale_h = shape_render_area_rect.height / shape_h
                scale_w = shape_render_area_rect.width / shape_w_eff
                cell_size = max(1, min(scale_h, scale_w))

                self.game_area_renderer._render_single_shape(
                    shape_sub_surf, shp, int(cell_size)
                )
        except ValueError as sub_err:
            print(f"Error subsurface shape preview: {sub_err}")
            pygame.draw.rect(surf, VisConfig.RED, clipped_preview_rect, 1)
        except Exception as e:
            print(f"Error rendering demo shape preview: {e}")
            pygame.draw.rect(surf, VisConfig.RED, clipped_preview_rect, 1)

    def _render_too_small_message(self, text: str, area_rect: pygame.Rect):
        try:
            font = self.game_area_renderer.fonts.get("ui") or pygame.font.SysFont(
                None, 24
            )
            err_surf = font.render(text, True, VisConfig.GRAY)
            target_rect = err_surf.get_rect(center=area_rect.center)
            self.screen.blit(err_surf, target_rect)
        except Exception as e:
            print(f"Error rendering 'too small' message: {e}")


File: ui\input_handler.py
# File: ui/input_handler.py
import pygame
from typing import Tuple, Callable, Optional

# Type Aliases for Callbacks
HandleDemoInputCallback = Callable[[pygame.event.Event], None]
ToggleTrainingRunCallback = Callable[[], None]  # Renamed
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]
StartDemoModeCallback = Callable[[], None]
ExitDemoModeCallback = Callable[[], None]

# Forward declaration for type hinting
if False:
    from .renderer import UIRenderer


class InputHandler:
    """Handles Pygame events and triggers callbacks based on application state."""

    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",
        toggle_training_run_cb: ToggleTrainingRunCallback,  # Renamed
        request_cleanup_cb: RequestCleanupCallback,
        cancel_cleanup_cb: CancelCleanupCallback,
        confirm_cleanup_cb: ConfirmCleanupCallback,
        exit_app_cb: ExitAppCallback,
        start_demo_mode_cb: StartDemoModeCallback,
        exit_demo_mode_cb: ExitDemoModeCallback,
        handle_demo_input_cb: HandleDemoInputCallback,
    ):
        self.screen = screen
        self.renderer = renderer
        self.toggle_training_run_cb = toggle_training_run_cb  # Renamed
        self.request_cleanup_cb = request_cleanup_cb
        self.cancel_cleanup_cb = cancel_cleanup_cb
        self.confirm_cleanup_cb = confirm_cleanup_cb
        self.exit_app_cb = exit_app_cb
        self.start_demo_mode_cb = start_demo_mode_cb
        self.exit_demo_mode_cb = exit_demo_mode_cb
        self.handle_demo_input_cb = handle_demo_input_cb

        self._update_button_rects()

    def _update_button_rects(self):
        """Calculates button rects based on initial layout assumptions."""
        # These rects are primarily for click detection, actual rendering is in LeftPanel
        self.run_btn_rect = pygame.Rect(10, 10, 100, 40)  # Renamed
        self.cleanup_btn_rect = pygame.Rect(self.run_btn_rect.right + 10, 10, 160, 40)
        self.demo_btn_rect = pygame.Rect(self.cleanup_btn_rect.right + 10, 10, 120, 40)
        # Confirmation buttons are positioned dynamically during rendering
        sw, sh = self.screen.get_size()
        self.confirm_yes_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_no_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

    def handle_input(self, app_state: str, cleanup_confirmation_active: bool) -> bool:
        """
        Processes Pygame events. Returns True to continue running, False to exit.
        """
        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            mouse_pos = (0, 0)  # Handle cases where display might not be initialized

        # Update confirmation button positions dynamically based on screen size
        sw, sh = self.screen.get_size()
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

        if app_state == "MainMenu" and not cleanup_confirmation_active:
            if hasattr(self.renderer, "check_hover"):
                self.renderer.check_hover(mouse_pos, app_state)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return self.exit_app_cb()

            if event.type == pygame.VIDEORESIZE:
                try:
                    new_w, new_h = max(320, event.w), max(240, event.h)
                    self.screen = pygame.display.set_mode(
                        (new_w, new_h), pygame.RESIZABLE
                    )
                    self._update_ui_screen_references(self.screen)
                    self._update_button_rects()  # Recalculate button rects
                    if hasattr(self.renderer, "force_redraw"):
                        self.renderer.force_redraw()
                    print(f"Window resized: {new_w}x{new_h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")
                continue  # Skip other event handling for this frame

            # --- Cleanup Confirmation Mode ---
            if cleanup_confirmation_active:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.cancel_cleanup_cb()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.confirm_yes_rect.collidepoint(mouse_pos):
                        self.confirm_cleanup_cb()
                    elif self.confirm_no_rect.collidepoint(mouse_pos):
                        self.cancel_cleanup_cb()
                continue  # Don't process other inputs during confirmation

            # --- Playing (Demo) Mode ---
            elif app_state == "Playing":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_demo_mode_cb()
                    else:
                        # Delegate other key presses to the demo input handler
                        self.handle_demo_input_cb(event)

            # --- Main Menu Mode ---
            elif app_state == "MainMenu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return self.exit_app_cb()
                    elif event.key == pygame.K_p:  # 'P' key toggles training
                        self.toggle_training_run_cb()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Check button clicks using the stored rects
                    if self.run_btn_rect.collidepoint(mouse_pos):
                        self.toggle_training_run_cb()
                    elif self.cleanup_btn_rect.collidepoint(mouse_pos):
                        self.request_cleanup_cb()
                    elif self.demo_btn_rect.collidepoint(mouse_pos):
                        self.start_demo_mode_cb()

            # --- Error Mode ---
            elif app_state == "Error":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return self.exit_app_cb()

        return True  # Continue running

    def _update_ui_screen_references(self, new_screen: pygame.Surface):
        """Updates the screen reference in the renderer and its sub-components."""
        components_to_update = [
            self.renderer,
            getattr(self.renderer, "left_panel", None),
            getattr(self.renderer, "game_area", None),
            getattr(self.renderer, "overlays", None),
            getattr(self.renderer, "tooltips", None),
            getattr(self.renderer, "demo_renderer", None),
        ]
        for component in components_to_update:
            if component and hasattr(component, "screen"):
                component.screen = new_screen


File: ui\overlays.py
# File: ui/overlays.py
# (No significant changes needed, this file was already focused)
import pygame
import time
import traceback
from typing import Tuple
from config import VisConfig


class OverlayRenderer:
    """Renders overlay elements like confirmation dialogs and status messages."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()

    def _init_fonts(self):
        """Initializes fonts used for overlays."""
        fonts = {}
        try:
            fonts["overlay_title"] = pygame.font.SysFont(None, 36)
            fonts["overlay_text"] = pygame.font.SysFont(None, 24)
        except Exception as e:
            print(f"Warning: SysFont error for overlay fonts: {e}. Using default.")
            fonts["overlay_title"] = pygame.font.Font(None, 36)
            fonts["overlay_text"] = pygame.font.Font(None, 24)
        return fonts

    def render_cleanup_confirmation(self):
        """Renders the confirmation dialog for cleanup. Does not flip display."""
        try:
            current_width, current_height = self.screen.get_size()

            # Semi-transparent background overlay
            overlay_surface = pygame.Surface(
                (current_width, current_height), pygame.SRCALPHA
            )
            overlay_surface.fill((0, 0, 0, 200))  # Black with alpha
            self.screen.blit(overlay_surface, (0, 0))

            center_x, center_y = current_width // 2, current_height // 2

            # --- Render Text Lines ---
            if "overlay_title" not in self.fonts or "overlay_text" not in self.fonts:
                print("ERROR: Overlay fonts not loaded!")
                # Draw basic fallback text
                fallback_font = pygame.font.Font(None, 30)
                err_surf = fallback_font.render("CONFIRM CLEANUP?", True, VisConfig.RED)
                self.screen.blit(
                    err_surf, err_surf.get_rect(center=(center_x, center_y - 30))
                )
                yes_surf = fallback_font.render("YES", True, VisConfig.WHITE)
                no_surf = fallback_font.render("NO", True, VisConfig.WHITE)
                self.screen.blit(
                    yes_surf, yes_surf.get_rect(center=(center_x - 60, center_y + 50))
                )
                self.screen.blit(
                    no_surf, no_surf.get_rect(center=(center_x + 60, center_y + 50))
                )
                return  # Stop here if fonts failed

            # Use loaded fonts
            prompt_l1 = self.fonts["overlay_title"].render(
                "DELETE CURRENT RUN DATA?", True, VisConfig.RED
            )
            prompt_l2 = self.fonts["overlay_text"].render(
                "(Agent Checkpoint & Buffer State)", True, VisConfig.WHITE
            )
            prompt_l3 = self.fonts["overlay_text"].render(
                "This action cannot be undone!", True, VisConfig.YELLOW
            )

            # Position and blit text
            self.screen.blit(
                prompt_l1, prompt_l1.get_rect(center=(center_x, center_y - 60))
            )
            self.screen.blit(
                prompt_l2, prompt_l2.get_rect(center=(center_x, center_y - 25))
            )
            self.screen.blit(prompt_l3, prompt_l3.get_rect(center=(center_x, center_y)))

            # --- Render Buttons ---
            # Recalculate rects based on current screen size for responsiveness
            confirm_yes_rect = pygame.Rect(center_x - 110, center_y + 30, 100, 40)
            confirm_no_rect = pygame.Rect(center_x + 10, center_y + 30, 100, 40)

            pygame.draw.rect(
                self.screen, (0, 150, 0), confirm_yes_rect, border_radius=5
            )  # Green YES
            pygame.draw.rect(
                self.screen, (150, 0, 0), confirm_no_rect, border_radius=5
            )  # Red NO

            yes_text = self.fonts["overlay_text"].render("YES", True, VisConfig.WHITE)
            no_text = self.fonts["overlay_text"].render("NO", True, VisConfig.WHITE)

            self.screen.blit(
                yes_text, yes_text.get_rect(center=confirm_yes_rect.center)
            )
            self.screen.blit(no_text, no_text.get_rect(center=confirm_no_rect.center))

        except pygame.error as pg_err:
            print(f"Pygame Error in render_cleanup_confirmation: {pg_err}")
            traceback.print_exc()
        except Exception as e:
            print(f"Error in render_cleanup_confirmation: {e}")
            traceback.print_exc()

    def render_status_message(self, message: str, last_message_time: float) -> bool:
        """
        Renders a status message (e.g., after cleanup) temporarily at the bottom center.
        Does not flip display. Returns True if a message was rendered.
        """
        # Check if message exists and hasn't timed out
        if not message or (time.time() - last_message_time >= 5.0):
            return False

        try:
            if "overlay_text" not in self.fonts:  # Check if font loaded
                print(
                    "Warning: Cannot render status message, overlay_text font missing."
                )
                return False

            current_width, current_height = self.screen.get_size()
            lines = message.split("\n")
            max_width = 0
            msg_surfs = []

            # Render each line and find max width
            for line in lines:
                msg_surf = self.fonts["overlay_text"].render(
                    line,
                    True,
                    VisConfig.YELLOW,
                    VisConfig.BLACK,  # Yellow text on black bg
                )
                msg_surfs.append(msg_surf)
                max_width = max(max_width, msg_surf.get_width())

            if not msg_surfs:
                return False  # No lines to render

            # Calculate background size and position
            total_height = (
                sum(s.get_height() for s in msg_surfs) + max(0, len(lines) - 1) * 2
            )
            padding = 5
            bg_rect = pygame.Rect(
                0, 0, max_width + padding * 2, total_height + padding * 2
            )
            bg_rect.midbottom = (
                current_width // 2,
                current_height - 10,
            )  # Position at bottom center

            # Draw background and border
            pygame.draw.rect(self.screen, VisConfig.BLACK, bg_rect, border_radius=3)
            pygame.draw.rect(self.screen, VisConfig.YELLOW, bg_rect, 1, border_radius=3)

            # Draw text lines centered within the background
            current_y = bg_rect.top + padding
            for msg_surf in msg_surfs:
                msg_rect = msg_surf.get_rect(midtop=(bg_rect.centerx, current_y))
                self.screen.blit(msg_surf, msg_rect)
                current_y += msg_surf.get_height() + 2  # Move Y for next line

            return True  # Message was rendered
        except Exception as e:
            print(f"Error rendering status message: {e}")
            traceback.print_exc()
            return False  # Message render failed


File: ui\plotter.py
# File: ui/plotter.py
import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
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
        self.plot_update_interval: float = 1.0
        self.rolling_window_sizes = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW

        self.colors = {
            "rl_score": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[0]),
            "game_score": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[1]),
            "policy_loss": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[3]),
            "value_loss": normalize_color_for_matplotlib(VisConfig.BLUE),
            "entropy": normalize_color_for_matplotlib((150, 150, 150)),
            "len": normalize_color_for_matplotlib(VisConfig.BLUE),
            "sps": normalize_color_for_matplotlib(VisConfig.LIGHTG),
            "best_game": normalize_color_for_matplotlib((255, 165, 0)),
            "lr": normalize_color_for_matplotlib((255, 0, 255)),
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
        }

    def create_plot_surface(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:

        if target_width <= 10 or target_height <= 10 or not plot_data:
            return None

        data_keys = [
            "episode_scores",
            "game_scores",
            "policy_loss",
            "value_loss",
            "entropy",
            "episode_lengths",
            "sps_values",
            "best_game_score_history",
            "lr_values",
        ]
        data_lists = {key: list(plot_data.get(key, deque())) for key in data_keys}

        # --- REMOVED Plotter Debug Print ---
        # print(f"[Plotter Debug] Data lengths: ...")
        # print(f"  PLoss sample: ...")
        # print(f"  VLoss sample: ...")
        # print(f"  Entropy sample: ...")
        # --- END REMOVED ---

        has_meaningful_data = any(
            len(data_lists.get(key, [])) > 0
            for key in ["episode_scores", "policy_loss", "value_loss", "entropy"]
        )
        # Only prevent plotting if *all* key data series are empty
        # Allow plotting if at least scores are present, even if losses aren't yet
        has_any_data = any(len(d) > 0 for d in data_lists.values())
        if not has_any_data:
            return None  # Return None if absolutely no data exists for any plot

        fig = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                dpi = 90
                fig_width_in = max(1, target_width / dpi)
                fig_height_in = max(1, target_height / dpi)

                fig, axes = plt.subplots(
                    3, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
                )
                fig.subplots_adjust(
                    hspace=0.30,
                    wspace=0.15,
                    left=0.08,
                    right=0.98,
                    bottom=0.10,
                    top=0.92,
                )
                axes_flat = axes.flatten()

                max_len = max((len(d) for d in data_lists.values() if d), default=0)
                plot_window_label = (
                    f"Latest {min(self.plot_data_window, max_len)} Updates"
                )

                # Pass the actual data lists to render_single_plot
                render_single_plot(
                    axes_flat[0],
                    data_lists["episode_scores"],
                    "RL Score",
                    self.colors["rl_score"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="RL Score",
                )
                render_single_plot(
                    axes_flat[1],
                    data_lists["game_scores"],
                    "Game Score",
                    self.colors["game_score"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Game Score",
                )
                render_single_plot(
                    axes_flat[2],
                    data_lists["policy_loss"],
                    "Policy Loss",
                    self.colors["policy_loss"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Policy Loss",
                )
                render_single_plot(
                    axes_flat[3],
                    data_lists["value_loss"],
                    "Value Loss",
                    self.colors["value_loss"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Value Loss",
                )
                render_single_plot(
                    axes_flat[4],
                    data_lists["entropy"],
                    "Entropy",
                    self.colors["entropy"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Entropy",
                )
                render_single_plot(
                    axes_flat[5],
                    data_lists["episode_lengths"],
                    "Ep Length",
                    self.colors["len"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="Episode Length",
                )
                render_single_plot(
                    axes_flat[6],
                    data_lists["sps_values"],
                    "Steps/Sec",
                    self.colors["sps"],
                    self.rolling_window_sizes,
                    xlabel=plot_window_label,
                    placeholder_text="SPS",
                )
                render_single_plot(
                    axes_flat[7],
                    data_lists["best_game_score_history"],
                    "Best Game Score",
                    self.colors["best_game"],
                    [],
                    xlabel=plot_window_label,
                    placeholder_text="Best Game Score",
                )
                render_single_plot(
                    axes_flat[8],
                    data_lists["lr_values"],
                    "Learning Rate",
                    self.colors["lr"],
                    [],
                    xlabel=plot_window_label,
                    y_log_scale=True,
                    placeholder_text="Learning Rate",
                )

                for ax in axes_flat:
                    ax.tick_params(axis="x", rotation=0)

                buf = BytesIO()
                fig.savefig(
                    buf,
                    format="png",
                    transparent=False,
                    facecolor=plt.rcParams["figure.facecolor"],
                )
                buf.seek(0)
                plot_img_surface = pygame.image.load(buf).convert()
                buf.close()

                current_size = plot_img_surface.get_size()
                if current_size != (target_width, target_height):
                    plot_img_surface = pygame.transform.smoothscale(
                        plot_img_surface, (target_width, target_height)
                    )

                return plot_img_surface

        except Exception as e:
            print(f"Error creating plot surface: {e}")
            traceback.print_exc()
            return None
        finally:
            if fig is not None:
                plt.close(fig)

    def get_cached_or_updated_plot(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        current_time = time.time()
        has_data = any(d for d in plot_data.values())
        needs_update_time = (
            current_time - self.last_plot_update_time > self.plot_update_interval
        )
        size_changed = self.plot_surface and self.plot_surface.get_size() != (
            target_width,
            target_height,
        )
        first_plot_needed = has_data and self.plot_surface is None
        can_create_plot = target_width > 50 and target_height > 50

        if can_create_plot and (needs_update_time or size_changed or first_plot_needed):
            if has_data:
                new_plot_surface = self.create_plot_surface(
                    plot_data, target_width, target_height
                )
                # Only update cache if plot creation was successful
                if new_plot_surface:
                    self.plot_surface = new_plot_surface
                # If creation failed (e.g., due to error), keep the old cached plot (if any)
                # else: self.plot_surface = None # Optionally clear cache on failure
                self.last_plot_update_time = current_time
            elif not has_data:  # No data at all, clear cache
                self.plot_surface = None

        return self.plot_surface


File: ui\plot_utils.py
# File: ui/plot_utils.py
import pygame
import numpy as np
from typing import Dict, Optional, Deque, List, Union, Tuple
import matplotlib
import warnings
from io import BytesIO
import traceback
import math

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, StatsConfig


def normalize_color_for_matplotlib(
    color_tuple_0_255: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        return tuple(c / 255.0 for c in color_tuple_0_255)
    else:
        print(f"Warning: Invalid color tuple {color_tuple_0_255}, using black.")
        return (0.0, 0.0, 0.0)


try:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "figure.facecolor": "#262626",
            "axes.facecolor": "#303030",
            "axes.edgecolor": "#707070",
            "axes.labelcolor": "#D0D0D0",
            "xtick.color": "#C0C0C0",
            "ytick.color": "#C0C0C0",
            "grid.color": "#505050",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "font.small_title_values": 7,
            "axes.titlepad": 12,
        }
    )
except Exception as e:
    print(f"Warning: Failed to set Matplotlib style: {e}")


TREND_SLOPE_TOLERANCE = 1e-5
TREND_MIN_LINEWIDTH = 1.5
TREND_MAX_LINEWIDTH = 3.5
TREND_COLOR_STABLE = normalize_color_for_matplotlib(VisConfig.YELLOW)
TREND_COLOR_INCREASING = normalize_color_for_matplotlib((0, 200, 0))
TREND_COLOR_DECREASING = normalize_color_for_matplotlib((200, 0, 0))
TREND_SLOPE_SCALE_FACTOR = 5.0
TREND_BACKGROUND_ALPHA = 0.15

TREND_LINE_COLOR = (1.0, 1.0, 1.0)
TREND_LINE_STYLE = (0, (5, 10))
TREND_LINE_WIDTH = 0.75
TREND_LINE_ALPHA = 0.7
TREND_LINE_ZORDER = 10

MIN_ALPHA = 0.3
MAX_ALPHA = 1.0
MIN_DATA_AVG_LINEWIDTH = 1
MAX_DATA_AVG_LINEWIDTH = 3


def calculate_trend_line(data: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Calculates the slope and intercept of the linear regression line.
    Returns (slope, intercept) or None if calculation fails.
    """
    n_points = len(data)
    if n_points < 2:
        return None
    try:
        x_coords = np.arange(n_points)
        finite_mask = np.isfinite(data)
        if np.sum(finite_mask) < 2:
            return None
        coeffs = np.polyfit(x_coords[finite_mask], data[finite_mask], 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return None
        return slope, intercept
    except (np.linalg.LinAlgError, ValueError):
        return None


def get_trend_color(slope: float) -> Tuple[float, float, float]:
    """
    Maps a slope to a color (Red -> Yellow(Stable) -> Green).
    Assumes positive slope is "good" and negative is "bad".
    The caller should adjust the slope sign based on metric goal.
    """
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_COLOR_STABLE
    norm_slope = math.atan(slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0)
    norm_slope = np.clip(norm_slope, -1.0, 1.0)
    if norm_slope > 0:
        t = norm_slope
        color = tuple(
            TREND_COLOR_STABLE[i] * (1 - t) + TREND_COLOR_INCREASING[i] * t
            for i in range(3)
        )
    else:
        t = abs(norm_slope)
        color = tuple(
            TREND_COLOR_STABLE[i] * (1 - t) + TREND_COLOR_DECREASING[i] * t
            for i in range(3)
        )
    return tuple(np.clip(c, 0.0, 1.0) for c in color)


def get_trend_linewidth(slope: float) -> float:
    """Maps the *magnitude* of a slope to a border linewidth."""
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_MIN_LINEWIDTH
    norm_slope_mag = abs(math.atan(slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0))
    norm_slope_mag = np.clip(norm_slope_mag, 0.0, 1.0)
    linewidth = TREND_MIN_LINEWIDTH + norm_slope_mag * (
        TREND_MAX_LINEWIDTH - TREND_MIN_LINEWIDTH
    )
    return linewidth


def _interpolate_visual_property(
    rank: int, total_ranks: int, min_val: float, max_val: float
) -> float:
    """
    Linearly interpolates alpha or linewidth based on rank.
    Rank 0 corresponds to max_val (most prominent).
    Rank (total_ranks - 1) corresponds to min_val (least prominent).
    """
    if total_ranks <= 1:
        return max_val
    inverted_rank = (total_ranks - 1) - rank
    fraction = inverted_rank / max(1, total_ranks - 1)
    value = min_val + (max_val - min_val) * fraction
    return np.clip(value, min_val, max_val)


def _format_value(value: float, is_loss: bool) -> str:
    """Formats value based on magnitude and whether it's a loss."""
    if not np.isfinite(value):
        return "N/A"
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.1e}"
    if abs(value) >= 1000:
        return f"{value:.2g}"
    if is_loss:
        return f"{value:.3f}"
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.2f}"


def _format_slope(slope: float) -> str:
    """Formats slope value for display in the legend."""
    if not np.isfinite(slope):
        return "N/A"
    sign = "+" if slope >= 0 else ""
    if abs(slope) < 1e-4:
        return f"{sign}{slope:.1e}"
    elif abs(slope) < 0.1:
        return f"{sign}{slope:.3f}"
    else:
        return f"{sign}{slope:.2f}"


def render_single_plot(
    ax,
    data: List[Union[float, int]],
    label: str,
    color: Tuple[float, float, float],
    rolling_window_sizes: List[int],
    xlabel: Optional[str] = None,
    show_placeholder: bool = True,
    placeholder_text: Optional[str] = None,
    y_log_scale: bool = False,
):
    """
    Renders data with linearly scaled alpha/linewidth. Trend line is thin, white, dashed.
    Title is two lines: Label, then compact value pairs.
    Applies a background tint and border to the entire subplot based on trend desirability.
    Legend now includes current values and trend slope.
    Handles empty data explicitly to show placeholder.
    """
    try:
        data_np = np.array(data, dtype=float)
        finite_mask = np.isfinite(data_np)
        valid_data = data_np[finite_mask]
    except (ValueError, TypeError):
        valid_data = np.array([])

    n_points = len(valid_data)
    placeholder_text_color = normalize_color_for_matplotlib(VisConfig.GRAY)

    # --- Explicitly handle n_points == 0 case ---
    if n_points == 0:
        if show_placeholder:
            p_text = placeholder_text if placeholder_text else f"{label}\n(No data)"
            ax.text(
                0.5,
                0.5,
                p_text,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color=placeholder_text_color,
            )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"{label} (N/A)", fontsize=plt.rcParams["axes.titlesize"])
        ax.grid(False)
        ax.patch.set_facecolor(plt.rcParams["axes.facecolor"])
        ax.patch.set_edgecolor(plt.rcParams["axes.edgecolor"])
        ax.patch.set_linewidth(0.5)
        return  # Exit early if no data
    # --- End n_points == 0 handling ---

    # --- Continue with plotting if n_points > 0 ---
    trend_params = calculate_trend_line(valid_data)
    trend_slope = trend_params[0] if trend_params is not None else 0.0

    is_lower_better = "loss" in label.lower() or "entropy" in label.lower()

    effective_slope_for_color = -trend_slope if is_lower_better else trend_slope
    trend_indicator_color = get_trend_color(effective_slope_for_color)
    trend_indicator_lw = get_trend_linewidth(trend_slope)

    plotted_windows = sorted([w for w in rolling_window_sizes if n_points >= w])
    total_ranks = 1 + len(plotted_windows)

    current_val = valid_data[-1]
    best_val = np.min(valid_data) if is_lower_better else np.max(valid_data)

    title_line1 = f"{label}"
    value_pairs = []
    value_pairs.append(
        f"Now:{_format_value(current_val, is_lower_better)}|Best:{_format_value(best_val, is_lower_better)}"
    )

    avg_values = {}
    for avg_window in plotted_windows:
        try:
            avg_values[avg_window] = np.mean(valid_data[-avg_window:])
        except Exception:
            avg_values[avg_window] = np.nan

    avg_windows_sorted = sorted(avg_values.keys())
    for i in range(0, len(avg_windows_sorted), 2):
        win1 = avg_windows_sorted[i]
        avg1 = avg_values.get(win1, np.nan)
        pair_str = f"A{win1}:{_format_value(avg1, is_lower_better)}"

        if i + 1 < len(avg_windows_sorted):
            win2 = avg_windows_sorted[i + 1]
            avg2 = avg_values.get(win2, np.nan)
            pair_str += f"|A{win2}:{_format_value(avg2, is_lower_better)}"
            if np.isfinite(avg1) and np.isfinite(avg2):
                diff = avg1 - avg2
                diff_sign = "+" if diff >= 0 else ""
                pair_str += f"(D:{diff_sign}{_format_value(diff, is_lower_better)})"
        value_pairs.append(pair_str)

    title_line2 = " • ".join(value_pairs)

    ax.set_title(
        title_line1,
        loc="left",
        fontsize=plt.rcParams["axes.titlesize"],
        pad=plt.rcParams.get("axes.titlepad", 6),
    )
    ax.text(
        0.01,
        0.99,
        title_line2,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=plt.rcParams.get("font.small_title_values", 7),
        color=plt.rcParams["axes.labelcolor"],
    )

    try:
        x_coords = np.arange(n_points)
        plotted_legend_items = False

        # Plot Raw Data
        raw_data_rank = total_ranks - 1
        raw_data_alpha = _interpolate_visual_property(
            raw_data_rank, total_ranks, MIN_ALPHA, MAX_ALPHA
        )
        raw_data_lw = _interpolate_visual_property(
            raw_data_rank,
            total_ranks,
            MIN_DATA_AVG_LINEWIDTH,
            MAX_DATA_AVG_LINEWIDTH,
        )
        raw_label = f"Raw: {_format_value(current_val, is_lower_better)}"
        ax.plot(
            x_coords,
            valid_data,
            color=color,
            linewidth=raw_data_lw,
            label=raw_label,
            alpha=raw_data_alpha,
        )
        plotted_legend_items = True

        # Plot Rolling Averages
        for i, avg_window in enumerate(plotted_windows):
            avg_rank = len(plotted_windows) - 1 - i
            current_alpha = _interpolate_visual_property(
                avg_rank, total_ranks, MIN_ALPHA, MAX_ALPHA
            )
            current_avg_lw = _interpolate_visual_property(
                avg_rank,
                total_ranks,
                MIN_DATA_AVG_LINEWIDTH,
                MAX_DATA_AVG_LINEWIDTH,
            )
            weights = np.ones(avg_window) / avg_window
            rolling_avg = np.convolve(valid_data, weights, mode="valid")
            avg_x_coords = np.arange(avg_window - 1, n_points)
            linestyle = "-"
            if len(avg_x_coords) == len(rolling_avg):
                last_avg_val = rolling_avg[-1] if len(rolling_avg) > 0 else np.nan
                avg_label = (
                    f"Avg {avg_window}: {_format_value(last_avg_val, is_lower_better)}"
                )
                ax.plot(
                    avg_x_coords,
                    rolling_avg,
                    color=color,
                    linewidth=current_avg_lw,
                    alpha=current_alpha,
                    linestyle=linestyle,
                    label=avg_label,
                )
                plotted_legend_items = True

        # Plot Trend Line
        if trend_params is not None and n_points >= 2:
            slope, intercept = trend_params
            x_trend = np.array([0, n_points - 1])
            y_trend = slope * x_trend + intercept
            trend_label = f"Trend: {_format_slope(slope)}"
            ax.plot(
                x_trend,
                y_trend,
                color=TREND_LINE_COLOR,
                linestyle=TREND_LINE_STYLE,
                linewidth=TREND_LINE_WIDTH,
                alpha=TREND_LINE_ALPHA,
                label=trend_label,
                zorder=TREND_LINE_ZORDER,
            )
            plotted_legend_items = True

        ax.tick_params(axis="both", which="major")
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(
            True,
            linestyle=plt.rcParams["grid.linestyle"],
            alpha=plt.rcParams["grid.alpha"],
        )

        min_val_plot = np.min(valid_data)
        max_val_plot = np.max(valid_data)
        padding_factor = 0.1
        range_val = max_val_plot - min_val_plot
        if abs(range_val) < 1e-6:
            padding = (
                max(abs(max_val_plot * padding_factor), 0.5)
                if max_val_plot != 0
                else 0.5
            )
        else:
            padding = range_val * padding_factor
        padding = max(padding, 1e-6)
        ax.set_ylim(min_val_plot - padding, max_val_plot + padding)

        if y_log_scale and min_val_plot > 1e-9:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(min_val_plot * 0.9, 1e-9))
        else:
            ax.set_yscale("linear")

        if n_points > 1:
            ax.set_xlim(-0.02 * n_points, n_points - 1 + 0.02 * n_points)
        elif n_points == 1:
            ax.set_xlim(-0.5, 0.5)

        if n_points > 1000:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

            def format_func(value, tick_number):
                val_int = int(value)
                if val_int >= 1_000_000:
                    return f"{val_int/1_000_000:.1f}M"
                if val_int >= 1_000:
                    return f"{val_int/1_000:.0f}k"
                return f"{val_int}"

            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        elif n_points > 10:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))

        if plotted_legend_items:
            ax.legend(loc="best", fontsize=plt.rcParams["legend.fontsize"])

    except Exception as plot_err:
        print(f"ERROR during render_single_plot for '{label}': {plot_err}")
        traceback.print_exc()
        error_text_color = normalize_color_for_matplotlib(VisConfig.RED)
        ax.text(
            0.5,
            0.5,
            f"Plot Error\n({label})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color=error_text_color,
        )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)

    # Apply background tint and border based on trend
    bg_color_with_alpha = (*trend_indicator_color, TREND_BACKGROUND_ALPHA)
    ax.patch.set_facecolor(bg_color_with_alpha)
    ax.patch.set_edgecolor(trend_indicator_color)
    ax.patch.set_linewidth(trend_indicator_lw)


File: ui\renderer.py
# File: ui/renderer.py
import pygame
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque

from config import VisConfig, EnvConfig, TensorBoardConfig, DemoConfig
from environment.game_state import GameState
from .panels import LeftPanelRenderer, GameAreaRenderer
from .overlays import OverlayRenderer
from .tooltips import TooltipRenderer
from .plotter import Plotter
from .demo_renderer import DemoRenderer


class UIRenderer:
    """Orchestrates rendering of all UI components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = Plotter()
        self.left_panel = LeftPanelRenderer(screen, vis_config, self.plotter)
        self.game_area = GameAreaRenderer(screen, vis_config)
        self.overlays = OverlayRenderer(screen, vis_config)
        self.tooltips = TooltipRenderer(screen, vis_config)
        self.demo_config = DemoConfig()
        self.demo_renderer = DemoRenderer(
            screen, vis_config, self.demo_config, self.game_area
        )
        self.last_plot_update_time = 0

    def check_hover(self, mouse_pos: Tuple[int, int], app_state: str):
        """Passes hover check to the tooltip renderer."""
        if app_state == "MainMenu":
            self.tooltips.update_rects_and_texts(
                self.left_panel.get_stat_rects(), self.left_panel.get_tooltip_texts()
            )
            self.tooltips.check_hover(mouse_pos)
        else:
            self.tooltips.hovered_stat_key = None
            self.tooltips.stat_rects.clear()

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0

    def render_all(
        self,
        app_state: str,
        is_training_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_confirmation_active: bool,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        demo_env: Optional[GameState] = None,
        update_progress: float = 0.0,  # Added update_progress parameter
    ):
        """Renders UI based on the application state."""
        try:
            if app_state == "MainMenu":
                self._render_main_menu(
                    is_training_running=is_training_running,
                    status=status,
                    stats_summary=stats_summary,
                    envs=envs,
                    num_envs=num_envs,
                    env_config=env_config,
                    cleanup_message=cleanup_message,
                    last_cleanup_message_time=last_cleanup_message_time,
                    tensorboard_log_dir=tensorboard_log_dir,
                    plot_data=plot_data,
                    update_progress=update_progress,  # Pass progress
                )
            elif app_state == "Playing":
                if demo_env:
                    self.demo_renderer.render(demo_env, env_config)
                else:
                    print("Error: Attempting to render demo mode without demo_env.")
                    self._render_simple_message("Demo Env Error!", VisConfig.RED)
            elif app_state == "Initializing":
                self._render_initializing_screen(status)
            elif app_state == "Error":
                self._render_error_screen(status)

            # Overlays are rendered on top
            if cleanup_confirmation_active and app_state != "Error":
                self.overlays.render_cleanup_confirmation()
            elif not cleanup_confirmation_active:
                self.overlays.render_status_message(
                    cleanup_message, last_cleanup_message_time
                )

            # Render tooltips last
            if app_state == "MainMenu" and not cleanup_confirmation_active:
                self.tooltips.render_tooltip()

            pygame.display.flip()

        except pygame.error as e:
            print(f"Pygame rendering error in render_all: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"Unexpected critical rendering error in render_all: {e}")
            traceback.print_exc()
            try:
                self._render_simple_message("Critical Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception:
                pass

    def _render_main_menu(
        self,
        is_training_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        update_progress: float,  # Added update_progress parameter
    ):
        """Renders the main training dashboard view."""
        self.screen.fill(VisConfig.BLACK)

        # Render Left Panel (Pass update_progress)
        self.left_panel.render(
            is_training_running=is_training_running,
            status=status,
            stats_summary=stats_summary,
            tensorboard_log_dir=tensorboard_log_dir,
            plot_data=plot_data,
            app_state="MainMenu",
            update_progress=update_progress,  # Pass progress
        )

        # Render Game Area
        self.game_area.render(envs, num_envs, env_config)

    def _render_initializing_screen(
        self, status_message: str = "Initializing RL Components..."
    ):
        """Renders the initializing screen with a status message."""
        self._render_simple_message(status_message, VisConfig.WHITE)

    def _render_error_screen(self, status_message: str):
        """Renders the error screen."""
        try:
            self.screen.fill((40, 0, 0))
            font_title = pygame.font.SysFont(None, 70)
            font_msg = pygame.font.SysFont(None, 30)

            title_surf = font_title.render("APPLICATION ERROR", True, VisConfig.RED)
            title_rect = title_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() // 3)
            )

            msg_surf = font_msg.render(
                f"Status: {status_message}", True, VisConfig.YELLOW
            )
            msg_rect = msg_surf.get_rect(
                center=(self.screen.get_width() // 2, title_rect.bottom + 30)
            )

            exit_surf = font_msg.render(
                "Press ESC or close window to exit.", True, VisConfig.WHITE
            )
            exit_rect = exit_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() * 0.8)
            )

            self.screen.blit(title_surf, title_rect)
            self.screen.blit(msg_surf, msg_rect)
            self.screen.blit(exit_surf, exit_rect)

        except Exception as e:
            print(f"Error rendering error screen: {e}")
            self._render_simple_message(f"Error State: {status_message}", VisConfig.RED)

    def _render_simple_message(self, message: str, color: Tuple[int, int, int]):
        """Renders a simple centered text message."""
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)
            text_surf = font.render(message, True, color)
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
        except Exception as e:
            print(f"Error rendering simple message '{message}': {e}")


File: ui\tooltips.py
# File: ui/tooltips.py
# (No significant changes needed, this file was already focused)
import pygame
from typing import Tuple, Dict, Optional
from config import VisConfig


class TooltipRenderer:
    """Handles rendering of tooltips when hovering over specific UI elements."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.font_tooltip = self._init_font()
        self.hovered_stat_key: Optional[str] = None
        self.stat_rects: Dict[str, pygame.Rect] = {}  # Rects to check for hover
        self.tooltip_texts: Dict[str, str] = {}  # Text corresponding to each rect key

    def _init_font(self):
        """Initializes the font used for tooltips."""
        try:
            # Smaller font for tooltips
            return pygame.font.SysFont(None, 18)
        except Exception as e:
            print(f"Warning: SysFont error for tooltip font: {e}. Using default.")
            return pygame.font.Font(None, 18)

    def check_hover(self, mouse_pos: Tuple[int, int]):
        """Checks if the mouse is hovering over any registered stat rect."""
        self.hovered_stat_key = None
        # Iterate in reverse order of drawing to prioritize top elements
        for key, rect in reversed(self.stat_rects.items()):
            # Ensure rect is valid before checking collision
            if (
                rect
                and rect.width > 0
                and rect.height > 0
                and rect.collidepoint(mouse_pos)
            ):
                self.hovered_stat_key = key
                return  # Found one, stop checking

    def render_tooltip(self):
        """Renders the tooltip if a stat element is being hovered over. Does not flip display."""
        if not self.hovered_stat_key or self.hovered_stat_key not in self.tooltip_texts:
            return  # No active hover or no text defined for this key

        tooltip_text = self.tooltip_texts[self.hovered_stat_key]
        mouse_pos = pygame.mouse.get_pos()

        # --- Text Wrapping Logic ---
        lines = []
        max_width = 300  # Max tooltip width in pixels
        words = tooltip_text.split(" ")
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}" if current_line else word
            try:
                test_surf = self.font_tooltip.render(test_line, True, VisConfig.BLACK)
                if test_surf.get_width() <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            except Exception as e:
                print(f"Warning: Font render error during tooltip wrap: {e}")
                lines.append(current_line)  # Add what we had
                current_line = word  # Start new line
        lines.append(current_line)  # Add the last line

        # --- Render Wrapped Text ---
        line_surfs = []
        total_height = 0
        max_line_width = 0
        try:
            for line in lines:
                if not line:
                    continue  # Skip empty lines
                surf = self.font_tooltip.render(line, True, VisConfig.BLACK)
                line_surfs.append(surf)
                total_height += surf.get_height()
                max_line_width = max(max_line_width, surf.get_width())
        except Exception as e:
            print(f"Warning: Font render error creating tooltip surfaces: {e}")
            return  # Cannot render tooltip

        if not line_surfs:
            return  # No valid lines to render

        # --- Calculate Tooltip Rect and Draw ---
        padding = 5
        tooltip_rect = pygame.Rect(
            mouse_pos[0] + 15,  # Offset from cursor x
            mouse_pos[1] + 10,  # Offset from cursor y
            max_line_width + padding * 2,
            total_height + padding * 2,
        )

        # Clamp tooltip rect to stay within screen bounds
        tooltip_rect.clamp_ip(self.screen.get_rect())

        try:
            # Draw background and border
            pygame.draw.rect(
                self.screen, VisConfig.YELLOW, tooltip_rect, border_radius=3
            )
            pygame.draw.rect(
                self.screen, VisConfig.BLACK, tooltip_rect, 1, border_radius=3
            )

            # Draw text lines onto the screen
            current_y = tooltip_rect.y + padding
            for surf in line_surfs:
                self.screen.blit(surf, (tooltip_rect.x + padding, current_y))
                current_y += surf.get_height()
        except Exception as e:
            print(f"Warning: Error drawing tooltip background/text: {e}")

    def update_rects_and_texts(
        self, rects: Dict[str, pygame.Rect], texts: Dict[str, str]
    ):
        """Updates the dictionaries used for hover detection and text lookup. Called by UIRenderer."""
        self.stat_rects = rects
        self.tooltip_texts = texts


File: ui\__init__.py
from .renderer import UIRenderer
from .input_handler import InputHandler

__all__ = ["UIRenderer", "InputHandler"]


File: ui\panels\game_area.py
# File: ui/panels/game_area.py
import pygame
import math
import traceback
from typing import List, Tuple
from config import VisConfig, EnvConfig
from environment.game_state import GameState
from environment.shape import Shape
from environment.triangle import Triangle
import numpy as np


class GameAreaRenderer:
    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()

    def _init_fonts(self):
        fonts = {}
        try:
            fonts["env_score"] = pygame.font.SysFont(None, 18)
            fonts["env_overlay"] = pygame.font.SysFont(None, 36)
            fonts["ui"] = pygame.font.SysFont(None, 24)
        except Exception as e:
            print(f"Warning: SysFont error: {e}. Using default.")
            fonts["env_score"] = pygame.font.Font(None, 18)
            fonts["env_overlay"] = pygame.font.Font(None, 36)
            fonts["ui"] = pygame.font.Font(None, 24)
        return fonts

    def render(self, envs: List[GameState], num_envs: int, env_config: EnvConfig):
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(300, self.vis_config.LEFT_PANEL_WIDTH))
        ga_rect = pygame.Rect(lp_width, 0, current_width - lp_width, current_height)

        if num_envs <= 0 or ga_rect.width <= 0 or ga_rect.height <= 0:
            return

        render_limit = self.vis_config.NUM_ENVS_TO_RENDER
        num_to_render = num_envs if render_limit <= 0 else min(num_envs, render_limit)
        if num_to_render <= 0:
            return

        cols_env, rows_env, cell_w, cell_h = self._calculate_grid_layout(
            ga_rect, num_to_render
        )

        min_cell_dim = 30
        if cell_w > min_cell_dim and cell_h > min_cell_dim:
            self._render_env_grid(
                envs,
                num_to_render,
                env_config,
                ga_rect,
                cols_env,
                rows_env,
                cell_w,
                cell_h,
            )
        else:
            self._render_too_small_message(ga_rect, cell_w, cell_h)

        if num_to_render < num_envs:
            self._render_render_limit_text(ga_rect, num_to_render, num_envs)

    def _calculate_grid_layout(
        self, ga_rect: pygame.Rect, num_to_render: int
    ) -> Tuple[int, int, int, int]:
        aspect_ratio = ga_rect.width / max(1, ga_rect.height)
        cols_env = max(1, int(math.sqrt(num_to_render * aspect_ratio)))
        rows_env = max(1, math.ceil(num_to_render / cols_env))
        total_spacing_w = (cols_env + 1) * self.vis_config.ENV_SPACING
        total_spacing_h = (rows_env + 1) * self.vis_config.ENV_SPACING
        cell_w = max(1, (ga_rect.width - total_spacing_w) // cols_env)
        cell_h = max(1, (ga_rect.height - total_spacing_h) // rows_env)
        return cols_env, rows_env, cell_w, cell_h

    def _render_env_grid(
        self, envs, num_to_render, env_config, ga_rect, cols, rows, cell_w, cell_h
    ):
        env_idx = 0
        for r in range(rows):
            for c in range(cols):
                if env_idx >= num_to_render:
                    break
                env_x = ga_rect.x + self.vis_config.ENV_SPACING * (c + 1) + c * cell_w
                env_y = ga_rect.y + self.vis_config.ENV_SPACING * (r + 1) + r * cell_h
                env_rect = pygame.Rect(env_x, env_y, cell_w, cell_h)

                clipped_env_rect = env_rect.clip(self.screen.get_rect())
                if clipped_env_rect.width <= 0 or clipped_env_rect.height <= 0:
                    env_idx += 1
                    continue

                try:
                    sub_surf = self.screen.subsurface(clipped_env_rect)
                    self._render_single_env(sub_surf, envs[env_idx], env_config)

                except ValueError as subsurface_error:
                    print(
                        f"Warning: Subsurface error env {env_idx} ({clipped_env_rect}): {subsurface_error}"
                    )
                    pygame.draw.rect(self.screen, (0, 0, 50), clipped_env_rect, 1)
                except Exception as e_render_env:
                    print(f"Error rendering env {env_idx}: {e_render_env}")
                    traceback.print_exc()
                    pygame.draw.rect(self.screen, (50, 0, 50), clipped_env_rect, 1)

                env_idx += 1

    def _render_single_env(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        cell_w = surf.get_width()
        cell_h = surf.get_height()
        if cell_w <= 0 or cell_h <= 0:
            return

        bg_color = VisConfig.GRAY
        if env.is_line_clearing():
            bg_color = VisConfig.LINE_CLEAR_FLASH_COLOR
        elif env.is_game_over_flashing():
            bg_color = VisConfig.GAME_OVER_FLASH_COLOR
        elif env.is_blinking():
            bg_color = VisConfig.YELLOW
        elif env.is_over():
            bg_color = VisConfig.DARK_RED
        elif env.is_frozen():
            bg_color = (30, 30, 100)
        surf.fill(bg_color)

        shape_area_height_ratio = 0.20
        grid_area_height = math.floor(cell_h * (1.0 - shape_area_height_ratio))
        shape_area_height = cell_h - grid_area_height
        shape_area_y = grid_area_height

        grid_surf = None
        shape_surf = None
        if grid_area_height > 0 and cell_w > 0:
            try:
                grid_rect = pygame.Rect(0, 0, cell_w, grid_area_height)
                grid_surf = surf.subsurface(grid_rect)
            except ValueError as e:
                print(f"Warning: Grid subsurface error ({grid_rect}): {e}")
                pygame.draw.rect(surf, VisConfig.RED, grid_rect, 1)

        if shape_area_height > 0 and cell_w > 0:
            try:
                shape_rect = pygame.Rect(0, shape_area_y, cell_w, shape_area_height)
                shape_surf = surf.subsurface(shape_rect)
                shape_surf.fill((35, 35, 35))
            except ValueError as e:
                print(f"Warning: Shape subsurface error ({shape_rect}): {e}")
                pygame.draw.rect(surf, VisConfig.RED, shape_rect, 1)

        if grid_surf:
            self._render_single_env_grid(grid_surf, env, env_config)

        if shape_surf:
            self._render_shape_previews(shape_surf, env)

        try:
            score_surf = self.fonts["env_score"].render(
                f"GS: {env.game_score} R: {env.score:.1f}",
                True,
                VisConfig.WHITE,
                (0, 0, 0, 180),
            )
            surf.blit(score_surf, (2, 2))
        except Exception as e:
            print(f"Error rendering score: {e}")

        if env.is_over():
            self._render_overlay_text(surf, "GAME OVER", VisConfig.RED)
        elif env.is_line_clearing():
            self._render_overlay_text(surf, "Line Clear!", VisConfig.BLUE)

    def _render_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        try:
            overlay_font = self.fonts["env_overlay"]
            text_surf = overlay_font.render(
                text,
                True,
                VisConfig.WHITE,
                (color[0] // 2, color[1] // 2, color[2] // 2, 220),
            )
            text_rect = text_surf.get_rect(center=surf.get_rect().center)
            surf.blit(text_surf, text_rect)
        except Exception as e:
            print(f"Error rendering overlay text '{text}': {e}")

    def _render_single_env_grid(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        try:
            padding = self.vis_config.ENV_GRID_PADDING
            drawable_w = max(1, surf.get_width() - 2 * padding)
            drawable_h = max(1, surf.get_height() - 2 * padding)

            grid_rows = env_config.ROWS
            grid_cols_effective_width = env_config.COLS * 0.75 + 0.25

            if grid_rows <= 0 or grid_cols_effective_width <= 0:
                return

            scale_w_based = drawable_w / grid_cols_effective_width
            scale_h_based = drawable_h / grid_rows
            final_scale = min(scale_w_based, scale_h_based)
            if final_scale <= 0:
                return

            final_grid_pixel_w = grid_cols_effective_width * final_scale
            final_grid_pixel_h = grid_rows * final_scale
            tri_cell_h = max(1, final_scale)
            tri_cell_w = max(1, final_scale)

            grid_ox = padding + (drawable_w - final_grid_pixel_w) / 2
            grid_oy = padding + (drawable_h - final_grid_pixel_h) / 2

            is_highlighting = env.is_highlighting_cleared()
            cleared_coords = (
                set(env.get_cleared_triangle_coords()) if is_highlighting else set()
            )
            highlight_color = self.vis_config.LINE_CLEAR_HIGHLIGHT_COLOR

            if hasattr(env, "grid") and hasattr(env.grid, "triangles"):
                for r in range(env.grid.rows):
                    for c in range(env.grid.cols):
                        if not (
                            0 <= r < len(env.grid.triangles)
                            and 0 <= c < len(env.grid.triangles[r])
                        ):
                            continue
                        t = env.grid.triangles[r][c]
                        if not t.is_death:
                            if not hasattr(t, "get_points"):
                                continue
                            try:
                                pts = t.get_points(
                                    ox=grid_ox,
                                    oy=grid_oy,
                                    cw=int(tri_cell_w),
                                    ch=int(tri_cell_h),
                                )
                                if is_highlighting and (r, c) in cleared_coords:
                                    color = highlight_color
                                elif t.is_occupied:
                                    color = t.color if t.color else VisConfig.RED
                                else:
                                    color = VisConfig.LIGHTG

                                pygame.draw.polygon(surf, color, pts)
                                pygame.draw.polygon(surf, VisConfig.GRAY, pts, 1)
                            except Exception as e_render:
                                pass
            else:
                pygame.draw.rect(surf, VisConfig.RED, surf.get_rect(), 2)
                err_txt = self.fonts["ui"].render(
                    "Invalid Grid Data", True, VisConfig.RED
                )
                surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))

        except Exception as e:
            print(f"Unexpected Render Error in _render_single_env_grid: {e}")
            traceback.print_exc()
            pygame.draw.rect(surf, VisConfig.RED, surf.get_rect(), 2)

    def _render_shape_previews(self, surf: pygame.Surface, env: GameState):
        available_shapes = env.get_shapes()
        if not available_shapes:
            return

        surf_w = surf.get_width()
        surf_h = surf.get_height()
        if surf_w <= 0 or surf_h <= 0:
            return

        num_shapes = len(available_shapes)
        padding = 4
        total_padding_needed = (num_shapes + 1) * padding
        available_width_for_shapes = surf_w - total_padding_needed

        if available_width_for_shapes <= 0:
            return

        width_per_shape = available_width_for_shapes / num_shapes
        height_limit = surf_h - 2 * padding
        preview_dim = max(5, min(width_per_shape, height_limit))

        start_x = (
            padding
            + (surf_w - (num_shapes * preview_dim + (num_shapes - 1) * padding)) / 2
        )
        start_y = padding + (surf_h - preview_dim) / 2

        current_x = start_x
        for shape in available_shapes:
            preview_rect = pygame.Rect(current_x, start_y, preview_dim, preview_dim)
            if preview_rect.right > surf_w - padding:
                break

            # --- ADDED CHECK: Skip rendering if shape is None ---
            if shape is None:
                # Optionally draw an empty box or just skip
                pygame.draw.rect(surf, (50, 50, 50), preview_rect, 1, border_radius=2)
                current_x += preview_dim + padding
                continue
            # --- END ADDED CHECK ---

            try:
                temp_shape_surf = pygame.Surface(
                    (preview_dim, preview_dim), pygame.SRCALPHA
                )
                temp_shape_surf.fill((0, 0, 0, 0))

                min_r, min_c, max_r, max_c = shape.bbox()  # Now safe to call
                shape_h_cells = max(1, max_r - min_r + 1)
                shape_w_cells_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)

                scale_h = preview_dim / shape_h_cells
                scale_w = preview_dim / shape_w_cells_eff
                cell_size = max(1, min(scale_h, scale_w))

                self._render_single_shape(temp_shape_surf, shape, int(cell_size))

                surf.blit(temp_shape_surf, preview_rect.topleft)
                current_x += preview_dim + padding

            except Exception as e:
                print(f"Error rendering shape preview: {e}")  # Keep error log
                pygame.draw.rect(surf, VisConfig.RED, preview_rect, 1)
                current_x += preview_dim + padding

    def _render_single_shape(self, surf: pygame.Surface, shape: Shape, cell_size: int):
        if not shape or not shape.triangles or cell_size <= 0:
            return
        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h_cells = max_r - min_r + 1
        shape_w_cells_eff = (max_c - min_c + 1) * 0.75 + 0.25
        if shape_w_cells_eff <= 0 or shape_h_cells <= 0:
            return

        total_w_pixels = shape_w_cells_eff * cell_size
        total_h_pixels = shape_h_cells * cell_size

        offset_x = (surf.get_width() - total_w_pixels) / 2 - min_c * (cell_size * 0.75)
        offset_y = (surf.get_height() - total_h_pixels) / 2 - min_r * cell_size

        for dr, dc, up in shape.triangles:
            tri = Triangle(row=dr, col=dc, is_up=up)
            try:
                pts = tri.get_points(
                    ox=offset_x, oy=offset_y, cw=cell_size, ch=cell_size
                )
                pygame.draw.polygon(surf, shape.color, pts)
            except Exception as e:
                print(f"Warning: Error rendering shape preview tri ({dr},{dc}): {e}")

    def _render_too_small_message(self, ga_rect: pygame.Rect, cell_w: int, cell_h: int):
        try:
            err_surf = self.fonts["ui"].render(
                f"Envs Too Small ({cell_w}x{cell_h})", True, VisConfig.GRAY
            )
            self.screen.blit(err_surf, err_surf.get_rect(center=ga_rect.center))
        except Exception as e:
            print(f"Error rendering 'too small' message: {e}")

    def _render_render_limit_text(
        self, ga_rect: pygame.Rect, num_rendered: int, num_total: int
    ):
        try:
            info_surf = self.fonts["ui"].render(
                f"Rendering {num_rendered}/{num_total} Envs",
                True,
                VisConfig.YELLOW,
                VisConfig.BLACK,
            )
            self.screen.blit(
                info_surf,
                info_surf.get_rect(bottomright=(ga_rect.right - 5, ga_rect.bottom - 5)),
            )
        except Exception as e:
            print(f"Error rendering limit text: {e}")


File: ui\panels\left_panel.py
# File: ui/panels/left_panel.py
import pygame
import os
import time
from typing import Dict, Any, Optional, Deque, Tuple

from config import (
    VisConfig,
    StatsConfig,
    PPOConfig,
    RNNConfig,
    DEVICE,
    TensorBoardConfig,
)
from config.general import TOTAL_TRAINING_STEPS
from ui.plotter import Plotter

from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    TBStatusRenderer,
    PlotAreaRenderer,
)

TOOLTIP_TEXTS = {
    "Status": "Current application state: Ready, Collecting Experience, Updating Agent, Confirm Cleanup, Cleaning, or Error.",
    "Run Button": "Click to Start/Stop training run (or press 'P').",
    "Cleanup Button": "Click to DELETE agent ckpt for CURRENT run ONLY, then re-init.",
    "Play Demo Button": "Click to enter interactive play mode.",
    "Device": f"Computation device detected ({DEVICE.type.upper()}).",
    "Network": f"Actor-Critic (CNN+MLP Fusion -> Optional LSTM:{RNNConfig.USE_RNN})",
    "TensorBoard Status": "Indicates TB logging status and log directory.",
    # Add tooltips for compact status line elements if desired
    "Steps Info": "Global Steps / Total Planned Steps",
    "Episodes Info": "Total Completed Episodes",
    "SPS Info": "Steps Per Second (Collection + Update Avg)",
    "Update Progress": "Progress of the current agent neural network update cycle.",
}


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.stat_rects: Dict[str, pygame.Rect] = {}

        self.button_status_renderer = ButtonStatusRenderer(self.screen, self.fonts)
        self.info_text_renderer = InfoTextRenderer(self.screen, self.fonts)
        self.tb_status_renderer = TBStatusRenderer(self.screen, self.fonts)
        self.plot_area_renderer = PlotAreaRenderer(
            self.screen, self.fonts, self.plotter
        )

    def _init_fonts(self):
        fonts = {}
        font_configs = {
            "ui": 24,
            "status": 28,
            "logdir": 16,
            "plot_placeholder": 20,
            "notification_label": 16,
            "plot_title_values": 8,
            "progress_bar": 14,  # Font for progress bar percentage
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
        is_training_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        app_state: str,
        update_progress: float,  # Added update_progress
    ):
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(300, self.vis_config.LEFT_PANEL_WIDTH))
        lp_rect = pygame.Rect(0, 0, lp_width, current_height)

        status_color_map = {
            "Ready": (30, 30, 30),
            "Collecting Experience": (30, 40, 30),  # Greenish tint
            "Updating Agent": (30, 30, 50),  # Bluish tint
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Initializing": (40, 40, 40),
        }
        bg_color = status_color_map.get(status, (30, 30, 30))
        pygame.draw.rect(self.screen, bg_color, lp_rect)
        self.stat_rects.clear()

        current_y = 10

        # Render Buttons and Compact Status Block (Pass update_progress)
        next_y, rects_bs = self.button_status_renderer.render(
            y_start=current_y,
            panel_width=lp_width,
            app_state=app_state,
            is_training_running=is_training_running,
            status=status,
            stats_summary=stats_summary,
            update_progress=update_progress,  # Pass progress
        )
        self.stat_rects.update(rects_bs)
        current_y = next_y

        # Render Info Text Block
        next_y, rects_info = self.info_text_renderer.render(
            current_y + 5, stats_summary, lp_width
        )
        self.stat_rects.update(rects_info)
        current_y = next_y

        # Render TensorBoard Status
        next_y, rects_tb = self.tb_status_renderer.render(
            current_y + 10, tensorboard_log_dir, lp_width
        )
        self.stat_rects.update(rects_tb)
        current_y = next_y

        # Render Plot Area
        self.plot_area_renderer.render(
            current_y + 15, lp_width, current_height, plot_data, status
        )

    def get_stat_rects(self) -> Dict[str, pygame.Rect]:
        return self.stat_rects.copy()

    def get_tooltip_texts(self) -> Dict[str, str]:
        return TOOLTIP_TEXTS


File: ui\panels\__init__.py
from .left_panel import LeftPanelRenderer
from .game_area import GameAreaRenderer

__all__ = ["LeftPanelRenderer", "GameAreaRenderer"]


File: ui\panels\left_panel_components\button_status_renderer.py
# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
from typing import Dict, Tuple, Optional, Any
from config import VisConfig, TOTAL_TRAINING_STEPS


class ButtonStatusRenderer:
    """Renders the top buttons, compact status block, and update progress bar."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("status", pygame.font.Font(None, 28))
        self.status_label_font = fonts.get(
            "notification_label", pygame.font.Font(None, 16)
        )
        self.progress_font = fonts.get("progress_bar", pygame.font.Font(None, 14))

    def _draw_button(self, rect: pygame.Rect, text: str, color: Tuple[int, int, int]):
        """Helper to draw a single button."""
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        ui_font = self.fonts.get("ui")
        if ui_font:
            lbl_surf = ui_font.render(text, True, VisConfig.WHITE)
            self.screen.blit(lbl_surf, lbl_surf.get_rect(center=rect.center))
        else:
            pygame.draw.line(
                self.screen, VisConfig.RED, rect.topleft, rect.bottomright, 2
            )
            pygame.draw.line(
                self.screen, VisConfig.RED, rect.topright, rect.bottomleft, 2
            )

    def _render_compact_status(
        self, y_start: int, panel_width: int, status: str, stats_summary: Dict[str, Any]
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the compact status block."""
        stat_rects: Dict[str, pygame.Rect] = {}
        x_margin = 10
        line_height = self.status_font.get_linesize()
        label_line_height = self.status_label_font.get_linesize()
        current_y = y_start

        # Line 1: Status
        status_text = f"Status: {status}"
        status_color = VisConfig.YELLOW
        if status == "Error":
            status_color = VisConfig.RED
        elif status == "Collecting Experience":
            status_color = VisConfig.GOOGLE_COLORS[0]  # Green
        elif status == "Updating Agent":
            status_color = VisConfig.GOOGLE_COLORS[2]  # Blue
        elif status == "Ready":
            status_color = VisConfig.WHITE

        status_surf = self.status_font.render(status_text, True, status_color)
        status_rect = status_surf.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(status_surf, status_rect)
        stat_rects["Status"] = status_rect
        current_y += line_height

        # Line 2: Steps | Episodes | SPS
        global_step = stats_summary.get("global_step", 0)
        total_episodes = stats_summary.get("total_episodes", 0)
        sps = stats_summary.get("steps_per_second", 0.0)

        steps_str = f"{global_step/1e6:.2f}M/{TOTAL_TRAINING_STEPS/1e6:.1f}M Steps"
        eps_str = f"{total_episodes} Eps"
        sps_str = f"{sps:.0f} SPS"

        line2_text = f"{steps_str}  |  {eps_str}  |  {sps_str}"
        line2_surf = self.status_label_font.render(line2_text, True, VisConfig.LIGHTG)
        line2_rect = line2_surf.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(line2_surf, line2_rect)

        # Add individual rects for tooltips on line 2 elements
        steps_surf = self.status_label_font.render(steps_str, True, VisConfig.LIGHTG)
        eps_surf = self.status_label_font.render(eps_str, True, VisConfig.LIGHTG)
        sps_surf = self.status_label_font.render(sps_str, True, VisConfig.LIGHTG)

        steps_rect = steps_surf.get_rect(topleft=(x_margin, current_y))
        eps_rect = eps_surf.get_rect(
            midleft=(steps_rect.right + 10, steps_rect.centery)
        )
        sps_rect = sps_surf.get_rect(midleft=(eps_rect.right + 10, eps_rect.centery))

        stat_rects["Steps Info"] = steps_rect
        stat_rects["Episodes Info"] = eps_rect
        stat_rects["SPS Info"] = sps_rect

        current_y += label_line_height + 2

        return current_y, stat_rects

    def _render_progress_bar(
        self, y_start: int, panel_width: int, progress: float
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the agent update progress bar."""
        stat_rects: Dict[str, pygame.Rect] = {}
        x_margin = 10
        bar_height = 18
        bar_width = panel_width - 2 * x_margin
        current_y = y_start

        if bar_width <= 0:
            return current_y, stat_rects

        # Background
        bg_rect = pygame.Rect(x_margin, current_y, bar_width, bar_height)
        pygame.draw.rect(
            self.screen, (60, 60, 80), bg_rect, border_radius=3
        )  # Darker blue bg

        # Foreground (Progress)
        progress_width = int(bar_width * progress)
        if progress_width > 0:
            fg_rect = pygame.Rect(x_margin, current_y, progress_width, bar_height)
            pygame.draw.rect(
                self.screen, VisConfig.GOOGLE_COLORS[2], fg_rect, border_radius=3
            )  # Blue progress

        # Border
        pygame.draw.rect(self.screen, VisConfig.LIGHTG, bg_rect, 1, border_radius=3)

        # Percentage Text
        if self.progress_font:
            progress_text = f"{progress:.0%}"
            text_surf = self.progress_font.render(progress_text, True, VisConfig.WHITE)
            text_rect = text_surf.get_rect(center=bg_rect.center)
            self.screen.blit(text_surf, text_rect)

        stat_rects["Update Progress"] = bg_rect  # Tooltip for the whole bar
        current_y += bar_height + 5  # Add padding below bar

        return current_y, stat_rects

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_training_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        update_progress: float,  # Added update_progress
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders buttons, status, and progress bar. Returns next_y, stat_rects."""
        stat_rects: Dict[str, pygame.Rect] = {}
        next_y = y_start

        # Render Buttons (only in MainMenu)
        if app_state == "MainMenu":
            button_h = 40
            button_y = y_start
            run_btn_w = 100
            cleanup_btn_w = 160
            demo_btn_w = 120
            spacing = 10

            run_btn_rect = pygame.Rect(spacing, button_y, run_btn_w, button_h)
            cleanup_btn_rect = pygame.Rect(
                run_btn_rect.right + spacing, button_y, cleanup_btn_w, button_h
            )
            demo_btn_rect = pygame.Rect(
                cleanup_btn_rect.right + spacing, button_y, demo_btn_w, button_h
            )

            run_button_text = "Run"
            run_button_color = (70, 70, 70)  # Default gray
            if is_training_running:
                run_button_text = "Stop"
                if status == "Collecting Experience":
                    run_button_color = (40, 80, 40)  # Green
                elif status == "Updating Agent":
                    run_button_color = (40, 40, 80)  # Blue
                else:
                    run_button_color = (80, 80, 40)  # Yellowish if other training state
            elif status == "Ready":
                run_button_color = (40, 40, 80)  # Ready blue

            self._draw_button(run_btn_rect, run_button_text, run_button_color)
            self._draw_button(cleanup_btn_rect, "Cleanup This Run", (100, 40, 40))
            self._draw_button(demo_btn_rect, "Play Demo", (40, 100, 40))

            stat_rects["Run Button"] = run_btn_rect
            stat_rects["Cleanup Button"] = cleanup_btn_rect
            stat_rects["Play Demo Button"] = demo_btn_rect

            next_y = run_btn_rect.bottom + 10

        # Render Compact Status Block
        status_y = next_y if app_state == "MainMenu" else y_start
        next_y, status_rects = self._render_compact_status(
            status_y, panel_width, status, stats_summary
        )
        stat_rects.update(status_rects)

        # Render Progress Bar (only if updating)
        if status == "Updating Agent" and update_progress >= 0:
            next_y, progress_rects = self._render_progress_bar(
                next_y, panel_width, update_progress
            )
            stat_rects.update(progress_rects)

        return next_y, stat_rects


File: ui\panels\left_panel_components\info_text_renderer.py
# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
from typing import Dict, Any, Tuple
from config import (
    VisConfig,
    StatsConfig,
    PPOConfig,
    RNNConfig,
    DEVICE,
    TOTAL_TRAINING_STEPS,
)


class InfoTextRenderer:
    """Renders essential non-plotted information text."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],  # Keep stats_summary for potential future use
        panel_width: int,
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the info text block. Returns next_y and stat_rects."""
        stat_rects: Dict[str, pygame.Rect] = {}
        ui_font = self.fonts.get("ui")
        if not ui_font:
            return y_start, stat_rects  # Return 0 height if no font

        line_height = ui_font.get_linesize()

        # --- Kept essential, non-plotted info ---
        info_lines = [
            ("Device", f"{DEVICE.type.upper()}"),
            (
                "Network",
                f"Actor-Critic (CNN+MLP->LSTM:{RNNConfig.USE_RNN})",
            ),
            # Add any other essential non-plotted info here if needed
        ]
        # --- Removed: Global Steps, Episodes, SPS (moved to status) ---
        # --- Removed: Policy Loss, Value Loss, Entropy, LR (plotted) ---

        last_y = y_start
        x_pos_key, x_pos_val_offset = 10, 5

        # Add a small gap before this section
        current_y = y_start + 5

        for idx, (key, value_str) in enumerate(info_lines):
            line_y = current_y + idx * line_height
            try:
                key_surf = ui_font.render(f"{key}:", True, VisConfig.LIGHTG)
                key_rect = key_surf.get_rect(topleft=(x_pos_key, line_y))
                self.screen.blit(key_surf, key_rect)

                value_surf = ui_font.render(f"{value_str}", True, VisConfig.WHITE)
                value_rect = value_surf.get_rect(
                    topleft=(key_rect.right + x_pos_val_offset, line_y)
                )

                # Simple clipping for value text
                clip_width = max(0, panel_width - value_rect.left - 10)
                if value_rect.width > clip_width:
                    self.screen.blit(
                        value_surf,
                        value_rect,
                        area=pygame.Rect(0, 0, clip_width, value_rect.height),
                    )
                else:
                    self.screen.blit(value_surf, value_rect)

                # Store rect for tooltip
                combined_rect = key_rect.union(value_rect)
                combined_rect.width = min(
                    combined_rect.width, panel_width - x_pos_key - 10
                )
                stat_rects[key] = combined_rect
                last_y = combined_rect.bottom
            except Exception as e:
                print(f"Error rendering stat line '{key}': {e}")
                last_y = line_y + line_height

        # Return position below the last rendered line
        return last_y, stat_rects


File: ui\panels\left_panel_components\notification_renderer.py
# File: ui/panels/left_panel_components/notification_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple
from config import VisConfig, StatsConfig
import numpy as np


class NotificationRenderer:
    """Renders the notification area with best scores/loss."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts

    def _format_steps_ago(self, current_step: int, best_step: int) -> str:
        """Formats the difference in steps into a readable string."""
        if best_step <= 0 or current_step <= best_step:
            return "Now"
        diff = current_step - best_step
        if diff < 1000:
            return f"{diff} steps ago"
        elif diff < 1_000_000:
            return f"{diff / 1000:.1f}k steps ago"
        else:
            return f"{diff / 1_000_000:.1f}M steps ago"

    def _render_line(
        self,
        area_rect: pygame.Rect,
        y_pos: int,
        label: str,
        current_val: Any,
        prev_val: Any,
        best_step: int,
        val_format: str,
        current_step: int,
    ) -> pygame.Rect:
        """Renders a single line within the notification area."""
        label_font = self.fonts.get("notification_label")
        value_font = self.fonts.get("notification")
        if not label_font or not value_font:
            return pygame.Rect(0, y_pos, 0, 0)

        padding = 5
        label_color, value_color = VisConfig.LIGHTG, VisConfig.WHITE
        prev_color, time_color = VisConfig.GRAY, (180, 180, 100)

        label_surf = label_font.render(label, True, label_color)
        label_rect = label_surf.get_rect(topleft=(area_rect.left + padding, y_pos))
        self.screen.blit(label_surf, label_rect)
        current_x = label_rect.right + 4

        current_val_str = "N/A"
        # --- Convert to float *before* checking ---
        val_as_float: Optional[float] = None
        if isinstance(
            current_val, (int, float, np.number)
        ):  # Check against np.number too
            try:
                val_as_float = float(current_val)
            except (ValueError, TypeError):
                val_as_float = None  # Conversion failed

        # --- Use the converted float for checks and formatting ---
        if val_as_float is not None and np.isfinite(val_as_float):
            try:
                current_val_str = val_format.format(val_as_float)
            except (ValueError, TypeError) as fmt_err:
                current_val_str = "ErrFmt"

        val_surf = value_font.render(current_val_str, True, value_color)
        val_rect = val_surf.get_rect(topleft=(current_x, y_pos))
        self.screen.blit(val_surf, val_rect)
        current_x = val_rect.right + 4

        prev_val_str = "(N/A)"
        # --- Convert prev_val to float before checking ---
        prev_val_as_float: Optional[float] = None
        if isinstance(prev_val, (int, float, np.number)):
            try:
                prev_val_as_float = float(prev_val)
            except (ValueError, TypeError):
                prev_val_as_float = None

        if prev_val_as_float is not None and np.isfinite(prev_val_as_float):
            try:
                prev_val_str = f"({val_format.format(prev_val_as_float)})"
            except (ValueError, TypeError):
                prev_val_str = "(ErrFmt)"

        prev_surf = label_font.render(prev_val_str, True, prev_color)
        prev_rect = prev_surf.get_rect(topleft=(current_x, y_pos + 1))
        self.screen.blit(prev_surf, prev_rect)
        current_x = prev_rect.right + 6

        steps_ago_str = self._format_steps_ago(current_step, best_step)
        time_surf = label_font.render(steps_ago_str, True, time_color)
        time_rect = time_surf.get_rect(topleft=(current_x, y_pos + 1))

        available_width = area_rect.right - time_rect.left - padding
        clip_rect = pygame.Rect(0, 0, max(0, available_width), time_rect.height)
        if time_rect.width > available_width > 0:
            self.screen.blit(time_surf, time_rect, area=clip_rect)
        elif available_width > 0:
            self.screen.blit(time_surf, time_rect)

        union_rect = label_rect.union(val_rect).union(prev_rect).union(time_rect)
        union_rect.width = min(union_rect.width, area_rect.width - 2 * padding)
        return union_rect

    def render(
        self, area_rect: pygame.Rect, stats_summary: Dict[str, Any]
    ) -> Dict[str, pygame.Rect]:
        """Renders the notification content."""
        stat_rects: Dict[str, pygame.Rect] = {}
        pygame.draw.rect(self.screen, (45, 45, 45), area_rect, border_radius=3)
        pygame.draw.rect(self.screen, VisConfig.LIGHTG, area_rect, 1, border_radius=3)
        stat_rects["Notification Area"] = area_rect

        value_font = self.fonts.get("notification")
        if not value_font:
            return stat_rects

        padding = 5
        line_height = value_font.get_linesize()
        current_step = stats_summary.get("global_step", 0)
        y = area_rect.top + padding

        # --- Pass values from summary, _render_line handles conversion/check ---
        rect_rl = self._render_line(
            area_rect,
            y,
            "RL Score:",
            stats_summary.get("best_score", -float("inf")),
            stats_summary.get("previous_best_score", -float("inf")),
            stats_summary.get("best_score_step", 0),
            "{:.2f}",
            current_step,
        )
        stat_rects["Best RL Score Info"] = rect_rl.clip(area_rect)
        y += line_height

        rect_game = self._render_line(
            area_rect,
            y,
            "Game Score:",
            stats_summary.get("best_game_score", -float("inf")),
            stats_summary.get("previous_best_game_score", -float("inf")),
            stats_summary.get("best_game_score_step", 0),
            "{:.0f}",
            current_step,
        )
        stat_rects["Best Game Score Info"] = rect_game.clip(area_rect)
        y += line_height

        rect_loss = self._render_line(
            area_rect,
            y,
            "Loss:",
            stats_summary.get("best_loss", float("inf")),
            stats_summary.get("previous_best_loss", float("inf")),
            stats_summary.get("best_loss_step", 0),
            "{:.4f}",
            current_step,
        )
        stat_rects["Best Loss Info"] = rect_loss.clip(area_rect)

        return stat_rects


File: ui\panels\left_panel_components\plot_area_renderer.py
# File: ui/panels/left_panel_components/plot_area_renderer.py
import pygame
from typing import Dict, Deque, Optional
from config import VisConfig
from ui.plotter import Plotter  # Import Plotter


class PlotAreaRenderer:
    """Renders the plot area using a Plotter instance."""

    def __init__(
        self,
        screen: pygame.Surface,
        fonts: Dict[str, pygame.font.Font],
        plotter: Plotter,  # Pass plotter instance
    ):
        self.screen = screen
        self.fonts = fonts
        self.plotter = plotter

    def render(
        self,
        y_start: int,
        panel_width: int,
        screen_height: int,
        plot_data: Dict[str, Deque],
        status: str,
    ):
        """Renders the plot area."""
        plot_area_height = screen_height - y_start - 10
        plot_area_width = panel_width - 20

        if plot_area_width <= 50 or plot_area_height <= 50:
            return

        plot_surface = self.plotter.get_cached_or_updated_plot(
            plot_data, plot_area_width, plot_area_height
        )
        plot_area_rect = pygame.Rect(10, y_start, plot_area_width, plot_area_height)

        if plot_surface:
            self.screen.blit(plot_surface, plot_area_rect.topleft)
        else:
            # Render placeholder
            pygame.draw.rect(self.screen, (40, 40, 40), plot_area_rect, 1)
            placeholder_text = "Waiting for data..."
            if status == "Buffering":
                placeholder_text = "Buffering... Waiting for plot data..."
            elif status == "Error":
                placeholder_text = "Plotting disabled due to error."
            elif not plot_data or not any(plot_data.values()):
                placeholder_text = "No plot data yet..."

            placeholder_font = self.fonts.get("plot_placeholder")
            if placeholder_font:
                placeholder_surf = placeholder_font.render(
                    placeholder_text, True, VisConfig.GRAY
                )
                placeholder_rect = placeholder_surf.get_rect(
                    center=plot_area_rect.center
                )
                blit_pos = (
                    max(plot_area_rect.left, placeholder_rect.left),
                    max(plot_area_rect.top, placeholder_rect.top),
                )
                clip_area_rect = plot_area_rect.clip(placeholder_rect)
                blit_area = clip_area_rect.move(
                    -placeholder_rect.left, -placeholder_rect.top
                )
                if blit_area.width > 0 and blit_area.height > 0:
                    self.screen.blit(placeholder_surf, blit_pos, area=blit_area)
            else:  # Fallback cross
                pygame.draw.line(
                    self.screen,
                    VisConfig.GRAY,
                    plot_area_rect.topleft,
                    plot_area_rect.bottomright,
                )
                pygame.draw.line(
                    self.screen,
                    VisConfig.GRAY,
                    plot_area_rect.topright,
                    plot_area_rect.bottomleft,
                )


File: ui\panels\left_panel_components\tb_status_renderer.py
# File: ui/panels/left_panel_components/tb_status_renderer.py
import pygame
import os
from typing import Dict, Optional, Tuple
from config import VisConfig, TensorBoardConfig


class TBStatusRenderer:
    """Renders the TensorBoard status line."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts

    def _shorten_path(self, path: str, max_chars: int) -> str:
        """Attempts to shorten a path string for display."""
        if len(path) <= max_chars:
            return path
        try:
            rel_path = os.path.relpath(path)
        except ValueError:  # Handle different drives on Windows
            rel_path = path
        if len(rel_path) <= max_chars:
            return rel_path

        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            short_path = os.path.join("...", *parts[-2:])
            if len(short_path) <= max_chars:
                return short_path
        # Fallback: Ellipsis + end of basename
        basename = os.path.basename(path)
        return (
            "..." + basename[-(max_chars - 3) :]
            if len(basename) > max_chars - 3
            else basename
        )

    def render(
        self, y_start: int, log_dir: Optional[str], panel_width: int
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the TB status. Returns next_y and stat_rects."""
        stat_rects: Dict[str, pygame.Rect] = {}
        ui_font = self.fonts.get("ui")
        logdir_font = self.fonts.get("logdir")
        if not ui_font or not logdir_font:
            return y_start + 30, stat_rects

        tb_active = (
            TensorBoardConfig.LOG_HISTOGRAMS
            or TensorBoardConfig.LOG_IMAGES
            or TensorBoardConfig.LOG_SHAPE_PLACEMENT_Q_VALUES
        )
        tb_color = VisConfig.GOOGLE_COLORS[0] if tb_active else VisConfig.GRAY
        tb_text = f"TensorBoard: {'Logging Active' if tb_active else 'Logging Minimal'}"

        tb_surf = ui_font.render(tb_text, True, tb_color)
        tb_rect = tb_surf.get_rect(topleft=(10, y_start))
        self.screen.blit(tb_surf, tb_rect)
        stat_rects["TensorBoard Status"] = tb_rect
        last_y = tb_rect.bottom

        if log_dir:
            try:
                panel_char_width = max(
                    10, panel_width // max(1, logdir_font.size("A")[0])
                )
                short_log_dir = self._shorten_path(log_dir, panel_char_width)
            except Exception:
                short_log_dir = os.path.basename(log_dir)

            dir_surf = logdir_font.render(
                f"Log Dir: {short_log_dir}", True, VisConfig.LIGHTG
            )
            dir_rect = dir_surf.get_rect(topleft=(10, tb_rect.bottom + 2))

            clip_width = max(0, panel_width - dir_rect.left - 10)
            if dir_rect.width > clip_width:
                self.screen.blit(
                    dir_surf,
                    dir_rect,
                    area=pygame.Rect(0, 0, clip_width, dir_rect.height),
                )
            else:
                self.screen.blit(dir_surf, dir_rect)

            combined_tb_rect = tb_rect.union(dir_rect)
            combined_tb_rect.width = min(
                combined_tb_rect.width, panel_width - 10 - combined_tb_rect.left
            )
            stat_rects["TensorBoard Status"] = combined_tb_rect
            last_y = dir_rect.bottom

        return last_y, stat_rects


File: ui\panels\left_panel_components\__init__.py
from .button_status_renderer import ButtonStatusRenderer
from .notification_renderer import NotificationRenderer
from .info_text_renderer import InfoTextRenderer
from .tb_status_renderer import TBStatusRenderer
from .plot_area_renderer import PlotAreaRenderer

__all__ = [
    "ButtonStatusRenderer",
    "NotificationRenderer",
    "InfoTextRenderer",
    "TBStatusRenderer",
    "PlotAreaRenderer",
]


File: utils\helpers.py
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

    # Check MPS first (for Macs) - This will be false on your PC
    if torch.backends.mps.is_available():
        device_str = "mps"
    # Check CUDA next (for NVIDIA GPUs) - This SHOULD become true
    elif torch.cuda.is_available():
        device_str = "cuda"
    # Fallback to CPU
    else:
        device_str = "cpu"

    print(f"Using device: {device_str.upper()}")
    if device_str == "cuda":
        # This line should execute once fixed
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    elif device_str == "mps":
        print("MPS device found on MacOS.") # Won't execute on PC
    else:
        print("No CUDA or MPS device found, falling back to CPU.") # This is what's happening now

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
            if arr.dtype == np.object_:  # Indicates ragged array
                raise ValueError(
                    "Cannot convert ragged list/tuple to float32 numpy array."
                )
            return arr
        else:
            # Attempt conversion for single numbers or other types
            return np.array([data], dtype=np.float32)
    except (ValueError, TypeError, RuntimeError) as e:
        print(
            f"CRITICAL ERROR in ensure_numpy conversion: {e}. Input type: {type(data)}. Data (partial): {str(data)[:100]}"
        )
        raise ValueError(f"ensure_numpy failed: {e}") from e


def save_object(obj: Any, filepath: str):
    """Saves an arbitrary Python object to a file using cloudpickle."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            cloudpickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error saving object to {filepath}: {e}")
        raise e  # Re-raise after loggingcpu


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
        raise e  # Re-raise after logging

File: utils\init_checks.py
# File: utils/init_checks.py
import sys
import traceback
import numpy as np
from config import EnvConfig
from environment.game_state import GameState


def run_pre_checks() -> bool:
    """Performs basic checks on GameState and configuration compatibility."""
    print("--- Pre-Run Checks ---")
    try:
        print("Checking GameState and Configuration Compatibility...")
        env_config_instance = EnvConfig()

        gs_test = GameState()
        gs_test.reset()
        s_test_dict = gs_test.get_state()

        if not isinstance(s_test_dict, dict):
            raise TypeError(
                f"GameState.get_state() should return a dict, but got {type(s_test_dict)}"
            )
        print("GameState state type check PASSED (returned dict).")

        # Check grid shape
        if "grid" not in s_test_dict:
            raise KeyError("State dictionary missing 'grid' key.")
        grid_state = s_test_dict["grid"]
        expected_grid_shape = env_config_instance.GRID_STATE_SHAPE
        if not isinstance(grid_state, np.ndarray):
            raise TypeError(
                f"State 'grid' component should be numpy array, but got {type(grid_state)}"
            )
        if grid_state.shape != expected_grid_shape:
            raise ValueError(
                f"State 'grid' shape mismatch! GameState:{grid_state.shape}, EnvConfig:{expected_grid_shape}"
            )
        print(f"GameState 'grid' state shape check PASSED (Shape: {grid_state.shape}).")

        # Check 'shapes' component (features)
        if "shapes" not in s_test_dict:
            raise KeyError("State dictionary missing 'shapes' key.")
        shape_state = s_test_dict["shapes"]
        expected_shape_shape = (
            env_config_instance.NUM_SHAPE_SLOTS,
            env_config_instance.SHAPE_FEATURES_PER_SHAPE,
        )
        if not isinstance(shape_state, np.ndarray):
            raise TypeError(
                f"State 'shapes' component should be numpy array, but got {type(shape_state)}"
            )
        if shape_state.shape != expected_shape_shape:
            raise ValueError(
                f"State 'shapes' feature shape mismatch! GameState:{shape_state.shape}, EnvConfig:{expected_shape_shape}"
            )
        print(
            f"GameState 'shapes' feature shape check PASSED (Shape: {shape_state.shape})."
        )

        # Check 'shape_availability' component
        if "shape_availability" not in s_test_dict:
            raise KeyError("State dictionary missing 'shape_availability' key.")
        availability_state = s_test_dict["shape_availability"]
        expected_availability_shape = (env_config_instance.SHAPE_AVAILABILITY_DIM,)
        if not isinstance(availability_state, np.ndarray):
            raise TypeError(
                f"State 'shape_availability' component should be numpy array, but got {type(availability_state)}"
            )
        if availability_state.shape != expected_availability_shape:
            raise ValueError(
                f"State 'shape_availability' shape mismatch! GameState:{availability_state.shape}, EnvConfig:{expected_availability_shape}"
            )
        print(
            f"GameState 'shape_availability' state shape check PASSED (Shape: {availability_state.shape})."
        )

        # --- UPDATED: Check 'explicit_features' component (with new dimension) ---
        if "explicit_features" not in s_test_dict:
            raise KeyError("State dictionary missing 'explicit_features' key.")
        explicit_features_state = s_test_dict["explicit_features"]
        expected_explicit_features_shape = (
            env_config_instance.EXPLICIT_FEATURES_DIM,
        )  # Use config dim
        if not isinstance(explicit_features_state, np.ndarray):
            raise TypeError(
                f"State 'explicit_features' component should be numpy array, but got {type(explicit_features_state)}"
            )
        if explicit_features_state.shape != expected_explicit_features_shape:
            raise ValueError(
                f"State 'explicit_features' shape mismatch! GameState:{explicit_features_state.shape}, EnvConfig:{expected_explicit_features_shape}"
            )
        print(
            f"GameState 'explicit_features' state shape check PASSED (Shape: {explicit_features_state.shape})."
        )
        # --- END UPDATED ---

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
    except (ValueError, AttributeError, TypeError, KeyError) as e:
        print(f"FATAL ERROR during pre-run checks: {e}")
    except Exception as e:
        print(f"FATAL ERROR during GameState pre-check: {e}")
        traceback.print_exc()
    sys.exit(1)  # Exit if checks fail


File: utils\types.py
# File: utils/types.py
from typing import NamedTuple, Union, Tuple, List, Dict, Any, Optional
import numpy as np
import torch

StateType = Dict[str, np.ndarray]
ActionType = int
AgentStateDict = Dict[str, Any]


File: utils\__init__.py


File: visualization\__init__.py


