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
        self.is_running = False
        self.cleanup_confirmation_active = False
        self.last_cleanup_message_time = 0.0
        self.cleanup_message = ""
        self.status = "Initializing Components"

        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None
        self.envs: List[GameState] = []
        self.agent: Optional[PPOAgent] = None
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.trainer: Optional[Trainer] = None
        self.demo_env: Optional[GameState] = None

        # Initial component initialization (not a re-init)
        self._initialize_core_components(is_reinit=False)

        self.app_state = "MainMenu"
        self.status = "Ready"
        print("Initialization Complete. Ready.")
        print(f"--- tensorboard --logdir {os.path.abspath(BASE_LOG_DIR)} ---")

    # --- MODIFIED: Added is_reinit flag ---
    def _initialize_core_components(self, is_reinit: bool = False):
        """Initializes Renderer, RL components, Demo Env, and Input Handler."""
        try:
            # Renderer only needs init once
            if not is_reinit:
                self.renderer = UIRenderer(self.screen, self.vis_config)
                self.renderer.render_all(
                    app_state=self.app_state,
                    is_running=self.is_running,
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
                )
                pygame.time.delay(100)

            # Initialize RL components, passing the reinit flag
            self._initialize_rl_components(is_reinit=is_reinit)

            # Demo env and input handler only need init once
            if not is_reinit:
                self._initialize_demo_env()
                self.input_handler = InputHandler(
                    self.screen,
                    self.renderer,
                    self._toggle_running,
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

    # --- MODIFIED: Added is_reinit flag ---
    def _initialize_rl_components(self, is_reinit: bool = False):
        """Initializes RL components using helper functions (now for PPO)."""
        print(f"Initializing RL components (PPO)... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            # Environments and Agent always need re-initialization
            self.envs = initialize_envs(self.num_envs, self.env_config)
            self.agent = initialize_agent(
                self.model_config, self.ppo_config, self.rnn_config, self.env_config
            )

            # Stats recorder needs the reinit flag to avoid re-logging graph/hparams
            self.stats_recorder = initialize_stats_recorder(
                stats_config=self.stats_config,
                tb_config=self.tensorboard_config,
                config_dict=self.config_dict,
                agent=self.agent,
                env_config=self.env_config,
                rnn_config=self.rnn_config,
                is_reinit=is_reinit,  # Pass the flag here
            )
            if self.stats_recorder is None:
                raise RuntimeError("Stats Recorder initialization failed unexpectedly.")

            # Trainer always needs re-initialization
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

    # --- END MODIFIED ---

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
    def _toggle_running(self):
        """Starts or stops the PPO training run."""
        if self.app_state != "MainMenu":
            return
        self.is_running = not self.is_running
        print(f"PPO Run {'STARTED' if self.is_running else 'STOPPED'}")
        if not self.is_running:
            self._try_save_checkpoint()

    def _request_cleanup(self):
        if self.app_state != "MainMenu":
            return
        was_running = self.is_running
        self.is_running = False
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
        return False  # Signal to main loop to exit

    def _start_demo_mode(self):
        if self.demo_env is None:
            print("Cannot start demo mode: Demo environment failed to initialize.")
            return
        if self.app_state == "MainMenu":
            print("Entering Demo Mode...")
            self.is_running = False
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
                    # Indicate invalid placement attempt visually? Maybe later.
                    action_taken = True  # Still counts as input handled

            if self.demo_env.is_over():
                print("[Demo] Game Over! Press ESC to exit.")

    # --- Core Logic Methods ---
    def _cleanup_data(self):
        """Deletes current run's checkpoint and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.app_state = "Initializing"
        self.is_running = False
        self.status = "Cleaning"
        messages = []

        # Render cleaning screen
        if self.renderer:
            try:
                self.renderer.render_all(
                    app_state=self.app_state,
                    is_running=False,
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
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

        # Close existing components safely
        if self.trainer:
            print("[Cleanup] Running trainer cleanup...")
            try:
                self.trainer.cleanup(save_final=False)  # Don't save during cleanup
            except Exception as e:
                print(f"Error during trainer cleanup: {e}")
        if self.stats_recorder:
            print("[Cleanup] Closing stats recorder...")
            try:
                # Don't close the writer itself, just detach components
                # self.stats_recorder.close() # This might close the writer we need
                pass  # Let the new recorder potentially reuse the writer
            except Exception as e:
                print(f"Error closing stats recorder: {e}")

        # Delete checkpoint file for the current run
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

        # Re-initialize RL components
        print("[Cleanup] Re-initializing RL components...")
        try:
            # --- MODIFIED: Pass is_reinit=True ---
            self._initialize_rl_components(is_reinit=True)
            # --- END MODIFIED ---

            if self.demo_env:  # Reset demo env as well
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
        if self.app_state == "MainMenu" and not self.is_running and self.trainer:
            print("Saving checkpoint on stop...")
            try:
                self.trainer.maybe_save_checkpoint(force_save=True)
            except Exception as e:
                print(f"Error saving checkpoint on stop: {e}")

    def _update(self):
        """Updates the application state and performs training steps (PPO)."""
        should_perform_training_iteration = False

        if self.app_state == "MainMenu":
            if self.cleanup_confirmation_active:
                self.status = "Confirm Cleanup"
            elif not self.is_running and self.status != "Error":
                self.status = "Ready"
            elif not self.trainer:
                if self.status != "Error":
                    self.status = "Error: Trainer Missing"
            elif self.is_running:
                self.status = "Training"
                should_perform_training_iteration = True
            else:  # Not running
                if self.status != "Error":
                    self.status = "Ready"

            if should_perform_training_iteration:
                if not self.trainer:
                    print("Error: Trainer became unavailable during _update.")
                    self.status = "Error: Trainer Lost"
                    self.is_running = False
                else:
                    try:
                        self.trainer.perform_training_iteration()
                    except Exception as e:
                        print(
                            f"\n--- ERROR DURING TRAINING ITERATION (Step: {getattr(self.trainer, 'global_step', 'N/A')}) ---"
                        )
                        traceback.print_exc()
                        print("--- Stopping training due to error. ---")
                        self.is_running = False
                        self.status = "Error: Training Iteration Failed"
                        # Keep app_state as MainMenu but show error status? Or switch to Error state?
                        # Switching to Error state might be clearer.
                        self.app_state = "Error"

        elif self.app_state == "Playing":
            if self.demo_env and hasattr(self.demo_env, "_update_timers"):
                self.demo_env._update_timers()  # Ensure demo timers update
            self.status = "Playing Demo"
        elif self.app_state == "Initializing":
            self.status = "Initializing..."  # Or more specific status if available
        elif self.app_state == "Error":
            # Status should already be set by the error source
            pass

    def _render(self):
        """Renders the UI based on the current application state."""
        stats_summary = {}
        plot_data: Dict[str, Deque] = {}

        # Get stats unless in pure initializing state
        if self.app_state != "Initializing":
            if self.stats_recorder:
                current_step = getattr(self.trainer, "global_step", 0)
                try:
                    stats_summary = self.stats_recorder.get_summary(current_step)
                except Exception as e:
                    print(f"Error getting stats summary: {e}")
                    stats_summary = {"global_step": current_step}  # Basic fallback
                try:
                    plot_data = self.stats_recorder.get_plot_data()
                except Exception as e:
                    print(f"Error getting plot data: {e}")
                    plot_data = {}
            elif self.app_state == "Error":
                # Try to get step count even if recorder failed
                stats_summary = {"global_step": getattr(self.trainer, "global_step", 0)}

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
                is_running=self.is_running,
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
            )
        except Exception as render_all_err:
            print(f"CRITICAL ERROR in renderer.render_all: {render_all_err}")
            traceback.print_exc()
            # Attempt to render a basic error screen via the renderer if possible
            try:
                self.app_state = "Error"
                self.status = "Render Error"
                self.renderer._render_error_screen(self.status)
                pygame.display.flip()  # Need flip here
            except Exception as e:
                print(f"Error rendering error screen: {e}")

        # Clear temporary cleanup message after timeout
        if time.time() - self.last_cleanup_message_time >= 5.0:
            self.cleanup_message = ""

    def _perform_cleanup(self):
        """Handles final cleanup of resources."""
        print("Exiting application...")
        if self.trainer:
            print("Performing final trainer cleanup...")
            try:
                # Save unless cleanup was just performed or in error state
                save_on_exit = self.status != "Cleaning" and self.app_state != "Error"
                self.trainer.cleanup(save_final=save_on_exit)
            except Exception as final_cleanup_err:
                print(f"Error during final trainer cleanup: {final_cleanup_err}")
        elif self.stats_recorder:  # Close stats recorder even if trainer failed/absent
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
        running_flag = True
        try:
            while running_flag:
                start_frame_time = time.perf_counter()

                # --- Input Handling ---
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
                        running_flag = False  # Exit on unhandled input error
                else:  # Basic exit if handler failed init
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
                    break  # Exit if input handler requested it

                # --- Update ---
                try:
                    self._update()
                except Exception as update_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN UPDATE LOOP ({self.app_state}) ---"
                    )
                    traceback.print_exc()
                    self.status = "Error: Update Loop Failed"
                    self.app_state = "Error"
                    self.is_running = False  # Stop training on update error

                # --- Render ---
                try:
                    self._render()
                except Exception as render_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN RENDER LOOP ({self.app_state}) ---"
                    )
                    traceback.print_exc()
                    self.status = "Error: Render Loop Failed"
                    self.app_state = "Error"
                    # Continue running to display error? Or exit? Let's try continuing.

                # --- Frame Rate Control (Optional) ---
                # Let Pygame handle drawing timing, avoid manual sleep unless necessary
                # self.clock.tick(self.vis_config.FPS if self.vis_config.FPS > 0 else 0)
                # A tiny sleep can prevent 100% CPU usage in idle states if needed
                if not self.is_running and self.app_state == "MainMenu":
                    time.sleep(0.01)

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
        # Attempt cleanup even if init failed partially
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
