import sys
import pygame
import os
import time
import traceback
from typing import List, Tuple, Optional, Dict, Any, Deque

from utils.helpers import set_random_seeds, get_device
from config.general import set_device as set_config_device

determined_device = get_device()
set_config_device(determined_device)

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
    ObsNormConfig,
    TransformerConfig,
    RANDOM_SEED,
    MODEL_SAVE_PATH,
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    RUN_LOG_DIR,
    TOTAL_TRAINING_STEPS,
)
from config.general import DEVICE

from environment.game_state import GameState, StateType
from agent.ppo_agent import PPOAgent

from training.trainer import Trainer
from stats.stats_recorder import StatsRecorderBase
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler
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
        self.obs_norm_config = ObsNormConfig()
        self.transformer_config = TransformerConfig()

        self.device = DEVICE
        if self.device is None:
            print("FATAL: Device was not set correctly before MainApp init.")
            sys.exit(1)

        self.config_dict = load_and_validate_configs()
        self.num_envs = self.env_config.NUM_ENVS

        initialize_directories()
        self.screen, self.clock = initialize_pygame(self.vis_config)

        self.app_state = "Initializing"
        self.is_process_running = False
        self.cleanup_confirmation_active = False
        self.last_cleanup_message_time = 0.0
        self.cleanup_message = ""
        self.status = "Initializing Components"
        self.update_progress_details: Dict[str, Any] = {}

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
                # Initialize Renderer first (contains LeftPanelRenderer)
                self.renderer = UIRenderer(self.screen, self.vis_config)
                self.renderer.render_all(  # Initial render before components load
                    app_state=self.app_state,
                    is_process_running=self.is_process_running,
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
                    update_progress_details={},
                )
                pygame.display.flip()  # Show initial screen
                pygame.time.delay(100)

            # Initialize RL components
            self._initialize_rl_components(is_reinit=is_reinit)

            if not is_reinit:
                # Initialize Demo Env
                self._initialize_demo_env()

                self.input_handler = InputHandler(
                    screen=self.screen,
                    renderer=self.renderer,  # Pass renderer
                    toggle_training_run_cb=self._toggle_training_run,
                    request_cleanup_cb=self._request_cleanup,
                    cancel_cleanup_cb=self._cancel_cleanup,
                    confirm_cleanup_cb=self._confirm_cleanup,
                    exit_app_cb=self._exit_app,
                    start_demo_mode_cb=self._start_demo_mode,
                    exit_demo_mode_cb=self._exit_demo_mode,
                    handle_demo_mouse_motion_cb=self._handle_demo_mouse_motion,  # New
                    handle_demo_mouse_button_down_cb=self._handle_demo_mouse_button_down,  # New
                    start_debug_mode_cb=self._start_debug_mode,
                    exit_debug_mode_cb=self._exit_debug_mode,
                    handle_debug_input_cb=self._handle_debug_input,
                )
                # Set the InputHandler reference in the LeftPanelRenderer
                if self.renderer and self.renderer.left_panel:
                    self.renderer.left_panel.input_handler = self.input_handler

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
                model_config=self.model_config,
                ppo_config=self.ppo_config,
                rnn_config=self.rnn_config,
                env_config=self.env_config,
                transformer_config=self.transformer_config,
                device=self.device,
            )
            self.stats_recorder = initialize_stats_recorder(
                stats_config=self.stats_config,
                tb_config=self.tensorboard_config,
                config_dict=self.config_dict,
                agent=self.agent,
                env_config=self.env_config,
                rnn_config=self.rnn_config,
                transformer_config=self.transformer_config,
                is_reinit=is_reinit,
            )
            if self.stats_recorder is None:
                raise RuntimeError("Stats Recorder init failed.")

            self.trainer = initialize_trainer(
                envs=self.envs,
                agent=self.agent,
                stats_recorder=self.stats_recorder,
                env_config=self.env_config,
                ppo_config=self.ppo_config,
                rnn_config=self.rnn_config,
                train_config=self.train_config,
                model_config=self.model_config,
                obs_norm_config=self.obs_norm_config,
                transformer_config=self.transformer_config,
                device=self.device,
                model_save_path=MODEL_SAVE_PATH,
                load_checkpoint_path=self.train_config.LOAD_CHECKPOINT_PATH,
            )
            print(f"RL components initialized in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Error during RL component initialization: {e}")
            raise e

    def _initialize_demo_env(self):
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
            print("Warning: Demo/Debug mode may be unavailable.")

    # --- Input Handler Callbacks ---
    def _toggle_training_run(self):
        """Starts or stops the training run process."""
        if self.app_state != "MainMenu":
            print("Cannot toggle run outside MainMenu.")
            return
        if not self.trainer:
            print("Cannot toggle run: Trainer not initialized.")
            return

        if not self.is_process_running:
            if self.trainer.global_step >= TOTAL_TRAINING_STEPS:
                print(
                    f"Training already completed ({self.trainer.global_step}/{TOTAL_TRAINING_STEPS} steps). Cannot restart."
                )
                self.status = "Training Complete"
                return

            self.is_process_running = True
            self.status = "Collecting Experience"

        else:
            print("Stopping process...")
            self.is_process_running = False
            self._try_save_checkpoint()
            self.app_state = "MainMenu"
            self.status = "Ready"

    def _request_cleanup(self):
        if self.is_process_running:
            print("Cannot request cleanup while process is running.")
            return
        if self.app_state != "MainMenu":
            print("Cannot request cleanup outside MainMenu.")
            return
        self.cleanup_confirmation_active = True
        print("Cleanup requested. Confirm action.")

    def _start_demo_mode(self):
        if self.is_process_running:
            print("Cannot start demo mode while process is running.")
            return
        if self.demo_env is None:
            print("Cannot start demo mode: Demo environment failed to initialize.")
            return
        if self.app_state != "MainMenu":
            print("Cannot start demo mode outside MainMenu.")
            return
        print("Entering Demo Mode...")
        self._try_save_checkpoint()
        self.app_state = "Playing"
        self.status = "Playing Demo"
        self.demo_env.reset()

    def _start_debug_mode(self):
        if self.is_process_running:
            print("Cannot start debug mode while process is running.")
            return
        if self.demo_env is None:
            print("Cannot start debug mode: Demo environment failed to initialize.")
            return
        if self.app_state != "MainMenu":
            print("Cannot start debug mode outside MainMenu.")
            return
        print("Entering Debug Mode...")
        self._try_save_checkpoint()
        self.app_state = "Debug"
        self.status = "Debugging Grid"
        self.demo_env.reset()

    def _exit_debug_mode(self):
        if self.app_state == "Debug":
            print("Exiting Debug Mode...")
            self.app_state = "MainMenu"
            self.status = "Ready"

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

    def _exit_demo_mode(self):
        if self.app_state == "Playing":
            print("Exiting Demo Mode...")
            if self.demo_env:
                self.demo_env.deselect_dragged_shape()
            self.app_state = "MainMenu"
            self.status = "Ready"

    def _handle_demo_mouse_motion(self, mouse_pos: Tuple[int, int]):
        """Handles mouse movement during demo mode for shape dragging and snapping."""
        if self.app_state != "Playing" or self.demo_env is None:
            return
        if self.demo_env.is_frozen() or self.demo_env.is_over():
            return
        if self.demo_env.demo_dragged_shape_idx is None:
            return  # Only update snapping if a shape is being dragged

        # Map mouse position to grid coordinates
        grid_coords = self._map_screen_to_grid(mouse_pos)
        # Update the snapped position in the game state
        self.demo_env.update_snapped_position(grid_coords)

    def _handle_demo_mouse_button_down(self, event: pygame.event.Event):
        """Handles mouse clicks during demo mode for selection and placement."""
        if self.app_state != "Playing" or self.demo_env is None:
            return
        if self.demo_env.is_frozen() or self.demo_env.is_over():
            return
        if event.button != 1:  # Only handle left clicks
            return

        mouse_pos = event.pos

        # 1. Check for click on shape previews
        clicked_preview_index = self._map_screen_to_preview(mouse_pos)
        if clicked_preview_index is not None:
            if clicked_preview_index == self.demo_env.demo_dragged_shape_idx:
                # Clicked on the already dragged shape preview -> deselect
                self.demo_env.deselect_dragged_shape()
            else:
                # Clicked on a different (or no) shape preview -> select new one
                self.demo_env.select_shape_for_drag(clicked_preview_index)
            return  # Handled click on preview

        # 2. Check for click on the game grid
        grid_coords = self._map_screen_to_grid(mouse_pos)
        if grid_coords is not None:
            # Clicked on the grid - attempt placement if snapped
            if (
                self.demo_env.demo_dragged_shape_idx is not None
                and self.demo_env.demo_snapped_position == grid_coords
            ):
                placed = self.demo_env.place_dragged_shape()
                if placed and self.demo_env.is_over():
                    print("[Demo] Game Over! Press ESC to exit.")
            else:
                # Clicked on grid but not snapped or no shape dragged -> deselect
                self.demo_env.deselect_dragged_shape()
            return  # Handled click on grid

        # 3. Clicked outside previews and grid -> deselect
        self.demo_env.deselect_dragged_shape()

    def _handle_debug_input(self, event: pygame.event.Event):
        """Handles input during debug mode (clicks, reset)."""
        if self.app_state != "Debug" or self.demo_env is None:
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                print("[Debug] Resetting grid...")
                self.demo_env.reset()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
            mouse_pos = event.pos
            # Map click position to grid coordinates
            clicked_coords = self._map_screen_to_grid(mouse_pos)
            if clicked_coords:
                row, col = clicked_coords
                self.demo_env.toggle_triangle_debug(row, col)

    def _map_screen_to_grid(
        self, screen_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Maps screen coordinates to grid row/col in Debug/Demo mode."""
        if self.renderer is None or self.demo_env is None:
            return None
        if self.app_state not in ["Playing", "Debug"]:
            return None

        screen_width, screen_height = self.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        # Recalculate game area rect (same as in DemoRenderer)
        game_rect, clipped_game_rect = (
            self.renderer.demo_renderer._calculate_game_area_rect(
                screen_width,
                screen_height,
                padding,
                hud_height,
                help_height,
                self.env_config,
            )
        )

        if not clipped_game_rect.collidepoint(screen_pos):
            return None  # Click outside game area

        # Calculate relative position within the clipped game area
        relative_x = screen_pos[0] - clipped_game_rect.left
        relative_y = screen_pos[1] - clipped_game_rect.top

        # Calculate triangle size and grid offset (same as in DemoRenderer)
        tri_cell_w, tri_cell_h = (
            self.renderer.demo_renderer._calculate_demo_triangle_size(
                clipped_game_rect.width, clipped_game_rect.height, self.env_config
            )
        )
        grid_ox, grid_oy = self.renderer.demo_renderer._calculate_grid_offset(
            clipped_game_rect.width, clipped_game_rect.height, self.env_config
        )

        if tri_cell_w <= 0 or tri_cell_h <= 0:
            return None

        # Adjust relative position by grid offset
        grid_relative_x = relative_x - grid_ox
        grid_relative_y = relative_y - grid_oy

        # Approximate row/col calculation (might need refinement based on triangle geometry)
        # This is a simplified version assuming near-rectangular cells for mapping
        # A more precise method would involve checking which triangle polygon contains the point
        approx_row = int(grid_relative_y / tri_cell_h)
        # Adjust col based on row and horizontal offset of triangles
        approx_col = int(grid_relative_x / (tri_cell_w * 0.75))

        # Basic bounds check
        if (
            0 <= approx_row < self.env_config.ROWS
            and 0 <= approx_col < self.env_config.COLS
        ):
            # Refine based on which half of the "diamond" the click is in
            # TODO: Implement more precise point-in-triangle test if needed.
            # For now, return the approximate row/col if it's valid and not death.
            if (
                self.demo_env.grid.valid(approx_row, approx_col)
                and not self.demo_env.grid.triangles[approx_row][approx_col].is_death
            ):
                return approx_row, approx_col

        return None

    def _map_screen_to_preview(self, screen_pos: Tuple[int, int]) -> Optional[int]:
        """Maps screen coordinates to a shape preview index if clicked."""
        if self.renderer is None or self.demo_env is None or self.input_handler is None:
            return None
        if self.app_state != "Playing":
            return None

        # Use the shape preview rects calculated by the InputHandler/DemoRenderer
        # We need the InputHandler to store these rects after they are calculated in DemoRenderer
        if hasattr(self.input_handler, "shape_preview_rects"):
            for idx, rect in self.input_handler.shape_preview_rects.items():
                if rect.collidepoint(screen_pos):
                    return idx
        else:
            # Fallback or recalculate if rects aren't stored in input_handler
            # This requires DemoRenderer layout logic to be duplicated or accessed
            # For simplicity, assume InputHandler will store them.
            pass

        return None

    # --- Core Logic Methods ---
    def _cleanup_data(self):
        """Deletes current run's checkpoint and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.app_state = "Initializing"
        self.is_process_running = False
        self.status = "Cleaning"
        messages = []
        if self.renderer:
            try:
                self.renderer.render_all(
                    app_state=self.app_state,
                    is_process_running=False,
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
                    update_progress_details={},
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
                if hasattr(self.stats_recorder, "close"):
                    self.stats_recorder.close()
            except Exception as e:
                print(f"Error closing stats recorder: {e}")
        print("[Cleanup] Deleting agent checkpoint file/dir...")
        try:
            save_dir = os.path.dirname(MODEL_SAVE_PATH)
            if os.path.isdir(save_dir):
                import shutil

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
            and not self.is_process_running
            and self.trainer
        ):
            print("Saving checkpoint on stop...")
            try:
                self.trainer.maybe_save_checkpoint(force_save=True)
            except Exception as e:
                print(f"Error saving checkpoint on stop: {e}")

    def _update(self):
        """Updates the application state and performs training steps."""
        self.update_progress_details = {}
        if not self.trainer:
            if self.app_state != "Error" and self.app_state != "Initializing":
                self.status = "Error: Trainer Missing"
                self.app_state = "Error"
            return

        if self.is_process_running:
            try:
                if self.trainer.global_step >= TOTAL_TRAINING_STEPS:
                    print(
                        f"\n--- Training Complete ({self.trainer.global_step}/{TOTAL_TRAINING_STEPS} steps) ---"
                    )
                    self.is_process_running = False
                    self.app_state = "MainMenu"
                    self.status = "Training Complete"
                    self._try_save_checkpoint()
                    return

                self.trainer.perform_training_iteration()

                if self.trainer.global_step >= TOTAL_TRAINING_STEPS:
                    print(
                        f"\n--- Training Complete ({self.trainer.global_step}/{TOTAL_TRAINING_STEPS} steps) ---"
                    )
                    self.is_process_running = False
                    self.app_state = "MainMenu"
                    self.status = "Training Complete"
                    self._try_save_checkpoint()
                    return

                self.status = self.trainer.get_current_phase()
                self.update_progress_details = (
                    self.trainer.get_update_progress_details()
                )

            except Exception as e:
                error_phase = "TRAINING"
                print(f"\n--- ERROR DURING {error_phase} ITERATION ---")
                traceback.print_exc()
                self.status = f"Error: {error_phase} Failed"
                self.app_state = "Error"
                self.is_process_running = False
        elif not self.is_process_running:
            if self.app_state == "Playing":
                if self.demo_env and hasattr(self.demo_env, "_update_timers"):
                    self.demo_env._update_timers()
                self.status = "Playing Demo"
            elif self.app_state == "Debug":
                if self.demo_env and hasattr(self.demo_env, "_update_timers"):
                    self.demo_env._update_timers()
                self.status = "Debugging Grid"
            elif self.app_state == "Initializing":
                self.status = "Initializing..."
            elif self.app_state == "MainMenu":
                if self.cleanup_confirmation_active:
                    self.status = "Confirm Cleanup"
                elif self.status != "Error" and self.status != "Training Complete":
                    self.status = "Ready"
            elif self.app_state == "Error":
                pass

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
            return

        try:
            self.renderer.render_all(
                app_state=self.app_state,
                is_process_running=self.is_process_running,
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
                update_progress_details=self.update_progress_details,
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
                save_on_exit = (
                    self.status != "Cleaning" and self.app_state != "Error"
                ) or self.status == "Training Complete"
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
                # 1. Handle Input
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
                    # Fallback basic event handling if input_handler fails
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running_flag = False
                        if (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_ESCAPE
                        ):
                            if self.app_state == "Playing":
                                self._exit_demo_mode()
                            elif self.app_state == "Debug":
                                self._exit_debug_mode()
                            elif not self.cleanup_confirmation_active:
                                running_flag = False
                    if not running_flag:
                        break

                if not running_flag:
                    break

                # 2. Update State
                try:
                    self._update()
                except Exception as update_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN UPDATE LOOP ({self.app_state}) ---"
                    )
                    traceback.print_exc()
                    self.status = "Error: Update Loop Failed"
                    self.app_state = "Error"
                    self.is_process_running = False

                # 3. Render Frame
                try:
                    self._render()
                except Exception as render_err:
                    print(
                        f"\n--- UNHANDLED ERROR IN RENDER LOOP ({self.app_state}) ---"
                    )
                    traceback.print_exc()
                    self.status = "Error: Render Loop Failed"
                    self.app_state = "Error"

                # 4. Frame Limiting / Yielding
                is_updating = "Updating" in self.status
                is_active_phase = self.is_process_running or self.app_state in [
                    "Playing",
                    "Debug",
                ]

                if not is_active_phase:
                    time.sleep(0.010)
                elif is_active_phase and not is_updating:
                    time.sleep(0.001)

                if self.vis_config.FPS > 0:
                    self.clock.tick(self.vis_config.FPS)

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
    original_stdout, original_stderr = sys.stdout, sys.stderr
    logger = TeeLogger(log_filepath, original_stdout)
    sys.stdout, sys.stderr = logger, logger
    app_instance, exit_code = None, 0
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
        sys.stdout, sys.stderr = original_stdout, original_stderr
        print(f"Console logging restored. Full log should be in: {log_filepath}")
        sys.exit(exit_code)
