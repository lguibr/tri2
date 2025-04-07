File: analyze_profile.py
# File: analyze_profile_v2.py
import pstats
from pstats import SortKey

profile_file = "profile_output.prof"
output_file_cumulative = "profile_summary_cumulative.txt"
output_file_tottime = "profile_summary_tottime.txt"
num_lines_to_print = 50  # You can adjust how many lines to show

try:
    # --- Sort by Cumulative Time ---
    print(
        f"Saving top {num_lines_to_print} cumulative time stats to {output_file_cumulative}..."
    )
    with open(output_file_cumulative, "w") as f_cum:
        # Pass the file handle directly as the stream
        stats_cum = pstats.Stats(profile_file, stream=f_cum)
        stats_cum.sort_stats(SortKey.CUMULATIVE).print_stats(num_lines_to_print)
        # 'with open' handles closing/flushing
    print("Done.")

    # --- Sort by Total Time (Internal) ---
    print(
        f"Saving top {num_lines_to_print} total time (tottime) stats to {output_file_tottime}..."
    )
    with open(output_file_tottime, "w") as f_tot:
        # Pass the file handle directly as the stream
        stats_tot = pstats.Stats(profile_file, stream=f_tot)
        stats_tot.sort_stats(SortKey.TIME).print_stats(
            num_lines_to_print
        )  # SortKey.TIME is 'tottime'
        # 'with open' handles closing/flushing
    print("Done.")

    print(
        f"\nAnalysis complete. Check '{output_file_cumulative}' and '{output_file_tottime}'."
    )

except FileNotFoundError:
    print(f"ERROR: Profile file '{profile_file}' not found.")
except Exception as e:
    print(f"An error occurred during profile analysis: {e}")


File: app_init.py
# File: app_init.py
import pygame
import time
import traceback
import sys
import torch
from typing import TYPE_CHECKING, List, Optional, Dict, Any

from config import (
    VisConfig,
    EnvConfig,
    # Removed PPOConfig
    RNNConfig,
    ModelConfig,
    StatsConfig,
    # Removed RewardConfig
    TensorBoardConfig,
    DemoConfig,
    # Removed ObsNormConfig
    TransformerConfig,
    RANDOM_SEED,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
    DEVICE,
)
from environment.game_state import GameState

# Removed PPOAgent import
# from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler

# Removed init.rl_components_ppo import
from training.checkpoint_manager import CheckpointManager

# Removed RolloutCollector import
from app_state import AppState

if TYPE_CHECKING:
    from main_pygame import MainApp
    from agent.base_agent import BaseAgent  # Hypothetical base class for NN agent


class AppInitializer:
    """Handles the initialization of core application components."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Config instances
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        # Removed self.ppo_config
        self.rnn_config = RNNConfig()
        self.train_config = app.train_config_instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        # Removed self.reward_config
        # Removed self.obs_norm_config
        self.transformer_config = TransformerConfig()

        # Components to be initialized
        self.envs: List[GameState] = []  # Keep for potential multi-env display
        self.agent: Optional["BaseAgent"] = None  # Agent is now the NN
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        # Removed self.rollout_collector

    def initialize_all(self, is_reinit: bool = False):
        """Initializes all core components."""
        try:
            # GPU Memory Info (Keep)
            if self.app.device.type == "cuda":
                try:
                    self.app.total_gpu_memory_bytes = torch.cuda.get_device_properties(
                        self.app.device
                    ).total_memory
                    print(
                        f"Total GPU Memory: {self.app.total_gpu_memory_bytes / (1024**3):.2f} GB"
                    )
                except Exception as e:
                    print(f"Warning: Could not get total GPU memory: {e}")

            # Renderer and Initial Render (Keep)
            if not is_reinit:
                self.app.renderer = UIRenderer(self.app.screen, self.vis_config)
                # Adapt render_all call later if needed
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=False,  # No PPO process running
                    status=self.app.status,
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
                    update_progress_details={},  # Keep for potential NN training progress
                    agent_param_count=0,
                    worker_counts={},  # Remove worker counts for now
                )
                pygame.display.flip()
                pygame.time.delay(100)

            # Initialize "RL" components (NN, Stats, Checkpoint Manager)
            self.initialize_rl_components(
                is_reinit=is_reinit, checkpoint_to_load=self.app.checkpoint_to_load
            )

            # Demo Env and Input Handler (Keep)
            if not is_reinit:
                self.initialize_demo_env()
                self.initialize_input_handler()

            # Calculate NN parameter count if agent exists
            if self.agent and hasattr(self.agent, "network"):
                try:
                    self.agent_param_count = sum(
                        p.numel()
                        for p in self.agent.network.parameters()
                        if p.requires_grad
                    )
                except Exception as e:
                    print(f"Warning: Could not calculate agent parameters: {e}")
                    self.agent_param_count = 0

        except Exception as init_err:
            print(f"FATAL ERROR during component initialization: {init_err}")
            traceback.print_exc()
            if self.app.renderer:
                try:
                    self.app.app_state = AppState.ERROR
                    self.app.status = "Initialization Failed"
                    self.app.renderer._render_error_screen(self.app.status)
                    pygame.display.flip()
                    time.sleep(5)
                except Exception:
                    pass
            pygame.quit()
            sys.exit(1)

    def initialize_rl_components(
        self, is_reinit: bool = False, checkpoint_to_load: Optional[str] = None
    ):
        """Initializes NN Agent, Stats Recorder, Checkpoint Manager."""
        print(f"Initializing AlphaZero components... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            # Initialize Envs (only needed for visualization now)
            # self.envs = initialize_envs(self.env_config.NUM_ENVS, self.env_config) # Removed multi-env init
            self.envs = []  # No parallel envs needed for core logic now

            # --- Initialize Agent (Neural Network) ---
            # Replace with AlphaZero NN initialization later
            # For now, set agent to None or a placeholder
            self.agent = None  # Placeholder - Initialize AlphaZero NN here later
            print("Agent (NN) initialization SKIPPED (placeholder).")
            # Example placeholder for future NN init:
            # self.agent = initialize_alphazero_agent(
            #     model_config=self.model_config,
            #     rnn_config=self.rnn_config,
            #     env_config=self.env_config,
            #     transformer_config=self.transformer_config,
            #     device=self.app.device,
            # )
            # --- End Agent Init ---

            # --- Initialize Stats Recorder ---
            # Adapt initialize_stats_recorder if needed (e.g., remove PPO hparams)
            # Need to create a simplified version or adapt existing one
            # For now, assume it's adapted or create a placeholder
            try:
                # Assuming init.stats_init exists and is adapted
                from init.stats_init import initialize_stats_recorder

                self.stats_recorder = initialize_stats_recorder(
                    stats_config=self.stats_config,
                    tb_config=self.tensorboard_config,
                    config_dict=self.app.config_dict,
                    # Pass agent=None if NN not ready, or pass the NN agent
                    agent=self.agent,
                    env_config=self.env_config,
                    rnn_config=self.rnn_config,  # Keep for potential NN config logging
                    transformer_config=self.transformer_config,  # Keep for potential NN config logging
                    is_reinit=is_reinit,
                )
            except ImportError:
                print(
                    "Warning: init.stats_init.initialize_stats_recorder not found. Skipping stats recorder init."
                )
                self.stats_recorder = None  # Fallback
            except Exception as stats_init_err:
                print(f"Error initializing stats recorder: {stats_init_err}")
                traceback.print_exc()
                self.stats_recorder = None

            if self.stats_recorder is None:
                print("Warning: Stats Recorder initialization failed or skipped.")
                # Decide if this is critical - maybe allow running without stats?
                # raise RuntimeError("Stats Recorder init failed.")
            # --- End Stats Recorder Init ---

            # --- Initialize Checkpoint Manager ---
            # Checkpoint manager now handles NN agent state and stats aggregator state
            self.checkpoint_manager = CheckpointManager(
                # Pass the NN agent (or None if not ready)
                agent=self.agent,
                # Pass stats aggregator if it exists
                stats_aggregator=getattr(self.stats_recorder, "aggregator", None),
                base_checkpoint_dir=BASE_CHECKPOINT_DIR,
                run_checkpoint_dir=get_run_checkpoint_dir(),
                load_checkpoint_path_config=checkpoint_to_load,
                device=self.app.device,
                # obs_rms_dict=None, # Removed Obs RMS
            )
            # --- End Checkpoint Manager Init ---

            # --- Load Checkpoint ---
            if self.checkpoint_manager.get_checkpoint_path_to_load():
                self.checkpoint_manager.load_checkpoint()  # Loads NN state and stats state
                # Get initial step/episode count from loaded stats
                loaded_global_step, initial_episode_count = (
                    self.checkpoint_manager.get_initial_state()
                )
                # Sync episode count if needed (e.g., if MCTS tracks episodes)
                # self.mcts_manager.state.episode_count = initial_episode_count # Example
            # --- End Load Checkpoint ---

            print(
                f"AlphaZero components initialized in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            print(f"Error during AlphaZero component initialization: {e}")
            traceback.print_exc()
            raise e  # Re-raise to be caught by initialize_all

    def initialize_demo_env(self):
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

    def initialize_input_handler(self):
        """Initializes the Input Handler."""
        if not self.app.renderer:
            print("ERROR: Cannot initialize InputHandler before Renderer.")
            return

        self.app.input_handler = InputHandler(
            screen=self.app.screen,
            renderer=self.app.renderer,
            # Removed toggle_training_run_cb
            request_cleanup_cb=self.app.logic.request_cleanup,
            cancel_cleanup_cb=self.app.logic.cancel_cleanup,
            confirm_cleanup_cb=self.app.logic.confirm_cleanup,
            exit_app_cb=self.app.logic.exit_app,
            start_demo_mode_cb=self.app.logic.start_demo_mode,
            exit_demo_mode_cb=self.app.logic.exit_demo_mode,
            handle_demo_mouse_motion_cb=self.app.logic.handle_demo_mouse_motion,
            handle_demo_mouse_button_down_cb=self.app.logic.handle_demo_mouse_button_down,
            start_debug_mode_cb=self.app.logic.start_debug_mode,
            exit_debug_mode_cb=self.app.logic.exit_debug_mode,
            handle_debug_input_cb=self.app.logic.handle_debug_input,
        )
        if self.app.renderer and self.app.renderer.left_panel:
            self.app.renderer.left_panel.input_handler = self.app.input_handler
            if hasattr(self.app.renderer.left_panel, "button_status_renderer"):
                self.app.renderer.left_panel.button_status_renderer.input_handler_ref = (
                    self.app.input_handler
                )

    def close_stats_recorder(self, is_cleanup: bool = False):
        """Safely closes the stats recorder."""
        print(
            f"[AppInitializer] close_stats_recorder called (is_cleanup={is_cleanup})..."
        )
        if self.stats_recorder:
            print("[AppInitializer] Stats recorder exists, attempting close...")
            try:
                if hasattr(self.stats_recorder, "close"):
                    self.stats_recorder.close(is_cleanup=is_cleanup)
                    print("[AppInitializer] stats_recorder.close() executed.")
                else:
                    print("[AppInitializer] stats_recorder has no close method.")
            except Exception as log_e:
                print(f"[AppInitializer] Error closing stats recorder on exit: {log_e}")
                traceback.print_exc()
        else:
            print("[AppInitializer] No stats recorder instance to close.")
        print("[AppInitializer] close_stats_recorder finished.")


File: app_logic.py
# File: app_logic.py
import pygame
import time
import traceback
import os
import shutil
from typing import TYPE_CHECKING, Tuple

from app_state import AppState

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppLogic:
    """Handles the core application logic and state transitions."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def check_initial_completion_status(self):
        """Checks if training target (e.g., games played) was met upon loading."""
        # This needs adaptation based on how AlphaZero training progress is tracked.
        # For now, assume completion isn't automatically checked this way.
        # Example: Check against games played or NN training steps in aggregator
        # if (
        #     self.app.initializer.stats_recorder
        #     and hasattr(self.app.initializer.stats_recorder, "aggregator")
        # ):
        #     aggregator = self.app.initializer.stats_recorder.aggregator
        #     target_games = getattr(aggregator.storage, "training_target_games", 0) # Example target
        #     current_games = getattr(aggregator.storage, "total_episodes", 0)
        #     if target_games > 0 and current_games >= target_games:
        #         self.app.status = "Target Reached"
        #         print(f"Target games already reached ({current_games:,}/{target_games:,}). Ready.")
        pass  # Keep simple for now

    def update_status_and_check_completion(self):
        """Updates the status text based on application state."""
        # Removed PPO-specific status logic (Collecting, Updating)
        # Removed completion check based on PPO steps

        # Update status based on AppState
        if self.app.app_state == AppState.MAIN_MENU:
            if self.app.cleanup_confirmation_active:
                self.app.status = "Confirm Cleanup"
            else:
                # Check if training is "complete" based on new criteria if needed
                # For now, just set to Ready if not confirming cleanup
                self.app.status = "Ready"
        elif self.app.app_state == AppState.PLAYING:
            self.app.status = "Playing Demo"
        elif self.app.app_state == AppState.DEBUG:
            self.app.status = "Debugging Grid"
        elif self.app.app_state == AppState.INITIALIZING:
            self.app.status = "Initializing..."
        elif self.app.app_state == AppState.ERROR:
            # Status should already be set by the error handler
            pass

    # Removed toggle_training_run method

    def request_cleanup(self):
        # Removed check for is_process_running
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot request cleanup outside MainMenu.")
            return
        self.app.cleanup_confirmation_active = True
        self.app.status = "Confirm Cleanup"  # Update status
        print("Cleanup requested. Confirm action.")

    def start_demo_mode(self):
        # Removed check for is_process_running
        if self.app.initializer.demo_env is None:
            print("Cannot start demo mode: Demo environment failed to initialize.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot start demo mode outside MainMenu.")
            return
        print("Entering Demo Mode...")
        self.try_save_checkpoint()  # Save NN weights before switching mode?
        self.app.app_state = AppState.PLAYING
        self.app.status = "Playing Demo"
        self.app.initializer.demo_env.reset()

    def start_debug_mode(self):
        # Removed check for is_process_running
        if self.app.initializer.demo_env is None:
            print("Cannot start debug mode: Demo environment failed to initialize.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot start debug mode outside MainMenu.")
            return
        print("Entering Debug Mode...")
        self.try_save_checkpoint()  # Save NN weights before switching mode?
        self.app.app_state = AppState.DEBUG
        self.app.status = "Debugging Grid"
        self.app.initializer.demo_env.reset()

    def exit_debug_mode(self):
        if self.app.app_state == AppState.DEBUG:
            print("Exiting Debug Mode...")
            self.app.app_state = AppState.MAIN_MENU
            # Removed setting is_process_running and pause_event
            self.check_initial_completion_status()  # Re-check status
            self.app.status = "Ready"  # Set status back to Ready

    def cancel_cleanup(self):
        self.app.cleanup_confirmation_active = False
        self.app.cleanup_message = "Cleanup cancelled."
        self.app.last_cleanup_message_time = time.time()
        self.app.status = "Ready"  # Set status back
        print("Cleanup cancelled by user.")

    def confirm_cleanup(self):
        print("Cleanup confirmed by user. Starting process...")
        try:
            self._cleanup_data()
        except Exception as e:
            print(f"FATAL ERROR during cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Cleanup Failed Critically"
            self.app.app_state = AppState.ERROR
        finally:
            self.app.cleanup_confirmation_active = False
            # Status is set within _cleanup_data or error handling
            print(
                f"Cleanup process finished. State: {self.app.app_state}, Status: {self.app.status}"
            )

    def exit_app(self) -> bool:
        print("Exit requested.")
        self.app.stop_event.set()  # Keep stop event for main loop exit
        return False

    def exit_demo_mode(self):
        if self.app.app_state == AppState.PLAYING:
            print("Exiting Demo Mode...")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.deselect_dragged_shape()
            self.app.app_state = AppState.MAIN_MENU
            # Removed setting is_process_running and pause_event
            self.check_initial_completion_status()  # Re-check status
            self.app.status = "Ready"  # Set status back to Ready

    def handle_demo_mouse_motion(self, mouse_pos: Tuple[int, int]):
        if (
            self.app.app_state != AppState.PLAYING
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if demo_env.demo_dragged_shape_idx is None:
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        demo_env.update_snapped_position(grid_coords)

    def handle_demo_mouse_button_down(self, event: pygame.event.Event):
        if (
            self.app.app_state != AppState.PLAYING
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if event.button != 1:
            return
        mouse_pos = event.pos
        clicked_preview_index = self.app.ui_utils.map_screen_to_preview(mouse_pos)
        if clicked_preview_index is not None:
            if clicked_preview_index == demo_env.demo_dragged_shape_idx:
                demo_env.deselect_dragged_shape()
            else:
                demo_env.select_shape_for_drag(clicked_preview_index)
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        if grid_coords is not None:
            if (
                demo_env.demo_dragged_shape_idx is not None
                and demo_env.demo_snapped_position == grid_coords
            ):
                placed = demo_env.place_dragged_shape()
                if placed and demo_env.is_over():
                    print("[Demo] Game Over! Press ESC to exit.")
            else:
                demo_env.deselect_dragged_shape()
            return
        demo_env.deselect_dragged_shape()

    def handle_debug_input(self, event: pygame.event.Event):
        if (
            self.app.app_state != AppState.DEBUG
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                print("[Debug] Resetting grid...")
                demo_env.reset()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            clicked_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
            if clicked_coords:
                row, col = clicked_coords
                demo_env.toggle_triangle_debug(row, col)

    def _cleanup_data(self):
        """Deletes current run's checkpoint and re-initializes components."""
        from config.general import get_run_checkpoint_dir  # Local import

        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        # Removed pause_event setting and is_process_running
        self.app.app_state = AppState.INITIALIZING
        self.app.status = "Cleaning"
        messages = []

        # Render cleaning status
        if self.app.renderer:
            try:
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=False,  # No process running now
                    status=self.app.status,
                    stats_summary={},
                    envs=[],
                    num_envs=0,
                    env_config=self.app.env_config,
                    cleanup_confirmation_active=False,
                    cleanup_message="",
                    last_cleanup_message_time=0,
                    tensorboard_log_dir=None,
                    plot_data={},
                    demo_env=self.app.initializer.demo_env,
                    update_progress_details={},
                    agent_param_count=getattr(
                        self.app.initializer, "agent_param_count", 0
                    ),
                    worker_counts={},  # Removed worker counts
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

        # --- Stop existing worker threads (if any) ---
        # Keep this structure in case new workers (MCTS/NN) are added later
        print("[Cleanup] Stopping existing worker threads (if any)...")
        self.app.worker_manager.stop_worker_threads()
        print("[Cleanup] Existing worker threads stopped.")
        # --- End Stop Workers ---

        # --- Close Stats Recorder ---
        print("[Cleanup] Closing stats recorder...")
        self.app.initializer.close_stats_recorder(is_cleanup=True)
        print("[Cleanup] Stats recorder closed.")
        # --- End Close Stats ---

        # --- Delete Checkpoint Directory ---
        print("[Cleanup] Deleting agent checkpoint file/dir...")
        try:
            save_dir = get_run_checkpoint_dir()
            if os.path.isdir(save_dir):
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
        print("[Cleanup] Checkpoint deletion attempt finished.")
        # --- End Delete Checkpoint ---

        time.sleep(0.1)
        print("[Cleanup] Re-initializing components...")
        try:
            # Re-initialize components (NN, Stats, Checkpoint Manager)
            self.app.initializer.initialize_rl_components(
                is_reinit=True, checkpoint_to_load=None
            )
            print("[Cleanup] Components re-initialized.")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
                print("[Cleanup] Demo env reset.")

            # --- Start new worker threads (if any) ---
            # Keep this structure for future workers
            print("[Cleanup] Starting new worker threads (if any)...")
            self.app.worker_manager.start_worker_threads()
            print("[Cleanup] New worker threads started.")
            # --- End Start Workers ---

            print("[Cleanup] Component re-initialization and worker start successful.")
            messages.append("Components re-initialized.")

            # --- Set state after cleanup ---
            # Removed is_process_running and pause_event
            self.app.status = "Ready"
            self.app.app_state = AppState.MAIN_MENU
            print("[Cleanup] Application state set to MAIN_MENU.")
            # --- End Set State ---

        except Exception as e:
            print(f"FATAL ERROR during re-initialization after cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Re-init Failed"
            self.app.app_state = AppState.ERROR
            messages.append("ERROR RE-INITIALIZING COMPONENTS!")
            if self.app.renderer:
                try:
                    self.app.renderer._render_error_screen(self.app.status)
                except Exception as render_err_final:
                    print(f"Warning: Failed to render error screen: {render_err_final}")

        self.app.cleanup_message = "\n".join(messages)
        self.app.last_cleanup_message_time = time.time()
        print(
            f"--- CLEANUP DATA COMPLETE (Final State: {self.app.app_state}, Status: {self.app.status}) ---"
        )

    def try_save_checkpoint(self):
        """Saves checkpoint (e.g., NN weights) if in main menu."""
        # Removed check for is_process_running
        if (
            self.app.app_state == AppState.MAIN_MENU
            and self.app.initializer.checkpoint_manager
            and self.app.initializer.stats_recorder  # Check if stats exist
            and hasattr(self.app.initializer.stats_recorder, "aggregator")
        ):
            print("Saving checkpoint...")
            try:
                # Get step/episode count from aggregator
                current_step = getattr(
                    self.app.initializer.stats_recorder.aggregator.storage,
                    "current_global_step",
                    0,
                )
                episode_count = getattr(
                    self.app.initializer.stats_recorder.aggregator.storage,
                    "total_episodes",
                    0,
                )
                # Target step is now managed within aggregator/checkpoint manager
                target_step = getattr(
                    self.app.initializer.checkpoint_manager, "training_target_step", 0
                )

                self.app.initializer.checkpoint_manager.save_checkpoint(
                    current_step,
                    episode_count,
                    training_target_step=target_step,
                    is_final=False,
                )
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                traceback.print_exc()

    def save_final_checkpoint(self):
        """Saves the final checkpoint (e.g., NN weights)."""
        if (
            self.app.initializer.checkpoint_manager
            and self.app.initializer.stats_recorder
            and hasattr(self.app.initializer.stats_recorder, "aggregator")
        ):
            save_on_exit = (
                self.app.status != "Cleaning" and self.app.app_state != AppState.ERROR
            )

            if save_on_exit:
                print("Performing final checkpoint save...")
                try:
                    current_step = getattr(
                        self.app.initializer.stats_recorder.aggregator.storage,
                        "current_global_step",
                        0,
                    )
                    episode_count = getattr(
                        self.app.initializer.stats_recorder.aggregator.storage,
                        "total_episodes",
                        0,
                    )
                    target_step = getattr(
                        self.app.initializer.checkpoint_manager,
                        "training_target_step",
                        0,
                    )
                    self.app.initializer.checkpoint_manager.save_checkpoint(
                        current_step,
                        episode_count,
                        training_target_step=target_step,
                        is_final=True,
                    )
                except Exception as final_save_err:
                    print(f"Error during final checkpoint save: {final_save_err}")
                    traceback.print_exc()
            else:
                print("Skipping final checkpoint save.")


File: app_setup.py
# File: app_setup.py
import os
import pygame
from typing import Tuple, Dict, Any

from config import (
    VisConfig,
    get_run_checkpoint_dir,
    get_run_log_dir,
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
    pygame.display.set_caption("AlphaTri Trainer")  # Updated caption
    clock = pygame.time.Clock()
    print("Pygame initialized.")
    return screen, clock


def initialize_directories():
    """Creates necessary runtime directories using dynamic paths."""
    run_checkpoint_dir = get_run_checkpoint_dir()
    run_log_dir = get_run_log_dir()
    # Console log dir is created within main_pygame before logger init

    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"Ensured directories exist: {run_checkpoint_dir}, {run_log_dir}")


def load_and_validate_configs() -> Dict[str, Any]:
    """Loads all config classes and returns the combined config dictionary."""
    config_dict = get_config_dict()
    print_config_info_and_validate()
    return config_dict


File: app_state.py
# File: app_state.py
from enum import Enum, auto


class AppState(Enum):
    INITIALIZING = "Initializing"
    MAIN_MENU = "MainMenu"
    PLAYING = "Playing"  # Demo Mode
    DEBUG = "Debug"
    CLEANUP_CONFIRM = "Confirm Cleanup"  # Not used directly, handled by flag
    CLEANING = "Cleaning"  # Intermediate state during cleanup
    ERROR = "Error"
    UNKNOWN = "Unknown"  # Fallback


File: app_ui_utils.py
# File: app_ui_utils.py
import pygame
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from main_pygame import MainApp
    from environment.game_state import GameState
    from ui.renderer import UIRenderer


class AppUIUtils:
    """Utility functions related to mapping screen coordinates to game elements."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def map_screen_to_grid(
        self, screen_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Maps screen coordinates to grid row/column for demo/debug."""
        if (
            self.app.renderer is None
            or self.app.initializer.demo_env is None
            or self.app.renderer.demo_renderer is None
        ):
            return None
        if self.app.app_state not in [
            self.app.app_state.PLAYING,
            self.app.app_state.DEBUG,
        ]:
            return None

        demo_env: "GameState" = self.app.initializer.demo_env
        renderer: "UIRenderer" = self.app.renderer

        screen_width, screen_height = self.app.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        _, clipped_game_rect = renderer.demo_renderer._calculate_game_area_rect(
            screen_width,
            screen_height,
            padding,
            hud_height,
            help_height,
            self.app.env_config,
        )

        if not clipped_game_rect.collidepoint(screen_pos):
            return None

        relative_x = screen_pos[0] - clipped_game_rect.left
        relative_y = screen_pos[1] - clipped_game_rect.top

        tri_cell_w, tri_cell_h = renderer.demo_renderer._calculate_demo_triangle_size(
            clipped_game_rect.width, clipped_game_rect.height, self.app.env_config
        )
        grid_ox, grid_oy = renderer.demo_renderer._calculate_grid_offset(
            clipped_game_rect.width, clipped_game_rect.height, self.app.env_config
        )

        if tri_cell_w <= 0 or tri_cell_h <= 0:
            return None

        grid_relative_x = relative_x - grid_ox
        grid_relative_y = relative_y - grid_oy

        # Approximate calculation (might need refinement based on triangle geometry)
        approx_row = int(grid_relative_y / tri_cell_h)
        approx_col = int(grid_relative_x / (tri_cell_w * 0.75))

        if (
            0 <= approx_row < self.app.env_config.ROWS
            and 0 <= approx_col < self.app.env_config.COLS
        ):
            if (
                demo_env.grid.valid(approx_row, approx_col)
                and not demo_env.grid.triangles[approx_row][approx_col].is_death
            ):
                return approx_row, approx_col
        return None

    def map_screen_to_preview(self, screen_pos: Tuple[int, int]) -> Optional[int]:
        """Maps screen coordinates to a shape preview index."""
        if (
            self.app.renderer is None
            or self.app.initializer.demo_env is None
            or self.app.input_handler is None
        ):
            return None
        if self.app.app_state != self.app.app_state.PLAYING:
            return None

        # Access preview rects directly from the input handler
        if hasattr(self.app.input_handler, "shape_preview_rects"):
            for idx, rect in self.app.input_handler.shape_preview_rects.items():
                if rect.collidepoint(screen_pos):
                    return idx
        return None


File: app_workers.py
# File: app_workers.py
import threading
import queue
import time
import traceback
from typing import TYPE_CHECKING, Optional  # Added Optional

# Removed worker imports
# from workers import EnvironmentRunner, TrainingWorker

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppWorkerManager:
    """Manages the creation, starting, and stopping of worker threads."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Remove specific worker thread attributes
        # self.env_runner_thread: Optional[EnvironmentRunner] = None
        # self.training_worker_thread: Optional[TrainingWorker] = None
        # Add placeholders for future workers if needed
        self.mcts_worker_thread: Optional[threading.Thread] = None
        self.nn_training_worker_thread: Optional[threading.Thread] = None
        print("[AppWorkerManager] Initialized (No workers started by default).")

    def start_worker_threads(self):
        """Creates and starts worker threads (MCTS, NN Training - Placeholder)."""
        # --- This needs to be implemented based on AlphaZero architecture ---
        print(
            "[AppWorkerManager] start_worker_threads called (Placeholder - No workers started)."
        )
        # Example structure:
        # if not self.app.initializer.agent or not self.app.initializer.mcts_manager:
        #     print("ERROR: Cannot start workers, core components not initialized.")
        #     self.app.app_state = self.app.app_state.ERROR
        #     self.app.status = "Worker Init Failed"
        #     return
        #
        # print("Starting AlphaZero worker threads...")
        # self.app.stop_event.clear()
        # # self.app.pause_event.clear() # Removed pause event
        #
        # # MCTS Self-Play Worker(s)
        # self.mcts_worker_thread = MCTSSelfPlayWorker(...)
        # self.mcts_worker_thread.start()
        #
        # # NN Training Worker
        # self.nn_training_worker_thread = NNTrainingWorker(...)
        # self.nn_training_worker_thread.start()
        #
        # print("AlphaZero worker threads started.")
        pass

    def stop_worker_threads(self):
        """Signals worker threads to stop and waits for them to join."""
        if self.app.stop_event.is_set():
            print("[AppWorkerManager] Stop event already set.")
            return

        print("[AppWorkerManager] Stopping worker threads (Placeholder)...")
        self.app.stop_event.set()
        # self.app.pause_event.clear() # Removed pause event

        join_timeout = 5.0

        # --- Join future worker threads ---
        if self.mcts_worker_thread and self.mcts_worker_thread.is_alive():
            print("[AppWorkerManager] Joining MCTS worker...")
            self.mcts_worker_thread.join(timeout=join_timeout)
            if self.mcts_worker_thread.is_alive():
                print("[AppWorkerManager] MCTS worker thread did not join cleanly.")
            self.mcts_worker_thread = None

        if self.nn_training_worker_thread and self.nn_training_worker_thread.is_alive():
            print("[AppWorkerManager] Joining NN Training worker...")
            self.nn_training_worker_thread.join(timeout=join_timeout)
            if self.nn_training_worker_thread.is_alive():
                print(
                    "[AppWorkerManager] NN Training worker thread did not join cleanly."
                )
            self.nn_training_worker_thread = None
        # --- End Join ---

        # Clear queues if used by workers
        # Example:
        # while not self.app.experience_queue.empty():
        #     try:
        #         self.app.experience_queue.get_nowait()
        #     except queue.Empty:
        #         break

        print("[AppWorkerManager] Worker threads stopped.")


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
import pygame
import sys
import time
import threading
import logging
import argparse
import os
import traceback
from typing import Optional, List, Dict, Any
import torch

# Resource Monitoring Import (Keep)
try:
    import psutil
except ImportError:
    print("Warning: psutil not found. CPU/Memory usage monitoring will be disabled.")
    print("Install it using: pip install psutil")
    psutil = None

# Path Adjustment (Keep)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Config Imports (Updated)
try:
    from config import (
        VisConfig,
        EnvConfig,
        # Removed PPOConfig
        ModelConfig,
        StatsConfig,
        TrainConfig,
        TensorBoardConfig,
        DemoConfig,
        RNNConfig,
        TransformerConfig,
        # Removed ObsNormConfig
        # Removed RewardConfig
        DEVICE,
        RANDOM_SEED,
        # Removed TOTAL_TRAINING_STEPS
        BASE_CHECKPOINT_DIR,
        BASE_LOG_DIR,
        set_device,
        get_run_id,
        set_run_id,
        get_run_checkpoint_dir,
        get_run_log_dir,
        get_console_log_dir,
        print_config_info_and_validate,
        get_config_dict,
    )
    from utils.helpers import get_device as get_torch_device, set_random_seeds
    from logger import TeeLogger
    from utils.init_checks import run_pre_checks
    from utils.types import AgentStateDict  # Keep for potential NN state dict type

except ImportError as e:
    print(f"Error importing configuration classes or utils: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during config/util import: {e}")
    traceback.print_exc()
    sys.exit(1)

# Component Imports (Updated)
try:
    from environment.game_state import GameState
    from ui.renderer import UIRenderer

    # Removed PPOAgent import
    # Removed RolloutStorage, RolloutCollector imports
    # Removed worker imports (EnvironmentRunner, TrainingWorker)

    # Keep stats imports
    from stats import (
        StatsRecorderBase,
        SimpleStatsRecorder,
        TensorBoardStatsRecorder,
        StatsAggregator,
    )

    # Keep CheckpointManager import (will manage NN weights now)
    from training.checkpoint_manager import (
        CheckpointManager,
        find_latest_run_and_checkpoint,
    )
    from app_state import AppState
    from app_init import AppInitializer
    from app_logic import AppLogic
    from app_workers import AppWorkerManager  # Keep manager structure
    from app_setup import (
        initialize_pygame,
        initialize_directories,
        load_and_validate_configs,
    )
    from app_ui_utils import AppUIUtils
    from ui.input_handler import InputHandler
    import queue  # Keep queue if needed for future workers

except ImportError as e:
    print(f"Error importing application components: {e}")
    traceback.print_exc()
    sys.exit(1)

# Logging Setup (Keep)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MainApp:
    """Main application class orchestrating Pygame UI, game logic, and potentially AlphaZero components."""

    def __init__(self, checkpoint_to_load: Optional[str] = None):
        # --- Configuration (Updated) ---
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        # Removed self.ppo_config
        self.rnn_config = RNNConfig()
        self.train_config_instance = TrainConfig()
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        # Removed self.reward_config
        # Removed self.obs_norm_config
        self.transformer_config = TransformerConfig()
        self.config_dict = get_config_dict()

        # --- Core Components (Keep) ---
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None

        # --- State (Updated) ---
        self.app_state: AppState = AppState.INITIALIZING
        # Removed self.is_process_running
        self.status: str = "Initializing..."
        self.running: bool = True
        self.update_progress_details: Dict[str, Any] = (
            {}
        )  # Keep for potential NN training progress

        # --- Threading & Communication (Updated) ---
        self.stop_event = threading.Event()  # Keep for main loop exit signal
        # Removed self.pause_event (was tied to PPO workers)
        # Removed self.experience_queue

        # --- RL Components (Managed by Initializer - Updated) ---
        # Placeholders for NN agent, stats, checkpoint manager
        self.envs: List[GameState] = []  # Only for visualization now
        self.agent: Optional[Any] = None  # Placeholder for AlphaZero NN Agent
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        # Removed self.rollout_collector
        self.demo_env: Optional[GameState] = None

        # --- Helper Classes (Keep) ---
        self.device = get_torch_device()
        set_device(self.device)
        self.checkpoint_to_load = checkpoint_to_load
        self.initializer = AppInitializer(self)
        self.logic = AppLogic(self)
        self.worker_manager = AppWorkerManager(self)  # Manages future workers
        self.ui_utils = AppUIUtils(self)

        # --- UI State (Keep) ---
        self.cleanup_confirmation_active: bool = False
        self.cleanup_message: str = ""
        self.last_cleanup_message_time: float = 0.0
        self.total_gpu_memory_bytes: Optional[int] = None

        # --- Resource Monitoring (Keep) ---
        self.last_resource_update_time: float = 0.0
        self.resource_update_interval: float = 1.0

    def initialize(self):
        """Initializes Pygame, directories, configs, and core components."""
        logger.info("--- Application Initialization ---")
        self.screen, self.clock = initialize_pygame(self.vis_config)
        initialize_directories()
        # Configs are instantiated in __init__

        set_random_seeds(RANDOM_SEED)
        run_pre_checks()

        # Initialize core components via AppInitializer (now initializes NN, Stats, etc.)
        self.app_state = AppState.INITIALIZING
        self.initializer.initialize_all()

        # Set input handler reference in renderer (Keep)
        if self.renderer and self.initializer.app.input_handler:
            self.renderer.set_input_handler(self.initializer.app.input_handler)

        # Start worker threads (Placeholder - will start MCTS/NN workers later)
        self.worker_manager.start_worker_threads()

        # Check initial completion status (Adapt later if needed)
        self.logic.check_initial_completion_status()
        self.status = "Ready"  # Default status after init
        self.app_state = AppState.MAIN_MENU

        logger.info("--- Initialization Complete ---")
        if self.tensorboard_config.LOG_DIR:
            tb_path = os.path.abspath(get_run_log_dir())
            logger.info(f"--- TensorBoard logs: tensorboard --logdir {tb_path} ---")

    def _update_resource_stats(self):
        """Updates CPU, Memory, and GPU usage in the StatsAggregator."""
        current_time = time.time()
        if (
            current_time - self.last_resource_update_time
            < self.resource_update_interval
        ):
            return

        # Check if stats recorder and aggregator exist
        if not self.initializer.stats_recorder or not hasattr(
            self.initializer.stats_recorder, "aggregator"
        ):
            return

        aggregator = self.initializer.stats_recorder.aggregator
        storage = aggregator.storage

        cpu_percent, mem_percent, gpu_mem_percent = 0.0, 0.0, 0.0

        if psutil:
            try:
                cpu_percent = psutil.cpu_percent()
                mem_percent = psutil.virtual_memory().percent
            except Exception as e:
                logger.warning(f"Error getting CPU/Mem usage: {e}")

        if self.device.type == "cuda" and self.total_gpu_memory_bytes:
            try:
                allocated = torch.cuda.memory_allocated(self.device)
                gpu_mem_percent = (allocated / self.total_gpu_memory_bytes) * 100.0
            except Exception as e:
                logger.warning(f"Error getting GPU memory usage: {e}")
                gpu_mem_percent = 0.0

        # Update aggregator storage directly (thread-safe via lock)
        with aggregator._lock:
            storage.current_cpu_usage = cpu_percent
            storage.current_memory_usage = mem_percent
            storage.current_gpu_memory_usage_percent = gpu_mem_percent
            # Append to deques if they exist (check needed after refactor)
            if hasattr(storage, "cpu_usage"):
                storage.cpu_usage.append(cpu_percent)
            if hasattr(storage, "memory_usage"):
                storage.memory_usage.append(mem_percent)
            if hasattr(storage, "gpu_memory_usage_percent"):
                storage.gpu_memory_usage_percent.append(gpu_mem_percent)

        self.last_resource_update_time = current_time

    def run_main_loop(self):
        """The main application loop."""
        logger.info("Starting main application loop...")
        while self.running:
            dt = self.clock.tick(self.vis_config.FPS) / 1000.0

            # Handle Input (Keep)
            if self.input_handler:
                self.running = self.input_handler.handle_input(
                    self.app_state.value, self.cleanup_confirmation_active
                )
                if not self.running:
                    self.stop_event.set()
                    break
            else:  # Fallback exit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        self.stop_event.set()
                        break
                if not self.running:
                    break

            # --- Fetch Update Progress Details (Keep for potential NN training) ---
            # This needs to be adapted to get progress from the NN training worker if implemented
            self.update_progress_details = {}  # Placeholder
            # Example:
            # if self.worker_manager.nn_training_worker_thread and self.worker_manager.nn_training_worker_thread.is_alive():
            #     try:
            #         self.update_progress_details = self.worker_manager.nn_training_worker_thread.get_update_progress_details()
            #     except Exception as e:
            #         logger.warning(f"Could not get update progress details: {e}")
            #         self.update_progress_details = {}
            # else:
            #     self.update_progress_details = {}
            # --- End Fetch Update Progress Details ---

            # Update Logic & State (Simplified)
            self.logic.update_status_and_check_completion()

            # Update Resource Stats (Keep)
            self._update_resource_stats()

            # Update Demo Env Timers (Keep)
            if self.initializer.demo_env:
                try:
                    # Timers are updated internally by demo_env.step or manually if needed
                    # self.initializer.demo_env._update_timers() # Call if step isn't called automatically
                    pass  # Assume timers update within demo logic or step
                except Exception as timer_err:
                    logger.error(
                        f"Error updating demo env timers: {timer_err}", exc_info=False
                    )

            # Render UI (Adapted)
            if self.renderer:
                plot_data = {}
                stats_summary = {}
                tb_log_dir = None
                agent_params = 0
                # Removed worker_counts

                if self.initializer.stats_recorder:
                    plot_data = self.initializer.stats_recorder.get_plot_data()
                    current_step = 0
                    if hasattr(self.initializer.stats_recorder, "aggregator"):
                        current_step = getattr(
                            self.initializer.stats_recorder.aggregator.storage,
                            "current_global_step",
                            0,
                        )
                    stats_summary = self.initializer.stats_recorder.get_summary(
                        current_step
                    )
                    if isinstance(
                        self.initializer.stats_recorder, TensorBoardStatsRecorder
                    ):
                        tb_log_dir = self.initializer.stats_recorder.log_dir

                if self.initializer.agent:
                    agent_params = self.initializer.agent_param_count

                # Adapt render_all call
                self.renderer.render_all(
                    app_state=self.app_state.value,
                    is_process_running=False,  # No PPO process running
                    status=self.status,
                    stats_summary=stats_summary,
                    envs=self.initializer.envs,  # Pass empty list or visualized envs
                    num_envs=self.env_config.NUM_ENVS,  # Pass total configured (for layout)
                    env_config=self.env_config,
                    cleanup_confirmation_active=self.cleanup_confirmation_active,
                    cleanup_message=self.cleanup_message,
                    last_cleanup_message_time=self.last_cleanup_message_time,
                    tensorboard_log_dir=tb_log_dir,
                    plot_data=plot_data,
                    demo_env=self.initializer.demo_env,
                    update_progress_details=self.update_progress_details,  # Pass NN progress later
                    agent_param_count=agent_params,
                    worker_counts={},  # Removed worker counts
                )
            else:  # Fallback render
                self.screen.fill((20, 0, 0))
                font = pygame.font.Font(None, 30)
                text_surf = font.render("Renderer Error", True, (255, 50, 50))
                self.screen.blit(
                    text_surf, text_surf.get_rect(center=self.screen.get_rect().center)
                )
                pygame.display.flip()

        logger.info("Main application loop exited.")

    def shutdown(self):
        """Cleans up resources and exits."""
        logger.info("Initiating shutdown sequence...")

        # Signal threads to stop (Keep stop_event)
        logger.info("Setting stop event for worker threads (if any).")
        self.stop_event.set()
        # Removed pause_event clearing

        # Stop and join worker threads (Placeholder)
        logger.info("Stopping worker threads (if any)...")
        self.worker_manager.stop_worker_threads()
        logger.info("Worker threads stopped.")

        # Save final checkpoint (NN weights, stats)
        logger.info("Attempting final checkpoint save...")
        self.logic.save_final_checkpoint()
        logger.info("Final checkpoint save attempt finished.")

        # Close stats recorder (Keep)
        logger.info("Closing stats recorder (before pygame.quit)...")
        self.initializer.close_stats_recorder()
        logger.info("Stats recorder closed.")

        # Quit Pygame (Keep)
        logger.info("Quitting Pygame...")
        pygame.quit()
        logger.info("Pygame quit.")
        logger.info("Shutdown complete.")


# Global variable for TeeLogger instance (Keep)
tee_logger_instance: Optional[TeeLogger] = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlphaTri Trainer"
    )  # Updated description
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        # Updated help text
        help="Path to a specific checkpoint file to load (e.g., NN weights). Overrides auto-resume.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    # Setup Logging (Keep, checkpoint finding logic might adapt)
    if args.load_checkpoint:
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                print(f"Using Run ID from checkpoint path: {get_run_id()}")
            else:
                get_run_id()
        except Exception:
            get_run_id()
    else:
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            print(f"Resuming Run ID: {get_run_id()}")
        else:
            get_run_id()

    # Setup TeeLogger (Keep)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        log_file_dir = get_console_log_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, "console_output.log")
        tee_logger_instance = TeeLogger(log_file_path, sys.stdout)
        sys.stdout = tee_logger_instance
        sys.stderr = tee_logger_instance
    except Exception as e:
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)

    # Set logging level (Keep)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.info(f"Logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {get_run_id()}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")

    # Create and Run Application (Keep structure)
    app = None
    exit_code = 0
    try:
        app = MainApp(checkpoint_to_load=args.load_checkpoint)
        app.initialize()
        app.run_main_loop()
        app.shutdown()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down.")
        if app:
            app.shutdown()
        else:
            pygame.quit()
        exit_code = 130
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in main: {e}", exc_info=True)
        if app:
            app.shutdown()
        else:
            pygame.quit()
        exit_code = 1
    finally:
        # Restore stdout/stderr (Keep)
        print("[Main Finally] Restoring stdout/stderr and closing logger...")
        if tee_logger_instance:
            try:
                if isinstance(sys.stdout, TeeLogger):
                    sys.stdout.flush()
                if isinstance(sys.stderr, TeeLogger):
                    sys.stderr.flush()
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                tee_logger_instance.close()
                print("[Main Finally] TeeLogger closed and streams restored.")
            except Exception as log_close_err:
                original_stdout.write(f"ERROR closing TeeLogger: {log_close_err}\n")
                traceback.print_exc(file=original_stderr)
        print(f"[Main Finally] Exiting with code {exit_code}.")
        sys.exit(exit_code)


File: out.md
File: analyze_profile.py
# File: analyze_profile_v2.py
import pstats
from pstats import SortKey

profile_file = "profile_output.prof"
output_file_cumulative = "profile_summary_cumulative.txt"
output_file_tottime = "profile_summary_tottime.txt"
num_lines_to_print = 50  # You can adjust how many lines to show

try:
    # --- Sort by Cumulative Time ---
    print(
        f"Saving top {num_lines_to_print} cumulative time stats to {output_file_cumulative}..."
    )
    with open(output_file_cumulative, "w") as f_cum:
        # Pass the file handle directly as the stream
        stats_cum = pstats.Stats(profile_file, stream=f_cum)
        stats_cum.sort_stats(SortKey.CUMULATIVE).print_stats(num_lines_to_print)
        # 'with open' handles closing/flushing
    print("Done.")

    # --- Sort by Total Time (Internal) ---
    print(
        f"Saving top {num_lines_to_print} total time (tottime) stats to {output_file_tottime}..."
    )
    with open(output_file_tottime, "w") as f_tot:
        # Pass the file handle directly as the stream
        stats_tot = pstats.Stats(profile_file, stream=f_tot)
        stats_tot.sort_stats(SortKey.TIME).print_stats(
            num_lines_to_print
        )  # SortKey.TIME is 'tottime'
        # 'with open' handles closing/flushing
    print("Done.")

    print(
        f"\nAnalysis complete. Check '{output_file_cumulative}' and '{output_file_tottime}'."
    )

except FileNotFoundError:
    print(f"ERROR: Profile file '{profile_file}' not found.")
except Exception as e:
    print(f"An error occurred during profile analysis: {e}")


File: app_init.py
# File: app_init.py
import pygame
import time
import traceback
import sys
import torch
from typing import TYPE_CHECKING, List, Optional, Dict, Any

from config import (
    VisConfig,
    EnvConfig,
    # Removed PPOConfig
    RNNConfig,
    ModelConfig,
    StatsConfig,
    # Removed RewardConfig
    TensorBoardConfig,
    DemoConfig,
    # Removed ObsNormConfig
    TransformerConfig,
    RANDOM_SEED,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
    DEVICE,
)
from environment.game_state import GameState

# Removed PPOAgent import
# from agent.ppo_agent import PPOAgent
from stats.stats_recorder import StatsRecorderBase
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler

# Removed init.rl_components_ppo import
from training.checkpoint_manager import CheckpointManager

# Removed RolloutCollector import
from app_state import AppState

if TYPE_CHECKING:
    from main_pygame import MainApp
    from agent.base_agent import BaseAgent  # Hypothetical base class for NN agent


class AppInitializer:
    """Handles the initialization of core application components."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Config instances
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        # Removed self.ppo_config
        self.rnn_config = RNNConfig()
        self.train_config = app.train_config_instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        # Removed self.reward_config
        # Removed self.obs_norm_config
        self.transformer_config = TransformerConfig()

        # Components to be initialized
        self.envs: List[GameState] = []  # Keep for potential multi-env display
        self.agent: Optional["BaseAgent"] = None  # Agent is now the NN
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        # Removed self.rollout_collector

    def initialize_all(self, is_reinit: bool = False):
        """Initializes all core components."""
        try:
            # GPU Memory Info (Keep)
            if self.app.device.type == "cuda":
                try:
                    self.app.total_gpu_memory_bytes = torch.cuda.get_device_properties(
                        self.app.device
                    ).total_memory
                    print(
                        f"Total GPU Memory: {self.app.total_gpu_memory_bytes / (1024**3):.2f} GB"
                    )
                except Exception as e:
                    print(f"Warning: Could not get total GPU memory: {e}")

            # Renderer and Initial Render (Keep)
            if not is_reinit:
                self.app.renderer = UIRenderer(self.app.screen, self.vis_config)
                # Adapt render_all call later if needed
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=False,  # No PPO process running
                    status=self.app.status,
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
                    update_progress_details={},  # Keep for potential NN training progress
                    agent_param_count=0,
                    worker_counts={},  # Remove worker counts for now
                )
                pygame.display.flip()
                pygame.time.delay(100)

            # Initialize "RL" components (NN, Stats, Checkpoint Manager)
            self.initialize_rl_components(
                is_reinit=is_reinit, checkpoint_to_load=self.app.checkpoint_to_load
            )

            # Demo Env and Input Handler (Keep)
            if not is_reinit:
                self.initialize_demo_env()
                self.initialize_input_handler()

            # Calculate NN parameter count if agent exists
            if self.agent and hasattr(self.agent, "network"):
                try:
                    self.agent_param_count = sum(
                        p.numel()
                        for p in self.agent.network.parameters()
                        if p.requires_grad
                    )
                except Exception as e:
                    print(f"Warning: Could not calculate agent parameters: {e}")
                    self.agent_param_count = 0

        except Exception as init_err:
            print(f"FATAL ERROR during component initialization: {init_err}")
            traceback.print_exc()
            if self.app.renderer:
                try:
                    self.app.app_state = AppState.ERROR
                    self.app.status = "Initialization Failed"
                    self.app.renderer._render_error_screen(self.app.status)
                    pygame.display.flip()
                    time.sleep(5)
                except Exception:
                    pass
            pygame.quit()
            sys.exit(1)

    def initialize_rl_components(
        self, is_reinit: bool = False, checkpoint_to_load: Optional[str] = None
    ):
        """Initializes NN Agent, Stats Recorder, Checkpoint Manager."""
        print(f"Initializing AlphaZero components... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            # Initialize Envs (only needed for visualization now)
            # self.envs = initialize_envs(self.env_config.NUM_ENVS, self.env_config) # Removed multi-env init
            self.envs = []  # No parallel envs needed for core logic now

            # --- Initialize Agent (Neural Network) ---
            # Replace with AlphaZero NN initialization later
            # For now, set agent to None or a placeholder
            self.agent = None  # Placeholder - Initialize AlphaZero NN here later
            print("Agent (NN) initialization SKIPPED (placeholder).")
            # Example placeholder for future NN init:
            # self.agent = initialize_alphazero_agent(
            #     model_config=self.model_config,
            #     rnn_config=self.rnn_config,
            #     env_config=self.env_config,
            #     transformer_config=self.transformer_config,
            #     device=self.app.device,
            # )
            # --- End Agent Init ---

            # --- Initialize Stats Recorder ---
            # Adapt initialize_stats_recorder if needed (e.g., remove PPO hparams)
            # Need to create a simplified version or adapt existing one
            # For now, assume it's adapted or create a placeholder
            try:
                # Assuming init.stats_init exists and is adapted
                from init.stats_init import initialize_stats_recorder

                self.stats_recorder = initialize_stats_recorder(
                    stats_config=self.stats_config,
                    tb_config=self.tensorboard_config,
                    config_dict=self.app.config_dict,
                    # Pass agent=None if NN not ready, or pass the NN agent
                    agent=self.agent,
                    env_config=self.env_config,
                    rnn_config=self.rnn_config,  # Keep for potential NN config logging
                    transformer_config=self.transformer_config,  # Keep for potential NN config logging
                    is_reinit=is_reinit,
                )
            except ImportError:
                print(
                    "Warning: init.stats_init.initialize_stats_recorder not found. Skipping stats recorder init."
                )
                self.stats_recorder = None  # Fallback
            except Exception as stats_init_err:
                print(f"Error initializing stats recorder: {stats_init_err}")
                traceback.print_exc()
                self.stats_recorder = None

            if self.stats_recorder is None:
                print("Warning: Stats Recorder initialization failed or skipped.")
                # Decide if this is critical - maybe allow running without stats?
                # raise RuntimeError("Stats Recorder init failed.")
            # --- End Stats Recorder Init ---

            # --- Initialize Checkpoint Manager ---
            # Checkpoint manager now handles NN agent state and stats aggregator state
            self.checkpoint_manager = CheckpointManager(
                # Pass the NN agent (or None if not ready)
                agent=self.agent,
                # Pass stats aggregator if it exists
                stats_aggregator=getattr(self.stats_recorder, "aggregator", None),
                base_checkpoint_dir=BASE_CHECKPOINT_DIR,
                run_checkpoint_dir=get_run_checkpoint_dir(),
                load_checkpoint_path_config=checkpoint_to_load,
                device=self.app.device,
                # obs_rms_dict=None, # Removed Obs RMS
            )
            # --- End Checkpoint Manager Init ---

            # --- Load Checkpoint ---
            if self.checkpoint_manager.get_checkpoint_path_to_load():
                self.checkpoint_manager.load_checkpoint()  # Loads NN state and stats state
                # Get initial step/episode count from loaded stats
                loaded_global_step, initial_episode_count = (
                    self.checkpoint_manager.get_initial_state()
                )
                # Sync episode count if needed (e.g., if MCTS tracks episodes)
                # self.mcts_manager.state.episode_count = initial_episode_count # Example
            # --- End Load Checkpoint ---

            print(
                f"AlphaZero components initialized in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            print(f"Error during AlphaZero component initialization: {e}")
            traceback.print_exc()
            raise e  # Re-raise to be caught by initialize_all

    def initialize_demo_env(self):
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

    def initialize_input_handler(self):
        """Initializes the Input Handler."""
        if not self.app.renderer:
            print("ERROR: Cannot initialize InputHandler before Renderer.")
            return

        self.app.input_handler = InputHandler(
            screen=self.app.screen,
            renderer=self.app.renderer,
            # Removed toggle_training_run_cb
            request_cleanup_cb=self.app.logic.request_cleanup,
            cancel_cleanup_cb=self.app.logic.cancel_cleanup,
            confirm_cleanup_cb=self.app.logic.confirm_cleanup,
            exit_app_cb=self.app.logic.exit_app,
            start_demo_mode_cb=self.app.logic.start_demo_mode,
            exit_demo_mode_cb=self.app.logic.exit_demo_mode,
            handle_demo_mouse_motion_cb=self.app.logic.handle_demo_mouse_motion,
            handle_demo_mouse_button_down_cb=self.app.logic.handle_demo_mouse_button_down,
            start_debug_mode_cb=self.app.logic.start_debug_mode,
            exit_debug_mode_cb=self.app.logic.exit_debug_mode,
            handle_debug_input_cb=self.app.logic.handle_debug_input,
        )
        if self.app.renderer and self.app.renderer.left_panel:
            self.app.renderer.left_panel.input_handler = self.app.input_handler
            if hasattr(self.app.renderer.left_panel, "button_status_renderer"):
                self.app.renderer.left_panel.button_status_renderer.input_handler_ref = (
                    self.app.input_handler
                )

    def close_stats_recorder(self, is_cleanup: bool = False):
        """Safely closes the stats recorder."""
        print(
            f"[AppInitializer] close_stats_recorder called (is_cleanup={is_cleanup})..."
        )
        if self.stats_recorder:
            print("[AppInitializer] Stats recorder exists, attempting close...")
            try:
                if hasattr(self.stats_recorder, "close"):
                    self.stats_recorder.close(is_cleanup=is_cleanup)
                    print("[AppInitializer] stats_recorder.close() executed.")
                else:
                    print("[AppInitializer] stats_recorder has no close method.")
            except Exception as log_e:
                print(f"[AppInitializer] Error closing stats recorder on exit: {log_e}")
                traceback.print_exc()
        else:
            print("[AppInitializer] No stats recorder instance to close.")
        print("[AppInitializer] close_stats_recorder finished.")


File: app_logic.py
# File: app_logic.py
import pygame
import time
import traceback
import os
import shutil
from typing import TYPE_CHECKING, Tuple

from app_state import AppState

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppLogic:
    """Handles the core application logic and state transitions."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def check_initial_completion_status(self):
        """Checks if training target (e.g., games played) was met upon loading."""
        # This needs adaptation based on how AlphaZero training progress is tracked.
        # For now, assume completion isn't automatically checked this way.
        # Example: Check against games played or NN training steps in aggregator
        # if (
        #     self.app.initializer.stats_recorder
        #     and hasattr(self.app.initializer.stats_recorder, "aggregator")
        # ):
        #     aggregator = self.app.initializer.stats_recorder.aggregator
        #     target_games = getattr(aggregator.storage, "training_target_games", 0) # Example target
        #     current_games = getattr(aggregator.storage, "total_episodes", 0)
        #     if target_games > 0 and current_games >= target_games:
        #         self.app.status = "Target Reached"
        #         print(f"Target games already reached ({current_games:,}/{target_games:,}). Ready.")
        pass  # Keep simple for now

    def update_status_and_check_completion(self):
        """Updates the status text based on application state."""
        # Removed PPO-specific status logic (Collecting, Updating)
        # Removed completion check based on PPO steps

        # Update status based on AppState
        if self.app.app_state == AppState.MAIN_MENU:
            if self.app.cleanup_confirmation_active:
                self.app.status = "Confirm Cleanup"
            else:
                # Check if training is "complete" based on new criteria if needed
                # For now, just set to Ready if not confirming cleanup
                self.app.status = "Ready"
        elif self.app.app_state == AppState.PLAYING:
            self.app.status = "Playing Demo"
        elif self.app.app_state == AppState.DEBUG:
            self.app.status = "Debugging Grid"
        elif self.app.app_state == AppState.INITIALIZING:
            self.app.status = "Initializing..."
        elif self.app.app_state == AppState.ERROR:
            # Status should already be set by the error handler
            pass

    # Removed toggle_training_run method

    def request_cleanup(self):
        # Removed check for is_process_running
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot request cleanup outside MainMenu.")
            return
        self.app.cleanup_confirmation_active = True
        self.app.status = "Confirm Cleanup"  # Update status
        print("Cleanup requested. Confirm action.")

    def start_demo_mode(self):
        # Removed check for is_process_running
        if self.app.initializer.demo_env is None:
            print("Cannot start demo mode: Demo environment failed to initialize.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot start demo mode outside MainMenu.")
            return
        print("Entering Demo Mode...")
        self.try_save_checkpoint()  # Save NN weights before switching mode?
        self.app.app_state = AppState.PLAYING
        self.app.status = "Playing Demo"
        self.app.initializer.demo_env.reset()

    def start_debug_mode(self):
        # Removed check for is_process_running
        if self.app.initializer.demo_env is None:
            print("Cannot start debug mode: Demo environment failed to initialize.")
            return
        if self.app.app_state != AppState.MAIN_MENU:
            print("Cannot start debug mode outside MainMenu.")
            return
        print("Entering Debug Mode...")
        self.try_save_checkpoint()  # Save NN weights before switching mode?
        self.app.app_state = AppState.DEBUG
        self.app.status = "Debugging Grid"
        self.app.initializer.demo_env.reset()

    def exit_debug_mode(self):
        if self.app.app_state == AppState.DEBUG:
            print("Exiting Debug Mode...")
            self.app.app_state = AppState.MAIN_MENU
            # Removed setting is_process_running and pause_event
            self.check_initial_completion_status()  # Re-check status
            self.app.status = "Ready"  # Set status back to Ready

    def cancel_cleanup(self):
        self.app.cleanup_confirmation_active = False
        self.app.cleanup_message = "Cleanup cancelled."
        self.app.last_cleanup_message_time = time.time()
        self.app.status = "Ready"  # Set status back
        print("Cleanup cancelled by user.")

    def confirm_cleanup(self):
        print("Cleanup confirmed by user. Starting process...")
        try:
            self._cleanup_data()
        except Exception as e:
            print(f"FATAL ERROR during cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Cleanup Failed Critically"
            self.app.app_state = AppState.ERROR
        finally:
            self.app.cleanup_confirmation_active = False
            # Status is set within _cleanup_data or error handling
            print(
                f"Cleanup process finished. State: {self.app.app_state}, Status: {self.app.status}"
            )

    def exit_app(self) -> bool:
        print("Exit requested.")
        self.app.stop_event.set()  # Keep stop event for main loop exit
        return False

    def exit_demo_mode(self):
        if self.app.app_state == AppState.PLAYING:
            print("Exiting Demo Mode...")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.deselect_dragged_shape()
            self.app.app_state = AppState.MAIN_MENU
            # Removed setting is_process_running and pause_event
            self.check_initial_completion_status()  # Re-check status
            self.app.status = "Ready"  # Set status back to Ready

    def handle_demo_mouse_motion(self, mouse_pos: Tuple[int, int]):
        if (
            self.app.app_state != AppState.PLAYING
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if demo_env.demo_dragged_shape_idx is None:
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        demo_env.update_snapped_position(grid_coords)

    def handle_demo_mouse_button_down(self, event: pygame.event.Event):
        if (
            self.app.app_state != AppState.PLAYING
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if event.button != 1:
            return
        mouse_pos = event.pos
        clicked_preview_index = self.app.ui_utils.map_screen_to_preview(mouse_pos)
        if clicked_preview_index is not None:
            if clicked_preview_index == demo_env.demo_dragged_shape_idx:
                demo_env.deselect_dragged_shape()
            else:
                demo_env.select_shape_for_drag(clicked_preview_index)
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        if grid_coords is not None:
            if (
                demo_env.demo_dragged_shape_idx is not None
                and demo_env.demo_snapped_position == grid_coords
            ):
                placed = demo_env.place_dragged_shape()
                if placed and demo_env.is_over():
                    print("[Demo] Game Over! Press ESC to exit.")
            else:
                demo_env.deselect_dragged_shape()
            return
        demo_env.deselect_dragged_shape()

    def handle_debug_input(self, event: pygame.event.Event):
        if (
            self.app.app_state != AppState.DEBUG
            or self.app.initializer.demo_env is None
        ):
            return
        demo_env = self.app.initializer.demo_env
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                print("[Debug] Resetting grid...")
                demo_env.reset()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            clicked_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
            if clicked_coords:
                row, col = clicked_coords
                demo_env.toggle_triangle_debug(row, col)

    def _cleanup_data(self):
        """Deletes current run's checkpoint and re-initializes components."""
        from config.general import get_run_checkpoint_dir  # Local import

        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        # Removed pause_event setting and is_process_running
        self.app.app_state = AppState.INITIALIZING
        self.app.status = "Cleaning"
        messages = []

        # Render cleaning status
        if self.app.renderer:
            try:
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=False,  # No process running now
                    status=self.app.status,
                    stats_summary={},
                    envs=[],
                    num_envs=0,
                    env_config=self.app.env_config,
                    cleanup_confirmation_active=False,
                    cleanup_message="",
                    last_cleanup_message_time=0,
                    tensorboard_log_dir=None,
                    plot_data={},
                    demo_env=self.app.initializer.demo_env,
                    update_progress_details={},
                    agent_param_count=getattr(
                        self.app.initializer, "agent_param_count", 0
                    ),
                    worker_counts={},  # Removed worker counts
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

        # --- Stop existing worker threads (if any) ---
        # Keep this structure in case new workers (MCTS/NN) are added later
        print("[Cleanup] Stopping existing worker threads (if any)...")
        self.app.worker_manager.stop_worker_threads()
        print("[Cleanup] Existing worker threads stopped.")
        # --- End Stop Workers ---

        # --- Close Stats Recorder ---
        print("[Cleanup] Closing stats recorder...")
        self.app.initializer.close_stats_recorder(is_cleanup=True)
        print("[Cleanup] Stats recorder closed.")
        # --- End Close Stats ---

        # --- Delete Checkpoint Directory ---
        print("[Cleanup] Deleting agent checkpoint file/dir...")
        try:
            save_dir = get_run_checkpoint_dir()
            if os.path.isdir(save_dir):
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
        print("[Cleanup] Checkpoint deletion attempt finished.")
        # --- End Delete Checkpoint ---

        time.sleep(0.1)
        print("[Cleanup] Re-initializing components...")
        try:
            # Re-initialize components (NN, Stats, Checkpoint Manager)
            self.app.initializer.initialize_rl_components(
                is_reinit=True, checkpoint_to_load=None
            )
            print("[Cleanup] Components re-initialized.")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
                print("[Cleanup] Demo env reset.")

            # --- Start new worker threads (if any) ---
            # Keep this structure for future workers
            print("[Cleanup] Starting new worker threads (if any)...")
            self.app.worker_manager.start_worker_threads()
            print("[Cleanup] New worker threads started.")
            # --- End Start Workers ---

            print("[Cleanup] Component re-initialization and worker start successful.")
            messages.append("Components re-initialized.")

            # --- Set state after cleanup ---
            # Removed is_process_running and pause_event
            self.app.status = "Ready"
            self.app.app_state = AppState.MAIN_MENU
            print("[Cleanup] Application state set to MAIN_MENU.")
            # --- End Set State ---

        except Exception as e:
            print(f"FATAL ERROR during re-initialization after cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Re-init Failed"
            self.app.app_state = AppState.ERROR
            messages.append("ERROR RE-INITIALIZING COMPONENTS!")
            if self.app.renderer:
                try:
                    self.app.renderer._render_error_screen(self.app.status)
                except Exception as render_err_final:
                    print(f"Warning: Failed to render error screen: {render_err_final}")

        self.app.cleanup_message = "\n".join(messages)
        self.app.last_cleanup_message_time = time.time()
        print(
            f"--- CLEANUP DATA COMPLETE (Final State: {self.app.app_state}, Status: {self.app.status}) ---"
        )

    def try_save_checkpoint(self):
        """Saves checkpoint (e.g., NN weights) if in main menu."""
        # Removed check for is_process_running
        if (
            self.app.app_state == AppState.MAIN_MENU
            and self.app.initializer.checkpoint_manager
            and self.app.initializer.stats_recorder  # Check if stats exist
            and hasattr(self.app.initializer.stats_recorder, "aggregator")
        ):
            print("Saving checkpoint...")
            try:
                # Get step/episode count from aggregator
                current_step = getattr(
                    self.app.initializer.stats_recorder.aggregator.storage,
                    "current_global_step",
                    0,
                )
                episode_count = getattr(
                    self.app.initializer.stats_recorder.aggregator.storage,
                    "total_episodes",
                    0,
                )
                # Target step is now managed within aggregator/checkpoint manager
                target_step = getattr(
                    self.app.initializer.checkpoint_manager, "training_target_step", 0
                )

                self.app.initializer.checkpoint_manager.save_checkpoint(
                    current_step,
                    episode_count,
                    training_target_step=target_step,
                    is_final=False,
                )
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                traceback.print_exc()

    def save_final_checkpoint(self):
        """Saves the final checkpoint (e.g., NN weights)."""
        if (
            self.app.initializer.checkpoint_manager
            and self.app.initializer.stats_recorder
            and hasattr(self.app.initializer.stats_recorder, "aggregator")
        ):
            save_on_exit = (
                self.app.status != "Cleaning" and self.app.app_state != AppState.ERROR
            )

            if save_on_exit:
                print("Performing final checkpoint save...")
                try:
                    current_step = getattr(
                        self.app.initializer.stats_recorder.aggregator.storage,
                        "current_global_step",
                        0,
                    )
                    episode_count = getattr(
                        self.app.initializer.stats_recorder.aggregator.storage,
                        "total_episodes",
                        0,
                    )
                    target_step = getattr(
                        self.app.initializer.checkpoint_manager,
                        "training_target_step",
                        0,
                    )
                    self.app.initializer.checkpoint_manager.save_checkpoint(
                        current_step,
                        episode_count,
                        training_target_step=target_step,
                        is_final=True,
                    )
                except Exception as final_save_err:
                    print(f"Error during final checkpoint save: {final_save_err}")
                    traceback.print_exc()
            else:
                print("Skipping final checkpoint save.")


File: app_setup.py
# File: app_setup.py
import os
import pygame
from typing import Tuple, Dict, Any

from config import (
    VisConfig,
    get_run_checkpoint_dir,
    get_run_log_dir,
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
    pygame.display.set_caption("AlphaTri Trainer")  # Updated caption
    clock = pygame.time.Clock()
    print("Pygame initialized.")
    return screen, clock


def initialize_directories():
    """Creates necessary runtime directories using dynamic paths."""
    run_checkpoint_dir = get_run_checkpoint_dir()
    run_log_dir = get_run_log_dir()
    # Console log dir is created within main_pygame before logger init

    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"Ensured directories exist: {run_checkpoint_dir}, {run_log_dir}")


def load_and_validate_configs() -> Dict[str, Any]:
    """Loads all config classes and returns the combined config dictionary."""
    config_dict = get_config_dict()
    print_config_info_and_validate()
    return config_dict


File: app_state.py
# File: app_state.py
from enum import Enum, auto


class AppState(Enum):
    INITIALIZING = "Initializing"
    MAIN_MENU = "MainMenu"
    PLAYING = "Playing"  # Demo Mode
    DEBUG = "Debug"
    CLEANUP_CONFIRM = "Confirm Cleanup"  # Not used directly, handled by flag
    CLEANING = "Cleaning"  # Intermediate state during cleanup
    ERROR = "Error"
    UNKNOWN = "Unknown"  # Fallback


File: app_ui_utils.py
# File: app_ui_utils.py
import pygame
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from main_pygame import MainApp
    from environment.game_state import GameState
    from ui.renderer import UIRenderer


class AppUIUtils:
    """Utility functions related to mapping screen coordinates to game elements."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def map_screen_to_grid(
        self, screen_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Maps screen coordinates to grid row/column for demo/debug."""
        if (
            self.app.renderer is None
            or self.app.initializer.demo_env is None
            or self.app.renderer.demo_renderer is None
        ):
            return None
        if self.app.app_state not in [
            self.app.app_state.PLAYING,
            self.app.app_state.DEBUG,
        ]:
            return None

        demo_env: "GameState" = self.app.initializer.demo_env
        renderer: "UIRenderer" = self.app.renderer

        screen_width, screen_height = self.app.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        _, clipped_game_rect = renderer.demo_renderer._calculate_game_area_rect(
            screen_width,
            screen_height,
            padding,
            hud_height,
            help_height,
            self.app.env_config,
        )

        if not clipped_game_rect.collidepoint(screen_pos):
            return None

        relative_x = screen_pos[0] - clipped_game_rect.left
        relative_y = screen_pos[1] - clipped_game_rect.top

        tri_cell_w, tri_cell_h = renderer.demo_renderer._calculate_demo_triangle_size(
            clipped_game_rect.width, clipped_game_rect.height, self.app.env_config
        )
        grid_ox, grid_oy = renderer.demo_renderer._calculate_grid_offset(
            clipped_game_rect.width, clipped_game_rect.height, self.app.env_config
        )

        if tri_cell_w <= 0 or tri_cell_h <= 0:
            return None

        grid_relative_x = relative_x - grid_ox
        grid_relative_y = relative_y - grid_oy

        # Approximate calculation (might need refinement based on triangle geometry)
        approx_row = int(grid_relative_y / tri_cell_h)
        approx_col = int(grid_relative_x / (tri_cell_w * 0.75))

        if (
            0 <= approx_row < self.app.env_config.ROWS
            and 0 <= approx_col < self.app.env_config.COLS
        ):
            if (
                demo_env.grid.valid(approx_row, approx_col)
                and not demo_env.grid.triangles[approx_row][approx_col].is_death
            ):
                return approx_row, approx_col
        return None

    def map_screen_to_preview(self, screen_pos: Tuple[int, int]) -> Optional[int]:
        """Maps screen coordinates to a shape preview index."""
        if (
            self.app.renderer is None
            or self.app.initializer.demo_env is None
            or self.app.input_handler is None
        ):
            return None
        if self.app.app_state != self.app.app_state.PLAYING:
            return None

        # Access preview rects directly from the input handler
        if hasattr(self.app.input_handler, "shape_preview_rects"):
            for idx, rect in self.app.input_handler.shape_preview_rects.items():
                if rect.collidepoint(screen_pos):
                    return idx
        return None


File: app_workers.py
# File: app_workers.py
import threading
import queue
import time
import traceback
from typing import TYPE_CHECKING, Optional  # Added Optional

# Removed worker imports
# from workers import EnvironmentRunner, TrainingWorker

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppWorkerManager:
    """Manages the creation, starting, and stopping of worker threads."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Remove specific worker thread attributes
        # self.env_runner_thread: Optional[EnvironmentRunner] = None
        # self.training_worker_thread: Optional[TrainingWorker] = None
        # Add placeholders for future workers if needed
        self.mcts_worker_thread: Optional[threading.Thread] = None
        self.nn_training_worker_thread: Optional[threading.Thread] = None
        print("[AppWorkerManager] Initialized (No workers started by default).")

    def start_worker_threads(self):
        """Creates and starts worker threads (MCTS, NN Training - Placeholder)."""
        # --- This needs to be implemented based on AlphaZero architecture ---
        print(
            "[AppWorkerManager] start_worker_threads called (Placeholder - No workers started)."
        )
        # Example structure:
        # if not self.app.initializer.agent or not self.app.initializer.mcts_manager:
        #     print("ERROR: Cannot start workers, core components not initialized.")
        #     self.app.app_state = self.app.app_state.ERROR
        #     self.app.status = "Worker Init Failed"
        #     return
        #
        # print("Starting AlphaZero worker threads...")
        # self.app.stop_event.clear()
        # # self.app.pause_event.clear() # Removed pause event
        #
        # # MCTS Self-Play Worker(s)
        # self.mcts_worker_thread = MCTSSelfPlayWorker(...)
        # self.mcts_worker_thread.start()
        #
        # # NN Training Worker
        # self.nn_training_worker_thread = NNTrainingWorker(...)
        # self.nn_training_worker_thread.start()
        #
        # print("AlphaZero worker threads started.")
        pass

    def stop_worker_threads(self):
        """Signals worker threads to stop and waits for them to join."""
        if self.app.stop_event.is_set():
            print("[AppWorkerManager] Stop event already set.")
            return

        print("[AppWorkerManager] Stopping worker threads (Placeholder)...")
        self.app.stop_event.set()
        # self.app.pause_event.clear() # Removed pause event

        join_timeout = 5.0

        # --- Join future worker threads ---
        if self.mcts_worker_thread and self.mcts_worker_thread.is_alive():
            print("[AppWorkerManager] Joining MCTS worker...")
            self.mcts_worker_thread.join(timeout=join_timeout)
            if self.mcts_worker_thread.is_alive():
                print("[AppWorkerManager] MCTS worker thread did not join cleanly.")
            self.mcts_worker_thread = None

        if self.nn_training_worker_thread and self.nn_training_worker_thread.is_alive():
            print("[AppWorkerManager] Joining NN Training worker...")
            self.nn_training_worker_thread.join(timeout=join_timeout)
            if self.nn_training_worker_thread.is_alive():
                print(
                    "[AppWorkerManager] NN Training worker thread did not join cleanly."
                )
            self.nn_training_worker_thread = None
        # --- End Join ---

        # Clear queues if used by workers
        # Example:
        # while not self.app.experience_queue.empty():
        #     try:
        #         self.app.experience_queue.get_nowait()
        #     except queue.Empty:
        #         break

        print("[AppWorkerManager] Worker threads stopped.")


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
import pygame
import sys
import time
import threading
import logging
import argparse
import os
import traceback
from typing import Optional, List, Dict, Any
import torch

# Resource Monitoring Import (Keep)
try:
    import psutil
except ImportError:
    print("Warning: psutil not found. CPU/Memory usage monitoring will be disabled.")
    print("Install it using: pip install psutil")
    psutil = None

# Path Adjustment (Keep)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Config Imports (Updated)
try:
    from config import (
        VisConfig,
        EnvConfig,
        # Removed PPOConfig
        ModelConfig,
        StatsConfig,
        TrainConfig,
        TensorBoardConfig,
        DemoConfig,
        RNNConfig,
        TransformerConfig,
        # Removed ObsNormConfig
        # Removed RewardConfig
        DEVICE,
        RANDOM_SEED,
        # Removed TOTAL_TRAINING_STEPS
        BASE_CHECKPOINT_DIR,
        BASE_LOG_DIR,
        set_device,
        get_run_id,
        set_run_id,
        get_run_checkpoint_dir,
        get_run_log_dir,
        get_console_log_dir,
        print_config_info_and_validate,
        get_config_dict,
    )
    from utils.helpers import get_device as get_torch_device, set_random_seeds
    from logger import TeeLogger
    from utils.init_checks import run_pre_checks
    from utils.types import AgentStateDict  # Keep for potential NN state dict type

except ImportError as e:
    print(f"Error importing configuration classes or utils: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during config/util import: {e}")
    traceback.print_exc()
    sys.exit(1)

# Component Imports (Updated)
try:
    from environment.game_state import GameState
    from ui.renderer import UIRenderer

    # Removed PPOAgent import
    # Removed RolloutStorage, RolloutCollector imports
    # Removed worker imports (EnvironmentRunner, TrainingWorker)

    # Keep stats imports
    from stats import (
        StatsRecorderBase,
        SimpleStatsRecorder,
        TensorBoardStatsRecorder,
        StatsAggregator,
    )

    # Keep CheckpointManager import (will manage NN weights now)
    from training.checkpoint_manager import (
        CheckpointManager,
        find_latest_run_and_checkpoint,
    )
    from app_state import AppState
    from app_init import AppInitializer
    from app_logic import AppLogic
    from app_workers import AppWorkerManager  # Keep manager structure
    from app_setup import (
        initialize_pygame,
        initialize_directories,
        load_and_validate_configs,
    )
    from app_ui_utils import AppUIUtils
    from ui.input_handler import InputHandler
    import queue  # Keep queue if needed for future workers

except ImportError as e:
    print(f"Error importing application components: {e}")
    traceback.print_exc()
    sys.exit(1)

# Logging Setup (Keep)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class MainApp:
    """Main application class orchestrating Pygame UI, game logic, and potentially AlphaZero components."""

    def __init__(self, checkpoint_to_load: Optional[str] = None):
        # --- Configuration (Updated) ---
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        # Removed self.ppo_config
        self.rnn_config = RNNConfig()
        self.train_config_instance = TrainConfig()
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.tensorboard_config = TensorBoardConfig()
        self.demo_config = DemoConfig()
        # Removed self.reward_config
        # Removed self.obs_norm_config
        self.transformer_config = TransformerConfig()
        self.config_dict = get_config_dict()

        # --- Core Components (Keep) ---
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None

        # --- State (Updated) ---
        self.app_state: AppState = AppState.INITIALIZING
        # Removed self.is_process_running
        self.status: str = "Initializing..."
        self.running: bool = True
        self.update_progress_details: Dict[str, Any] = (
            {}
        )  # Keep for potential NN training progress

        # --- Threading & Communication (Updated) ---
        self.stop_event = threading.Event()  # Keep for main loop exit signal
        # Removed self.pause_event (was tied to PPO workers)
        # Removed self.experience_queue

        # --- RL Components (Managed by Initializer - Updated) ---
        # Placeholders for NN agent, stats, checkpoint manager
        self.envs: List[GameState] = []  # Only for visualization now
        self.agent: Optional[Any] = None  # Placeholder for AlphaZero NN Agent
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        # Removed self.rollout_collector
        self.demo_env: Optional[GameState] = None

        # --- Helper Classes (Keep) ---
        self.device = get_torch_device()
        set_device(self.device)
        self.checkpoint_to_load = checkpoint_to_load
        self.initializer = AppInitializer(self)
        self.logic = AppLogic(self)
        self.worker_manager = AppWorkerManager(self)  # Manages future workers
        self.ui_utils = AppUIUtils(self)

        # --- UI State (Keep) ---
        self.cleanup_confirmation_active: bool = False
        self.cleanup_message: str = ""
        self.last_cleanup_message_time: float = 0.0
        self.total_gpu_memory_bytes: Optional[int] = None

        # --- Resource Monitoring (Keep) ---
        self.last_resource_update_time: float = 0.0
        self.resource_update_interval: float = 1.0

    def initialize(self):
        """Initializes Pygame, directories, configs, and core components."""
        logger.info("--- Application Initialization ---")
        self.screen, self.clock = initialize_pygame(self.vis_config)
        initialize_directories()
        # Configs are instantiated in __init__

        set_random_seeds(RANDOM_SEED)
        run_pre_checks()

        # Initialize core components via AppInitializer (now initializes NN, Stats, etc.)
        self.app_state = AppState.INITIALIZING
        self.initializer.initialize_all()

        # Set input handler reference in renderer (Keep)
        if self.renderer and self.initializer.app.input_handler:
            self.renderer.set_input_handler(self.initializer.app.input_handler)

        # Start worker threads (Placeholder - will start MCTS/NN workers later)
        self.worker_manager.start_worker_threads()

        # Check initial completion status (Adapt later if needed)
        self.logic.check_initial_completion_status()
        self.status = "Ready"  # Default status after init
        self.app_state = AppState.MAIN_MENU

        logger.info("--- Initialization Complete ---")
        if self.tensorboard_config.LOG_DIR:
            tb_path = os.path.abspath(get_run_log_dir())
            logger.info(f"--- TensorBoard logs: tensorboard --logdir {tb_path} ---")

    def _update_resource_stats(self):
        """Updates CPU, Memory, and GPU usage in the StatsAggregator."""
        current_time = time.time()
        if (
            current_time - self.last_resource_update_time
            < self.resource_update_interval
        ):
            return

        # Check if stats recorder and aggregator exist
        if not self.initializer.stats_recorder or not hasattr(
            self.initializer.stats_recorder, "aggregator"
        ):
            return

        aggregator = self.initializer.stats_recorder.aggregator
        storage = aggregator.storage

        cpu_percent, mem_percent, gpu_mem_percent = 0.0, 0.0, 0.0

        if psutil:
            try:
                cpu_percent = psutil.cpu_percent()
                mem_percent = psutil.virtual_memory().percent
            except Exception as e:
                logger.warning(f"Error getting CPU/Mem usage: {e}")

        if self.device.type == "cuda" and self.total_gpu_memory_bytes:
            try:
                allocated = torch.cuda.memory_allocated(self.device)
                gpu_mem_percent = (allocated / self.total_gpu_memory_bytes) * 100.0
            except Exception as e:
                logger.warning(f"Error getting GPU memory usage: {e}")
                gpu_mem_percent = 0.0

        # Update aggregator storage directly (thread-safe via lock)
        with aggregator._lock:
            storage.current_cpu_usage = cpu_percent
            storage.current_memory_usage = mem_percent
            storage.current_gpu_memory_usage_percent = gpu_mem_percent
            # Append to deques if they exist (check needed after refactor)
            if hasattr(storage, "cpu_usage"):
                storage.cpu_usage.append(cpu_percent)
            if hasattr(storage, "memory_usage"):
                storage.memory_usage.append(mem_percent)
            if hasattr(storage, "gpu_memory_usage_percent"):
                storage.gpu_memory_usage_percent.append(gpu_mem_percent)

        self.last_resource_update_time = current_time

    def run_main_loop(self):
        """The main application loop."""
        logger.info("Starting main application loop...")
        while self.running:
            dt = self.clock.tick(self.vis_config.FPS) / 1000.0

            # Handle Input (Keep)
            if self.input_handler:
                self.running = self.input_handler.handle_input(
                    self.app_state.value, self.cleanup_confirmation_active
                )
                if not self.running:
                    self.stop_event.set()
                    break
            else:  # Fallback exit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        self.stop_event.set()
                        break
                if not self.running:
                    break

            # --- Fetch Update Progress Details (Keep for potential NN training) ---
            # This needs to be adapted to get progress from the NN training worker if implemented
            self.update_progress_details = {}  # Placeholder
            # Example:
            # if self.worker_manager.nn_training_worker_thread and self.worker_manager.nn_training_worker_thread.is_alive():
            #     try:
            #         self.update_progress_details = self.worker_manager.nn_training_worker_thread.get_update_progress_details()
            #     except Exception as e:
            #         logger.warning(f"Could not get update progress details: {e}")
            #         self.update_progress_details = {}
            # else:
            #     self.update_progress_details = {}
            # --- End Fetch Update Progress Details ---

            # Update Logic & State (Simplified)
            self.logic.update_status_and_check_completion()

            # Update Resource Stats (Keep)
            self._update_resource_stats()

            # Update Demo Env Timers (Keep)
            if self.initializer.demo_env:
                try:
                    # Timers are updated internally by demo_env.step or manually if needed
                    # self.initializer.demo_env._update_timers() # Call if step isn't called automatically
                    pass  # Assume timers update within demo logic or step
                except Exception as timer_err:
                    logger.error(
                        f"Error updating demo env timers: {timer_err}", exc_info=False
                    )

            # Render UI (Adapted)
            if self.renderer:
                plot_data = {}
                stats_summary = {}
                tb_log_dir = None
                agent_params = 0
                # Removed worker_counts

                if self.initializer.stats_recorder:
                    plot_data = self.initializer.stats_recorder.get_plot_data()
                    current_step = 0
                    if hasattr(self.initializer.stats_recorder, "aggregator"):
                        current_step = getattr(
                            self.initializer.stats_recorder.aggregator.storage,
                            "current_global_step",
                            0,
                        )
                    stats_summary = self.initializer.stats_recorder.get_summary(
                        current_step
                    )
                    if isinstance(
                        self.initializer.stats_recorder, TensorBoardStatsRecorder
                    ):
                        tb_log_dir = self.initializer.stats_recorder.log_dir

                if self.initializer.agent:
                    agent_params = self.initializer.agent_param_count

                # Adapt render_all call
                self.renderer.render_all(
                    app_state=self.app_state.value,
                    is_process_running=False,  # No PPO process running
                    status=self.status,
                    stats_summary=stats_summary,
                    envs=self.initializer.envs,  # Pass empty list or visualized envs
                    num_envs=self.env_config.NUM_ENVS,  # Pass total configured (for layout)
                    env_config=self.env_config,
                    cleanup_confirmation_active=self.cleanup_confirmation_active,
                    cleanup_message=self.cleanup_message,
                    last_cleanup_message_time=self.last_cleanup_message_time,
                    tensorboard_log_dir=tb_log_dir,
                    plot_data=plot_data,
                    demo_env=self.initializer.demo_env,
                    update_progress_details=self.update_progress_details,  # Pass NN progress later
                    agent_param_count=agent_params,
                    worker_counts={},  # Removed worker counts
                )
            else:  # Fallback render
                self.screen.fill((20, 0, 0))
                font = pygame.font.Font(None, 30)
                text_surf = font.render("Renderer Error", True, (255, 50, 50))
                self.screen.blit(
                    text_surf, text_surf.get_rect(center=self.screen.get_rect().center)
                )
                pygame.display.flip()

        logger.info("Main application loop exited.")

    def shutdown(self):
        """Cleans up resources and exits."""
        logger.info("Initiating shutdown sequence...")

        # Signal threads to stop (Keep stop_event)
        logger.info("Setting stop event for worker threads (if any).")
        self.stop_event.set()
        # Removed pause_event clearing

        # Stop and join worker threads (Placeholder)
        logger.info("Stopping worker threads (if any)...")
        self.worker_manager.stop_worker_threads()
        logger.info("Worker threads stopped.")

        # Save final checkpoint (NN weights, stats)
        logger.info("Attempting final checkpoint save...")
        self.logic.save_final_checkpoint()
        logger.info("Final checkpoint save attempt finished.")

        # Close stats recorder (Keep)
        logger.info("Closing stats recorder (before pygame.quit)...")
        self.initializer.close_stats_recorder()
        logger.info("Stats recorder closed.")

        # Quit Pygame (Keep)
        logger.info("Quitting Pygame...")
        pygame.quit()
        logger.info("Pygame quit.")
        logger.info("Shutdown complete.")


# Global variable for TeeLogger instance (Keep)
tee_logger_instance: Optional[TeeLogger] = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlphaTri Trainer"
    )  # Updated description
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        # Updated help text
        help="Path to a specific checkpoint file to load (e.g., NN weights). Overrides auto-resume.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    # Setup Logging (Keep, checkpoint finding logic might adapt)
    if args.load_checkpoint:
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                print(f"Using Run ID from checkpoint path: {get_run_id()}")
            else:
                get_run_id()
        except Exception:
            get_run_id()
    else:
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            print(f"Resuming Run ID: {get_run_id()}")
        else:
            get_run_id()

    # Setup TeeLogger (Keep)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        log_file_dir = get_console_log_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, "console_output.log")
        tee_logger_instance = TeeLogger(log_file_path, sys.stdout)
        sys.stdout = tee_logger_instance
        sys.stderr = tee_logger_instance
    except Exception as e:
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)

    # Set logging level (Keep)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.info(f"Logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {get_run_id()}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")

    # Create and Run Application (Keep structure)
    app = None
    exit_code = 0
    try:
        app = MainApp(checkpoint_to_load=args.load_checkpoint)
        app.initialize()
        app.run_main_loop()
        app.shutdown()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down.")
        if app:
            app.shutdown()
        else:
            pygame.quit()
        exit_code = 130
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in main: {e}", exc_info=True)
        if app:
            app.shutdown()
        else:
            pygame.quit()
        exit_code = 1
    finally:
        # Restore stdout/stderr (Keep)
        print("[Main Finally] Restoring stdout/stderr and closing logger...")
        if tee_logger_instance:
            try:
                if isinstance(sys.stdout, TeeLogger):
                    sys.stdout.flush()
                if isinstance(sys.stderr, TeeLogger):
                    sys.stderr.flush()
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                tee_logger_instance.close()
                print("[Main Finally] TeeLogger closed and streams restored.")
            except Exception as log_close_err:
                original_stdout.write(f"ERROR closing TeeLogger: {log_close_err}\n")
                traceback.print_exc(file=original_stderr)
        print(f"[Main Finally] Exiting with code {exit_code}.")
        sys.exit(exit_code)


File: random_data.md
random data

i m running it on a 

Device name	lg-blade
Processor	12th Gen Intel(R) Core(TM) i9-12900H   2.50 GHz
Installed RAM	64,0 GB (63,7 GB usable)
Device ID	ECD2E981-1D85-4676-88C5-48D7A1929F1A
Product ID	00342-20830-23926-AAOEM
System type	64-bit operating system, x64-based processor
Pen and touch	No pen or touch input is available for this display
WIth a NVIDIA RTX 3080TI Mobile

On a random/initial run the average score is 60
and episodio lengh 17 


File: requirements.txt
pygame>=2.1.0
numpy>=1.20.0
torch>=1.10.0
tensorboard
cloudpickle
matplotlib
psutil

File: wish.md
**AlphaZero Core Concepts Explained**

AlphaZero learns by combining a powerful Neural Network (NN) with Monte Carlo Tree Search (MCTS) through self-play.

1.  **Neural Network (NN):**
    *   **Input:** The current game state (`GameState.get_state()`).
    *   **Output:** Two heads:
        *   **Policy Head (π):** Predicts a probability distribution over all possible *next moves* from the current state. This acts as a prior, guiding the MCTS search towards promising moves.
        *   **Value Head (v):** Predicts the expected *outcome* of the game from the current state (e.g., a value between -1 for loss, 0 for draw, +1 for win).
    *   **Architecture:** Often uses convolutional layers (like ResNet blocks) to process the board state, followed by fully connected layers leading to the policy and value heads. Your `ModelConfig` provides a good starting point.

2.  **Monte Carlo Tree Search (MCTS):**
    *   **Purpose:** For a given game state, MCTS explores the possible future game trajectories to determine the *best* move to make *right now*. It builds a search tree where nodes are game states and edges are actions.
    *   **Process (for each move decision):**
        *   **Selection:** Start at the root node (current state). Traverse down the tree by repeatedly selecting the child node with the highest score according to a formula (like UCB1: `ValueEstimate + ExplorationBonus * sqrt(log(ParentVisits) / ChildVisits)`). The NN's policy output (π) influences the ExplorationBonus, biasing the search towards moves the NN thinks are good. The NN's value output (v) can be used as the initial ValueEstimate for new nodes.
        *   **Expansion:** When a leaf node (a state not yet fully explored or added to the tree) is reached, expand it by adding one or more children representing possible next states after taking valid actions. Get the policy (π) and value (v) for this leaf state from the NN. Use π to initialize child node priors and v as the initial value estimate for this node.
        *   **Simulation (Rollout):** *Crucially, the original AlphaZero often doesn't perform full random rollouts.* Instead, the **NN's value prediction (v)** for the expanded leaf node is directly used as the estimated outcome of the game from that point. This makes the search much more efficient than random simulations.
        *   **Backpropagation:** Update the statistics (visit count `N`, total action value `W` or `Q`) of all nodes along the path from the expanded leaf node back up to the root, using the value (v) obtained during expansion/simulation. The value estimate for a node becomes `Q = W / N`.
    *   **Output:** After running many simulations (e.g., 100, 800, 1600), MCTS provides an *improved* policy distribution for the root state (current state). This distribution is usually based on the visit counts (`N`) of the children nodes of the root (often normalized: `probs = N^(1/temperature)`). This improved policy is the target the NN learns to predict.

**Self-Play:**
    *   The agent plays games against itself.
    *   For each move in a game:
        *   Run MCTS from the current state using the *current* NN.
        *   Select the actual move to play based on the MCTS result (e.g., sample proportionally to visit counts, especially early in the game to encourage exploration; later, deterministically pick the most visited move).
        *   Update the game state.
    *   At the end of the game, record the final outcome (Win=+1, Loss=-1, Draw=0).
    *   Store the collected data for each step: `(state, mcts_policy_target, final_outcome)`.

**Training:**
    *   Periodically (or continuously), sample batches of `(state, mcts_policy_target, final_outcome)` data collected from self-play games.
    *   Train the NN:
        *   **Policy Loss:** Minimize the difference (e.g., cross-entropy) between the NN's policy output (π) for the `state` and the `mcts_policy_target`.
        *   **Value Loss:** Minimize the difference (e.g., mean squared error) between the NN's value output (v) for the `state` and the actual `final_outcome` (z).
        *   Combine these losses (often with regularization) and update the NN weights using an optimizer (e.g., Adam).

**Analogy:** The NN is like an improving intuition about the game. MCTS is like focused thinking/planning using that intuition to find the best immediate move. Self-play generates the experience (games) needed for the intuition (NN) to learn from the thinking (MCTS) and the results (game outcomes).

**Does MCTS need huge amounts of random data?** Not exactly. MCTS *itself* is the search process. The *self-play* phase generates the data *using* MCTS guided by the NN. The quality of this data improves as the NN gets better. You need many self-play games, but the moves within those games are intelligently selected by MCTS, not purely random (except maybe in the simulation phase if you choose to implement it that way, but using the NN value is standard).

**Step-by-Step Implementation Plan**

Here's a high-level, detailed plan to refactor towards AlphaZero:

**Phase 1: Implement AlphaZero Components**

3.  **Define Neural Network (`AlphaZeroNet`):**
    *   Create a new file (e.g., `agent/alphazero_net.py`).
    *   Define a class `AlphaZeroNet(torch.nn.Module)`.
    *   Use `ModelConfig` to define the architecture (e.g., CNN backbone similar to existing, potentially ResNet blocks).
    *   Implement the `forward` method:
        *   Input: State dictionary from `GameState.get_state()`. Process grid, shapes, features appropriately.
        *   Output: `policy_logits` (raw scores before softmax, shape `[batch_size, action_dim]`) and `value` (scalar estimate, shape `[batch_size, 1]`).
    *   Implement `get_state_dict` and `load_state_dict` (standard PyTorch).
4.  **Implement MCTS:**
    *   Create new files (e.g., `mcts/node.py`, `mcts/search.py`).
    *   **`MCTSNode` Class:** Represents a node in the search tree. Stores:
        *   `state`: The game state this node represents (can be lightweight if needed).
        *   `parent`: Reference to the parent node.
        *   `children`: Dictionary mapping action -> child `MCTSNode`.
        *   `visit_count (N)`: How many times this node was visited during backpropagation.
        *   `total_action_value (W)` or `mean_action_value (Q)`: Sum or average of values backpropagated through this node.
        *   `prior_probability (P)`: Policy prior from the NN for the action leading to this node (stored in the child).
        *   `action_taken`: The action that led from the parent to this node.
        *   `is_expanded`: Boolean flag.
    *   **`MCTS` Class:** Orchestrates the search.
        *   `__init__(self, nn_agent: AlphaZeroNet, env_config: EnvConfig, mcts_config)`: Takes the NN and configs.
        *   `run_simulations(self, root_state: GameState, num_simulations: int)`: Main MCTS loop.
            *   Creates the `root_node`.
            *   Repeatedly calls `_select`, `_expand`, `_simulate` (using NN value), `_backpropagate`.
        *   `_select(self, node: MCTSNode)`: Traverses the tree using UCB1 or PUCT formula, returning the leaf node to expand.
        *   `_expand(self, node: MCTSNode)`: If node isn't terminal, get valid actions, get policy/value from NN for the node's state, create child nodes, initialize their priors (P).
        *   `_simulate(self, node: MCTSNode)`: **Crucially, just return the value (v) predicted by the NN during the `_expand` step for this node.** No random rollout needed typically.
        *   `_backpropagate(self, node: MCTSNode, value: float)`: Update `N` and `W` (or `Q`) for the node and its ancestors up to the root.
        *   `get_policy_target(self, root_node: MCTSNode, temperature: float)`: After simulations, calculate the improved policy target based on child visit counts (`N^(1/temperature)`), normalized. Returns a probability distribution over actions.

**Phase 2: Implement Workers and Integration**

5.  **Implement Self-Play Worker:**
    *   Create a class (e.g., `workers/self_play_worker.py`).
    *   Takes the NN, `EnvConfig`, MCTS instance (or creates one), a shared data buffer (e.g., `queue.Queue` or custom buffer), stop/pause events.
    *   `run()` method:
        *   Loops indefinitely (until `stop_event`).
        *   Plays a full game:
            *   `game = GameState()`, `game.reset()`.
            *   `game_data = []` (to store `(state, policy_target, player)` tuples for this game).
            *   While `not game.is_over()`:
                *   `policy_target = mcts.run_simulations(game.get_state(), num_simulations)`
                *   `current_state_features = game.get_state()` # Get state *before* the move
                *   `game_data.append((current_state_features, policy_target, game.current_player))` # Store state and MCTS target
                *   `action = choose_action(policy_target, temperature)` # Choose actual move (probabilistic early, deterministic later)
                *   `_, done = game.step(action)`
            *   `final_outcome = determine_outcome(game)` # Get win/loss/draw (+1/-1/0)
            *   Assign the `final_outcome` to all stored tuples in `game_data`.
            *   Put `game_data` into the shared buffer.
            *   Log episode stats via `StatsAggregator`.
6.  **Implement Training Worker:**
    *   Create a class (e.g., `workers/training_worker.py`).
    *   Takes the NN, optimizer, shared data buffer, `StatsAggregator`, stop event.
    *   `run()` method:
        *   Loops indefinitely (until `stop_event`).
        *   Samples a batch of `(state, policy_target, outcome)` from the buffer.
        *   Performs NN forward pass: `policy_logits, value = nn(batch_states)`.
        *   Calculates policy loss (e.g., `CrossEntropyLoss(policy_logits, batch_policy_targets)`).
        *   Calculates value loss (e.g., `MSELoss(value, batch_outcomes)`).
        *   Calculates total loss (+ regularization).
        *   Performs `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
        *   Logs losses and other training metrics (LR, etc.) via `StatsAggregator.record_step()`.
7.  **Integrate Components:**
    *   **`AppInitializer`:** Instantiate `AlphaZeroNet`, `MCTS`, `SelfPlayWorker`, `TrainingWorker`, shared buffer, optimizer. Pass references correctly.
    *   **`AppWorkerManager`:** Modify `start_worker_threads` and `stop_worker_threads` to manage the new `SelfPlayWorker` and `TrainingWorker` threads.
    *   **`CheckpointManager`:** Update `save_checkpoint` to include `nn.state_dict()`, `optimizer.state_dict()`, and the updated `stats_aggregator.state_dict()`. Update `load_checkpoint` accordingly.
    *   **`StatsAggregator` / Loggers:** Ensure they track and log `policy_loss`, `value_loss`, and potentially MCTS statistics passed via `record_step` or `record_episode`.
    *   **`main_pygame.py`:** Ensure the main loop correctly starts/stops workers, fetches stats from the aggregator for rendering, and handles shutdown gracefully.
    *   **UI (`LeftPanelRenderer`, `Plotter`):** Update to display new stats (NN losses) and remove obsolete PPO stats.

**Phase 3: Refinement and Tuning**

8.  **Debugging:** Thoroughly test interactions between MCTS, NN, self-play, and training.
9.  **Tuning:** Adjust hyperparameters:
    *   MCTS: `num_simulations`, UCB1/PUCT exploration constant (`c_puct`).
    *   Self-Play: Temperature parameter for action selection.
    *   NN: Architecture (`ModelConfig`), learning rate, optimizer parameters, regularization strength.
    *   Training: Batch size, buffer size, training frequency vs. self-play generation speed.
10. **Profiling:** Use `analyze_profile.py` to identify bottlenecks (MCTS or NN inference are common).

File: config\constants.py

"""
Defines constants shared across different modules, primarily visual elements,
to avoid circular imports and keep configuration clean.
"""

# Colors (RGB tuples 0-255)
WHITE: tuple[int, int, int] = (255, 255, 255)
BLACK: tuple[int, int, int] = (0, 0, 0)
LIGHTG: tuple[int, int, int] = (140, 140, 140)
GRAY: tuple[int, int, int] = (50, 50, 50)
RED: tuple[int, int, int] = (255, 50, 50)
DARK_RED: tuple[int, int, int] = (80, 10, 10)
BLUE: tuple[int, int, int] = (50, 50, 255)
YELLOW: tuple[int, int, int] = (255, 255, 100)
GOOGLE_COLORS: list[tuple[int, int, int]] = [
    (15, 157, 88),  # Green
    (244, 180, 0),  # Yellow/Orange
    (66, 133, 244),  # Blue
    (219, 68, 55),  # Red
]
LINE_CLEAR_FLASH_COLOR: tuple[int, int, int] = (180, 180, 220)
LINE_CLEAR_HIGHLIGHT_COLOR: tuple[int, int, int, int] = (255, 255, 0, 180)  # RGBA
GAME_OVER_FLASH_COLOR: tuple[int, int, int] = (255, 0, 0)

# Add other simple, shared constants here if needed.


File: config\core.py
# File: config/core.py
import torch
from typing import List, Tuple, Optional

from .constants import (
    WHITE,
    BLACK,
    LIGHTG,
    GRAY,
    RED,
    DARK_RED,
    BLUE,
    YELLOW,
    GOOGLE_COLORS,
    LINE_CLEAR_FLASH_COLOR,
    LINE_CLEAR_HIGHLIGHT_COLOR,
    GAME_OVER_FLASH_COLOR,
)


class VisConfig:
    NUM_ENVS_TO_RENDER = 16
    FPS = 60
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00
    LEFT_PANEL_RATIO = 0.7
    ENV_SPACING = 2
    ENV_GRID_PADDING = 2

    WHITE = WHITE
    BLACK = BLACK
    LIGHTG = LIGHTG
    GRAY = GRAY
    RED = RED
    DARK_RED = DARK_RED
    BLUE = BLUE
    YELLOW = YELLOW
    GOOGLE_COLORS = GOOGLE_COLORS
    LINE_CLEAR_FLASH_COLOR = LINE_CLEAR_FLASH_COLOR
    LINE_CLEAR_HIGHLIGHT_COLOR = LINE_CLEAR_HIGHLIGHT_COLOR
    GAME_OVER_FLASH_COLOR = GAME_OVER_FLASH_COLOR


class EnvConfig:
    NUM_ENVS = 1  # Set to 1, as we removed multi-env collection logic
    ROWS = 8
    COLS = 15
    GRID_FEATURES_PER_CELL = 2
    SHAPE_FEATURES_PER_SHAPE = 5
    NUM_SHAPE_SLOTS = 3
    EXPLICIT_FEATURES_DIM = 10
    CALCULATE_POTENTIAL_OUTCOMES_IN_STATE = False  # Keep False for now

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
        # Action space might change for MCTS (e.g., just placement coords)
        # Keep original for now, but likely needs refactoring later.
        return self.NUM_SHAPE_SLOTS * (self.ROWS * self.COLS)


# Removed RewardConfig


# Removed PPOConfig


class RNNConfig:  # Keep for potential future NN architectures
    USE_RNN = False  # Default to False
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2


class TransformerConfig:  # Keep for potential future NN architectures
    USE_TRANSFORMER = False  # Default to False
    TRANSFORMER_D_MODEL = 256
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_DIM_FEEDFORWARD = 512
    TRANSFORMER_NUM_LAYERS = 3
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "relu"


class TrainConfig:  # Simplified for general training/checkpointing
    # Checkpointing might be handled differently (e.g., saving NN weights + MCTS stats)
    CHECKPOINT_SAVE_FREQ = (
        50  # Frequency might mean something else now (e.g., training steps, games)
    )
    LOAD_CHECKPOINT_PATH: Optional[str] = None


class ModelConfig:  # Keep structure for the AlphaZero NN
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        CONV_CHANNELS = [64, 128, 256]
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True

        SHAPE_FEATURE_MLP_DIMS = [128, 128]
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU

        COMBINED_FC_DIMS = [1024, 256]
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.0


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [25, 50, 100]
    CONSOLE_LOG_FREQ = 1
    PLOT_DATA_WINDOW = 100_000


class TensorBoardConfig:
    LOG_HISTOGRAMS = False
    HISTOGRAM_LOG_FREQ = 20  # Frequency might mean training steps/games
    LOG_IMAGES = False  # Keep ability to log images (e.g., board states)
    IMAGE_LOG_FREQ = 20  # Frequency might mean training steps/games
    LOG_DIR: Optional[str] = None


class DemoConfig:  # Keep as is
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Arrows]=Move | [Q/E]=Cycle Shape | [Space]=Place | [ESC]=Exit"
    DEBUG_HELP_TEXT = "[Click]=Toggle Triangle | [R]=Reset Grid | [ESC]=Exit"

File: config\general.py
import torch
import os
import time
from typing import Optional

# --- Base Directories ---
BASE_CHECKPOINT_DIR = "checkpoints"
BASE_LOG_DIR = "logs"

# --- Device ---
DEVICE: Optional[torch.device] = None


def set_device(device: torch.device):
    """Sets the global DEVICE variable."""
    global DEVICE
    DEVICE = device
    print(f"[Config] Global DEVICE set to: {DEVICE}")


# --- Random Seed ---
RANDOM_SEED = 42

# --- Run ID and Paths (Dynamically Determined) ---
_current_run_id: Optional[str] = None


def get_run_id() -> str:
    """Gets the current run ID, generating one if not set."""
    global _current_run_id
    if _current_run_id is None:
        _current_run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"[Config] Generated new RUN_ID: {_current_run_id}")
    return _current_run_id


def set_run_id(run_id: str):
    """Sets the run ID, typically when resuming a run."""
    global _current_run_id
    if _current_run_id is not None and _current_run_id != run_id:
        print(
            f"[Config] WARNING: Overwriting existing RUN_ID '{_current_run_id}' with '{run_id}'."
        )
    elif _current_run_id is None:
        print(f"[Config] Setting RUN_ID to resumed ID: {run_id}")
    _current_run_id = run_id


def get_run_checkpoint_dir() -> str:
    """Gets the checkpoint directory for the current run."""
    # Checkpoints will now likely store NN weights, maybe MCTS stats
    return os.path.join(BASE_CHECKPOINT_DIR, get_run_id())


def get_run_log_dir() -> str:
    """Gets the TensorBoard log directory for the current run."""
    # Ensure the base 'tensorboard' subdirectory exists within BASE_LOG_DIR
    tb_base = os.path.join(BASE_LOG_DIR, "tensorboard")
    return os.path.join(tb_base, get_run_id())


def get_console_log_dir() -> str:
    """Gets the directory for console logs for the current run."""
    # Place console logs directly within the run-specific log directory
    return get_run_log_dir()


def get_model_save_path() -> str:
    """Gets the base model save path for the current run (adapt name later)."""
    # Rename from ppo_agent_state to something more generic like 'alphatri_nn.pth'
    return os.path.join(get_run_checkpoint_dir(), "alphatri_nn.pth")


File: config\utils.py
import torch
from typing import Dict, Any
from .core import (
    VisConfig,
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    DemoConfig,
    TransformerConfig,
)

from .general import DEVICE, RANDOM_SEED, get_run_id


def get_config_dict() -> Dict[str, Any]:
    """Returns a flat dictionary of all relevant config values for logging."""
    all_configs = {}

    def flatten_class(cls, prefix=""):
        d = {}
        # Use cls() to instantiate if needed for properties, but be careful
        instance = None
        try:
            instance = cls()
        except Exception:
            instance = None  # Cannot instantiate, rely on class vars

        for k, v in vars(cls).items():
            if (
                not k.startswith("__")
                and not callable(v)
                and not isinstance(v, type)
                and not hasattr(v, "__module__")  # Exclude modules
            ):
                # Check if it's a property descriptor on the class
                is_property = isinstance(getattr(cls, k, None), property)

                if is_property and instance:
                    try:
                        v = getattr(instance, k)  # Get property value from instance
                    except Exception:
                        continue  # Skip if property access fails
                elif is_property and not instance:
                    continue  # Skip properties if instance couldn't be created

                d[f"{prefix}{k}"] = v
        return d

    # Flatten core config classes
    all_configs.update(flatten_class(VisConfig, "Vis."))
    all_configs.update(flatten_class(EnvConfig, "Env."))
    all_configs.update(flatten_class(RNNConfig, "RNN."))
    all_configs.update(flatten_class(TrainConfig, "Train."))
    all_configs.update(flatten_class(ModelConfig.Network, "Model.Net."))
    all_configs.update(flatten_class(StatsConfig, "Stats."))
    all_configs.update(flatten_class(TensorBoardConfig, "TB."))
    all_configs.update(flatten_class(DemoConfig, "Demo."))
    all_configs.update(flatten_class(TransformerConfig, "Transformer."))

    # Add general config values
    all_configs["General.DEVICE"] = str(DEVICE) if DEVICE else "None"
    all_configs["General.RANDOM_SEED"] = RANDOM_SEED
    all_configs["General.RUN_ID"] = get_run_id()  # Use the getter function

    # Filter out None paths
    all_configs = {
        k: v for k, v in all_configs.items() if not (k.endswith("_PATH") and v is None)
    }

    # Convert non-basic types to strings for logging
    for key, value in all_configs.items():
        if isinstance(value, type) and issubclass(value, torch.nn.Module):
            all_configs[key] = value.__name__
        elif isinstance(value, (list, tuple)):
            all_configs[key] = str(value)
        elif not isinstance(value, (int, float, str, bool)):
            # Check for None explicitly before converting to string
            if value is None:
                all_configs[key] = "None"
            else:
                all_configs[key] = str(value)

    return all_configs


File: config\validation.py
# File: config/validation.py
from .core import (
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    VisConfig,
    TransformerConfig,
)
from .general import (
    DEVICE,
    # Removed TOTAL_TRAINING_STEPS
    get_run_id,
    get_run_log_dir,
    get_run_checkpoint_dir,
    get_model_save_path,
)


def print_config_info_and_validate():
    env_config_instance = EnvConfig()
    rnn_config_instance = RNNConfig()
    transformer_config_instance = TransformerConfig()
    vis_config_instance = VisConfig()
    train_config_instance = TrainConfig()

    run_id = get_run_id()
    run_log_dir = get_run_log_dir()
    run_checkpoint_dir = get_run_checkpoint_dir()

    print("-" * 70)
    print(f"RUN ID: {run_id}")
    print(f"Log Directory: {run_log_dir}")
    print(f"Checkpoint Directory: {run_checkpoint_dir}")
    print(f"Device: {DEVICE}")
    print(
        f"TB Logging: Histograms={'ON' if TensorBoardConfig.LOG_HISTOGRAMS else 'OFF'}, "
        f"Images={'ON' if TensorBoardConfig.LOG_IMAGES else 'OFF'}"
    )

    if train_config_instance.LOAD_CHECKPOINT_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD CHECKPOINT specified: {train_config_instance.LOAD_CHECKPOINT_PATH} ***\n"
            "*** CheckpointManager will attempt to load this path (e.g., NN weights). ***\n"
            + "*" * 70
        )
    else:
        print(
            "--- No explicit checkpoint path. CheckpointManager will attempt auto-resume if applicable. ---"
        )

    print("--- Training Algorithm: AlphaZero (MCTS + NN) ---")  # Updated description

    # Removed PPO specific prints

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
        f"--- Using Transformer: {transformer_config_instance.USE_TRANSFORMER}"
        + (
            f" (d_model={transformer_config_instance.TRANSFORMER_D_MODEL}, nhead={transformer_config_instance.TRANSFORMER_NHEAD}, layers={transformer_config_instance.TRANSFORMER_NUM_LAYERS})"
            if transformer_config_instance.USE_TRANSFORMER
            else ""
        )
        + " ---"
    )
    # Removed ObsNorm print

    print(
        f"Config: Env=(R={env_config_instance.ROWS}, C={env_config_instance.COLS}), "
        f"GridState={env_config_instance.GRID_STATE_SHAPE}, "
        f"ShapeState={env_config_instance.SHAPE_STATE_DIM}, "
        f"ActionDim={env_config_instance.ACTION_DIM}"
    )
    cnn_str = str(ModelConfig.Network.CONV_CHANNELS).replace(" ", "")
    mlp_str = str(ModelConfig.Network.COMBINED_FC_DIMS).replace(" ", "")
    shape_mlp_cfg_str = str(ModelConfig.Network.SHAPE_FEATURE_MLP_DIMS).replace(" ", "")
    print(
        f"Network Base: CNN={cnn_str}, ShapeMLP={shape_mlp_cfg_str}, Fusion={mlp_str}"
    )  # Adapted description

    print(f"Training: NUM_ENVS={env_config_instance.NUM_ENVS}")  # Removed total steps
    print(
        f"Stats: AVG_WINDOWS={StatsConfig.STATS_AVG_WINDOW}, Console Log Freq={StatsConfig.CONSOLE_LOG_FREQ} (episodes/updates)"  # Adapted freq description
    )

    if env_config_instance.NUM_ENVS >= 1024:  # Keep warning, though NUM_ENVS is now 1
        device_type = DEVICE.type if DEVICE else "UNKNOWN"
        print(
            "*" * 70
            + f"\n*** Warning: NUM_ENVS={env_config_instance.NUM_ENVS}. Monitor system resources. ***"
            + (
                "\n*** Using MPS device. Performance varies. Force CPU via env var if needed. ***"
                if device_type == "mps"
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
# config/__init__.py
# This file marks the 'config' directory as a Python package.

# Import core configuration classes to make them available directly under 'config'
from .core import (
    VisConfig,
    EnvConfig,
    # Removed RewardConfig
    # Removed PPOConfig
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    DemoConfig,
    # Removed ObsNormConfig
    TransformerConfig,
)

# Import general configuration settings and functions
from .general import (
    DEVICE,
    RANDOM_SEED,
    # Removed TOTAL_TRAINING_STEPS
    BASE_CHECKPOINT_DIR,
    BASE_LOG_DIR,
    set_device,
    get_run_id,
    set_run_id,
    get_run_checkpoint_dir,
    get_run_log_dir,
    get_console_log_dir,
    get_model_save_path,
)

# Import utility functions
from .utils import get_config_dict

# Import validation function
from .validation import print_config_info_and_validate

# Import constants
from .constants import (
    WHITE,
    BLACK,
    LIGHTG,
    GRAY,
    RED,
    DARK_RED,
    BLUE,
    YELLOW,
    GOOGLE_COLORS,
    LINE_CLEAR_FLASH_COLOR,
    LINE_CLEAR_HIGHLIGHT_COLOR,
    GAME_OVER_FLASH_COLOR,
)


# Define __all__ to control what 'from config import *' imports
__all__ = [
    # Core Configs
    "VisConfig",
    "EnvConfig",
    # Removed RewardConfig
    # Removed PPOConfig
    "RNNConfig",
    "TrainConfig",
    "ModelConfig",
    "StatsConfig",
    "TensorBoardConfig",
    "DemoConfig",
    # Removed ObsNormConfig
    "TransformerConfig",
    # General Configs
    "DEVICE",
    "RANDOM_SEED",
    # Removed TOTAL_TRAINING_STEPS
    "BASE_CHECKPOINT_DIR",
    "BASE_LOG_DIR",
    "set_device",
    "get_run_id",
    "set_run_id",
    "get_run_checkpoint_dir",
    "get_run_log_dir",
    "get_console_log_dir",
    "get_model_save_path",
    # Utils
    "get_config_dict",
    "print_config_info_and_validate",
    # Constants
    "WHITE",
    "BLACK",
    "LIGHTG",
    "GRAY",
    "RED",
    "DARK_RED",
    "BLUE",
    "YELLOW",
    "GOOGLE_COLORS",
    "LINE_CLEAR_FLASH_COLOR",
    "LINE_CLEAR_HIGHLIGHT_COLOR",
    "GAME_OVER_FLASH_COLOR",
]


File: environment\game_demo_logic.py
# File: environment/game_demo_logic.py
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .game_state import GameState
    from .shape import Shape


class GameDemoLogic:
    """Handles logic specific to the interactive demo and debug modes."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def update_demo_selection_after_placement(self, placed_slot_index: int):
        """Selects the next available shape slot after placement in demo mode."""
        num_slots = self.gs.env_config.NUM_SHAPE_SLOTS
        if num_slots <= 0:
            return
        available_indices = [i for i, s in enumerate(self.gs.shapes) if s is not None]
        if not available_indices:
            self.gs.demo_selected_shape_idx = 0
        else:
            self.gs.demo_selected_shape_idx = available_indices[0]

    def select_shape_for_drag(self, shape_index: int):
        """Selects a shape to be dragged by the mouse."""
        if self.gs.game_over or self.gs.freeze_time > 0:
            return
        if (
            0 <= shape_index < len(self.gs.shapes)
            and self.gs.shapes[shape_index] is not None
        ):
            self.gs.demo_dragged_shape_idx = shape_index
            self.gs.demo_selected_shape_idx = shape_index
            self.gs.demo_snapped_position = None
            print(f"[Demo] Dragging shape index: {shape_index}")
        else:
            self.gs.demo_dragged_shape_idx = None
            print(f"[Demo] Invalid shape index {shape_index} or shape is None.")

    def deselect_dragged_shape(self):
        """Deselects the currently dragged shape."""
        if self.gs.demo_dragged_shape_idx is not None:
            print(f"[Demo] Deselected shape index: {self.gs.demo_dragged_shape_idx}")
            self.gs.demo_dragged_shape_idx = None
            self.gs.demo_snapped_position = None

    def update_snapped_position(self, grid_pos: Optional[Tuple[int, int]]):
        """Updates the snapped position if the dragged shape can be placed there."""
        if self.gs.demo_dragged_shape_idx is None:
            self.gs.demo_snapped_position = None
            return

        shape_to_check = self.gs.shapes[self.gs.demo_dragged_shape_idx]
        if shape_to_check is None:
            self.gs.demo_snapped_position = None
            return

        if grid_pos is not None:
            target_row, target_col = grid_pos
            if self.gs.grid.can_place(shape_to_check, target_row, target_col):
                if self.gs.demo_snapped_position != grid_pos:
                    self.gs.demo_snapped_position = grid_pos
            else:
                self.gs.demo_snapped_position = None
        else:
            self.gs.demo_snapped_position = None

    def place_dragged_shape(self) -> bool:
        """Attempts to place the currently dragged and snapped shape."""
        if self.gs.game_over or self.gs.freeze_time > 0:
            return False
        if (
            self.gs.demo_dragged_shape_idx is None
            or self.gs.demo_snapped_position is None
        ):
            print("[Demo] Cannot place: No shape dragged or not snapped.")
            return False

        shape_slot_index = self.gs.demo_dragged_shape_idx
        target_row, target_col = self.gs.demo_snapped_position
        shape_to_place = self.gs.shapes[shape_slot_index]

        if shape_to_place is not None and self.gs.grid.can_place(
            shape_to_place, target_row, target_col
        ):
            print(
                f"[Demo] Placing shape {shape_slot_index} at {target_row},{target_col}"
            )
            locations_per_shape = self.gs.grid.rows * self.gs.grid.cols
            action_index = shape_slot_index * locations_per_shape + (
                target_row * self.gs.grid.cols + target_col
            )
            _, _ = self.gs.logic.step(action_index)  # Use logic helper

            self.gs.demo_dragged_shape_idx = None
            self.gs.demo_snapped_position = None
            return True
        else:
            print(f"[Demo] Invalid placement attempt at {target_row},{target_col}")
            return False

    def get_dragged_shape_info(
        self,
    ) -> Tuple[Optional["Shape"], Optional[Tuple[int, int]]]:
        """Returns the currently dragged shape object and its snapped position."""
        if self.gs.demo_dragged_shape_idx is None:
            return None, None
        shape = self.gs.shapes[self.gs.demo_dragged_shape_idx]
        return shape, self.gs.demo_snapped_position

    def toggle_triangle_debug(self, row: int, col: int):
        """Toggles the state of a triangle for debugging and checks for lines."""
        if not self.gs.grid.valid(row, col):
            print(f"[Debug] Invalid coords: ({row}, {col})")
            return

        triangle = self.gs.grid.triangles[row][col]
        if triangle.is_death:
            print(f"[Debug] Cannot toggle death cell: ({row}, {col})")
            return

        triangle.is_occupied = not triangle.is_occupied
        if triangle.is_occupied:
            triangle.color = self.gs.vis_config.YELLOW
        else:
            triangle.color = None
        print(
            f"[Debug] Toggled ({row}, {col}) to {'Occupied' if triangle.is_occupied else 'Empty'}"
        )

        lines_cleared, triangles_cleared, cleared_coords = self.gs.grid.clear_lines()
        line_clear_reward = self.gs.logic._calculate_line_clear_reward(
            triangles_cleared
        )  # Use logic helper

        if triangles_cleared > 0:
            print(
                f"[Debug] Cleared {triangles_cleared} triangles in {lines_cleared} lines."
            )
            self.gs.game_score += triangles_cleared * 2
            self.gs.triangles_cleared_this_episode += triangles_cleared
            self.gs.blink_time = 0.5
            self.gs.freeze_time = 0.5
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.5
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                triangles_cleared,
                line_clear_reward,
            )
        else:
            if self.gs.line_clear_highlight_time <= 0:
                self.gs.cleared_triangles_coords = []
            self.gs.last_line_clear_info = None

        self.gs.game_over = False
        self.gs.game_over_flash_time = 0.0


File: environment\game_logic.py
# File: environment/game_logic.py
from typing import TYPE_CHECKING, List, Tuple, Optional
import time

if TYPE_CHECKING:
    from .game_state import GameState
    from .shape import Shape


class GameLogic:
    """Handles the core game mechanics like stepping, placement, and line clearing."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        # Note: GameState.valid_actions() now checks is_frozen() first.
        # This method assumes the game is not frozen when called.
        if self.gs.game_over:  # Still check game_over here
            return []

        valid_action_indices: List[int] = []
        locations_per_shape = self.gs.grid.rows * self.gs.grid.cols
        for shape_slot_index, current_shape in enumerate(self.gs.shapes):
            if not current_shape:
                continue
            for target_row in range(self.gs.grid.rows):
                for target_col in range(self.gs.grid.cols):
                    if self.gs.grid.can_place(current_shape, target_row, target_col):
                        action_index = shape_slot_index * locations_per_shape + (
                            target_row * self.gs.grid.cols + target_col
                        )
                        valid_action_indices.append(action_index)
        return valid_action_indices

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        locations_per_shape = self.gs.grid.rows * self.gs.grid.cols
        shape_slot_index = action_index // locations_per_shape
        position_index = action_index % locations_per_shape
        target_row = position_index // self.gs.grid.cols
        target_col = position_index % self.gs.grid.cols
        return (shape_slot_index, target_row, target_col)

    def _check_fundamental_game_over(self) -> bool:
        """Checks if any available shape can be placed anywhere."""
        for current_shape in self.gs.shapes:
            if not current_shape:
                continue
            for target_row in range(self.gs.grid.rows):
                for target_col in range(self.gs.grid.cols):
                    if self.gs.grid.can_place(current_shape, target_row, target_col):
                        return False
        return True

    def _calculate_placement_reward(self, placed_shape: "Shape") -> float:
        return self.gs.rewards.REWARD_PLACE_PER_TRI * len(placed_shape.triangles)

    def _calculate_line_clear_reward(self, triangles_cleared: int) -> float:
        return triangles_cleared * self.gs.rewards.REWARD_PER_CLEARED_TRIANGLE

    def _calculate_state_penalty(self) -> float:
        penalty = 0.0
        max_height = self.gs.grid.get_max_height()
        bumpiness = self.gs.grid.get_bumpiness()
        num_holes = self.gs.grid.count_holes()
        penalty += max_height * self.gs.rewards.PENALTY_MAX_HEIGHT_FACTOR
        penalty += bumpiness * self.gs.rewards.PENALTY_BUMPINESS_FACTOR
        penalty += num_holes * self.gs.rewards.PENALTY_HOLE_PER_HOLE
        return penalty

    def _handle_invalid_placement(self) -> float:
        self.gs._last_action_valid = False
        return self.gs.rewards.PENALTY_INVALID_MOVE

    def _handle_game_over_state_change(self) -> float:
        if self.gs.game_over:
            return 0.0
        self.gs.game_over = True
        if self.gs.freeze_time <= 0:  # Only set freeze if not already frozen
            self.gs.freeze_time = 1.0
        self.gs.game_over_flash_time = 0.6
        return self.gs.rewards.PENALTY_GAME_OVER

    def _handle_valid_placement(
        self,
        shape_to_place: "Shape",
        shape_slot_index: int,
        target_row: int,
        target_col: int,
    ) -> float:
        self.gs._last_action_valid = True
        step_reward = 0.0

        step_reward += self._calculate_placement_reward(shape_to_place)
        holes_before = self.gs.grid.count_holes()

        self.gs.grid.place(shape_to_place, target_row, target_col)
        self.gs.shapes[shape_slot_index] = None
        self.gs.game_score += len(shape_to_place.triangles)
        self.gs.pieces_placed_this_episode += 1

        lines_cleared, triangles_cleared, cleared_coords = self.gs.grid.clear_lines()
        line_clear_reward = self._calculate_line_clear_reward(triangles_cleared)
        step_reward += line_clear_reward
        self.gs.triangles_cleared_this_episode += triangles_cleared

        if triangles_cleared > 0:
            self.gs.game_score += triangles_cleared * 2
            self.gs.blink_time = 0.5
            self.gs.freeze_time = 0.5  # Set freeze time for animation
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.5
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                triangles_cleared,
                line_clear_reward,
            )

        holes_after = self.gs.grid.count_holes()
        new_holes_created = max(0, holes_after - holes_before)

        step_reward += self._calculate_state_penalty()
        step_reward += new_holes_created * self.gs.rewards.PENALTY_NEW_HOLE

        if all(s is None for s in self.gs.shapes):
            from .shape import Shape  # Local import to avoid cycle

            self.gs.shapes = [
                Shape() for _ in range(self.gs.env_config.NUM_SHAPE_SLOTS)
            ]

        if self._check_fundamental_game_over():
            step_reward += self._handle_game_over_state_change()

        self.gs.demo_logic.update_demo_selection_after_placement(shape_slot_index)
        return step_reward

    def step(self, action_index: int) -> Tuple[float, bool]:
        """Performs one game step based on the action index."""
        # Update timers at the very beginning of the step
        self.gs._update_timers()

        # Check game over state *after* timer update
        if self.gs.game_over:
            return (0.0, True)

        # Check if frozen *after* timer update
        if self.gs.is_frozen():
            # If frozen, still calculate potential change but return 0 extrinsic reward
            # print(f"[GameLogic] Step called while frozen ({self.gs.freeze_time:.3f}s left). Skipping action.") # DEBUG
            current_potential = self.gs.features.calculate_potential()
            pbrs_reward = (
                (self.gs.ppo_config.GAMMA * current_potential - self.gs._last_potential)
                if self.gs.rewards.ENABLE_PBRS
                else 0.0
            )
            self.gs._last_potential = current_potential
            total_reward = (
                self.gs.rewards.REWARD_ALIVE_STEP + pbrs_reward
            )  # Still give alive reward? Maybe not if frozen? Let's keep it for now.
            self.gs.score += total_reward
            return (
                total_reward,
                False,
            )  # Return False for done, as game is just paused

        # --- If not frozen and not game over, proceed with action ---
        shape_slot_index, target_row, target_col = self.decode_action(action_index)

        shape_to_place = (
            self.gs.shapes[shape_slot_index]
            if 0 <= shape_slot_index < len(self.gs.shapes)
            else None
        )
        # Check if the specific action is valid (placement possible)
        is_valid_placement = shape_to_place is not None and self.gs.grid.can_place(
            shape_to_place, target_row, target_col
        )

        potential_before_action = self.gs.features.calculate_potential()

        if is_valid_placement:
            extrinsic_reward = self._handle_valid_placement(
                shape_to_place, shape_slot_index, target_row, target_col
            )
        else:
            # An invalid action was chosen (e.g., by the agent or debug click)
            # print(f"[GameLogic] Invalid placement attempt: Action {action_index} -> Slot {shape_slot_index}, Pos ({target_row},{target_col})") # DEBUG
            extrinsic_reward = self._handle_invalid_placement()
            # Check if *any* move is possible after this invalid attempt
            if self._check_fundamental_game_over():
                extrinsic_reward += self._handle_game_over_state_change()

        # Add alive reward only if the game didn't end *during* this step
        if not self.gs.game_over:
            extrinsic_reward += self.gs.rewards.REWARD_ALIVE_STEP

        potential_after_action = self.gs.features.calculate_potential()
        pbrs_reward = 0.0
        if self.gs.rewards.ENABLE_PBRS:
            pbrs_reward = (
                self.gs.ppo_config.GAMMA * potential_after_action
                - potential_before_action
            )

        total_reward = extrinsic_reward + pbrs_reward
        self.gs._last_potential = potential_after_action

        self.gs.score += total_reward
        return (total_reward, self.gs.game_over)


File: environment\game_state.py
# File: environment/game_state.py
import time
import numpy as np
from typing import List, Optional, Tuple, Dict

from config import EnvConfig, RewardConfig, PPOConfig, VisConfig
from .grid import Grid
from .shape import Shape
from .game_logic import GameLogic
from .game_state_features import GameStateFeatures
from .game_demo_logic import GameDemoLogic

StateType = Dict[str, np.ndarray]


class GameState:
    """
    Represents the state of a single game instance.
    Delegates logic to helper classes: GameLogic, GameStateFeatures, GameDemoLogic.
    Timer updates are now primarily handled within GameLogic.step().
    """

    def __init__(self):
        self.env_config = EnvConfig()
        self.rewards = RewardConfig()
        self.ppo_config = PPOConfig()
        self.vis_config = VisConfig()

        self.grid = Grid(self.env_config)
        self.shapes: List[Optional[Shape]] = []
        self.score: float = 0.0
        self.game_score: int = 0
        self.triangles_cleared_this_episode: int = 0
        self.pieces_placed_this_episode: int = 0

        # Timers
        self.blink_time: float = 0.0
        self._last_timer_update_time: float = (
            time.monotonic()
        )  # Use monotonic clock for intervals
        self.freeze_time: float = 0.0
        self.line_clear_flash_time: float = 0.0
        self.line_clear_highlight_time: float = 0.0
        self.game_over_flash_time: float = 0.0
        self.cleared_triangles_coords: List[Tuple[int, int]] = []
        self.last_line_clear_info: Optional[Tuple[int, int, float]] = None

        self.game_over: bool = False
        self._last_action_valid: bool = True
        self._last_potential: float = 0.0

        # Demo state
        self.demo_selected_shape_idx: int = 0
        self.demo_dragged_shape_idx: Optional[int] = None
        self.demo_snapped_position: Optional[Tuple[int, int]] = None

        # Helper classes
        self.logic = GameLogic(self)
        self.features = GameStateFeatures(self)
        self.demo_logic = GameDemoLogic(self)

        self.reset()

    def reset(self) -> StateType:
        """Resets the game to its initial state."""
        self.grid = Grid(self.env_config)
        self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]
        self.score = 0.0
        self.game_score = 0
        self.triangles_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0

        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.line_clear_flash_time = 0.0
        self.line_clear_highlight_time = 0.0
        self.game_over_flash_time = 0.0
        self.cleared_triangles_coords = []
        self.last_line_clear_info = None

        self.game_over = False
        self._last_action_valid = True
        self._last_timer_update_time = time.monotonic()  # Reset timer time

        self.demo_selected_shape_idx = 0
        self.demo_dragged_shape_idx = None
        self.demo_snapped_position = None

        self._last_potential = self.features.calculate_potential()

        return self.get_state()

    def step(self, action_index: int) -> Tuple[float, bool]:
        """
        Performs one game step based on the action index.
        Timer updates are handled within GameLogic.step().
        """
        return self.logic.step(action_index)

    def get_state(self) -> StateType:
        """Returns the current game state as a dictionary of numpy arrays."""
        return self.features.get_state()

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        # Removed is_frozen check here. Logic.step handles frozen state.
        # The collector also checks is_frozen separately.
        return self.logic.valid_actions()

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        return self.logic.decode_action(action_index)

    def is_over(self) -> bool:
        return self.game_over

    def is_frozen(self) -> bool:
        # Check freeze_time before allowing actions or progression
        is_currently_frozen = self.freeze_time > 0
        return is_currently_frozen

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

    def get_shapes(self) -> List[Optional[Shape]]:
        return self.shapes

    def _update_timers(self):
        """Updates timers for visual effects based on elapsed time."""
        now = time.monotonic()  # Use monotonic clock
        delta_time = now - self._last_timer_update_time
        self._last_timer_update_time = now

        # Ensure delta_time is non-negative
        delta_time = max(0.0, delta_time)

        self.freeze_time = max(0, self.freeze_time - delta_time)
        self.blink_time = max(0, self.blink_time - delta_time)
        self.line_clear_flash_time = max(0, self.line_clear_flash_time - delta_time)
        self.line_clear_highlight_time = max(
            0, self.line_clear_highlight_time - delta_time
        )
        self.game_over_flash_time = max(0, self.game_over_flash_time - delta_time)

        # Clear highlight coords only when highlight time runs out
        if self.line_clear_highlight_time <= 0 and self.cleared_triangles_coords:
            self.cleared_triangles_coords = []
        # Clear line clear info only when flash time runs out
        if self.line_clear_flash_time <= 0 and self.last_line_clear_info is not None:
            self.last_line_clear_info = None

    # --- Demo Mode Methods (Delegated) ---
    def select_shape_for_drag(self, shape_index: int):
        self.demo_logic.select_shape_for_drag(shape_index)

    def deselect_dragged_shape(self):
        self.demo_logic.deselect_dragged_shape()

    def update_snapped_position(self, grid_pos: Optional[Tuple[int, int]]):
        self.demo_logic.update_snapped_position(grid_pos)

    def place_dragged_shape(self) -> bool:
        # Update timers before placing shape in demo mode as well
        self._update_timers()
        return self.demo_logic.place_dragged_shape()

    def get_dragged_shape_info(
        self,
    ) -> Tuple[Optional[Shape], Optional[Tuple[int, int]]]:
        return self.demo_logic.get_dragged_shape_info()

    def toggle_triangle_debug(self, row: int, col: int):
        # Update timers before toggling in debug mode
        self._update_timers()
        self.demo_logic.toggle_triangle_debug(row, col)


File: environment\game_state_features.py
# File: environment/game_state_features.py
import numpy as np
from typing import TYPE_CHECKING, Dict, List
import copy

if TYPE_CHECKING:
    from .game_state import GameState

StateType = Dict[str, np.ndarray]


class GameStateFeatures:
    """Handles calculation of state features and potential outcomes."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def calculate_potential(self) -> float:
        """Calculates the potential function based on current grid state for PBRS."""
        if not self.gs.rewards.ENABLE_PBRS:
            return 0.0

        potential = 0.0
        max_height = self.gs.grid.get_max_height()
        num_holes = self.gs.grid.count_holes()
        bumpiness = self.gs.grid.get_bumpiness()

        potential += self.gs.rewards.PBRS_HEIGHT_COEF * max_height
        potential += self.gs.rewards.PBRS_HOLE_COEF * num_holes
        potential += self.gs.rewards.PBRS_BUMPINESS_COEF * bumpiness

        return potential

    def _calculate_potential_placement_outcomes(self) -> Dict[str, float]:
        """Calculates potential outcomes (tris cleared, holes, height, bumpiness) for valid moves."""
        valid_actions = self.gs.logic.valid_actions()  # Use logic helper
        if not valid_actions:
            return {
                "max_tris_cleared": 0.0,
                "min_holes": 0.0,
                "min_height": float(self.gs.grid.get_max_height()),
                "min_bump": float(self.gs.grid.get_bumpiness()),
            }

        max_triangles_cleared = 0
        min_new_holes = float("inf")
        min_resulting_height = float("inf")
        min_resulting_bumpiness = float("inf")
        initial_holes = self.gs.grid.count_holes()

        for action_index in valid_actions:
            shape_slot_index, target_row, target_col = self.gs.logic.decode_action(
                action_index
            )
            shape_to_place = self.gs.shapes[shape_slot_index]
            if shape_to_place is None:
                continue

            temp_grid = copy.deepcopy(self.gs.grid)
            temp_grid.place(shape_to_place, target_row, target_col)
            _, triangles_cleared, _ = temp_grid.clear_lines()
            holes_after = temp_grid.count_holes()
            height_after = temp_grid.get_max_height()
            bumpiness_after = temp_grid.get_bumpiness()
            new_holes_created = max(0, holes_after - initial_holes)

            max_triangles_cleared = max(max_triangles_cleared, triangles_cleared)
            min_new_holes = min(min_new_holes, new_holes_created)
            min_resulting_height = min(min_resulting_height, height_after)
            min_resulting_bumpiness = min(min_resulting_bumpiness, bumpiness_after)

        if min_new_holes == float("inf"):
            min_new_holes = 0.0
        if min_resulting_height == float("inf"):
            min_resulting_height = float(self.gs.grid.get_max_height())
        if min_resulting_bumpiness == float("inf"):
            min_resulting_bumpiness = float(self.gs.grid.get_bumpiness())

        return {
            "max_tris_cleared": float(max_triangles_cleared),
            "min_holes": float(min_new_holes),
            "min_height": float(min_resulting_height),
            "min_bump": float(min_resulting_bumpiness),
        }

    def get_state(self) -> StateType:
        """Returns the current game state as a dictionary of numpy arrays."""
        grid_state = self.gs.grid.get_feature_matrix()

        shape_features_per = self.gs.env_config.SHAPE_FEATURES_PER_SHAPE
        num_shapes_expected = self.gs.env_config.NUM_SHAPE_SLOTS
        shape_feature_matrix = np.zeros(
            (num_shapes_expected, shape_features_per), dtype=np.float32
        )
        max_tris_norm = 6.0
        max_h_norm = float(self.gs.grid.rows)
        max_w_norm = float(self.gs.grid.cols)
        for i in range(num_shapes_expected):
            s = self.gs.shapes[i] if i < len(self.gs.shapes) else None
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

        shape_availability_dim = self.gs.env_config.SHAPE_AVAILABILITY_DIM
        shape_availability_vector = np.zeros(shape_availability_dim, dtype=np.float32)
        for i in range(min(num_shapes_expected, shape_availability_dim)):
            if i < len(self.gs.shapes) and self.gs.shapes[i] is not None:
                shape_availability_vector[i] = 1.0

        explicit_features_dim = self.gs.env_config.EXPLICIT_FEATURES_DIM
        explicit_features_vector = np.zeros(explicit_features_dim, dtype=np.float32)
        num_holes = self.gs.grid.count_holes()
        col_heights = self.gs.grid.get_column_heights()
        avg_height = np.mean(col_heights) if col_heights else 0
        max_height = max(col_heights) if col_heights else 0
        bumpiness = self.gs.grid.get_bumpiness()
        max_possible_holes = self.gs.env_config.ROWS * self.gs.env_config.COLS
        max_possible_bumpiness = self.gs.env_config.ROWS * (self.gs.env_config.COLS - 1)
        explicit_features_vector[0] = np.clip(
            num_holes / max(1, max_possible_holes), 0.0, 1.0
        )
        explicit_features_vector[1] = np.clip(
            avg_height / self.gs.env_config.ROWS, 0.0, 1.0
        )
        explicit_features_vector[2] = np.clip(
            max_height / self.gs.env_config.ROWS, 0.0, 1.0
        )
        explicit_features_vector[3] = np.clip(
            bumpiness / max(1, max_possible_bumpiness), 0.0, 1.0
        )
        explicit_features_vector[4] = np.clip(
            self.gs.triangles_cleared_this_episode / 500.0, 0.0, 1.0
        )
        explicit_features_vector[5] = np.clip(
            self.gs.pieces_placed_this_episode / 500.0, 0.0, 1.0
        )

        if self.gs.env_config.CALCULATE_POTENTIAL_OUTCOMES_IN_STATE:
            potential_outcomes = self._calculate_potential_placement_outcomes()
            max_possible_tris_cleared = (
                self.gs.env_config.ROWS * self.gs.env_config.COLS
            )
            max_possible_new_holes = max_possible_holes
            explicit_features_vector[6] = np.clip(
                potential_outcomes["max_tris_cleared"]
                / max(1, max_possible_tris_cleared),
                0.0,
                1.0,
            )
            explicit_features_vector[7] = np.clip(
                potential_outcomes["min_holes"] / max(1, max_possible_new_holes),
                0.0,
                1.0,
            )
            explicit_features_vector[8] = np.clip(
                potential_outcomes["min_height"] / self.gs.env_config.ROWS, 0.0, 1.0
            )
            explicit_features_vector[9] = np.clip(
                potential_outcomes["min_bump"] / max(1, max_possible_bumpiness),
                0.0,
                1.0,
            )
        else:
            explicit_features_vector[6:10] = 0.0

        state_dict: StateType = {
            "grid": grid_state.astype(np.float32),
            "shapes": shape_feature_matrix.reshape(-1).astype(np.float32),
            "shape_availability": shape_availability_vector.astype(np.float32),
            "explicit_features": explicit_features_vector.astype(np.float32),
        }
        return state_dict


File: environment\grid.py
import numpy as np
from typing import List, Tuple, Set, Dict

from config import EnvConfig


from .triangle import Triangle
from .shape import Shape


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig): 
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.triangles: List[List[Triangle]] = []
        self._create(env_config)  # Pass config to _create
        self._link_neighbors()  # Link neighbors after creation
        self._identify_playable_lines()  # Identify all potential lines

    def _create(self, env_config: EnvConfig) -> None:
        """Initializes the grid with playable and death cells."""
        # Example pattern for a hexagon-like board within the grid bounds
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]  # Specific to 8 rows

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
            row_triangles: List[Triangle] = []
            base_playable_cols = cols_per_row[r]

            # Calculate padding for death cells
            initial_death_cols_left = (
                (self.cols - base_playable_cols) // 2
                if base_playable_cols < self.cols
                else 0
            )
            initial_first_death_col_right = initial_death_cols_left + base_playable_cols

            # Adjustment for Specific Hex Grid Pattern (makes it slightly narrower)
            adjusted_death_cols_left = initial_death_cols_left + 1
            adjusted_first_death_col_right = initial_first_death_col_right - 1

            for c in range(self.cols):
                is_death_cell = (
                    (c < adjusted_death_cols_left)
                    or (
                        c >= adjusted_first_death_col_right
                        and adjusted_first_death_col_right > adjusted_death_cols_left
                    )
                    or (base_playable_cols <= 2)  # Treat very narrow rows as death
                )
                is_up_cell = (r + c) % 2 == 0  
                triangle = Triangle(r, c, is_up=is_up_cell, is_death=is_death_cell)
                row_triangles.append(triangle)
            self.triangles.append(row_triangles)

    def _link_neighbors(self) -> None:
        """Iterates through the grid and sets neighbor references for each triangle."""
        for r in range(self.rows):
            for c in range(self.cols):
                current_tri = self.triangles[r][c]

                # Neighbor Left (X)
                if self.valid(r, c - 1):
                    current_tri.neighbor_left = self.triangles[r][c - 1]

                # Neighbor Right (Y)
                if self.valid(r, c + 1):
                    current_tri.neighbor_right = self.triangles[r][c + 1]

                # Neighbor Vertical (Z)
                if current_tri.is_up:
                    # Up triangle's vertical neighbor is below
                    if self.valid(r + 1, c):
                        current_tri.neighbor_vert = self.triangles[r + 1][c]
                else:
                    # Down triangle's vertical neighbor is above
                    if self.valid(r - 1, c):
                        current_tri.neighbor_vert = self.triangles[r - 1][c]

    def _identify_playable_lines(self):
        """
        Identifies all sets of playable triangles that form a complete line
        along horizontal (1-thick) and diagonal (2-thick) axes.
        Stores these sets for efficient checking later.
        """
        self.potential_lines: List[Set[Triangle]] = []

        # 1. Horizontal Lines (1-thick)
        for r in range(self.rows):
            line_triangles: List[Triangle] = []
            is_playable_row = False
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if not tri.is_death:
                    line_triangles.append(tri)
                    is_playable_row = True
                else:
                    # If we hit a death cell after finding playable cells, store the line segment
                    if line_triangles:
                        self.potential_lines.append(set(line_triangles))
                        line_triangles = []
                    # Reset if the row started with death cells
                    if not is_playable_row:
                        line_triangles = []

            # Add the last segment if the row ended with playable cells
            if line_triangles:
                self.potential_lines.append(set(line_triangles))

        # Helper function to generate single diagonal lines
        def get_single_diagonal_lines(
            k_func, r_range, c_func
        ) -> Dict[int, Set[Triangle]]:
            single_lines: Dict[int, Set[Triangle]] = {}
            min_k = float("inf")
            max_k = float("-inf")
            # Determine k range by checking all valid cells
            for r_check in range(self.rows):
                for c_check in range(self.cols):
                    if (
                        self.valid(r_check, c_check)
                        and not self.triangles[r_check][c_check].is_death
                    ):
                        k_val = k_func(r_check, c_check)
                        min_k = min(min_k, k_val)
                        max_k = max(max_k, k_val)

            if min_k > max_k:  # No playable cells found
                return {}

            for k in range(min_k, max_k + 1):
                line_triangles: List[Triangle] = []
                is_playable_line = False
                for r in r_range:
                    c = c_func(k, r)
                    if self.valid(r, c):
                        tri = self.triangles[r][c]
                        if not tri.is_death:
                            line_triangles.append(tri)
                            is_playable_line = True
                        else:
                            if line_triangles:  # End of a playable segment
                                if k not in single_lines:
                                    single_lines[k] = set()
                                single_lines[k].update(line_triangles)
                            if not is_playable_line:  # Reset if started with death
                                line_triangles = []
                            else:  # Break segment
                                line_triangles = []
                    elif (
                        line_triangles
                    ):  # End of playable segment due to invalid coords
                        if k not in single_lines:
                            single_lines[k] = set()
                        single_lines[k].update(line_triangles)
                        line_triangles = []

                if line_triangles:  # Add last segment
                    if k not in single_lines:
                        single_lines[k] = set()
                    single_lines[k].update(line_triangles)
            return single_lines

        # 2. Diagonal Lines TL-BR (k = c - r) - Combine adjacent k and k+1
        single_diag_tlbr = get_single_diagonal_lines(
            k_func=lambda r, c: c - r,
            r_range=range(self.rows),
            c_func=lambda k, r: k + r,
        )
        # Determine the actual range of k present in the dictionary keys
        k_values_tlbr = sorted(single_diag_tlbr.keys())
        if k_values_tlbr:
            for i in range(len(k_values_tlbr) - 1):
                k1 = k_values_tlbr[i]
                k2 = k_values_tlbr[i + 1]
                # Check if keys are adjacent (k+1)
                if k2 == k1 + 1:
                    # Combine the sets for the 2-thick diagonal line
                    combined_line = single_diag_tlbr[k1].union(single_diag_tlbr[k2])
                    if combined_line:  # Ensure not empty
                        self.potential_lines.append(combined_line)

        # 3. Diagonal Lines TR-BL (k = r + c) - Combine adjacent k and k+1
        single_diag_trbl = get_single_diagonal_lines(
            k_func=lambda r, c: r + c,
            r_range=range(self.rows),
            c_func=lambda k, r: k - r,
        )
        # Determine the actual range of k present in the dictionary keys
        k_values_trbl = sorted(single_diag_trbl.keys())
        if k_values_trbl:
            for i in range(len(k_values_trbl) - 1):
                k1 = k_values_trbl[i]
                k2 = k_values_trbl[i + 1]
                # Check if keys are adjacent (k+1)
                if k2 == k1 + 1:
                    # Combine the sets for the 2-thick diagonal line
                    combined_line = single_diag_trbl[k1].union(single_diag_trbl[k2])
                    if combined_line:  # Ensure not empty
                        self.potential_lines.append(combined_line)

        # Filter out potential lines that might be empty due to edge cases
        self.potential_lines = [line for line in self.potential_lines if line]
        # print(f"[Grid] Identified {len(self.potential_lines)} potential playable lines (H: 1-thick, D: 2-thick).")

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(
        self, shape_to_place: Shape, target_row: int, target_col: int
    ) -> bool:
        """Checks if a shape can be placed at the target location."""
        for dr, dc, is_up_shape_tri in shape_to_place.triangles:
            nr, nc = target_row + dr, target_col + dc
            if not self.valid(nr, nc):
                return False  # Out of bounds
            grid_triangle = self.triangles[nr][nc]
            # Cannot place on death cells, occupied cells, or cells with mismatching orientation
            if (
                grid_triangle.is_death
                or grid_triangle.is_occupied
                or (grid_triangle.is_up != is_up_shape_tri)
            ):
                return False
        return True  # All shape triangles can be placed

    def place(self, shape_to_place: Shape, target_row: int, target_col: int) -> None:
        """Places a shape onto the grid (assumes can_place was checked)."""
        for dr, dc, _ in shape_to_place.triangles:
            nr, nc = target_row + dr, target_col + dc
            if self.valid(nr, nc):
                grid_triangle = self.triangles[nr][nc]
                # Only occupy non-death, non-occupied cells
                if not grid_triangle.is_death and not grid_triangle.is_occupied:
                    grid_triangle.is_occupied = True
                    grid_triangle.color = shape_to_place.color

    def clear_lines(self) -> Tuple[int, int, List[Tuple[int, int]]]:
        """
        Checks for completed lines based on pre-identified potential lines.
        Clears all triangles belonging to any completed lines simultaneously.
        Returns the number of lines cleared, total triangles cleared, and their coordinates.
        """
        cleared_triangles_in_this_step: Set[Triangle] = set()
        lines_cleared_count = 0

        # Iterate through all potential lines identified during initialization
        for line_set in self.potential_lines:
            is_complete = True
            if not line_set:
                is_complete = False
            else:
                for triangle in line_set:
                    if not triangle.is_occupied:
                        is_complete = False
                        break

            if is_complete:
                cleared_triangles_in_this_step.update(line_set)
                lines_cleared_count += 1

        triangles_cleared_count = 0
        cleared_triangles_coords: List[Tuple[int, int]] = []

        if not cleared_triangles_in_this_step:
            return 0, 0, []  # Return 3 values even if none cleared

        for triangle in cleared_triangles_in_this_step:
            if not triangle.is_death and triangle.is_occupied:
                triangles_cleared_count += 1
                triangle.is_occupied = False
                triangle.color = None
                cleared_triangles_coords.append((triangle.row, triangle.col))

        return lines_cleared_count, triangles_cleared_count, cleared_triangles_coords


    def get_column_heights(self) -> List[int]:
        """Calculates the height of occupied cells in each column."""
        heights = [0] * self.cols
        for c in range(self.cols):
            for r in range(self.rows - 1, -1, -1):  # Iterate from top down
                triangle = self.triangles[r][c]
                if triangle.is_occupied and not triangle.is_death:
                    heights[c] = r + 1  # Height is row index + 1
                    break  # Found highest occupied cell in this column
        return heights

    def get_max_height(self) -> int:
        """Returns the maximum height across all columns."""
        heights = self.get_column_heights()
        return max(heights) if heights else 0

    def get_bumpiness(self) -> int:
        """Calculates the total absolute difference between adjacent column heights."""
        heights = self.get_column_heights()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def count_holes(self) -> int:
        """Counts the number of empty, non-death cells below an occupied cell in the same column."""
        holes = 0
        for c in range(self.cols):
            occupied_above_found = False
            for r in range(self.rows - 1, -1, -1):  # Iterate from top down
                triangle = self.triangles[r][c]
                if triangle.is_death:
                    occupied_above_found = (
                        False  # Reset if we hit a death cell column top
                    )
                    continue  # Skip death cells

                if triangle.is_occupied:
                    occupied_above_found = True
                elif not triangle.is_occupied and occupied_above_found:
                    # Found an empty cell below an occupied one in this column
                    holes += 1
        return holes

    def get_feature_matrix(self) -> np.ndarray:
        """Returns the grid state as a 2-channel numpy array (Occupancy, Orientation)."""
        # Channel 0: Occupancy (1.0 if occupied and not death, 0.0 otherwise)
        # Channel 1: Orientation (1.0 if pointing up and not death, 0.0 otherwise)
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                triangle = self.triangles[r][c]
                if not triangle.is_death:
                    grid_state[0, r, c] = 1.0 if triangle.is_occupied else 0.0
                    grid_state[1, r, c] = 1.0 if triangle.is_up else 0.0
        return grid_state


File: environment\shape.py
import random
from typing import List, Tuple

from config.constants import GOOGLE_COLORS


class Shape:
    """Represents a polyomino-like shape made of triangles."""

    def __init__(self) -> None:
        # List of (relative_row, relative_col, is_up) tuples defining the shape
        self.triangles: List[Tuple[int, int, bool]] = []
        # GOOGLE_COLORS is now imported from constants
        self.color: Tuple[int, int, int] = random.choice(GOOGLE_COLORS)
        self._generate()  # Generate the shape structure

    def _generate(self) -> None:
        """Generates a random shape by adding adjacent triangles."""
        num_triangles_in_shape = random.randint(1, 5)
        first_triangle_is_up = random.choice([True, False])
        # Add the root triangle at relative coordinates (0,0)
        self.triangles.append((0, 0, first_triangle_is_up))

        # Add remaining triangles adjacent to existing ones
        for _ in range(num_triangles_in_shape - 1):
            # Find valid neighbors of the *last added* triangle
            if not self.triangles:
                break  # Should not happen
            last_rel_row, last_rel_col, last_is_up = self.triangles[-1]
            valid_neighbors = self._find_valid_neighbors(
                last_rel_row, last_rel_col, last_is_up
            )
            if valid_neighbors:
                self.triangles.append(random.choice(valid_neighbors))
            # else: Could break early if no valid neighbors found, shape < n

    def _find_valid_neighbors(
        self, r: int, c: int, is_up: bool
    ) -> List[Tuple[int, int, bool]]:
        """Finds potential neighbor triangles that are not already part of the shape."""
        potential_neighbors: List[Tuple[int, int, bool]]
        if is_up:  # Neighbors of an UP triangle are DOWN triangles
            potential_neighbors = [
                (r, c - 1, False),
                (r, c + 1, False),
                (r + 1, c, False),
            ]
        else:  # Neighbors of a DOWN triangle are UP triangles
            potential_neighbors = [(r, c - 1, True), (r, c + 1, True), (r - 1, c, True)]
        # Return only neighbors that are not already in self.triangles
        valid_neighbors = [n for n in potential_neighbors if n not in self.triangles]
        return valid_neighbors

    def bbox(self) -> Tuple[int, int, int, int]:
        """Calculates the bounding box (min_r, min_c, max_r, max_c) of the shape."""
        if not self.triangles:
            return (0, 0, 0, 0)
        rows = [t[0] for t in self.triangles]
        cols = [t[1] for t in self.triangles]
        return (min(rows), min(cols), max(rows), max(cols))


File: environment\triangle.py
from typing import Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .grid import Grid  


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
        # Neighbors based on shared edges
        self.neighbor_left: Optional["Triangle"] = (
            None  # Corresponds to TS 'X' direction neighbor
        )
        self.neighbor_right: Optional["Triangle"] = (
            None  # Corresponds to TS 'Y' direction neighbor
        )
        self.neighbor_vert: Optional["Triangle"] = (
            None  # Corresponds to TS 'Z' direction neighbor (vertical)
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

    def get_line_neighbors(
        self,
    ) -> Tuple[Optional["Triangle"], Optional["Triangle"], Optional["Triangle"]]:
        """Returns neighbors relevant for line checking (left, right, vertical)."""
        return self.neighbor_left, self.neighbor_right, self.neighbor_vert


File: environment\__init__.py


File: stats\aggregator.py
# File: stats/aggregator.py
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, List
import numpy as np
import threading

from config import StatsConfig
from .aggregator_storage import AggregatorStorage
from .aggregator_logic import AggregatorLogic


class StatsAggregator:
    """
    Handles aggregation and storage of training statistics using deques.
    Calculates rolling averages and tracks best values. Does not perform logging.
    Includes locks for thread safety. Delegates storage and logic to helper classes.
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

        self._lock = threading.Lock()
        self.storage = AggregatorStorage(plot_window)
        self.logic = AggregatorLogic(self.storage)

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
        triangles_cleared: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            current_step = (
                global_step
                if global_step is not None
                else self.storage.current_global_step
            )
            update_info = self.logic.update_episode_stats(
                episode_score,
                episode_length,
                episode_num,
                current_step,
                game_score,
                triangles_cleared,
            )
            return update_info

    def record_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Records step data, now likely related to NN training steps."""
        with self._lock:
            update_info = self.logic.update_step_stats(step_data)
            return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            if current_global_step is None:
                current_global_step = self.storage.current_global_step
            summary = self.logic.calculate_summary(
                current_global_step, self.summary_avg_window
            )
            return summary

    def get_plot_data(self) -> Dict[str, Deque]:
        with self._lock:
            return self.storage.get_all_plot_deques()

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            state = self.storage.state_dict()
            state["plot_window"] = self.plot_window
            state["avg_windows"] = self.avg_windows
            return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        with self._lock:
            print("[StatsAggregator] Loading state...")
            self.plot_window = state_dict.get("plot_window", self.plot_window)
            self.avg_windows = state_dict.get("avg_windows", self.avg_windows)
            self.summary_avg_window = self.avg_windows[0] if self.avg_windows else 100

            self.storage.load_state_dict(state_dict, self.plot_window)

            print("[StatsAggregator] State loaded.")
            print(f"  -> Loaded total_episodes: {self.storage.total_episodes}")
            print(f"  -> Loaded best_score: {self.storage.best_score}")
            print(
                f"  -> Loaded start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.storage.start_time))}"
            )
            print(
                f"  -> Loaded training_target_step: {self.storage.training_target_step}"
            )
            print(
                f"  -> Loaded current_global_step: {self.storage.current_global_step}"
            )


File: stats\aggregator_logic.py
# File: stats/aggregator_logic.py
from collections import deque
from typing import Deque, Dict, Any, Optional, List
import numpy as np
import time

from .aggregator_storage import AggregatorStorage


class AggregatorLogic:
    """Handles the calculation logic for StatsAggregator."""

    def __init__(self, storage: AggregatorStorage):
        self.storage = storage

    def update_episode_stats(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        current_step: int,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Updates storage with episode data and checks for bests."""
        update_info = {"new_best_rl": False, "new_best_game": False}

        self.storage.episode_scores.append(episode_score)
        self.storage.episode_lengths.append(episode_length)
        if game_score is not None:
            self.storage.game_scores.append(game_score)
        if triangles_cleared is not None:
            self.storage.episode_triangles_cleared.append(triangles_cleared)
            self.storage.total_triangles_cleared += triangles_cleared
        self.storage.total_episodes = episode_num

        if episode_score > self.storage.best_score:
            self.storage.previous_best_score = self.storage.best_score
            self.storage.best_score = episode_score
            self.storage.best_score_step = current_step
            update_info["new_best_rl"] = True

        if game_score is not None and game_score > self.storage.best_game_score:
            self.storage.previous_best_game_score = self.storage.best_game_score
            self.storage.best_game_score = float(game_score)
            self.storage.best_game_score_step = current_step
            update_info["new_best_game"] = True

        self.storage.best_rl_score_history.append(self.storage.best_score)
        current_best_game = (
            int(self.storage.best_game_score)
            if self.storage.best_game_score > -float("inf")
            else 0
        )
        self.storage.best_game_score_history.append(current_best_game)

        return update_info

    def update_step_stats(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Updates storage with step data and checks for best loss."""
        g_step = step_data.get("global_step", self.storage.current_global_step)
        if g_step > self.storage.current_global_step:
            self.storage.current_global_step = g_step

        if "training_target_step" in step_data:
            self.storage.training_target_step = step_data["training_target_step"]

        update_info = {
            "new_best_loss": False,
            "new_best_policy_loss": False,
        }  # Added policy loss flag

        # Append to deques
        # --- NN Policy Loss ---
        if "policy_loss" in step_data and step_data["policy_loss"] is not None:
            current_policy_loss = step_data["policy_loss"]
            if np.isfinite(current_policy_loss):
                self.storage.policy_losses.append(current_policy_loss)
                if current_policy_loss < self.storage.best_policy_loss and g_step > 0:
                    self.storage.previous_best_policy_loss = (
                        self.storage.best_policy_loss
                    )
                    self.storage.best_policy_loss = current_policy_loss
                    self.storage.best_policy_loss_step = g_step
                    update_info["new_best_policy_loss"] = True
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Policy Loss: {current_policy_loss}"
                )
        # --- End NN Policy Loss ---

        # --- NN Value Loss ---
        if "value_loss" in step_data and step_data["value_loss"] is not None:
            current_value_loss = step_data["value_loss"]
            if np.isfinite(current_value_loss):
                self.storage.value_losses.append(current_value_loss)
                if current_value_loss < self.storage.best_value_loss and g_step > 0:
                    self.storage.previous_best_value_loss = self.storage.best_value_loss
                    self.storage.best_value_loss = current_value_loss
                    self.storage.best_value_loss_step = g_step
                    update_info["new_best_loss"] = (
                        True  # Keep original flag name for value loss
                    )
            else:
                print(
                    f"[Aggregator Warning] Received non-finite Value Loss: {current_value_loss}"
                )
        # --- End NN Value Loss ---

        # Removed Entropy

        # Append other optional metrics
        optional_metrics = [
            # Removed grad_norm, update_steps_per_second, minibatch_update_sps
            ("avg_max_q", "avg_max_qs"),
            ("beta", "beta_values"),
            ("buffer_size", "buffer_sizes"),
            ("lr", "lr_values"),
            ("epsilon", "epsilon_values"),
            ("cpu_usage", "cpu_usage"),
            ("memory_usage", "memory_usage"),
            ("gpu_memory_usage_percent", "gpu_memory_usage_percent"),
        ]
        for data_key, deque_name in optional_metrics:
            if data_key in step_data and step_data[data_key] is not None:
                val = step_data[data_key]
                if np.isfinite(val):
                    # Ensure deque exists before appending
                    if hasattr(self.storage, deque_name):
                        getattr(self.storage, deque_name).append(val)
                    else:
                        print(
                            f"[Aggregator Warning] Deque '{deque_name}' not found in storage."
                        )
                else:
                    print(f"[Aggregator Warning] Received non-finite {data_key}: {val}")

        # Update scalar values
        scalar_updates = {
            # Removed SPS scalars
            "beta": "current_beta",
            "buffer_size": "current_buffer_size",
            "lr": "current_lr",
            "epsilon": "current_epsilon",
            "cpu_usage": "current_cpu_usage",
            "memory_usage": "current_memory_usage",
            "gpu_memory_usage_percent": "current_gpu_memory_usage_percent",
        }
        for data_key, storage_key in scalar_updates.items():
            if data_key in step_data and step_data[data_key] is not None:
                val = step_data[data_key]
                if np.isfinite(val):
                    setattr(self.storage, storage_key, val)
                else:
                    print(f"[Aggregator Warning] Received non-finite {data_key}: {val}")

        return update_info

    def calculate_summary(
        self, current_global_step: int, summary_avg_window: int
    ) -> Dict[str, Any]:
        """Calculates the summary dictionary based on stored data."""

        def safe_mean(q_name: str, default=0.0) -> float:
            # Check if deque exists before accessing
            if not hasattr(self.storage, q_name):
                return default
            deque_instance = self.storage.get_deque(q_name)
            window_data = list(deque_instance)[-summary_avg_window:]
            finite_data = [x for x in window_data if np.isfinite(x)]
            return float(np.mean(finite_data)) if finite_data else default

        summary = {
            "avg_score_window": safe_mean("episode_scores"),
            "avg_length_window": safe_mean("episode_lengths"),
            "policy_loss": safe_mean("policy_losses"),  # Added policy loss
            "value_loss": safe_mean("value_losses"),
            # Removed entropy, avg_update_sps, avg_minibatch_sps
            "avg_max_q_window": safe_mean("avg_max_qs"),
            "avg_game_score_window": safe_mean("game_scores"),
            "avg_triangles_cleared_window": safe_mean("episode_triangles_cleared"),
            "avg_lr_window": safe_mean("lr_values", default=self.storage.current_lr),
            "avg_cpu_window": safe_mean("cpu_usage"),
            "avg_memory_window": safe_mean("memory_usage"),
            "avg_gpu_memory_window": safe_mean("gpu_memory_usage_percent"),
            "total_episodes": self.storage.total_episodes,
            "beta": self.storage.current_beta,
            "buffer_size": self.storage.current_buffer_size,
            # Removed SPS scalars
            "global_step": current_global_step,
            "current_lr": self.storage.current_lr,
            "best_score": self.storage.best_score,
            "previous_best_score": self.storage.previous_best_score,
            "best_score_step": self.storage.best_score_step,
            "best_game_score": self.storage.best_game_score,
            "previous_best_game_score": self.storage.previous_best_game_score,
            "best_game_score_step": self.storage.best_game_score_step,
            "best_loss": self.storage.best_value_loss,  # Keep as value loss best
            "previous_best_loss": self.storage.previous_best_value_loss,
            "best_loss_step": self.storage.best_value_loss_step,
            "best_policy_loss": self.storage.best_policy_loss,  # Added policy loss best
            "previous_best_policy_loss": self.storage.previous_best_policy_loss,
            "best_policy_loss_step": self.storage.best_policy_loss_step,
            "num_ep_scores": len(self.storage.episode_scores),
            "num_losses": len(
                self.storage.value_losses
            ),  # Maybe rename to num_value_losses?
            "summary_avg_window_size": summary_avg_window,
            "start_time": self.storage.start_time,
            "training_target_step": self.storage.training_target_step,
            "current_cpu_usage": self.storage.current_cpu_usage,
            "current_memory_usage": self.storage.current_memory_usage,
            "current_gpu_memory_usage_percent": self.storage.current_gpu_memory_usage_percent,
        }
        return summary


File: stats\aggregator_storage.py
from collections import deque
from typing import Deque, Dict, Any,
import time


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator."""

    def __init__(self, plot_window: int):
        self.plot_window = plot_window

        # --- Deques for Plotting ---
        # Removed: self.policy_losses
        # Removed: self.value_losses
        # Removed: self.entropies
        # Removed: self.grad_norms
        self.avg_max_qs: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep if Q-values are estimated by NN
        self.episode_scores: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep RL score if used, or repurpose
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_triangles_cleared: Deque[int] = deque(maxlen=plot_window)
        # Removed: self.sps_values
        # Removed: self.update_steps_per_second_values
        # Removed: self.minibatch_update_sps_values
        self.buffer_sizes: Deque[int] = deque(
            maxlen=plot_window
        )  # Might be useful for MCTS buffer/NN training data
        self.beta_values: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep if used (e.g., PER)
        self.best_rl_score_history: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep RL score if used
        self.best_game_score_history: Deque[int] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep for NN training LR
        self.epsilon_values: Deque[float] = deque(
            maxlen=plot_window
        )  # Keep if epsilon-greedy is used in MCTS/NN
        self.cpu_usage: Deque[float] = deque(maxlen=plot_window)
        self.memory_usage: Deque[float] = deque(maxlen=plot_window)
        self.gpu_memory_usage_percent: Deque[float] = deque(maxlen=plot_window)

        # --- Scalar State Variables ---
        self.total_episodes = 0
        self.total_triangles_cleared = 0
        self.current_epsilon: float = 0.0
        self.current_beta: float = 0.0
        self.current_buffer_size: int = 0
        self.current_global_step: int = (
            0  # Might represent training steps or games played now
        )

        self.current_lr: float = 0.0  # Keep for NN
        self.start_time: float = time.time()
        self.training_target_step: int = 0  # Target might be games played or NN steps
        self.current_cpu_usage: float = 0.0
        self.current_memory_usage: float = 0.0
        self.current_gpu_memory_usage_percent: float = 0.0

        # --- Best Value Tracking ---
        self.best_score: float = -float("inf")  # Keep RL score if used
        self.previous_best_score: float = -float("inf")
        self.best_score_step: int = 0
        self.best_game_score: float = -float("inf")
        self.previous_best_game_score: float = -float("inf")
        self.best_game_score_step: int = 0
        self.best_value_loss: float = float("inf")  # Keep for NN value head loss
        self.previous_best_value_loss: float = float("inf")
        self.best_value_loss_step: int = 0
        # Add best policy loss?
        self.best_policy_loss: float = float("inf")
        self.previous_best_policy_loss: float = float("inf")
        self.best_policy_loss_step: int = 0

    def get_deque(self, name: str) -> Deque:
        """Safely gets a deque attribute."""
        return getattr(self, name, deque(maxlen=self.plot_window))

    def get_all_plot_deques(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        deque_names = [
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            # Add NN losses if tracked
            "policy_losses",  # Added back for NN policy head
            "value_losses",  # Kept for NN value head
        ]
        # Filter out names that might not exist if loaded from old state
        return {
            name: self.get_deque(name).copy()
            for name in deque_names
            if hasattr(self, name)
        }

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the storage for saving."""
        state = {}
        # Deques
        deque_names = [
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            # Add NN losses if tracked
            "policy_losses",  # Added back for NN policy head
            "value_losses",  # Kept for NN value head
        ]
        for name in deque_names:
            if hasattr(self, name):  # Check if deque exists before saving
                state[name] = list(self.get_deque(name))

        # Scalar State Variables
        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_epsilon",
            "current_beta",
            "current_buffer_size",
            "current_global_step",
            # Removed SPS scalars
            "current_lr",
            "start_time",
            "training_target_step",
            "current_cpu_usage",
            "current_memory_usage",
            "current_gpu_memory_usage_percent",
        ]
        for key in scalar_keys:
            state[key] = getattr(self, key, 0)

        # Best Value Tracking
        best_value_keys = [
            "best_score",
            "previous_best_score",
            "best_score_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
            "best_policy_loss",  # Added policy loss tracking
            "previous_best_policy_loss",
            "best_policy_loss_step",
        ]
        for key in best_value_keys:
            state[key] = getattr(self, key, 0)

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        self.plot_window = plot_window

        deque_names = [
            "avg_max_qs",
            "episode_scores",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "buffer_sizes",
            "beta_values",
            "best_rl_score_history",
            "best_game_score_history",
            "lr_values",
            "epsilon_values",
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            "policy_losses",
            "value_losses",
        ]
        for key in deque_names:
            data_to_load = state_dict.get(key)
            if data_to_load is not None:
                try:
                    if isinstance(data_to_load, (list, tuple)):
                        setattr(self, key, deque(data_to_load, maxlen=self.plot_window))
                    else:
                        print(
                            f"  -> Warning: Invalid type for deque '{key}'. Initializing empty."
                        )
                        setattr(self, key, deque(maxlen=self.plot_window))
                except Exception as e:
                    print(f"  -> Error loading deque '{key}': {e}. Initializing empty.")
                    setattr(self, key, deque(maxlen=self.plot_window))
            else:
                # Ensure deque exists even if not in state_dict
                setattr(self, key, deque(maxlen=self.plot_window))

        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_epsilon",
            "current_beta",
            "current_buffer_size",
            "current_global_step",
            "current_lr",
            "start_time",
            "training_target_step",
            "current_cpu_usage",
            "current_memory_usage",
            "current_gpu_memory_usage_percent",
        ]
        default_values = {
            "start_time": time.time(),
            "training_target_step": 0,
            "current_global_step": 0,
        }
        for key in scalar_keys:
            value_to_load = state_dict.get(key, default_values.get(key, 0))
            setattr(self, key, value_to_load)

        # Best Value Tracking
        best_value_keys = [
            "best_score",
            "previous_best_score",
            "best_score_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
            "best_policy_loss",
            "previous_best_policy_loss",
            "best_policy_loss_step",
        ]
        default_best = {
            "best_score": -float("inf"),
            "previous_best_score": -float("inf"),
            "best_game_score": -float("inf"),
            "previous_best_game_score": -float("inf"),
            "best_value_loss": float("inf"),
            "previous_best_value_loss": float("inf"),
            "best_policy_loss": float("inf"),  # Added policy loss tracking
            "previous_best_policy_loss": float("inf"),
        }
        for key in best_value_keys:
            setattr(self, key, state_dict.get(key, default_best.get(key, 0)))

        # Ensure current_global_step exists after loading
        if not hasattr(self, "current_global_step"):
            self.current_global_step = 0
        # Ensure training_target_step exists after loading
        if not hasattr(self, "training_target_step"):
            self.training_target_step = 0
        # Ensure policy loss tracking exists
        if not hasattr(self, "best_policy_loss"):
            self.best_policy_loss = float("inf")
        if not hasattr(self, "previous_best_policy_loss"):
            self.previous_best_policy_loss = float("inf")
        if not hasattr(self, "best_policy_loss_step"):
            self.best_policy_loss_step = 0


File: stats\simple_stats_recorder.py
# File: stats/simple_stats_recorder.py
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List
import numpy as np
import torch
import threading

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from config import StatsConfig


class SimpleStatsRecorder(StatsRecorderBase):
    """
    Logs aggregated statistics to the console periodically. Thread-safe.
    Delegates data storage and aggregation to a StatsAggregator instance.
    Provides no-op implementations for histogram, image, hparam, graph logging.
    """

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_log_interval: int = StatsConfig.CONSOLE_LOG_FREQ,
    ):
        self.aggregator = aggregator
        # Console log interval might now represent episodes or training steps
        self.console_log_interval = (
            max(1, console_log_interval) if console_log_interval > 0 else -1
        )
        self.last_log_time: float = time.time()
        self.start_time: float = time.time()
        self.summary_avg_window = self.aggregator.summary_avg_window
        # Counter might track episodes or training steps now
        self.updates_since_last_log = 0  # Renamed from rollouts_since_last_log

        self._lock = threading.Lock()

        print(
            f"[SimpleStatsRecorder] Initialized. Console Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'} updates/episodes"
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
        triangles_cleared: Optional[int] = None,
    ):
        """Records episode stats and prints new bests to console. Thread-safe."""
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
        )
        current_step = (
            global_step
            if global_step is not None
            else self.aggregator.storage.current_global_step
        )
        step_info = f"at Step ~{current_step/1e6:.1f}M"  # Step might mean NN steps now

        # Print new bests immediately
        if update_info.get("new_best_rl"):
            prev_str = (
                f"{self.aggregator.storage.previous_best_score:.2f}"
                if self.aggregator.storage.previous_best_score > -float("inf")
                else "N/A"
            )
            print(
                f"\n--- 🏆 New Best RL: {self.aggregator.storage.best_score:.2f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_game"):
            prev_str = (
                f"{self.aggregator.storage.previous_best_game_score:.0f}"
                if self.aggregator.storage.previous_best_game_score > -float("inf")
                else "N/A"
            )
            print(
                f"--- 🎮 New Best Game: {self.aggregator.storage.best_game_score:.0f} {step_info} (Prev: {prev_str}) ---"
            )
        # Check for new best NN losses
        if update_info.get("new_best_loss"):  # Value loss
            prev_str = (
                f"{self.aggregator.storage.previous_best_value_loss:.4f}"
                if self.aggregator.storage.previous_best_value_loss < float("inf")
                else "N/A"
            )
            print(
                f"---📉 New Best V.Loss: {self.aggregator.storage.best_value_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_policy_loss"):  # Policy loss
            prev_str = (
                f"{self.aggregator.storage.previous_best_policy_loss:.4f}"
                if self.aggregator.storage.previous_best_policy_loss < float("inf")
                else "N/A"
            )
            print(
                f"---📉 New Best P.Loss: {self.aggregator.storage.best_policy_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )

        # Trigger console log based on episode count if interval is set
        log_now = False
        with self._lock:
            self.updates_since_last_log += 1
            if (
                self.console_log_interval > 0
                and self.updates_since_last_log >= self.console_log_interval
            ):
                log_now = True
                self.updates_since_last_log = 0

        if log_now:
            self.log_summary(current_step)

    def record_step(self, step_data: Dict[str, Any]):
        """Records step stats (e.g., NN update) and triggers console logging if interval met. Thread-safe."""
        update_info = self.aggregator.record_step(step_data)
        g_step = step_data.get(
            "global_step", self.aggregator.storage.current_global_step
        )

        # Print new best loss immediately if it occurred during this step's update
        if update_info.get("new_best_loss"):  # Value loss
            prev_str = (
                f"{self.aggregator.storage.previous_best_value_loss:.4f}"
                if self.aggregator.storage.previous_best_value_loss < float("inf")
                else "N/A"
            )
            step_info = f"at Step ~{g_step/1e6:.1f}M"
            print(
                f"---📉 New Best V.Loss: {self.aggregator.storage.best_value_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )
        if update_info.get("new_best_policy_loss"):  # Policy loss
            prev_str = (
                f"{self.aggregator.storage.previous_best_policy_loss:.4f}"
                if self.aggregator.storage.previous_best_policy_loss < float("inf")
                else "N/A"
            )
            step_info = f"at Step ~{g_step/1e6:.1f}M"
            print(
                f"---📉 New Best P.Loss: {self.aggregator.storage.best_policy_loss:.4f} {step_info} (Prev: {prev_str}) ---"
            )

        # Increment counter and check logging frequency (thread-safe)
        # Logging frequency might now be based on NN updates instead of rollouts
        log_now = False
        with self._lock:
            # Increment counter if an NN update occurred (check for loss keys)
            if "policy_loss" in step_data or "value_loss" in step_data:
                self.updates_since_last_log += 1
                if (
                    self.console_log_interval > 0
                    and self.updates_since_last_log >= self.console_log_interval
                ):
                    log_now = True
                    self.updates_since_last_log = 0

        if log_now:
            self.log_summary(g_step)

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Gets the summary dictionary from the aggregator (thread-safe)."""
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        """Gets the plot data deques from the aggregator (thread-safe)."""
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        """Logs the current summary statistics to the console."""
        summary = self.get_summary(global_step)
        elapsed_runtime = time.time() - self.aggregator.storage.start_time
        runtime_hrs = elapsed_runtime / 3600

        best_score_val = (
            f"{summary['best_score']:.2f}"
            if summary["best_score"] > -float("inf")
            else "N/A"
        )
        best_game_score_val = (
            f"{summary['best_game_score']:.0f}"
            if summary["best_game_score"] > -float("inf")
            else "N/A"
        )
        best_v_loss_val = (
            f"{summary['best_loss']:.4f}"  # Value loss
            if summary["best_loss"] < float("inf")
            else "N/A"
        )
        best_p_loss_val = (
            f"{summary['best_policy_loss']:.4f}"  # Policy loss
            if summary["best_policy_loss"] < float("inf")
            else "N/A"
        )
        avg_window_size = summary.get("summary_avg_window_size", "?")

        # Removed SPS
        log_str = (
            f"[{runtime_hrs:.1f}h|Console] Step: {global_step/1e6:<6.2f}M | "  # Step might mean NN steps
            f"Ep: {summary['total_episodes']:<7} | "
            # f"RLScore(Avg{avg_window_size}): {summary['avg_score_window']:<6.2f} (Best: {best_score_val}) | " # Keep RL score?
            f"GameScore(Avg{avg_window_size}): {summary['avg_game_score_window']:<6.0f} (Best: {best_game_score_val}) | "
            f"V.Loss(Avg{avg_window_size}): {summary['value_loss']:.4f} (Best: {best_v_loss_val}) | "
            f"P.Loss(Avg{avg_window_size}): {summary['policy_loss']:.4f} (Best: {best_p_loss_val}) | "
            f"LR: {summary['current_lr']:.1e}"
        )
        avg_tris_cleared = summary.get("avg_triangles_cleared_window", 0.0)
        log_str += f" | TrisClr(Avg{avg_window_size}): {avg_tris_cleared:.1f}"

        print(log_str)

        self.last_log_time = time.time()

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

    def close(self, is_cleanup: bool = False):
        """Closes the recorder (no action needed for simple console logger)."""
        print(f"[SimpleStatsRecorder] Closed (is_cleanup={is_cleanup}).")


File: stats\stats_recorder.py
# File: stats/stats_recorder.py
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Dict, Any, Optional, Union
import numpy as np
import torch

# Removed: from .stats_recorder import StatsRecorderBase


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
        triangles_cleared: Optional[int] = None,
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
    def close(self, is_cleanup: bool = False):
        """Perform any necessary cleanup."""
        pass


File: stats\tb_histogram_logger.py
# File: stats/tb_histogram_logger.py
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from typing import Union, List, Optional
import numpy as np


class TBHistogramLogger:
    """Handles logging histograms to TensorBoard based on frequency."""

    def __init__(
        self, writer: Optional[SummaryWriter], lock: threading.Lock, log_interval: int
    ):
        self.writer = writer
        self._lock = lock
        self.log_interval = log_interval  # Interval in terms of updates/rollouts
        self.last_log_step = -1
        self.rollouts_since_last_log = 0  # Track rollouts internally

    def should_log(self, global_step: int) -> bool:
        """Checks if enough rollouts have passed and the global step has advanced."""
        if not self.writer or self.log_interval <= 0:
            return False
        # Check based on internal rollout counter and if step has advanced
        return (
            self.rollouts_since_last_log >= self.log_interval
            and global_step > self.last_log_step
        )

    def log_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Logs a histogram if the interval condition is met."""
        if not self.should_log(global_step):
            # Increment counter even if not logging this specific histogram
            # This assumes record_step increments the counter externally
            # Let's manage the counter internally based on when log_histogram is called
            # self.rollouts_since_last_log += 1 # No, let external call manage this
            return

        with self._lock:
            # Double check condition inside lock
            if global_step > self.last_log_step:
                try:
                    self.writer.add_histogram(tag, values, global_step)
                    self.last_log_step = global_step
                    # Reset counter after successful log
                    # self.rollouts_since_last_log = 0 # Resetting counter is handled in record_step
                except Exception as e:
                    print(f"Error logging histogram '{tag}': {e}")

    def increment_rollout_counter(self):
        """Increments the internal counter, called after each update/rollout."""
        if self.log_interval > 0:
            self.rollouts_since_last_log += 1

    def reset_rollout_counter(self):
        """Resets the counter, called after logging."""
        self.rollouts_since_last_log = 0


File: stats\tb_hparam_logger.py
# File: stats/tb_hparam_logger.py
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from typing import Dict, Any, Optional
import traceback


class TBHparamLogger:
    """Handles logging hyperparameters and final metrics to TensorBoard."""

    def __init__(
        self,
        writer: Optional[SummaryWriter],
        lock: threading.Lock,
        hparam_dict: Optional[Dict[str, Any]],
    ):
        self.writer = writer
        self._lock = lock
        self.hparam_dict = hparam_dict if hparam_dict else {}
        self.initial_hparams_logged = False

    def _filter_hparams(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
        """Filters hyperparameters to types supported by TensorBoard."""
        return {
            k: v
            for k, v in hparams.items()
            if isinstance(v, (int, float, str, bool, torch.Tensor))
        }

    def log_initial_hparams(self):
        """Logs hyperparameters at the beginning of the run."""
        if not self.writer or not self.hparam_dict or self.initial_hparams_logged:
            return
        with self._lock:
            try:
                initial_metrics = {
                    "hparam/final_best_rl_score": -float("inf"),
                    "hparam/final_best_game_score": -float("inf"),
                    "hparam/final_best_loss": float("inf"),
                    "hparam/final_total_episodes": 0,
                }
                filtered_hparams = self._filter_hparams(self.hparam_dict)
                self.writer.add_hparams(filtered_hparams, initial_metrics, run_name=".")
                self.initial_hparams_logged = True
                print("[TensorBoardStatsRecorder] Hyperparameters logged.")
            except Exception as e:
                print(f"Error logging initial hyperparameters: {e}")
                traceback.print_exc()

    def log_final_hparams(
        self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]
    ):
        """Logs final hyperparameters and metrics."""
        if not self.writer:
            return
        with self._lock:
            try:
                filtered_hparams = self._filter_hparams(hparam_dict)
                filtered_metrics = {
                    k: v for k, v in metric_dict.items() if isinstance(v, (int, float))
                }
                self.writer.add_hparams(
                    filtered_hparams, filtered_metrics, run_name="."
                )
                print("[TensorBoardStatsRecorder] Final hparams and metrics logged.")
            except Exception as e:
                print(f"Error logging final hyperparameters/metrics: {e}")
                traceback.print_exc()

    def log_final_hparams_from_summary(self, final_summary: Dict[str, Any]):
        """Logs final hparams using the stored hparam_dict and metrics from summary."""
        if not self.hparam_dict:
            print(
                "[TensorBoardStatsRecorder] Skipping final hparam logging (hparam_dict not set)."
            )
            return
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
        self.log_final_hparams(self.hparam_dict, final_metrics)


File: stats\tb_image_logger.py
# File: stats/tb_image_logger.py
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from typing import Union, Optional
import numpy as np
import traceback

from .tb_log_utils import format_image_for_tb


class TBImageLogger:
    """Handles logging images to TensorBoard based on frequency."""

    def __init__(
        self, writer: Optional[SummaryWriter], lock: threading.Lock, log_interval: int
    ):
        self.writer = writer
        self._lock = lock
        self.log_interval = log_interval  # Interval in terms of updates/rollouts
        self.last_log_step = -1
        self.rollouts_since_last_log = 0  # Track rollouts internally

    def should_log(self, global_step: int) -> bool:
        """Checks if enough rollouts have passed and the global step has advanced."""
        if not self.writer or self.log_interval <= 0:
            return False
        return (
            self.rollouts_since_last_log >= self.log_interval
            and global_step > self.last_log_step
        )

    def log_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Logs an image if the interval condition is met."""
        if not self.should_log(global_step):
            # self.rollouts_since_last_log += 1 # No, let external call manage this
            return

        with self._lock:
            # Double check condition inside lock
            if global_step > self.last_log_step:
                try:
                    image_tensor = format_image_for_tb(image)
                    self.writer.add_image(
                        tag, image_tensor, global_step, dataformats="CHW"
                    )
                    self.last_log_step = global_step
                    # self.rollouts_since_last_log = 0 # Resetting counter is handled in record_step
                except Exception as e:
                    print(f"Error logging image '{tag}': {e}")
                    # traceback.print_exc() # Optional: more detail

    def increment_rollout_counter(self):
        """Increments the internal counter, called after each update/rollout."""
        if self.log_interval > 0:
            self.rollouts_since_last_log += 1

    def reset_rollout_counter(self):
        """Resets the counter, called after logging."""
        self.rollouts_since_last_log = 0


File: stats\tb_log_utils.py
# File: stats/tb_log_utils.py
import torch
import numpy as np
from typing import Union


def format_image_for_tb(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Formats an image (numpy or tensor) into CHW format for TensorBoard."""
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[-1] in [1, 3, 4]:  # HWC
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        elif image.ndim == 2:  # HW (grayscale)
            image_tensor = torch.from_numpy(image).unsqueeze(0)
        else:  # Assume CHW or other format, pass through
            image_tensor = torch.from_numpy(image)
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] not in [1, 3, 4]:  # Likely HWC
            if image.shape[-1] in [1, 3, 4]:
                image_tensor = image.permute(2, 0, 1)
            else:  # Unknown format, pass through
                image_tensor = image
        elif image.ndim == 2:  # HW
            image_tensor = image.unsqueeze(0)
        else:  # Assume CHW or other format
            image_tensor = image
    else:
        raise TypeError(f"Unsupported image type for TensorBoard: {type(image)}")

    # Ensure correct data type (e.g., uint8 or float) - TB handles this mostly
    return image_tensor


File: stats\tb_scalar_logger.py
# File: stats/tb_scalar_logger.py
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from typing import Dict, Any, Optional

from .aggregator import StatsAggregator  # For type hinting


class TBScalarLogger:
    """Handles logging scalar values to TensorBoard."""

    def __init__(self, writer: Optional[SummaryWriter], lock: threading.Lock):
        self.writer = writer
        self._lock = lock

    def log_episode_scalars(
        self,
        g_step: int,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        game_score: Optional[int],
        triangles_cleared: Optional[int],
        update_info: Dict[str, Any],
        aggregator: StatsAggregator,  # Pass aggregator for best values
    ):
        """Logs scalars related to a completed episode."""
        if not self.writer:
            return
        with self._lock:
            try:
                self.writer.add_scalar("Episode/Score", episode_score, g_step)
                self.writer.add_scalar("Episode/Length", episode_length, g_step)
                if game_score is not None:
                    self.writer.add_scalar("Episode/Game Score", game_score, g_step)
                if triangles_cleared is not None:
                    self.writer.add_scalar(
                        "Episode/Triangles Cleared", triangles_cleared, g_step
                    )
                self.writer.add_scalar("Progress/Total Episodes", episode_num, g_step)

                # --- Corrected Access ---
                if update_info.get("new_best_rl"):
                    self.writer.add_scalar(
                        "Best Performance/RL Score",
                        aggregator.storage.best_score,
                        g_step,
                    )
                if update_info.get("new_best_game"):
                    self.writer.add_scalar(
                        "Best Performance/Game Score",
                        aggregator.storage.best_game_score,
                        g_step,
                    )
                # --- End Correction ---
            except Exception as e:
                print(f"Error writing episode scalars to TensorBoard: {e}")

    def log_step_scalars(
        self,
        g_step: int,
        step_data: Dict[str, Any],
        update_info: Dict[str, Any],
        aggregator: StatsAggregator,  # Pass aggregator for best values
    ):
        """Logs scalars related to a training/environment step."""
        if not self.writer:
            return
        with self._lock:
            try:
                scalar_map = {
                    "policy_loss": "Loss/Policy Loss",  # Keep policy loss for NN
                    "value_loss": "Loss/Value Loss",  # Keep value loss for NN
                    # Removed entropy, grad_norm, sps_collection, update_steps_per_second, minibatch_update_sps
                    "avg_max_q": "Debug/Avg Max Q",  # Keep if NN estimates Q
                    "beta": "Debug/Beta",  # Keep if PER used
                    "buffer_size": "Debug/Buffer Size",  # Keep for MCTS/NN buffer
                    "lr": "Train/Learning Rate",  # Keep for NN
                    "epsilon": "Train/Epsilon",  # Keep if used
                    "update_time": "Performance/Update Time",  # Keep for NN update time
                    "step_time": "Performance/Total Step Time",  # Keep if relevant
                    "cpu_usage": "Resource/CPU Usage (%)",
                    "memory_usage": "Resource/Memory Usage (%)",
                    "gpu_memory_usage_percent": "Resource/GPU Memory Usage (%)",
                }
                for key, tag in scalar_map.items():
                    if key in step_data and step_data[key] is not None:
                        self.writer.add_scalar(tag, step_data[key], g_step)

                # --- Corrected Access ---
                if update_info.get("new_best_loss"):  # Value loss
                    self.writer.add_scalar(
                        "Best Performance/Value Loss",
                        aggregator.storage.best_value_loss,
                        g_step,
                    )
                if update_info.get("new_best_policy_loss"):  # Policy loss
                    self.writer.add_scalar(
                        "Best Performance/Policy Loss",
                        aggregator.storage.best_policy_loss,
                        g_step,
                    )
                # --- End Correction ---
            except Exception as e:
                print(f"Error writing step scalars to TensorBoard: {e}")


File: stats\tensorboard_logger.py
# File: stats/tensorboard_logger.py
import time
import traceback
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import threading

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from .simple_stats_recorder import SimpleStatsRecorder
from config import (
    TensorBoardConfig,
    EnvConfig,
    RNNConfig,
)  # Keep RNNConfig for potential future use

# Removed ActorCriticNetwork import

# Import helper modules
from .tb_log_utils import format_image_for_tb
from .tb_scalar_logger import TBScalarLogger
from .tb_histogram_logger import TBHistogramLogger
from .tb_image_logger import TBImageLogger
from .tb_hparam_logger import TBHparamLogger


class TensorBoardStatsRecorder(StatsRecorderBase):
    """
    Logs aggregated statistics, histograms, images, and hyperparameters to TensorBoard. Thread-safe.
    Uses a SimpleStatsRecorder for console logging and a StatsAggregator for data handling.
    Delegates specific logging tasks to helper classes.
    """

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_recorder: SimpleStatsRecorder,
        log_dir: str,
        hparam_dict: Optional[Dict[str, Any]] = None,
        model_for_graph: Optional[torch.nn.Module] = None,  # Changed type hint
        dummy_input_for_graph: Optional[Any] = None,  # Changed type hint
        histogram_log_interval: int = TensorBoardConfig.HISTOGRAM_LOG_FREQ,
        image_log_interval: int = TensorBoardConfig.IMAGE_LOG_FREQ,
        env_config: Optional[EnvConfig] = None,
        rnn_config: Optional[RNNConfig] = None,
    ):
        self.aggregator = aggregator
        self.console_recorder = console_recorder
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None
        self._lock = threading.Lock()

        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"[TensorBoardStatsRecorder] Initialized. Logging to: {self.log_dir}")

            # Initialize helper loggers
            self.scalar_logger = TBScalarLogger(self.writer, self._lock)
            self.histogram_logger = TBHistogramLogger(
                self.writer, self._lock, histogram_log_interval
            )
            self.image_logger = TBImageLogger(
                self.writer, self._lock, image_log_interval
            )
            self.hparam_logger = TBHparamLogger(self.writer, self._lock, hparam_dict)

            if model_for_graph and dummy_input_for_graph:
                self.record_graph(model_for_graph, dummy_input_for_graph)
            else:
                print("[TensorBoardStatsRecorder] Model graph logging skipped.")

            self.hparam_logger.log_initial_hparams()

        except Exception as e:
            print(f"FATAL: Error initializing TensorBoard SummaryWriter: {e}")
            traceback.print_exc()
            self.writer = None
            self.scalar_logger = None
            self.histogram_logger = None
            self.image_logger = None
            self.hparam_logger = None

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
    ):
        """Records episode stats to TensorBoard and delegates to console recorder."""
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
        )
        g_step = (
            global_step
            if global_step is not None
            else getattr(self.aggregator.storage, "current_global_step", 0)
        )

        if self.scalar_logger:
            self.scalar_logger.log_episode_scalars(
                g_step,
                episode_score,
                episode_length,
                episode_num,
                game_score,
                triangles_cleared,
                update_info,
                self.aggregator,
            )

        self.console_recorder.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
        )

    def record_step(self, step_data: Dict[str, Any]):
        """Records step stats (e.g., NN training step) to TensorBoard and console."""
        update_info = self.aggregator.record_step(step_data)
        g_step = step_data.get(
            "global_step",
            getattr(self.aggregator.storage, "current_global_step", 0),
        )

        if self.scalar_logger:
            self.scalar_logger.log_step_scalars(
                g_step, step_data, update_info, self.aggregator
            )

        # Increment histogram/image counters if an update occurred
        # Check for a key indicating an NN update, e.g., 'policy_loss' or 'value_loss'
        if "policy_loss" in step_data or "value_loss" in step_data:
            if self.histogram_logger:
                self.histogram_logger.increment_rollout_counter()
                if self.histogram_logger.should_log(g_step):
                    self.histogram_logger.reset_rollout_counter()  # Reset only if logged
            if self.image_logger:
                self.image_logger.increment_rollout_counter()
                if self.image_logger.should_log(g_step):
                    self.image_logger.reset_rollout_counter()  # Reset only if logged

        self.console_recorder.record_step(step_data)

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Records a histogram to TensorBoard using the helper."""
        if self.histogram_logger:
            self.histogram_logger.log_histogram(tag, values, global_step)

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Records an image to TensorBoard using the helper."""
        if self.image_logger:
            self.image_logger.log_image(tag, image, global_step)

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Records final hyperparameters and metrics using the helper."""
        if self.hparam_logger:
            self.hparam_logger.log_final_hparams(hparam_dict, metric_dict)

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[Any] = None
    ):
        """Records the model graph to TensorBoard."""
        if not self.writer:
            return
        if input_to_model is None:
            print("Warning: Cannot record graph without dummy input.")
            return
        with self._lock:
            try:
                # Ensure model is on CPU for graph tracing if needed
                original_device = next(iter(model.parameters())).device
                model.cpu()
                # Convert input to CPU if it's a tensor or tuple of tensors
                if isinstance(input_to_model, torch.Tensor):
                    dummy_input_cpu = input_to_model.cpu()
                elif isinstance(input_to_model, tuple):
                    dummy_input_cpu = tuple(
                        i.cpu() if isinstance(i, torch.Tensor) else i
                        for i in input_to_model
                    )
                else:
                    dummy_input_cpu = input_to_model  # Assume compatible

                self.writer.add_graph(model, dummy_input_cpu, verbose=False)
                print("[TensorBoardStatsRecorder] Model graph logged.")
                model.to(original_device)  # Move model back
            except Exception as e:
                print(f"Error logging model graph: {e}.")
                traceback.print_exc()
                try:
                    model.to(original_device)  # Attempt to move back even on error
                except Exception:
                    pass

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        self.console_recorder.log_summary(global_step)

    def close(self, is_cleanup: bool = False):
        """Closes the TensorBoard writer and logs final hparams unless cleaning up."""
        print(f"[TensorBoardStatsRecorder] Close called (is_cleanup={is_cleanup})...")
        if not self.writer:
            print(
                "[TensorBoardStatsRecorder] Writer was not initialized or already closed."
            )
            self.console_recorder.close(is_cleanup=is_cleanup)
            return

        with self._lock:
            print("[TensorBoardStatsRecorder] Acquired lock for closing.")
            try:
                if not is_cleanup and self.hparam_logger:
                    print("[TensorBoardStatsRecorder] Logging final hparams...")
                    final_step = getattr(
                        self.aggregator.storage, "current_global_step", 0
                    )
                    final_summary = self.get_summary(final_step)
                    self.hparam_logger.log_final_hparams_from_summary(final_summary)
                    print("[TensorBoardStatsRecorder] Final hparams logged.")
                elif is_cleanup:
                    print(
                        "[TensorBoardStatsRecorder] Skipping final hparams logging due to cleanup."
                    )

                print("[TensorBoardStatsRecorder] Flushing writer...")
                self.writer.flush()
                print("[TensorBoardStatsRecorder] Writer flushed.")
                print("[TensorBoardStatsRecorder] Closing writer...")
                self.writer.close()
                self.writer = None
                print("[TensorBoardStatsRecorder] Writer closed successfully.")
            except Exception as e:
                print(f"[TensorBoardStatsRecorder] Error during writer close: {e}")
                traceback.print_exc()
            finally:
                print(
                    "[TensorBoardStatsRecorder] Releasing lock after closing attempt."
                )

        self.console_recorder.close(is_cleanup=is_cleanup)
        print("[TensorBoardStatsRecorder] Close method finished.")


File: stats\__init__.py
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
import os
import torch
import traceback
import re
import time
from typing import Optional, Dict, Tuple
import pickle

from utils.running_mean_std import RunningMeanStd
from stats.aggregator import StatsAggregator  # Updated import path
from config.general import TOTAL_TRAINING_STEPS


def find_latest_run_and_checkpoint(
    base_checkpoint_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Finds the latest run directory and the latest checkpoint within it.
    Returns (run_id, checkpoint_path) or (None, None).
    """
    latest_run_id = None
    latest_run_mtime = 0

    if not os.path.isdir(base_checkpoint_dir):
        print(
            f"[CheckpointFinder] Base checkpoint directory not found: {base_checkpoint_dir}"
        )
        return None, None

    try:
        for item in os.listdir(base_checkpoint_dir):
            item_path = os.path.join(base_checkpoint_dir, item)
            if os.path.isdir(item_path) and item.startswith("run_"):
                try:
                    mtime = os.path.getmtime(item_path)
                    if mtime > latest_run_mtime:
                        latest_run_mtime = mtime
                        latest_run_id = item
                except OSError:
                    continue
    except OSError as e:
        print(
            f"[CheckpointFinder] Error listing base checkpoint directory {base_checkpoint_dir}: {e}"
        )
        return None, None

    if latest_run_id is None:
        print(f"[CheckpointFinder] No run directories found in {base_checkpoint_dir}.")
        return None, None

    latest_run_dir = os.path.join(base_checkpoint_dir, latest_run_id)
    print(f"[CheckpointFinder] Identified latest run directory: {latest_run_dir}")

    latest_checkpoint_path = find_latest_checkpoint_in_dir(latest_run_dir)

    if latest_checkpoint_path:
        print(
            f"[CheckpointFinder] Found latest checkpoint in run '{latest_run_id}': {os.path.basename(latest_checkpoint_path)}"
        )
        return latest_run_id, latest_checkpoint_path
    else:
        print(
            f"[CheckpointFinder] No valid checkpoints found in the latest run directory: {latest_run_dir}"
        )
        return latest_run_id, None


def find_latest_checkpoint_in_dir(checkpoint_dir: str) -> Optional[str]:
    """
    Finds the latest checkpoint file (step_*.pth or FINAL_*.pth) in a specific directory.
    Prioritizes FINAL checkpoint if it exists.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = []
    final_checkpoint = None
    step_pattern = re.compile(r"step_(\d+)_agent_state\.pth")

    try:
        for filename in os.listdir(checkpoint_dir):
            full_path = os.path.join(checkpoint_dir, filename)
            if not os.path.isfile(full_path):
                continue

            if filename == "FINAL_agent_state.pth":
                final_checkpoint = full_path
                # Don't return immediately, check if step checkpoints are newer
                # return final_checkpoint # Removed immediate return
            else:
                match = step_pattern.match(filename)
                if match:
                    step = int(match.group(1))
                    checkpoints.append((step, full_path))

    except OSError as e:
        print(f"[CheckpointFinder] Error listing directory {checkpoint_dir}: {e}")
        return None

    # Prioritize FINAL if it exists and no step checkpoints are newer
    if final_checkpoint:
        final_mtime = os.path.getmtime(final_checkpoint)
        newer_step_checkpoints = [
            cp for step, cp in checkpoints if os.path.getmtime(cp) > final_mtime
        ]
        if not newer_step_checkpoints:
            print(f"[CheckpointFinder] Using FINAL checkpoint: {final_checkpoint}")
            return final_checkpoint

    if not checkpoints:
        # If no step checkpoints and FINAL wasn't returned, return FINAL if it exists
        return final_checkpoint

    # Otherwise, return the latest step checkpoint
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


class CheckpointManager:
    """Handles loading and saving of agent states, observation normalization, and stats."""

    def __init__(
        self,
        # agent: PPOAgent,
        stats_aggregator: StatsAggregator,  # Use updated import
        base_checkpoint_dir: str,
        run_checkpoint_dir: str,
        load_checkpoint_path_config: Optional[str],
        device: torch.device,
        obs_rms_dict: Optional[Dict[str, RunningMeanStd]] = None,
    ):
        # self.agent = agent
        self.stats_aggregator = stats_aggregator
        self.base_checkpoint_dir = base_checkpoint_dir
        self.run_checkpoint_dir = run_checkpoint_dir
        self.device = device
        self.obs_rms_dict = obs_rms_dict if obs_rms_dict else {}

        self.global_step = 0
        self.episode_count = 0
        # Initialize target step from config, will be overwritten by load if successful
        self.training_target_step = TOTAL_TRAINING_STEPS

        self.run_id_to_load_from: Optional[str] = None
        self.checkpoint_path_to_load: Optional[str] = None

        if load_checkpoint_path_config:
            print(
                f"[CheckpointManager] Using explicit checkpoint path from config: {load_checkpoint_path_config}"
            )
            if os.path.isfile(load_checkpoint_path_config):
                self.checkpoint_path_to_load = load_checkpoint_path_config
                try:
                    parent_dir = os.path.dirname(load_checkpoint_path_config)
                    self.run_id_to_load_from = os.path.basename(parent_dir)
                    if not self.run_id_to_load_from.startswith("run_"):
                        self.run_id_to_load_from = None
                except Exception:
                    self.run_id_to_load_from = None
                if self.run_id_to_load_from:
                    print(
                        f"[CheckpointManager] Extracted run_id '{self.run_id_to_load_from}' from explicit path."
                    )
                else:
                    print(
                        "[CheckpointManager] Could not determine run_id from explicit path."
                    )
            else:
                print(
                    f"[CheckpointManager] WARNING: Explicit checkpoint path not found: {load_checkpoint_path_config}. Starting fresh."
                )
        else:
            print(
                f"[CheckpointManager] No explicit checkpoint path. Searching for latest run in: {self.base_checkpoint_dir}"
            )
            latest_run_id, latest_checkpoint_path = find_latest_run_and_checkpoint(
                self.base_checkpoint_dir
            )
            if latest_run_id and latest_checkpoint_path:
                print(
                    f"[CheckpointManager] Found latest run '{latest_run_id}' with checkpoint: {os.path.basename(latest_checkpoint_path)}"
                )
                self.run_id_to_load_from = latest_run_id
                self.checkpoint_path_to_load = latest_checkpoint_path
            elif latest_run_id:
                print(
                    f"[CheckpointManager] Found latest run directory '{latest_run_id}' but no valid checkpoint inside. Starting fresh."
                )
            else:
                print(
                    f"[CheckpointManager] No previous runs found in {self.base_checkpoint_dir}. Starting fresh."
                )

        # Ensure aggregator has the initial target step
        if self.stats_aggregator:
            self.stats_aggregator.storage.training_target_step = (
                self.training_target_step
            )

    def get_run_id_to_load_from(self) -> Optional[str]:
        return self.run_id_to_load_from

    def get_checkpoint_path_to_load(self) -> Optional[str]:
        return self.checkpoint_path_to_load

    def load_checkpoint(self):
        """Loads agent state, observation normalization, and stats aggregator state."""
        if not self.checkpoint_path_to_load:
            print(
                "[CheckpointManager] No checkpoint path specified for loading. Skipping load."
            )
            # Ensure initial target step is set in aggregator
            if self.stats_aggregator:
                self.stats_aggregator.storage.training_target_step = (
                    self.training_target_step
                )
                self.stats_aggregator.storage.total_episodes = (
                    self.episode_count
                )  # Sync episode count
            return

        if not os.path.isfile(self.checkpoint_path_to_load):
            print(
                f"[CheckpointManager] LOAD ERROR: Checkpoint file not found: {self.checkpoint_path_to_load}"
            )
            self._reset_all_states()  # This sets target step and updates aggregator
            return

        print(
            f"[CheckpointManager] Loading checkpoint from: {self.checkpoint_path_to_load}"
        )
        loaded_target_step = None
        agent_load_successful = False
        try:
            checkpoint = torch.load(
                self.checkpoint_path_to_load,
                map_location=self.device,
                weights_only=False,
            )

            if "agent_state_dict" in checkpoint:
                # Agent load_state_dict now handles internal errors more gracefully
                self.agent.load_state_dict(checkpoint["agent_state_dict"])
                # We assume if no critical error was raised, it's "successful" enough to proceed
                agent_load_successful = True
                print("  -> Agent state loading attempted.")
            else:
                print(
                    "  -> WARNING: 'agent_state_dict' key missing. Agent state NOT loaded."
                )

            self.global_step = checkpoint.get("global_step", 0)
            print(f"  -> Loaded Global Step: {self.global_step}")

            if "stats_aggregator_state_dict" in checkpoint and self.stats_aggregator:
                try:
                    self.stats_aggregator.load_state_dict(
                        checkpoint["stats_aggregator_state_dict"]
                    )
                    print("  -> Stats Aggregator state loaded successfully.")
                    self.episode_count = self.stats_aggregator.storage.total_episodes
                    # Get target step loaded by aggregator
                    loaded_target_step = getattr(
                        self.stats_aggregator.storage, "training_target_step", None
                    )
                    if loaded_target_step is not None:
                        print(
                            f"  -> Loaded Training Target Step from Stats: {loaded_target_step}"
                        )
                    else:
                        print(
                            "  -> WARNING: 'training_target_step' not found in loaded stats."
                        )
                    loaded_start_time = self.stats_aggregator.storage.start_time
                    print(
                        f"  -> Loaded Run Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(loaded_start_time))}"
                    )
                except Exception as stats_err:
                    print(
                        f"  -> ERROR loading Stats Aggregator state: {stats_err}. Stats reset."
                    )
                    self._reset_aggregator_state()  # Resets aggregator, including target step
                    self.episode_count = 0
            elif self.stats_aggregator:
                print(
                    "  -> WARNING: 'stats_aggregator_state_dict' not found. Stats Aggregator reset."
                )
                self._reset_aggregator_state()  # Resets aggregator, including target step
                self.episode_count = 0
            else:
                # Fallback if no aggregator (shouldn't happen in normal flow)
                self.episode_count = checkpoint.get("episode_count", 0)
                # Try loading target step directly from checkpoint if stats failed/missing
                loaded_target_step = checkpoint.get("training_target_step", None)
                if loaded_target_step is not None:
                    print(
                        f"  -> Loaded Training Target Step from Checkpoint (fallback): {loaded_target_step}"
                    )

            print(
                f"  -> Resuming from Step: {self.global_step}, Ep: {self.episode_count}"
            )

            if "obs_rms_state_dict" in checkpoint and self.obs_rms_dict:
                rms_state_dict = checkpoint["obs_rms_state_dict"]
                loaded_keys = set()
                for key, rms_instance in self.obs_rms_dict.items():
                    if key in rms_state_dict:
                        try:
                            rms_data = rms_state_dict[key]
                            # Ensure data is numpy before loading into RMS
                            if isinstance(rms_data.get("mean"), torch.Tensor):
                                rms_data["mean"] = rms_data["mean"].cpu().numpy()
                            if isinstance(rms_data.get("var"), torch.Tensor):
                                rms_data["var"] = rms_data["var"].cpu().numpy()
                            rms_instance.load_state_dict(rms_data)
                            loaded_keys.add(key)
                            print(f"  -> Loaded Obs RMS for '{key}'")
                        except Exception as rms_load_err:
                            print(
                                f"  -> ERROR loading Obs RMS for '{key}': {rms_load_err}. RMS for this key reset."
                            )
                            rms_instance.reset()
                    else:
                        print(
                            f"  -> WARNING: Obs RMS state for '{key}' not found in checkpoint. Using initial RMS."
                        )
                        rms_instance.reset()
                extra_keys = set(rms_state_dict.keys()) - loaded_keys
                if extra_keys:
                    print(
                        f"  -> WARNING: Checkpoint contains unused Obs RMS keys: {extra_keys}"
                    )
            elif self.obs_rms_dict:
                print(
                    "  -> WARNING: Obs RMS state dict ('obs_rms_state_dict') not found. Using initial RMS for all keys."
                )
                for rms in self.obs_rms_dict.values():
                    rms.reset()

            # Determine final training target step
            if loaded_target_step is not None and loaded_target_step > self.global_step:
                self.training_target_step = loaded_target_step
                print(
                    f"[CheckpointManager] Using loaded Training Target Step: {self.training_target_step}"
                )
            else:
                # Calculate new target based on current global step + config steps
                self.training_target_step = self.global_step + TOTAL_TRAINING_STEPS
                if loaded_target_step is not None:
                    print(
                        f"[CheckpointManager] WARNING: Loaded target step ({loaded_target_step}) is not valid or already reached. Calculating new target."
                    )
                print(
                    f"[CheckpointManager] Calculated new Training Target Step: {self.training_target_step} (Current Step {self.global_step} + Config Steps {TOTAL_TRAINING_STEPS})"
                )

            # Ensure aggregator has the final target step
            if self.stats_aggregator:
                self.stats_aggregator.storage.training_target_step = (
                    self.training_target_step
                )

            print("[CheckpointManager] Checkpoint loading finished.")

        except pickle.UnpicklingError as e:
            print(f"  -> ERROR loading checkpoint (UnpicklingError): {e}. State reset.")
            traceback.print_exc()
            self._reset_all_states()
        except KeyError as e:
            print(f"  -> ERROR loading checkpoint: Missing key '{e}'. State reset.")
            traceback.print_exc()
            self._reset_all_states()
        except Exception as e:
            print(f"  -> ERROR loading checkpoint ('{e}'). State reset.")
            traceback.print_exc()
            self._reset_all_states()

        # Final check: if agent load failed, ensure target step is calculated fresh
        if not agent_load_successful:
            print(
                "[CheckpointManager] Agent load was unsuccessful. Recalculating target step."
            )
            self.training_target_step = self.global_step + TOTAL_TRAINING_STEPS
            if self.stats_aggregator:
                self.stats_aggregator.storage.training_target_step = (
                    self.training_target_step
                )
            print(
                f"[CheckpointManager] Final Training Target Step set to: {self.training_target_step}"
            )

    def _reset_aggregator_state(self):
        """Helper to reset only the stats aggregator state."""
        if self.stats_aggregator:
            avg_windows = self.stats_aggregator.avg_windows
            plot_window = self.stats_aggregator.plot_window
            self.stats_aggregator.__init__(
                avg_windows=avg_windows, plot_window=plot_window
            )
            # Ensure target step is reset in the new aggregator instance
            self.stats_aggregator.storage.training_target_step = (
                self.training_target_step
            )
            self.stats_aggregator.storage.total_episodes = 0  # Reset episode count too

    def _reset_all_states(self):
        """Helper to reset all managed states on critical load failure."""
        print("[CheckpointManager] Resetting all managed states due to load failure.")
        self.global_step = 0
        self.episode_count = 0
        # Set target step based on config *before* resetting aggregator
        self.training_target_step = TOTAL_TRAINING_STEPS
        if self.obs_rms_dict:
            for rms in self.obs_rms_dict.values():
                rms.reset()
        self._reset_aggregator_state()  # Resets aggregator and sets its target step

    def save_checkpoint(
        self,
        global_step: int,
        episode_count: int,  # This episode count might be slightly behind aggregator's if called mid-rollout
        training_target_step: int,  # Pass the target step known by the caller (e.g., main thread)
        is_final: bool = False,
    ):
        """Saves agent, observation normalization, and stats aggregator state."""
        prefix = "FINAL" if is_final else f"step_{global_step}"
        save_dir = self.run_checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{prefix}_agent_state.pth"
        full_save_path = os.path.join(save_dir, filename)

        print(f"[CheckpointManager] Saving checkpoint ({prefix}) to {save_dir}...")
        temp_save_path = full_save_path + ".tmp"
        try:
            agent_save_data = self.agent.get_state_dict()

            obs_rms_save_data = {}
            if self.obs_rms_dict:
                for key, rms_instance in self.obs_rms_dict.items():
                    rms_state = rms_instance.state_dict()
                    # Ensure data is numpy before saving
                    if isinstance(rms_state.get("mean"), torch.Tensor):
                        rms_state["mean"] = rms_state["mean"].cpu().numpy()
                    if isinstance(rms_state.get("var"), torch.Tensor):
                        rms_state["var"] = rms_state["var"].cpu().numpy()
                    obs_rms_save_data[key] = rms_state

            stats_aggregator_save_data = {}
            aggregator_episode_count = episode_count  # Use passed value as default
            aggregator_target_step = training_target_step  # Use passed value as default
            if self.stats_aggregator:
                # Ensure the target step is up-to-date before saving stats
                self.stats_aggregator.storage.training_target_step = (
                    training_target_step
                )
                stats_aggregator_save_data = self.stats_aggregator.state_dict()
                # Use episode count and target step from aggregator storage for consistency
                aggregator_episode_count = self.stats_aggregator.storage.total_episodes
                aggregator_target_step = (
                    self.stats_aggregator.storage.training_target_step
                )

            checkpoint_data = {
                "global_step": global_step,
                "episode_count": aggregator_episode_count,  # Save aggregator's count
                "training_target_step": aggregator_target_step,  # Save aggregator's target
                "agent_state_dict": agent_save_data,
                "obs_rms_state_dict": obs_rms_save_data,
                "stats_aggregator_state_dict": stats_aggregator_save_data,
            }

            torch.save(checkpoint_data, temp_save_path)
            os.replace(temp_save_path, full_save_path)
            print(f"  -> Checkpoint saved: {filename}")
        except Exception as e:
            print(f"  -> ERROR saving checkpoint: {e}")
            traceback.print_exc()
            if os.path.exists(temp_save_path):
                try:
                    os.remove(temp_save_path)
                except OSError:
                    pass

    def get_initial_state(self) -> Tuple[int, int]:
        """Returns the initial global step and episode count after potential loading."""
        return self.global_step, self.episode_count


File: training\training_utils.py
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
from .checkpoint_manager import CheckpointManager

__all__ = ["CheckpointManager"]


File: ui\demo_renderer.py
# File: ui/demo_renderer.py
import pygame
import math
import traceback
from typing import Tuple, Dict

from config import VisConfig, EnvConfig, DemoConfig, RED
from environment.game_state import GameState
from .panels.game_area import GameAreaRenderer  # Keep for grid rendering logic
from .demo_components.grid_renderer import DemoGridRenderer
from .demo_components.preview_renderer import DemoPreviewRenderer
from .demo_components.hud_renderer import DemoHudRenderer


class DemoRenderer:
    """
    Handles rendering specifically for the interactive Demo/Debug Mode.
    Delegates rendering tasks to sub-components.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,  # Pass GameAreaRenderer for shared logic/fonts
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer  # Keep reference

        # Initialize sub-renderers
        self.grid_renderer = DemoGridRenderer(
            screen, vis_config, demo_config, game_area_renderer
        )
        self.preview_renderer = DemoPreviewRenderer(
            screen, vis_config, demo_config, game_area_renderer
        )
        self.hud_renderer = DemoHudRenderer(
            screen, vis_config, demo_config, game_area_renderer
        )

        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

    def render(
        self, demo_env: GameState, env_config: EnvConfig, is_debug: bool = False
    ):
        """Renders the entire demo/debug mode screen."""
        if not demo_env:
            print("Error: DemoRenderer called with demo_env=None")
            return

        bg_color = self.hud_renderer.determine_background_color(demo_env)
        self.screen.fill(bg_color)

        screen_width, screen_height = self.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        game_rect, clipped_game_rect = self.grid_renderer.calculate_game_area_rect(
            screen_width, screen_height, padding, hud_height, help_height, env_config
        )

        if clipped_game_rect.width > 10 and clipped_game_rect.height > 10:
            self.grid_renderer.render_game_area(
                demo_env, env_config, clipped_game_rect, bg_color, is_debug
            )
        else:
            self.hud_renderer.render_too_small_message(
                "Demo Area Too Small", clipped_game_rect
            )

        if not is_debug:
            self.shape_preview_rects = self.preview_renderer.render_shape_previews_area(
                demo_env, screen_width, clipped_game_rect, padding
            )
        else:
            self.shape_preview_rects.clear()

        self.hud_renderer.render_hud(
            demo_env, screen_width, game_rect.bottom + 10, is_debug
        )
        self.hud_renderer.render_help_text(screen_width, screen_height, is_debug)

    # Expose calculation methods if needed by InputHandler
    def _calculate_game_area_rect(self, *args, **kwargs):
        return self.grid_renderer.calculate_game_area_rect(*args, **kwargs)

    def _calculate_demo_triangle_size(self, *args, **kwargs):
        return self.grid_renderer.calculate_demo_triangle_size(*args, **kwargs)

    def _calculate_grid_offset(self, *args, **kwargs):
        return self.grid_renderer.calculate_grid_offset(*args, **kwargs)

    def get_shape_preview_rects(self) -> Dict[int, pygame.Rect]:
        """Returns the dictionary of screen-relative shape preview rects."""
        # Get rects from the preview renderer
        return self.preview_renderer.get_shape_preview_rects()


File: ui\input_handler.py
# File: ui/input_handler.py
import pygame
from typing import Tuple, Callable, Dict, TYPE_CHECKING

# Type Aliases for Callbacks
HandleDemoMouseMotionCallback = Callable[[Tuple[int, int]], None]
HandleDemoMouseButtonDownCallback = Callable[[pygame.event.Event], None]
# Removed ToggleTrainingRunCallback
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]
StartDemoModeCallback = Callable[[], None]
ExitDemoModeCallback = Callable[[], None]
StartDebugModeCallback = Callable[[], None]
ExitDebugModeCallback = Callable[[], None]
HandleDebugInputCallback = Callable[[pygame.event.Event], None]

if TYPE_CHECKING:
    from .renderer import UIRenderer
    from app_state import AppState


class InputHandler:
    """Handles Pygame events and triggers callbacks based on application state."""

    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",
        # Removed toggle_training_run_cb
        request_cleanup_cb: RequestCleanupCallback,
        cancel_cleanup_cb: CancelCleanupCallback,
        confirm_cleanup_cb: ConfirmCleanupCallback,
        exit_app_cb: ExitAppCallback,
        start_demo_mode_cb: StartDemoModeCallback,
        exit_demo_mode_cb: ExitDemoModeCallback,
        handle_demo_mouse_motion_cb: HandleDemoMouseMotionCallback,
        handle_demo_mouse_button_down_cb: HandleDemoMouseButtonDownCallback,
        start_debug_mode_cb: StartDebugModeCallback,
        exit_debug_mode_cb: ExitDebugModeCallback,
        handle_debug_input_cb: HandleDebugInputCallback,
    ):
        self.screen = screen
        self.renderer = renderer
        # Removed self.toggle_training_run_cb
        self.request_cleanup_cb = request_cleanup_cb
        self.cancel_cleanup_cb = cancel_cleanup_cb
        self.confirm_cleanup_cb = confirm_cleanup_cb
        self.exit_app_cb = exit_app_cb
        self.start_demo_mode_cb = start_demo_mode_cb
        self.exit_demo_mode_cb = exit_demo_mode_cb
        self.handle_demo_mouse_motion_cb = handle_demo_mouse_motion_cb
        self.handle_demo_mouse_button_down_cb = handle_demo_mouse_button_down_cb
        self.start_debug_mode_cb = start_debug_mode_cb
        self.exit_debug_mode_cb = exit_debug_mode_cb
        self.handle_debug_input_cb = handle_debug_input_cb

        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

        # Button rects (Run button removed)
        # self.run_btn_rect = pygame.Rect(0, 0, 0, 0) # Removed
        self.cleanup_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.demo_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.debug_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_yes_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_no_rect = pygame.Rect(0, 0, 0, 0)
        self._update_button_rects()

    def _update_button_rects(self):
        """Calculates button rects based on initial layout assumptions."""
        button_height = 40
        button_y_pos = 10
        # Removed run_button_width
        cleanup_button_width = 160
        demo_button_width = 120
        debug_button_width = 120
        button_spacing = 10

        # Start directly with cleanup button
        current_x = button_spacing
        # self.run_btn_rect = pygame.Rect(button_spacing, button_y_pos, run_button_width, button_height) # Removed
        # current_x = self.run_btn_rect.right + button_spacing # Removed
        self.cleanup_btn_rect = pygame.Rect(
            current_x, button_y_pos, cleanup_button_width, button_height
        )
        current_x = self.cleanup_btn_rect.right + button_spacing
        self.demo_btn_rect = pygame.Rect(
            current_x, button_y_pos, demo_button_width, button_height
        )
        current_x = self.demo_btn_rect.right + button_spacing
        self.debug_btn_rect = pygame.Rect(
            current_x, button_y_pos, debug_button_width, button_height
        )

        sw, sh = self.screen.get_size()
        self.confirm_yes_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_no_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

    def handle_input(
        self, app_state_str: str, cleanup_confirmation_active: bool
    ) -> bool:
        """Processes Pygame events. Returns True to continue running, False to exit."""
        from app_state import AppState

        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            mouse_pos = (0, 0)

        sw, sh = self.screen.get_size()
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

        if (
            app_state_str == AppState.PLAYING.value
            and self.renderer
            and self.renderer.demo_renderer
        ):
            self.shape_preview_rects = (
                self.renderer.demo_renderer.get_shape_preview_rects()
            )
        else:
            self.shape_preview_rects.clear()

        # Removed hover check

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
                    self._update_button_rects()
                    if hasattr(self.renderer, "force_redraw"):
                        self.renderer.force_redraw()
                    print(f"Window resized: {new_w}x{new_h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")
                continue

            if cleanup_confirmation_active:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.cancel_cleanup_cb()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.confirm_yes_rect.collidepoint(mouse_pos):
                        self.confirm_cleanup_cb()
                    elif self.confirm_no_rect.collidepoint(mouse_pos):
                        self.cancel_cleanup_cb()
                continue

            elif app_state_str == AppState.PLAYING.value:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_demo_mode_cb()
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_demo_mouse_motion_cb(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_demo_mouse_button_down_cb(event)

            elif app_state_str == AppState.DEBUG.value:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_debug_mode_cb()
                    else:
                        self.handle_debug_input_cb(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_debug_input_cb(event)

            elif app_state_str == AppState.MAIN_MENU.value:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return self.exit_app_cb()
                    # Removed 'P' key binding for toggle run
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Removed run_btn_rect check
                    if self.cleanup_btn_rect.collidepoint(mouse_pos):
                        self.request_cleanup_cb()
                    elif self.demo_btn_rect.collidepoint(mouse_pos):
                        self.start_demo_mode_cb()
                    elif self.debug_btn_rect.collidepoint(mouse_pos):
                        self.start_debug_mode_cb()

            elif app_state_str == AppState.ERROR.value:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return self.exit_app_cb()

        return True

    def _update_ui_screen_references(self, new_screen: pygame.Surface):
        """Updates the screen reference in the renderer and its sub-components."""
        components_to_update = [
            self.renderer,
            getattr(self.renderer, "left_panel", None),
            getattr(self.renderer, "game_area", None),
            getattr(self.renderer, "overlays", None),
            # Removed tooltips
            getattr(self.renderer, "demo_renderer", None),
            getattr(
                getattr(self.renderer, "demo_renderer", None), "grid_renderer", None
            ),
            getattr(
                getattr(self.renderer, "demo_renderer", None), "preview_renderer", None
            ),
            getattr(
                getattr(self.renderer, "demo_renderer", None), "hud_renderer", None
            ),
        ]
        for component in components_to_update:
            if component and hasattr(component, "screen"):
                component.screen = new_screen


File: ui\overlays.py
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
from typing import Dict, Optional, Deque
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
        self.plot_update_interval: float = 0.2
        self.rolling_window_sizes = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW

        self.colors = {
            "rl_score": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[0]),
            "game_score": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[1]),
            "policy_loss": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[3]
            ),  # Keep for NN policy loss
            "value_loss": normalize_color_for_matplotlib(
                VisConfig.BLUE
            ),  # Keep for NN value loss
            # Removed entropy
            "len": normalize_color_for_matplotlib(VisConfig.BLUE),
            # Removed minibatch_sps
            "tris_cleared": normalize_color_for_matplotlib(VisConfig.YELLOW),
            # Removed lr (can add back if needed for NN)
            "cpu": normalize_color_for_matplotlib((255, 165, 0)),
            "memory": normalize_color_for_matplotlib((0, 191, 255)),
            "gpu_mem": normalize_color_for_matplotlib((123, 104, 238)),
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
        }

    def create_plot_surface(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:

        if target_width <= 10 or target_height <= 10 or not plot_data:
            return None

        # Updated data keys
        data_keys = [
            "game_scores",
            "episode_triangles_cleared",
            "episode_scores",  # Keep RL score? Or remove? Keep for now.
            "episode_lengths",
            "policy_losses",  # Added NN policy loss
            "value_losses",  # Kept NN value loss
            # Removed: minibatch_update_sps_values, lr_values, entropies
            "cpu_usage",
            "memory_usage",
            "gpu_memory_usage_percent",
            # Add placeholders for other potential plots
            "placeholder1",
            "placeholder2",
            "placeholder3",
        ]
        data_lists = {key: list(plot_data.get(key, deque())) for key in data_keys}

        has_any_data = any(len(d) > 0 for d in data_lists.values())
        if not has_any_data:
            return None

        fig = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                dpi = 90
                fig_width_in = max(1, target_width / dpi)
                fig_height_in = max(1, target_height / dpi)

                # Keep 4x3 layout for now, fill unused plots with placeholders
                fig, axes = plt.subplots(
                    4, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
                )
                fig.subplots_adjust(
                    hspace=0.18,
                    wspace=0.10,
                    left=0.08,
                    right=0.98,
                    bottom=0.05,
                    top=0.95,
                )
                axes_flat = axes.flatten()

                # Row 1: Performance Metrics
                render_single_plot(
                    axes_flat[0],
                    data_lists["game_scores"],
                    "Game Score",
                    self.colors["game_score"],
                    self.rolling_window_sizes,
                    placeholder_text="Game Score",
                )
                render_single_plot(
                    axes_flat[1],
                    data_lists["episode_triangles_cleared"],
                    "Tris Cleared / Ep",
                    self.colors["tris_cleared"],
                    self.rolling_window_sizes,
                    placeholder_text="Triangles Cleared",
                )
                render_single_plot(
                    axes_flat[2],
                    data_lists["episode_scores"],
                    "RL Score",
                    self.colors["rl_score"],
                    self.rolling_window_sizes,
                    placeholder_text="RL Score",
                )

                # Row 2: Training Dynamics / NN Losses
                render_single_plot(
                    axes_flat[3],
                    data_lists["episode_lengths"],
                    "Ep Length",
                    self.colors["len"],
                    self.rolling_window_sizes,
                    placeholder_text="Episode Length",
                )
                render_single_plot(
                    axes_flat[4],
                    data_lists["policy_losses"],
                    "Policy Loss",
                    self.colors["policy_loss"],
                    self.rolling_window_sizes,
                    placeholder_text="Policy Loss",
                )  # NN Policy Loss
                render_single_plot(
                    axes_flat[5],
                    data_lists["value_losses"],
                    "Value Loss",
                    self.colors["value_loss"],
                    self.rolling_window_sizes,
                    placeholder_text="Value Loss",
                )  # NN Value Loss

                # Row 3: Resource Usage
                render_single_plot(
                    axes_flat[6],
                    data_lists["cpu_usage"],
                    "CPU Usage (%)",
                    self.colors["cpu"],
                    self.rolling_window_sizes,
                    placeholder_text="CPU %",
                )
                render_single_plot(
                    axes_flat[7],
                    data_lists["memory_usage"],
                    "Memory Usage (%)",
                    self.colors["memory"],
                    self.rolling_window_sizes,
                    placeholder_text="Mem %",
                )
                render_single_plot(
                    axes_flat[8],
                    data_lists["gpu_memory_usage_percent"],
                    "GPU Memory (%)",
                    self.colors["gpu_mem"],
                    self.rolling_window_sizes,
                    placeholder_text="GPU Mem %",
                )

                # Row 4: Placeholders / Future Plots
                render_single_plot(
                    axes_flat[9],
                    data_lists["placeholder1"],
                    "Future Plot 1",
                    self.colors["placeholder"],
                    [],
                    placeholder_text="Future Plot 1",
                )
                render_single_plot(
                    axes_flat[10],
                    data_lists["placeholder2"],
                    "Future Plot 2",
                    self.colors["placeholder"],
                    [],
                    placeholder_text="Future Plot 2",
                )
                render_single_plot(
                    axes_flat[11],
                    data_lists["placeholder3"],
                    "Future Plot 3",
                    self.colors["placeholder"],
                    [],
                    placeholder_text="Future Plot 3",
                )

                # Remove x-axis labels/ticks for inner plots
                for i, ax in enumerate(axes_flat):
                    if i < 9:  # Adjust based on layout (4x3 -> 9 inner plots)
                        ax.set_xticklabels([])
                        ax.set_xlabel("")
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
                if new_plot_surface:
                    self.plot_surface = new_plot_surface
                self.last_plot_update_time = current_time
            elif not has_data:
                self.plot_surface = None

        return self.plot_surface


File: ui\plot_utils.py
# File: ui/plot_utils.py
import numpy as np
from typing import Optional, List, Union, Tuple
import matplotlib
import traceback
import math

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig


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
            "font.size": 9,  # Base font size slightly increased
            "axes.labelsize": 9,  # Increased label size
            "axes.titlesize": 11,  # Increased title size
            "xtick.labelsize": 8,  # Increased tick label size
            "ytick.labelsize": 8,  # Increased tick label size
            "legend.fontsize": 8,  # Increased legend font size
            "figure.facecolor": "#262626",
            "axes.facecolor": "#303030",
            "axes.edgecolor": "#707070",
            "axes.labelcolor": "#D0D0D0",
            "xtick.color": "#C0C0C0",
            "ytick.color": "#C0C0C0",
            "grid.color": "#505050",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "axes.titlepad": 6,  # Reduced title padding
            "legend.frameon": True,  # Add frame to legend
            "legend.framealpha": 0.85,  # Increased legend background alpha
            "legend.facecolor": "#202020",  # Darker legend background
            "legend.title_fontsize": 8,  # Increased legend title font size
        }
    )
except Exception as e:
    print(f"Warning: Failed to set Matplotlib style: {e}")


TREND_SLOPE_TOLERANCE = 1e-5
TREND_MIN_LINEWIDTH = 1
TREND_MAX_LINEWIDTH = 2
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

MIN_ALPHA = 0.4
MAX_ALPHA = 1.0
MIN_DATA_AVG_LINEWIDTH = 1
MAX_DATA_AVG_LINEWIDTH = 2

# Y_PADDING_FACTOR = 0.20 # Removed vertical padding factor


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
        return float(max_val)  # Ensure float
    inverted_rank = (total_ranks - 1) - rank
    fraction = inverted_rank / max(1, total_ranks - 1)
    # --- Explicitly cast to float before subtraction/addition ---
    f_min_val = float(min_val)
    f_max_val = float(max_val)
    value = f_min_val + (f_max_val - f_min_val) * fraction
    # --- End explicit cast ---
    # Clip using original min/max in case casting caused issues, ensure float return
    return float(np.clip(value, min_val, max_val))


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
    show_placeholder: bool = True,
    placeholder_text: Optional[str] = None,
    y_log_scale: bool = False,
):
    """
    Renders data with linearly scaled alpha/linewidth. Trend line is thin, white, dashed.
    Title is just the label. Detailed values moved to legend. Best value shown as legend title.
    Applies a background tint and border to the entire subplot based on trend desirability.
    Legend now includes current values and trend slope, placed at center-left.
    Handles empty data explicitly to show placeholder.
    Removed vertical padding, removed horizontal padding and xlabel.
    Increased title and legend font sizes, increased legend background alpha.
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
    best_val_str = f"Best: {_format_value(best_val, is_lower_better)}"

    # Set only the main title
    ax.set_title(
        label,
        loc="left",
        fontsize=plt.rcParams["axes.titlesize"],
        pad=plt.rcParams.get("axes.titlepad", 6),
    )

    try:
        x_coords = np.arange(n_points)
        plotted_legend_items = False
        min_y_overall = float("inf")
        max_y_overall = float("-inf")

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
        min_y_overall = min(min_y_overall, np.min(valid_data))
        max_y_overall = max(max_y_overall, np.max(valid_data))
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
                if len(rolling_avg) > 0:
                    min_y_overall = min(min_y_overall, np.min(rolling_avg))
                    max_y_overall = max(max_y_overall, np.max(rolling_avg))
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
            # Don't include trend line in min/max calculation for ylim
            plotted_legend_items = True

        ax.tick_params(axis="both", which="major")
        ax.grid(
            True,
            linestyle=plt.rcParams["grid.linestyle"],
            alpha=plt.rcParams["grid.alpha"],
        )

        # --- Adjust Y-axis limits WITHOUT padding ---
        if np.isfinite(min_y_overall) and np.isfinite(max_y_overall):
            if abs(max_y_overall - min_y_overall) < 1e-6:  # Handle constant data
                # Add a tiny epsilon to avoid zero range
                epsilon = max(abs(min_y_overall * 0.01), 1e-6)
                ax.set_ylim(min_y_overall - epsilon, max_y_overall + epsilon)
            else:
                ax.set_ylim(min_y_overall, max_y_overall)
        # else: Keep default limits if min/max calculation failed

        if y_log_scale and min_y_overall > 1e-9:
            ax.set_yscale("log")
            # Adjust log scale limits if needed, ensuring bottom is positive
            current_bottom, current_top = ax.get_ylim()
            new_bottom = max(current_bottom, 1e-9)  # Ensure bottom is positive
            if new_bottom >= current_top:  # Prevent invalid limits
                new_bottom = current_top / 10
            ax.set_ylim(bottom=new_bottom, top=current_top)
        else:
            ax.set_yscale("linear")

        # --- Adjust X-axis limits (remove padding) ---
        if n_points > 1:
            ax.set_xlim(0, n_points - 1)  # Set limits tightly
        elif n_points == 1:
            ax.set_xlim(-0.5, 0.5)  # Keep slight padding for single point

        # --- X-axis Ticks Formatting ---
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

        # --- Legend ---
        if plotted_legend_items:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(0, 0.5),
                title=best_val_str,
                fontsize=plt.rcParams["legend.fontsize"],
            )

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
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque

from config import VisConfig, EnvConfig, DemoConfig
from environment.game_state import GameState
from .panels import LeftPanelRenderer, GameAreaRenderer
from .overlays import OverlayRenderer
from .plotter import Plotter
from .demo_renderer import DemoRenderer
from .input_handler import InputHandler
from app_state import AppState


class UIRenderer:
    """Orchestrates rendering of all UI components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = Plotter()
        self.left_panel = LeftPanelRenderer(screen, vis_config, self.plotter)
        self.game_area = GameAreaRenderer(screen, vis_config)
        self.overlays = OverlayRenderer(screen, vis_config)
        self.demo_config = DemoConfig()
        self.demo_renderer = DemoRenderer(
            screen, vis_config, self.demo_config, self.game_area
        )
        self.last_plot_update_time = 0

    def set_input_handler(self, input_handler: InputHandler):
        """Sets the InputHandler reference after it's initialized."""
        self.left_panel.input_handler = input_handler
        if hasattr(self.left_panel, "button_status_renderer"):
            self.left_panel.button_status_renderer.input_handler_ref = input_handler

    def check_hover(self, mouse_pos: Tuple[int, int], app_state_str: str):
        """Placeholder for hover checks if needed later."""
        pass

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0

    def render_all(
        self,
        app_state: str,
        is_process_running: bool,  # Keep for potential MCTS/NN status
        status: str,
        stats_summary: Dict[str, Any],
        envs: List[GameState],  # Keep for visualization
        num_envs: int,  # Keep for visualization layout
        env_config: EnvConfig,
        cleanup_confirmation_active: bool,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        demo_env: Optional[GameState] = None,
        update_progress_details: Dict[str, Any] = {},  # Keep for potential NN progress
        agent_param_count: int = 0,  # Keep for NN param count
        worker_counts: Dict[
            str, int
        ] = {},  # Keep structure for potential future workers
    ):
        """Renders UI based on the application state."""
        try:
            current_app_state = (
                AppState(app_state)
                if app_state in AppState._value2member_map_
                else AppState.UNKNOWN
            )

            if current_app_state == AppState.MAIN_MENU:
                self._render_main_menu(
                    is_process_running=is_process_running,  # Pass to left panel
                    status=status,
                    stats_summary=stats_summary,
                    envs=envs,
                    num_envs=num_envs,
                    env_config=env_config,
                    cleanup_message=cleanup_message,
                    last_cleanup_message_time=last_cleanup_message_time,
                    tensorboard_log_dir=tensorboard_log_dir,
                    plot_data=plot_data,
                    update_progress_details=update_progress_details,  # Pass to left panel
                    app_state=app_state,
                    agent_param_count=agent_param_count,
                    worker_counts=worker_counts,  # Pass to left panel
                )
            elif current_app_state == AppState.PLAYING:
                if demo_env:
                    self.demo_renderer.render(demo_env, env_config, is_debug=False)
                else:
                    self._render_simple_message("Demo Env Error!", VisConfig.RED)
            elif current_app_state == AppState.DEBUG:
                if demo_env:
                    self.demo_renderer.render(demo_env, env_config, is_debug=True)
                else:
                    self._render_simple_message("Debug Env Error!", VisConfig.RED)
            elif current_app_state == AppState.INITIALIZING:
                self._render_initializing_screen(status)
            elif current_app_state == AppState.ERROR:
                self._render_error_screen(status)

            if cleanup_confirmation_active and current_app_state != AppState.ERROR:
                self.overlays.render_cleanup_confirmation()
            elif not cleanup_confirmation_active:
                self.overlays.render_status_message(
                    cleanup_message, last_cleanup_message_time
                )

            pygame.display.flip()

        except pygame.error as e:
            print(f"Pygame rendering error in render_all: {e}")
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
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        update_progress_details: Dict[str, Any],
        app_state: str,
        agent_param_count: int,
        worker_counts: Dict[str, int],
    ):
        """Renders the main dashboard view."""
        self.screen.fill(VisConfig.BLACK)
        current_width, current_height = self.screen.get_size()
        left_panel_ratio = max(0.1, min(0.9, self.vis_config.LEFT_PANEL_RATIO))
        lp_width = int(current_width * left_panel_ratio)
        ga_width = current_width - lp_width
        min_lp_width = 300
        if lp_width < min_lp_width and current_width > min_lp_width:
            lp_width = min_lp_width
            ga_width = max(0, current_width - lp_width)
        elif current_width <= min_lp_width:
            lp_width = current_width
            ga_width = 0

        self.left_panel.render(
            panel_width=lp_width,
            is_process_running=is_process_running,  # Pass for potential future use
            status=status,
            stats_summary=stats_summary,
            tensorboard_log_dir=tensorboard_log_dir,
            plot_data=plot_data,
            app_state=app_state,
            update_progress_details=update_progress_details,  # Pass for potential NN progress
            agent_param_count=agent_param_count,
            worker_counts=worker_counts,  # Pass for potential future use
        )
        # Render game area only if width is sufficient
        if ga_width > 0:
            self.game_area.render(
                envs=envs,  # Pass envs for visualization
                num_envs=num_envs,  # Pass num_envs for layout
                env_config=env_config,
                panel_width=ga_width,
                panel_x_offset=lp_width,
            )

    def _render_initializing_screen(self, status_message: str = "Initializing..."):
        self._render_simple_message(status_message, VisConfig.WHITE)

    def _render_error_screen(self, status_message: str):
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
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)
            text_surf = font.render(message, True, color)
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
        except Exception as e:
            print(f"Error rendering simple message '{message}': {e}")


File: ui\tooltips.py
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


File: ui\demo_components\grid_renderer.py
# File: ui/demo_components/grid_renderer.py
import pygame
import math
import traceback
from typing import Tuple

from config import VisConfig, EnvConfig, DemoConfig, RED, BLUE
from config.constants import LINE_CLEAR_FLASH_COLOR, GAME_OVER_FLASH_COLOR
from environment.game_state import GameState
from environment.triangle import Triangle
from ui.panels.game_area import GameAreaRenderer  # Import for base rendering


class DemoGridRenderer:
    """Renders the main game grid area for Demo/Debug mode."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,  # Use fonts/methods from GameAreaRenderer
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer  # Store reference
        self.overlay_font = self.game_area_renderer.fonts.get(
            "env_overlay", pygame.font.Font(None, 36)
        )
        self.invalid_placement_color = (0, 0, 0, 150)

    def calculate_game_area_rect(
        self,
        screen_width: int,
        screen_height: int,
        padding: int,
        hud_height: int,
        help_height: int,
        env_config: EnvConfig,
    ) -> Tuple[pygame.Rect, pygame.Rect]:
        """Calculates the main game area rectangle, maintaining aspect ratio."""
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

    def render_game_area(
        self,
        demo_env: GameState,
        env_config: EnvConfig,
        clipped_game_rect: pygame.Rect,
        bg_color: Tuple[int, int, int],
        is_debug: bool,
    ):
        """Renders the central game grid and placement preview."""
        try:
            game_surf = self.screen.subsurface(clipped_game_rect)
            game_surf.fill(bg_color)

            # Use the existing grid rendering logic from GameAreaRenderer
            self.game_area_renderer._render_single_env_grid(
                game_surf, demo_env, env_config
            )

            if not is_debug:
                tri_cell_w, tri_cell_h = self.calculate_demo_triangle_size(
                    clipped_game_rect.width, clipped_game_rect.height, env_config
                )
                if tri_cell_w > 0 and tri_cell_h > 0:
                    grid_ox, grid_oy = self.calculate_grid_offset(
                        clipped_game_rect.width, clipped_game_rect.height, env_config
                    )
                    self._render_dragged_shape(
                        game_surf,
                        demo_env,
                        tri_cell_w,
                        tri_cell_h,
                        grid_ox,
                        grid_oy,
                        clipped_game_rect.topleft,
                    )

            # Render overlays (delegated to HUD renderer now)
            # if demo_env.is_over():
            #     self._render_demo_overlay_text(game_surf, "GAME OVER", RED)
            # elif demo_env.is_line_clearing() and demo_env.last_line_clear_info:
            #     lines, tris, score = demo_env.last_line_clear_info
            #     line_str = "Line" if lines == 1 else "Lines"
            #     clear_msg = f"{lines} {line_str} Cleared! ({tris} Tris, +{score:.2f} pts)"
            #     self._render_demo_overlay_text(game_surf, clear_msg, BLUE)

        except ValueError as e:
            print(f"Error subsurface demo game ({clipped_game_rect}): {e}")
            pygame.draw.rect(self.screen, RED, clipped_game_rect, 1)
        except Exception as render_e:
            print(f"Error rendering demo game area: {render_e}")
            traceback.print_exc()
            pygame.draw.rect(self.screen, RED, clipped_game_rect, 1)

    def calculate_demo_triangle_size(
        self, surf_w: int, surf_h: int, env_config: EnvConfig
    ) -> Tuple[int, int]:
        """Calculates the size of triangles for rendering within the demo area."""
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

    def calculate_grid_offset(
        self, surf_w: int, surf_h: int, env_config: EnvConfig
    ) -> Tuple[float, float]:
        """Calculates the top-left offset for centering the grid rendering."""
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

    def _render_dragged_shape(
        self,
        surf: pygame.Surface,
        env: GameState,
        cell_w: int,
        cell_h: int,
        grid_offset_x: float,
        grid_offset_y: float,
        game_area_offset: Tuple[int, int],
    ):
        """Renders the shape being dragged, either snapped or following the mouse."""
        if cell_w <= 0 or cell_h <= 0:
            return
        dragged_shape, snapped_pos = env.get_dragged_shape_info()
        if dragged_shape is None:
            return

        is_valid_placement = snapped_pos is not None
        preview_alpha = 150
        if is_valid_placement:
            shape_rgb = dragged_shape.color
            preview_color_rgba = (
                shape_rgb[0],
                shape_rgb[1],
                shape_rgb[2],
                preview_alpha,
            )
        else:
            preview_color_rgba = (50, 50, 50, 100)

        temp_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))

        ref_x, ref_y = 0, 0
        if snapped_pos:
            snap_r, snap_c = snapped_pos
            ref_x = grid_offset_x + snap_c * (cell_w * 0.75)
            ref_y = grid_offset_y + snap_r * cell_h
        else:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            mouse_x -= game_area_offset[0]
            mouse_y -= game_area_offset[1]
            min_r, min_c, max_r, max_c = dragged_shape.bbox()
            shape_h_cells = max_r - min_r + 1
            shape_w_cells_eff = (max_c - min_c + 1) * 0.75 + 0.25
            shape_pixel_w = shape_w_cells_eff * cell_w
            shape_pixel_h = shape_h_cells * cell_h
            ref_x = mouse_x - (shape_pixel_w / 2) - (min_c * cell_w * 0.75)
            ref_y = mouse_y - (shape_pixel_h / 2) - (min_r * cell_h)

        for dr, dc, is_up in dragged_shape.triangles:
            tri_x = ref_x + dc * (cell_w * 0.75)
            tri_y = ref_y + dr * cell_h
            temp_tri = Triangle(0, 0, is_up)
            try:
                points = temp_tri.get_points(ox=tri_x, oy=tri_y, cw=cell_w, ch=cell_h)
                pygame.draw.polygon(temp_surface, preview_color_rgba, points)
            except Exception:
                pass

        surf.blit(temp_surface, (0, 0))


File: ui\demo_components\hud_renderer.py
# File: ui/demo_components/hud_renderer.py
import pygame
from typing import Tuple

from config import VisConfig, DemoConfig, RED, BLUE, WHITE, LIGHTG, DARK_RED
from config.constants import LINE_CLEAR_FLASH_COLOR, GAME_OVER_FLASH_COLOR
from environment.game_state import GameState
from ui.panels.game_area import GameAreaRenderer  # Import for fonts


class DemoHudRenderer:
    """Renders the HUD, help text, and overlays for Demo/Debug mode."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,  # Use fonts from GameAreaRenderer
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer  # Store reference
        self._init_demo_fonts()  # Initialize specific demo fonts

    def _init_demo_fonts(self):
        """Initializes fonts used specifically in demo mode HUD/Help."""
        try:
            self.demo_hud_font = pygame.font.SysFont(
                None, self.demo_config.HUD_FONT_SIZE
            )
            self.demo_help_font = pygame.font.SysFont(
                None, self.demo_config.HELP_FONT_SIZE
            )
        except Exception as e:
            print(f"Warning: SysFont error for demo fonts: {e}. Using default.")
            self.demo_hud_font = pygame.font.Font(None, self.demo_config.HUD_FONT_SIZE)
            self.demo_help_font = pygame.font.Font(
                None, self.demo_config.HELP_FONT_SIZE
            )

    def determine_background_color(self, demo_env: GameState) -> Tuple[int, int, int]:
        """Determines the background color based on game state."""
        if demo_env.is_line_clearing():
            return LINE_CLEAR_FLASH_COLOR
        if demo_env.is_game_over_flashing():
            return GAME_OVER_FLASH_COLOR
        if demo_env.is_over():
            return DARK_RED
        if demo_env.is_frozen():
            return (30, 30, 100)
        return self.demo_config.BACKGROUND_COLOR

    def render_hud(
        self, demo_env: GameState, screen_width: int, hud_y: int, is_debug: bool
    ):
        """Renders the score and triangles cleared HUD."""
        if is_debug:
            hud_text = (
                f"DEBUG MODE | Tris Cleared: {demo_env.triangles_cleared_this_episode}"
            )
        else:
            hud_text = f"Score: {demo_env.game_score} | Tris Cleared: {demo_env.triangles_cleared_this_episode}"
        try:
            hud_surf = self.demo_hud_font.render(hud_text, True, WHITE)
            hud_rect = hud_surf.get_rect(midtop=(screen_width // 2, hud_y))
            self.screen.blit(hud_surf, hud_rect)
        except Exception as e:
            print(f"HUD render error: {e}")

    def render_help_text(self, screen_width: int, screen_height: int, is_debug: bool):
        """Renders the control help text at the bottom."""
        if is_debug:
            help_text = self.demo_config.DEBUG_HELP_TEXT
        else:
            help_text = (
                "[Click Preview]=Select/Deselect | [Click Grid]=Place | [ESC]=Exit"
            )
        try:
            help_surf = self.demo_help_font.render(help_text, True, LIGHTG)
            help_rect = help_surf.get_rect(
                centerx=screen_width // 2, bottom=screen_height - 10
            )
            self.screen.blit(help_surf, help_rect)
        except Exception as e:
            print(f"Help render error: {e}")

    def render_demo_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders centered overlay text (e.g., GAME OVER)."""
        # This method might be better placed here if it's specific to demo overlays
        # Or keep it in the grid renderer if it's used there too.
        # For now, assuming it's called by the grid renderer.
        # If called directly by DemoRenderer, it needs access to the game_surf.
        pass  # Logic moved to GridRenderer for now, can be moved back if needed.

    def render_too_small_message(self, text: str, area_rect: pygame.Rect):
        """Renders a message indicating the area is too small."""
        try:
            font = self.game_area_renderer.fonts.get("ui", pygame.font.Font(None, 24))
            err_surf = font.render(
                text, True, LIGHTG
            )  # Use LIGHTG for less alarming message
            target_rect = err_surf.get_rect(center=area_rect.center)
            self.screen.blit(err_surf, target_rect)
        except Exception as e:
            print(f"Error rendering 'too small' message: {e}")


File: ui\demo_components\preview_renderer.py
# File: ui/demo_components/preview_renderer.py
import pygame
from typing import Tuple, Dict

from config import VisConfig, DemoConfig, RED, BLUE, GRAY
from environment.game_state import GameState
from ui.panels.game_area import GameAreaRenderer  # Import for base rendering


class DemoPreviewRenderer:
    """Renders the shape preview area in Demo mode."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,  # Use fonts/methods from GameAreaRenderer
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer  # Store reference
        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

    def render_shape_previews_area(
        self,
        demo_env: GameState,
        screen_width: int,
        clipped_game_rect: pygame.Rect,
        padding: int,
    ) -> Dict[int, pygame.Rect]:
        """Renders the shape preview area. Returns dict of preview rects."""
        self.shape_preview_rects.clear()  # Clear previous rects
        preview_area_w = min(150, screen_width - clipped_game_rect.right - padding // 2)
        if preview_area_w <= 20:
            return self.shape_preview_rects

        preview_area_rect = pygame.Rect(
            clipped_game_rect.right + padding // 2,
            clipped_game_rect.top,
            preview_area_w,
            clipped_game_rect.height,
        )
        clipped_preview_area_rect = preview_area_rect.clip(self.screen.get_rect())
        if (
            clipped_preview_area_rect.width <= 0
            or clipped_preview_area_rect.height <= 0
        ):
            return self.shape_preview_rects

        try:
            preview_area_surf = self.screen.subsurface(clipped_preview_area_rect)
            self.shape_preview_rects = self._render_demo_shape_previews(
                preview_area_surf, demo_env, preview_area_rect.topleft
            )
        except ValueError as e:
            print(f"Error subsurface demo shape preview area: {e}")
            pygame.draw.rect(self.screen, RED, clipped_preview_area_rect, 1)
        except Exception as e:
            print(f"Error rendering demo shape previews: {e}")
            traceback.print_exc()
        return self.shape_preview_rects

    def _render_demo_shape_previews(
        self, surf: pygame.Surface, env: GameState, area_topleft: Tuple[int, int]
    ) -> Dict[int, pygame.Rect]:
        """Renders the small previews of available shapes. Returns dict of screen rects."""
        calculated_rects: Dict[int, pygame.Rect] = {}
        surf.fill((25, 25, 25))
        all_slots = env.shapes
        selected_idx = env.demo_selected_shape_idx
        dragged_idx = env.demo_dragged_shape_idx

        num_slots = env.env_config.NUM_SHAPE_SLOTS
        surf_w, surf_h = surf.get_size()
        preview_padding = 5

        if num_slots <= 0:
            return calculated_rects

        preview_h = max(20, (surf_h - (num_slots + 1) * preview_padding) / num_slots)
        preview_w = max(20, surf_w - 2 * preview_padding)
        current_preview_y = preview_padding

        for i in range(num_slots):
            shape_in_slot = all_slots[i] if i < len(all_slots) else None
            preview_rect_local = pygame.Rect(
                preview_padding, current_preview_y, preview_w, preview_h
            )
            preview_rect_screen = preview_rect_local.move(area_topleft)
            calculated_rects[i] = preview_rect_screen

            clipped_preview_rect = preview_rect_local.clip(surf.get_rect())
            if clipped_preview_rect.width <= 0 or clipped_preview_rect.height <= 0:
                current_preview_y += preview_h + preview_padding
                continue

            bg_color = (40, 40, 40)
            border_color = GRAY
            border_width = 1
            if i == selected_idx and shape_in_slot is not None:
                border_color = self.demo_config.SELECTED_SHAPE_HIGHLIGHT_COLOR
                border_width = 3
            elif i == dragged_idx:
                border_color = (100, 100, 255)
                border_width = 2

            pygame.draw.rect(surf, bg_color, clipped_preview_rect, border_radius=3)
            pygame.draw.rect(
                surf, border_color, clipped_preview_rect, border_width, border_radius=3
            )

            if shape_in_slot is not None:
                self._render_single_shape_in_preview_box(
                    surf, shape_in_slot, preview_rect_local, clipped_preview_rect
                )

            current_preview_y += preview_h + preview_padding
        return calculated_rects

    def _render_single_shape_in_preview_box(
        self,
        surf: pygame.Surface,
        shape_obj,
        preview_rect: pygame.Rect,
        clipped_preview_rect: pygame.Rect,
    ):
        """Renders a single shape scaled to fit within its preview box."""
        try:
            inner_padding = 2
            shape_render_area_rect = pygame.Rect(
                inner_padding,
                inner_padding,
                clipped_preview_rect.width - 2 * inner_padding,
                clipped_preview_rect.height - 2 * inner_padding,
            )
            if shape_render_area_rect.width <= 0 or shape_render_area_rect.height <= 0:
                return

            temp_surf = pygame.Surface(shape_render_area_rect.size, pygame.SRCALPHA)
            temp_surf.fill((0, 0, 0, 0))

            min_r, min_c, max_r, max_c = shape_obj.bbox()
            shape_h_cells = max(1, max_r - min_r + 1)
            shape_w_cells_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)
            scale_h = shape_render_area_rect.height / shape_h_cells
            scale_w = shape_render_area_rect.width / shape_w_cells_eff
            cell_size = max(1, min(scale_h, scale_w))

            # Use the GameAreaRenderer's shape rendering logic
            self.game_area_renderer._render_single_shape(
                temp_surf, shape_obj, int(cell_size)
            )

            surf.blit(
                temp_surf, shape_render_area_rect.move(preview_rect.topleft).topleft
            )

        except ValueError as sub_err:
            print(f"Error subsurface shape preview: {sub_err}")
            pygame.draw.rect(surf, RED, clipped_preview_rect, 1)
        except Exception as e:
            print(f"Error rendering demo shape preview: {e}")
            pygame.draw.rect(surf, RED, clipped_preview_rect, 1)

    def get_shape_preview_rects(self) -> Dict[int, pygame.Rect]:
        """Returns the dictionary of screen-relative shape preview rects."""
        return self.shape_preview_rects.copy()


File: ui\panels\game_area.py
# File: ui/panels/game_area.py
import pygame
import math
import traceback
from typing import List, Tuple
from config import (
    VisConfig,
    EnvConfig,
    BLACK,
    BLUE,
    RED,
    GRAY,
    YELLOW,
    LIGHTG,
)  # Added GRAY, YELLOW, LIGHTG
from environment.game_state import GameState
from environment.shape import Shape
from environment.triangle import Triangle


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

    def render(
        self,
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        panel_width: int,
        panel_x_offset: int,
    ):
        current_height = self.screen.get_height()
        ga_rect = pygame.Rect(panel_x_offset, 0, panel_width, current_height)

        if num_envs <= 0 or ga_rect.width <= 0 or ga_rect.height <= 0:
            pygame.draw.rect(self.screen, (10, 10, 10), ga_rect)
            pygame.draw.rect(self.screen, (50, 50, 50), ga_rect, 1)
            return

        render_limit = self.vis_config.NUM_ENVS_TO_RENDER
        num_to_render = min(num_envs, render_limit) if render_limit > 0 else num_envs

        if num_to_render <= 0:
            pygame.draw.rect(self.screen, (10, 10, 10), ga_rect)
            pygame.draw.rect(self.screen, (50, 50, 50), ga_rect, 1)
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
        if ga_rect.width <= 0 or ga_rect.height <= 0:
            return 0, 0, 0, 0
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
        cell_w, cell_h = surf.get_width(), surf.get_height()
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

        grid_surf, shape_surf = None, None
        if grid_area_height > 0 and cell_w > 0:
            try:
                grid_surf = surf.subsurface(pygame.Rect(0, 0, cell_w, grid_area_height))
            except ValueError:
                pygame.draw.rect(
                    surf, VisConfig.RED, pygame.Rect(0, 0, cell_w, grid_area_height), 1
                )
        if shape_area_height > 0 and cell_w > 0:
            try:
                shape_rect = pygame.Rect(0, shape_area_y, cell_w, shape_area_height)
                shape_surf = surf.subsurface(shape_rect)
                shape_surf.fill((35, 35, 35))
            except ValueError:
                pygame.draw.rect(
                    surf,
                    VisConfig.RED,
                    pygame.Rect(0, shape_area_y, cell_w, shape_area_height),
                    1,
                )

        if grid_surf:
            self._render_single_env_grid(grid_surf, env, env_config)
        if shape_surf:
            self._render_shape_previews(shape_surf, env)

        try:
            score_text = f"GS: {env.game_score} R: {env.score:.1f}"
            score_surf = self.fonts["env_score"].render(
                score_text, True, VisConfig.WHITE, (0, 0, 0, 180)
            )
            surf.blit(score_surf, (2, 2))
        except Exception as e:
            print(f"Error rendering score: {e}")

        if env.is_over():
            self._render_overlay_text(surf, "GAME OVER", VisConfig.RED)
        elif env.is_line_clearing() and env.last_line_clear_info:
            lines, tris, score = env.last_line_clear_info
            line_str = "Line" if lines == 1 else "Lines"
            clear_msg = f"{lines} {line_str} Cleared! ({tris} Tris, +{score:.2f} pts)"
            self._render_overlay_text(surf, clear_msg, BLUE)

    def _render_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        try:
            overlay_font = self.fonts["env_overlay"]
            max_width = surf.get_width() * 0.9
            font_size = 36
            text_surf = overlay_font.render(text, True, VisConfig.WHITE)
            while text_surf.get_width() > max_width and font_size > 10:
                font_size -= 2
                overlay_font = pygame.font.SysFont(None, font_size)
                text_surf = overlay_font.render(text, True, VisConfig.WHITE)
            bg_color_rgba = (color[0] // 2, color[1] // 2, color[2] // 2, 220)
            text_surf_with_bg = overlay_font.render(
                text, True, VisConfig.WHITE, bg_color_rgba
            )
            text_rect = text_surf_with_bg.get_rect(center=surf.get_rect().center)
            surf.blit(text_surf_with_bg, text_rect)
        except Exception as e:
            print(f"Error rendering overlay text '{text}': {e}")

    def _render_single_env_grid(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        try:
            padding = self.vis_config.ENV_GRID_PADDING
            drawable_w, drawable_h = max(1, surf.get_width() - 2 * padding), max(
                1, surf.get_height() - 2 * padding
            )
            grid_rows, grid_cols_eff_width = (
                env_config.ROWS,
                env_config.COLS * 0.75 + 0.25,
            )
            if grid_rows <= 0 or grid_cols_eff_width <= 0:
                return

            scale_w, scale_h = drawable_w / grid_cols_eff_width, drawable_h / grid_rows
            final_scale = min(scale_w, scale_h)
            if final_scale <= 0:
                return

            final_grid_pixel_w, final_grid_pixel_h = (
                grid_cols_eff_width * final_scale,
                grid_rows * final_scale,
            )
            tri_cell_w, tri_cell_h = max(1, final_scale), max(1, final_scale)
            grid_ox, grid_oy = (
                padding + (drawable_w - final_grid_pixel_w) / 2,
                padding + (drawable_h - final_grid_pixel_h) / 2,
            )

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
                        if not t.is_death and hasattr(t, "get_points"):
                            try:
                                pts = t.get_points(
                                    ox=grid_ox,
                                    oy=grid_oy,
                                    cw=int(tri_cell_w),
                                    ch=int(tri_cell_h),
                                )
                                color = VisConfig.LIGHTG
                                if is_highlighting and (r, c) in cleared_coords:
                                    color = highlight_color
                                elif t.is_occupied:
                                    color = t.color if t.color else VisConfig.RED
                                pygame.draw.polygon(surf, color, pts)
                                pygame.draw.polygon(surf, VisConfig.GRAY, pts, 1)
                            except Exception:
                                pass
            else:
                pygame.draw.rect(surf, VisConfig.RED, surf.get_rect(), 2)
                err_txt = self.fonts["ui"].render(
                    "Invalid Grid Data", True, VisConfig.RED
                )
                surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))
        except Exception as e:
            pygame.draw.rect(surf, VisConfig.RED, surf.get_rect(), 2)

    def _render_shape_previews(self, surf: pygame.Surface, env: GameState):
        available_shapes = env.get_shapes()
        if not available_shapes:
            return
        surf_w, surf_h = surf.get_width(), surf.get_height()
        if surf_w <= 0 or surf_h <= 0:
            return

        num_shapes = len(available_shapes)
        padding = 4
        total_padding = (num_shapes + 1) * padding
        available_width = surf_w - total_padding
        if available_width <= 0:
            return

        width_per_shape = available_width / num_shapes
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
            if shape is None:
                pygame.draw.rect(surf, (50, 50, 50), preview_rect, 1, border_radius=2)
                current_x += preview_dim + padding
                continue
            try:
                temp_shape_surf = pygame.Surface(
                    (preview_dim, preview_dim), pygame.SRCALPHA
                )
                temp_shape_surf.fill((0, 0, 0, 0))
                min_r, min_c, max_r, max_c = shape.bbox()
                shape_h, shape_w_eff = max(1, max_r - min_r + 1), max(
                    1, (max_c - min_c + 1) * 0.75 + 0.25
                )
                scale_h, scale_w = preview_dim / shape_h, preview_dim / shape_w_eff
                cell_size = max(1, min(scale_h, scale_w))
                self._render_single_shape(temp_shape_surf, shape, int(cell_size))
                surf.blit(temp_shape_surf, preview_rect.topleft)
                current_x += preview_dim + padding
            except Exception as e:
                pygame.draw.rect(surf, VisConfig.RED, preview_rect, 1)
                current_x += preview_dim + padding

    def _render_single_shape(self, surf: pygame.Surface, shape: Shape, cell_size: int):
        if not shape or not shape.triangles or cell_size <= 0:
            return
        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h, shape_w_eff = max(1, max_r - min_r + 1), max(
            1, (max_c - min_c + 1) * 0.75 + 0.25
        )
        if shape_w_eff <= 0 or shape_h <= 0:
            return

        total_w, total_h = shape_w_eff * cell_size, shape_h * cell_size
        offset_x = (surf.get_width() - total_w) / 2 - min_c * (cell_size * 0.75)
        offset_y = (surf.get_height() - total_h) / 2 - min_r * cell_size

        for dr, dc, up in shape.triangles:
            tri = Triangle(row=dr, col=dc, is_up=up)
            try:
                pts = tri.get_points(
                    ox=offset_x, oy=offset_y, cw=cell_size, ch=cell_size
                )
                pygame.draw.polygon(surf, shape.color, pts)
            except Exception:
                pass

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
from typing import Dict, Any, Optional, Deque

from config import (
    VisConfig,
    RNNConfig,
    TransformerConfig,
    ModelConfig,
    # Removed TOTAL_TRAINING_STEPS
)
from config.general import DEVICE

from ui.plotter import Plotter
from ui.input_handler import InputHandler
from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    TBStatusRenderer,
    PlotAreaRenderer,
)
from app_state import AppState


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.input_handler: Optional[InputHandler] = None

        self.button_status_renderer = ButtonStatusRenderer(self.screen, self.fonts)
        self.info_text_renderer = InfoTextRenderer(self.screen, self.fonts)
        self.tb_status_renderer = TBStatusRenderer(self.screen, self.fonts)
        self.plot_area_renderer = PlotAreaRenderer(
            self.screen, self.fonts, self.plotter
        )
        self.rnn_config = RNNConfig()
        self.transformer_config = TransformerConfig()
        self.model_config_net = ModelConfig.Network()

    def _init_fonts(self):
        """Initializes fonts used in the left panel."""
        fonts = {}
        font_configs = {
            "ui": 24,
            "status": 28,
            "logdir": 16,
            "plot_placeholder": 20,
            "notification_label": 16,
            "plot_title_values": 8,
            "progress_bar": 14,
            "notification": 18,
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
        panel_width: int,
        is_process_running: bool,  # Keep for potential future use (MCTS/NN running)
        status: str,
        stats_summary: Dict[str, Any],
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        app_state: str,
        update_progress_details: Dict[str, Any],  # Keep for potential NN progress
        agent_param_count: int,
        worker_counts: Dict[str, int],  # Keep structure, content will change
    ):
        """Renders the entire left panel within the given width."""
        current_height = self.screen.get_height()
        lp_rect = pygame.Rect(0, 0, panel_width, current_height)

        # Simplified status mapping
        status_color_map = {
            "Ready": (30, 30, 30),
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Debugging Grid": (40, 30, 40),
            "Initializing": (40, 40, 40),
            # Add potential future states
            "Running MCTS": (30, 40, 30),
            "Training NN": (30, 30, 50),
        }
        base_status = status.split(" (")[0] if "(" in status else status
        bg_color = status_color_map.get(base_status, (30, 30, 30))

        pygame.draw.rect(self.screen, bg_color, lp_rect)
        current_y = 10

        # Render Buttons and Status (simplified)
        next_y = self.button_status_renderer.render(
            y_start=current_y,
            panel_width=panel_width,
            app_state=app_state,
            is_process_running=is_process_running,  # Pass for potential future use
            status=status,
            stats_summary=stats_summary,
            update_progress_details=update_progress_details,  # Pass for potential NN progress
        )
        current_y = next_y

        # Render Info Text (simplified)
        next_y = self.info_text_renderer.render(
            current_y + 5,
            stats_summary,
            panel_width,
            agent_param_count,
            worker_counts,  # Pass potentially adapted worker counts
        )
        current_y = next_y

        # Render TB Status (unchanged)
        next_y = self.tb_status_renderer.render(
            current_y + 10, tensorboard_log_dir, panel_width
        )
        current_y = next_y

        # Render Plots (unchanged condition, but plots themselves are simplified)
        if app_state == AppState.MAIN_MENU.value:
            self.plot_area_renderer.render(
                y_start=current_y + 5,
                panel_width=panel_width,
                screen_height=current_height,
                plot_data=plot_data,
                status=status,  # Pass status for placeholder text
            )


File: ui\panels\__init__.py
from .left_panel import LeftPanelRenderer
from .game_area import GameAreaRenderer

__all__ = ["LeftPanelRenderer", "GameAreaRenderer"]


File: ui\panels\left_panel_components\button_status_renderer.py
# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
import time
import math
from typing import Dict, Tuple, Any, Optional

from config import WHITE, YELLOW, RED, GOOGLE_COLORS, LIGHTG
from utils.helpers import format_eta
from ui.input_handler import InputHandler


class ButtonStatusRenderer:
    """Renders the top buttons (excluding Run/Stop), and compact status block."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("status", pygame.font.Font(None, 28))
        self.status_label_font = fonts.get(
            "notification_label", pygame.font.Font(None, 16)
        )
        # Removed self.progress_font
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.input_handler_ref: Optional[InputHandler] = None

    def _draw_button(
        self,
        rect: pygame.Rect,
        text: str,
        color: Tuple[int, int, int],
        enabled: bool = True,
    ):
        """Helper to draw a single button, optionally grayed out."""
        final_color = color if enabled else tuple(max(30, c // 2) for c in color[:3])
        pygame.draw.rect(self.screen, final_color, rect, border_radius=5)
        text_color = WHITE if enabled else (150, 150, 150)
        if self.ui_font:
            label_surface = self.ui_font.render(text, True, text_color)
            self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        else:
            pygame.draw.line(self.screen, RED, rect.topleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, RED, rect.topright, rect.bottomleft, 2)

    def _render_compact_status(
        self, y_start: int, panel_width: int, status: str, stats_summary: Dict[str, Any]
    ) -> int:
        """Renders the compact status block below buttons."""
        x_margin, current_y = 10, y_start
        line_height_status = self.status_font.get_linesize()
        line_height_label = self.status_label_font.get_linesize()

        status_text = f"Status: {status}"
        status_color = YELLOW
        if "Error" in status:
            status_color = RED
        elif "Ready" in status:
            status_color = WHITE
        elif "Debugging" in status:
            status_color = (200, 100, 200)
        elif "Playing" in status:
            status_color = (100, 150, 200)
        elif "Initializing" in status:
            status_color = LIGHTG
        elif "Cleaning" in status:
            status_color = (200, 100, 100)
        elif "Confirm" in status:
            status_color = (200, 150, 50)

        status_surface = self.status_font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(status_surface, status_rect)
        current_y += line_height_status

        # Display basic info like total episodes/games played
        global_step = stats_summary.get(
            "global_step", 0
        )  # Step might mean games or NN steps
        total_episodes = stats_summary.get("total_episodes", 0)

        global_step_str = f"{global_step:,}".replace(",", "_")
        eps_str = f"~{total_episodes} Eps"  # Or Games

        line2_text = f"{global_step_str} Steps | {eps_str}"  # Simplified info line
        line2_surface = self.status_label_font.render(line2_text, True, LIGHTG)
        line2_rect = line2_surface.get_rect(topleft=(x_margin, current_y))
        self.screen.blit(line2_surface, line2_rect)

        current_y += line_height_label + 2
        return current_y

    # Removed _render_single_progress_bar
    # Removed _render_detailed_progress_bars

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_process_running: bool,  # Keep for potential future use (e.g., MCTS running)
        status: str,
        stats_summary: Dict[str, Any],
        update_progress_details: Dict[str, Any],  # Keep for potential NN progress
    ) -> int:
        """Renders buttons (excluding Run/Stop) and status. Returns next_y."""
        from app_state import AppState

        next_y = y_start

        # Get button rects from InputHandler
        # Removed run_btn_rect
        cleanup_btn_rect = (
            self.input_handler_ref.cleanup_btn_rect
            if self.input_handler_ref
            else pygame.Rect(10, y_start, 160, 40)
        )
        demo_btn_rect = (
            self.input_handler_ref.demo_btn_rect
            if self.input_handler_ref
            else pygame.Rect(cleanup_btn_rect.right + 10, y_start, 120, 40)
        )
        debug_btn_rect = (
            self.input_handler_ref.debug_btn_rect
            if self.input_handler_ref
            else pygame.Rect(demo_btn_rect.right + 10, y_start, 120, 40)
        )

        # Removed Run/Stop button rendering

        # Render other buttons
        buttons_enabled = app_state == AppState.MAIN_MENU.value
        self._draw_button(
            cleanup_btn_rect, "Cleanup This Run", (100, 40, 40), enabled=buttons_enabled
        )
        self._draw_button(
            demo_btn_rect, "Play Demo", (40, 100, 40), enabled=buttons_enabled
        )
        self._draw_button(
            debug_btn_rect, "Debug Mode", (100, 40, 100), enabled=buttons_enabled
        )
        # Set next_y below the buttons
        button_bottom = max(
            cleanup_btn_rect.bottom, demo_btn_rect.bottom, debug_btn_rect.bottom
        )
        next_y = button_bottom + 10

        # Render Status Block
        status_block_y = next_y
        # Removed check for "Updating Agent" status to render progress bars
        # Always render the compact status block now
        next_y = self._render_compact_status(
            status_block_y, panel_width, status, stats_summary
        )

        # Render NN training progress bar if applicable (using update_progress_details)
        # Example placeholder:
        # if update_progress_details and update_progress_details.get('phase') == 'Training NN':
        #     next_y = self._render_detailed_progress_bars(...) # Adapt this call

        return next_y


File: ui\panels\left_panel_components\info_text_renderer.py
# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple

from config import RNNConfig, TransformerConfig, ModelConfig, WHITE, LIGHTG, GRAY


class InfoTextRenderer:
    """Renders essential non-plotted information text."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.rnn_config = RNNConfig()
        self.transformer_config = TransformerConfig()
        self.model_config_net = ModelConfig.Network()
        self.resource_font = fonts.get("logdir", pygame.font.Font(None, 16))
        self.stats_summary_cache: Dict[str, Any] = {}

    def _get_network_description(self) -> str:
        """Builds a description string based on network components."""
        # Adapt based on AlphaZero NN architecture later
        return "AlphaZero Neural Network"  # Placeholder

    def _get_network_details(self) -> str:
        """Builds a detailed string of network configuration."""
        # Adapt based on AlphaZero NN architecture later
        details = []
        # Example: Add details about ResNet blocks, policy/value heads if known
        # cnn_str = str(self.model_config_net.CONV_CHANNELS).replace(" ", "")
        # details.append(f"CNN Base: {cnn_str}")
        return "Details TBD"  # Placeholder

    def _get_live_resource_usage(self) -> Dict[str, str]:
        """Fetches live CPU, Memory, and GPU Memory usage from cached summary."""
        from config.general import DEVICE

        usage = {"CPU": "N/A", "Mem": "N/A", "GPU Mem": "N/A"}
        cpu_val = self.stats_summary_cache.get("current_cpu_usage")
        mem_val = self.stats_summary_cache.get("current_memory_usage")
        gpu_val = self.stats_summary_cache.get("current_gpu_memory_usage_percent")

        if cpu_val is not None:
            usage["CPU"] = f"{cpu_val:.1f}%"
        if mem_val is not None:
            usage["Mem"] = f"{mem_val:.1f}%"  # Fixed quote

        device_type = DEVICE.type if DEVICE else "cpu"
        if gpu_val is not None:
            usage["GPU Mem"] = (
                f"{gpu_val:.1f}%" if device_type == "cuda" else "N/A (CPU)"
            )
        elif device_type != "cuda":
            usage["GPU Mem"] = "N/A (CPU)"
        return usage

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],
        panel_width: int,
        agent_param_count: int,  # Keep for NN param count
        worker_counts: Dict[str, int],  # Keep structure, but content will change
    ) -> int:
        """Renders the info text block. Returns next_y."""
        from config.general import DEVICE

        self.stats_summary_cache = stats_summary
        ui_font, detail_font, resource_font = (
            self.fonts.get("ui"),
            self.fonts.get("logdir"),
            self.resource_font,
        )
        if not ui_font or not detail_font or not resource_font:
            return y_start

        line_height_ui, line_height_detail, line_height_resource = (
            ui_font.get_linesize(),
            detail_font.get_linesize(),
            resource_font.get_linesize(),
        )
        device_type_str = DEVICE.type.upper() if DEVICE else "CPU"
        network_desc, network_details = (
            self._get_network_description(),
            self._get_network_details(),
        )
        param_str = (
            f"{agent_param_count / 1e6:.2f} M" if agent_param_count > 0 else "N/A"
        )
        start_time_unix = stats_summary.get("start_time", 0.0)
        start_time_str = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_unix))
            if start_time_unix > 0
            else "N/A"
        )
        # Removed worker_str and lr_str (can be added back if needed)

        info_lines = [
            ("Device", device_type_str),
            ("Network", network_desc),
            ("Params", param_str),
            # ("LR", lr_str), # Removed LR for now
            # ("Workers", worker_str), # Removed Workers for now
            ("Run Started", start_time_str),
        ]
        last_y, x_pos_key, x_pos_val_offset, current_y = y_start, 10, 5, y_start + 5

        for idx, (key, value_str) in enumerate(info_lines):
            line_y = current_y + idx * line_height_ui
            try:
                key_surf = ui_font.render(f"{key}:", True, LIGHTG)
                key_rect = key_surf.get_rect(topleft=(x_pos_key, line_y))
                self.screen.blit(key_surf, key_rect)
                value_surf = ui_font.render(f"{value_str}", True, WHITE)
                value_rect = value_surf.get_rect(
                    topleft=(key_rect.right + x_pos_val_offset, line_y)
                )
                clip_width = max(0, panel_width - value_rect.left - 10)
                blit_area = (
                    pygame.Rect(0, 0, clip_width, value_rect.height)
                    if value_rect.width > clip_width
                    else None
                )
                self.screen.blit(value_surf, value_rect, area=blit_area)
                last_y = key_rect.union(value_rect).bottom
            except Exception as e:
                print(f"Error rendering stat line '{key}': {e}")
                last_y = line_y + line_height_ui

        current_y = last_y + 2
        try:
            detail_surf = detail_font.render(network_details, True, WHITE)
            detail_rect = detail_surf.get_rect(topleft=(x_pos_key, current_y))
            clip_width_detail = max(0, panel_width - detail_rect.left - 10)
            blit_area_detail = (
                pygame.Rect(0, 0, clip_width_detail, detail_rect.height)
                if detail_rect.width > clip_width_detail
                else None
            )
            self.screen.blit(detail_surf, detail_rect, area=blit_area_detail)
            last_y = detail_rect.bottom
        except Exception as e:
            print(f"Error rendering network details: {e}")
            last_y = current_y + line_height_detail

        current_y = last_y + 4
        resource_usage = self._get_live_resource_usage()
        resource_str = f"Live Usage | CPU: {resource_usage['CPU']} | Mem: {resource_usage['Mem']} | GPU Mem: {resource_usage['GPU Mem']}"
        try:
            resource_surf = resource_font.render(resource_str, True, WHITE)
            resource_rect = resource_surf.get_rect(topleft=(x_pos_key, current_y))
            clip_width_resource = max(0, panel_width - resource_rect.left - 10)
            blit_area_resource = (
                pygame.Rect(0, 0, clip_width_resource, resource_rect.height)
                if resource_rect.width > clip_width_resource
                else None
            )
            self.screen.blit(resource_surf, resource_rect, area=blit_area_resource)
            last_y = resource_rect.bottom
        except Exception as e:
            print(f"Error rendering resource usage: {e}")
            last_y = current_y + line_height_resource

        return last_y


File: ui\panels\left_panel_components\notification_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple, Optional
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
        val_as_float: Optional[float] = None
        if isinstance(current_val, (int, float, np.number)):
            try:
                val_as_float = float(current_val)
            except (ValueError, TypeError):
                val_as_float = None

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
from typing import Dict, Deque, Any, Optional, Tuple
import numpy as np
from config import (
    VisConfig,
    LIGHTG,
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    GRAY,
)  # Added GRAY
from ui.plotter import Plotter


class PlotAreaRenderer:
    """Renders the plot area using a Plotter instance."""

    def __init__(
        self,
        screen: pygame.Surface,
        fonts: Dict[str, pygame.font.Font],
        plotter: Plotter,
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
        plot_area_y_start = y_start
        plot_area_height = screen_height - plot_area_y_start - 10
        plot_area_width = panel_width - 20

        if plot_area_width <= 50 or plot_area_height <= 50:
            return

        plot_surface = self.plotter.get_cached_or_updated_plot(
            plot_data, plot_area_width, plot_area_height
        )
        plot_area_rect = pygame.Rect(
            10, plot_area_y_start, plot_area_width, plot_area_height
        )

        if plot_surface:
            self.screen.blit(plot_surface, plot_area_rect.topleft)
        else:
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
                placeholder_surf = placeholder_font.render(placeholder_text, True, GRAY)
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
                    GRAY,
                    plot_area_rect.topleft,
                    plot_area_rect.bottomright,
                )
                pygame.draw.line(
                    self.screen,
                    GRAY,
                    plot_area_rect.topright,
                    plot_area_rect.bottomleft,
                )


File: ui\panels\left_panel_components\pretrain_status_renderer.py
# File: ui/panels/left_panel_components/pretrain_status_renderer.py
import pygame
from typing import Dict, Any, Tuple, Optional
from config import YELLOW, LIGHTG, GOOGLE_COLORS
from utils.helpers import format_eta  # Import from new location


class PretrainStatusRenderer:
    """Renders the status and progress of the pre-training phase."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("pretrain_status", pygame.font.Font(None, 20))
        self.progress_font = fonts.get(
            "pretrain_progress_bar", pygame.font.Font(None, 14)
        )
        self.detail_font = fonts.get("pretrain_detail", pygame.font.Font(None, 16))

    def render(
        self, y_start: int, pretrain_info: Dict[str, Any], panel_width: int
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the pre-training status block. Returns next_y and stat_rects."""
        stat_rects: Dict[str, pygame.Rect] = {}
        if not pretrain_info or not self.status_font or not self.detail_font:
            return y_start, stat_rects

        x_margin = 10
        current_y = y_start
        line_height_status = self.status_font.get_linesize()
        line_height_detail = self.detail_font.get_linesize()

        phase = pretrain_info.get("phase", "Unknown")
        status_text = f"Pre-Train: {phase}"
        status_color = YELLOW
        detail_text = ""

        overall_eta_str = format_eta(
            pretrain_info.get("overall_eta_seconds", pretrain_info.get("eta_seconds"))
        )
        if phase == "Random Play":
            games = pretrain_info.get("games_played", 0)
            target = pretrain_info.get("target_games", 0)
            pps = pretrain_info.get("plays_per_second", 0.0)
            num_envs = pretrain_info.get("num_envs", 0)
            status_text = f"Pre-Train: Random Play ({games:,}/{target:,})"
            detail_text = (
                f"{num_envs} Envs | {pps:.1f} Plays/s | ETA: {overall_eta_str}"
            )
            status_color = GOOGLE_COLORS[1]
        elif phase == "Sorting Games":
            status_text = "Pre-Train: Sorting Games..."
            status_color = (200, 150, 50)
        elif phase == "Replaying Top K":
            replayed = pretrain_info.get("games_replayed", 0)
            target = pretrain_info.get("target_games", 0)
            transitions = pretrain_info.get("transitions_collected", 0)
            status_text = f"Pre-Train: Replaying ({replayed:,}/{target:,})"
            detail_text = f"Collecting Transitions ({transitions:,})"
            status_color = (100, 180, 180)
        elif phase == "Updating Agent":
            epoch = pretrain_info.get("epoch", 0)
            total_epochs = pretrain_info.get("total_epochs", 0)
            status_text = f"Pre-Train: Updating (Epoch {epoch}/{total_epochs})"
            detail_text = f"Overall ETA: {overall_eta_str}"  # Show ETA here
            status_color = GOOGLE_COLORS[2]

        # Render Status Line
        status_surface = self.status_font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(topleft=(x_margin, current_y))
        clip_width_status = max(0, panel_width - status_rect.left - x_margin)
        if status_rect.width > clip_width_status:
            self.screen.blit(
                status_surface,
                status_rect,
                area=pygame.Rect(0, 0, clip_width_status, status_rect.height),
            )
        else:
            self.screen.blit(status_surface, status_rect)
        stat_rects["Pre-training Status"] = status_rect
        current_y += line_height_status

        # Render Detail Line
        if detail_text:
            detail_surface = self.detail_font.render(detail_text, True, LIGHTG)
            detail_rect = detail_surface.get_rect(topleft=(x_margin + 2, current_y))
            clip_width_detail = max(0, panel_width - detail_rect.left - x_margin)
            if detail_rect.width > clip_width_detail:
                self.screen.blit(
                    detail_surface,
                    detail_rect,
                    area=pygame.Rect(0, 0, clip_width_detail, detail_rect.height),
                )
            else:
                self.screen.blit(detail_surface, detail_rect)
            current_y += line_height_detail
        current_y += 5  # Add final padding

        return current_y, stat_rects


File: ui\panels\left_panel_components\tb_status_renderer.py
# File: ui/panels/left_panel_components/tb_status_renderer.py
import pygame
import os
from typing import Dict, Optional, Tuple
from config import (
    VisConfig,
    TensorBoardConfig,
    GRAY,
    LIGHTG,
    GOOGLE_COLORS,
    WHITE,  # Added WHITE
)


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
        except ValueError:
            rel_path = path
        if len(rel_path) <= max_chars:
            return rel_path
        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            short_path = os.path.join("...", *parts[-2:])
            if len(short_path) <= max_chars:
                return short_path
        basename = os.path.basename(path)
        return (
            "..." + basename[-(max_chars - 3) :]
            if len(basename) > max_chars - 3
            else basename
        )

    def render(
        self, y_start: int, log_dir: Optional[str], panel_width: int
    ) -> int:  # Removed returning rects
        """Renders the TB status. Returns next_y."""
        # Removed stat_rects initialization
        ui_font, logdir_font = self.fonts.get("ui"), self.fonts.get("logdir")
        if not ui_font or not logdir_font:
            return y_start + 30

        tb_active = (
            TensorBoardConfig.LOG_HISTOGRAMS
            or TensorBoardConfig.LOG_IMAGES
            or TensorBoardConfig.LOG_SHAPE_PLACEMENT_Q_VALUES
        )
        tb_color = GOOGLE_COLORS[0] if tb_active else GRAY
        tb_text = f"TensorBoard: {'Logging Active' if tb_active else 'Logging Minimal'}"

        tb_surf = ui_font.render(tb_text, True, tb_color)
        tb_rect = tb_surf.get_rect(topleft=(10, y_start))
        self.screen.blit(tb_surf, tb_rect)
        # Removed stat_rects update
        last_y = tb_rect.bottom

        if log_dir:
            try:
                panel_char_width = max(
                    10, panel_width // max(1, logdir_font.size("A")[0])
                )
            except Exception:
                panel_char_width = 30  # Fallback
            short_log_dir = self._shorten_path(log_dir, panel_char_width)

            # --- Change color here ---
            dir_surf = logdir_font.render(f"Log Dir: {short_log_dir}", True, WHITE)
            # --- End change color ---
            dir_rect = dir_surf.get_rect(topleft=(10, tb_rect.bottom + 2))
            clip_width = max(0, panel_width - dir_rect.left - 10)
            blit_area = (
                pygame.Rect(0, 0, clip_width, dir_rect.height)
                if dir_rect.width > clip_width
                else None
            )
            self.screen.blit(dir_surf, dir_rect, area=blit_area)

            # Removed combined_tb_rect calculation and stat_rects update
            last_y = dir_rect.bottom

        return last_y  # Return only next_y


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
import torch
import numpy as np
import random
import os
import pickle
import cloudpickle
import math  # Added for format_eta
from typing import Union, Any, Optional  # Added Optional for format_eta


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
        print("MPS device found on MacOS.")  # Won't execute on PC
    else:
        print(
            "No CUDA or MPS device found, falling back to CPU."
        )  # This is what's happening now

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
        raise e  # Re-raise after logging


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


def format_eta(seconds: Optional[float]) -> str:
    """Formats seconds into a human-readable HH:MM:SS or MM:SS string."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "N/A"
    if seconds > 3600 * 24 * 30:  # Cap at roughly a month
        return ">1 month"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"


File: utils\init_checks.py
# File: utils/init_checks.py
import sys
import traceback
import numpy as np

from environment.game_state import GameState


def run_pre_checks() -> bool:
    """Performs basic checks on GameState and configuration compatibility."""
    try:
        from config import EnvConfig
    except ImportError as e:
        print(f"FATAL ERROR: Could not import EnvConfig during pre-check: {e}")
        print(
            "This might indicate an issue with the config package structure or an ongoing import cycle."
        )
        sys.exit(1)

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

        if "shapes" not in s_test_dict:
            raise KeyError("State dictionary missing 'shapes' key.")
        shape_state = s_test_dict["shapes"]
        expected_shape_shape = (env_config_instance.SHAPE_STATE_DIM,)
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

        if "explicit_features" not in s_test_dict:
            raise KeyError("State dictionary missing 'explicit_features' key.")
        explicit_features_state = s_test_dict["explicit_features"]
        expected_explicit_features_shape = (env_config_instance.EXPLICIT_FEATURES_DIM,)
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

        if env_config_instance.CALCULATE_POTENTIAL_OUTCOMES_IN_STATE:
            print("Potential outcome calculation is ENABLED in EnvConfig.")
        else:
            print("Potential outcome calculation is DISABLED in EnvConfig.")

        # Removed PBRS check

        _ = gs_test.valid_actions()
        print("GameState valid_actions check PASSED.")
        if not hasattr(gs_test, "game_score"):
            raise AttributeError("GameState missing 'game_score' attribute!")
        print("GameState 'game_score' attribute check PASSED.")
        if not hasattr(gs_test, "triangles_cleared_this_episode"):
            raise AttributeError(
                "GameState missing 'triangles_cleared_this_episode' attribute!"
            )
        print("GameState 'triangles_cleared_this_episode' attribute check PASSED.")

        del gs_test
        print("--- Pre-Run Checks Complete ---")
        return True
    except (NameError, ImportError) as e:
        print(f"FATAL ERROR: Import/Name error during pre-check: {e}")
    except (ValueError, AttributeError, TypeError, KeyError) as e:
        print(f"FATAL ERROR during pre-run checks: {e}")
    except Exception as e:
        print(f"FATAL ERROR during GameState pre-check: {e}")
        traceback.print_exc()
    sys.exit(1)


File: utils\types.py
from typing import Dict, Any
import numpy as np

StateType = Dict[str, np.ndarray]
ActionType = int
AgentStateDict = Dict[str, Any]


File: utils\__init__.py
# File: utils/__init__.py
from .helpers import (
    get_device,
    set_random_seeds,
    ensure_numpy,
    save_object,
    load_object,
    format_eta,  # Added format_eta
)
from .init_checks import run_pre_checks
from .types import StateType, ActionType, AgentStateDict
from .running_mean_std import RunningMeanStd


__all__ = [
    "get_device",
    "set_random_seeds",
    "ensure_numpy",
    "save_object",
    "load_object",
    "format_eta",  # Added format_eta
    "run_pre_checks",
    "StateType",
    "ActionType",
    "AgentStateDict",
    "RunningMeanStd",
]


File: visualization\__init__.py


