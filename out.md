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
import pygame
import time
import traceback
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # Import scheduler
from typing import TYPE_CHECKING, List, Optional

from config import (
    ModelConfig,
    StatsConfig,
    DemoConfig,
    MCTSConfig,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
)
from environment.game_state import GameState
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregator
from stats.simple_stats_recorder import SimpleStatsRecorder
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler
from training.checkpoint_manager import CheckpointManager
from app_state import AppState
from mcts import MCTS
from agent.alphazero_net import AlphaZeroNet
from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker

if TYPE_CHECKING:
    from main_pygame import MainApp
    from torch.optim.lr_scheduler import _LRScheduler


class AppInitializer:
    """Handles the initialization of core application components."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Config instances
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        self.train_config = app.train_config_instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.demo_config = DemoConfig()
        self.mcts_config = MCTSConfig()

        # Components to be initialized
        self.agent: Optional[AlphaZeroNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional["_LRScheduler"] = None  # Add scheduler attribute
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.stats_aggregator: Optional[StatsAggregator] = None
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.mcts: Optional[MCTS] = None
        self.self_play_workers: List[SelfPlayWorker] = []
        self.training_worker: Optional[TrainingWorker] = None

    def initialize_all(self, is_reinit: bool = False):
        """Initializes all core components."""
        try:
            self._check_gpu_memory()
            if not is_reinit:
                self._initialize_ui_early()

            self.initialize_rl_components(
                is_reinit=is_reinit, checkpoint_to_load=self.app.checkpoint_to_load
            )

            if not is_reinit:
                self.initialize_demo_env()
                self.initialize_input_handler()

            self._calculate_agent_params()
            self.initialize_workers()  # Workers now need the scheduler

        except Exception as init_err:
            self._handle_init_error(init_err)

    def _check_gpu_memory(self):
        """Checks and prints total GPU memory if available."""
        if self.app.device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(self.app.device)
                self.app.total_gpu_memory_bytes = props.total_memory
                print(f"Total GPU Memory: {props.total_memory / (1024**3):.2f} GB")
            except Exception as e:
                print(f"Warning: Could not get total GPU memory: {e}")

    def _initialize_ui_early(self):
        """Initializes the renderer and performs an initial render."""
        self.app.renderer = UIRenderer(self.app.screen, self.vis_config)
        self.app.renderer.render_all(
            app_state=self.app.app_state.value,
            is_process_running=False,
            status=self.app.status,
            stats_summary={},
            envs=[],
            num_envs=0,
            env_config=self.env_config,
            cleanup_confirmation_active=False,
            cleanup_message="",
            last_cleanup_message_time=0,
            plot_data={},
            demo_env=None,
            update_progress_details={},
            agent_param_count=0,
            worker_counts={},
            best_game_state_data=None,
        )
        pygame.display.flip()
        pygame.time.delay(100)  # Allow UI to update

    def _calculate_agent_params(self):
        """Calculates the number of trainable parameters in the agent."""
        if self.agent:
            try:
                self.agent_param_count = sum(
                    p.numel() for p in self.agent.parameters() if p.requires_grad
                )
            except Exception as e:
                print(f"Warning: Could not calculate agent parameters: {e}")
                self.agent_param_count = 0

    def _handle_init_error(self, error: Exception):
        """Handles fatal errors during initialization."""
        print(f"FATAL ERROR during component initialization: {error}")
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
        """Initializes NN Agent, Optimizer, Scheduler, MCTS, Stats, Checkpoint Manager."""
        print(f"Initializing AlphaZero components... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            self._init_agent()
            self._init_optimizer_and_scheduler()  # Renamed method
            self._init_mcts()
            self._init_stats()
            self._init_checkpoint_manager(
                checkpoint_to_load
            )  # Checkpoint manager needs scheduler now
            print(
                f"AlphaZero components initialized in {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            print(f"Error during AlphaZero component initialization: {e}")
            traceback.print_exc()
            raise e

    def _init_agent(self):
        self.agent = AlphaZeroNet(
            env_config=self.env_config, model_config=self.model_config.Network()
        ).to(self.app.device)
        print(f"AlphaZeroNet initialized on device: {self.app.device}.")

    def _init_optimizer_and_scheduler(self):
        """Initializes the optimizer and the learning rate scheduler."""
        if not self.agent:
            raise RuntimeError("Agent must be initialized before Optimizer.")
        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.train_config.LEARNING_RATE,
            weight_decay=self.train_config.WEIGHT_DECAY,
        )
        print(f"Optimizer initialized (Adam, LR={self.train_config.LEARNING_RATE}).")

        # Initialize Scheduler (if enabled)
        if self.train_config.USE_LR_SCHEDULER:
            if self.train_config.SCHEDULER_TYPE == "CosineAnnealingLR":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.train_config.SCHEDULER_T_MAX,
                    eta_min=self.train_config.SCHEDULER_ETA_MIN,
                )
                print(
                    f"LR Scheduler initialized (CosineAnnealingLR, T_max={self.train_config.SCHEDULER_T_MAX}, eta_min={self.train_config.SCHEDULER_ETA_MIN})."
                )
            # Add other scheduler types here if needed
            # elif self.train_config.SCHEDULER_TYPE == "OneCycleLR":
            #     self.scheduler = OneCycleLR(...)
            else:
                print(
                    f"Warning: Unknown scheduler type '{self.train_config.SCHEDULER_TYPE}'. No scheduler initialized."
                )
                self.scheduler = None
        else:
            print("LR Scheduler is DISABLED.")
            self.scheduler = None

    def _init_mcts(self):
        if not self.agent:
            raise RuntimeError("Agent must be initialized before MCTS.")
        self.mcts = MCTS(
            network_predictor=self.agent.predict,
            config=self.mcts_config,
            env_config=self.env_config,
        )
        print("MCTS initialized with AlphaZeroNet predictor.")

    def _init_stats(self):
        print("Initializing StatsAggregator and SimpleStatsRecorder...")
        self.stats_aggregator = StatsAggregator(
            avg_windows=self.stats_config.STATS_AVG_WINDOW,
            plot_window=self.stats_config.PLOT_DATA_WINDOW,
        )
        self.stats_recorder = SimpleStatsRecorder(
            aggregator=self.stats_aggregator,
            console_log_interval=self.stats_config.CONSOLE_LOG_FREQ,
            train_config=self.train_config,
        )
        print("StatsAggregator and SimpleStatsRecorder initialized.")

    def _init_checkpoint_manager(self, checkpoint_to_load: Optional[str]):
        if not self.agent or not self.optimizer or not self.stats_aggregator:
            raise RuntimeError(
                "Agent, Optimizer, StatsAggregator needed for CheckpointManager."
            )
        # Pass the scheduler to the CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            agent=self.agent,
            optimizer=self.optimizer,
            scheduler=self.scheduler,  # Pass scheduler
            stats_aggregator=self.stats_aggregator,
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            run_checkpoint_dir=get_run_checkpoint_dir(),
            load_checkpoint_path_config=checkpoint_to_load,
            device=self.app.device,
        )
        if self.checkpoint_manager.get_checkpoint_path_to_load():
            self.checkpoint_manager.load_checkpoint()

    def initialize_workers(self):
        """Initializes worker threads (Self-Play, Training). Does NOT start them."""
        print("Initializing worker threads...")
        if (
            not self.agent
            or not self.mcts
            or not self.stats_aggregator
            or not self.optimizer
            # Scheduler is optional, so don't check it here
        ):
            print("ERROR: Cannot initialize workers, core RL components missing.")
            return

        self._init_self_play_workers()
        self._init_training_worker()  # Training worker needs scheduler
        num_sp = len(self.self_play_workers)
        print(f"Worker threads initialized ({num_sp} Self-Play, 1 Training).")

    def _init_self_play_workers(self):
        self.self_play_workers = []
        num_sp_workers = self.train_config.NUM_SELF_PLAY_WORKERS
        print(f"Initializing {num_sp_workers} SelfPlayWorker(s)...")
        for i in range(num_sp_workers):
            worker = SelfPlayWorker(
                worker_id=i,
                agent=self.agent,
                mcts=self.mcts,
                experience_queue=self.app.experience_queue,
                stats_aggregator=self.stats_aggregator,
                stop_event=self.app.stop_event,
                env_config=self.env_config,
                mcts_config=self.mcts_config,
                device=self.app.device,
            )
            self.self_play_workers.append(worker)
            print(f"  SelfPlayWorker-{i} initialized.")

    def _init_training_worker(self):
        # Pass the scheduler to the TrainingWorker
        self.training_worker = TrainingWorker(
            agent=self.agent,
            optimizer=self.optimizer,
            scheduler=self.scheduler,  # Pass scheduler
            experience_queue=self.app.experience_queue,
            stats_aggregator=self.stats_aggregator,
            stop_event=self.app.stop_event,
            train_config=self.train_config,
            device=self.app.device,
        )
        print("TrainingWorker initialized.")

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
            start_run_cb=self.app.logic.start_run,
            stop_run_cb=self.app.logic.stop_run,
        )
        if self.app.input_handler:
            self.app.input_handler.app_ref = self.app
        if self.app.renderer and self.app.renderer.left_panel:
            self.app.renderer.left_panel.input_handler = self.app.input_handler
            if hasattr(self.app.renderer.left_panel, "button_status_renderer"):
                btn_renderer = self.app.renderer.left_panel.button_status_renderer
                btn_renderer.input_handler_ref = self.app.input_handler
                btn_renderer.app_ref = self.app

    def close_stats_recorder(self, is_cleanup: bool = False):
        """Safely closes the stats recorder (if it exists)."""
        print(
            f"[AppInitializer] close_stats_recorder called (is_cleanup={is_cleanup})..."
        )
        if self.stats_recorder and hasattr(self.stats_recorder, "close"):
            print("[AppInitializer] Stats recorder exists, attempting close...")
            try:
                self.stats_recorder.close(is_cleanup=is_cleanup)
                print("[AppInitializer] stats_recorder.close() executed.")
            except Exception as log_e:
                print(f"[AppInitializer] Error closing stats recorder on exit: {log_e}")
                traceback.print_exc()
        else:
            print("[AppInitializer] No stats recorder instance or close method.")
        print("[AppInitializer] close_stats_recorder finished.")


File: app_logic.py
import pygame
import time
import traceback
import os
import shutil
from typing import TYPE_CHECKING, Tuple

from app_state import AppState
from config.general import get_run_checkpoint_dir 

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppLogic:
    """Handles the core application logic and state transitions."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def check_initial_completion_status(self):
        """Checks if training target was met upon loading (placeholder)."""
        pass 

    def update_status_and_check_completion(self):
        """Updates the status text based on application state."""
        is_running = self.app.worker_manager.is_any_worker_running()
        state = self.app.app_state
        if state == AppState.MAIN_MENU:
            self.app.status = (
                "Confirm Cleanup"
                if self.app.cleanup_confirmation_active
                else "Running AlphaZero" if is_running else "Ready"
            )
        elif state == AppState.PLAYING:
            self.app.status = "Playing Demo"
        elif state == AppState.DEBUG:
            self.app.status = "Debugging Grid"
        elif state == AppState.INITIALIZING:
            self.app.status = "Initializing..."

    # --- Worker Control ---
    def start_run(self):
        """Starts both self-play and training workers."""
        if (
            self.app.app_state != AppState.MAIN_MENU
            or self.app.worker_manager.is_any_worker_running()
        ):
            print("Cannot start run: Not in Main Menu or already running.")
            return
        print("Starting AlphaZero Run (Self-Play & Training)...")
        self.app.worker_manager.start_all_workers()
        self.update_status_and_check_completion()

    def stop_run(self):
        """Stops both self-play and training workers."""
        if not self.app.worker_manager.is_any_worker_running():
            print("Run not currently active.")
            return
        print("Stopping AlphaZero Run...")
        self.app.worker_manager.stop_all_workers()
        self.update_status_and_check_completion()

    # --- Mode Transitions & Cleanup ---
    def request_cleanup(self):
        if self.app.app_state != AppState.MAIN_MENU:
            return
        if self.app.worker_manager.is_any_worker_running():
            self._set_temp_message("Stop Run before Cleanup!")
            return
        self.app.cleanup_confirmation_active = True
        self.app.status = "Confirm Cleanup"
        print("Cleanup requested. Confirm action.")

    def start_demo_mode(self):
        if self._can_start_mode("Demo"):
            print("Entering Demo Mode...")
            self.try_save_checkpoint()
            self.app.app_state = AppState.PLAYING
            self.app.status = "Playing Demo"
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()

    def start_debug_mode(self):
        if self._can_start_mode("Debug"):
            print("Entering Debug Mode...")
            self.try_save_checkpoint()
            self.app.app_state = AppState.DEBUG
            self.app.status = "Debugging Grid"
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()

    def _can_start_mode(self, mode_name: str) -> bool:
        """Checks if demo/debug mode can be started."""
        if self.app.initializer.demo_env is None:
            print(f"Cannot start {mode_name}: Env not initialized.")
            return False
        if self.app.app_state != AppState.MAIN_MENU:
            print(f"Cannot start {mode_name} mode outside MainMenu.")
            return False
        if self.app.worker_manager.is_any_worker_running():
            self._set_temp_message(f"Stop Run before {mode_name}!")
            return False
        return True

    def exit_demo_mode(self):
        if self.app.app_state == AppState.PLAYING:
            print("Exiting Demo Mode...")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.deselect_dragged_shape()
            self._return_to_main_menu()

    def exit_debug_mode(self):
        if self.app.app_state == AppState.DEBUG:
            print("Exiting Debug Mode...")
            self._return_to_main_menu()

    def _return_to_main_menu(self):
        """Helper to transition back to the main menu state."""
        self.app.app_state = AppState.MAIN_MENU
        self.check_initial_completion_status()
        self.update_status_and_check_completion()

    def cancel_cleanup(self):
        self.app.cleanup_confirmation_active = False
        self._set_temp_message("Cleanup cancelled.")
        self.update_status_and_check_completion()
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
            print(
                f"Cleanup process finished. State: {self.app.app_state}, Status: {self.app.status}"
            )

    def exit_app(self) -> bool:
        print("Exit requested.")
        self.app.stop_event.set()
        self.app.worker_manager.stop_all_workers()
        return False  # Signal main loop to stop

    # --- Input Handling Callbacks ---
    def handle_demo_mouse_motion(self, mouse_pos: Tuple[int, int]):
        if self.app.app_state != AppState.PLAYING or not self.app.initializer.demo_env:
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if demo_env.demo_dragged_shape_idx is None:
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        demo_env.update_snapped_position(grid_coords)

    def handle_demo_mouse_button_down(self, event: pygame.event.Event):
        if self.app.app_state != AppState.PLAYING or not self.app.initializer.demo_env:
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over() or event.button != 1:
            return

        mouse_pos = event.pos
        clicked_preview = self.app.ui_utils.map_screen_to_preview(mouse_pos)
        if clicked_preview is not None:
            action = (
                demo_env.deselect_dragged_shape
                if clicked_preview == demo_env.demo_dragged_shape_idx
                else lambda: demo_env.select_shape_for_drag(clicked_preview)
            )
            action()
            return

        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        if (
            grid_coords is not None
            and demo_env.demo_dragged_shape_idx is not None
            and demo_env.demo_snapped_position == grid_coords
        ):
            placed = demo_env.place_dragged_shape()
            if placed and demo_env.is_over():
                print("[Demo] Game Over! Press ESC to exit.")
        else:
            demo_env.deselect_dragged_shape()

    def handle_debug_input(self, event: pygame.event.Event):
        if self.app.app_state != AppState.DEBUG or not self.app.initializer.demo_env:
            return
        demo_env = self.app.initializer.demo_env
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            print("[Debug] Resetting grid...")
            demo_env.reset()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            clicked_coords = self.app.ui_utils.map_screen_to_grid(event.pos)
            if clicked_coords:
                demo_env.toggle_triangle_debug(*clicked_coords)

    # --- Internal Helpers ---
    def _set_temp_message(self, message: str):
        """Sets a temporary message to be displayed."""
        self.app.cleanup_message = message
        self.app.last_cleanup_message_time = time.time()

    def _cleanup_data(self):
        """Deletes current run's checkpoint and re-initializes components."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.app.app_state = AppState.INITIALIZING
        self.app.status = "Cleaning"
        messages = []
        self._render_during_cleanup()

        print("[Cleanup] Stopping existing worker threads (if any)...")
        self.app.worker_manager.stop_all_workers()
        print("[Cleanup] Existing worker threads stopped.")
        print("[Cleanup] Closing stats recorder...")
        self.app.initializer.close_stats_recorder(is_cleanup=True)
        print("[Cleanup] Stats recorder closed.")

        messages.append(self._delete_checkpoint_dir())
        time.sleep(0.1)

        print("[Cleanup] Re-initializing components...")
        try:
            self.app.initializer.initialize_rl_components(
                is_reinit=True, checkpoint_to_load=None
            )
            print("[Cleanup] Components re-initialized.")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
            self.app.initializer.initialize_workers()
            print("[Cleanup] Workers re-initialized (not started).")
            messages.append("Components re-initialized.")
            self.app.status = "Ready"
            self.app.app_state = AppState.MAIN_MENU
        except Exception as e:
            print(f"FATAL ERROR during re-initialization after cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Re-init Failed"
            self.app.app_state = AppState.ERROR
            messages.append("ERROR RE-INITIALIZING COMPONENTS!")
            if self.app.renderer:
                self.app.renderer._render_error_screen(self.app.status)

        self._set_temp_message("\n".join(messages))
        print(f"--- CLEANUP DATA COMPLETE (Final State: {self.app.app_state}) ---")

    def _render_during_cleanup(self):
        """Renders the screen while cleanup is in progress."""
        if self.app.renderer:
            try:
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=False,
                    status=self.app.status,
                    stats_summary={},
                    envs=[],
                    num_envs=0,
                    env_config=self.app.env_config,
                    cleanup_confirmation_active=False,
                    cleanup_message="",
                    last_cleanup_message_time=0,
                    plot_data={},
                    demo_env=self.app.initializer.demo_env,
                    update_progress_details={},
                    agent_param_count=getattr(
                        self.app.initializer, "agent_param_count", 0
                    ),
                    worker_counts={},
                    best_game_state_data=None,
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

    def _delete_checkpoint_dir(self) -> str:
        """Deletes the checkpoint directory and returns a status message."""
        print("[Cleanup] Deleting agent checkpoint file/dir...")
        msg = ""
        try:
            save_dir = get_run_checkpoint_dir()
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
                msg = f"Run checkpoint directory deleted: {save_dir}"
            else:
                msg = f"Run checkpoint directory not found: {save_dir}"
        except OSError as e:
            msg = f"Error deleting checkpoint dir: {e}"
        print(f"  - {msg}")
        print("[Cleanup] Checkpoint deletion attempt finished.")
        return msg

    def try_save_checkpoint(self):
        """Saves checkpoint if in main menu and workers are not running."""
        if (
            self.app.app_state != AppState.MAIN_MENU
            or self.app.worker_manager.is_any_worker_running()
        ):
            return
        if (
            not self.app.initializer.checkpoint_manager
            or not self.app.initializer.stats_aggregator
        ):
            return

        print("Saving checkpoint...")
        try:
            agg_storage = self.app.initializer.stats_aggregator.storage
            current_step = getattr(agg_storage, "current_global_step", 0)
            episode_count = getattr(agg_storage, "total_episodes", 0)
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
        """Saves the final checkpoint."""
        if (
            not self.app.initializer.checkpoint_manager
            or not self.app.initializer.stats_aggregator
        ):
            return
        save_on_exit = (
            self.app.status != "Cleaning" and self.app.app_state != AppState.ERROR
        )
        if save_on_exit:
            print("Performing final checkpoint save...")
            try:
                agg_storage = self.app.initializer.stats_aggregator.storage
                current_step = getattr(agg_storage, "current_global_step", 0)
                episode_count = getattr(agg_storage, "total_episodes", 0)
                target_step = getattr(
                    self.app.initializer.checkpoint_manager, "training_target_step", 0
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
from enum import Enum


class AppState(Enum):
    INITIALIZING = "Initializing"
    MAIN_MENU = "MainMenu"
    PLAYING = "Playing" 
    DEBUG = "Debug"
    CLEANUP_CONFIRM = "Confirm Cleanup"
    CLEANING = "Cleaning"
    ERROR = "Error"
    UNKNOWN = "Unknown"


File: app_ui_utils.py
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
from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Any
import logging

from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker
from environment.game_state import GameState

if TYPE_CHECKING:
    from main_pygame import MainApp

logger = logging.getLogger(__name__)


class AppWorkerManager:
    """Manages the creation, starting, and stopping of worker threads."""

    def __init__(self, app: "MainApp"):
        self.app = app
        self.self_play_worker_threads: List[SelfPlayWorker] = (
            []
        )  # Holds running thread objects
        self.training_worker_thread: Optional[TrainingWorker] = (
            None  # Holds running thread object
        )
        print("[AppWorkerManager] Initialized.")

    def get_active_worker_counts(self) -> Dict[str, int]:
        """Returns the count of currently active workers by type."""
        # Count based on living threads stored in this manager
        sp_count = sum(1 for w in self.self_play_worker_threads if w and w.is_alive())
        tr_count = 1 if self.is_training_running() else 0
        return {"SelfPlay": sp_count, "Training": tr_count}

    def is_self_play_running(self) -> bool:
        """Checks if *any* self-play worker thread is active."""
        return any(
            w is not None and w.is_alive() for w in self.self_play_worker_threads
        )

    def is_training_running(self) -> bool:
        """Checks if the training worker thread is active."""
        return (
            self.training_worker_thread is not None
            and self.training_worker_thread.is_alive()
        )

    def is_any_worker_running(self) -> bool:
        """Checks if any worker thread is active."""
        return self.is_self_play_running() or self.is_training_running()

    def get_worker_render_data(self, max_envs: int) -> List[Optional[Dict[str, Any]]]:
        """
        Retrieves render data (state copy and stats) from active self-play workers.
        Returns a list of dictionaries [{state: GameState, stats: Dict}, ... ] or None.
        Accesses worker instances directly from the initializer.
        """
        render_data_list: List[Optional[Dict[str, Any]]] = []
        count = 0
        # Get worker instances from the initializer, as they persist even if thread stops/restarts
        worker_instances = self.app.initializer.self_play_workers

        for worker in worker_instances:
            if count >= max_envs:
                break  # Limit reached

            # Check if the worker instance exists and is *currently running*
            if (
                worker
                and worker.is_alive()
                and hasattr(worker, "get_current_render_data")
            ):
                try:
                    # Fetch the combined state and stats dictionary
                    data: Dict[str, Any] = worker.get_current_render_data()
                    render_data_list.append(data)
                except Exception as e:
                    logger.error(
                        f"Error getting render data from worker {worker.worker_id}: {e}"
                    )
                    render_data_list.append(None)  # Append None on error
            else:
                # Append None if worker doesn't exist, isn't alive, or lacks method
                render_data_list.append(None)

            count += 1

        # Pad with None if fewer workers than max_envs
        while len(render_data_list) < max_envs:
            render_data_list.append(None)

        return render_data_list

    def start_all_workers(self):
        """Starts all initialized worker threads if they are not already running."""
        if self.is_any_worker_running():
            logger.warning(
                "[AppWorkerManager] Attempted to start workers, but some are already running."
            )
            return

        # Check if necessary components are initialized
        if (
            not self.app.initializer.agent
            or not self.app.initializer.mcts
            or not self.app.initializer.stats_aggregator
            or not self.app.initializer.optimizer
        ):
            logger.error(
                "[AppWorkerManager] ERROR: Cannot start workers, core RL components missing."
            )
            self.app.app_state = self.app.app_state.ERROR
            self.app.status = "Worker Init Failed: Missing Components"
            return

        # Check if worker instances exist in the initializer
        if (
            not self.app.initializer.self_play_workers
            or not self.app.initializer.training_worker
        ):
            logger.error(
                "[AppWorkerManager] ERROR: Workers not initialized in AppInitializer."
            )
            self.app.app_state = self.app.app_state.ERROR
            self.app.status = "Worker Init Failed: Not Initialized"
            return

        logger.info("[AppWorkerManager] Starting all worker threads...")
        self.app.stop_event.clear()  # Ensure stop event is clear

        # --- Start Self-Play Workers ---
        self.self_play_worker_threads = []  # Clear list of active threads
        for i, worker_instance in enumerate(self.app.initializer.self_play_workers):
            if worker_instance:
                # Check if the thread associated with this instance is alive
                if not worker_instance.is_alive():
                    try:
                        # Recreate the thread object using the instance's init args
                        # This ensures we have a fresh thread if the previous one finished/crashed
                        recreated_worker = SelfPlayWorker(
                            **worker_instance.get_init_args()
                        )
                        # Replace the instance in the initializer list (important for rendering)
                        self.app.initializer.self_play_workers[i] = recreated_worker
                        worker_to_start = recreated_worker
                        logger.info(f"  Recreated SelfPlayWorker-{i}.")
                    except Exception as e:
                        logger.error(f"  ERROR recreating SelfPlayWorker-{i}: {e}")
                        continue  # Skip starting this worker
                else:
                    # If already alive (shouldn't happen based on initial check, but safe)
                    worker_to_start = worker_instance
                    logger.warning(
                        f"  SelfPlayWorker-{i} was already alive during start sequence."
                    )

                # Add to our list of *running* threads and start
                self.self_play_worker_threads.append(worker_to_start)
                worker_to_start.start()
                logger.info(f"  SelfPlayWorker-{i} thread started.")
            else:
                logger.error(
                    f"[AppWorkerManager] ERROR: SelfPlayWorker instance {i} is None during start."
                )

        # --- Start Training Worker ---
        training_instance = self.app.initializer.training_worker
        if training_instance:
            if not training_instance.is_alive():
                try:
                    # Recreate thread object if needed
                    recreated_worker = TrainingWorker(
                        **training_instance.get_init_args()
                    )
                    self.app.initializer.training_worker = (
                        recreated_worker  # Update initializer's instance
                    )
                    self.training_worker_thread = (
                        recreated_worker  # Update manager's running thread ref
                    )
                    logger.info("  Recreated TrainingWorker.")
                except Exception as e:
                    logger.error(f"  ERROR recreating TrainingWorker: {e}")
                    self.training_worker_thread = None  # Failed to recreate
            else:
                self.training_worker_thread = (
                    training_instance  # Use existing live thread instance
                )
                logger.warning(
                    "  TrainingWorker was already alive during start sequence."
                )

            if self.training_worker_thread:
                # Start the thread (safe to call start() again on already started thread)
                self.training_worker_thread.start()
                logger.info("  TrainingWorker thread started.")
        else:
            logger.error(
                "[AppWorkerManager] ERROR: TrainingWorker instance is None during start."
            )

        # Final status update
        if self.is_any_worker_running():
            self.app.status = "Running AlphaZero"
            num_sp = len(self.self_play_worker_threads)
            num_tr = 1 if self.is_training_running() else 0
            logger.info(
                f"[AppWorkerManager] Workers started ({num_sp} SP, {num_tr} TR)."
            )

    def stop_all_workers(self, join_timeout: float = 5.0):
        """Signals ALL worker threads to stop and waits for them to join."""
        # Check if there's anything to stop
        worker_instances_exist = (
            self.app.initializer.self_play_workers
            or self.app.initializer.training_worker
        )
        if not self.is_any_worker_running() and not worker_instances_exist:
            logger.info("[AppWorkerManager] No workers initialized or running to stop.")
            return
        elif not self.is_any_worker_running():
            logger.info("[AppWorkerManager] No workers currently running to stop.")
            # Proceed to clear queue etc. even if not running
        else:
            logger.info("[AppWorkerManager] Stopping ALL worker threads...")
            self.app.stop_event.set()  # Signal threads to stop

        # Collect threads that need joining (use initializer instances as the source of truth)
        threads_to_join: List[Tuple[str, threading.Thread]] = []
        for i, worker in enumerate(self.app.initializer.self_play_workers):
            # Check if the instance exists and the thread is alive
            if worker and worker.is_alive():
                threads_to_join.append((f"SelfPlayWorker-{i}", worker))

        if (
            self.app.initializer.training_worker
            and self.app.initializer.training_worker.is_alive()
        ):
            threads_to_join.append(
                ("TrainingWorker", self.app.initializer.training_worker)
            )

        # Join threads with timeout
        start_join_time = time.time()
        if not threads_to_join:
            logger.info("[AppWorkerManager] No active threads found to join.")
        else:
            logger.info(
                f"[AppWorkerManager] Attempting to join {len(threads_to_join)} threads..."
            )
            for name, thread in threads_to_join:
                # Calculate remaining timeout dynamically
                elapsed_time = time.time() - start_join_time
                remaining_timeout = max(
                    0.1, join_timeout - elapsed_time
                )  # Ensure minimum timeout
                logger.info(
                    f"[AppWorkerManager] Joining {name} (timeout: {remaining_timeout:.1f}s)..."
                )
                thread.join(timeout=remaining_timeout)
                if thread.is_alive():
                    logger.warning(
                        f"[AppWorkerManager] WARNING: {name} thread did not join cleanly after timeout."
                    )
                else:
                    logger.info(f"[AppWorkerManager] {name} joined.")

        # Clear internal references to running threads
        self.self_play_worker_threads = []
        self.training_worker_thread = None

        # Clear the experience queue after stopping workers
        logger.info("[AppWorkerManager] Clearing experience queue...")
        cleared_count = 0
        while not self.app.experience_queue.empty():
            try:
                self.app.experience_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error clearing queue item: {e}")
                break  # Stop clearing on error
        logger.info(
            f"[AppWorkerManager] Cleared {cleared_count} items from experience queue."
        )

        logger.info("[AppWorkerManager] All worker threads stopped.")
        self.app.status = "Ready"  # Update status after stopping


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
from typing import TextIO, Optional


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
from typing import Optional, Dict, Any, List
import queue
import numpy as np
from collections import deque

script_dir = os.path.dirname(os.path.abspath(__file__))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# --- Config and Utils Imports ---
try:
    from config import (
        VisConfig,
        EnvConfig,
        TrainConfig,
        MCTSConfig,
        RANDOM_SEED,
        BASE_CHECKPOINT_DIR,
        set_device,
        get_run_id,
        set_run_id,
        get_run_log_dir,
        get_console_log_dir,
        get_config_dict,
    )
    from utils.helpers import get_device as get_torch_device, set_random_seeds
    from logger import TeeLogger
    from utils.init_checks import run_pre_checks
except ImportError as e:
    print(f"Error importing config/utils: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# --- App Component Imports ---
try:
    from environment.game_state import GameState
    from ui.renderer import UIRenderer
    from stats import StatsAggregator
    from training.checkpoint_manager import (
        find_latest_run_and_checkpoint,
    )
    from app_state import AppState
    from app_init import AppInitializer
    from app_logic import AppLogic
    from app_workers import AppWorkerManager
    from app_setup import (
        initialize_pygame,
        initialize_directories,
    )
    from app_ui_utils import AppUIUtils
    from ui.input_handler import InputHandler
    from agent.alphazero_net import AlphaZeroNet
    from workers.training_worker import ProcessedExperienceBatch
except ImportError as e:
    print(f"Error importing app components: {e}\n{traceback.format_exc()}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
# Reduce default logging noise
logging.getLogger("mcts").setLevel(logging.WARNING)
logging.getLogger("workers.self_play_worker").setLevel(logging.INFO)
logging.getLogger("workers.training_worker").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)  # Main app logger

# --- Constants ---
LOOP_TIMING_INTERVAL = 60


class MainApp:
    """Main application class orchestrating Pygame UI and AlphaZero components."""

    def __init__(self, checkpoint_to_load: Optional[str] = None):
        # Config Instances
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        self.train_config_instance = TrainConfig()
        self.mcts_config = MCTSConfig()

        # Core Components
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None

        # State
        self.app_state: AppState = AppState.INITIALIZING
        self.status: str = "Initializing..."
        self.running: bool = True
        self.update_progress_details: Dict[str, Any] = {}

        # Threading & Communication
        self.stop_event = threading.Event()
        self.experience_queue: queue.Queue[ProcessedExperienceBatch] = queue.Queue(
            maxsize=self.train_config_instance.BUFFER_CAPACITY
        )
        logger.info(
            f"Experience Queue Size: {self.train_config_instance.BUFFER_CAPACITY}"
        )

        # RL Components (Managed by Initializer)
        self.agent: Optional[AlphaZeroNet] = None
        self.stats_aggregator: Optional[StatsAggregator] = None
        self.demo_env: Optional[GameState] = None

        # Helper Classes
        self.device = get_torch_device()
        set_device(self.device)
        self.checkpoint_to_load = checkpoint_to_load
        self.initializer = AppInitializer(self)
        self.logic = AppLogic(self)
        self.worker_manager = AppWorkerManager(self)
        self.ui_utils = AppUIUtils(self)

        # UI State
        self.cleanup_confirmation_active: bool = False
        self.cleanup_message: str = ""
        self.last_cleanup_message_time: float = 0.0
        self.total_gpu_memory_bytes: Optional[int] = None

        # Timing
        self.frame_count = 0
        self.loop_times = deque(maxlen=LOOP_TIMING_INTERVAL)

    def initialize(self):
        """Initializes Pygame, directories, configs, and core components."""
        logger.info("--- Application Initialization ---")
        self.screen, self.clock = initialize_pygame(self.vis_config)
        initialize_directories()
        set_random_seeds(RANDOM_SEED)
        run_pre_checks()

        self.app_state = AppState.INITIALIZING
        self.initializer.initialize_all()

        self.agent = self.initializer.agent
        self.stats_aggregator = self.initializer.stats_aggregator
        self.demo_env = self.initializer.demo_env

        if self.renderer and self.input_handler:
            self.renderer.set_input_handler(self.input_handler)
            if hasattr(self.input_handler, "app_ref"):
                self.input_handler.app_ref = self

        self.logic.check_initial_completion_status()
        self.status = "Ready"
        self.app_state = AppState.MAIN_MENU
        logger.info("--- Initialization Complete ---")

    def _handle_input(self) -> bool:
        """Handles user input using the InputHandler."""
        if self.input_handler:
            return self.input_handler.handle_input(
                self.app_state.value, self.cleanup_confirmation_active
            )
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.logic.exit_app()
                    return False
            return True

    def _update_state(self):
        """Updates application logic and status."""
        self.update_progress_details = {}
        self.logic.update_status_and_check_completion()

        # Update demo env timers if in demo/debug mode
        if self.app_state in [AppState.PLAYING, AppState.DEBUG] and self.demo_env:
            try:
                self.demo_env._update_timers()
            except Exception as e:
                logger.error(f"Error updating demo env timers: {e}")

    def _prepare_render_data(self) -> Dict[str, Any]:
        """Gathers all necessary data for rendering the current frame."""
        render_data = {
            "app_state": self.app_state.value,
            "status": self.status,
            "cleanup_confirmation_active": self.cleanup_confirmation_active,
            "cleanup_message": self.cleanup_message,
            "last_cleanup_message_time": self.last_cleanup_message_time,
            "update_progress_details": self.update_progress_details,
            "demo_env": self.demo_env,  # Pass the demo env object itself
            "env_config": self.env_config,
            "num_envs": self.train_config_instance.NUM_SELF_PLAY_WORKERS,
            "plot_data": {},
            "stats_summary": {},
            "best_game_state_data": None,
            "agent_param_count": 0,
            "worker_counts": {},
            "is_process_running": False,
            "worker_render_data": [],
        }

        if self.stats_aggregator:
            render_data["plot_data"] = self.stats_aggregator.get_plot_data()
            current_step = self.stats_aggregator.storage.current_global_step
            render_data["stats_summary"] = self.stats_aggregator.get_summary(
                current_step
            )
            render_data["best_game_state_data"] = (
                self.stats_aggregator.get_best_game_state_data()
            )

        if self.initializer:
            render_data["agent_param_count"] = self.initializer.agent_param_count

        if self.worker_manager:
            render_data["worker_counts"] = (
                self.worker_manager.get_active_worker_counts()
            )
            render_data["is_process_running"] = (
                self.worker_manager.is_any_worker_running()
            )
            if (
                render_data["is_process_running"]
                and self.app_state == AppState.MAIN_MENU
            ):
                num_to_render = self.vis_config.NUM_ENVS_TO_RENDER
                if num_to_render > 0:
                    render_data["worker_render_data"] = (
                        self.worker_manager.get_worker_render_data(num_to_render)
                    )

        return render_data

    def _render_frame(self, render_data: Dict[str, Any]):
        """Renders the UI frame using the collected data."""
        if self.renderer:
            self.renderer.render_all(**render_data)
        else:
            try:
                self.screen.fill((20, 0, 0))
                font = pygame.font.Font(None, 30)
                text_surf = font.render(
                    "Renderer Initialization Failed!", True, (255, 50, 50)
                )
                self.screen.blit(
                    text_surf, text_surf.get_rect(center=self.screen.get_rect().center)
                )
                pygame.display.flip()
            except Exception:
                pass

    def _log_loop_timing(self, loop_start_time: float):
        """Logs average loop time periodically."""
        self.loop_times.append(time.monotonic() - loop_start_time)
        self.frame_count += 1

    def run_main_loop(self):
        """The main application loop."""
        logger.info("Starting main application loop...")
        while self.running:
            loop_start_time = time.monotonic()

            if not self.clock or not self.screen:
                logger.error("Pygame clock or screen not initialized. Exiting.")
                break

            dt = self.clock.tick(self.vis_config.FPS) / 1000.0

            self.running = self._handle_input()
            if not self.running:
                break

            self._update_state()  # Updates demo timers if needed
            render_data = self._prepare_render_data()
            self._render_frame(render_data)

            self._log_loop_timing(loop_start_time)

        logger.info("Main application loop exited.")

    def shutdown(self):
        """Cleans up resources and exits."""
        logger.info("Initiating shutdown sequence...")
        logger.info("Stopping worker threads...")
        self.worker_manager.stop_all_workers()
        logger.info("Worker threads stopped.")
        logger.info("Attempting final checkpoint save...")
        self.logic.save_final_checkpoint()
        logger.info("Final checkpoint save attempt finished.")
        logger.info("Closing stats recorder (before pygame.quit)...")
        self.initializer.close_stats_recorder()
        logger.info("Stats recorder closed.")
        logger.info("Quitting Pygame...")
        pygame.quit()
        logger.info("Pygame quit.")
        logger.info("Shutdown complete.")


# --- Main Execution ---
tee_logger_instance: Optional[TeeLogger] = None


def setup_logging_and_run_id(args: argparse.Namespace):
    """Sets up logging (including TeeLogger) and determines the run ID."""
    global tee_logger_instance
    run_id_source = "New"

    if args.load_checkpoint:
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                run_id_source = f"Explicit Checkpoint ({get_run_id()})"
            else:
                get_run_id()
                run_id_source = (
                    f"New (Explicit Ckpt Path Invalid: {args.load_checkpoint})"
                )
        except Exception as e:
            logger.warning(
                f"Could not determine run_id from checkpoint path '{args.load_checkpoint}': {e}. Generating new."
            )
            get_run_id()
            run_id_source = f"New (Error parsing ckpt path)"
    else:
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            run_id_source = f"Resumed Latest ({get_run_id()})"
        else:
            get_run_id()
            run_id_source = f"New (No previous runs found)"

    current_run_id = get_run_id()
    print(f"Run ID: {current_run_id} (Source: {run_id_source})")

    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        log_file_dir = get_run_log_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, "console_output.log")
        tee_logger_instance = TeeLogger(log_file_path, sys.stdout)
        sys.stdout = tee_logger_instance
        sys.stderr = tee_logger_instance
        print(f"Console output will be mirrored to: {log_file_path}")
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)

    # Configure Python's logging system
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    # Set default levels for noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    # Set default levels for our workers unless overridden by args.log_level
    if log_level <= logging.INFO:
        logging.getLogger("mcts").setLevel(logging.WARNING)  # Keep MCTS quieter
        logging.getLogger("workers.self_play_worker").setLevel(logging.INFO)
        logging.getLogger("workers.training_worker").setLevel(logging.INFO)
    # If DEBUG is requested, set workers/MCTS to DEBUG too
    if log_level == logging.DEBUG:
        logging.getLogger("mcts").setLevel(logging.DEBUG)
        logging.getLogger("workers.self_play_worker").setLevel(logging.DEBUG)
        logging.getLogger("workers.training_worker").setLevel(logging.DEBUG)

    logger.info(f"Logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {current_run_id}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")

    return original_stdout, original_stderr


def cleanup_logging(original_stdout, original_stderr, exit_code):
    """Restores standard output/error and closes logger."""
    print("[Main Finally] Restoring stdout/stderr and closing logger...")
    logger = logging.getLogger(__name__)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Trainer")
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint file (.pth) to load.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    original_stdout, original_stderr = setup_logging_and_run_id(args)

    app = None
    exit_code = 0
    try:
        app = MainApp(checkpoint_to_load=args.load_checkpoint)
        app.initialize()
        app.run_main_loop()

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Shutting down gracefully...")
        if app:
            app.logic.exit_app()
        else:
            pygame.quit()
        exit_code = 130

    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        if app:
            app.logic.exit_app()
        else:
            pygame.quit()
        exit_code = 1

    finally:
        if app:
            app.shutdown()
        cleanup_logging(original_stdout, original_stderr, exit_code)


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
import pygame
import time
import traceback
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # Import scheduler
from typing import TYPE_CHECKING, List, Optional

from config import (
    ModelConfig,
    StatsConfig,
    DemoConfig,
    MCTSConfig,
    BASE_CHECKPOINT_DIR,
    get_run_checkpoint_dir,
)
from environment.game_state import GameState
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregator
from stats.simple_stats_recorder import SimpleStatsRecorder
from ui.renderer import UIRenderer
from ui.input_handler import InputHandler
from training.checkpoint_manager import CheckpointManager
from app_state import AppState
from mcts import MCTS
from agent.alphazero_net import AlphaZeroNet
from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker

if TYPE_CHECKING:
    from main_pygame import MainApp
    from torch.optim.lr_scheduler import _LRScheduler


class AppInitializer:
    """Handles the initialization of core application components."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Config instances
        self.vis_config = app.vis_config
        self.env_config = app.env_config
        self.train_config = app.train_config_instance
        self.model_config = ModelConfig()
        self.stats_config = StatsConfig()
        self.demo_config = DemoConfig()
        self.mcts_config = MCTSConfig()

        # Components to be initialized
        self.agent: Optional[AlphaZeroNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional["_LRScheduler"] = None  # Add scheduler attribute
        self.stats_recorder: Optional[StatsRecorderBase] = None
        self.stats_aggregator: Optional[StatsAggregator] = None
        self.demo_env: Optional[GameState] = None
        self.agent_param_count: int = 0
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.mcts: Optional[MCTS] = None
        self.self_play_workers: List[SelfPlayWorker] = []
        self.training_worker: Optional[TrainingWorker] = None

    def initialize_all(self, is_reinit: bool = False):
        """Initializes all core components."""
        try:
            self._check_gpu_memory()
            if not is_reinit:
                self._initialize_ui_early()

            self.initialize_rl_components(
                is_reinit=is_reinit, checkpoint_to_load=self.app.checkpoint_to_load
            )

            if not is_reinit:
                self.initialize_demo_env()
                self.initialize_input_handler()

            self._calculate_agent_params()
            self.initialize_workers()  # Workers now need the scheduler

        except Exception as init_err:
            self._handle_init_error(init_err)

    def _check_gpu_memory(self):
        """Checks and prints total GPU memory if available."""
        if self.app.device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(self.app.device)
                self.app.total_gpu_memory_bytes = props.total_memory
                print(f"Total GPU Memory: {props.total_memory / (1024**3):.2f} GB")
            except Exception as e:
                print(f"Warning: Could not get total GPU memory: {e}")

    def _initialize_ui_early(self):
        """Initializes the renderer and performs an initial render."""
        self.app.renderer = UIRenderer(self.app.screen, self.vis_config)
        self.app.renderer.render_all(
            app_state=self.app.app_state.value,
            is_process_running=False,
            status=self.app.status,
            stats_summary={},
            envs=[],
            num_envs=0,
            env_config=self.env_config,
            cleanup_confirmation_active=False,
            cleanup_message="",
            last_cleanup_message_time=0,
            plot_data={},
            demo_env=None,
            update_progress_details={},
            agent_param_count=0,
            worker_counts={},
            best_game_state_data=None,
        )
        pygame.display.flip()
        pygame.time.delay(100)  # Allow UI to update

    def _calculate_agent_params(self):
        """Calculates the number of trainable parameters in the agent."""
        if self.agent:
            try:
                self.agent_param_count = sum(
                    p.numel() for p in self.agent.parameters() if p.requires_grad
                )
            except Exception as e:
                print(f"Warning: Could not calculate agent parameters: {e}")
                self.agent_param_count = 0

    def _handle_init_error(self, error: Exception):
        """Handles fatal errors during initialization."""
        print(f"FATAL ERROR during component initialization: {error}")
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
        """Initializes NN Agent, Optimizer, Scheduler, MCTS, Stats, Checkpoint Manager."""
        print(f"Initializing AlphaZero components... Re-init: {is_reinit}")
        start_time = time.time()
        try:
            self._init_agent()
            self._init_optimizer_and_scheduler()  # Renamed method
            self._init_mcts()
            self._init_stats()
            self._init_checkpoint_manager(
                checkpoint_to_load
            )  # Checkpoint manager needs scheduler now
            print(
                f"AlphaZero components initialized in {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            print(f"Error during AlphaZero component initialization: {e}")
            traceback.print_exc()
            raise e

    def _init_agent(self):
        self.agent = AlphaZeroNet(
            env_config=self.env_config, model_config=self.model_config.Network()
        ).to(self.app.device)
        print(f"AlphaZeroNet initialized on device: {self.app.device}.")

    def _init_optimizer_and_scheduler(self):
        """Initializes the optimizer and the learning rate scheduler."""
        if not self.agent:
            raise RuntimeError("Agent must be initialized before Optimizer.")
        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.train_config.LEARNING_RATE,
            weight_decay=self.train_config.WEIGHT_DECAY,
        )
        print(f"Optimizer initialized (Adam, LR={self.train_config.LEARNING_RATE}).")

        # Initialize Scheduler (if enabled)
        if self.train_config.USE_LR_SCHEDULER:
            if self.train_config.SCHEDULER_TYPE == "CosineAnnealingLR":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.train_config.SCHEDULER_T_MAX,
                    eta_min=self.train_config.SCHEDULER_ETA_MIN,
                )
                print(
                    f"LR Scheduler initialized (CosineAnnealingLR, T_max={self.train_config.SCHEDULER_T_MAX}, eta_min={self.train_config.SCHEDULER_ETA_MIN})."
                )
            # Add other scheduler types here if needed
            # elif self.train_config.SCHEDULER_TYPE == "OneCycleLR":
            #     self.scheduler = OneCycleLR(...)
            else:
                print(
                    f"Warning: Unknown scheduler type '{self.train_config.SCHEDULER_TYPE}'. No scheduler initialized."
                )
                self.scheduler = None
        else:
            print("LR Scheduler is DISABLED.")
            self.scheduler = None

    def _init_mcts(self):
        if not self.agent:
            raise RuntimeError("Agent must be initialized before MCTS.")
        self.mcts = MCTS(
            network_predictor=self.agent.predict,
            config=self.mcts_config,
            env_config=self.env_config,
        )
        print("MCTS initialized with AlphaZeroNet predictor.")

    def _init_stats(self):
        print("Initializing StatsAggregator and SimpleStatsRecorder...")
        self.stats_aggregator = StatsAggregator(
            avg_windows=self.stats_config.STATS_AVG_WINDOW,
            plot_window=self.stats_config.PLOT_DATA_WINDOW,
        )
        self.stats_recorder = SimpleStatsRecorder(
            aggregator=self.stats_aggregator,
            console_log_interval=self.stats_config.CONSOLE_LOG_FREQ,
            train_config=self.train_config,
        )
        print("StatsAggregator and SimpleStatsRecorder initialized.")

    def _init_checkpoint_manager(self, checkpoint_to_load: Optional[str]):
        if not self.agent or not self.optimizer or not self.stats_aggregator:
            raise RuntimeError(
                "Agent, Optimizer, StatsAggregator needed for CheckpointManager."
            )
        # Pass the scheduler to the CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            agent=self.agent,
            optimizer=self.optimizer,
            scheduler=self.scheduler,  # Pass scheduler
            stats_aggregator=self.stats_aggregator,
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            run_checkpoint_dir=get_run_checkpoint_dir(),
            load_checkpoint_path_config=checkpoint_to_load,
            device=self.app.device,
        )
        if self.checkpoint_manager.get_checkpoint_path_to_load():
            self.checkpoint_manager.load_checkpoint()

    def initialize_workers(self):
        """Initializes worker threads (Self-Play, Training). Does NOT start them."""
        print("Initializing worker threads...")
        if (
            not self.agent
            or not self.mcts
            or not self.stats_aggregator
            or not self.optimizer
            # Scheduler is optional, so don't check it here
        ):
            print("ERROR: Cannot initialize workers, core RL components missing.")
            return

        self._init_self_play_workers()
        self._init_training_worker()  # Training worker needs scheduler
        num_sp = len(self.self_play_workers)
        print(f"Worker threads initialized ({num_sp} Self-Play, 1 Training).")

    def _init_self_play_workers(self):
        self.self_play_workers = []
        num_sp_workers = self.train_config.NUM_SELF_PLAY_WORKERS
        print(f"Initializing {num_sp_workers} SelfPlayWorker(s)...")
        for i in range(num_sp_workers):
            worker = SelfPlayWorker(
                worker_id=i,
                agent=self.agent,
                mcts=self.mcts,
                experience_queue=self.app.experience_queue,
                stats_aggregator=self.stats_aggregator,
                stop_event=self.app.stop_event,
                env_config=self.env_config,
                mcts_config=self.mcts_config,
                device=self.app.device,
            )
            self.self_play_workers.append(worker)
            print(f"  SelfPlayWorker-{i} initialized.")

    def _init_training_worker(self):
        # Pass the scheduler to the TrainingWorker
        self.training_worker = TrainingWorker(
            agent=self.agent,
            optimizer=self.optimizer,
            scheduler=self.scheduler,  # Pass scheduler
            experience_queue=self.app.experience_queue,
            stats_aggregator=self.stats_aggregator,
            stop_event=self.app.stop_event,
            train_config=self.train_config,
            device=self.app.device,
        )
        print("TrainingWorker initialized.")

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
            start_run_cb=self.app.logic.start_run,
            stop_run_cb=self.app.logic.stop_run,
        )
        if self.app.input_handler:
            self.app.input_handler.app_ref = self.app
        if self.app.renderer and self.app.renderer.left_panel:
            self.app.renderer.left_panel.input_handler = self.app.input_handler
            if hasattr(self.app.renderer.left_panel, "button_status_renderer"):
                btn_renderer = self.app.renderer.left_panel.button_status_renderer
                btn_renderer.input_handler_ref = self.app.input_handler
                btn_renderer.app_ref = self.app

    def close_stats_recorder(self, is_cleanup: bool = False):
        """Safely closes the stats recorder (if it exists)."""
        print(
            f"[AppInitializer] close_stats_recorder called (is_cleanup={is_cleanup})..."
        )
        if self.stats_recorder and hasattr(self.stats_recorder, "close"):
            print("[AppInitializer] Stats recorder exists, attempting close...")
            try:
                self.stats_recorder.close(is_cleanup=is_cleanup)
                print("[AppInitializer] stats_recorder.close() executed.")
            except Exception as log_e:
                print(f"[AppInitializer] Error closing stats recorder on exit: {log_e}")
                traceback.print_exc()
        else:
            print("[AppInitializer] No stats recorder instance or close method.")
        print("[AppInitializer] close_stats_recorder finished.")


File: app_logic.py
import pygame
import time
import traceback
import os
import shutil
from typing import TYPE_CHECKING, Tuple

from app_state import AppState
from config.general import get_run_checkpoint_dir 

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppLogic:
    """Handles the core application logic and state transitions."""

    def __init__(self, app: "MainApp"):
        self.app = app

    def check_initial_completion_status(self):
        """Checks if training target was met upon loading (placeholder)."""
        pass 

    def update_status_and_check_completion(self):
        """Updates the status text based on application state."""
        is_running = self.app.worker_manager.is_any_worker_running()
        state = self.app.app_state
        if state == AppState.MAIN_MENU:
            self.app.status = (
                "Confirm Cleanup"
                if self.app.cleanup_confirmation_active
                else "Running AlphaZero" if is_running else "Ready"
            )
        elif state == AppState.PLAYING:
            self.app.status = "Playing Demo"
        elif state == AppState.DEBUG:
            self.app.status = "Debugging Grid"
        elif state == AppState.INITIALIZING:
            self.app.status = "Initializing..."

    # --- Worker Control ---
    def start_run(self):
        """Starts both self-play and training workers."""
        if (
            self.app.app_state != AppState.MAIN_MENU
            or self.app.worker_manager.is_any_worker_running()
        ):
            print("Cannot start run: Not in Main Menu or already running.")
            return
        print("Starting AlphaZero Run (Self-Play & Training)...")
        self.app.worker_manager.start_all_workers()
        self.update_status_and_check_completion()

    def stop_run(self):
        """Stops both self-play and training workers."""
        if not self.app.worker_manager.is_any_worker_running():
            print("Run not currently active.")
            return
        print("Stopping AlphaZero Run...")
        self.app.worker_manager.stop_all_workers()
        self.update_status_and_check_completion()

    # --- Mode Transitions & Cleanup ---
    def request_cleanup(self):
        if self.app.app_state != AppState.MAIN_MENU:
            return
        if self.app.worker_manager.is_any_worker_running():
            self._set_temp_message("Stop Run before Cleanup!")
            return
        self.app.cleanup_confirmation_active = True
        self.app.status = "Confirm Cleanup"
        print("Cleanup requested. Confirm action.")

    def start_demo_mode(self):
        if self._can_start_mode("Demo"):
            print("Entering Demo Mode...")
            self.try_save_checkpoint()
            self.app.app_state = AppState.PLAYING
            self.app.status = "Playing Demo"
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()

    def start_debug_mode(self):
        if self._can_start_mode("Debug"):
            print("Entering Debug Mode...")
            self.try_save_checkpoint()
            self.app.app_state = AppState.DEBUG
            self.app.status = "Debugging Grid"
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()

    def _can_start_mode(self, mode_name: str) -> bool:
        """Checks if demo/debug mode can be started."""
        if self.app.initializer.demo_env is None:
            print(f"Cannot start {mode_name}: Env not initialized.")
            return False
        if self.app.app_state != AppState.MAIN_MENU:
            print(f"Cannot start {mode_name} mode outside MainMenu.")
            return False
        if self.app.worker_manager.is_any_worker_running():
            self._set_temp_message(f"Stop Run before {mode_name}!")
            return False
        return True

    def exit_demo_mode(self):
        if self.app.app_state == AppState.PLAYING:
            print("Exiting Demo Mode...")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.deselect_dragged_shape()
            self._return_to_main_menu()

    def exit_debug_mode(self):
        if self.app.app_state == AppState.DEBUG:
            print("Exiting Debug Mode...")
            self._return_to_main_menu()

    def _return_to_main_menu(self):
        """Helper to transition back to the main menu state."""
        self.app.app_state = AppState.MAIN_MENU
        self.check_initial_completion_status()
        self.update_status_and_check_completion()

    def cancel_cleanup(self):
        self.app.cleanup_confirmation_active = False
        self._set_temp_message("Cleanup cancelled.")
        self.update_status_and_check_completion()
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
            print(
                f"Cleanup process finished. State: {self.app.app_state}, Status: {self.app.status}"
            )

    def exit_app(self) -> bool:
        print("Exit requested.")
        self.app.stop_event.set()
        self.app.worker_manager.stop_all_workers()
        return False  # Signal main loop to stop

    # --- Input Handling Callbacks ---
    def handle_demo_mouse_motion(self, mouse_pos: Tuple[int, int]):
        if self.app.app_state != AppState.PLAYING or not self.app.initializer.demo_env:
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over():
            return
        if demo_env.demo_dragged_shape_idx is None:
            return
        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        demo_env.update_snapped_position(grid_coords)

    def handle_demo_mouse_button_down(self, event: pygame.event.Event):
        if self.app.app_state != AppState.PLAYING or not self.app.initializer.demo_env:
            return
        demo_env = self.app.initializer.demo_env
        if demo_env.is_frozen() or demo_env.is_over() or event.button != 1:
            return

        mouse_pos = event.pos
        clicked_preview = self.app.ui_utils.map_screen_to_preview(mouse_pos)
        if clicked_preview is not None:
            action = (
                demo_env.deselect_dragged_shape
                if clicked_preview == demo_env.demo_dragged_shape_idx
                else lambda: demo_env.select_shape_for_drag(clicked_preview)
            )
            action()
            return

        grid_coords = self.app.ui_utils.map_screen_to_grid(mouse_pos)
        if (
            grid_coords is not None
            and demo_env.demo_dragged_shape_idx is not None
            and demo_env.demo_snapped_position == grid_coords
        ):
            placed = demo_env.place_dragged_shape()
            if placed and demo_env.is_over():
                print("[Demo] Game Over! Press ESC to exit.")
        else:
            demo_env.deselect_dragged_shape()

    def handle_debug_input(self, event: pygame.event.Event):
        if self.app.app_state != AppState.DEBUG or not self.app.initializer.demo_env:
            return
        demo_env = self.app.initializer.demo_env
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            print("[Debug] Resetting grid...")
            demo_env.reset()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            clicked_coords = self.app.ui_utils.map_screen_to_grid(event.pos)
            if clicked_coords:
                demo_env.toggle_triangle_debug(*clicked_coords)

    # --- Internal Helpers ---
    def _set_temp_message(self, message: str):
        """Sets a temporary message to be displayed."""
        self.app.cleanup_message = message
        self.app.last_cleanup_message_time = time.time()

    def _cleanup_data(self):
        """Deletes current run's checkpoint and re-initializes components."""
        print("\n--- CLEANUP DATA INITIATED (Current Run Only) ---")
        self.app.app_state = AppState.INITIALIZING
        self.app.status = "Cleaning"
        messages = []
        self._render_during_cleanup()

        print("[Cleanup] Stopping existing worker threads (if any)...")
        self.app.worker_manager.stop_all_workers()
        print("[Cleanup] Existing worker threads stopped.")
        print("[Cleanup] Closing stats recorder...")
        self.app.initializer.close_stats_recorder(is_cleanup=True)
        print("[Cleanup] Stats recorder closed.")

        messages.append(self._delete_checkpoint_dir())
        time.sleep(0.1)

        print("[Cleanup] Re-initializing components...")
        try:
            self.app.initializer.initialize_rl_components(
                is_reinit=True, checkpoint_to_load=None
            )
            print("[Cleanup] Components re-initialized.")
            if self.app.initializer.demo_env:
                self.app.initializer.demo_env.reset()
            self.app.initializer.initialize_workers()
            print("[Cleanup] Workers re-initialized (not started).")
            messages.append("Components re-initialized.")
            self.app.status = "Ready"
            self.app.app_state = AppState.MAIN_MENU
        except Exception as e:
            print(f"FATAL ERROR during re-initialization after cleanup: {e}")
            traceback.print_exc()
            self.app.status = "Error: Re-init Failed"
            self.app.app_state = AppState.ERROR
            messages.append("ERROR RE-INITIALIZING COMPONENTS!")
            if self.app.renderer:
                self.app.renderer._render_error_screen(self.app.status)

        self._set_temp_message("\n".join(messages))
        print(f"--- CLEANUP DATA COMPLETE (Final State: {self.app.app_state}) ---")

    def _render_during_cleanup(self):
        """Renders the screen while cleanup is in progress."""
        if self.app.renderer:
            try:
                self.app.renderer.render_all(
                    app_state=self.app.app_state.value,
                    is_process_running=False,
                    status=self.app.status,
                    stats_summary={},
                    envs=[],
                    num_envs=0,
                    env_config=self.app.env_config,
                    cleanup_confirmation_active=False,
                    cleanup_message="",
                    last_cleanup_message_time=0,
                    plot_data={},
                    demo_env=self.app.initializer.demo_env,
                    update_progress_details={},
                    agent_param_count=getattr(
                        self.app.initializer, "agent_param_count", 0
                    ),
                    worker_counts={},
                    best_game_state_data=None,
                )
                pygame.display.flip()
                pygame.time.delay(100)
            except Exception as render_err:
                print(f"Warning: Error rendering during cleanup start: {render_err}")

    def _delete_checkpoint_dir(self) -> str:
        """Deletes the checkpoint directory and returns a status message."""
        print("[Cleanup] Deleting agent checkpoint file/dir...")
        msg = ""
        try:
            save_dir = get_run_checkpoint_dir()
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
                msg = f"Run checkpoint directory deleted: {save_dir}"
            else:
                msg = f"Run checkpoint directory not found: {save_dir}"
        except OSError as e:
            msg = f"Error deleting checkpoint dir: {e}"
        print(f"  - {msg}")
        print("[Cleanup] Checkpoint deletion attempt finished.")
        return msg

    def try_save_checkpoint(self):
        """Saves checkpoint if in main menu and workers are not running."""
        if (
            self.app.app_state != AppState.MAIN_MENU
            or self.app.worker_manager.is_any_worker_running()
        ):
            return
        if (
            not self.app.initializer.checkpoint_manager
            or not self.app.initializer.stats_aggregator
        ):
            return

        print("Saving checkpoint...")
        try:
            agg_storage = self.app.initializer.stats_aggregator.storage
            current_step = getattr(agg_storage, "current_global_step", 0)
            episode_count = getattr(agg_storage, "total_episodes", 0)
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
        """Saves the final checkpoint."""
        if (
            not self.app.initializer.checkpoint_manager
            or not self.app.initializer.stats_aggregator
        ):
            return
        save_on_exit = (
            self.app.status != "Cleaning" and self.app.app_state != AppState.ERROR
        )
        if save_on_exit:
            print("Performing final checkpoint save...")
            try:
                agg_storage = self.app.initializer.stats_aggregator.storage
                current_step = getattr(agg_storage, "current_global_step", 0)
                episode_count = getattr(agg_storage, "total_episodes", 0)
                target_step = getattr(
                    self.app.initializer.checkpoint_manager, "training_target_step", 0
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
from enum import Enum


class AppState(Enum):
    INITIALIZING = "Initializing"
    MAIN_MENU = "MainMenu"
    PLAYING = "Playing" 
    DEBUG = "Debug"
    CLEANUP_CONFIRM = "Confirm Cleanup"
    CLEANING = "Cleaning"
    ERROR = "Error"
    UNKNOWN = "Unknown"


File: app_ui_utils.py
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
from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Any
import logging

from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker
from environment.game_state import GameState

if TYPE_CHECKING:
    from main_pygame import MainApp

logger = logging.getLogger(__name__)


class AppWorkerManager:
    """Manages the creation, starting, and stopping of worker threads."""

    def __init__(self, app: "MainApp"):
        self.app = app
        self.self_play_worker_threads: List[SelfPlayWorker] = (
            []
        )  # Holds running thread objects
        self.training_worker_thread: Optional[TrainingWorker] = (
            None  # Holds running thread object
        )
        print("[AppWorkerManager] Initialized.")

    def get_active_worker_counts(self) -> Dict[str, int]:
        """Returns the count of currently active workers by type."""
        # Count based on living threads stored in this manager
        sp_count = sum(1 for w in self.self_play_worker_threads if w and w.is_alive())
        tr_count = 1 if self.is_training_running() else 0
        return {"SelfPlay": sp_count, "Training": tr_count}

    def is_self_play_running(self) -> bool:
        """Checks if *any* self-play worker thread is active."""
        return any(
            w is not None and w.is_alive() for w in self.self_play_worker_threads
        )

    def is_training_running(self) -> bool:
        """Checks if the training worker thread is active."""
        return (
            self.training_worker_thread is not None
            and self.training_worker_thread.is_alive()
        )

    def is_any_worker_running(self) -> bool:
        """Checks if any worker thread is active."""
        return self.is_self_play_running() or self.is_training_running()

    def get_worker_render_data(self, max_envs: int) -> List[Optional[Dict[str, Any]]]:
        """
        Retrieves render data (state copy and stats) from active self-play workers.
        Returns a list of dictionaries [{state: GameState, stats: Dict}, ... ] or None.
        Accesses worker instances directly from the initializer.
        """
        render_data_list: List[Optional[Dict[str, Any]]] = []
        count = 0
        # Get worker instances from the initializer, as they persist even if thread stops/restarts
        worker_instances = self.app.initializer.self_play_workers

        for worker in worker_instances:
            if count >= max_envs:
                break  # Limit reached

            # Check if the worker instance exists and is *currently running*
            if (
                worker
                and worker.is_alive()
                and hasattr(worker, "get_current_render_data")
            ):
                try:
                    # Fetch the combined state and stats dictionary
                    data: Dict[str, Any] = worker.get_current_render_data()
                    render_data_list.append(data)
                except Exception as e:
                    logger.error(
                        f"Error getting render data from worker {worker.worker_id}: {e}"
                    )
                    render_data_list.append(None)  # Append None on error
            else:
                # Append None if worker doesn't exist, isn't alive, or lacks method
                render_data_list.append(None)

            count += 1

        # Pad with None if fewer workers than max_envs
        while len(render_data_list) < max_envs:
            render_data_list.append(None)

        return render_data_list

    def start_all_workers(self):
        """Starts all initialized worker threads if they are not already running."""
        if self.is_any_worker_running():
            logger.warning(
                "[AppWorkerManager] Attempted to start workers, but some are already running."
            )
            return

        # Check if necessary components are initialized
        if (
            not self.app.initializer.agent
            or not self.app.initializer.mcts
            or not self.app.initializer.stats_aggregator
            or not self.app.initializer.optimizer
        ):
            logger.error(
                "[AppWorkerManager] ERROR: Cannot start workers, core RL components missing."
            )
            self.app.app_state = self.app.app_state.ERROR
            self.app.status = "Worker Init Failed: Missing Components"
            return

        # Check if worker instances exist in the initializer
        if (
            not self.app.initializer.self_play_workers
            or not self.app.initializer.training_worker
        ):
            logger.error(
                "[AppWorkerManager] ERROR: Workers not initialized in AppInitializer."
            )
            self.app.app_state = self.app.app_state.ERROR
            self.app.status = "Worker Init Failed: Not Initialized"
            return

        logger.info("[AppWorkerManager] Starting all worker threads...")
        self.app.stop_event.clear()  # Ensure stop event is clear

        # --- Start Self-Play Workers ---
        self.self_play_worker_threads = []  # Clear list of active threads
        for i, worker_instance in enumerate(self.app.initializer.self_play_workers):
            if worker_instance:
                # Check if the thread associated with this instance is alive
                if not worker_instance.is_alive():
                    try:
                        # Recreate the thread object using the instance's init args
                        # This ensures we have a fresh thread if the previous one finished/crashed
                        recreated_worker = SelfPlayWorker(
                            **worker_instance.get_init_args()
                        )
                        # Replace the instance in the initializer list (important for rendering)
                        self.app.initializer.self_play_workers[i] = recreated_worker
                        worker_to_start = recreated_worker
                        logger.info(f"  Recreated SelfPlayWorker-{i}.")
                    except Exception as e:
                        logger.error(f"  ERROR recreating SelfPlayWorker-{i}: {e}")
                        continue  # Skip starting this worker
                else:
                    # If already alive (shouldn't happen based on initial check, but safe)
                    worker_to_start = worker_instance
                    logger.warning(
                        f"  SelfPlayWorker-{i} was already alive during start sequence."
                    )

                # Add to our list of *running* threads and start
                self.self_play_worker_threads.append(worker_to_start)
                worker_to_start.start()
                logger.info(f"  SelfPlayWorker-{i} thread started.")
            else:
                logger.error(
                    f"[AppWorkerManager] ERROR: SelfPlayWorker instance {i} is None during start."
                )

        # --- Start Training Worker ---
        training_instance = self.app.initializer.training_worker
        if training_instance:
            if not training_instance.is_alive():
                try:
                    # Recreate thread object if needed
                    recreated_worker = TrainingWorker(
                        **training_instance.get_init_args()
                    )
                    self.app.initializer.training_worker = (
                        recreated_worker  # Update initializer's instance
                    )
                    self.training_worker_thread = (
                        recreated_worker  # Update manager's running thread ref
                    )
                    logger.info("  Recreated TrainingWorker.")
                except Exception as e:
                    logger.error(f"  ERROR recreating TrainingWorker: {e}")
                    self.training_worker_thread = None  # Failed to recreate
            else:
                self.training_worker_thread = (
                    training_instance  # Use existing live thread instance
                )
                logger.warning(
                    "  TrainingWorker was already alive during start sequence."
                )

            if self.training_worker_thread:
                # Start the thread (safe to call start() again on already started thread)
                self.training_worker_thread.start()
                logger.info("  TrainingWorker thread started.")
        else:
            logger.error(
                "[AppWorkerManager] ERROR: TrainingWorker instance is None during start."
            )

        # Final status update
        if self.is_any_worker_running():
            self.app.status = "Running AlphaZero"
            num_sp = len(self.self_play_worker_threads)
            num_tr = 1 if self.is_training_running() else 0
            logger.info(
                f"[AppWorkerManager] Workers started ({num_sp} SP, {num_tr} TR)."
            )

    def stop_all_workers(self, join_timeout: float = 5.0):
        """Signals ALL worker threads to stop and waits for them to join."""
        # Check if there's anything to stop
        worker_instances_exist = (
            self.app.initializer.self_play_workers
            or self.app.initializer.training_worker
        )
        if not self.is_any_worker_running() and not worker_instances_exist:
            logger.info("[AppWorkerManager] No workers initialized or running to stop.")
            return
        elif not self.is_any_worker_running():
            logger.info("[AppWorkerManager] No workers currently running to stop.")
            # Proceed to clear queue etc. even if not running
        else:
            logger.info("[AppWorkerManager] Stopping ALL worker threads...")
            self.app.stop_event.set()  # Signal threads to stop

        # Collect threads that need joining (use initializer instances as the source of truth)
        threads_to_join: List[Tuple[str, threading.Thread]] = []
        for i, worker in enumerate(self.app.initializer.self_play_workers):
            # Check if the instance exists and the thread is alive
            if worker and worker.is_alive():
                threads_to_join.append((f"SelfPlayWorker-{i}", worker))

        if (
            self.app.initializer.training_worker
            and self.app.initializer.training_worker.is_alive()
        ):
            threads_to_join.append(
                ("TrainingWorker", self.app.initializer.training_worker)
            )

        # Join threads with timeout
        start_join_time = time.time()
        if not threads_to_join:
            logger.info("[AppWorkerManager] No active threads found to join.")
        else:
            logger.info(
                f"[AppWorkerManager] Attempting to join {len(threads_to_join)} threads..."
            )
            for name, thread in threads_to_join:
                # Calculate remaining timeout dynamically
                elapsed_time = time.time() - start_join_time
                remaining_timeout = max(
                    0.1, join_timeout - elapsed_time
                )  # Ensure minimum timeout
                logger.info(
                    f"[AppWorkerManager] Joining {name} (timeout: {remaining_timeout:.1f}s)..."
                )
                thread.join(timeout=remaining_timeout)
                if thread.is_alive():
                    logger.warning(
                        f"[AppWorkerManager] WARNING: {name} thread did not join cleanly after timeout."
                    )
                else:
                    logger.info(f"[AppWorkerManager] {name} joined.")

        # Clear internal references to running threads
        self.self_play_worker_threads = []
        self.training_worker_thread = None

        # Clear the experience queue after stopping workers
        logger.info("[AppWorkerManager] Clearing experience queue...")
        cleared_count = 0
        while not self.app.experience_queue.empty():
            try:
                self.app.experience_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error clearing queue item: {e}")
                break  # Stop clearing on error
        logger.info(
            f"[AppWorkerManager] Cleared {cleared_count} items from experience queue."
        )

        logger.info("[AppWorkerManager] All worker threads stopped.")
        self.app.status = "Ready"  # Update status after stopping


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
from typing import TextIO, Optional


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
from typing import Optional, Dict, Any, List
import queue
import numpy as np
from collections import deque

script_dir = os.path.dirname(os.path.abspath(__file__))

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# --- Config and Utils Imports ---
try:
    from config import (
        VisConfig,
        EnvConfig,
        TrainConfig,
        MCTSConfig,
        RANDOM_SEED,
        BASE_CHECKPOINT_DIR,
        set_device,
        get_run_id,
        set_run_id,
        get_run_log_dir,
        get_console_log_dir,
        get_config_dict,
    )
    from utils.helpers import get_device as get_torch_device, set_random_seeds
    from logger import TeeLogger
    from utils.init_checks import run_pre_checks
except ImportError as e:
    print(f"Error importing config/utils: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# --- App Component Imports ---
try:
    from environment.game_state import GameState
    from ui.renderer import UIRenderer
    from stats import StatsAggregator
    from training.checkpoint_manager import (
        find_latest_run_and_checkpoint,
    )
    from app_state import AppState
    from app_init import AppInitializer
    from app_logic import AppLogic
    from app_workers import AppWorkerManager
    from app_setup import (
        initialize_pygame,
        initialize_directories,
    )
    from app_ui_utils import AppUIUtils
    from ui.input_handler import InputHandler
    from agent.alphazero_net import AlphaZeroNet
    from workers.training_worker import ProcessedExperienceBatch
except ImportError as e:
    print(f"Error importing app components: {e}\n{traceback.format_exc()}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
# Reduce default logging noise
logging.getLogger("mcts").setLevel(logging.WARNING)
logging.getLogger("workers.self_play_worker").setLevel(logging.INFO)
logging.getLogger("workers.training_worker").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)  # Main app logger

# --- Constants ---
LOOP_TIMING_INTERVAL = 60


class MainApp:
    """Main application class orchestrating Pygame UI and AlphaZero components."""

    def __init__(self, checkpoint_to_load: Optional[str] = None):
        # Config Instances
        self.vis_config = VisConfig()
        self.env_config = EnvConfig()
        self.train_config_instance = TrainConfig()
        self.mcts_config = MCTSConfig()

        # Core Components
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.renderer: Optional[UIRenderer] = None
        self.input_handler: Optional[InputHandler] = None

        # State
        self.app_state: AppState = AppState.INITIALIZING
        self.status: str = "Initializing..."
        self.running: bool = True
        self.update_progress_details: Dict[str, Any] = {}

        # Threading & Communication
        self.stop_event = threading.Event()
        self.experience_queue: queue.Queue[ProcessedExperienceBatch] = queue.Queue(
            maxsize=self.train_config_instance.BUFFER_CAPACITY
        )
        logger.info(
            f"Experience Queue Size: {self.train_config_instance.BUFFER_CAPACITY}"
        )

        # RL Components (Managed by Initializer)
        self.agent: Optional[AlphaZeroNet] = None
        self.stats_aggregator: Optional[StatsAggregator] = None
        self.demo_env: Optional[GameState] = None

        # Helper Classes
        self.device = get_torch_device()
        set_device(self.device)
        self.checkpoint_to_load = checkpoint_to_load
        self.initializer = AppInitializer(self)
        self.logic = AppLogic(self)
        self.worker_manager = AppWorkerManager(self)
        self.ui_utils = AppUIUtils(self)

        # UI State
        self.cleanup_confirmation_active: bool = False
        self.cleanup_message: str = ""
        self.last_cleanup_message_time: float = 0.0
        self.total_gpu_memory_bytes: Optional[int] = None

        # Timing
        self.frame_count = 0
        self.loop_times = deque(maxlen=LOOP_TIMING_INTERVAL)

    def initialize(self):
        """Initializes Pygame, directories, configs, and core components."""
        logger.info("--- Application Initialization ---")
        self.screen, self.clock = initialize_pygame(self.vis_config)
        initialize_directories()
        set_random_seeds(RANDOM_SEED)
        run_pre_checks()

        self.app_state = AppState.INITIALIZING
        self.initializer.initialize_all()

        self.agent = self.initializer.agent
        self.stats_aggregator = self.initializer.stats_aggregator
        self.demo_env = self.initializer.demo_env

        if self.renderer and self.input_handler:
            self.renderer.set_input_handler(self.input_handler)
            if hasattr(self.input_handler, "app_ref"):
                self.input_handler.app_ref = self

        self.logic.check_initial_completion_status()
        self.status = "Ready"
        self.app_state = AppState.MAIN_MENU
        logger.info("--- Initialization Complete ---")

    def _handle_input(self) -> bool:
        """Handles user input using the InputHandler."""
        if self.input_handler:
            return self.input_handler.handle_input(
                self.app_state.value, self.cleanup_confirmation_active
            )
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.logic.exit_app()
                    return False
            return True

    def _update_state(self):
        """Updates application logic and status."""
        self.update_progress_details = {}
        self.logic.update_status_and_check_completion()

        # Update demo env timers if in demo/debug mode
        if self.app_state in [AppState.PLAYING, AppState.DEBUG] and self.demo_env:
            try:
                self.demo_env._update_timers()
            except Exception as e:
                logger.error(f"Error updating demo env timers: {e}")

    def _prepare_render_data(self) -> Dict[str, Any]:
        """Gathers all necessary data for rendering the current frame."""
        render_data = {
            "app_state": self.app_state.value,
            "status": self.status,
            "cleanup_confirmation_active": self.cleanup_confirmation_active,
            "cleanup_message": self.cleanup_message,
            "last_cleanup_message_time": self.last_cleanup_message_time,
            "update_progress_details": self.update_progress_details,
            "demo_env": self.demo_env,  # Pass the demo env object itself
            "env_config": self.env_config,
            "num_envs": self.train_config_instance.NUM_SELF_PLAY_WORKERS,
            "plot_data": {},
            "stats_summary": {},
            "best_game_state_data": None,
            "agent_param_count": 0,
            "worker_counts": {},
            "is_process_running": False,
            "worker_render_data": [],
        }

        if self.stats_aggregator:
            render_data["plot_data"] = self.stats_aggregator.get_plot_data()
            current_step = self.stats_aggregator.storage.current_global_step
            render_data["stats_summary"] = self.stats_aggregator.get_summary(
                current_step
            )
            render_data["best_game_state_data"] = (
                self.stats_aggregator.get_best_game_state_data()
            )

        if self.initializer:
            render_data["agent_param_count"] = self.initializer.agent_param_count

        if self.worker_manager:
            render_data["worker_counts"] = (
                self.worker_manager.get_active_worker_counts()
            )
            render_data["is_process_running"] = (
                self.worker_manager.is_any_worker_running()
            )
            if (
                render_data["is_process_running"]
                and self.app_state == AppState.MAIN_MENU
            ):
                num_to_render = self.vis_config.NUM_ENVS_TO_RENDER
                if num_to_render > 0:
                    render_data["worker_render_data"] = (
                        self.worker_manager.get_worker_render_data(num_to_render)
                    )

        return render_data

    def _render_frame(self, render_data: Dict[str, Any]):
        """Renders the UI frame using the collected data."""
        if self.renderer:
            self.renderer.render_all(**render_data)
        else:
            try:
                self.screen.fill((20, 0, 0))
                font = pygame.font.Font(None, 30)
                text_surf = font.render(
                    "Renderer Initialization Failed!", True, (255, 50, 50)
                )
                self.screen.blit(
                    text_surf, text_surf.get_rect(center=self.screen.get_rect().center)
                )
                pygame.display.flip()
            except Exception:
                pass

    def _log_loop_timing(self, loop_start_time: float):
        """Logs average loop time periodically."""
        self.loop_times.append(time.monotonic() - loop_start_time)
        self.frame_count += 1

    def run_main_loop(self):
        """The main application loop."""
        logger.info("Starting main application loop...")
        while self.running:
            loop_start_time = time.monotonic()

            if not self.clock or not self.screen:
                logger.error("Pygame clock or screen not initialized. Exiting.")
                break

            dt = self.clock.tick(self.vis_config.FPS) / 1000.0

            self.running = self._handle_input()
            if not self.running:
                break

            self._update_state()  # Updates demo timers if needed
            render_data = self._prepare_render_data()
            self._render_frame(render_data)

            self._log_loop_timing(loop_start_time)

        logger.info("Main application loop exited.")

    def shutdown(self):
        """Cleans up resources and exits."""
        logger.info("Initiating shutdown sequence...")
        logger.info("Stopping worker threads...")
        self.worker_manager.stop_all_workers()
        logger.info("Worker threads stopped.")
        logger.info("Attempting final checkpoint save...")
        self.logic.save_final_checkpoint()
        logger.info("Final checkpoint save attempt finished.")
        logger.info("Closing stats recorder (before pygame.quit)...")
        self.initializer.close_stats_recorder()
        logger.info("Stats recorder closed.")
        logger.info("Quitting Pygame...")
        pygame.quit()
        logger.info("Pygame quit.")
        logger.info("Shutdown complete.")


# --- Main Execution ---
tee_logger_instance: Optional[TeeLogger] = None


def setup_logging_and_run_id(args: argparse.Namespace):
    """Sets up logging (including TeeLogger) and determines the run ID."""
    global tee_logger_instance
    run_id_source = "New"

    if args.load_checkpoint:
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                run_id_source = f"Explicit Checkpoint ({get_run_id()})"
            else:
                get_run_id()
                run_id_source = (
                    f"New (Explicit Ckpt Path Invalid: {args.load_checkpoint})"
                )
        except Exception as e:
            logger.warning(
                f"Could not determine run_id from checkpoint path '{args.load_checkpoint}': {e}. Generating new."
            )
            get_run_id()
            run_id_source = f"New (Error parsing ckpt path)"
    else:
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            run_id_source = f"Resumed Latest ({get_run_id()})"
        else:
            get_run_id()
            run_id_source = f"New (No previous runs found)"

    current_run_id = get_run_id()
    print(f"Run ID: {current_run_id} (Source: {run_id_source})")

    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        log_file_dir = get_run_log_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, "console_output.log")
        tee_logger_instance = TeeLogger(log_file_path, sys.stdout)
        sys.stdout = tee_logger_instance
        sys.stderr = tee_logger_instance
        print(f"Console output will be mirrored to: {log_file_path}")
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)

    # Configure Python's logging system
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    # Set default levels for noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    # Set default levels for our workers unless overridden by args.log_level
    if log_level <= logging.INFO:
        logging.getLogger("mcts").setLevel(logging.WARNING)  # Keep MCTS quieter
        logging.getLogger("workers.self_play_worker").setLevel(logging.INFO)
        logging.getLogger("workers.training_worker").setLevel(logging.INFO)
    # If DEBUG is requested, set workers/MCTS to DEBUG too
    if log_level == logging.DEBUG:
        logging.getLogger("mcts").setLevel(logging.DEBUG)
        logging.getLogger("workers.self_play_worker").setLevel(logging.DEBUG)
        logging.getLogger("workers.training_worker").setLevel(logging.DEBUG)

    logger.info(f"Logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {current_run_id}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")

    return original_stdout, original_stderr


def cleanup_logging(original_stdout, original_stderr, exit_code):
    """Restores standard output/error and closes logger."""
    print("[Main Finally] Restoring stdout/stderr and closing logger...")
    logger = logging.getLogger(__name__)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Trainer")
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint file (.pth) to load.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    original_stdout, original_stderr = setup_logging_and_run_id(args)

    app = None
    exit_code = 0
    try:
        app = MainApp(checkpoint_to_load=args.load_checkpoint)
        app.initialize()
        app.run_main_loop()

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Shutting down gracefully...")
        if app:
            app.logic.exit_app()
        else:
            pygame.quit()
        exit_code = 130

    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        if app:
            app.logic.exit_app()
        else:
            pygame.quit()
        exit_code = 1

    finally:
        if app:
            app.shutdown()
        cleanup_logging(original_stdout, original_stderr, exit_code)


File: requirements.txt
pygame>=2.1.0
numpy>=1.20.0
torch>=1.10.0
tensorboard
cloudpickle
matplotlib
psutil
numba>=0.55.0

File: agent\alphazero_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List  # Added List
import numpy as np  # Added numpy

from config import ModelConfig, EnvConfig
from utils.types import StateType, ActionType


class ResidualBlock(nn.Module):
    """Basic Residual Block for CNN."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """
    Neural Network for AlphaZero.
    Takes game state features and outputs policy logits and a value estimate.
    Includes methods for single and batched predictions.
    """

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        model_config: Optional[ModelConfig.Network] = None,
    ):
        super().__init__()
        self.env_cfg = env_config if env_config else EnvConfig()
        self.model_cfg = model_config if model_config else ModelConfig.Network()

        # --- Input Processing Layers ---
        grid_input_channels = self.env_cfg.GRID_STATE_SHAPE[0]
        conv_channels = self.model_cfg.CONV_CHANNELS
        current_channels = grid_input_channels
        conv_layers = []
        for out_channels in conv_channels:
            conv_layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=self.model_cfg.CONV_KERNEL_SIZE,
                    stride=self.model_cfg.CONV_STRIDE,
                    padding=self.model_cfg.CONV_PADDING,
                    bias=not self.model_cfg.USE_BATCHNORM_CONV,
                )
            )
            if self.model_cfg.USE_BATCHNORM_CONV:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(self.model_cfg.CONV_ACTIVATION())
            # Add ResidualBlock *after* activation
            conv_layers.append(ResidualBlock(out_channels))
            current_channels = out_channels
        self.conv_backbone = nn.Sequential(*conv_layers)

        conv_output_size = self._calculate_conv_output_size(
            (grid_input_channels, self.env_cfg.ROWS, self.env_cfg.COLS)
        )

        shape_input_dim = self.env_cfg.SHAPE_STATE_DIM
        shape_mlp_dims = self.model_cfg.SHAPE_FEATURE_MLP_DIMS
        shape_layers = []
        current_shape_dim = shape_input_dim
        for dim in shape_mlp_dims:
            shape_layers.append(nn.Linear(current_shape_dim, dim))
            shape_layers.append(self.model_cfg.SHAPE_MLP_ACTIVATION())
            current_shape_dim = dim
        self.shape_mlp = nn.Sequential(*shape_layers)
        shape_output_dim = current_shape_dim if shape_mlp_dims else shape_input_dim

        other_features_dim = (
            self.env_cfg.SHAPE_AVAILABILITY_DIM + self.env_cfg.EXPLICIT_FEATURES_DIM
        )

        combined_input_dim = conv_output_size + shape_output_dim + other_features_dim
        combined_fc_dims = self.model_cfg.COMBINED_FC_DIMS
        fusion_layers = []
        current_combined_dim = combined_input_dim
        for dim in combined_fc_dims:
            fusion_layers.append(
                nn.Linear(
                    current_combined_dim, dim, bias=not self.model_cfg.USE_BATCHNORM_FC
                )
            )
            if self.model_cfg.USE_BATCHNORM_FC:
                fusion_layers.append(nn.BatchNorm1d(dim))
            fusion_layers.append(self.model_cfg.COMBINED_ACTIVATION())
            if self.model_cfg.DROPOUT_FC > 0:
                fusion_layers.append(nn.Dropout(self.model_cfg.DROPOUT_FC))
            current_combined_dim = dim
        self.fusion_mlp = nn.Sequential(*fusion_layers)
        fusion_output_dim = current_combined_dim

        self.policy_head = nn.Linear(fusion_output_dim, self.env_cfg.ACTION_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh()
        )

    def _calculate_conv_output_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Helper to calculate the flattened output size of the conv backbone."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.conv_backbone(dummy_input)
            return int(torch.flatten(output, 1).shape[1])

    def forward(
        self, state: StateType  # Expects Tensors in the dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        Assumes input state dictionary contains tensors.
        """
        grid = state["grid"]
        shapes = state["shapes"]
        shape_availability = state["shape_availability"]
        explicit_features = state["explicit_features"]

        conv_out = self.conv_backbone(grid)
        flat_conv_out = torch.flatten(conv_out, 1)

        if self.model_cfg.SHAPE_FEATURE_MLP_DIMS:
            shape_out = self.shape_mlp(shapes)
        else:
            shape_out = shapes

        other_features = torch.cat([shape_availability, explicit_features], dim=-1)
        combined_features = torch.cat([flat_conv_out, shape_out, other_features], dim=1)

        fused_out = self.fusion_mlp(combined_features)

        policy_logits = self.policy_head(fused_out)
        value = self.value_head(fused_out)

        return policy_logits, value

    def _prepare_state_batch(self, state_numpy_list: List[StateType]) -> StateType:
        """Converts a list of numpy state dicts to a batched tensor state dict."""
        device = next(self.parameters()).device
        # Initialize structure to hold batched data for each state component
        batched_tensors = {key: [] for key in state_numpy_list[0].keys()}

        for state_numpy in state_numpy_list:
            for key, value in state_numpy.items():
                if key in batched_tensors:
                    batched_tensors[key].append(value)
                else:
                    # This shouldn't happen if all state dicts have the same keys
                    print(
                        f"Warning: Key {key} not found in initial state dict during batching."
                    )

        # Convert lists of numpy arrays to batched tensors
        final_batched_states = {
            k: torch.from_numpy(np.stack(v)).to(device)
            for k, v in batched_tensors.items()
        }
        return final_batched_states

    def predict_batch(
        self,
        state_numpy_list: List[
            StateType
        ],  # Expects list of numpy arrays from GameState
    ) -> Tuple[List[Dict[ActionType, float]], List[float]]:
        """
        Performs batched prediction for MCTS.
        Takes a list of state dictionaries (numpy arrays), converts to a batch tensor,
        runs inference, applies softmax, and returns lists of policy dicts and scalar values.
        """
        if not state_numpy_list:
            return [], []

        # Prepare the batch of states
        batched_state_tensors = self._prepare_state_batch(state_numpy_list)

        self.eval()
        with torch.no_grad():
            policy_logits_batch, value_batch = self.forward(batched_state_tensors)

        # Process results back into lists
        policy_probs_batch = F.softmax(policy_logits_batch, dim=-1).cpu().numpy()
        values = value_batch.squeeze(-1).cpu().numpy().tolist()  # Squeeze value output

        policy_dicts = []
        for policy_probs in policy_probs_batch:
            policy_dicts.append({i: float(prob) for i, prob in enumerate(policy_probs)})

        return policy_dicts, values

    def predict(
        self, state_numpy: StateType  # Expects numpy arrays from GameState
    ) -> Tuple[Dict[ActionType, float], float]:
        """
        Convenience method for single prediction (e.g., initial root prediction).
        Uses predict_batch internally.
        """
        policy_list, value_list = self.predict_batch([state_numpy])
        return policy_list[0], value_list[0]

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns the model's state dictionary."""
        return self.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the model's state dictionary."""
        super().load_state_dict(state_dict)


File: agent\__init__.py


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
DARK_GRAY: tuple[int, int, int] = (30, 30, 30)
RED: tuple[int, int, int] = (255, 50, 50)
DARK_RED: tuple[int, int, int] = (80, 10, 10)
BLUE: tuple[int, int, int] = (50, 50, 255)
YELLOW: tuple[int, int, int] = (255, 255, 100)
GREEN: tuple[int, int, int] = (50, 200, 50)
DARK_GREEN: tuple[int, int, int] = (20, 80, 20)  # Added Dark Green
ORANGE: tuple[int, int, int] = (255, 165, 0)
PURPLE: tuple[int, int, int] = (128, 0, 128)
CYAN: tuple[int, int, int] = (0, 255, 255)

GOOGLE_COLORS: list[tuple[int, int, int]] = [
    (15, 157, 88),  # Green
    (244, 180, 0),  # Yellow/Orange
    (66, 133, 244),  # Blue
    (219, 68, 55),  # Red
]
LINE_CLEAR_FLASH_COLOR: tuple[int, int, int] = (180, 180, 220)
LINE_CLEAR_HIGHLIGHT_COLOR: tuple[int, int, int, int] = (255, 255, 0, 180)  # RGBA
GAME_OVER_FLASH_COLOR: tuple[int, int, int] = (255, 0, 0)

# MCTS Visualization Colors
MCTS_NODE_WIN_COLOR: tuple[int, int, int] = DARK_GREEN  # Use darker green for node fill
MCTS_NODE_LOSS_COLOR: tuple[int, int, int] = DARK_RED
MCTS_NODE_NEUTRAL_COLOR: tuple[int, int, int] = DARK_GRAY  # Use darker gray
MCTS_NODE_BORDER_COLOR: tuple[int, int, int] = GRAY  # Lighter border
MCTS_NODE_SELECTED_BORDER_COLOR: tuple[int, int, int] = YELLOW
MCTS_EDGE_COLOR: tuple[int, int, int] = GRAY  # Lighter edge color
MCTS_EDGE_HIGHLIGHT_COLOR: tuple[int, int, int] = WHITE
MCTS_INFO_TEXT_COLOR: tuple[int, int, int] = WHITE
MCTS_NODE_TEXT_COLOR: tuple[int, int, int] = WHITE
MCTS_NODE_PRIOR_COLOR: tuple[int, int, int] = CYAN
MCTS_NODE_SCORE_COLOR: tuple[int, int, int] = ORANGE
MCTS_MINI_GRID_BG_COLOR: tuple[int, int, int] = (40, 40, 40)  # Background for mini-grid
MCTS_MINI_GRID_LINE_COLOR: tuple[int, int, int] = (70, 70, 70)  # Lines for mini-grid
MCTS_MINI_GRID_OCCUPIED_COLOR: tuple[int, int, int] = (
    200,
    200,
    200,
)  # Occupied cells in mini-grid


File: config\core.py
# File: config/core.py
import torch
from typing import List, Tuple, Optional

from .constants import (
    WHITE,
    BLACK,
    LIGHTG,
    GRAY,
    DARK_GRAY,
    RED,
    DARK_RED,
    BLUE,
    YELLOW,
    GREEN,
    DARK_GREEN,
    ORANGE,
    PURPLE,
    CYAN,
    GOOGLE_COLORS,
    LINE_CLEAR_FLASH_COLOR,
    LINE_CLEAR_HIGHLIGHT_COLOR,
    GAME_OVER_FLASH_COLOR,
    MCTS_NODE_WIN_COLOR,
    MCTS_NODE_LOSS_COLOR,
    MCTS_NODE_NEUTRAL_COLOR,
    MCTS_NODE_BORDER_COLOR,
    MCTS_NODE_SELECTED_BORDER_COLOR,
    MCTS_EDGE_COLOR,
    MCTS_EDGE_HIGHLIGHT_COLOR,
    MCTS_INFO_TEXT_COLOR,
    MCTS_NODE_TEXT_COLOR,
    MCTS_NODE_PRIOR_COLOR,
    MCTS_NODE_SCORE_COLOR,
    MCTS_MINI_GRID_BG_COLOR,
    MCTS_MINI_GRID_LINE_COLOR,
    MCTS_MINI_GRID_OCCUPIED_COLOR,
)


class MCTSConfig:
    """Configuration parameters for the Monte Carlo Tree Search."""

    PUCT_C: float = 1.5
    NUM_SIMULATIONS: int = 5000
    TEMPERATURE_INITIAL: float = 1.0
    TEMPERATURE_FINAL: float = 0.01
    TEMPERATURE_ANNEAL_STEPS: int = 30
    DIRICHLET_ALPHA: float = 0.3
    DIRICHLET_EPSILON: float = 0.25
    MAX_SEARCH_DEPTH: int = 100


class VisConfig:
    NUM_ENVS_TO_RENDER = 4
    FPS = 0
    SCREEN_WIDTH = 1600
    SCREEN_HEIGHT = 900
    VISUAL_STEP_DELAY = 0.00  # Keep at 0 for workers
    LEFT_PANEL_RATIO = 0.5
    ENV_SPACING = 2
    ENV_GRID_PADDING = 1

    WHITE = WHITE
    BLACK = BLACK
    LIGHTG = LIGHTG
    GRAY = GRAY
    DARK_GRAY = DARK_GRAY
    RED = RED
    DARK_RED = DARK_RED
    BLUE = BLUE
    YELLOW = YELLOW
    GREEN = GREEN
    DARK_GREEN = DARK_GREEN
    ORANGE = ORANGE
    PURPLE = PURPLE
    CYAN = CYAN
    GOOGLE_COLORS = GOOGLE_COLORS
    LINE_CLEAR_FLASH_COLOR = LINE_CLEAR_FLASH_COLOR
    LINE_CLEAR_HIGHLIGHT_COLOR = LINE_CLEAR_HIGHLIGHT_COLOR
    GAME_OVER_FLASH_COLOR = GAME_OVER_FLASH_COLOR
    MCTS_NODE_WIN_COLOR = MCTS_NODE_WIN_COLOR
    MCTS_NODE_LOSS_COLOR = MCTS_NODE_LOSS_COLOR
    MCTS_NODE_NEUTRAL_COLOR = MCTS_NODE_NEUTRAL_COLOR
    MCTS_NODE_BORDER_COLOR = MCTS_NODE_BORDER_COLOR
    MCTS_NODE_SELECTED_BORDER_COLOR = MCTS_NODE_SELECTED_BORDER_COLOR
    MCTS_EDGE_COLOR = MCTS_EDGE_COLOR
    MCTS_EDGE_HIGHLIGHT_COLOR = MCTS_EDGE_HIGHLIGHT_COLOR
    MCTS_INFO_TEXT_COLOR = MCTS_INFO_TEXT_COLOR
    MCTS_NODE_TEXT_COLOR = MCTS_NODE_TEXT_COLOR
    MCTS_NODE_PRIOR_COLOR = MCTS_NODE_PRIOR_COLOR
    MCTS_NODE_SCORE_COLOR = MCTS_NODE_SCORE_COLOR
    MCTS_MINI_GRID_BG_COLOR = MCTS_MINI_GRID_BG_COLOR
    MCTS_MINI_GRID_LINE_COLOR = MCTS_MINI_GRID_LINE_COLOR
    MCTS_MINI_GRID_OCCUPIED_COLOR = MCTS_MINI_GRID_OCCUPIED_COLOR


class EnvConfig:
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


class RNNConfig:
    USE_RNN = False
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2


class TransformerConfig:
    USE_TRANSFORMER = False
    TRANSFORMER_D_MODEL = 256
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_DIM_FEEDFORWARD = 512
    TRANSFORMER_NUM_LAYERS = 3
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_ACTIVATION = "relu"


class TrainConfig:
    """Configuration parameters for the Training Worker."""

    CHECKPOINT_SAVE_FREQ = 1000
    LOAD_CHECKPOINT_PATH: Optional[str] = None
    NUM_SELF_PLAY_WORKERS: int = 8
    BATCH_SIZE: int = 512
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    NUM_TRAINING_STEPS_PER_ITER: int = 100
    MIN_BUFFER_SIZE_TO_TRAIN: int = 2000
    BUFFER_CAPACITY: int = 200000
    POLICY_LOSS_WEIGHT: float = 1.0
    VALUE_LOSS_WEIGHT: float = 1.0
    USE_LR_SCHEDULER: bool = True
    SCHEDULER_TYPE: str = "CosineAnnealingLR"
    SCHEDULER_T_MAX: int = 100000
    SCHEDULER_ETA_MIN: float = 1e-6


class ModelConfig:
    class Network:
        _env_cfg_instance = EnvConfig()
        HEIGHT = _env_cfg_instance.ROWS
        WIDTH = _env_cfg_instance.COLS
        del _env_cfg_instance

        CONV_CHANNELS = [32, 64, 128]
        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1
        CONV_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_CONV = True
        SHAPE_FEATURE_MLP_DIMS = [64]
        SHAPE_MLP_ACTIVATION = torch.nn.ReLU
        COMBINED_FC_DIMS = [256, 128]
        COMBINED_ACTIVATION = torch.nn.ReLU
        USE_BATCHNORM_FC = True
        DROPOUT_FC = 0.1


class StatsConfig:
    STATS_AVG_WINDOW: List[int] = [100, 500]
    CONSOLE_LOG_FREQ = 100
    PLOT_DATA_WINDOW = 10000


class DemoConfig:
    BACKGROUND_COLOR = (10, 10, 20)
    SELECTED_SHAPE_HIGHLIGHT_COLOR = BLUE
    HUD_FONT_SIZE = 24
    HELP_FONT_SIZE = 18
    HELP_TEXT = "[Click Preview]=Select/Deselect | [Click Grid]=Place | [ESC]=Exit"
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
    # Updated filename for AlphaZero NN
    return os.path.join(get_run_checkpoint_dir(), "alphazero_nn.pth")


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
    DemoConfig,
    TransformerConfig,
    MCTSConfig,  
)

from .general import DEVICE, RANDOM_SEED, get_run_id


def get_config_dict() -> Dict[str, Any]:
    """Returns a flat dictionary of all relevant config values for logging."""
    all_configs = {}

    def flatten_class(cls, prefix=""):
        d = {}
        instance = None
        try:
            instance = cls()
        except Exception:
            instance = None

        for k, v in vars(cls).items():
            if (
                not k.startswith("__")
                and not callable(v)
                and not isinstance(v, type)
                and not hasattr(v, "__module__")
            ):
                is_property = isinstance(getattr(cls, k, None), property)
                if is_property and instance:
                    try:
                        v = getattr(instance, k)
                    except Exception:
                        continue
                elif is_property and not instance:
                    continue
                d[f"{prefix}{k}"] = v
        return d

    # Flatten core config classes
    all_configs.update(flatten_class(VisConfig, "Vis."))
    all_configs.update(flatten_class(EnvConfig, "Env."))
    all_configs.update(flatten_class(RNNConfig, "RNN."))
    all_configs.update(flatten_class(TrainConfig, "Train."))
    all_configs.update(flatten_class(ModelConfig.Network, "Model.Net."))
    all_configs.update(flatten_class(StatsConfig, "Stats."))
    # all_configs.update(flatten_class(TensorBoardConfig, "TB.")) # Removed
    all_configs.update(flatten_class(DemoConfig, "Demo."))
    all_configs.update(flatten_class(TransformerConfig, "Transformer."))
    all_configs.update(flatten_class(MCTSConfig, "MCTS."))  # Flatten MCTSConfig

    # Add general config values
    all_configs["General.DEVICE"] = str(DEVICE) if DEVICE else "None"
    all_configs["General.RANDOM_SEED"] = RANDOM_SEED
    all_configs["General.RUN_ID"] = get_run_id()

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
            all_configs[key] = str(value) if value is not None else "None"

    return all_configs


File: config\validation.py
# File: config/validation.py
from .core import (
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    VisConfig,
    TransformerConfig,
    MCTSConfig,
)
from .general import (
    DEVICE,
    get_run_id,
    get_run_log_dir,
    get_run_checkpoint_dir,
)


def print_config_info_and_validate():
    env_config_instance = EnvConfig()
    rnn_config_instance = RNNConfig()
    transformer_config_instance = TransformerConfig()
    train_config_instance = TrainConfig()
    mcts_config_instance = MCTSConfig()
    vis_config_instance = VisConfig()  # Get instance for NUM_ENVS_TO_RENDER
    stats_config_instance = StatsConfig()  # Get instance for logging frequency

    run_id = get_run_id()
    run_log_dir = get_run_log_dir()
    run_checkpoint_dir = get_run_checkpoint_dir()

    print("-" * 70)
    print(f"RUN ID: {run_id}")
    print(f"Log Directory: {run_log_dir}")
    print(f"Checkpoint Directory: {run_checkpoint_dir}")
    print(f"Device: {DEVICE}")

    if train_config_instance.LOAD_CHECKPOINT_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD CHECKPOINT specified: {train_config_instance.LOAD_CHECKPOINT_PATH} ***\n"
            "*** CheckpointManager will attempt to load this path (NN weights, Optimizer, Stats). ***\n"
            + "*" * 70
        )
    else:
        print(
            "--- No explicit checkpoint path. CheckpointManager will attempt auto-resume if applicable. ---"
        )

    print("--- Training Algorithm: AlphaZero (MCTS + NN) ---")

    if rnn_config_instance.USE_RNN:
        print(
            f"--- Warning: RNN configured ON ({rnn_config_instance.LSTM_HIDDEN_SIZE}, {rnn_config_instance.LSTM_NUM_LAYERS}) but not used by AlphaZeroNet ---"
        )
    if transformer_config_instance.USE_TRANSFORMER:
        print(
            f"--- Warning: Transformer configured ON ({transformer_config_instance.TRANSFORMER_D_MODEL}, {transformer_config_instance.TRANSFORMER_NHEAD}, {transformer_config_instance.TRANSFORMER_NUM_LAYERS}) but not used by AlphaZeroNet ---"
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
    dropout = ModelConfig.Network.DROPOUT_FC
    print(
        f"Network Base: CNN={cnn_str}, ShapeMLP={shape_mlp_cfg_str}, Fusion={mlp_str}, Dropout={dropout}"
    )

    print(
        f"MCTS: Sims={mcts_config_instance.NUM_SIMULATIONS}, "
        f"PUCT_C={mcts_config_instance.PUCT_C:.2f}, "
        f"Temp={mcts_config_instance.TEMPERATURE_INITIAL:.2f}->{mcts_config_instance.TEMPERATURE_FINAL:.2f} (Steps: {mcts_config_instance.TEMPERATURE_ANNEAL_STEPS}), "
        f"Dirichlet(α={mcts_config_instance.DIRICHLET_ALPHA:.2f}, ε={mcts_config_instance.DIRICHLET_EPSILON:.2f})"
    )

    scheduler_info = "DISABLED"
    if train_config_instance.USE_LR_SCHEDULER:
        scheduler_info = f"{train_config_instance.SCHEDULER_TYPE} (T_max={train_config_instance.SCHEDULER_T_MAX:,}, eta_min={train_config_instance.SCHEDULER_ETA_MIN:.1e})"
    print(
        f"Training: Batch={train_config_instance.BATCH_SIZE}, LR={train_config_instance.LEARNING_RATE:.1e}, "
        f"WD={train_config_instance.WEIGHT_DECAY:.1e}, Scheduler={scheduler_info}"
    )
    print(
        f"Buffer: Capacity={train_config_instance.BUFFER_CAPACITY:,}, MinSize={train_config_instance.MIN_BUFFER_SIZE_TO_TRAIN:,}, Steps/Iter={train_config_instance.NUM_TRAINING_STEPS_PER_ITER}"
    )
    print(
        f"Workers: Self-Play={train_config_instance.NUM_SELF_PLAY_WORKERS}, Training=1"
    )
    print(
        f"Stats: AVG_WINDOWS={stats_config_instance.STATS_AVG_WINDOW}, Console Log Freq={stats_config_instance.CONSOLE_LOG_FREQ} (updates/episodes)"
    )

    render_info = (
        f"Live Self-Play Workers (Max: {vis_config_instance.NUM_ENVS_TO_RENDER})"
    )
    print(f"--- Rendering {render_info} in Game Area ---")
    print("-" * 70)


File: config\__init__.py
# File: config/__init__.py
# config/__init__.py
# This file marks the 'config' directory as a Python package.

# Import core configuration classes to make them available directly under 'config'
from .core import (
    VisConfig,
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    # TensorBoardConfig removed
    DemoConfig,
    TransformerConfig,
    MCTSConfig,
)

# Import general configuration settings and functions
from .general import (
    DEVICE,
    RANDOM_SEED,
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
    DARK_GRAY,
    RED,
    DARK_RED,
    BLUE,
    YELLOW,
    GREEN,
    DARK_GREEN,  # Added DARK_GREEN import
    ORANGE,
    PURPLE,
    CYAN,
    GOOGLE_COLORS,
    LINE_CLEAR_FLASH_COLOR,
    LINE_CLEAR_HIGHLIGHT_COLOR,
    GAME_OVER_FLASH_COLOR,
    # MCTS Colors (also available directly)
    MCTS_NODE_WIN_COLOR,
    MCTS_NODE_LOSS_COLOR,
    MCTS_NODE_NEUTRAL_COLOR,
    MCTS_NODE_BORDER_COLOR,
    MCTS_NODE_SELECTED_BORDER_COLOR,
    MCTS_EDGE_COLOR,
    MCTS_EDGE_HIGHLIGHT_COLOR,
    MCTS_INFO_TEXT_COLOR,
    MCTS_NODE_TEXT_COLOR,
    MCTS_NODE_PRIOR_COLOR,
    MCTS_NODE_SCORE_COLOR,
    MCTS_MINI_GRID_BG_COLOR,
    MCTS_MINI_GRID_LINE_COLOR,
    MCTS_MINI_GRID_OCCUPIED_COLOR,
)


# Define __all__ to control what 'from config import *' imports
__all__ = [
    # Core Configs
    "VisConfig",
    "EnvConfig",
    "RNNConfig",
    "TrainConfig",
    "ModelConfig",
    "StatsConfig",
    # "TensorBoardConfig", # Removed
    "DemoConfig",
    "TransformerConfig",
    "MCTSConfig",
    # General Configs
    "DEVICE",
    "RANDOM_SEED",
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
    "DARK_GRAY",
    "RED",
    "DARK_RED",
    "BLUE",
    "YELLOW",
    "GREEN",
    "DARK_GREEN",  # Added DARK_GREEN export
    "ORANGE",
    "PURPLE",
    "CYAN",
    "GOOGLE_COLORS",
    "LINE_CLEAR_FLASH_COLOR",
    "LINE_CLEAR_HIGHLIGHT_COLOR",
    "GAME_OVER_FLASH_COLOR",
    # MCTS Colors
    "MCTS_NODE_WIN_COLOR",
    "MCTS_NODE_LOSS_COLOR",
    "MCTS_NODE_NEUTRAL_COLOR",
    "MCTS_NODE_BORDER_COLOR",
    "MCTS_NODE_SELECTED_BORDER_COLOR",
    "MCTS_EDGE_COLOR",
    "MCTS_EDGE_HIGHLIGHT_COLOR",
    "MCTS_INFO_TEXT_COLOR",
    "MCTS_NODE_TEXT_COLOR",
    "MCTS_NODE_PRIOR_COLOR",
    "MCTS_NODE_SCORE_COLOR",
    "MCTS_MINI_GRID_BG_COLOR",
    "MCTS_MINI_GRID_LINE_COLOR",
    "MCTS_MINI_GRID_OCCUPIED_COLOR",
]


File: environment\game_demo_logic.py
# File: environment/game_demo_logic.py
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .game_state import GameState
    from .shape import Shape


class GameDemoLogic:
    """Handles logic specific to the interactive Demo/Debug mode."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def select_shape_for_drag(self, shape_index: int):
        """Selects a shape for dragging if available."""
        if (
            0 <= shape_index < len(self.gs.shapes)
            and self.gs.shapes[shape_index] is not None
        ):
            # If clicking the already selected shape, deselect it
            if self.gs.demo_selected_shape_idx == shape_index:
                self.gs.demo_selected_shape_idx = -1  # Indicate no selection
                self.gs.demo_dragged_shape_idx = None
                self.gs.demo_snapped_position = None
            else:
                self.gs.demo_selected_shape_idx = shape_index
                self.gs.demo_dragged_shape_idx = shape_index
                self.gs.demo_snapped_position = None  # Reset snap on new selection

    def deselect_dragged_shape(self):
        """Deselects any currently dragged shape."""
        self.gs.demo_dragged_shape_idx = None
        self.gs.demo_snapped_position = None
        # Keep demo_selected_shape_idx as is, maybe user wants to re-drag later

    def update_snapped_position(self, grid_pos: Optional[Tuple[int, int]]):
        """Updates the snapped position based on grid hover."""
        if self.gs.demo_dragged_shape_idx is None:
            self.gs.demo_snapped_position = None
            return

        shape_to_drag = self.gs.shapes[self.gs.demo_dragged_shape_idx]
        if shape_to_drag is None:
            self.gs.demo_snapped_position = None
            return

        if grid_pos is not None and self.gs.grid.can_place(
            shape_to_drag, grid_pos[0], grid_pos[1]
        ):
            self.gs.demo_snapped_position = grid_pos
        else:
            self.gs.demo_snapped_position = None

    def place_dragged_shape(self) -> bool:
        """Attempts to place the currently dragged shape at the snapped position."""
        if (
            self.gs.demo_dragged_shape_idx is not None
            and self.gs.demo_snapped_position is not None
        ):
            shape_idx = self.gs.demo_dragged_shape_idx
            r, c = self.gs.demo_snapped_position

            # Encode the action based on the demo state
            action_index = self.gs.logic.encode_action(shape_idx, r, c)

            # Use the core game logic step function
            _, done = self.gs.logic.step(action_index)

            # Reset demo drag state after placement attempt
            self.gs.demo_dragged_shape_idx = None
            self.gs.demo_snapped_position = None
            self.gs.demo_selected_shape_idx = -1  # Deselect after placement

            return not done  # Return True if placement was successful (game not over)
        return False

    def get_dragged_shape_info(
        self,
    ) -> Tuple[Optional["Shape"], Optional[Tuple[int, int]]]:
        """Returns the currently dragged shape and its snapped position."""
        if self.gs.demo_dragged_shape_idx is None:
            return None, None
        shape = self.gs.shapes[self.gs.demo_dragged_shape_idx]
        return shape, self.gs.demo_snapped_position

    def toggle_triangle_debug(self, row: int, col: int):
        """Toggles the occupied state of a triangle in debug mode and checks for line clears."""
        if not self.gs.grid.valid(row, col):
            return

        tri = self.gs.grid.triangles[row][col]
        if tri.is_death:
            return  # Cannot toggle death cells

        # Toggle state
        tri.is_occupied = not tri.is_occupied
        self.gs.grid._occupied_np[row, col] = tri.is_occupied  # Update numpy array
        tri.color = self.gs.vis_config.WHITE if tri.is_occupied else None

        # Manually trigger line clear check for the toggled triangle
        toggled_triangle_set = {tri}
        lines_cleared, tris_cleared, cleared_coords = self.gs.grid.clear_lines(
            newly_occupied_triangles=toggled_triangle_set  # Pass the toggled tri
        )

        if lines_cleared > 0:
            # Update score and visual timers if lines were cleared
            self.gs.triangles_cleared_this_episode += tris_cleared
            score_increase = (lines_cleared**2) * 10 + tris_cleared
            self.gs.game_score += score_increase
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.6
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                tris_cleared,
                score_increase,
            )
            print(f"[Debug] Cleared {lines_cleared} lines ({tris_cleared} tris).")


File: environment\game_logic.py
# File: environment/game_logic.py
import random
from typing import TYPE_CHECKING, List, Tuple, Optional, Set

from .shape import Shape
from .triangle import Triangle

if TYPE_CHECKING:
    from .game_state import GameState


class GameLogic:
    """Handles core game mechanics like placing shapes, clearing lines, and game over conditions."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def step(self, action_index: int) -> Tuple[Optional[float], bool]:
        """
        Performs one game step based on the action index.
        Returns (reward, is_game_over). Reward is currently always None.
        """
        shape_slot_index, target_row, target_col = self.decode_action(action_index)

        # Validate shape index and shape existence
        if not (0 <= shape_slot_index < len(self.gs.shapes)):
            self.gs._last_action_valid = False
            self.gs.game_over = True
            self.gs.game_over_flash_time = 0.5
            return None, True
        shape_to_place = self.gs.shapes[shape_slot_index]

        if shape_to_place is None or not self.gs.grid.can_place(
            shape_to_place, target_row, target_col
        ):
            self.gs._last_action_valid = False
            self.gs.game_over = True
            self.gs.game_over_flash_time = 0.5  # Visual effect timer
            return None, True

        # Place the shape and get the set of newly occupied triangles
        newly_occupied = self.gs.grid.place(shape_to_place, target_row, target_col)
        self.gs.pieces_placed_this_episode += 1
        self.gs.shapes[shape_slot_index] = None  # Remove shape from slot

        # Clear lines using the optimized method
        lines_cleared, tris_cleared, cleared_coords = self.gs.grid.clear_lines(
            newly_occupied_triangles=newly_occupied
        )

        if lines_cleared > 0:
            self.gs.triangles_cleared_this_episode += tris_cleared
            # Update score (simple scoring for now)
            score_increase = (lines_cleared**2) * 10 + tris_cleared
            self.gs.game_score += score_increase
            # Set visual effect timers
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.6
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                tris_cleared,
                score_increase,
            )

        # Refill shape slots if needed (now checks if all are empty)
        self._refill_shape_slots()

        # Check game over condition (if any slot cannot place its shape)
        # This check needs to happen *after* potential refill
        if self._check_game_over():
            self.gs.game_over = True
            self.gs.game_over_flash_time = 0.5  # Visual effect timer
            return None, True

        self.gs._last_action_valid = True
        return None, False  # Reward is None, game not over

    def _refill_shape_slots(self):
        """Refills shape slots with new random shapes ONLY if all slots are empty."""
        # Check if all slots are currently empty (None)
        if all(s is None for s in self.gs.shapes):
            num_slots = self.gs.env_config.NUM_SHAPE_SLOTS
            self.gs.shapes = [Shape() for _ in range(num_slots)]

    def _check_game_over(self) -> bool:
        """Checks if the game is over because no available shape can be placed."""
        # If all slots are empty, the game is definitely not over yet (wait for refill)
        if all(s is None for s in self.gs.shapes):
            return False

        for shape in self.gs.shapes:
            if shape is not None:
                # Check if this shape has at least one valid placement anywhere
                for r in range(self.gs.grid.rows):
                    for c in range(self.gs.grid.cols):
                        if self.gs.grid.can_place(shape, r, c):
                            return False  # Found a valid move, game not over
        # If we get here, it means there was at least one shape, but none could be placed
        return True

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        valid_action_indices = []
        for shape_slot_index, shape in enumerate(self.gs.shapes):
            if shape is None:
                continue
            for r in range(self.gs.grid.rows):
                for c in range(self.gs.grid.cols):
                    if self.gs.grid.can_place(shape, r, c):
                        action_index = self.encode_action(shape_slot_index, r, c)
                        valid_action_indices.append(action_index)
        return valid_action_indices

    def encode_action(self, shape_slot_index: int, row: int, col: int) -> int:
        """Encodes (shape_slot, row, col) into a single action index."""
        return (
            shape_slot_index * (self.gs.grid.rows * self.gs.grid.cols)
            + row * self.gs.grid.cols
            + col
        )

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        grid_size = self.gs.grid.rows * self.gs.grid.cols
        shape_slot_index = action_index // grid_size
        remainder = action_index % grid_size
        row = remainder // self.gs.grid.cols
        col = remainder % self.gs.grid.cols
        return shape_slot_index, row, col


File: environment\game_state.py
import time
import numpy as np
from typing import List, Optional, Tuple, Dict
import copy  # Keep copy for grid deepcopy if needed

from config import EnvConfig, VisConfig
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
    Visual effect timers are managed but DO NOT block core logic execution.
    _update_timers() should only be called externally for UI/Demo rendering.
    """

    def __init__(self):
        self.env_config = EnvConfig()
        self.vis_config = VisConfig()

        self.grid = Grid(self.env_config)
        # Initialize shapes list with correct number of None slots
        self.shapes: List[Optional[Shape]] = [None] * self.env_config.NUM_SHAPE_SLOTS
        self.game_score: int = 0
        self.triangles_cleared_this_episode: int = 0
        self.pieces_placed_this_episode: int = 0

        # Timers for VISUAL effects only
        self.blink_time: float = 0.0
        self._last_timer_update_time: float = time.monotonic()
        self.freeze_time: float = 0.0
        self.line_clear_flash_time: float = 0.0
        self.line_clear_highlight_time: float = 0.0
        self.game_over_flash_time: float = 0.0
        self.cleared_triangles_coords: List[Tuple[int, int]] = []
        self.last_line_clear_info: Optional[Tuple[int, int, float]] = None

        self.game_over: bool = False
        self._last_action_valid: bool = True

        # Demo state
        self.demo_selected_shape_idx: int = -1  # Start with no selection
        self.demo_dragged_shape_idx: Optional[int] = None
        self.demo_snapped_position: Optional[Tuple[int, int]] = None

        # Helper classes
        self.logic = GameLogic(self)
        self.features = GameStateFeatures(self)
        self.demo_logic = GameDemoLogic(self)

        self.reset()

    def reset(self) -> StateType:
        """Resets the game to its initial state."""
        self.grid = Grid(self.env_config)  # Recreate grid
        # Ensure shapes list has the correct number of None slots initially
        self.shapes = [None] * self.env_config.NUM_SHAPE_SLOTS
        self.game_score = 0
        self.triangles_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0

        # Reset visual timers
        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.line_clear_flash_time = 0.0
        self.line_clear_highlight_time = 0.0
        self.game_over_flash_time = 0.0
        self.cleared_triangles_coords = []
        self.last_line_clear_info = None

        self.game_over = False
        self._last_action_valid = True
        self._last_timer_update_time = time.monotonic()

        self.demo_selected_shape_idx = -1
        self.demo_dragged_shape_idx = None
        self.demo_snapped_position = None

        # Generate the initial batch of shapes
        self.logic._refill_shape_slots()

        return self.get_state()

    def step(self, action_index: int) -> Tuple[Optional[StateType], bool]:
        """
        Performs one game step based on the action index using GameLogic.
        Returns (None, is_game_over). State should be fetched via get_state().
        """
        _, done = self.logic.step(action_index)
        return None, done

    def get_state(self) -> StateType:
        """Returns the current game state as a dictionary of numpy arrays."""
        return self.features.get_state()

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        return self.logic.valid_actions()

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        return self.logic.decode_action(action_index)

    def is_over(self) -> bool:
        return self.game_over

    # --- Visual State Check Methods (Used by UI/Demo) ---
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

    def get_shapes(self) -> List[Optional[Shape]]:
        return self.shapes

    def get_outcome(self) -> float:
        """Determines the outcome of the game. Returns 0 for now."""
        return 0.0  # Simple outcome for now

    def _update_timers(self):
        """Updates timers for visual effects based on elapsed time."""
        now = time.monotonic()
        delta_time = now - self._last_timer_update_time
        self._last_timer_update_time = now
        delta_time = max(0.0, delta_time)

        self.freeze_time = max(0, self.freeze_time - delta_time)
        self.blink_time = max(0, self.blink_time - delta_time)
        self.line_clear_flash_time = max(0, self.line_clear_flash_time - delta_time)
        self.line_clear_highlight_time = max(
            0, self.line_clear_highlight_time - delta_time
        )
        self.game_over_flash_time = max(0, self.game_over_flash_time - delta_time)

        if self.line_clear_highlight_time <= 0 and self.cleared_triangles_coords:
            self.cleared_triangles_coords = []

    # --- Demo Mode Methods (Delegated) ---
    def select_shape_for_drag(self, shape_index: int):
        self.demo_logic.select_shape_for_drag(shape_index)

    def deselect_dragged_shape(self):
        self.demo_logic.deselect_dragged_shape()

    def update_snapped_position(self, grid_pos: Optional[Tuple[int, int]]):
        self.demo_logic.update_snapped_position(grid_pos)

    def place_dragged_shape(self) -> bool:
        return self.demo_logic.place_dragged_shape()

    def get_dragged_shape_info(
        self,
    ) -> Tuple[Optional[Shape], Optional[Tuple[int, int]]]:
        return self.demo_logic.get_dragged_shape_info()

    def toggle_triangle_debug(self, row: int, col: int):
        self.demo_logic.toggle_triangle_debug(row, col)


File: environment\game_state_features.py
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

    # Removed calculate_potential (PBRS logic)

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

            # Use grid's deepcopy method for simulation
            temp_grid = self.gs.grid.deepcopy_grid()
            # Need to simulate placement and clearing on the copy
            newly_occupied_sim = set()
            for dr, dc, _ in shape_to_place.triangles:
                nr, nc = target_row + dr, target_col + dc
                if temp_grid.valid(nr, nc):
                    tri = temp_grid.triangles[nr][nc]
                    if not tri.is_death and not tri.is_occupied:
                        tri.is_occupied = True
                        temp_grid._occupied_np[nr, nc] = True  # Update numpy array too
                        newly_occupied_sim.add(tri)

            _, triangles_cleared, _ = temp_grid.clear_lines(
                newly_occupied_triangles=newly_occupied_sim
            )
            holes_after = temp_grid.count_holes()
            height_after = temp_grid.get_max_height()
            bumpiness_after = temp_grid.get_bumpiness()
            new_holes_created = max(0, holes_after - initial_holes)

            max_triangles_cleared = max(max_triangles_cleared, triangles_cleared)
            min_new_holes = min(min_new_holes, new_holes_created)
            min_resulting_height = min(min_resulting_height, height_after)
            min_resulting_bumpiness = min(min_resulting_bumpiness, bumpiness_after)
            del temp_grid  # Clean up copy

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
        death_mask_state = self.gs.grid.get_death_data()  # Get death mask

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
            "death_mask": death_mask_state.astype(np.bool_),  # Add death mask
        }
        return state_dict


File: environment\grid.py
import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Deque
from collections import deque  # Import deque for BFS
import numba  # Import Numba
import logging  # Import logging

from config import EnvConfig
from .triangle import Triangle
from .shape import Shape

logger = logging.getLogger(__name__)  # Add logger


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig):
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.triangles: List[List[Triangle]] = self._create(env_config)
        self._link_neighbors()

        # --- Internal NumPy arrays for Numba ---
        # Initialize these arrays based on the created grid
        self._occupied_np = np.array(
            [[tri.is_occupied for tri in row] for row in self.triangles], dtype=np.bool_
        )
        self._death_np = np.array(
            [[tri.is_death for tri in row] for row in self.triangles], dtype=np.bool_
        )
        # --- End Internal NumPy arrays ---

        # Store potential lines as frozensets for easy hashing/set operations
        self.potential_lines: Set[frozenset[Triangle]] = set()
        # Index mapping Triangle object to the set of lines it belongs to
        self._triangle_to_lines_map: Dict[Triangle, Set[frozenset[Triangle]]] = {}
        self._initialize_lines_and_index()  # This needs to run after _create and _link_neighbors

    def _create(self, env_config: EnvConfig) -> List[List[Triangle]]:
        """
        Initializes the grid with playable and death cells.
        The playable area defined by cols_per_row is further reduced by
        making the first and last cell of that range into death cells.
        """
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]
        if len(cols_per_row) != self.rows:
            raise ValueError("cols_per_row length mismatch")
        if max(cols_per_row) > self.cols:
            raise ValueError("cols_per_row exceeds EnvConfig.COLS")

        grid: List[List[Triangle]] = []
        for r in range(self.rows):
            row_tris: List[Triangle] = []
            intended_playable_width = cols_per_row[r]
            total_padding = self.cols - intended_playable_width
            pad_l = total_padding // 2
            pad_r = self.cols - (total_padding - pad_l)
            playable_start_col = pad_l + 1
            playable_end_col = pad_r - 1

            for c in range(self.cols):
                is_playable = (
                    intended_playable_width > 2
                    and playable_start_col <= c < playable_end_col
                )
                is_death = not is_playable
                is_up = (r + c) % 2 == 0
                row_tris.append(Triangle(r, c, is_up=is_up, is_death=is_death))
            grid.append(row_tris)
        return grid

    def _link_neighbors(self) -> None:
        """Sets neighbor references for each triangle."""
        for r in range(self.rows):
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if self.valid(r, c - 1):
                    tri.neighbor_left = self.triangles[r][c - 1]
                if self.valid(r, c + 1):
                    tri.neighbor_right = self.triangles[r][c + 1]
                nr, nc = (r + 1, c) if tri.is_up else (r - 1, c)
                if self.valid(nr, nc):
                    tri.neighbor_vert = self.triangles[nr][nc]

    def _get_line_neighbors(self, tri: Triangle, direction: str) -> List[Triangle]:
        """Helper to get relevant neighbors for line tracing in a specific direction."""
        neighbors = []
        if direction == "horizontal":
            if tri.neighbor_left:
                neighbors.append(tri.neighbor_left)
            if tri.neighbor_right:
                neighbors.append(tri.neighbor_right)
        elif direction == "diag1":
            if tri.is_up:
                if tri.neighbor_left and not tri.neighbor_left.is_up:
                    neighbors.append(tri.neighbor_left)
                if tri.neighbor_vert and not tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
            else:
                if tri.neighbor_right and tri.neighbor_right.is_up:
                    neighbors.append(tri.neighbor_right)
                if tri.neighbor_vert and tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
        elif direction == "diag2":
            if tri.is_up:
                if tri.neighbor_right and not tri.neighbor_right.is_up:
                    neighbors.append(tri.neighbor_right)
                if tri.neighbor_vert and not tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
            else:
                if tri.neighbor_left and tri.neighbor_left.is_up:
                    neighbors.append(tri.neighbor_left)
                if tri.neighbor_vert and tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
        return [n for n in neighbors if not n.is_death]

    def _initialize_lines_and_index(self) -> None:
        """
        Identifies all sets of playable triangles forming potential lines
        by tracing connections along horizontal and diagonal axes using BFS.
        Populates self.potential_lines and self._triangle_to_lines_map.
        """
        self.potential_lines = set()
        self._triangle_to_lines_map = {}
        visited_in_direction: Dict[str, Set[Triangle]] = {
            "horizontal": set(),
            "diag1": set(),
            "diag2": set(),
        }
        min_line_length = 3

        for r in range(self.rows):
            for c in range(self.cols):
                start_node = self.triangles[r][c]
                if start_node.is_death:
                    continue

                for direction in ["horizontal", "diag1", "diag2"]:
                    if start_node not in visited_in_direction[direction]:
                        current_line: Set[Triangle] = set()
                        queue: Deque[Triangle] = deque([start_node])
                        visited_this_bfs: Set[Triangle] = {start_node}

                        while queue:
                            tri = queue.popleft()
                            if not tri.is_death:
                                current_line.add(tri)
                            visited_in_direction[direction].add(tri)

                            neighbors = self._get_line_neighbors(tri, direction)
                            for neighbor in neighbors:
                                if neighbor not in visited_this_bfs:
                                    visited_this_bfs.add(neighbor)
                                    queue.append(neighbor)

                        if len(current_line) >= min_line_length:
                            line_frozenset = frozenset(current_line)
                            self.potential_lines.add(line_frozenset)
                            for tri_in_line in current_line:
                                if tri_in_line not in self._triangle_to_lines_map:
                                    self._triangle_to_lines_map[tri_in_line] = set()
                                self._triangle_to_lines_map[tri_in_line].add(
                                    line_frozenset
                                )

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(self, shape: Shape, r: int, c: int) -> bool:
        """Checks if a shape can be placed at the target location."""
        for dr, dc, is_up_shape in shape.triangles:
            nr, nc = r + dr, c + dc
            if not self.valid(nr, nc):
                return False
            # Use pre-computed numpy arrays for faster checks
            if (
                self._death_np[nr, nc]
                or self._occupied_np[nr, nc]
                or (self.triangles[nr][nc].is_up != is_up_shape)
            ):
                return False
        return True

    def place(self, shape: Shape, r: int, c: int) -> Set[Triangle]:
        """
        Places a shape onto the grid (assumes can_place was checked).
        Updates internal numpy arrays and returns the set of occupied Triangles.
        """
        newly_occupied: Set[Triangle] = set()
        for dr, dc, _ in shape.triangles:
            nr, nc = r + dr, c + dc
            # Check validity again just in case, though can_place should precede
            if self.valid(nr, nc):
                tri = self.triangles[nr][nc]
                # Check using numpy arrays first for speed
                if not self._death_np[nr, nc] and not self._occupied_np[nr, nc]:
                    # Update Triangle object
                    tri.is_occupied = True
                    tri.color = shape.color
                    # Update internal numpy array
                    self._occupied_np[nr, nc] = True
                    newly_occupied.add(tri)
        return newly_occupied

    def clear_lines(
        self, newly_occupied_triangles: Optional[Set[Triangle]] = None
    ) -> Tuple[int, int, List[Tuple[int, int]]]:
        """
        Checks and clears completed lines using the optimized index approach.
        Updates internal numpy arrays.
        Returns lines cleared count, total triangles cleared count, and their coordinates.
        """
        lines_to_check: Set[frozenset[Triangle]] = set()
        if newly_occupied_triangles:
            for tri in newly_occupied_triangles:
                if tri in self._triangle_to_lines_map:
                    lines_to_check.update(self._triangle_to_lines_map[tri])
        else:
            lines_to_check = self.potential_lines

        cleared_tris_total: Set[Triangle] = set()
        lines_cleared_count = 0

        for line_set in lines_to_check:
            if not line_set:
                continue
            # Check occupancy using the numpy array for speed
            if all(self._occupied_np[tri.row, tri.col] for tri in line_set):
                if not line_set.issubset(cleared_tris_total):
                    cleared_tris_total.update(line_set)
                    lines_cleared_count += 1

        tris_cleared_count = 0
        coords: List[Tuple[int, int]] = []
        if not cleared_tris_total:
            return 0, 0, []

        for tri in cleared_tris_total:
            # Check using numpy array first
            if (
                not self._death_np[tri.row, tri.col]
                and self._occupied_np[tri.row, tri.col]
            ):
                tris_cleared_count += 1
                # Update Triangle object
                tri.is_occupied = False
                tri.color = None
                # Update internal numpy array
                self._occupied_np[tri.row, tri.col] = False
                coords.append((tri.row, tri.col))

        return lines_cleared_count, tris_cleared_count, coords

    @staticmethod
    @numba.njit(cache=True)
    def _numba_get_column_heights(
        rows: int, cols: int, is_occupied: np.ndarray, is_death: np.ndarray
    ) -> np.ndarray:
        """Numba-accelerated calculation of column heights."""
        heights = np.zeros(cols, dtype=np.int32)
        for c in range(cols):
            col_max_r = -1
            for r in range(rows):
                if not is_death[r, c] and is_occupied[r, c]:
                    col_max_r = max(col_max_r, r)
            heights[c] = col_max_r + 1
        return heights

    def get_column_heights(self) -> List[int]:
        """Calculates the height of occupied cells in each column using Numba and internal arrays."""
        # Use the internal numpy arrays directly
        heights_np = self._numba_get_column_heights(
            self.rows, self.cols, self._occupied_np, self._death_np
        )
        return heights_np.tolist()

    def get_max_height(self) -> int:
        """Returns the maximum height across all columns."""
        heights = self.get_column_heights()
        return max(heights) if heights else 0

    @staticmethod
    @numba.njit(cache=True)
    def _numba_get_bumpiness(heights: np.ndarray) -> int:
        """Numba-accelerated calculation of bumpiness."""
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def get_bumpiness(self) -> int:
        """Calculates the total absolute difference between adjacent column heights using Numba."""
        heights_np = np.array(self.get_column_heights(), dtype=np.int32)
        return self._numba_get_bumpiness(heights_np)

    @staticmethod
    @numba.njit(cache=True)
    def _numba_count_holes(
        rows: int,
        cols: int,
        heights: np.ndarray,
        is_occupied: np.ndarray,
        is_death: np.ndarray,
    ) -> int:
        """Numba-accelerated calculation of holes."""
        holes = 0
        for c in range(cols):
            height = heights[c]
            if height > 0:
                for r in range(height):  # Iterate up to height-1
                    if not is_death[r, c] and not is_occupied[r, c]:
                        holes += 1
        return holes

    def count_holes(self) -> int:
        """Counts empty, non-death cells below the highest occupied cell in the same column using Numba."""
        heights_np = np.array(self.get_column_heights(), dtype=np.int32)
        # Use the internal numpy arrays directly
        return self._numba_count_holes(
            self.rows, self.cols, heights_np, self._occupied_np, self._death_np
        )

    def get_feature_matrix(self) -> np.ndarray:
        """Returns the grid state as a 2-channel numpy array (Occupancy, Orientation)."""
        # Use internal _occupied_np for the first channel
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        grid_state[0, :, :] = self._occupied_np.astype(np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if not self._death_np[r, c]:
                    grid_state[1, r, c] = 1.0 if self.triangles[r][c].is_up else -1.0
        return grid_state

    def get_color_data(self) -> List[List[Optional[Tuple[int, int, int]]]]:
        """Returns a 2D list of colors for occupied cells."""
        color_data = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                # Check internal numpy array first
                if self._occupied_np[r, c] and not self._death_np[r, c]:
                    color_data[r][c] = self.triangles[r][c].color
        return color_data

    def get_death_data(self) -> np.ndarray:
        """Returns a boolean numpy array indicating death cells (uses internal array)."""
        return self._death_np.copy()  # Return a copy to prevent external modification

    def deepcopy_grid(self) -> "Grid":
        """Creates a deep copy of the grid, including Triangle objects and numpy arrays."""
        new_grid = Grid.__new__(Grid)  # Create instance without calling __init__
        new_grid.rows = self.rows
        new_grid.cols = self.cols

        # Deep copy the list of lists of Triangles
        new_grid.triangles = [[tri.copy() for tri in row] for row in self.triangles]

        # Re-link neighbors within the new grid
        new_grid._link_neighbors()

        # Copy the numpy arrays
        new_grid._occupied_np = self._occupied_np.copy()
        new_grid._death_np = self._death_np.copy()

        # Rebuild the lines index based on the *new* triangle objects
        new_grid._initialize_lines_and_index()

        return new_grid


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

    def copy(self) -> "Shape":
        """Creates a shallow copy of the Shape object."""
        new_shape = Shape.__new__(
            Shape
        )  # Create a new instance without calling __init__
        new_shape.triangles = list(
            self.triangles
        )  # Copy the list (tuples inside are immutable)
        new_shape.color = self.color  # Copy the color tuple reference
        return new_shape


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
        # Neighbors based on shared edges - these will be linked by the Grid
        self.neighbor_left: Optional["Triangle"] = None
        self.neighbor_right: Optional["Triangle"] = None
        self.neighbor_vert: Optional["Triangle"] = None

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

    def copy(self) -> "Triangle":
        """Creates a copy of the Triangle object's state, excluding neighbors."""
        new_tri = Triangle.__new__(Triangle)  # Create instance without calling __init__
        new_tri.row = self.row
        new_tri.col = self.col
        new_tri.is_up = self.is_up
        new_tri.is_death = self.is_death
        new_tri.is_occupied = self.is_occupied
        new_tri.color = self.color  # Copy color reference (tuple is immutable)
        # Neighbors are intentionally set to None, they will be re-linked by Grid.deepcopy_grid
        new_tri.neighbor_left = None
        new_tri.neighbor_right = None
        new_tri.neighbor_vert = None
        return new_tri


File: environment\__init__.py


File: mcts\config.py
class MCTSConfig:
    """Configuration parameters for the Monte Carlo Tree Search."""

    # Exploration constant (c_puct in PUCT formula)
    # Balances exploitation (Q value) and exploration (prior P and visit counts N)
    # Higher values encourage exploring less-visited actions with high priors.
    PUCT_C: float = 1.5

    # Number of MCTS simulations to run for each move decision.
    # More simulations generally lead to stronger play but take more time.
    NUM_SIMULATIONS: int = 100

    # Temperature parameter for action selection during self-play.
    # Controls the randomness of move selection based on visit counts.
    # Higher temperature -> more exploration (sample proportionally to N^(1/temp))
    # Lower temperature -> more exploitation (closer to choosing the most visited action)
    # Often starts high (e.g., 1.0) and anneals to a small value (e.g., 0.1 or 0) during the game.
    TEMPERATURE_INITIAL: float = 1.0
    TEMPERATURE_FINAL: float = 0.01
    TEMPERATURE_ANNEAL_STEPS: int = (
        30  # Number of game steps over which to anneal temperature
    )

    # Dirichlet noise parameters for exploration at the root node during self-play.
    # Adds noise to the prior probabilities from the network to encourage exploration,
    # especially early in training.
    # Alpha determines the shape of the distribution, Epsilon the weight of the noise.
    DIRICHLET_ALPHA: float = 0.3
    DIRICHLET_EPSILON: float = 0.25

    # Maximum depth for the MCTS search tree (optional, can prevent excessive depth)
    MAX_SEARCH_DEPTH: int = 100


File: mcts\node.py
import math
import numpy as np
from typing import Dict, Optional, TYPE_CHECKING, Any, List

# Assuming GameState is hashable or identifiable
from environment.game_state import GameState
from utils.types import ActionType
from config import MCTSConfig  # Import from config package


class MCTSNode:
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(
        self,
        game_state: GameState,
        parent: Optional["MCTSNode"] = None,
        action_taken: Optional[ActionType] = None,
        prior: float = 0.0,
        config: Optional[MCTSConfig] = None,  # Pass config for PUCT_C
    ):
        self.game_state = game_state
        self.parent = parent
        self.action_taken = action_taken

        self.children: Dict[ActionType, "MCTSNode"] = {}
        self.is_expanded: bool = False
        self.is_terminal: bool = game_state.is_over()

        self.visit_count: int = 0
        self.total_action_value: float = 0.0
        self.mean_action_value: float = 0.0
        self.prior: float = prior

        self._config = config if config else MCTSConfig()  # Use default if None

    def get_ucb_score(self) -> float:
        """Calculates the PUCT score for this node (from the perspective of its parent)."""
        if self.parent is None:
            return self.mean_action_value  # Root node score

        exploration_bonus = (
            self._config.PUCT_C
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        q_value = self.mean_action_value
        return q_value + exploration_bonus

    def select_best_child(self) -> "MCTSNode":
        """Selects the child with the highest UCB score."""
        if not self.children:
            raise ValueError("Cannot select best child from a node with no children.")
        # Simple way to handle potential ties: add small random noise or just pick first max
        best_score = -float("inf")
        best_children = []
        for child in self.children.values():
            score = child.get_ucb_score()
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        if not best_children:
            raise RuntimeError("Could not select a best child node.")
        # Randomly pick among the best children in case of ties
        return np.random.choice(best_children)

    def backpropagate(self, value: float):
        """Updates the visit count and action value of this node and its ancestors."""
        node: Optional[MCTSNode] = self
        while node is not None:
            node.visit_count += 1
            node.total_action_value += value
            node.mean_action_value = node.total_action_value / node.visit_count
            node = node.parent


File: mcts\search.py
# File: mcts/search.py
import numpy as np
import time
import copy  # Keep for potential future use, but avoid for GameState
from typing import Dict, Optional, Tuple, Callable, Any, List
import logging
import torch  # Added for batching
import threading  # Added for stop_event

from environment.game_state import GameState  # Import GameState directly
from utils.types import ActionType, StateType
from .node import MCTSNode
from config import MCTSConfig, EnvConfig


# Updated signature for batch prediction
NetworkPredictor = Callable[
    [List[StateType]], Tuple[List[Dict[ActionType, float]], List[float]]
]
logger = logging.getLogger(__name__)


class MCTS:
    """Monte Carlo Tree Search implementation based on AlphaZero principles with batching."""

    def __init__(
        self,
        network_predictor: NetworkPredictor,  # Predictor now handles batches
        config: Optional[MCTSConfig] = None,
        env_config: Optional[EnvConfig] = None,
        batch_size: int = 8,  # Default MCTS batch size
        stop_event: Optional[threading.Event] = None,  # Add stop_event
    ):
        self.network_predictor = network_predictor
        self.config = config if config else MCTSConfig()
        self.env_config = env_config if env_config else EnvConfig()
        self.batch_size = batch_size  # Store batch size
        self.stop_event = stop_event  # Store stop_event
        self.log_prefix = "[MCTS]"
        logger.info(
            f"{self.log_prefix} Initialized with NN batch size: {self.batch_size}"
        )

    def _select_leaf(self, root_node: MCTSNode) -> Tuple[MCTSNode, int]:
        """Traverses the tree using PUCT until a leaf node is reached. Returns node and depth."""
        node = root_node
        depth = 0
        while node.is_expanded and not node.is_terminal:
            if self.stop_event and self.stop_event.is_set():
                raise InterruptedError("MCTS selection interrupted by stop event.")
            if depth >= self.config.MAX_SEARCH_DEPTH:
                break
            if not node.children:
                break
            node = node.select_best_child()
            depth += 1
        return node, depth

    def _expand_and_backpropagate_batch(
        self, nodes_to_expand: List[MCTSNode]
    ) -> Tuple[float, int]:
        """
        Expands a batch of leaf nodes using batched NN prediction and backpropagates results.
        Returns (total_nn_prediction_time, nodes_created_count).
        """
        if not nodes_to_expand:
            return 0.0, 0
        if self.stop_event and self.stop_event.is_set():
            raise InterruptedError("MCTS expansion interrupted by stop event.")

        batch_states = [node.game_state.get_state() for node in nodes_to_expand]
        total_nn_prediction_time = 0.0
        nodes_created_count = 0

        try:
            start_pred_time = time.monotonic()
            policy_probs_list, predicted_values = self.network_predictor(batch_states)
            total_nn_prediction_time = time.monotonic() - start_pred_time
            logger.info(
                f"{self.log_prefix} Batched NN Prediction ({len(batch_states)} states) took {total_nn_prediction_time:.4f}s."
            )
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Error during batched network prediction: {e}",
                exc_info=True,
            )
            for node in nodes_to_expand:
                if not node.is_expanded:
                    node.is_expanded = True
                    node.backpropagate(0.0)
            return total_nn_prediction_time, 0

        for i, node in enumerate(nodes_to_expand):
            if self.stop_event and self.stop_event.is_set():
                raise InterruptedError(
                    "MCTS expansion processing interrupted by stop event."
                )

            if node.is_expanded or node.is_terminal:
                if node.visit_count == 0:
                    node.backpropagate(predicted_values[i])
                continue

            policy_probs_dict = policy_probs_list[i]
            predicted_value = predicted_values[i]
            children_created_count_node = 0

            valid_actions = node.game_state.valid_actions()
            if not valid_actions:
                node.is_expanded = True
                node.is_terminal = True
                node.backpropagate(predicted_value)
                continue

            parent_state = node.game_state
            start_expand_time = time.monotonic()
            for action in valid_actions:
                try:
                    # --- Create new state and apply action ---
                    # This uses the grid's deepcopy method, which is necessary for now.
                    child_state = GameState()  # Create a fresh GameState
                    child_state.grid = (
                        parent_state.grid.deepcopy_grid()
                    )  # Use grid's deepcopy
                    child_state.shapes = [
                        s.copy() if s else None for s in parent_state.shapes
                    ]
                    child_state.game_score = parent_state.game_score
                    child_state.triangles_cleared_this_episode = (
                        parent_state.triangles_cleared_this_episode
                    )
                    child_state.pieces_placed_this_episode = (
                        parent_state.pieces_placed_this_episode
                    )
                    # Apply step to the *new* child_state
                    _, done = child_state.step(action)
                    # --- End state modification ---
                    logger.info(
                        f"{self.log_prefix} Created child state for action {action}. Parent score: {parent_state.game_score}, Child score: {child_state.game_score}, Done: {done}"
                    )

                    prior_prob = policy_probs_dict.get(action, 0.0)
                    child_node = MCTSNode(
                        game_state=child_state,
                        parent=node,
                        action_taken=action,
                        prior=prior_prob,
                        config=self.config,
                    )
                    if done:
                        child_node.is_terminal = True

                    node.children[action] = child_node
                    children_created_count_node += 1
                except Exception as child_creation_err:
                    logger.error(
                        f"{self.log_prefix} Error creating child for action {action}: {child_creation_err}",
                        exc_info=True,
                    )
                    continue
            expand_duration = time.monotonic() - start_expand_time
            logger.info(
                f"{self.log_prefix} Node expansion ({children_created_count_node} children) took {expand_duration:.4f}s."
            )

            node.is_expanded = True
            nodes_created_count += children_created_count_node
            node.backpropagate(predicted_value)

        return total_nn_prediction_time, nodes_created_count

    def run_simulations(
        self, root_state: GameState, num_simulations: int
    ) -> Tuple[MCTSNode, Dict[str, Any]]:
        """
        Runs the MCTS process for a given number of simulations using batching.
        Returns the root node and a dictionary of simulation statistics.
        """
        try:
            # --- Create a copy of the root state for the search ---
            search_root_state = GameState()
            search_root_state.grid = (
                root_state.grid.deepcopy_grid()
            )  # Use grid's deepcopy
            search_root_state.shapes = [
                s.copy() if s else None for s in root_state.shapes
            ]
            search_root_state.game_score = root_state.game_score
            search_root_state.triangles_cleared_this_episode = (
                root_state.triangles_cleared_this_episode
            )
            search_root_state.pieces_placed_this_episode = (
                root_state.pieces_placed_this_episode
            )
            search_root_state.game_over = root_state.game_over
            # --- End state copy ---
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Error copying root game state for MCTS: {e}",
                exc_info=True,
            )
            return MCTSNode(game_state=root_state, config=self.config), {
                "simulations_run": 0,
                "mcts_total_duration": 0.0,
                "total_nn_prediction_time": 0.0,
                "nodes_created": 0,
                "avg_leaf_depth": 0.0,
                "root_visits": 0,
            }

        root_node = MCTSNode(game_state=search_root_state, config=self.config)
        sim_start_time = time.monotonic()
        total_nn_prediction_time = 0.0
        nodes_created_this_run = 1
        total_leaf_depth = 0
        simulations_run = 0

        if root_node.is_terminal:
            logger.warning(
                f"{self.log_prefix} Root node is terminal. No simulations run."
            )
            return root_node, {
                "simulations_run": 0,
                "mcts_total_duration": 0.0,
                "total_nn_prediction_time": 0.0,
                "nodes_created": 1,
                "avg_leaf_depth": 0.0,
                "root_visits": 0,
            }

        try:
            initial_batch_time, initial_nodes_created = (
                self._expand_and_backpropagate_batch([root_node])
            )
            total_nn_prediction_time += initial_batch_time
            nodes_created_this_run += initial_nodes_created
            simulations_run += 1

            if root_node.is_expanded and not root_node.is_terminal:
                self._add_dirichlet_noise(root_node)

            leaves_to_expand: List[MCTSNode] = []
            for sim_num in range(simulations_run, num_simulations):
                if self.stop_event and self.stop_event.is_set():
                    logger.info(
                        f"{self.log_prefix} Stop event detected during simulation {sim_num+1}. Stopping MCTS."
                    )
                    break

                sim_start_step = time.monotonic()
                leaf_node, depth = self._select_leaf(root_node)
                total_leaf_depth += depth

                if leaf_node.is_terminal:
                    value = leaf_node.game_state.get_outcome()
                    leaf_node.backpropagate(value)
                    sim_duration_step = time.monotonic() - sim_start_step
                    logger.info(
                        f"{self.log_prefix} Sim {sim_num+1}/{num_simulations} hit terminal node. Backprop: {value:.2f}. Took {sim_duration_step:.5f}s"
                    )
                    continue

                leaves_to_expand.append(leaf_node)

                if (
                    len(leaves_to_expand) >= self.batch_size
                    or sim_num == num_simulations - 1
                ):
                    if leaves_to_expand:
                        batch_nn_time, batch_nodes_created = (
                            self._expand_and_backpropagate_batch(leaves_to_expand)
                        )
                        total_nn_prediction_time += batch_nn_time
                        nodes_created_this_run += batch_nodes_created
                        leaves_to_expand = []

            if leaves_to_expand:
                batch_nn_time, batch_nodes_created = (
                    self._expand_and_backpropagate_batch(leaves_to_expand)
                )
                total_nn_prediction_time += batch_nn_time
                nodes_created_this_run += batch_nodes_created

        except InterruptedError as e:
            logger.warning(f"{self.log_prefix} MCTS run interrupted: {e}")
            pass

        sim_duration_total = time.monotonic() - sim_start_time
        effective_sims = max(1, root_node.visit_count)
        avg_leaf_depth = (
            total_leaf_depth / effective_sims if effective_sims > 0 else 0.0
        )

        logger.info(
            f"{self.log_prefix} Finished {root_node.visit_count} effective simulations in {sim_duration_total:.4f}s. "
            f"Nodes created: {nodes_created_this_run}, "
            f"Total NN time: {total_nn_prediction_time:.4f}s, Avg Depth: {avg_leaf_depth:.1f}"
        )

        mcts_stats = {
            "simulations_run": root_node.visit_count,
            "mcts_total_duration": sim_duration_total,
            "total_nn_prediction_time": total_nn_prediction_time,
            "nodes_created": nodes_created_this_run,
            "avg_leaf_depth": avg_leaf_depth,
            "root_visits": root_node.visit_count,
        }
        return root_node, mcts_stats

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Adds Dirichlet noise to the prior probabilities of the root node's children."""
        if not node.children or self.config.DIRICHLET_ALPHA <= 0:
            return
        actions = list(node.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA] * len(actions))
        eps = self.config.DIRICHLET_EPSILON
        for i, action in enumerate(actions):
            child = node.children.get(action)
            if child:
                child.prior = (1 - eps) * child.prior + eps * noise[i]
        logger.info(f"{self.log_prefix} Applied Dirichlet noise to root node priors.")

    def get_policy_target(
        self, root_node: MCTSNode, temperature: float
    ) -> Dict[ActionType, float]:
        """Calculates the improved policy distribution based on visit counts."""
        if not root_node.children:
            return {}
        total_visits = sum(child.visit_count for child in root_node.children.values())
        if total_visits == 0:
            num_children = len(root_node.children)
            return (
                {a: 1.0 / num_children for a in root_node.children}
                if num_children > 0
                else {}
            )

        policy_target: Dict[ActionType, float] = {}
        if temperature == 0:
            best_action, max_visits = -1, -1
            for action, child in root_node.children.items():
                if child.visit_count > max_visits:
                    max_visits, best_action = child.visit_count, action
            if best_action != -1:
                for action in root_node.children:
                    policy_target[action] = 1.0 if action == best_action else 0.0
            else:
                num_children = len(root_node.children)
                prob = 1.0 / num_children if num_children > 0 else 0.0
                for action in root_node.children:
                    policy_target[action] = prob
        else:
            total_power, powered_counts = 0.0, {}
            for action, child in root_node.children.items():
                visit_count = max(0, child.visit_count)
                try:
                    powered_count = np.power(
                        np.float64(visit_count), 1.0 / temperature, dtype=np.float64
                    )
                    if np.isinf(powered_count):
                        logger.warning(
                            f"{self.log_prefix} Infinite powered count encountered for action {action} (visits={visit_count}, temp={temperature}). Clamping."
                        )
                        powered_count = np.finfo(np.float64).max / len(
                            root_node.children
                        )
                except (OverflowError, ValueError):
                    logger.warning(
                        f"{self.log_prefix} Power calculation overflow/error for action {action} (visits={visit_count}, temp={temperature}). Setting to 0."
                    )
                    powered_count = 0.0
                powered_counts[action] = powered_count
                if not np.isinf(powered_count):
                    total_power += powered_count

            if total_power <= 1e-9 or np.isinf(total_power):
                if np.isinf(total_power):
                    inf_actions = [
                        a
                        for a, pc in powered_counts.items()
                        if np.isinf(pc)
                        or pc > np.finfo(np.float64).max / (len(root_node.children) * 2)
                    ]
                    num_inf = len(inf_actions)
                    prob = 1.0 / num_inf if num_inf > 0 else 0.0
                    for action in root_node.children:
                        policy_target[action] = prob if action in inf_actions else 0.0
                    logger.warning(
                        f"{self.log_prefix} Total power infinite, assigned uniform prob to {num_inf} actions."
                    )
                else:
                    visited_children = [
                        a for a, c in root_node.children.items() if c.visit_count > 0
                    ]
                    num_visited = len(visited_children)
                    if num_visited > 0:
                        prob = 1.0 / num_visited
                        for action in root_node.children:
                            policy_target[action] = (
                                prob if action in visited_children else 0.0
                            )
                    elif root_node.children:
                        prob = 1.0 / len(root_node.children)
                        for action in root_node.children:
                            policy_target[action] = prob
                    else:
                        prob = 0.0
                    logger.warning(
                        f"{self.log_prefix} Total power near zero ({total_power}), assigned uniform prob."
                    )
            else:
                for action, powered_count in powered_counts.items():
                    policy_target[action] = float(powered_count / total_power)

        full_policy = np.zeros(self.env_config.ACTION_DIM, dtype=np.float32)
        policy_sum_check = 0.0
        for action, prob in policy_target.items():
            if 0 <= action < self.env_config.ACTION_DIM:
                full_policy[action] = prob
                policy_sum_check += prob
            else:
                logger.warning(
                    f"{self.log_prefix} MCTS produced invalid action index {action} in policy target."
                )

        if not np.isclose(policy_sum_check, 1.0, atol=1e-4):
            logger.warning(
                f"{self.log_prefix} Policy target sum is {policy_sum_check:.4f} before final conversion. Renormalizing."
            )
            current_sum = np.sum(full_policy)
            if current_sum > 1e-6:
                full_policy /= current_sum
            else:
                valid_actions = root_node.game_state.valid_actions()
                num_valid = len(valid_actions)
                if num_valid > 0:
                    prob = 1.0 / num_valid
                    full_policy.fill(0.0)
                    for action in valid_actions:
                        if 0 <= action < self.env_config.ACTION_DIM:
                            full_policy[action] = prob

        return {i: float(prob) for i, prob in enumerate(full_policy)}

    def choose_action(self, root_node: MCTSNode, temperature: float) -> ActionType:
        """Chooses an action based on MCTS visit counts and temperature."""
        policy_dict = self.get_policy_target(root_node, temperature)
        valid_actions_list = root_node.game_state.valid_actions()
        valid_actions_set = set(valid_actions_list)

        if not policy_dict or not valid_actions_list:
            if valid_actions_list:
                logger.warning(
                    f"{self.log_prefix} Policy dict empty/invalid, choosing random valid action."
                )
                return np.random.choice(valid_actions_list)
            else:
                logger.error(
                    f"{self.log_prefix} MCTS failed: no policy and no valid actions."
                )
                return -1

        filtered_actions, filtered_probs = [], []
        for action, prob in policy_dict.items():
            if action in valid_actions_set and prob > 1e-7:
                filtered_actions.append(action)
                filtered_probs.append(prob)

        if not filtered_actions:
            logger.warning(
                f"{self.log_prefix} MCTS policy effectively zero for all valid actions. Choosing uniformly among valid."
            )
            return np.random.choice(valid_actions_list)

        actions = np.array(filtered_actions)
        probabilities = np.array(filtered_probs, dtype=np.float64)

        prob_sum = np.sum(probabilities)
        if prob_sum <= 1e-7:
            logger.warning(
                f"{self.log_prefix} Filtered policy sum near zero ({prob_sum}). Choosing uniformly among filtered."
            )
            return np.random.choice(actions)

        probabilities /= prob_sum

        try:
            chosen_action = np.random.choice(actions, p=probabilities)
            return int(chosen_action)
        except ValueError as e:
            logger.error(
                f"{self.log_prefix} Error during np.random.choice: {e}. Probabilities sum: {np.sum(probabilities)}. Actions: {actions}. Probs: {probabilities}"
            )
            return np.random.choice(actions)


File: mcts\__init__.py
from .config import MCTSConfig
from .node import MCTSNode
from .search import MCTS

__all__ = ["MCTSConfig", "MCTSNode", "MCTS"]


File: stats\aggregator.py
# File: stats/aggregator.py
import time
from typing import (
    Deque,
    Dict,
    Any,
    Optional,
    List,
    TYPE_CHECKING,
)
import threading
import logging

from config import StatsConfig
from .aggregator_storage import AggregatorStorage
from .aggregator_logic import AggregatorLogic

if TYPE_CHECKING:
    from environment.game_state import GameState

logger = logging.getLogger(__name__)


class StatsAggregator:
    """
    Handles aggregation and storage of training statistics using deques.
    Calculates rolling averages and tracks best values. Does not perform logging.
    Includes locks for thread safety. Delegates storage and logic to helper classes.
    Refactored for clarity and AlphaZero focus.
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
        episode_outcome: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional["GameState"] = None,
    ) -> Dict[str, Any]:
        """Records episode stats and checks for new bests."""
        with self._lock:
            current_step = (
                global_step
                if global_step is not None
                else self.storage.current_global_step
            )
            update_info = self.logic.update_episode_stats(
                episode_outcome,
                episode_length,
                episode_num,
                current_step,
                game_score,
                triangles_cleared,
                game_state_for_best,
            )
            return update_info

    def record_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Records step data (e.g., NN training step, MCTS step) and checks for new bests.
        Handles both training worker and self-play worker data.
        """
        with self._lock:
            g_step = step_data.get("global_step")
            if g_step is not None and g_step > self.storage.current_global_step:
                self.storage.current_global_step = g_step
                logger.info(
                    f"[StatsAggregator] Global step updated to {g_step} from step_data."
                )
            elif g_step is None:
                # Use current step if not provided (e.g., for MCTS stats from self-play)
                g_step = self.storage.current_global_step
                logger.info(
                    f"[StatsAggregator] Using existing global step {g_step} for step_data."
                )

            # Explicitly pass MCTS stats if present
            mcts_sim_time = step_data.get("mcts_sim_time")
            mcts_nn_time = step_data.get("mcts_nn_time")
            mcts_nodes = step_data.get("mcts_nodes_explored")
            mcts_depth = step_data.get("mcts_avg_depth")

            update_info = self.logic.update_step_stats(
                step_data,
                g_step,
                mcts_sim_time=mcts_sim_time,
                mcts_nn_time=mcts_nn_time,
                mcts_nodes_explored=mcts_nodes,
                mcts_avg_depth=mcts_depth,
            )
            return update_info

    def get_summary(self, current_global_step: Optional[int] = None) -> Dict[str, Any]:
        """Calculates and returns the summary dictionary."""
        with self._lock:
            if current_global_step is None:
                current_global_step = self.storage.current_global_step
            summary = self.logic.calculate_summary(
                current_global_step, self.summary_avg_window
            )
            return summary

    def get_plot_data(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        with self._lock:
            return self.storage.get_all_plot_deques()

    def get_best_game_state_data(self) -> Optional[Dict[str, Any]]:
        """Returns the data needed to render the best game state found."""
        with self._lock:
            return (
                self.storage.best_game_state_data.copy()
                if self.storage.best_game_state_data
                else None
            )

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state for checkpointing."""
        with self._lock:
            state = self.storage.state_dict()
            state["plot_window"] = self.plot_window
            state["avg_windows"] = self.avg_windows
            return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the state from a checkpoint."""
        with self._lock:
            print("[StatsAggregator] Loading state...")
            self.plot_window = state_dict.get("plot_window", self.plot_window)
            self.avg_windows = state_dict.get("avg_windows", self.avg_windows)
            self.summary_avg_window = self.avg_windows[0] if self.avg_windows else 100

            self.storage.load_state_dict(state_dict, self.plot_window)

            print("[StatsAggregator] State loaded.")
            print(f"  -> Loaded total_episodes: {self.storage.total_episodes}")
            # print(f"  -> Loaded best_outcome: {self.storage.best_outcome}") # Less relevant now
            print(f"  -> Loaded best_game_score: {self.storage.best_game_score}")
            print(
                f"  -> Loaded start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.storage.start_time))}"
            )
            print(
                f"  -> Loaded current_global_step: {self.storage.current_global_step}"
            )
            if self.storage.best_game_state_data:
                print(
                    f"  -> Loaded best_game_state_data (Score: {self.storage.best_game_state_data.get('score', 'N/A')})"
                )
            else:
                print("  -> No best_game_state_data found in loaded state.")


File: stats\aggregator_logic.py
# File: stats/aggregator_logic.py
import time
import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging

from .aggregator_storage import AggregatorStorage

if TYPE_CHECKING:
    from environment.game_state import GameState

logger = logging.getLogger(__name__)


class AggregatorLogic:
    """Handles the logic for updating stats and calculating summaries."""

    def __init__(self, storage: AggregatorStorage):
        self.storage = storage

    def update_episode_stats(
        self,
        episode_outcome: float,
        episode_length: int,
        episode_num: int,
        current_step: int,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional[
            Dict[str, np.ndarray]
        ] = None,  # Expect state dict
    ) -> Dict[str, Any]:
        """Updates deques and best values related to episode completion."""
        update_info = {"new_best_game": False}
        self.storage.total_episodes += 1
        self.storage.episode_outcomes.append(episode_outcome)
        self.storage.episode_lengths.append(episode_length)

        if game_score is not None:
            self.storage.game_scores.append(game_score)
            if game_score > self.storage.best_game_score:
                self.storage.previous_best_game_score = self.storage.best_game_score
                self.storage.best_game_score = game_score
                self.storage.best_game_score_step = current_step
                self.storage.best_game_score_history.append(game_score)
                update_info["new_best_game"] = True
                # Store best game state data if provided (now expects state dict)
                if game_state_for_best and isinstance(game_state_for_best, dict):
                    self.storage.best_game_state_data = {
                        "score": game_score,
                        "step": current_step,
                        "game_state_dict": game_state_for_best,  # Store the state dict directly
                    }
                elif game_state_for_best:  # Log warning if wrong type passed
                    logger.warning(
                        f"Received game_state_for_best of type {type(game_state_for_best)}, expected dict. Skipping storage."
                    )
                    self.storage.best_game_state_data = (
                        None  # Clear previous best if new best has invalid state
                    )
                else:
                    self.storage.best_game_state_data = (
                        None  # Clear if no state provided for new best
                    )

            elif self.storage.best_game_score > -float("inf"):
                self.storage.best_game_score_history.append(
                    self.storage.best_game_score
                )

        if triangles_cleared is not None:
            self.storage.episode_triangles_cleared.append(triangles_cleared)
            self.storage.total_triangles_cleared += triangles_cleared

        # Update best outcome (less relevant now)
        if episode_outcome > self.storage.best_outcome:
            self.storage.previous_best_outcome = self.storage.best_outcome
            self.storage.best_outcome = episode_outcome
            self.storage.best_outcome_step = current_step
            update_info["new_best_outcome"] = True

        return update_info

    def update_step_stats(
        self,
        step_data: Dict[str, Any],
        g_step: int,
        mcts_sim_time: Optional[float] = None,
        mcts_nn_time: Optional[float] = None,
        mcts_nodes_explored: Optional[int] = None,
        mcts_avg_depth: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Updates deques and best values related to individual steps (training or MCTS)."""
        update_info = {
            "new_best_value_loss": False,
            "new_best_policy_loss": False,
            "new_best_mcts_sim_time": False,
        }

        # --- Log Global Step Update ---
        # Use INFO level as requested
        logger.info(
            f"[AggLogic] update_step_stats called. Incoming g_step: {g_step}, Stored step before update: {self.storage.current_global_step}"
        )
        if g_step > self.storage.current_global_step:
            self.storage.current_global_step = g_step
            logger.info(
                f"[AggLogic] Updated storage.current_global_step to: {self.storage.current_global_step}"
            )
        # --- End Log ---

        # Training Stats
        policy_loss = step_data.get("policy_loss")
        value_loss = step_data.get("value_loss")
        lr = step_data.get("lr")

        if policy_loss is not None and np.isfinite(policy_loss):
            self.storage.policy_losses.append(policy_loss)
            if policy_loss < self.storage.best_policy_loss:
                self.storage.previous_best_policy_loss = self.storage.best_policy_loss
                self.storage.best_policy_loss = policy_loss
                self.storage.best_policy_loss_step = g_step
                update_info["new_best_policy_loss"] = True

        if value_loss is not None and np.isfinite(value_loss):
            self.storage.value_losses.append(value_loss)
            if value_loss < self.storage.best_value_loss:
                self.storage.previous_best_value_loss = self.storage.best_value_loss
                self.storage.best_value_loss = value_loss
                self.storage.best_value_loss_step = g_step
                update_info["new_best_value_loss"] = True

        if lr is not None and np.isfinite(lr):
            self.storage.lr_values.append(lr)
            self.storage.current_lr = lr

        # MCTS Stats
        if mcts_sim_time is not None and np.isfinite(mcts_sim_time):
            self.storage.mcts_simulation_times.append(mcts_sim_time)
            if mcts_sim_time < self.storage.best_mcts_sim_time:
                self.storage.previous_best_mcts_sim_time = (
                    self.storage.best_mcts_sim_time
                )
                self.storage.best_mcts_sim_time = mcts_sim_time
                self.storage.best_mcts_sim_time_step = g_step
                update_info["new_best_mcts_sim_time"] = True

        if mcts_nn_time is not None and np.isfinite(mcts_nn_time):
            self.storage.mcts_nn_prediction_times.append(mcts_nn_time)
        if mcts_nodes_explored is not None and np.isfinite(mcts_nodes_explored):
            self.storage.mcts_nodes_explored.append(mcts_nodes_explored)
        if mcts_avg_depth is not None and np.isfinite(mcts_avg_depth):
            self.storage.mcts_avg_depths.append(mcts_avg_depth)

        # System Stats
        buffer_size = step_data.get("buffer_size")
        if buffer_size is not None:
            self.storage.buffer_sizes.append(buffer_size)
            self.storage.current_buffer_size = buffer_size

        self.storage.update_steps_per_second(g_step)

        # Intermediate Progress
        current_game = step_data.get("current_self_play_game_number")
        current_game_step = step_data.get("current_self_play_game_steps")
        training_steps = step_data.get("training_steps_performed")

        if current_game is not None:
            self.storage.current_self_play_game_number = current_game
        if current_game_step is not None:
            self.storage.current_self_play_game_steps = current_game_step
        if training_steps is not None:
            self.storage.training_steps_performed = training_steps

        return update_info

    def _calculate_average(self, deque, window_size):
        """Calculates the average of the last 'window_size' elements in a deque."""
        if not deque:
            return 0.0
        actual_window = min(window_size, len(deque))
        items = list(deque)[-actual_window:]
        return np.mean(items) if items else 0.0

    def calculate_summary(
        self, current_global_step: int, avg_window: int
    ) -> Dict[str, Any]:
        """Calculates the summary dictionary based on stored data."""
        summary = {}
        summary["global_step"] = current_global_step
        summary["total_episodes"] = self.storage.total_episodes
        summary["start_time"] = self.storage.start_time
        summary["runtime_seconds"] = time.time() - self.storage.start_time
        summary["buffer_size"] = self.storage.current_buffer_size
        summary["current_lr"] = self.storage.current_lr
        summary["summary_avg_window_size"] = avg_window

        summary["avg_episode_length"] = self._calculate_average(
            self.storage.episode_lengths, avg_window
        )
        summary["avg_game_score_window"] = self._calculate_average(
            self.storage.game_scores, avg_window
        )
        summary["avg_triangles_cleared"] = self._calculate_average(
            self.storage.episode_triangles_cleared, avg_window
        )
        summary["policy_loss"] = self._calculate_average(
            self.storage.policy_losses, avg_window
        )
        summary["value_loss"] = self._calculate_average(
            self.storage.value_losses, avg_window
        )
        summary["mcts_simulation_time_avg"] = self._calculate_average(
            self.storage.mcts_simulation_times, avg_window
        )
        summary["mcts_nn_prediction_time_avg"] = self._calculate_average(
            self.storage.mcts_nn_prediction_times, avg_window
        )
        summary["mcts_nodes_explored_avg"] = self._calculate_average(
            self.storage.mcts_nodes_explored, avg_window
        )
        summary["steps_per_second_avg"] = self._calculate_average(
            self.storage.steps_per_second, avg_window
        )

        summary["best_game_score"] = self.storage.best_game_score
        summary["best_game_score_step"] = self.storage.best_game_score_step
        summary["best_value_loss"] = self.storage.best_value_loss
        summary["best_value_loss_step"] = self.storage.best_value_loss_step
        summary["best_policy_loss"] = self.storage.best_policy_loss
        summary["best_policy_loss_step"] = self.storage.best_policy_loss_step
        summary["best_mcts_sim_time"] = self.storage.best_mcts_sim_time
        summary["best_mcts_sim_time_step"] = self.storage.best_mcts_sim_time_step

        summary["current_self_play_game_number"] = (
            self.storage.current_self_play_game_number
        )
        summary["current_self_play_game_steps"] = (
            self.storage.current_self_play_game_steps
        )
        summary["training_steps_performed"] = self.storage.training_steps_performed
        summary["training_target_step"] = self.storage.training_target_step

        return summary


File: stats\aggregator_storage.py
# File: stats/aggregator_storage.py
from collections import deque
from typing import Deque, Dict, Any, List, Optional
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AggregatorStorage:
    """Holds the data structures (deques and scalar values) for StatsAggregator.
    Refactored for AlphaZero focus. Resource usage removed."""

    def __init__(self, plot_window: int):
        self.plot_window = plot_window

        # --- Deques for Plotting ---
        # Training Stats
        self.policy_losses: Deque[float] = deque(maxlen=plot_window)
        self.value_losses: Deque[float] = deque(maxlen=plot_window)
        self.lr_values: Deque[float] = deque(maxlen=plot_window)
        # Game Stats
        self.episode_outcomes: Deque[float] = deque(maxlen=plot_window)  # -1, 0, 1
        self.episode_lengths: Deque[int] = deque(maxlen=plot_window)
        self.game_scores: Deque[int] = deque(maxlen=plot_window)
        self.episode_triangles_cleared: Deque[int] = deque(maxlen=plot_window)
        self.best_game_score_history: Deque[int] = deque(maxlen=plot_window)
        # MCTS Stats
        self.mcts_simulation_times: Deque[float] = deque(maxlen=plot_window)
        self.mcts_nn_prediction_times: Deque[float] = deque(maxlen=plot_window)
        self.mcts_nodes_explored: Deque[int] = deque(maxlen=plot_window)
        self.mcts_avg_depths: Deque[float] = deque(maxlen=plot_window)
        # System Stats
        self.buffer_sizes: Deque[int] = deque(maxlen=plot_window)
        self.steps_per_second: Deque[float] = deque(
            maxlen=plot_window
        )  # Added steps/sec deque
        self._last_step_time: Optional[float] = None
        self._last_step_count: Optional[int] = None

        # --- Scalar State Variables ---
        self.total_episodes: int = 0
        self.total_triangles_cleared: int = 0
        self.current_buffer_size: int = 0
        self.current_global_step: int = 0
        self.current_lr: float = 0.0
        self.start_time: float = time.time()
        self.training_target_step: int = 0

        # --- Intermediate Progress Tracking ---
        self.current_self_play_game_number: int = 0
        self.current_self_play_game_steps: int = 0
        self.training_steps_performed: int = 0

        # --- Best Value Tracking ---
        self.best_outcome: float = -float("inf")  # Less relevant now
        self.previous_best_outcome: float = -float("inf")  # Less relevant now
        self.best_outcome_step: int = 0  # Less relevant now
        self.best_game_score: float = -float("inf")
        self.previous_best_game_score: float = -float("inf")
        self.best_game_score_step: int = 0
        self.best_value_loss: float = float("inf")
        self.previous_best_value_loss: float = float("inf")
        self.best_value_loss_step: int = 0
        self.best_policy_loss: float = float("inf")
        self.previous_best_policy_loss: float = float("inf")
        self.best_policy_loss_step: int = 0
        self.best_mcts_sim_time: float = float("inf")  # Best is lowest time
        self.previous_best_mcts_sim_time: float = float("inf")
        self.best_mcts_sim_time_step: int = 0

        # --- Best Game State Data ---
        self.best_game_state_data: Optional[Dict[str, Any]] = None

    def get_deque(self, name: str) -> Deque:
        """Safely gets a deque attribute."""
        return getattr(self, name, deque(maxlen=self.plot_window))

    def get_all_plot_deques(self) -> Dict[str, Deque]:
        """Returns copies of all deques intended for plotting."""
        deque_names = [
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "best_game_score_history",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "mcts_nodes_explored",
            "mcts_avg_depths",
            "buffer_sizes",
            "steps_per_second",
        ]
        return {
            name: self.get_deque(name).copy()
            for name in deque_names
            if hasattr(self, name)
        }

    def update_steps_per_second(self, global_step: int):
        """Calculates and updates the steps per second deque."""
        current_time = time.time()
        if self._last_step_time is not None and self._last_step_count is not None:
            time_diff = current_time - self._last_step_time
            step_diff = global_step - self._last_step_count
            if (
                time_diff > 1e-3 and step_diff > 0
            ):  # Avoid division by zero and stale data
                sps = step_diff / time_diff
                self.steps_per_second.append(sps)
            elif (
                step_diff <= 0 and time_diff > 1.0
            ):  # If no steps for a while, record 0
                self.steps_per_second.append(0.0)

        # Update last step time/count for the next calculation
        self._last_step_time = current_time
        self._last_step_count = global_step

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the storage for saving."""
        state = {}
        deque_names = [
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "best_game_score_history",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "mcts_nodes_explored",
            "mcts_avg_depths",
            "buffer_sizes",
            "steps_per_second",
        ]
        for name in deque_names:
            if hasattr(self, name):
                deque_instance = getattr(self, name, None)
                if deque_instance is not None:
                    state[name] = list(deque_instance)

        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_buffer_size",
            "current_global_step",
            "current_lr",
            "start_time",
            "training_target_step",
            "current_self_play_game_number",
            "current_self_play_game_steps",
            "training_steps_performed",
            "_last_step_time",
            "_last_step_count",  # Include for sps calculation resume
        ]
        for key in scalar_keys:
            state[key] = getattr(self, key, None if key.startswith("_last") else 0)

        best_value_keys = [
            "best_outcome",
            "previous_best_outcome",
            "best_outcome_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
            "best_policy_loss",
            "previous_best_policy_loss",
            "best_policy_loss_step",
            "best_mcts_sim_time",
            "previous_best_mcts_sim_time",
            "best_mcts_sim_time_step",
        ]
        for key in best_value_keys:
            default = (
                0
                if "step" in key
                else (
                    float("inf") if ("loss" in key or "time" in key) else -float("inf")
                )
            )
            state[key] = getattr(self, key, default)

        # Serialize best game state data carefully
        if self.best_game_state_data:
            try:
                # Convert GameState object to a serializable format (e.g., its state dict)
                serializable_data = {
                    "score": self.best_game_state_data.get("score"),
                    "step": self.best_game_state_data.get("step"),
                    # Add other relevant scalar info if needed
                }
                # Save the game state's internal state (which should be serializable)
                game_state_obj = self.best_game_state_data.get("game_state")
                if game_state_obj and hasattr(game_state_obj, "get_state"):
                    serializable_data["game_state_dict"] = game_state_obj.get_state()

                state["best_game_state_data"] = serializable_data
            except Exception as e:
                logger.error(f"Error serializing best_game_state_data: {e}")
                state["best_game_state_data"] = None
        else:
            state["best_game_state_data"] = None

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], plot_window: int):
        """Loads the state from a dictionary."""
        self.plot_window = plot_window
        deque_names = [
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_outcomes",
            "episode_lengths",
            "game_scores",
            "episode_triangles_cleared",
            "best_game_score_history",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "mcts_nodes_explored",
            "mcts_avg_depths",
            "buffer_sizes",
            "steps_per_second",
        ]
        for key in deque_names:
            data = state_dict.get(key)
            if isinstance(data, (list, tuple)):
                setattr(self, key, deque(data, maxlen=self.plot_window))
            else:
                # Initialize empty deque if data is missing or invalid
                setattr(self, key, deque(maxlen=self.plot_window))
                if data is not None:
                    logger.warning(
                        f"Invalid data type for deque '{key}' in loaded state: {type(data)}. Initializing empty deque."
                    )

        scalar_keys = [
            "total_episodes",
            "total_triangles_cleared",
            "current_buffer_size",
            "current_global_step",
            "current_lr",
            "start_time",
            "training_target_step",
            "current_self_play_game_number",
            "current_self_play_game_steps",
            "training_steps_performed",
            "_last_step_time",
            "_last_step_count",  # Restore for sps calculation
        ]
        defaults = {
            "start_time": time.time(),
            "training_target_step": 0,
            "current_global_step": 0,
            "total_episodes": 0,
            "total_triangles_cleared": 0,
            "current_buffer_size": 0,
            "current_lr": 0.0,
            "current_self_play_game_number": 0,
            "current_self_play_game_steps": 0,
            "training_steps_performed": 0,
            "_last_step_time": None,
            "_last_step_count": None,
        }
        for key in scalar_keys:
            setattr(self, key, state_dict.get(key, defaults.get(key)))

        best_value_keys = [
            "best_outcome",
            "previous_best_outcome",
            "best_outcome_step",
            "best_game_score",
            "previous_best_game_score",
            "best_game_score_step",
            "best_value_loss",
            "previous_best_value_loss",
            "best_value_loss_step",
            "best_policy_loss",
            "previous_best_policy_loss",
            "best_policy_loss_step",
            "best_mcts_sim_time",
            "previous_best_mcts_sim_time",
            "best_mcts_sim_time_step",
        ]
        best_defaults = {
            "best_outcome": -float("inf"),
            "previous_best_outcome": -float("inf"),
            "best_outcome_step": 0,
            "best_game_score": -float("inf"),
            "previous_best_game_score": -float("inf"),
            "best_game_score_step": 0,
            "best_value_loss": float("inf"),
            "previous_best_value_loss": float("inf"),
            "best_value_loss_step": 0,
            "best_policy_loss": float("inf"),
            "previous_best_policy_loss": float("inf"),
            "best_policy_loss_step": 0,
            "best_mcts_sim_time": float("inf"),
            "previous_best_mcts_sim_time": float("inf"),
            "best_mcts_sim_time_step": 0,
        }
        for key in best_value_keys:
            setattr(self, key, state_dict.get(key, best_defaults.get(key)))

        # Deserialize best game state data (requires GameState class)
        loaded_best_data = state_dict.get("best_game_state_data")
        if loaded_best_data and isinstance(loaded_best_data, dict):
            try:
                from environment.game_state import GameState  # Local import

                temp_game_state = GameState()
                # We only saved the state dict, not the object itself
                # Reconstructing the exact state might be complex or impossible
                # Store the basic info (score, step) and maybe the state dict for inspection
                self.best_game_state_data = {
                    "score": loaded_best_data.get("score"),
                    "step": loaded_best_data.get("step"),
                    "game_state_dict": loaded_best_data.get(
                        "game_state_dict"
                    ),  # Store the dict
                    # Cannot easily reconstruct the full GameState object here
                }
            except ImportError:
                logger.error(
                    "Could not import GameState during best_game_state_data deserialization."
                )
                self.best_game_state_data = None
            except Exception as e:
                logger.error(f"Error deserializing best_game_state_data: {e}")
                self.best_game_state_data = None
        else:
            self.best_game_state_data = None

        # Ensure critical attributes exist after loading
        for attr, default_factory in [
            ("current_global_step", lambda: 0),
            ("best_game_score", lambda: -float("inf")),
            ("best_game_state_data", lambda: None),
            ("training_steps_performed", lambda: 0),
            ("current_self_play_game_number", lambda: 0),
            ("current_self_play_game_steps", lambda: 0),
            ("best_mcts_sim_time", lambda: float("inf")),
            ("steps_per_second", lambda: deque(maxlen=self.plot_window)),
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, default_factory())


File: stats\simple_stats_recorder.py
# File: stats/simple_stats_recorder.py
import time
from typing import Deque, Dict, Any, Optional, Union, List, TYPE_CHECKING
import numpy as np
import torch
import threading
import logging

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from config import StatsConfig, TrainConfig
from utils.helpers import format_eta  # Import helper

if TYPE_CHECKING:
    from environment.game_state import GameState

logger = logging.getLogger(__name__)


class SimpleStatsRecorder(StatsRecorderBase):
    """Logs aggregated statistics to the console periodically."""

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_log_interval: int = StatsConfig.CONSOLE_LOG_FREQ,
        train_config: Optional[TrainConfig] = None,
    ):
        self.aggregator = aggregator
        self.console_log_interval = (
            max(1, console_log_interval) if console_log_interval > 0 else -1
        )
        self.train_config = train_config if train_config else TrainConfig()
        self.last_log_time: float = time.time()
        self.summary_avg_window = self.aggregator.summary_avg_window
        self.updates_since_last_log = 0
        self._lock = threading.Lock()
        logger.info(
            f"[SimpleStatsRecorder] Initialized. Log Interval: {self.console_log_interval if self.console_log_interval > 0 else 'Disabled'} updates/episodes. Avg Window: {self.summary_avg_window}"
        )

    def _log_new_best(
        self,
        metric_name: str,
        current_best: float,
        previous_best: float,
        step: int,
        is_loss: bool,  # True if lower is better
        is_time: bool = False,
    ):
        """Logs a new best value achieved."""
        # Check if the current value is actually an improvement
        improvement_made = False
        if is_loss:  # Lower is better (losses, times)
            if np.isfinite(current_best) and current_best < previous_best:
                improvement_made = True
        else:  # Higher is better (scores)
            if np.isfinite(current_best) and current_best > previous_best:
                improvement_made = True

        if not improvement_made:
            return  # Don't log if it's not better than the previous best

        # Format the values
        if is_time:
            format_str = "{:.3f}s"
            prev_str = (
                format_str.format(previous_best)
                if np.isfinite(previous_best) and previous_best != float("inf")
                else "N/A"
            )
            current_str = format_str.format(current_best)
            prefix = "⏱️"
        elif is_loss:  # Includes losses
            format_str = "{:.4f}"
            prev_str = (
                format_str.format(previous_best)
                if np.isfinite(previous_best) and previous_best != float("inf")
                else "N/A"
            )
            current_str = format_str.format(current_best)
            prefix = "📉"
        else:  # Score
            format_str = "{:.0f}"
            prev_str = (
                format_str.format(previous_best)
                if np.isfinite(previous_best) and previous_best != -float("inf")
                else "N/A"
            )
            current_str = format_str.format(current_best)
            prefix = "🎮"

        step_info = f"at Step ~{step/1e6:.1f}M" if step > 0 else "at Start"
        logger.info(
            f"--- {prefix} New Best {metric_name}: {current_str} {step_info} (Prev: {prev_str}) ---"
        )

    def record_episode(
        self,
        episode_outcome: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional["GameState"] = None,
    ):
        """Records episode stats and prints new bests to console."""
        update_info = self.aggregator.record_episode(
            episode_outcome,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
            game_state_for_best,
        )
        current_step = (
            global_step
            if global_step is not None
            else self.aggregator.storage.current_global_step
        )

        # Check for new best game score
        if update_info.get("new_best_game"):
            self._log_new_best(
                "Game Score",
                self.aggregator.storage.best_game_score,
                self.aggregator.storage.previous_best_game_score,
                current_step,
                is_loss=False,  # Higher score is better
            )

        self._check_and_log_summary(current_step)

    def record_step(self, step_data: Dict[str, Any]):
        """Records step stats and triggers console logging if interval met."""
        update_info = self.aggregator.record_step(step_data)
        g_step = step_data.get(
            "global_step", self.aggregator.storage.current_global_step
        )

        # Check for new best losses and MCTS time
        if update_info.get("new_best_value_loss"):
            self._log_new_best(
                "V.Loss",
                self.aggregator.storage.best_value_loss,
                self.aggregator.storage.previous_best_value_loss,
                g_step,
                is_loss=True,  # Lower loss is better
            )
        if update_info.get("new_best_policy_loss"):
            self._log_new_best(
                "P.Loss",
                self.aggregator.storage.best_policy_loss,
                self.aggregator.storage.previous_best_policy_loss,
                g_step,
                is_loss=True,  # Lower loss is better
            )
        if update_info.get("new_best_mcts_sim_time"):
            self._log_new_best(
                "MCTS Sim Time",
                self.aggregator.storage.best_mcts_sim_time,
                self.aggregator.storage.previous_best_mcts_sim_time,
                g_step,
                is_loss=True,  # Lower time is better
                is_time=True,
            )

        self._check_and_log_summary(g_step)

    def _check_and_log_summary(self, global_step: int):
        """Checks if the logging interval is met and logs summary."""
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
            self.log_summary(global_step)

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Gets the summary dictionary from the aggregator."""
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        """Gets the plot data deques from the aggregator."""
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        """Logs the current summary statistics to the console."""
        summary = self.get_summary(global_step)
        runtime_hrs = (time.time() - self.aggregator.storage.start_time) / 3600
        best_score = (
            f"{summary['best_game_score']:.0f}"
            if summary["best_game_score"] > -float("inf")
            else "N/A"
        )
        avg_win = summary.get("summary_avg_window_size", "?")
        buf_size = summary.get("buffer_size", 0)
        min_buf = self.train_config.MIN_BUFFER_SIZE_TO_TRAIN
        phase = "Buffering" if buf_size < min_buf and global_step == 0 else "Training"
        steps_sec = summary.get(
            "steps_per_second_avg", 0.0
        )  # Use averaged value for summary

        current_game = summary.get("current_self_play_game_number", 0)
        current_game_step = summary.get("current_self_play_game_steps", 0)
        game_prog_str = (
            f"Game: {current_game}({current_game_step})" if current_game > 0 else ""
        )

        # --- Build Log String ---
        log_items = [
            f"[{runtime_hrs:.1f}h|{phase}]",
            f"Step: {global_step/1e6:<6.2f}M ({steps_sec:.1f}/s)",
            f"Ep: {summary['total_episodes']:<7,}".replace(",", "_"),
            f"Buf: {buf_size:,}/{min_buf:,}".replace(",", "_"),
            f"Score(Avg{avg_win}): {summary['avg_game_score_window']:<6.0f} (Best: {best_score})",
        ]

        # Training specific stats
        if global_step > 0 or phase == "Training":
            log_items.extend(
                [
                    f"VLoss(Avg{avg_win}): {summary['value_loss']:.4f}",
                    f"PLoss(Avg{avg_win}): {summary['policy_loss']:.4f}",
                    f"LR: {summary['current_lr']:.1e}",
                ]
            )
        else:  # Buffering phase
            log_items.append("Loss: N/A")

        # MCTS specific stats (show averages)
        mcts_sim_time_avg = summary.get("mcts_simulation_time_avg", 0.0)
        mcts_nn_time_avg = summary.get("mcts_nn_prediction_time_avg", 0.0)
        mcts_nodes_avg = summary.get("mcts_nodes_explored_avg", 0.0)
        if mcts_sim_time_avg > 0 or mcts_nn_time_avg > 0 or mcts_nodes_avg > 0:
            mcts_str = f"MCTS(Avg{avg_win}): SimT={mcts_sim_time_avg*1000:.1f}ms | NNT={mcts_nn_time_avg*1000:.1f}ms | Nodes={mcts_nodes_avg:.0f}"
            log_items.append(mcts_str)

        if game_prog_str:
            log_items.append(game_prog_str)  # Append game progress last

        # Calculate ETA if training target is set
        if self.aggregator.storage.training_target_step > 0 and steps_sec > 0:
            steps_remaining = self.aggregator.storage.training_target_step - global_step
            if steps_remaining > 0:
                eta_seconds = steps_remaining / steps_sec
                eta_str = format_eta(eta_seconds)
                log_items.append(f"ETA: {eta_str}")

        logger.info(" | ".join(log_items))
        self.last_log_time = time.time()

    # --- No-op methods for other recording types ---
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
        # Ensure final summary is logged if interval logging is enabled
        if self.console_log_interval > 0 and self.updates_since_last_log > 0:
            logger.info("[SimpleStatsRecorder] Logging final summary before closing...")
            self.log_summary(self.aggregator.storage.current_global_step)
        logger.info(f"[SimpleStatsRecorder] Closed (is_cleanup={is_cleanup}).")


File: stats\stats_recorder.py
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import (
    Deque,
    List,
    Dict,
    Any,
    Optional,
    Union,
    TYPE_CHECKING,
)
import numpy as np
import torch

if TYPE_CHECKING:
    from environment.game_state import GameState  # Import for type hinting


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
        game_state_for_best: Optional["GameState"] = None,
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


File: stats\__init__.py
from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from .simple_stats_recorder import SimpleStatsRecorder


__all__ = [
    "StatsRecorderBase",
    "StatsAggregator",
    "SimpleStatsRecorder",
]


File: training\checkpoint_manager.py
import os
import torch
import torch.optim as optim
import traceback
import re
import time
from typing import Optional, Tuple, Any, Dict
import pickle

from stats.aggregator import StatsAggregator
from agent.alphazero_net import AlphaZeroNet

# Import scheduler base class for type hinting
from torch.optim.lr_scheduler import _LRScheduler


# --- Checkpoint Finding Logic ---
def find_latest_run_and_checkpoint(
    base_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Finds the latest run directory and the latest checkpoint within it."""
    latest_run_id, latest_run_mtime = None, 0
    if not os.path.isdir(base_dir):
        return None, None
    try:
        for item in os.listdir(base_dir):
            path = os.path.join(base_dir, item)
            if os.path.isdir(path) and item.startswith("run_"):
                try:
                    mtime = os.path.getmtime(path)
                    if mtime > latest_run_mtime:
                        latest_run_mtime, latest_run_id = mtime, item
                except OSError:
                    continue
    except OSError as e:
        print(f"[CheckpointFinder] Error listing {base_dir}: {e}")
        return None, None

    if latest_run_id is None:
        print(f"[CheckpointFinder] No runs found in {base_dir}.")
        return None, None
    latest_run_dir = os.path.join(base_dir, latest_run_id)
    print(f"[CheckpointFinder] Latest run directory: {latest_run_dir}")
    latest_checkpoint = find_latest_checkpoint_in_dir(latest_run_dir)
    if latest_checkpoint:
        print(
            f"[CheckpointFinder] Found checkpoint: {os.path.basename(latest_checkpoint)}"
        )
    else:
        print(f"[CheckpointFinder] No valid checkpoints found in {latest_run_dir}")
    return latest_run_id, latest_checkpoint


def find_latest_checkpoint_in_dir(ckpt_dir: str) -> Optional[str]:
    """Finds the latest checkpoint file in a specific directory."""
    if not os.path.isdir(ckpt_dir):
        return None
    checkpoints, final_ckpt = [], None
    step_pattern = re.compile(r"step_(\d+)_alphazero_nn\.pth")
    final_name = "FINAL_alphazero_nn.pth"
    try:
        for fname in os.listdir(ckpt_dir):
            fpath = os.path.join(ckpt_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if fname == final_name:
                final_ckpt = fpath
            else:
                match = step_pattern.match(fname)
                checkpoints.append((int(match.group(1)), fpath)) if match else None
    except OSError as e:
        print(f"[CheckpointFinder] Error listing {ckpt_dir}: {e}")
        return None

    if final_ckpt:
        try:
            final_mtime = os.path.getmtime(final_ckpt)
            if not any(os.path.getmtime(cp) > final_mtime for _, cp in checkpoints):
                return final_ckpt
        except OSError:
            pass
    if not checkpoints:
        return final_ckpt
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


# --- Checkpoint Manager Class ---
class CheckpointManager:
    """Handles loading and saving of agent, optimizer, scheduler, and stats states."""

    def __init__(
        self,
        agent: Optional[AlphaZeroNet],
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[_LRScheduler],  # Add scheduler
        stats_aggregator: Optional[StatsAggregator],
        base_checkpoint_dir: str,
        run_checkpoint_dir: str,
        load_checkpoint_path_config: Optional[str],
        device: torch.device,
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.scheduler = scheduler  # Store scheduler
        self.stats_aggregator = stats_aggregator
        self.base_checkpoint_dir = base_checkpoint_dir
        self.run_checkpoint_dir = run_checkpoint_dir
        self.device = device
        self.global_step = 0
        self.episode_count = 0
        self.training_target_step = 0
        self.run_id_to_load_from, self.checkpoint_path_to_load = (
            self._determine_checkpoint_to_load(load_checkpoint_path_config)
        )
        if self.stats_aggregator:
            self.stats_aggregator.storage.training_target_step = (
                self.training_target_step
            )

    def _determine_checkpoint_to_load(
        self, config_path: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Determines which checkpoint to load based on config or latest run."""
        if config_path:
            print(f"[CheckpointManager] Using explicit checkpoint path: {config_path}")
            if os.path.isfile(config_path):
                run_id = None
                try:
                    run_id = (
                        os.path.basename(os.path.dirname(config_path))
                        if os.path.basename(os.path.dirname(config_path)).startswith(
                            "run_"
                        )
                        else None
                    )
                except Exception:
                    pass
                print(
                    f"[CheckpointManager] Extracted run_id '{run_id}' from path."
                    if run_id
                    else "[CheckpointManager] Could not determine run_id from path."
                )
                return run_id, config_path
            else:
                print(
                    f"[CheckpointManager] WARNING: Explicit path not found: {config_path}. Starting fresh."
                )
                return None, None
        else:
            print(
                f"[CheckpointManager] Searching for latest run in: {self.base_checkpoint_dir}"
            )
            run_id, ckpt_path = find_latest_run_and_checkpoint(self.base_checkpoint_dir)
            if run_id and ckpt_path:
                print(
                    f"[CheckpointManager] Found latest run '{run_id}' with checkpoint."
                )
            elif run_id:
                print(
                    f"[CheckpointManager] Found latest run '{run_id}' but no checkpoint. Starting fresh."
                )
            else:
                print(f"[CheckpointManager] No previous runs found. Starting fresh.")
            return run_id, ckpt_path

    def get_run_id_to_load_from(self) -> Optional[str]:
        return self.run_id_to_load_from

    def get_checkpoint_path_to_load(self) -> Optional[str]:
        return self.checkpoint_path_to_load

    def load_checkpoint(self):
        """Loads agent, optimizer, scheduler, and stats aggregator state."""
        if not self.checkpoint_path_to_load or not os.path.isfile(
            self.checkpoint_path_to_load
        ):
            print(
                f"[CheckpointManager] Checkpoint not found or not specified: {self.checkpoint_path_to_load}. Skipping load."
            )
            self._reset_all_states()
            return
        print(f"[CheckpointManager] Loading checkpoint: {self.checkpoint_path_to_load}")
        try:
            checkpoint = torch.load(
                self.checkpoint_path_to_load,
                map_location=self.device,
                weights_only=False,  # Set to False to load optimizer/scheduler states
            )
            agent_ok = self._load_agent_state(checkpoint)
            opt_ok = self._load_optimizer_state(checkpoint)
            sched_ok = self._load_scheduler_state(checkpoint)  # Load scheduler
            stats_ok, loaded_target = self._load_stats_state(checkpoint)
            self.global_step = checkpoint.get("global_step", 0)
            print(f"  -> Loaded Global Step: {self.global_step}")
            if stats_ok:
                self.episode_count = self.stats_aggregator.storage.total_episodes
            else:
                self.episode_count = checkpoint.get("episode_count", 0)
            print(
                f"  -> Resuming from Step: {self.global_step}, Ep: {self.episode_count}"
            )
            self.training_target_step = (
                loaded_target
                if loaded_target is not None
                else checkpoint.get("training_target_step", 0)
            )
            if self.stats_aggregator:
                self.stats_aggregator.storage.training_target_step = (
                    self.training_target_step
                )
            print("[CheckpointManager] Checkpoint loading finished.")
            if not agent_ok:
                print("[CheckpointManager] Agent load was unsuccessful.")
            if not opt_ok:
                print("[CheckpointManager] Optimizer load was unsuccessful.")
            if not sched_ok:
                print("[CheckpointManager] Scheduler load was unsuccessful.")
            if not stats_ok:
                print("[CheckpointManager] Stats load was unsuccessful.")
        except (pickle.UnpicklingError, KeyError, Exception) as e:
            print(f"  -> ERROR loading checkpoint ('{e}'). State reset.")
            traceback.print_exc()
            self._reset_all_states()
        print(
            f"[CheckpointManager] Final Training Target Step set to: {self.training_target_step}"
        )

    def _load_agent_state(self, checkpoint: Dict[str, Any]) -> bool:
        """Loads the agent state dictionary."""
        if "agent_state_dict" not in checkpoint:
            print("  -> WARNING: 'agent_state_dict' missing.")
            return False
        if not self.agent:
            print("  -> WARNING: Agent not initialized.")
            return False
        try:
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            print("  -> Agent state loaded.")
            return True
        except Exception as e:
            print(f"  -> ERROR loading Agent state: {e}.")
            return False

    def _load_optimizer_state(self, checkpoint: Dict[str, Any]) -> bool:
        """Loads the optimizer state dictionary."""
        if "optimizer_state_dict" not in checkpoint:
            print("  -> WARNING: 'optimizer_state_dict' missing.")
            return False
        if not self.optimizer:
            print("  -> WARNING: Optimizer not initialized.")
            return False
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Move optimizer state to the correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print("  -> Optimizer state loaded.")
            return True
        except Exception as e:
            print(f"  -> ERROR loading Optimizer state: {e}.")
            return False

    def _load_scheduler_state(self, checkpoint: Dict[str, Any]) -> bool:
        """Loads the scheduler state dictionary."""
        if "scheduler_state_dict" not in checkpoint:
            # This is not necessarily a warning if scheduler wasn't used before
            print("  -> INFO: 'scheduler_state_dict' missing (may be expected).")
            return False
        if not self.scheduler:
            print(
                "  -> INFO: Scheduler not initialized for current run, skipping load."
            )
            return False
        try:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("  -> Scheduler state loaded.")
            return True
        except Exception as e:
            print(f"  -> ERROR loading Scheduler state: {e}.")
            return False

    def _load_stats_state(
        self, checkpoint: Dict[str, Any]
    ) -> Tuple[bool, Optional[int]]:
        """Loads the stats aggregator state."""
        loaded_target = None
        if "stats_aggregator_state_dict" not in checkpoint:
            print("  -> WARNING: 'stats_aggregator_state_dict' missing.")
            return False, loaded_target
        if not self.stats_aggregator:
            print("  -> WARNING: Stats Aggregator not initialized.")
            return False, loaded_target
        try:
            self.stats_aggregator.load_state_dict(
                checkpoint["stats_aggregator_state_dict"]
            )
            loaded_target = getattr(
                self.stats_aggregator.storage, "training_target_step", None
            )
            start_time = self.stats_aggregator.storage.start_time
            print("  -> Stats Aggregator state loaded.")
            if loaded_target is not None:
                print(f"  -> Loaded Training Target Step from Stats: {loaded_target}")
            print(
                f"  -> Loaded Run Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
            )
            return True, loaded_target
        except Exception as e:
            print(f"  -> ERROR loading Stats Aggregator state: {e}.")
            self._reset_aggregator_state()
            return False, loaded_target

    def _reset_aggregator_state(self):
        """Resets only the stats aggregator state."""
        if self.stats_aggregator:
            self.stats_aggregator.__init__(
                avg_windows=self.stats_aggregator.avg_windows,
                plot_window=self.stats_aggregator.plot_window,
            )
            self.stats_aggregator.storage.training_target_step = (
                self.training_target_step
            )
            self.stats_aggregator.storage.total_episodes = 0

    def _reset_all_states(self):
        """Resets all managed states on critical load failure."""
        print("[CheckpointManager] Resetting all managed states due to load failure.")
        self.global_step = 0
        self.episode_count = 0
        self.training_target_step = 0
        if self.optimizer:
            # Reset optimizer state
            self.optimizer.state = {}
            # Re-initialize scheduler if it exists, as its state depends on optimizer
            if self.scheduler:
                # Assuming CosineAnnealingLR, re-init with same params
                # This might need adjustment if using other schedulers
                try:
                    self.scheduler = type(self.scheduler)(
                        self.optimizer, **self.scheduler.state_dict()
                    )
                    print("  -> Scheduler re-initialized after optimizer reset.")
                except Exception as e:
                    print(f"  -> WARNING: Failed to re-initialize scheduler: {e}")
                    self.scheduler = None  # Fallback
            print("  -> Optimizer state reset.")
        self._reset_aggregator_state()

    def save_checkpoint(
        self,
        global_step: int,
        episode_count: int,
        training_target_step: int,
        is_final: bool = False,
    ):
        """Saves agent, optimizer, scheduler, and stats aggregator state."""
        prefix = "FINAL" if is_final else f"step_{global_step}"
        save_dir = self.run_checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{prefix}_alphazero_nn.pth"
        full_path = os.path.join(save_dir, filename)
        print(f"[CheckpointManager] Saving checkpoint ({prefix}) to {save_dir}...")
        temp_path = full_path + ".tmp"
        try:
            agent_sd = self.agent.state_dict() if self.agent else {}
            opt_sd = self.optimizer.state_dict() if self.optimizer else {}
            sched_sd = (
                self.scheduler.state_dict() if self.scheduler else {}
            )  # Get scheduler state
            stats_sd = {}
            agg_ep_count = episode_count
            agg_target_step = training_target_step
            if self.stats_aggregator:
                self.stats_aggregator.storage.training_target_step = (
                    training_target_step
                )
                stats_sd = self.stats_aggregator.state_dict()
                agg_ep_count = self.stats_aggregator.storage.total_episodes
                agg_target_step = self.stats_aggregator.storage.training_target_step

            checkpoint_data = {
                "global_step": global_step,
                "episode_count": agg_ep_count,
                "training_target_step": agg_target_step,
                "agent_state_dict": agent_sd,
                "optimizer_state_dict": opt_sd,
                "scheduler_state_dict": sched_sd,  # Save scheduler state
                "stats_aggregator_state_dict": stats_sd,
            }
            torch.save(checkpoint_data, temp_path)
            os.replace(temp_path, full_path)
            print(f"  -> Checkpoint saved: {filename}")
        except Exception as e:
            print(f"  -> ERROR saving checkpoint: {e}")
            traceback.print_exc()
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
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
# File: ui/input_handler.py
import pygame
from typing import Tuple, Callable, Dict, TYPE_CHECKING, Optional

# Type Aliases for Callbacks
HandleDemoMouseMotionCallback = Callable[[Tuple[int, int]], None]
HandleDemoMouseButtonDownCallback = Callable[[pygame.event.Event], None]
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]
StartDemoModeCallback = Callable[[], None]
ExitDemoModeCallback = Callable[[], None]
StartDebugModeCallback = Callable[[], None]
ExitDebugModeCallback = Callable[[], None]
HandleDebugInputCallback = Callable[[pygame.event.Event], None]
# MCTS Vis Callbacks Removed
# StartMCTSVisualizationCallback = Callable[[], None]
# ExitMCTSVisualizationCallback = Callable[[], None]
# HandleMCTSPanCallback = Callable[[int, int], None]
# HandleMCTSZoomCallback = Callable[[float, Tuple[int, int]], None]
# Combined Worker Control Callbacks
StartRunCallback = Callable[[], None]
StopRunCallback = Callable[[], None]


if TYPE_CHECKING:
    from .renderer import UIRenderer
    from app_state import AppState
    from main_pygame import MainApp


class InputHandler:
    """Handles Pygame events and triggers callbacks based on application state."""

    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",
        # Basic Callbacks
        request_cleanup_cb: RequestCleanupCallback,
        cancel_cleanup_cb: CancelCleanupCallback,
        confirm_cleanup_cb: ConfirmCleanupCallback,
        exit_app_cb: ExitAppCallback,
        # Demo Mode Callbacks
        start_demo_mode_cb: StartDemoModeCallback,
        exit_demo_mode_cb: ExitDemoModeCallback,
        handle_demo_mouse_motion_cb: HandleDemoMouseMotionCallback,
        handle_demo_mouse_button_down_cb: HandleDemoMouseButtonDownCallback,
        # Debug Mode Callbacks
        start_debug_mode_cb: StartDebugModeCallback,
        exit_debug_mode_cb: ExitDebugModeCallback,
        handle_debug_input_cb: HandleDebugInputCallback,
        # MCTS Vis Callbacks Removed
        # start_mcts_visualization_cb: StartMCTSVisualizationCallback,
        # exit_mcts_visualization_cb: ExitMCTSVisualizationCallback,
        # handle_mcts_pan_cb: HandleMCTSPanCallback,
        # handle_mcts_zoom_cb: HandleMCTSZoomCallback,
        # Combined Worker Control Callbacks
        start_run_cb: StartRunCallback,
        stop_run_cb: StopRunCallback,
    ):
        self.screen = screen
        self.renderer = renderer
        # Store Callbacks
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
        # MCTS Vis Callbacks Removed
        # self.start_mcts_visualization_cb = start_mcts_visualization_cb
        # self.exit_mcts_visualization_cb = exit_mcts_visualization_cb
        # self.handle_mcts_pan_cb = handle_mcts_pan_cb
        # self.handle_mcts_zoom_cb = handle_mcts_zoom_cb
        self.start_run_cb = start_run_cb
        self.stop_run_cb = stop_run_cb

        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

        # Button rects
        self.run_stop_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.cleanup_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.demo_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.debug_btn_rect = pygame.Rect(0, 0, 0, 0)
        # self.mcts_vis_btn_rect = pygame.Rect(0, 0, 0, 0) # MCTS Vis removed
        self.confirm_yes_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_no_rect = pygame.Rect(0, 0, 0, 0)
        self._update_button_rects()

        # MCTS Vis state removed
        # self.is_panning_mcts = False
        # self.last_pan_pos: Optional[Tuple[int, int]] = None
        self.app_ref: Optional["MainApp"] = None

    def _update_button_rects(self):
        """Calculates button rects based on initial layout assumptions."""
        button_height = 40
        button_y_pos = 10
        run_stop_button_width = 150
        cleanup_button_width = 160
        demo_button_width = 120
        debug_button_width = 120
        # mcts_vis_button_width = 140 # MCTS Vis removed
        button_spacing = 10

        current_x = button_spacing
        self.run_stop_btn_rect = pygame.Rect(
            current_x, button_y_pos, run_stop_button_width, button_height
        )
        current_x = self.run_stop_btn_rect.right + button_spacing * 2
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

            current_app_state = (
                AppState(app_state_str)
                if app_state_str in AppState._value2member_map_
                else AppState.UNKNOWN
            )

            if current_app_state == AppState.PLAYING:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.exit_demo_mode_cb()
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_demo_mouse_motion_cb(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_demo_mouse_button_down_cb(event)

            elif current_app_state == AppState.DEBUG:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_debug_mode_cb()
                    else:
                        self.handle_debug_input_cb(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_debug_input_cb(event)

            elif current_app_state == AppState.MAIN_MENU:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return self.exit_app_cb()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    is_running = (
                        self.app_ref.worker_manager.is_any_worker_running()
                        if self.app_ref
                        else False
                    )

                    if self.run_stop_btn_rect.collidepoint(mouse_pos):
                        if is_running:
                            self.stop_run_cb()
                        else:
                            self.start_run_cb()
                    elif not is_running:  # Only allow other buttons if not running
                        if self.cleanup_btn_rect.collidepoint(mouse_pos):
                            self.request_cleanup_cb()
                        elif self.demo_btn_rect.collidepoint(mouse_pos):
                            self.start_demo_mode_cb()
                        elif self.debug_btn_rect.collidepoint(mouse_pos):
                            self.start_debug_mode_cb()

            elif current_app_state == AppState.ERROR:
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
# File: ui/overlays.py
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
                "(NN Checkpoint & Stats)", True, VisConfig.WHITE
            )  # Updated text
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
from typing import Dict, Optional, Deque, Tuple
from collections import deque
import matplotlib
import time
from io import BytesIO
import logging
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import VisConfig, StatsConfig
from .plot_utils import render_single_plot, normalize_color_for_matplotlib

logger = logging.getLogger(__name__)


class Plotter:
    """Handles creation and caching of the multi-plot Matplotlib surface."""

    def __init__(self):
        self.plot_surface_cache: Optional[pygame.Surface] = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = (
            2.0  # Faster update for more responsive plots
        )
        self.rolling_window_sizes = StatsConfig.STATS_AVG_WINDOW
        self.plot_data_window = StatsConfig.PLOT_DATA_WINDOW
        self.colors = self._init_colors()

        self.fig: Optional[plt.Figure] = None
        self.axes: Optional[np.ndarray] = None  # Will be a 2D numpy array of axes
        self.last_target_size: Tuple[int, int] = (0, 0)
        self.last_data_hash: Optional[int] = None

        logger.info(
            f"[Plotter] Initialized with update interval: {self.plot_update_interval}s"
        )

    def _init_colors(self) -> Dict[str, Tuple[float, float, float]]:
        """Initializes plot colors."""
        return {
            "game_scores": normalize_color_for_matplotlib(VisConfig.GREEN),
            "policy_losses": normalize_color_for_matplotlib(VisConfig.RED),
            "value_losses": normalize_color_for_matplotlib(VisConfig.ORANGE),
            "episode_lengths": normalize_color_for_matplotlib(VisConfig.BLUE),
            "episode_outcomes": normalize_color_for_matplotlib(VisConfig.YELLOW),
            "episode_triangles_cleared": normalize_color_for_matplotlib(VisConfig.CYAN),
            "lr_values": normalize_color_for_matplotlib(VisConfig.GOOGLE_COLORS[2]),
            "buffer_sizes": normalize_color_for_matplotlib(VisConfig.PURPLE),
            "best_game_score_history": normalize_color_for_matplotlib(VisConfig.GREEN),
            # MCTS Stats Colors
            "mcts_simulation_times": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[1]
            ),
            "mcts_nn_prediction_times": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[3]
            ),
            "mcts_nodes_explored": normalize_color_for_matplotlib(VisConfig.LIGHTG),
            "mcts_avg_depths": normalize_color_for_matplotlib(VisConfig.WHITE),
            # System Stats Colors
            "steps_per_second": normalize_color_for_matplotlib(
                VisConfig.GOOGLE_COLORS[0]
            ),
            # Placeholder Color
            "placeholder": normalize_color_for_matplotlib(VisConfig.GRAY),
        }

    def _init_figure(self, target_width: int, target_height: int):
        """Initializes the Matplotlib figure and axes."""
        logger.info(
            f"[Plotter] Initializing Matplotlib figure for size {target_width}x{target_height}"
        )
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                logger.warning(f"Error closing previous figure: {e}")

        dpi = 96  # Standard DPI
        fig_width_in = max(1, target_width / dpi)
        fig_height_in = max(1, target_height / dpi)

        try:
            # 5 rows, 3 columns grid = 15 plots
            self.fig, self.axes = plt.subplots(
                5, 3, figsize=(fig_width_in, fig_height_in), dpi=dpi, sharex=False
            )
            self.fig.subplots_adjust(
                hspace=0.4,
                wspace=0.3,  # Increased spacing slightly
                left=0.08,
                right=0.97,
                bottom=0.06,
                top=0.96,  # Adjusted margins
            )
            self.last_target_size = (target_width, target_height)
            logger.info("[Plotter] Matplotlib figure initialized (5x3 grid).")
        except Exception as e:
            logger.error(f"Error creating Matplotlib figure: {e}", exc_info=True)
            self.fig = None
            self.axes = None
            self.last_target_size = (0, 0)

    def _get_data_hash(self, plot_data: Dict[str, Deque]) -> int:
        """Generates a simple hash based on data lengths and last elements."""
        hash_val = 0
        # Include all keys used in plot_defs
        keys_to_hash = [
            "game_scores",
            "episode_outcomes",
            "episode_lengths",
            "policy_losses",
            "value_losses",
            "lr_values",
            "episode_triangles_cleared",
            "best_game_score_history",
            "buffer_sizes",
            "mcts_simulation_times",
            "mcts_nn_prediction_times",
            "steps_per_second",
            "mcts_nodes_explored",
            "mcts_avg_depths",
        ]
        for key in sorted(keys_to_hash):
            dq = plot_data.get(key)  # Use .get() to handle missing keys gracefully
            if dq is None or not dq:  # Check if deque exists and is not empty
                continue  # Skip if key doesn't exist or deque is empty

            hash_val ^= hash(key)
            hash_val ^= len(dq)
            try:
                # Hash based on the last element to detect changes
                last_elem = dq[-1]
                if isinstance(last_elem, (int, float)):
                    hash_val ^= hash(f"{last_elem:.6f}")  # Format floats consistently
                else:
                    hash_val ^= hash(str(last_elem))
            except IndexError:
                pass  # Should not happen if dq is not empty
        return hash_val

    def _update_plot_data(self, plot_data: Dict[str, Deque]):
        """Updates the data on the existing Matplotlib axes."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()  # Flatten the 2D array of axes
            # --- Define Plot Order (5x3 Grid) ---
            plot_defs = [
                # Row 1: Game Performance
                ("game_scores", "Game Score", self.colors["game_scores"], False),
                ("episode_lengths", "Ep Length", self.colors["episode_lengths"], False),
                (
                    "episode_triangles_cleared",
                    "Tris Cleared/Ep",
                    self.colors["episode_triangles_cleared"],
                    False,
                ),
                # Row 2: Losses & LR
                (
                    "policy_losses",
                    "Policy Loss",
                    self.colors["policy_losses"],
                    True,
                ),  # Log scale often helpful for losses
                (
                    "value_losses",
                    "Value Loss",
                    self.colors["value_losses"],
                    True,
                ),  # Log scale often helpful for losses
                (
                    "lr_values",
                    "Learning Rate",
                    self.colors["lr_values"],
                    True,
                ),  # Log scale for LR is common
                # Row 3: Buffer & History
                ("buffer_sizes", "Buffer Size", self.colors["buffer_sizes"], False),
                (
                    "best_game_score_history",
                    "Best Score Hist",
                    self.colors["best_game_score_history"],
                    False,
                ),
                (
                    "episode_outcomes",
                    "Ep Outcome (-1,0,1)",
                    self.colors["episode_outcomes"],
                    False,
                ),  # Added outcome plot
                # Row 4: MCTS Timings / System
                (
                    "mcts_simulation_times",
                    "MCTS Sim Time (s)",
                    self.colors["mcts_simulation_times"],
                    False,
                ),
                (
                    "mcts_nn_prediction_times",
                    "MCTS NN Time (s)",
                    self.colors["mcts_nn_prediction_times"],
                    False,
                ),
                (
                    "steps_per_second",
                    "Steps/Sec",
                    self.colors["steps_per_second"],
                    False,
                ),
                # Row 5: MCTS Structure
                (
                    "mcts_nodes_explored",
                    "MCTS Nodes Explored",
                    self.colors["mcts_nodes_explored"],
                    False,
                ),
                (
                    "mcts_avg_depths",
                    "MCTS Avg Depth",
                    self.colors["mcts_avg_depths"],
                    False,
                ),
                (
                    "placeholder",
                    "Future Plot",
                    self.colors["placeholder"],
                    False,
                ),  # Keep one placeholder
            ]

            # Convert deques to lists for plotting (avoids issues with some matplotlib versions)
            data_lists = {
                key: list(plot_data.get(key, deque())) for key, _, _, _ in plot_defs
            }

            for i, (data_key, label, color, log_scale) in enumerate(plot_defs):
                if i >= len(axes_flat):
                    break  # Stop if we run out of axes
                ax = axes_flat[i]
                ax.clear()  # Clear previous plot content

                # Check if data exists before plotting
                current_data = data_lists.get(data_key, [])
                show_placeholder = (data_key == "placeholder") or not current_data

                render_single_plot(
                    ax,
                    current_data,
                    label,
                    color,
                    self.rolling_window_sizes,
                    show_placeholder=show_placeholder,
                    placeholder_text=label,  # Use label as placeholder text
                    y_log_scale=log_scale,
                )
                # Hide x-labels for plots not in the last row
                if i < len(axes_flat) - 3:  # Assuming 3 columns
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=0)  # Keep rotation at 0

            plot_update_duration = time.monotonic() - plot_update_start
            logger.info(f"[Plotter] Plot data updated in {plot_update_duration:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error updating plot data: {e}", exc_info=True)
            # Attempt to clear axes on error to avoid showing stale incorrect plots
            try:
                if self.axes is not None:
                    for ax in self.axes.flatten():
                        ax.clear()
            except Exception:
                pass  # Ignore errors during cleanup
            return False

    def _render_figure_to_surface(
        self, target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Renders the current Matplotlib figure to a Pygame surface."""
        if self.fig is None:
            logger.warning("[Plotter] Cannot render figure, not initialized.")
            return None

        render_start = time.monotonic()
        try:
            # Explicitly draw the canvas
            self.fig.canvas.draw()

            # Use BytesIO buffer to avoid disk I/O
            buf = BytesIO()
            self.fig.savefig(
                buf,
                format="png",
                transparent=False,  # Keep background
                facecolor=plt.rcParams[
                    "figure.facecolor"
                ],  # Use defined background color
            )
            buf.seek(0)

            # Load buffer into Pygame surface
            plot_img_surface = pygame.image.load(
                buf, "png"
            ).convert()  # Use convert() for performance
            buf.close()

            # --- Scaling ---
            # Scale the rendered surface to the exact target size if needed
            current_size = plot_img_surface.get_size()
            if current_size != (target_width, target_height):
                scale_diff = abs(current_size[0] - target_width) + abs(
                    current_size[1] - target_height
                )
                # Use smoothscale for larger differences, faster scale otherwise
                if scale_diff > 10:  # Arbitrary threshold
                    plot_img_surface = pygame.transform.smoothscale(
                        plot_img_surface, (target_width, target_height)
                    )
                else:
                    plot_img_surface = pygame.transform.scale(
                        plot_img_surface, (target_width, target_height)
                    )

            render_duration = time.monotonic() - render_start
            logger.info(
                f"[Plotter] Figure rendered to surface in {render_duration:.4f}s"
            )
            return plot_img_surface

        except Exception as e:
            logger.error(
                f"Error rendering Matplotlib figure to surface: {e}", exc_info=True
            )
            return None

    def get_cached_or_updated_plot(
        self, plot_data: Dict[str, Deque], target_width: int, target_height: int
    ) -> Optional[pygame.Surface]:
        """Returns the cached plot surface or creates/updates one if needed."""
        current_time = time.time()
        has_data = any(d for d in plot_data.values())  # Check if any data exists
        target_size = (target_width, target_height)

        # Check if figure needs reinitialization (first time, or size changed)
        needs_reinit = (
            self.fig is None
            or self.axes is None
            or self.last_target_size != target_size
        )

        # Check if data has changed or enough time has passed
        current_data_hash = self._get_data_hash(plot_data)
        data_changed = self.last_data_hash != current_data_hash
        time_elapsed = (
            current_time - self.last_plot_update_time
        ) > self.plot_update_interval
        needs_update = data_changed or time_elapsed

        # Check if target size is large enough to render plots meaningfully
        can_create_plot = target_width > 50 and target_height > 50

        # Handle cases where plotting is not possible or not needed
        if not can_create_plot:
            if self.plot_surface_cache is not None:
                logger.info(
                    "[Plotter] Target size too small, clearing cache and figure."
                )
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None  # Return None if area is too small

        if not has_data:
            if self.plot_surface_cache is not None:
                logger.info(
                    "[Plotter] No plot data available, clearing cache and figure."
                )
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes = None, None
                self.last_target_size = (0, 0)
            return None  # Return None if there's no data

        # --- Update or Reinitialize ---
        cache_status = "HIT"
        try:
            if needs_reinit:
                cache_status = "MISS (Re-init)"
                logger.info(
                    f"[Plotter] {cache_status}. Reason: fig/axes is None or size changed ({self.last_target_size} != {target_size})"
                )
                self._init_figure(target_width, target_height)
                if self.fig:
                    if self._update_plot_data(plot_data):
                        self.plot_surface_cache = self._render_figure_to_surface(
                            target_width, target_height
                        )
                        self.last_plot_update_time = current_time
                        self.last_data_hash = current_data_hash
                    else:
                        self.plot_surface_cache = None  # Update failed
                else:
                    self.plot_surface_cache = None  # Init failed

            elif needs_update:
                cache_status = (
                    f"MISS (Update - Data: {data_changed}, Time: {time_elapsed})"
                )
                logger.info(f"[Plotter] {cache_status}")
                if self._update_plot_data(plot_data):  # Update existing figure data
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    logger.warning(
                        "[Plotter] Plot data update failed, returning potentially stale cache."
                    )
                    cache_status = "ERROR (Update Failed)"

            elif self.plot_surface_cache is None:
                # Cache is None, but figure might exist and data hasn't changed recently
                # This can happen if the first render failed
                cache_status = "MISS (Cache None)"
                logger.info(f"[Plotter] {cache_status}, attempting re-render.")
                if self._update_plot_data(plot_data):  # Try updating data again
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = (
                        current_time  # Update time even on re-render
                    )
                    self.last_data_hash = current_data_hash

            # Log cache status for debugging performance
            # logger.info(f"[Plotter] Cache status: {cache_status}")

        except Exception as e:
            logger.error(
                f"[Plotter] Unexpected error in get_cached_or_updated_plot: {e}",
                exc_info=True,
            )
            self.plot_surface_cache = None
            if self.fig:
                plt.close(self.fig)  # Clean up figure on error
            self.fig, self.axes = None, None
            self.last_target_size = (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        """Ensure Matplotlib figure is closed when the plotter is garbage collected."""
        if self.fig:
            try:
                plt.close(self.fig)
                logger.info("[Plotter] Matplotlib figure closed in destructor.")
            except Exception as e:
                # Log error but don't crash during GC
                logger.error(f"[Plotter] Error closing figure in destructor: {e}")


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

# --- Constants ---
TREND_SLOPE_TOLERANCE = 1e-5
TREND_MIN_LINEWIDTH = 1
TREND_MAX_LINEWIDTH = 2
TREND_COLOR_STABLE = (1.0, 1.0, 0.0)  # Yellow
TREND_COLOR_INCREASING = (0.0, 0.8, 0.0)  # Green
TREND_COLOR_DECREASING = (0.8, 0.0, 0.0)  # Red
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


# --- Helper Functions ---
def normalize_color_for_matplotlib(
    color_tuple_0_255: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        return tuple(c / 255.0 for c in color_tuple_0_255)
    return (0.0, 0.0, 0.0)  # Default black


# --- Matplotlib Style Setup ---
try:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "#262626",
            "axes.facecolor": "#303030",
            "axes.edgecolor": "#707070",
            "axes.labelcolor": "#D0D0D0",
            "xtick.color": "#C0C0C0",
            "ytick.color": "#C0C0C0",
            "grid.color": "#505050",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "axes.titlepad": 6,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.facecolor": "#202020",
            "legend.title_fontsize": 8,
        }
    )
except Exception as e:
    print(f"Warning: Failed to set Matplotlib style: {e}")


# --- Trend Calculation ---
def calculate_trend_line(data: np.ndarray) -> Optional[Tuple[float, float]]:
    """Calculates the slope and intercept of the linear regression line."""
    n = len(data)
    x = np.arange(n)
    mask = np.isfinite(data)
    if np.sum(mask) < 2:
        return None
    try:
        coeffs = np.polyfit(x[mask], data[mask], 1)
        if not all(np.isfinite(c) for c in coeffs):
            return None
        return coeffs[0], coeffs[1]  # slope, intercept
    except (np.linalg.LinAlgError, ValueError):
        return None


def get_trend_color(slope: float, lower_is_better: bool) -> Tuple[float, float, float]:
    """Maps slope to color (Red -> Yellow -> Green)."""
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_COLOR_STABLE
    eff_slope = -slope if lower_is_better else slope
    norm_slope = np.clip(
        math.atan(eff_slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0), -1.0, 1.0
    )
    t = abs(norm_slope)
    base, target = (
        (TREND_COLOR_STABLE, TREND_COLOR_INCREASING)
        if norm_slope > 0
        else (TREND_COLOR_STABLE, TREND_COLOR_DECREASING)
    )
    color = tuple(base[i] * (1 - t) + target[i] * t for i in range(3))
    return tuple(np.clip(c, 0.0, 1.0) for c in color)


def get_trend_linewidth(slope: float) -> float:
    """Maps slope magnitude to border linewidth."""
    if abs(slope) < TREND_SLOPE_TOLERANCE:
        return TREND_MIN_LINEWIDTH
    norm_mag = np.clip(
        abs(math.atan(slope * TREND_SLOPE_SCALE_FACTOR) / (math.pi / 2.0)), 0.0, 1.0
    )
    return TREND_MIN_LINEWIDTH + norm_mag * (TREND_MAX_LINEWIDTH - TREND_MIN_LINEWIDTH)


# --- Visual Property Interpolation ---
def _interpolate_visual_property(
    rank: int, total_ranks: int, min_val: float, max_val: float
) -> float:
    """Linearly interpolates alpha/linewidth based on rank."""
    if total_ranks <= 1:
        return float(max_val)
    inv_rank = (total_ranks - 1) - rank
    fraction = inv_rank / max(1, total_ranks - 1)
    value = float(min_val) + (float(max_val) - float(min_val)) * fraction
    return float(np.clip(value, min_val, max_val))


# --- Value Formatting ---
def _format_value(value: float, is_loss: bool) -> str:
    """Formats value based on magnitude and whether it's a loss."""
    if not np.isfinite(value):
        return "N/A"
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.1e}"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", "_")
    if is_loss:
        return f"{value:.3f}"
    if abs(value) < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _format_slope(slope: float) -> str:
    """Formats slope value for display in the legend."""
    if not np.isfinite(slope):
        return "N/A"
    sign = "+" if slope >= 0 else ""
    abs_slope = abs(slope)
    if abs_slope < 1e-4:
        return f"{sign}{slope:.1e}"
    if abs_slope < 0.1:
        return f"{sign}{slope:.3f}"
    return f"{sign}{slope:.2f}"


# --- Main Plotting Function ---
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
    """Renders data with rolling averages, trend line, and informative legend."""
    try:
        data_np = np.array(data, dtype=float)
        valid_data = data_np[np.isfinite(data_np)]
    except (ValueError, TypeError):
        valid_data = np.array([])
    n_points = len(valid_data)
    is_lower_better = "loss" in label.lower()

    if n_points == 0:  # Handle empty data
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
                color=normalize_color_for_matplotlib(VisConfig.GRAY),
            )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"{label} (N/A)")
        ax.grid(False)
        ax.patch.set_facecolor(plt.rcParams["axes.facecolor"])
        ax.patch.set_edgecolor(plt.rcParams["axes.edgecolor"])
        ax.patch.set_linewidth(0.5)
        return

    trend_params = calculate_trend_line(valid_data)
    trend_slope = trend_params[0] if trend_params else 0.0
    trend_color = get_trend_color(trend_slope, is_lower_better)
    trend_lw = get_trend_linewidth(trend_slope)
    plotted_windows = sorted([w for w in rolling_window_sizes if n_points >= w])
    total_ranks = 1 + len(plotted_windows)
    current_val = valid_data[-1]
    best_val = np.min(valid_data) if is_lower_better else np.max(valid_data)
    best_val_str = f"Best: {_format_value(best_val, is_lower_better)}"
    ax.set_title(label, loc="left")

    try:
        x_coords = np.arange(n_points)
        plotted_legend = False
        min_y, max_y = float("inf"), float("-inf")
        plot_raw = len(plotted_windows) == 0 or n_points < min(
            rolling_window_sizes, default=10
        )
        if plot_raw:
            rank = 0
            alpha = _interpolate_visual_property(
                rank, total_ranks, MIN_ALPHA, MAX_ALPHA
            )
            lw = _interpolate_visual_property(
                rank, total_ranks, MIN_DATA_AVG_LINEWIDTH, MAX_DATA_AVG_LINEWIDTH
            )
            raw_label = f"Val: {_format_value(current_val, is_lower_better)}"
            ax.plot(
                x_coords,
                valid_data,
                color=color,
                linewidth=lw,
                label=raw_label,
                alpha=alpha,
            )
            min_y = min(min_y, np.min(valid_data))
            max_y = max(max_y, np.max(valid_data))
            plotted_legend = True

        for i, avg_win in enumerate(reversed(plotted_windows)):
            rank = i
            alpha = _interpolate_visual_property(
                rank, total_ranks, MIN_ALPHA, MAX_ALPHA
            )
            lw = _interpolate_visual_property(
                rank, total_ranks, MIN_DATA_AVG_LINEWIDTH, MAX_DATA_AVG_LINEWIDTH
            )
            weights = np.ones(avg_win) / avg_win
            rolling_avg = np.convolve(valid_data, weights, mode="valid")
            avg_x = np.arange(avg_win - 1, n_points)
            if len(avg_x) == len(rolling_avg):
                last_avg = rolling_avg[-1] if len(rolling_avg) > 0 else np.nan
                avg_label = f"Avg {avg_win}: {_format_value(last_avg, is_lower_better)}"
                ax.plot(
                    avg_x,
                    rolling_avg,
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                    linestyle="-",
                    label=avg_label,
                )
                if len(rolling_avg) > 0:
                    min_y = min(min_y, np.min(rolling_avg))
                    max_y = max(max_y, np.max(rolling_avg))
                plotted_legend = True

        if trend_params and n_points >= 2:
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
            plotted_legend = True

        ax.tick_params(axis="both", which="major")
        ax.grid(
            True,
            linestyle=plt.rcParams["grid.linestyle"],
            alpha=plt.rcParams["grid.alpha"],
        )
        if np.isfinite(min_y) and np.isfinite(max_y):
            yrange = max(max_y - min_y, 1e-6)
            pad = yrange * 0.05
            ax.set_ylim(min_y - pad, max_y + pad)
        if y_log_scale and min_y > 1e-9:
            ax.set_yscale("log")
            bottom, top = ax.get_ylim()
            new_bottom = max(bottom, 1e-9)
            if new_bottom >= top:
                new_bottom = top / 10
            ax.set_ylim(bottom=new_bottom, top=top)
        else:
            ax.set_yscale("linear")
        if n_points > 1:
            ax.set_xlim(0, n_points - 1)
        elif n_points == 1:
            ax.set_xlim(-0.5, 0.5)
        if n_points > 1000:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))

            def fmt_func(v, _):
                val = int(v)
                return (
                    f"{val/1e6:.1f}M"
                    if val >= 1e6
                    else (f"{val/1e3:.0f}k" if val >= 1e3 else f"{val}")
                )

            ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_func))
        elif n_points > 10:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        if plotted_legend:
            ax.legend(loc="center left", bbox_to_anchor=(0, 0.5), title=best_val_str)

    except Exception as plot_err:
        print(f"ERROR during render_single_plot for '{label}': {plot_err}")
        traceback.print_exc()
        ax.text(
            0.5,
            0.5,
            f"Plot Error\n({label})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color=normalize_color_for_matplotlib(VisConfig.RED),
        )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)

    ax.patch.set_facecolor((*trend_color, TREND_BACKGROUND_ALPHA))
    ax.patch.set_edgecolor(trend_color)
    ax.patch.set_linewidth(trend_lw)


File: ui\renderer.py
# File: ui/renderer.py
import pygame
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque
import logging

from config import VisConfig, EnvConfig, DemoConfig
from environment.game_state import GameState
from .panels import LeftPanelRenderer, GameAreaRenderer
from .overlays import OverlayRenderer
from .plotter import Plotter
from .demo_renderer import DemoRenderer
from .input_handler import InputHandler
from app_state import AppState

logger = logging.getLogger(__name__)


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
            # Pass the app reference if the button renderer needs it
            if hasattr(self.left_panel.button_status_renderer, "app_ref"):
                self.left_panel.button_status_renderer.app_ref = input_handler.app_ref

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0
        self.game_area.best_state_surface_cache = None
        self.game_area.placeholder_surface_cache = None
        logger.info("[Renderer] Forced redraw triggered.")

    def render_all(self, **kwargs):
        """Renders UI based on the application state."""
        try:
            app_state_str = kwargs.get("app_state", AppState.UNKNOWN.value)
            current_app_state = (
                AppState(app_state_str)
                if app_state_str in AppState._value2member_map_
                else AppState.UNKNOWN
            )

            # --- Main Rendering Logic ---
            if current_app_state == AppState.MAIN_MENU:
                self._render_main_menu(**kwargs)
            elif current_app_state == AppState.PLAYING:
                self._render_demo_mode(
                    kwargs.get("demo_env"), kwargs.get("env_config"), is_debug=False
                )
            elif current_app_state == AppState.DEBUG:
                self._render_demo_mode(
                    kwargs.get("demo_env"), kwargs.get("env_config"), is_debug=True
                )
            elif current_app_state == AppState.INITIALIZING:
                self._render_initializing_screen(
                    kwargs.get("status", "Initializing...")
                )
            elif current_app_state == AppState.ERROR:
                self._render_error_screen(kwargs.get("status", "Unknown Error"))
            else:  # Handle other potential states or default view
                self._render_simple_message(f"State: {app_state_str}", VisConfig.WHITE)

            # Render overlays on top (e.g., cleanup confirmation)
            if (
                kwargs.get("cleanup_confirmation_active")
                and current_app_state != AppState.ERROR
            ):
                self.overlays.render_cleanup_confirmation()
            elif not kwargs.get("cleanup_confirmation_active"):
                # Render temporary status messages (like 'Cleanup complete')
                self.overlays.render_status_message(
                    kwargs.get("cleanup_message", ""),
                    kwargs.get("last_cleanup_message_time", 0.0),
                )

            pygame.display.flip()  # Flip the display once after all rendering

        except pygame.error as e:
            logger.error(f"Pygame rendering error in render_all: {e}")
            # Attempt to render a minimal error message directly
            try:
                self._render_simple_message("Pygame Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception:
                pass
        except Exception as e:
            logger.critical(
                f"Unexpected critical rendering error in render_all: {e}", exc_info=True
            )
            try:
                self._render_simple_message("Critical Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception:
                pass

    def _render_main_menu(self, **kwargs):
        """Renders the main dashboard view with live worker envs."""
        self.screen.fill(VisConfig.BLACK)
        current_width, current_height = self.screen.get_size()
        lp_width, ga_width = self._calculate_panel_widths(current_width)

        # --- Render Left Panel (Stats, Controls, Plots) ---
        self.left_panel.render(
            panel_width=lp_width,
            is_process_running=kwargs.get("is_process_running", False),
            status=kwargs.get("status", ""),
            stats_summary=kwargs.get("stats_summary", {}),
            plot_data=kwargs.get("plot_data", {}),
            app_state=kwargs.get("app_state", ""),
            update_progress_details=kwargs.get("update_progress_details", {}),
            agent_param_count=kwargs.get("agent_param_count", 0),
            worker_counts=kwargs.get("worker_counts", {}),
        )

        # --- Render Game Area Panel (Live Worker Envs) ---
        if ga_width > 0:
            # Pass the worker_render_data list directly
            self.game_area.render(
                panel_width=ga_width,
                panel_x_offset=lp_width,
                worker_render_data=kwargs.get(
                    "worker_render_data", []
                ),  # List[Optional[Dict]]
                num_envs=kwargs.get("num_envs", 0),  # Total number of SP workers
                env_config=kwargs.get("env_config"),
                best_game_state_data=kwargs.get(
                    "best_game_state_data"
                ),  # Pass best game state data
            )

    def _calculate_panel_widths(self, current_width: int) -> Tuple[int, int]:
        """Calculates the widths for the left and game area panels."""
        # Ensure ratio is within reasonable bounds
        left_panel_ratio = max(0.2, min(0.8, self.vis_config.LEFT_PANEL_RATIO))
        lp_width = int(current_width * left_panel_ratio)
        ga_width = current_width - lp_width

        # Ensure left panel has a minimum width for readability, unless screen is tiny
        min_lp_width = 400  # Minimum width for left panel elements
        if lp_width < min_lp_width and current_width > min_lp_width:
            lp_width = min_lp_width
            ga_width = max(0, current_width - lp_width)  # Adjust game area width
        elif (
            current_width <= min_lp_width
        ):  # If screen too small, give all to left panel
            lp_width = current_width
            ga_width = 0

        return lp_width, ga_width

    def _render_demo_mode(
        self,
        demo_env: Optional[GameState],
        env_config: Optional[EnvConfig],
        is_debug: bool,
    ):
        """Renders the demo or debug mode using the DemoRenderer."""
        if demo_env and env_config and self.demo_renderer:
            self.demo_renderer.render(demo_env, env_config, is_debug=is_debug)
        else:
            mode = "Debug" if is_debug else "Demo"
            self._render_simple_message(
                f"{mode} Env Error or Renderer Missing!", VisConfig.RED
            )

    def _render_initializing_screen(self, status_message: str = "Initializing..."):
        """Renders a simple initializing message."""
        self._render_simple_message(status_message, VisConfig.WHITE)

    def _render_error_screen(self, status_message: str):
        """Renders the error screen with more detail."""
        try:
            self.screen.fill((40, 0, 0))  # Dark red background
            font_title = pygame.font.SysFont(None, 70)
            font_msg = pygame.font.SysFont(None, 30)

            title_surf = font_title.render("APPLICATION ERROR", True, VisConfig.RED)
            msg_surf = font_msg.render(
                f"Status: {status_message}", True, VisConfig.YELLOW
            )
            exit_surf = font_msg.render(
                "Press ESC or close window to exit.", True, VisConfig.WHITE
            )

            # Position elements
            title_rect = title_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() // 3)
            )
            msg_rect = msg_surf.get_rect(
                center=(self.screen.get_width() // 2, title_rect.bottom + 30)
            )
            exit_rect = exit_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() * 0.8)
            )

            self.screen.blit(title_surf, title_rect)
            self.screen.blit(msg_surf, msg_rect)
            self.screen.blit(exit_surf, exit_rect)

        except Exception as e:
            logger.error(f"Error rendering error screen itself: {e}")
            # Fallback to simple message if error screen fails
            self._render_simple_message(f"Error State: {status_message}", VisConfig.RED)

    def _render_simple_message(self, message: str, color: Tuple[int, int, int]):
        """Renders a simple centered message."""
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)  # Use a default system font
            if not font:  # Fallback if SysFont fails
                font = pygame.font.Font(None, 50)

            text_surf = font.render(message, True, color)
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
        except Exception as e:
            logger.error(f"Error rendering simple message '{message}': {e}")


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

from config import (
    VisConfig,
    EnvConfig,
    DemoConfig,
    RED,
    BLUE,
    WHITE,
    GRAY,
    BLACK,
)  # Added WHITE, GRAY, BLACK
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

            # Calculate grid rendering parameters
            tri_cell_w, tri_cell_h = self.calculate_demo_triangle_size(
                clipped_game_rect.width, clipped_game_rect.height, env_config
            )
            if tri_cell_w > 0 and tri_cell_h > 0:
                grid_ox, grid_oy = self.calculate_grid_offset(
                    clipped_game_rect.width, clipped_game_rect.height, env_config
                )

                # Render the grid directly using GameState object
                self._render_demo_grid_from_gamestate(
                    game_surf, demo_env, tri_cell_w, tri_cell_h, grid_ox, grid_oy
                )

                # Render dragged shape if not in debug mode
                if not is_debug:
                    self._render_dragged_shape(
                        game_surf,
                        demo_env,
                        tri_cell_w,
                        tri_cell_h,
                        grid_ox,
                        grid_oy,
                        clipped_game_rect.topleft,
                    )

            # Render overlays (Game Over, Line Clear) - using HUD renderer's logic if needed
            if demo_env.is_over():
                self._render_demo_overlay_text(game_surf, "GAME OVER", RED)
            elif demo_env.is_line_clearing() and demo_env.last_line_clear_info:
                lines, tris, score = demo_env.last_line_clear_info
                line_str = "Line" if lines == 1 else "Lines"
                clear_msg = (
                    f"{lines} {line_str} Cleared! ({tris} Tris, +{score:.2f} pts)"
                )
                self._render_demo_overlay_text(game_surf, clear_msg, BLUE)

        except ValueError as e:
            print(f"Error subsurface demo game ({clipped_game_rect}): {e}")
            pygame.draw.rect(self.screen, RED, clipped_game_rect, 1)
        except Exception as render_e:
            print(f"Error rendering demo game area: {render_e}")
            traceback.print_exc()
            pygame.draw.rect(self.screen, RED, clipped_game_rect, 1)

    def _render_demo_grid_from_gamestate(
        self,
        surf: pygame.Surface,
        env: GameState,
        cell_w: int,
        cell_h: int,
        grid_offset_x: float,
        grid_offset_y: float,
    ):
        """Renders the grid directly from the GameState object."""
        grid = env.grid
        for r in range(grid.rows):
            for c in range(grid.cols):
                if not (
                    0 <= r < len(grid.triangles) and 0 <= c < len(grid.triangles[r])
                ):
                    continue
                t = grid.triangles[r][c]
                if t.is_death:
                    continue  # Don't draw death cells

                try:
                    pts = t.get_points(
                        ox=grid_offset_x, oy=grid_offset_y, cw=cell_w, ch=cell_h
                    )
                    color = self.vis_config.LIGHTG  # Default empty color
                    if t.is_occupied:
                        color = (
                            t.color if t.color else self.vis_config.RED
                        )  # Use shape color or fallback

                    # Highlight cleared triangles
                    is_highlighted = (
                        env.is_highlighting_cleared()
                        and (r, c) in env.cleared_triangles_coords
                    )
                    if is_highlighted:
                        highlight_color = self.vis_config.LINE_CLEAR_HIGHLIGHT_COLOR
                        # Draw slightly larger background polygon
                        center_x = sum(p[0] for p in pts) / 3
                        center_y = sum(p[1] for p in pts) / 3
                        scale_factor = 1.2
                        highlight_pts = [
                            (
                                center_x + (p[0] - center_x) * scale_factor,
                                center_y + (p[1] - center_y) * scale_factor,
                            )
                            for p in pts
                        ]
                        pygame.draw.polygon(
                            surf,
                            (
                                highlight_color[0],
                                highlight_color[1],
                                highlight_color[2],
                            ),
                            highlight_pts,
                        )  # Use RGB for solid bg

                    # Draw the main triangle
                    pygame.draw.polygon(surf, color, pts)
                    # Draw border
                    pygame.draw.polygon(surf, self.vis_config.GRAY, pts, 1)

                except Exception as tri_err:
                    # logger.info(f"Error drawing triangle ({r},{c}): {tri_err}") # Use debug level
                    pass  # Ignore errors for single triangles

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
            preview_color_rgba = (50, 50, 50, 100)  # Invalid placement color

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
                # Optionally draw border for preview
                # pygame.draw.polygon(temp_surface, (200, 200, 200, 100), points, 1)
            except Exception:
                pass

        surf.blit(temp_surface, (0, 0))

    def _render_demo_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders centered overlay text (e.g., GAME OVER)."""
        try:
            font = self.overlay_font  # Use the font initialized for overlays
            if not font:
                return

            max_w = surf.get_width() * 0.9
            original_size = font.get_height()
            current_size = original_size

            surf_txt = font.render(text, True, WHITE)
            while surf_txt.get_width() > max_w and current_size > 8:
                current_size -= 2
                try:
                    font = pygame.font.SysFont(None, current_size)
                except:
                    font = pygame.font.Font(None, current_size)
                surf_txt = font.render(text, True, WHITE)

            bg_rgba = (color[0] // 2, color[1] // 2, color[2] // 2, 220)
            surf_bg = font.render(text, True, WHITE, bg_rgba)
            rect = surf_bg.get_rect(center=surf.get_rect().center)
            surf.blit(surf_bg, rect)
        except Exception as e:
            print(f"Error rendering overlay '{text}': {e}")


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


File: ui\mcts_visualizer\node_renderer.py
# File: ui/mcts_visualizer/node_renderer.py
import pygame
import math
from typing import Tuple, Optional, TYPE_CHECKING

from config import (
    VisConfig,
    WHITE,
    BLACK,
    RED,
    BLUE,
    YELLOW,
    GRAY,
    LIGHTG,
    CYAN,
    ORANGE,
)
from config.constants import (
    MCTS_NODE_WIN_COLOR,
    MCTS_NODE_LOSS_COLOR,
    MCTS_NODE_NEUTRAL_COLOR,
    MCTS_NODE_BORDER_COLOR,
    MCTS_NODE_SELECTED_BORDER_COLOR,
    MCTS_NODE_TEXT_COLOR,
    MCTS_NODE_PRIOR_COLOR,
    MCTS_NODE_SCORE_COLOR,
    MCTS_MINI_GRID_BG_COLOR,
    MCTS_MINI_GRID_LINE_COLOR,
    MCTS_MINI_GRID_OCCUPIED_COLOR,
    MCTS_EDGE_COLOR,
)
from mcts.node import MCTSNode

if TYPE_CHECKING:
    from ui.panels.game_area import GameAreaRenderer  # For type hinting


class MCTSNodeRenderer:
    """Renders a single MCTS node for visualization, including a mini-grid."""

    BASE_NODE_RADIUS = 25  # Increased base radius
    BASE_NODE_BORDER_WIDTH = 1
    BASE_FONT_SIZE = 10  # Smaller base font for more info
    MIN_NODE_RADIUS = 8
    MAX_NODE_RADIUS = 80
    MIN_FONT_SIZE = 6
    MAX_FONT_SIZE = 14
    GRID_RENDER_THRESHOLD_RADIUS = 15  # Only render grid if node radius is above this

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.font: Optional[pygame.font.Font] = None
        self.game_area_renderer: Optional["GameAreaRenderer"] = (
            None  # To render mini-grid
        )

    def set_game_area_renderer(self, renderer: "GameAreaRenderer"):
        """Sets the reference to the GameAreaRenderer."""
        self.game_area_renderer = renderer

    def _get_scaled_font(self, zoom: float) -> pygame.font.Font:
        """Gets a font scaled based on the zoom level."""
        scaled_size = int(self.BASE_FONT_SIZE * math.sqrt(zoom))
        clamped_size = max(self.MIN_FONT_SIZE, min(self.MAX_FONT_SIZE, scaled_size))
        try:
            # Consider caching fonts if performance becomes an issue
            return pygame.font.SysFont(None, clamped_size)
        except Exception:
            return pygame.font.Font(None, clamped_size)

    def _render_mini_grid(self, node: MCTSNode, surface: pygame.Surface):
        """Renders the game state grid onto the provided surface."""
        if not self.game_area_renderer:
            pygame.draw.line(surface, RED, (0, 0), surface.get_size(), 1)
            return

        # Use a simplified version of GameAreaRenderer's grid rendering
        try:
            padding = 1  # Minimal padding inside the node
            drawable_w = max(1, surface.get_width() - 2 * padding)
            drawable_h = max(1, surface.get_height() - 2 * padding)
            env_config = node.game_state.env_config  # Get config from node's state
            grid_rows, grid_cols_eff_width = (
                env_config.ROWS,
                env_config.COLS * 0.75 + 0.25,
            )
            if grid_rows <= 0 or grid_cols_eff_width <= 0:
                return

            scale_w = drawable_w / grid_cols_eff_width
            scale_h = drawable_h / grid_rows
            final_scale = min(scale_w, scale_h)
            if final_scale <= 0:
                return

            tri_cell_w, tri_cell_h = max(1, final_scale), max(1, final_scale)
            final_grid_pixel_w = grid_cols_eff_width * final_scale
            final_grid_pixel_h = grid_rows * final_scale
            grid_ox = padding + (drawable_w - final_grid_pixel_w) / 2
            grid_oy = padding + (drawable_h - final_grid_pixel_h) / 2

            surface.fill(MCTS_MINI_GRID_BG_COLOR)  # Background for the grid area

            grid = node.game_state.grid
            for r in range(grid.rows):
                for c in range(grid.cols):
                    if not (
                        0 <= r < len(grid.triangles) and 0 <= c < len(grid.triangles[r])
                    ):
                        continue
                    t = grid.triangles[r][c]
                    if not t.is_death and hasattr(t, "get_points"):
                        try:
                            pts = t.get_points(
                                ox=grid_ox,
                                oy=grid_oy,
                                cw=int(tri_cell_w),
                                ch=int(tri_cell_h),
                            )
                            color = MCTS_MINI_GRID_BG_COLOR  # Default empty color
                            if t.is_occupied:
                                # Use a simple bright color for occupied cells in mini-grid
                                color = MCTS_MINI_GRID_OCCUPIED_COLOR
                            pygame.draw.polygon(surface, color, pts)
                            # Draw subtle grid lines
                            pygame.draw.polygon(
                                surface, MCTS_MINI_GRID_LINE_COLOR, pts, 1
                            )
                        except Exception:
                            pass  # Ignore drawing errors for single triangles
        except Exception as e:
            print(f"Error rendering mini-grid: {e}")
            pygame.draw.line(surface, RED, (0, 0), surface.get_size(), 1)

    def render(
        self,
        node: MCTSNode,
        pos: Tuple[int, int],
        zoom: float,
        is_selected: bool = False,
    ):
        """Draws the node circle, mini-grid, and info, scaled by zoom."""
        self.font = self._get_scaled_font(zoom)
        if not self.font:
            return

        scaled_radius = int(self.BASE_NODE_RADIUS * zoom)
        node_radius = max(
            self.MIN_NODE_RADIUS, min(self.MAX_NODE_RADIUS, scaled_radius)
        )
        border_width = max(1, int(self.BASE_NODE_BORDER_WIDTH * zoom))

        value = node.mean_action_value
        if value > 0.1:
            color = MCTS_NODE_WIN_COLOR
        elif value < -0.1:
            color = MCTS_NODE_LOSS_COLOR
        else:
            color = MCTS_NODE_NEUTRAL_COLOR

        # Create surface for the node content (grid + border)
        node_diameter = node_radius * 2
        node_surface = pygame.Surface((node_diameter, node_diameter), pygame.SRCALPHA)
        node_surface.fill((0, 0, 0, 0))  # Transparent background

        # Render mini-grid if node is large enough
        if node_radius >= self.GRID_RENDER_THRESHOLD_RADIUS:
            grid_surface = pygame.Surface(
                (node_diameter, node_diameter), pygame.SRCALPHA
            )
            self._render_mini_grid(node, grid_surface)
            # Clip the grid to a circle
            pygame.draw.circle(
                grid_surface,
                (255, 255, 255, 0),
                (node_radius, node_radius),
                node_radius,
            )  # Transparent circle mask
            grid_surface.set_colorkey(
                (255, 255, 255, 0)
            )  # Make transparent area the colorkey
            node_surface.blit(grid_surface, (0, 0))
        else:
            # Draw solid color if too small for grid
            pygame.draw.circle(
                node_surface, color, (node_radius, node_radius), node_radius
            )

        # Draw border
        border_color = (
            MCTS_NODE_SELECTED_BORDER_COLOR if is_selected else MCTS_NODE_BORDER_COLOR
        )
        pygame.draw.circle(
            node_surface,
            border_color,
            (node_radius, node_radius),
            node_radius,
            border_width,
        )

        # Blit the node surface onto the main screen
        node_rect = node_surface.get_rect(center=pos)
        self.screen.blit(node_surface, node_rect)

        # Render text info below the node if radius is sufficient
        if node_radius > 10:
            visits_str = f"N:{node.visit_count}"
            value_str = f"Q:{value:.2f}"
            prior_str = f"P:{node.prior:.2f}"
            score_str = f"S:{node.game_state.game_score}"

            text_y_offset = node_rect.bottom + 2  # Start text below node rect
            line_height = self.font.get_linesize()

            visits_surf = self.font.render(visits_str, True, MCTS_NODE_TEXT_COLOR)
            value_surf = self.font.render(value_str, True, MCTS_NODE_TEXT_COLOR)
            prior_surf = self.font.render(prior_str, True, MCTS_NODE_PRIOR_COLOR)
            score_surf = self.font.render(score_str, True, MCTS_NODE_SCORE_COLOR)

            # Center text horizontally below the node
            self.screen.blit(
                visits_surf, visits_surf.get_rect(midtop=(pos[0], text_y_offset))
            )
            self.screen.blit(
                value_surf,
                value_surf.get_rect(midtop=(pos[0], text_y_offset + line_height)),
            )
            self.screen.blit(
                prior_surf,
                prior_surf.get_rect(midtop=(pos[0], text_y_offset + 2 * line_height)),
            )
            self.screen.blit(
                score_surf,
                score_surf.get_rect(midtop=(pos[0], text_y_offset + 3 * line_height)),
            )

    def draw_edge(
        self,
        parent_pos: Tuple[int, int],
        child_pos: Tuple[int, int],
        line_width: int = 1,
        color: Tuple[int, int, int] = MCTS_EDGE_COLOR,
    ):
        """Draws a line connecting parent and child nodes with variable width/color."""
        clamped_width = max(1, min(line_width, 5))
        pygame.draw.aaline(
            self.screen, color, parent_pos, child_pos
        )  # Use anti-aliased line


File: ui\mcts_visualizer\renderer.py
# File: ui/mcts_visualizer/renderer.py
import pygame
import math
from typing import Optional, Dict, Tuple, TYPE_CHECKING

from config import VisConfig, BLACK, WHITE, GRAY, YELLOW
from config.constants import (
    MCTS_INFO_TEXT_COLOR,
    MCTS_EDGE_COLOR,
    MCTS_EDGE_HIGHLIGHT_COLOR,
)
from mcts.node import MCTSNode
from .node_renderer import MCTSNodeRenderer
from .tree_layout import TreeLayout

if TYPE_CHECKING:
    from ui.panels.game_area import GameAreaRenderer


class MCTSVisualizer:
    """Renders the MCTS tree visualization with pan and zoom."""

    MIN_ZOOM = 0.1
    MAX_ZOOM = 5.0
    EDGE_HIGHLIGHT_THRESHOLD = 0.7  # Fraction of max visits to highlight edge

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        fonts: Dict[str, pygame.font.Font],
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = fonts
        self.node_renderer = MCTSNodeRenderer(screen, vis_config)
        self.info_font = fonts.get("ui", pygame.font.Font(None, 24))

        self.camera_offset_x = 0
        self.camera_offset_y = 0
        self.zoom_level = 1.0

        self.layout: Optional[TreeLayout] = None
        self.positions: Dict[MCTSNode, Tuple[int, int]] = {}

    def set_game_area_renderer(self, renderer: "GameAreaRenderer"):
        """Provides the GameAreaRenderer to the NodeRenderer for mini-grid drawing."""
        self.node_renderer.set_game_area_renderer(renderer)

    def reset_camera(self):
        """Resets camera pan and zoom."""
        self.camera_offset_x = 0
        self.camera_offset_y = 0
        self.zoom_level = 1.0
        print("MCTS Camera Reset")

    def pan_camera(self, delta_x: int, delta_y: int):
        """Pans the camera view."""
        self.camera_offset_x += delta_x
        self.camera_offset_y += delta_y

    def zoom_camera(self, factor: float, mouse_pos: Tuple[int, int]):
        """Zooms the camera view towards/away from the mouse position."""
        old_zoom = self.zoom_level
        self.zoom_level *= factor
        self.zoom_level = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.zoom_level))
        zoom_change = self.zoom_level / old_zoom

        world_mouse_x = (mouse_pos[0] - self.camera_offset_x) / old_zoom
        world_mouse_y = (mouse_pos[1] - self.camera_offset_y) / old_zoom

        new_offset_x = mouse_pos[0] - world_mouse_x * self.zoom_level
        new_offset_y = mouse_pos[1] - world_mouse_y * self.zoom_level

        self.camera_offset_x = new_offset_x
        self.camera_offset_y = new_offset_y

    def _world_to_screen(self, world_x: int, world_y: int) -> Tuple[int, int]:
        """Converts world coordinates (from layout) to screen coordinates."""
        screen_x = int(world_x * self.zoom_level + self.camera_offset_x)
        screen_y = int(world_y * self.zoom_level + self.camera_offset_y)
        return screen_x, screen_y

    def render(self, root_node: Optional[MCTSNode]):
        """Draws the MCTS tree and related info, applying camera transforms."""
        self.screen.fill(BLACK)

        if root_node is None:
            self._render_message("No MCTS data available.")
            return
        if not root_node.children and root_node.is_terminal:
            self._render_message("Root node is terminal.")
            pos = self._world_to_screen(self.screen.get_width() // 2, 100)
            self.node_renderer.render(root_node, pos, self.zoom_level, is_selected=True)
            self._render_info(root_node)
            return
        if not root_node.children and not root_node.is_expanded:
            self._render_message("MCTS Root not expanded (0 simulations?).")
            return

        if self.layout is None or self.layout.root != root_node:
            self.layout = TreeLayout(
                root_node, self.screen.get_width(), self.screen.get_height()
            )
            self.positions = self.layout.calculate_layout()

        max_child_visits = 0
        best_child_node: Optional[MCTSNode] = None
        if root_node.children:
            try:
                # Find best child based on visits for highlighting
                best_child_node = max(
                    root_node.children.values(), key=lambda n: n.visit_count
                )
                max_child_visits = best_child_node.visit_count
            except ValueError:  # Handle empty children dict case
                max_child_visits = 0
                best_child_node = None

        # Render edges first
        edges_to_render = []
        for node, world_pos in self.positions.items():
            if node.parent and node.parent in self.positions:
                parent_world_pos = self.positions[node.parent]
                parent_screen_pos = self._world_to_screen(*parent_world_pos)
                child_screen_pos = self._world_to_screen(*world_pos)

                line_width = 1
                edge_color = MCTS_EDGE_COLOR
                is_best_edge = False

                # Highlight edge from root to best child (based on visits)
                if (
                    node.parent == root_node
                    and node == best_child_node
                    and max_child_visits > 0
                ):
                    line_width = 3
                    edge_color = MCTS_EDGE_HIGHLIGHT_COLOR
                    is_best_edge = True

                edges_to_render.append(
                    (
                        (parent_screen_pos, child_screen_pos, line_width, edge_color),
                        is_best_edge,
                    )
                )

        # Sort edges to draw non-highlighted ones first
        edges_to_render.sort(
            key=lambda x: x[1]
        )  # False (non-best) comes before True (best)

        for edge_params, _ in edges_to_render:
            self.node_renderer.draw_edge(*edge_params)

        # Render nodes on top
        for node, world_pos in self.positions.items():
            screen_pos = self._world_to_screen(*world_pos)
            # Basic visibility check (culling) - expand bounds slightly
            render_radius = int(MCTSNodeRenderer.MAX_NODE_RADIUS * self.zoom_level)
            if (
                -render_radius < screen_pos[0] < self.screen.get_width() + render_radius
                and -render_radius
                < screen_pos[1]
                < self.screen.get_height() + render_radius
            ):
                self.node_renderer.render(
                    node, screen_pos, self.zoom_level, is_selected=(node == root_node)
                )

        self._render_info(root_node)

    def _render_message(self, message: str):
        """Displays a message centered on the screen."""
        if not self.info_font:
            return
        text_surf = self.info_font.render(message, True, WHITE)
        text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
        self.screen.blit(text_surf, text_rect)

    def _render_info(self, root_node: MCTSNode):
        """Displays information about the MCTS search and controls."""
        if not self.info_font:
            return
        sims = root_node.visit_count
        info_text = f"MCTS | Sims: {sims} | Zoom: {self.zoom_level:.2f}x | Drag=Pan | Scroll=Zoom | ESC=Exit"
        text_surf = self.info_font.render(info_text, True, MCTS_INFO_TEXT_COLOR)
        self.screen.blit(text_surf, (10, 10))

        if root_node.children:
            try:
                best_action_visits_node = max(
                    root_node.children.values(), key=lambda n: n.visit_count
                )
                best_action_visits = best_action_visits_node.action_taken

                best_action_q_node = max(
                    root_node.children.values(), key=lambda n: n.mean_action_value
                )
                best_action_q = best_action_q_node.action_taken

                best_action_text = f"Best Action (Visits): {best_action_visits} | Best Action (Q-Value): {best_action_q}"
                action_surf = self.info_font.render(best_action_text, True, YELLOW)
                self.screen.blit(action_surf, (10, 10 + self.info_font.get_linesize()))
            except ValueError:
                pass


File: ui\mcts_visualizer\tree_layout.py
# File: ui/mcts_visualizer/tree_layout.py
from typing import Dict, Tuple, Optional, List
from mcts.node import MCTSNode
import math


class TreeLayout:
    """Calculates positions for nodes in the MCTS tree for visualization."""

    HORIZONTAL_SPACING = 50
    VERTICAL_SPACING = 80
    SUBTREE_HORIZONTAL_PADDING = 10

    def __init__(self, root_node: MCTSNode, canvas_width: int, canvas_height: int):
        self.root = root_node
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.positions: Dict[MCTSNode, Tuple[int, int]] = {}
        self.subtree_widths: Dict[MCTSNode, int] = {}

    def calculate_layout(self) -> Dict[MCTSNode, Tuple[int, int]]:
        """Calculates and returns the positions for all nodes."""
        self._calculate_subtree_widths(self.root)
        self._calculate_positions(
            self.root, self.canvas_width // 2, 50
        )  # Start root at top-center
        return self.positions

    def _calculate_subtree_widths(self, node: MCTSNode):
        """Recursively calculates the horizontal space needed for each subtree."""
        if not node.children:
            self.subtree_widths[node] = self.HORIZONTAL_SPACING
            return

        total_width = 0
        for child in node.children.values():
            self._calculate_subtree_widths(child)
            total_width += self.subtree_widths[child]

        # Add padding between subtrees
        total_width += max(0, len(node.children) - 1) * self.SUBTREE_HORIZONTAL_PADDING
        # Ensure node itself has minimum spacing
        self.subtree_widths[node] = max(total_width, self.HORIZONTAL_SPACING)

    def _calculate_positions(self, node: MCTSNode, x: int, y: int):
        """Recursively calculates the (x, y) position for each node."""
        self.positions[node] = (x, y)

        if not node.children:
            return

        num_children = len(node.children)
        total_children_width = (
            self.subtree_widths[node] - self.HORIZONTAL_SPACING
        )  # Width excluding node itself
        current_x = x - total_children_width // 2

        child_list = list(node.children.values())  # Consistent order
        for i, child in enumerate(child_list):
            child_subtree_width = self.subtree_widths[child]
            child_x = current_x + child_subtree_width // 2
            child_y = y + self.VERTICAL_SPACING
            self._calculate_positions(child, child_x, child_y)
            current_x += child_subtree_width
            if i < num_children - 1:
                current_x += self.SUBTREE_HORIZONTAL_PADDING  # Add padding


File: ui\panels\game_area.py
import pygame
import math
import traceback
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import copy

from config import (
    VisConfig,
    EnvConfig,
    TrainConfig,
    BLACK,
    BLUE,
    RED,
    GRAY,
    YELLOW,
    LIGHTG,
    WHITE,
    DARK_RED,
    DARK_GREEN,
    GREEN,
    LINE_CLEAR_FLASH_COLOR,
    GAME_OVER_FLASH_COLOR,
    DARK_GRAY,
)
from environment.game_state import GameState, StateType
from environment.shape import Shape
from environment.triangle import Triangle

logger = logging.getLogger(__name__)


class GameAreaRenderer:
    """Renders the right panel: Shows live self-play worker environments and best game state."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()
        self.best_state_surface_cache: Optional[pygame.Surface] = None
        self.last_best_state_size: Tuple[int, int] = (0, 0)
        self.last_best_state_score: Optional[int] = None
        self.last_best_state_step: Optional[int] = None
        self.placeholder_surface_cache: Optional[pygame.Surface] = None
        self.last_placeholder_size: Tuple[int, int] = (0, 0)
        self.last_placeholder_message_key: str = ""

    def _init_fonts(self):
        """Initializes fonts used in the game area."""
        fonts = {}
        font_configs = {
            "env_score": 16,
            "env_overlay": 24,
            "env_info": 14,
            "ui": 24,
            "placeholder": 30,
            "placeholder_detail": 22,
            "best_state_title": 20,
            "best_state_score": 18,
            "best_state_step": 16,
        }
        for key, size in font_configs.items():
            try:
                fonts[key] = pygame.font.SysFont(None, size)
            except Exception:
                try:
                    fonts[key] = pygame.font.Font(None, size)
                except Exception as e:
                    logger.error(f"ERROR: Font '{key}' failed to load: {e}")
                    fonts[key] = None
        # Fallbacks
        if fonts.get("ui") is None:
            fonts["ui"] = pygame.font.Font(None, 24)
        if fonts.get("placeholder") is None:
            fonts["placeholder"] = pygame.font.Font(None, 30)
        if fonts.get("env_score") is None:
            fonts["env_score"] = pygame.font.Font(None, 16)
        if fonts.get("env_overlay") is None:
            fonts["env_overlay"] = pygame.font.Font(None, 24)
        if fonts.get("env_info") is None:
            fonts["env_info"] = pygame.font.Font(None, 14)
        if fonts.get("best_state_title") is None:
            fonts["best_state_title"] = pygame.font.Font(None, 20)
        if fonts.get("best_state_score") is None:
            fonts["best_state_score"] = pygame.font.Font(None, 18)
        if fonts.get("best_state_step") is None:
            fonts["best_state_step"] = pygame.font.Font(None, 16)
        return fonts

    def render(
        self,
        panel_width: int,
        panel_x_offset: int,
        worker_render_data: List[Optional[Dict[str, Any]]],
        num_envs: int,
        env_config: Optional[EnvConfig],
        best_game_state_data: Optional[Dict[str, Any]],
    ):
        """Renders the grid of live self-play worker environments and best game found."""
        current_height = self.screen.get_height()
        ga_rect = pygame.Rect(panel_x_offset, 0, panel_width, current_height)
        if ga_rect.width <= 0 or ga_rect.height <= 0:
            return

        pygame.draw.rect(self.screen, VisConfig.DARK_GRAY, ga_rect)

        best_game_display_height = (
            int(current_height * 0.2) if best_game_state_data else 0
        )
        best_game_rect = pygame.Rect(
            ga_rect.x, ga_rect.y, ga_rect.width, best_game_display_height
        )
        env_grid_rect = pygame.Rect(
            ga_rect.x,
            ga_rect.y + best_game_display_height,
            ga_rect.width,
            ga_rect.height - best_game_display_height,
        )

        if best_game_display_height > 0 and best_game_rect.height > 20:
            self._render_best_game(best_game_rect, best_game_state_data, env_config)
        elif best_game_state_data and best_game_rect.height <= 20:
            logger.warning("Best game display area too small to render.")

        num_to_render = self.vis_config.NUM_ENVS_TO_RENDER
        actual_render_data = worker_render_data[:num_to_render]

        if not actual_render_data or not env_config:
            self._render_placeholder(env_grid_rect, "Waiting for Self-Play Workers...")
            return

        cols_env, rows_env, cell_w, cell_h_total = self._calculate_grid_layout(
            env_grid_rect, len(actual_render_data)
        )

        if cell_w > 10 and cell_h_total > 40:
            self._render_env_grid(
                actual_render_data,
                env_config,
                env_grid_rect,
                cols_env,
                rows_env,
                cell_w,
                cell_h_total,
            )
        else:
            self._render_too_small_message(env_grid_rect, cell_w, cell_h_total)

        if len(actual_render_data) < num_envs:
            self._render_render_limit_text(ga_rect, len(actual_render_data), num_envs)

    def _render_placeholder(self, area_rect: pygame.Rect, message: str):
        """Renders a simple placeholder message."""
        pygame.draw.rect(self.screen, (60, 60, 70), area_rect, 1)
        font = self.fonts.get("placeholder")
        if font:
            text_surf = font.render(message, True, LIGHTG)
            self.screen.blit(text_surf, text_surf.get_rect(center=area_rect.center))

    def _calculate_grid_layout(
        self, available_rect: pygame.Rect, num_items: int
    ) -> Tuple[int, int, int, int]:
        """Calculates layout for multiple small environment grids."""
        if available_rect.width <= 0 or available_rect.height <= 0 or num_items <= 0:
            return 0, 0, 0, 0
        aspect = available_rect.width / max(1, available_rect.height)
        cols = max(1, int(math.sqrt(num_items * aspect)))
        rows = max(1, math.ceil(num_items / cols))
        while cols * rows < num_items:
            if (cols + 1) / rows < aspect:
                cols += 1
            else:
                rows += 1
        while cols * (rows - 1) >= num_items and rows > 1:
            rows -= 1
        while (cols - 1) * rows >= num_items and cols > 1:
            cols -= 1
        sp = self.vis_config.ENV_SPACING
        cw_total = max(1, (available_rect.width - (cols + 1) * sp) // cols)
        ch_total = max(1, (available_rect.height - (rows + 1) * sp) // rows)
        return cols, rows, cw_total, ch_total

    def _render_env_grid(
        self,
        worker_render_data: List[Optional[Dict[str, Any]]],
        env_config: EnvConfig,
        grid_area_rect: pygame.Rect,
        cols: int,
        rows: int,
        cell_w: int,
        cell_h_total: int,
    ):
        """Renders the grid of small environment previews."""
        env_idx = 0
        sp = self.vis_config.ENV_SPACING
        info_h, shapes_h = 12, 22
        grid_cell_h = max(1, cell_h_total - shapes_h - info_h - 2)

        for r in range(rows):
            for c in range(cols):
                if env_idx >= len(worker_render_data):
                    break
                env_x = grid_area_rect.x + sp * (c + 1) + c * cell_w
                env_y = grid_area_rect.y + sp * (r + 1) + r * cell_h_total
                env_rect_total = pygame.Rect(env_x, env_y, cell_w, cell_h_total)
                clip_rect_total = env_rect_total.clip(self.screen.get_rect())
                if clip_rect_total.width <= 0 or clip_rect_total.height <= 0:
                    env_idx += 1
                    continue

                render_data = worker_render_data[env_idx]
                grid_rect_local = pygame.Rect(0, 0, cell_w, grid_cell_h)
                shapes_rect_local = pygame.Rect(0, grid_cell_h + 1, cell_w, shapes_h)
                info_rect_local = pygame.Rect(
                    0, grid_cell_h + shapes_h + 2, cell_w, info_h
                )

                try:
                    cell_surf = self.screen.subsurface(clip_rect_total)
                    cell_surf.fill(VisConfig.DARK_GRAY)

                    if render_data and render_data.get("state_dict"):
                        env_state_dict: StateType = render_data["state_dict"]
                        env_stats: Dict[str, Any] = render_data.get("stats", {})

                        # Render Grid
                        if grid_rect_local.height > 0:
                            clipped_grid_rect_local = grid_rect_local.clip(
                                cell_surf.get_rect()
                            )
                            if (
                                clipped_grid_rect_local.width > 0
                                and clipped_grid_rect_local.height > 0
                            ):
                                grid_surf = cell_surf.subsurface(
                                    clipped_grid_rect_local
                                )
                                occupancy_grid = env_state_dict.get("grid")[0]
                                orientation_grid = env_state_dict.get("grid")[1]
                                death_mask = env_state_dict.get(
                                    "death_mask",
                                    np.zeros_like(occupancy_grid, dtype=bool),
                                )
                                self._render_single_env_grid_from_arrays(
                                    grid_surf,
                                    occupancy_grid,
                                    orientation_grid,
                                    death_mask,
                                    env_config,
                                    env_stats,  # Pass full stats dict
                                )

                        # Render Shapes
                        if shapes_rect_local.height > 0:
                            clipped_shapes_rect_local = shapes_rect_local.clip(
                                cell_surf.get_rect()
                            )
                            if (
                                clipped_shapes_rect_local.width > 0
                                and clipped_shapes_rect_local.height > 0
                            ):
                                shapes_surf = cell_surf.subsurface(
                                    clipped_shapes_rect_local
                                )
                                available_shapes_data = env_stats.get(
                                    "available_shapes_data", []
                                )
                                self._render_env_shapes_from_data(
                                    shapes_surf, available_shapes_data, env_config
                                )

                        # Render Info
                        if info_rect_local.height > 0:
                            clipped_info_rect_local = info_rect_local.clip(
                                cell_surf.get_rect()
                            )
                            if (
                                clipped_info_rect_local.width > 0
                                and clipped_info_rect_local.height > 0
                            ):
                                info_surf = cell_surf.subsurface(
                                    clipped_info_rect_local
                                )
                                self._render_env_info(info_surf, env_idx, env_stats)
                    else:  # Placeholder rendering
                        pygame.draw.rect(cell_surf, (20, 20, 20), cell_surf.get_rect())
                        pygame.draw.rect(
                            cell_surf, (60, 60, 60), cell_surf.get_rect(), 1
                        )
                        font = self.fonts.get("env_info")
                        if font:
                            status = (
                                render_data.get("stats", {}).get("status", "N/A")
                                if render_data
                                else "N/A"
                            )
                            text_surf = font.render(status, True, GRAY)
                            cell_surf.blit(
                                text_surf,
                                text_surf.get_rect(center=cell_surf.get_rect().center),
                            )

                    pygame.draw.rect(
                        cell_surf, VisConfig.LIGHTG, cell_surf.get_rect(), 1
                    )
                except ValueError as e:
                    logger.error(
                        f"Error creating subsurface for env cell {env_idx} ({clip_rect_total}): {e}"
                    )
                    pygame.draw.rect(self.screen, (50, 0, 50), clip_rect_total, 1)
                except Exception as e:
                    logger.error(
                        f"Error rendering env cell {env_idx}: {e}", exc_info=True
                    )
                    pygame.draw.rect(self.screen, (50, 0, 50), clip_rect_total, 1)
                env_idx += 1
            if env_idx >= len(worker_render_data):
                break

    def _render_single_env_grid_from_arrays(
        self,
        surf: pygame.Surface,
        occupancy_grid: np.ndarray,
        orientation_grid: np.ndarray,
        death_mask: np.ndarray,
        env_config: EnvConfig,
        stats: Dict[str, Any],  # Accept full stats dict
    ):
        """Renders the grid portion using occupancy, orientation, and death arrays."""
        cw, ch = surf.get_width(), surf.get_height()
        bg = VisConfig.GRAY
        surf.fill(bg)

        try:
            pad = self.vis_config.ENV_GRID_PADDING
            dw, dh = max(1, cw - 2 * pad), max(1, ch - 2 * pad)
            gr, gcw_eff = env_config.ROWS, env_config.COLS * 0.75 + 0.25
            if gr <= 0 or gcw_eff <= 0:
                return

            scale = min(dw / gcw_eff, dh / gr) if gr > 0 and gcw_eff > 0 else 0
            if scale <= 0:
                return

            tcw, tch = max(1, scale), max(1, scale)
            fpw, fph = gcw_eff * scale, gr * scale
            ox, oy = pad + (dw - fpw) / 2, pad + (dh - fph) / 2

            rows, cols = occupancy_grid.shape
            for r in range(rows):
                for c in range(cols):
                    if not (
                        0 <= r < death_mask.shape[0] and 0 <= c < death_mask.shape[1]
                    ):
                        continue

                    is_death = death_mask[r, c] > 0.5
                    is_occupied = occupancy_grid[r, c] > 0.5
                    expected_is_up = (r + c) % 2 == 0

                    temp_tri = Triangle(r, c, expected_is_up)
                    try:
                        pts = temp_tri.get_points(
                            ox=ox, oy=oy, cw=int(tcw), ch=int(tch)
                        )
                        color = VisConfig.LIGHTG

                        if is_death:
                            color = BLACK
                        elif is_occupied:
                            color = VisConfig.WHITE

                        pygame.draw.polygon(surf, color, pts)
                        if not is_death:
                            pygame.draw.polygon(surf, VisConfig.DARK_GRAY, pts, 1)
                    except Exception as tri_err:
                        pass
        except Exception as grid_err:
            logger.error(f"Error rendering grid from arrays: {grid_err}")
            pygame.draw.rect(surf, RED, surf.get_rect(), 2)

        # Score Overlay - Use game_score from stats dict
        try:
            score_font = self.fonts.get("env_score")
            score = stats.get("game_score", "?")  # Get score from the passed stats
            if score_font:
                score_surf = score_font.render(
                    f"Score: {score}", True, WHITE, (0, 0, 0, 180)
                )  # Changed label
                surf.blit(score_surf, (2, 2))
        except Exception as e:
            logger.error(f"Error rendering score overlay: {e}")

        # State Overlays
        status = stats.get("status", "")
        if "Error" in status:
            self._render_overlay_text(surf, "ERROR", RED)
        elif "Finished" in status:
            self._render_overlay_text(surf, "DONE", BLUE)

    def _render_env_shapes_from_data(
        self,
        surf: pygame.Surface,
        available_shapes_data: List[Optional[Dict[str, Any]]],
        env_config: EnvConfig,
    ):
        """Renders the available shapes using triangle list and color data."""
        sw, sh = surf.get_width(), surf.get_height()
        if sw <= 0 or sh <= 0:
            return

        num_slots = env_config.NUM_SHAPE_SLOTS
        pad = 2
        total_pad_w = (num_slots + 1) * pad
        avail_w = sw - total_pad_w
        if avail_w <= 0:
            return

        w_per = avail_w / num_slots
        h_lim = sh - 2 * pad
        dim = max(5, int(min(w_per, h_lim)))

        start_x = pad + (sw - (num_slots * dim + (num_slots - 1) * pad)) / 2
        start_y = pad + (sh - dim) / 2
        curr_x = start_x

        for i in range(num_slots):
            shape_data = (
                available_shapes_data[i] if i < len(available_shapes_data) else None
            )
            rect = pygame.Rect(int(curr_x), int(start_y), dim, dim)
            if rect.right > sw - pad:
                break

            pygame.draw.rect(surf, (50, 50, 50), rect, border_radius=2)

            if shape_data and isinstance(shape_data, dict):
                try:
                    temp_shape = Shape()
                    temp_shape.triangles = shape_data.get("triangles", [])
                    temp_shape.color = shape_data.get("color", WHITE)
                    self._render_single_shape_in_preview_box(surf, temp_shape, rect)
                except Exception as e:
                    logger.error(f"Error rendering shape preview from data: {e}")
                    pygame.draw.line(surf, RED, rect.topleft, rect.bottomright, 1)
            else:
                pygame.draw.line(surf, GRAY, rect.topleft, rect.bottomright, 1)
                pygame.draw.line(surf, GRAY, rect.topright, rect.bottomleft, 1)

            curr_x += dim + pad

    def _render_single_shape_in_preview_box(
        self,
        surf: pygame.Surface,
        shape_obj: Shape,
        preview_rect: pygame.Rect,
    ):
        """Renders a single shape scaled to fit within its preview box."""
        try:
            inner_padding = 2
            clipped_preview_rect = preview_rect.clip(surf.get_rect())
            if clipped_preview_rect.width <= 0 or clipped_preview_rect.height <= 0:
                return
            shape_surf = surf.subsurface(clipped_preview_rect)

            render_area_w = clipped_preview_rect.width - 2 * inner_padding
            render_area_h = clipped_preview_rect.height - 2 * inner_padding
            if render_area_w <= 0 or render_area_h <= 0:
                return

            temp_shape_surf = pygame.Surface(
                (render_area_w, render_area_h), pygame.SRCALPHA
            )
            temp_shape_surf.fill((0, 0, 0, 0))

            self._render_single_shape(
                temp_shape_surf, shape_obj, min(render_area_w, render_area_h)
            )

            shape_surf.blit(temp_shape_surf, (inner_padding, inner_padding))
        except ValueError as sub_err:
            logger.error(f"Error creating subsurface for shape preview: {sub_err}")
            pygame.draw.rect(surf, RED, preview_rect, 1)
        except Exception as e:
            logger.error(f"Error rendering single shape preview: {e}")
            pygame.draw.rect(surf, RED, preview_rect, 1)

    def _render_env_info(
        self, surf: pygame.Surface, worker_idx: int, stats: Dict[str, Any]
    ):
        """Renders worker ID and current game step with descriptive labels."""
        font = self.fonts.get("env_info")
        if not font:
            return
        game_step = stats.get("game_steps", "?")
        # Use more descriptive labels
        info_text = f"Worker: {worker_idx} | Step: {game_step}"
        try:
            text_surf = font.render(info_text, True, LIGHTG)
            # Center the text horizontally within the info area
            text_rect = text_surf.get_rect(centerx=surf.get_width() // 2, top=0)
            surf.blit(text_surf, text_rect)
        except Exception as e:
            logger.error(f"Error rendering env info: {e}")

    def _render_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders overlay text, scaling font if needed."""
        try:
            font = self.fonts.get("env_overlay")
            if not font:
                return
            max_w = surf.get_width() * 0.9
            original_size = font.get_height()
            current_size = original_size
            surf_txt = font.render(text, True, WHITE)
            while surf_txt.get_width() > max_w and current_size > 8:
                current_size -= 2
                try:
                    font = pygame.font.SysFont(None, current_size)
                except:
                    font = pygame.font.Font(None, current_size)
                surf_txt = font.render(text, True, WHITE)
            bg_rgba = (color[0] // 2, color[1] // 2, color[2] // 2, 220)
            surf_bg = font.render(text, True, WHITE, bg_rgba)
            rect = surf_bg.get_rect(center=surf.get_rect().center)
            surf.blit(surf_bg, rect)
        except Exception as e:
            logger.error(f"Error rendering overlay '{text}': {e}")

    def _render_single_shape(self, surf: pygame.Surface, shape: Shape, target_dim: int):
        """Renders a single shape scaled to fit within the target surface."""
        if not shape or not shape.triangles or target_dim <= 0:
            return
        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h_cells = max(1, max_r - min_r + 1)
        shape_w_cells_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)

        surf_w, surf_h = surf.get_size()
        scale_h = surf_h / shape_h_cells if shape_h_cells > 0 else surf_h
        scale_w = surf_w / shape_w_cells_eff if shape_w_cells_eff > 0 else surf_w
        scale = max(1, min(scale_h, scale_w))

        total_w_pixels = shape_w_cells_eff * scale
        total_h_pixels = shape_h_cells * scale
        ox = (surf_w - total_w_pixels) / 2 - min_c * (scale * 0.75)
        oy = (surf_h - total_h_pixels) / 2 - min_r * scale

        for dr, dc, is_up in shape.triangles:
            temp_tri = Triangle(0, 0, is_up)
            try:
                tri_x = ox + dc * (scale * 0.75)
                tri_y = oy + dr * scale
                pts = [
                    (int(p[0]), int(p[1]))
                    for p in temp_tri.get_points(
                        ox=tri_x, oy=tri_y, cw=int(scale), ch=int(scale)
                    )
                ]
                pygame.draw.polygon(surf, shape.color, pts)
                pygame.draw.polygon(surf, BLACK, pts, 1)
            except Exception as e:
                pass

    def _render_too_small_message(
        self, area_rect: pygame.Rect, cell_w: int, cell_h: int
    ):
        """Renders a message if the env cells are too small."""
        font = self.fonts.get("ui")
        if font:
            surf = font.render(f"Envs Too Small ({cell_w}x{cell_h})", True, GRAY)
            self.screen.blit(surf, surf.get_rect(center=area_rect.center))

    def _render_render_limit_text(
        self, ga_rect: pygame.Rect, num_rendered: int, num_total: int
    ):
        """Renders text indicating not all envs are shown."""
        font = self.fonts.get("ui")
        if font:
            surf = font.render(
                f"Rendering {num_rendered}/{num_total} Workers", True, YELLOW, BLACK
            )
            self.screen.blit(
                surf, surf.get_rect(bottomright=(ga_rect.right - 5, ga_rect.bottom - 5))
            )

    def _render_best_game(
        self,
        area_rect: pygame.Rect,
        best_game_data: Optional[Dict[str, Any]],
        env_config: Optional[EnvConfig],
    ):
        """Renders the best game state found so far."""
        pygame.draw.rect(self.screen, (20, 40, 20), area_rect)
        pygame.draw.rect(self.screen, GREEN, area_rect, 1)

        if not best_game_data or not env_config:
            logger.info(
                "[GameAreaRenderer] Best game data missing or not a dictionary. Rendering placeholder."
            )
            self._render_placeholder(area_rect, "No Best Game Yet")
            return

        title_font = self.fonts.get("best_state_title")
        score_font = self.fonts.get("best_state_score")
        step_font = self.fonts.get("best_state_step")

        padding = 5
        text_area_width = 100
        grid_area_width = area_rect.width - text_area_width - 3 * padding
        grid_area_height = area_rect.height - 2 * padding
        text_area_rect = pygame.Rect(
            area_rect.left + padding,
            area_rect.top + padding,
            text_area_width,
            grid_area_height,
        )
        grid_area_rect = pygame.Rect(
            text_area_rect.right + padding,
            area_rect.top + padding,
            grid_area_width,
            grid_area_height,
        )

        if title_font and score_font and step_font:
            score = best_game_data.get("score", "N/A")
            step = best_game_data.get("step", "N/A")
            title_surf = title_font.render("Best Game", True, YELLOW)
            score_surf = score_font.render(f"Score: {score}", True, WHITE)
            step_surf = step_font.render(f"Step: {step}", True, LIGHTG)
            title_rect = title_surf.get_rect(
                midtop=(text_area_rect.centerx, text_area_rect.top + 2)
            )
            score_rect = score_surf.get_rect(
                midtop=(text_area_rect.centerx, title_rect.bottom + 4)
            )
            step_rect = step_surf.get_rect(
                midtop=(text_area_rect.centerx, score_rect.bottom + 2)
            )
            self.screen.blit(title_surf, title_rect)
            self.screen.blit(score_surf, score_rect)
            self.screen.blit(step_surf, step_rect)

        if grid_area_rect.width > 10 and grid_area_rect.height > 10:
            game_state_dict = best_game_data.get("game_state_dict")
            if game_state_dict and isinstance(game_state_dict, dict):
                try:
                    if "grid" in game_state_dict:
                        grid_state_array = game_state_dict["grid"]
                        death_mask = game_state_dict.get(
                            "death_mask", np.zeros_like(grid_state_array[0], dtype=bool)
                        )
                        if (
                            isinstance(grid_state_array, np.ndarray)
                            and grid_state_array.ndim == 3
                            and grid_state_array.shape[0] >= 2
                        ):
                            self._render_grid_from_array(
                                grid_area_rect,
                                grid_state_array[0],
                                grid_state_array[1],
                                death_mask,
                                env_config,
                            )
                        else:
                            logger.warning(
                                f"Best game state 'grid' data invalid shape: {grid_state_array.shape if isinstance(grid_state_array, np.ndarray) else type(grid_state_array)}"
                            )
                            self._render_placeholder(
                                grid_area_rect, "Grid Data Invalid"
                            )
                    else:
                        logger.warning("Best game state dict missing 'grid' key.")
                        self._render_placeholder(grid_area_rect, "Grid Data Missing")
                except Exception as e:
                    logger.error(f"Error rendering best game grid: {e}", exc_info=True)
                    self._render_placeholder(grid_area_rect, "Grid Render Error")
            else:
                logger.info(
                    "[GameAreaRenderer] Best game state data dict missing or invalid. Rendering placeholder."
                )
                self._render_placeholder(grid_area_rect, "State Missing")

    def _render_grid_from_array(
        self,
        area_rect: pygame.Rect,
        occupancy_grid: np.ndarray,
        orientation_grid: np.ndarray,
        death_mask: np.ndarray,
        env_config: EnvConfig,
    ):
        """Simplified grid rendering directly from occupancy, orientation, and death arrays."""
        try:
            clipped_area_rect = area_rect.clip(self.screen.get_rect())
            if clipped_area_rect.width <= 0 or clipped_area_rect.height <= 0:
                return
            grid_surf = self.screen.subsurface(clipped_area_rect)
            grid_surf.fill(VisConfig.GRAY)

            rows, cols = occupancy_grid.shape
            if rows == 0 or cols == 0:
                return

            pad = 1
            dw, dh = max(1, clipped_area_rect.width - 2 * pad), max(
                1, clipped_area_rect.height - 2 * pad
            )
            gcw_eff = env_config.COLS * 0.75 + 0.25
            if env_config.ROWS <= 0 or gcw_eff <= 0:
                return

            scale = (
                min(dw / gcw_eff, dh / env_config.ROWS)
                if env_config.ROWS > 0 and gcw_eff > 0
                else 0
            )
            if scale <= 0:
                return

            tcw, tch = max(1, scale), max(1, scale)
            fpw, fph = gcw_eff * scale, env_config.ROWS * scale
            ox, oy = pad + (dw - fpw) / 2, pad + (dh - fph) / 2

            for r in range(rows):
                for c in range(cols):
                    if not (
                        0 <= r < death_mask.shape[0] and 0 <= c < death_mask.shape[1]
                    ):
                        continue

                    is_death = death_mask[r, c] > 0.5
                    is_occupied = occupancy_grid[r, c] > 0.5
                    expected_is_up = (r + c) % 2 == 0

                    temp_tri = Triangle(r, c, expected_is_up)
                    try:
                        pts = temp_tri.get_points(
                            ox=ox, oy=oy, cw=int(tcw), ch=int(tch)
                        )
                        color = VisConfig.LIGHTG

                        if is_death:
                            color = BLACK
                        elif is_occupied:
                            color = WHITE

                        pygame.draw.polygon(grid_surf, color, pts)
                        if not is_death:
                            pygame.draw.polygon(grid_surf, VisConfig.DARK_GRAY, pts, 1)
                    except Exception:
                        pass
        except ValueError as e:
            logger.error(
                f"Error creating subsurface for best grid ({clipped_area_rect}): {e}"
            )
            pygame.draw.rect(self.screen, RED, clipped_area_rect, 1)
        except Exception as e:
            logger.error(f"Error in _render_grid_from_array: {e}", exc_info=True)
            pygame.draw.rect(self.screen, RED, clipped_area_rect, 1)


File: ui\panels\left_panel.py
# File: ui/panels/left_panel.py
import pygame
from typing import Optional, Tuple, Dict, Any
import logging

from config import VisConfig
from ui.plotter import Plotter
from ui.input_handler import InputHandler
from .left_panel_components import (
    ButtonStatusRenderer,
    InfoTextRenderer,
    PlotAreaRenderer,
    NotificationRenderer,
)
from app_state import AppState

logger = logging.getLogger(__name__)


class LeftPanelRenderer:
    """Orchestrates rendering of the left panel using sub-components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.input_handler: Optional[InputHandler] = None

        # Initialize components
        self.button_status_renderer = ButtonStatusRenderer(self.screen, self.fonts)
        self.info_text_renderer = InfoTextRenderer(self.screen, self.fonts)
        self.notification_renderer = NotificationRenderer(self.screen, self.fonts)
        self.plot_area_renderer = PlotAreaRenderer(
            self.screen, self.fonts, self.plotter
        )

    def _init_fonts(self):
        """Initializes fonts used in the left panel."""
        fonts = {}
        font_configs = {
            "ui": 24,
            "status": 28,
            "detail": 16,
            "resource": 16,  # Kept for now, might remove later
            "notification_label": 16,
            "notification": 18,
            "plot_placeholder": 20,
            "plot_title_values": 8,
            "mcts_stats_label": 18,  # New font for MCTS labels
            "mcts_stats_value": 18,  # New font for MCTS values
        }
        for key, size in font_configs.items():
            try:
                fonts[key] = pygame.font.SysFont(None, size)
            except Exception:
                try:
                    fonts[key] = pygame.font.Font(None, size)
                except Exception as e:
                    logger.error(f"ERROR: Font '{key}' failed: {e}")
                    fonts[key] = None
        # Ensure essential fonts have fallbacks
        if fonts.get("ui") is None:
            fonts["ui"] = pygame.font.Font(None, 24)
        if fonts.get("status") is None:
            fonts["status"] = pygame.font.Font(None, 28)
        if fonts.get("mcts_stats_label") is None:
            fonts["mcts_stats_label"] = pygame.font.Font(None, 18)
        if fonts.get("mcts_stats_value") is None:
            fonts["mcts_stats_value"] = pygame.font.Font(None, 18)
        return fonts

    def _get_background_color(self, status: str) -> Tuple[int, int, int]:
        """Determines background color based on status."""
        status_color_map = {
            "Ready": (30, 30, 30),
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Debugging Grid": (40, 30, 40),
            "Initializing": (40, 40, 40),
            "Running AlphaZero": (30, 50, 30),
        }
        base_status = status.split(" (")[0] if "(" in status else status
        return status_color_map.get(base_status, (30, 30, 30))

    def render(self, panel_width: int, **kwargs):
        """Renders the entire left panel within the given width."""
        current_height = self.screen.get_height()
        lp_rect = pygame.Rect(0, 0, panel_width, current_height)
        status = kwargs.get("status", "")
        bg_color = self._get_background_color(status)
        pygame.draw.rect(self.screen, bg_color, lp_rect)

        current_y = 10
        # Define render order and estimated heights
        # InfoTextRenderer now includes MCTS stats, might need more height
        render_order: List[Tuple[callable, int, Dict[str, Any]]] = [
            (
                self.button_status_renderer.render,
                60,
                {
                    "app_state": kwargs.get("app_state", ""),
                    "is_process_running": kwargs.get("is_process_running", False),
                    "status": status,
                    "stats_summary": kwargs.get("stats_summary", {}),
                    "update_progress_details": kwargs.get(
                        "update_progress_details", {}
                    ),
                },
            ),
            (
                self.info_text_renderer.render,
                120,
                {  # Increased height estimate
                    "stats_summary": kwargs.get("stats_summary", {}),
                    "agent_param_count": kwargs.get("agent_param_count", 0),
                    "worker_counts": kwargs.get("worker_counts", {}),
                },
            ),
            (
                self.notification_renderer.render,
                70,
                {  # Reduced height estimate
                    "stats_summary": kwargs.get("stats_summary", {}),
                },
            ),
        ]

        # Render static components sequentially
        for render_func, fallback_height, func_kwargs in render_order:
            try:
                # Pass specific arguments required by each component
                if render_func == self.notification_renderer.render:
                    # Notification renderer takes rect directly
                    notification_rect = pygame.Rect(
                        10, current_y + 5, panel_width - 20, fallback_height
                    )
                    render_func(notification_rect, func_kwargs.get("stats_summary", {}))
                    next_y = notification_rect.bottom
                else:
                    next_y = render_func(
                        y_start=current_y + 5,
                        panel_width=panel_width,
                        **func_kwargs,  # Pass the specific kwargs for this function
                    )

                current_y = (
                    next_y
                    if isinstance(next_y, (int, float))
                    else current_y + fallback_height + 5
                )
            except Exception as e:
                logger.error(
                    f"Error rendering component {render_func.__name__}: {e}",
                    exc_info=True,
                )
                # Draw error box for the failed component
                error_rect = pygame.Rect(
                    10, current_y + 5, panel_width - 20, fallback_height
                )
                pygame.draw.rect(self.screen, VisConfig.RED, error_rect, 1)
                current_y += fallback_height + 5  # Fallback increment

        # --- Render Plots Area ---
        app_state_str = kwargs.get("app_state", AppState.UNKNOWN.value)
        # Render plots only when in the main menu
        should_render_plots = app_state_str == AppState.MAIN_MENU.value

        plot_y_start = current_y + 5
        try:
            self.plot_area_renderer.render(
                y_start=plot_y_start,
                panel_width=panel_width,
                screen_height=current_height,
                plot_data=kwargs.get("plot_data", {}),
                status=status,
                render_enabled=should_render_plots,
            )
        except Exception as e:
            logger.error(f"Error in plot_area_renderer: {e}", exc_info=True)
            plot_area_height = current_height - plot_y_start - 10
            plot_area_width = panel_width - 20
            if plot_area_width > 10 and plot_area_height > 10:
                plot_area_rect = pygame.Rect(
                    10, plot_y_start, plot_area_width, plot_area_height
                )
                pygame.draw.rect(self.screen, (80, 0, 0), plot_area_rect, 2)


File: ui\panels\__init__.py
from .left_panel import LeftPanelRenderer
from .game_area import GameAreaRenderer

__all__ = ["LeftPanelRenderer", "GameAreaRenderer"]


File: ui\panels\left_panel_components\button_status_renderer.py
# File: ui/panels/left_panel_components/button_status_renderer.py
import pygame
import time
import math
from typing import Dict, Tuple, Any, Optional, TYPE_CHECKING

from config import (
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    LIGHTG,
    GREEN,
    DARK_GREEN,
    BLUE,
    GRAY,  # Added GRAY
)
from config.core import (
    TrainConfig,
)
from utils.helpers import format_eta
from ui.input_handler import InputHandler

if TYPE_CHECKING:
    from main_pygame import MainApp


class ButtonStatusRenderer:
    """Renders the top buttons and compact status block, including buffering progress."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("status", pygame.font.Font(None, 28))
        self.detail_font = fonts.get("detail", pygame.font.Font(None, 16))
        self.progress_font = fonts.get("detail", pygame.font.Font(None, 14))
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.input_handler_ref: Optional[InputHandler] = None
        self.app_ref: Optional["MainApp"] = None
        self.train_config = TrainConfig()

    def _draw_button(
        self,
        rect: pygame.Rect,
        text: str,
        base_color: Tuple[int, int, int],
        active_color: Optional[Tuple[int, int, int]] = None,
        is_active: bool = False,
        enabled: bool = True,
    ):
        """Helper to draw a single button."""
        final_color = base_color
        if not enabled:
            final_color = tuple(
                max(30, c // 2) for c in base_color[:3]
            )  # Darken disabled
            final_color = GRAY  # Use consistent gray for disabled
        elif is_active and active_color:
            final_color = active_color

        pygame.draw.rect(self.screen, final_color, rect, border_radius=5)
        text_color = WHITE if enabled else (150, 150, 150)
        if self.ui_font:
            label_surface = self.ui_font.render(text, True, text_color)
            self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))
        else:
            pygame.draw.line(self.screen, RED, rect.topleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, RED, rect.topright, rect.bottomleft, 2)

    def _render_progress_bar(
        self,
        y_pos: int,
        panel_width: int,
        current_value: int,
        target_value: int,
        label: str,
    ) -> int:
        """Renders a progress bar."""
        if not self.progress_font:
            return y_pos

        bar_height = 18
        bar_width = panel_width * 0.8
        bar_x = (panel_width - bar_width) / 2
        bar_rect = pygame.Rect(bar_x, y_pos, bar_width, bar_height)

        progress = 0.0
        if target_value > 0:
            progress = min(1.0, max(0.0, current_value / target_value))

        bg_color = (50, 50, 50)
        border_color = LIGHTG
        pygame.draw.rect(self.screen, bg_color, bar_rect, border_radius=3)

        fill_width = int(bar_width * progress)
        fill_rect = pygame.Rect(bar_x, y_pos, fill_width, bar_height)
        fill_color = BLUE
        pygame.draw.rect(
            self.screen,
            fill_color,
            fill_rect,
            border_top_left_radius=3,
            border_bottom_left_radius=3,
            border_top_right_radius=3 if progress >= 1.0 else 0,
            border_bottom_right_radius=3 if progress >= 1.0 else 0,
        )

        pygame.draw.rect(self.screen, border_color, bar_rect, 1, border_radius=3)

        progress_text = f"{label}: {current_value:,}/{target_value:,}".replace(",", "_")
        text_surf = self.progress_font.render(progress_text, True, WHITE)
        text_rect = text_surf.get_rect(center=bar_rect.center)
        self.screen.blit(text_surf, text_rect)

        return int(bar_rect.bottom)

    def _render_compact_status(
        self,
        y_start: int,
        panel_width: int,
        status: str,
        stats_summary: Dict[str, Any],
        is_running: bool,
    ) -> int:
        """Renders the compact status block, including buffering progress if applicable."""
        x_margin, current_y = 10, y_start
        line_height_status = self.status_font.get_linesize()
        line_height_detail = self.detail_font.get_linesize()

        # 1. Render Status Text
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
        elif "Running AlphaZero" in status:
            status_color = GREEN

        try:
            status_surface = self.status_font.render(status_text, True, status_color)
            status_rect = status_surface.get_rect(topleft=(x_margin, current_y))
            self.screen.blit(status_surface, status_rect)
            current_y += line_height_status
        except Exception as e:
            print(f"Error rendering status text: {e}")
            current_y += 20

        # 2. Render Global Step and Episodes OR Buffering Progress
        global_step = stats_summary.get("global_step", 0)
        total_episodes = stats_summary.get("total_episodes", 0)
        buffer_size = stats_summary.get("buffer_size", 0)
        min_buffer = self.train_config.MIN_BUFFER_SIZE_TO_TRAIN

        is_buffering = is_running and global_step == 0 and buffer_size < min_buffer

        if not is_buffering:
            global_step_str = f"{global_step:,}".replace(",", "_")
            eps_str = f"{total_episodes:,}".replace(",", "_")
            line2_text = f"Step: {global_step_str} | Episodes: {eps_str}"
            try:
                line2_surface = self.detail_font.render(line2_text, True, LIGHTG)
                line2_rect = line2_surface.get_rect(topleft=(x_margin, current_y))
                clip_width = max(0, panel_width - line2_rect.left - x_margin)
                blit_area = (
                    pygame.Rect(0, 0, clip_width, line2_rect.height)
                    if line2_rect.width > clip_width
                    else None
                )
                self.screen.blit(line2_surface, line2_rect, area=blit_area)
                current_y += line_height_detail + 2
            except Exception as e:
                print(f"Error rendering step/ep text: {e}")
                current_y += 15
        else:
            current_y += 2
            next_y_after_bar = self._render_progress_bar(
                current_y,
                panel_width,
                buffer_size,
                min_buffer,
                "Buffering",
            )
            current_y = next_y_after_bar + 5

        return int(current_y)

    def render(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        update_progress_details: Dict[str, Any],
    ) -> int:
        """Renders buttons and status. Returns next_y."""
        from app_state import AppState

        next_y = y_start

        run_stop_btn_rect = (
            self.input_handler_ref.run_stop_btn_rect
            if self.input_handler_ref
            else pygame.Rect(10, y_start, 150, 40)  # Fallback rect
        )
        cleanup_btn_rect = (
            self.input_handler_ref.cleanup_btn_rect
            if self.input_handler_ref
            else pygame.Rect(
                run_stop_btn_rect.right + 10, y_start, 160, 40
            )  # Fallback rect
        )
        demo_btn_rect = (
            self.input_handler_ref.demo_btn_rect
            if self.input_handler_ref
            else pygame.Rect(
                cleanup_btn_rect.right + 10, y_start, 120, 40
            )  # Fallback rect
        )
        debug_btn_rect = (
            self.input_handler_ref.debug_btn_rect
            if self.input_handler_ref
            else pygame.Rect(
                demo_btn_rect.right + 10, y_start, 120, 40
            )  # Fallback rect
        )

        is_running = is_process_running  # Use the passed flag

        # Render Buttons
        run_stop_text = "Stop Run" if is_running else "Run AlphaZero"
        run_stop_base_color = (40, 80, 40)
        run_stop_active_color = (100, 40, 40)
        self._draw_button(
            run_stop_btn_rect,
            run_stop_text,
            run_stop_base_color,
            active_color=run_stop_active_color,
            is_active=is_running,
            enabled=(app_state == AppState.MAIN_MENU.value),
        )

        other_buttons_enabled = (
            app_state == AppState.MAIN_MENU.value
        ) and not is_running
        self._draw_button(
            cleanup_btn_rect,
            "Cleanup This Run",
            (100, 40, 40),
            enabled=other_buttons_enabled,
        )
        self._draw_button(
            demo_btn_rect, "Play Demo", (40, 100, 40), enabled=other_buttons_enabled
        )
        self._draw_button(
            debug_btn_rect, "Debug Mode", (100, 40, 100), enabled=other_buttons_enabled
        )

        button_bottom = max(
            run_stop_btn_rect.bottom,
            cleanup_btn_rect.bottom,
            demo_btn_rect.bottom,
            debug_btn_rect.bottom,
        )
        next_y = int(button_bottom) + 10

        # Render Status Block
        status_block_y = next_y
        next_y = self._render_compact_status(
            status_block_y, panel_width, status, stats_summary, is_running
        )

        return int(next_y)


File: ui\panels\left_panel_components\info_text_renderer.py
# File: ui/panels/left_panel_components/info_text_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple
import logging

from config import WHITE, LIGHTG, GRAY, YELLOW, GREEN
import config.general as config_general

logger = logging.getLogger(__name__)


class InfoTextRenderer:
    """Renders essential non-plotted information text."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.ui_font = fonts.get("ui", pygame.font.Font(None, 24))
        self.detail_font = fonts.get("detail", pygame.font.Font(None, 16))
        self.stats_summary_cache: Dict[str, Any] = {}

    def _get_network_description(self) -> str:
        """Builds a description string based on network components."""
        # Potentially add more detail here later if needed based on ModelConfig
        return "AlphaZero Neural Network"

    def _render_key_value_line(
        self,
        y_pos: int,
        panel_width: int,
        key: str,
        value: str,
        key_font: pygame.font.Font,
        value_font: pygame.font.Font,
        key_color=LIGHTG,
        value_color=WHITE,
    ) -> int:
        """Helper to render a single key-value line and return its bottom y."""
        x_pos_key = 10
        x_pos_val_offset = 5
        try:
            key_surf = key_font.render(f"{key}:", True, key_color)
            key_rect = key_surf.get_rect(topleft=(x_pos_key, y_pos))
            self.screen.blit(key_surf, key_rect)

            value_surf = value_font.render(f"{value}", True, value_color)
            value_rect = value_surf.get_rect(
                topleft=(key_rect.right + x_pos_val_offset, y_pos)
            )

            # Clip value text if it exceeds panel width
            clip_width = max(0, panel_width - value_rect.left - 10)
            blit_area = (
                pygame.Rect(0, 0, clip_width, value_rect.height)
                if value_rect.width > clip_width
                else None
            )
            self.screen.blit(value_surf, value_rect, area=blit_area)

            # Return the bottom-most y position of the rendered elements
            return max(key_rect.bottom, value_rect.bottom)
        except Exception as e:
            logger.error(f"Error rendering info line '{key}': {e}")
            # Fallback: increment y by line height if rendering fails
            return y_pos + key_font.get_linesize()

    def render(
        self,
        y_start: int,
        stats_summary: Dict[str, Any],
        panel_width: int,
        agent_param_count: int,
        worker_counts: Dict[str, int],
    ) -> int:
        """Renders the info text block. Returns next_y."""
        self.stats_summary_cache = stats_summary

        if not self.ui_font or not self.detail_font:
            logger.warning("Missing fonts for InfoTextRenderer.")
            return y_start

        current_y = y_start

        # --- General Info ---
        device_type_str = (
            config_general.DEVICE.type.upper() if config_general.DEVICE else "CPU"
        )
        network_desc = self._get_network_description()
        param_str = (
            f"{agent_param_count / 1e6:.2f} M" if agent_param_count > 0 else "N/A"
        )
        start_time_unix = stats_summary.get("start_time", 0.0)
        start_time_str = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_unix))
            if start_time_unix > 0
            else "N/A"
        )
        sp_workers = worker_counts.get("SelfPlay", 0)
        tr_workers = worker_counts.get("Training", 0)
        worker_str = f"SP: {sp_workers}, TR: {tr_workers}"
        # Use the instantaneous steps/sec calculated in summary
        steps_sec = stats_summary.get(
            "steps_per_second_avg", 0.0
        )  # Use average for display
        steps_sec_str = f"{steps_sec:.1f}"

        general_info_lines = [
            ("Device", device_type_str),
            ("Network", network_desc),
            ("Params", param_str),
            ("Workers", worker_str),
            ("Run Started", start_time_str),
            ("Steps/Sec (Avg)", steps_sec_str),  # Display current steps/sec
        ]

        # Render lines using the helper function
        line_spacing = 2  # Pixels between lines
        for key, value_str in general_info_lines:
            current_y = (
                self._render_key_value_line(
                    current_y, panel_width, key, value_str, self.ui_font, self.ui_font
                )
                + line_spacing
            )

        # Add a small buffer at the end before the next component
        return int(current_y) + 5


File: ui\panels\left_panel_components\notification_renderer.py
# File: ui/panels/left_panel_components/notification_renderer.py
import pygame
import time
from typing import Dict, Any, Tuple, Optional
from config import VisConfig, StatsConfig, WHITE, LIGHTG, GRAY, YELLOW, RED, GREEN
import numpy as np


class NotificationRenderer:
    """Renders the notification area (Simplified)."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.label_font = fonts.get("notification_label", pygame.font.Font(None, 16))
        self.value_font = fonts.get("notification", pygame.font.Font(None, 18))

    def render(
        self, area_rect: pygame.Rect, stats_summary: Dict[str, Any]
    ) -> Dict[str, pygame.Rect]:
        """Renders the simplified notification content (e.g., total episodes)."""
        stat_rects: Dict[str, pygame.Rect] = {}
        pygame.draw.rect(self.screen, (45, 45, 45), area_rect, border_radius=3)
        pygame.draw.rect(self.screen, LIGHTG, area_rect, 1, border_radius=3)
        stat_rects["Notification Area"] = area_rect

        if not self.value_font or not self.label_font:
            return stat_rects

        padding = 5
        line_height = self.value_font.get_linesize() + 2
        y = area_rect.top + padding

        # Display Total Episodes
        total_episodes = stats_summary.get("total_episodes", 0)
        label_surf = self.label_font.render("Total Episodes:", True, LIGHTG)
        value_surf = self.value_font.render(
            f"{total_episodes:,}".replace(",", "_"), True, WHITE
        )

        label_rect = label_surf.get_rect(topleft=(area_rect.left + padding, y))
        value_rect = value_surf.get_rect(topleft=(label_rect.right + 4, y))

        self.screen.blit(label_surf, label_rect)
        self.screen.blit(value_surf, value_rect)

        stat_rects["Total Episodes Info"] = label_rect.union(value_rect)

        # Best value rendering removed

        return stat_rects


File: ui\panels\left_panel_components\plot_area_renderer.py
# File: ui/panels/left_panel_components/plot_area_renderer.py
import pygame
from typing import Dict, Deque, Any, Optional, Tuple
import numpy as np
import logging

from config import (
    VisConfig,
    LIGHTG,
    WHITE,
    YELLOW,
    RED,
    GOOGLE_COLORS,
    GRAY,
)
from ui.plotter import Plotter

logger = logging.getLogger(__name__)


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
        self.placeholder_font = fonts.get(
            "plot_placeholder", pygame.font.Font(None, 20)
        )

    def render(
        self,
        y_start: int,
        panel_width: int,
        screen_height: int,
        plot_data: Dict[str, Deque],
        status: str,
        render_enabled: bool = True,  # Add the new argument with a default
    ):
        """Renders the plot area, conditionally based on render_enabled."""
        plot_area_y_start = y_start
        plot_area_height = screen_height - plot_area_y_start - 10
        plot_area_width = panel_width - 20

        if plot_area_width <= 50 or plot_area_height <= 50:
            # Optionally render a "too small" message if needed
            return

        plot_area_rect = pygame.Rect(
            10, plot_area_y_start, plot_area_width, plot_area_height
        )

        if not render_enabled:
            # Render placeholder if plots are explicitly disabled
            self._render_placeholder(plot_area_rect, "Plots Disabled")
            return

        # Attempt to get/create the plot surface only if enabled
        plot_surface = self.plotter.get_cached_or_updated_plot(
            plot_data, plot_area_width, plot_area_height
        )

        # Render the plot or a placeholder if data is missing/error
        if plot_surface:
            self.screen.blit(plot_surface, plot_area_rect.topleft)
        else:
            # Draw border and placeholder text
            placeholder_text = "Waiting for plot data..."
            if status == "Error":
                placeholder_text = "Plotting disabled due to error."
            elif not plot_data or not any(plot_data.values()):
                placeholder_text = "No plot data yet..."
            self._render_placeholder(plot_area_rect, placeholder_text)

    def _render_placeholder(self, plot_area_rect: pygame.Rect, message: str):
        """Renders a placeholder message within the plot area."""
        pygame.draw.rect(self.screen, (40, 40, 40), plot_area_rect, 1)
        if self.placeholder_font:
            placeholder_surf = self.placeholder_font.render(message, True, GRAY)
            placeholder_rect = placeholder_surf.get_rect(center=plot_area_rect.center)
            # Clip rendering to the plot area
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
        else:  # Fallback cross if font failed
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


File: ui\panels\left_panel_components\__init__.py
from .button_status_renderer import ButtonStatusRenderer
from .notification_renderer import NotificationRenderer
from .info_text_renderer import InfoTextRenderer

from .plot_area_renderer import PlotAreaRenderer


__all__ = [
    "ButtonStatusRenderer",
    "NotificationRenderer",
    "InfoTextRenderer",
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
]


File: visualization\__init__.py


File: workers\self_play_worker.py
# File: workers/self_play_worker.py
import threading
import time
import queue
import traceback
import torch
import copy
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
import logging
import numpy as np

from environment.game_state import GameState, StateType
from environment.shape import Shape  # Import Shape
from mcts import MCTS
from config import EnvConfig, MCTSConfig, TrainConfig
from utils.types import ActionType

if TYPE_CHECKING:
    from agent.alphazero_net import AlphaZeroNet
    from stats.aggregator import StatsAggregator

ExperienceTuple = Tuple[StateType, Dict[ActionType, float], int]
ProcessedExperienceBatch = List[Tuple[StateType, Dict[ActionType, float], float]]

logger = logging.getLogger(__name__)


class SelfPlayWorker(threading.Thread):
    """Plays games using MCTS to generate training data."""

    INTERMEDIATE_STATS_INTERVAL_SEC = 5.0
    MCTS_NN_BATCH_SIZE = 32

    def __init__(
        self,
        worker_id: int,
        agent: "AlphaZeroNet",
        mcts: MCTS,
        experience_queue: queue.Queue,
        stats_aggregator: "StatsAggregator",
        stop_event: threading.Event,
        env_config: EnvConfig,
        mcts_config: MCTSConfig,
        device: torch.device,
        max_game_steps: Optional[int] = None,
    ):
        super().__init__(daemon=True, name=f"SelfPlayWorker-{worker_id}")
        self.worker_id = worker_id
        self.agent = agent
        self.mcts = MCTS(
            network_predictor=self.agent.predict_batch,
            config=mcts_config,
            env_config=env_config,
            batch_size=self.MCTS_NN_BATCH_SIZE,
            stop_event=stop_event,
        )
        self.experience_queue = experience_queue
        self.stats_aggregator = stats_aggregator
        self.stop_event = stop_event
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.device = device
        self.max_game_steps = max_game_steps if max_game_steps else float("inf")
        self.log_prefix = f"[SelfPlayWorker-{self.worker_id}]"
        self.last_intermediate_stats_time = 0.0

        self._current_game_state_lock = threading.Lock()
        self._current_render_state_dict: Optional[StateType] = None
        self._last_stats: Dict[str, Any] = {
            "status": "Initialized",
            "game_steps": 0,
            "game_score": 0,
            "mcts_sim_time": 0.0,
            "mcts_nn_time": 0.0,
            "mcts_nodes_explored": 0,
            "mcts_avg_depth": 0.0,
            "available_shapes_data": [],  # Add field for shape render data
        }
        logger.info(
            f"{self.log_prefix} Initialized with MCTS Batch Size: {self.MCTS_NN_BATCH_SIZE}"
        )

    def get_init_args(self) -> Dict[str, Any]:
        """Returns arguments needed to re-initialize the thread."""
        return {
            "worker_id": self.worker_id,
            "agent": self.agent,
            "mcts": self.mcts,
            "experience_queue": self.experience_queue,
            "stats_aggregator": self.stats_aggregator,
            "stop_event": self.stop_event,
            "env_config": self.env_config,
            "mcts_config": self.mcts_config,
            "device": self.device,
            "max_game_steps": (
                self.max_game_steps if self.max_game_steps != float("inf") else None
            ),
        }

    def get_current_render_data(self) -> Dict[str, Any]:
        """Returns a dictionary containing state dict reference and last stats (thread-safe)."""
        with self._current_game_state_lock:
            state_dict_ref = self._current_render_state_dict
            stats_copy = self._last_stats.copy()
        return {"state_dict": state_dict_ref, "stats": stats_copy}

    def _update_render_state(self, game_state: GameState, stats: Dict[str, Any]):
        """Updates the state dictionary reference and stats exposed for rendering (thread-safe)."""
        with self._current_game_state_lock:
            try:
                self._current_render_state_dict = game_state.get_state()
            except Exception as e:
                logger.error(f"{self.log_prefix} Error getting game state dict: {e}")
                self._current_render_state_dict = None

            # Extract serializable shape data for rendering
            available_shapes_data = []
            for shape_obj in game_state.shapes:
                if shape_obj:
                    available_shapes_data.append(
                        {"triangles": shape_obj.triangles, "color": shape_obj.color}
                    )
                else:
                    available_shapes_data.append(None)

            ui_stats = {
                "status": stats.get(
                    "status", self._last_stats.get("status", "Unknown")
                ),
                "game_steps": stats.get(
                    "game_steps", self._last_stats.get("game_steps", 0)
                ),
                "game_score": game_state.game_score,  # Explicitly include current game score
                "mcts_sim_time": stats.get(
                    "mcts_total_duration", self._last_stats.get("mcts_sim_time", 0.0)
                ),
                "mcts_nn_time": stats.get(
                    "total_nn_prediction_time",
                    self._last_stats.get("mcts_nn_time", 0.0),
                ),
                "mcts_nodes_explored": stats.get(
                    "nodes_created", self._last_stats.get("mcts_nodes_explored", 0)
                ),
                "mcts_avg_depth": stats.get(
                    "avg_leaf_depth", self._last_stats.get("mcts_avg_depth", 0.0)
                ),
                "available_shapes_data": available_shapes_data,
            }
            self._last_stats.update(ui_stats)

    def _get_temperature(self, game_step: int) -> float:
        """Calculates the MCTS temperature based on the game step."""
        if game_step < self.mcts_config.TEMPERATURE_ANNEAL_STEPS:
            progress = game_step / max(1, self.mcts_config.TEMPERATURE_ANNEAL_STEPS)
            return (
                self.mcts_config.TEMPERATURE_INITIAL * (1 - progress)
                + self.mcts_config.TEMPERATURE_FINAL * progress
            )
        return self.mcts_config.TEMPERATURE_FINAL

    def _play_one_game(self) -> Optional[ProcessedExperienceBatch]:
        """Plays a single game and returns the processed experience."""
        current_game_num = self.stats_aggregator.storage.total_episodes + 1
        logger.info(f"{self.log_prefix} Starting game {current_game_num}")
        start_time = time.monotonic()
        game_data: List[ExperienceTuple] = []
        game = GameState()
        current_state_features = game.reset()
        game_steps = 0
        self.last_intermediate_stats_time = time.monotonic()

        self._update_render_state(game, {"status": "Starting", "game_steps": 0})

        recording_step = {
            "current_self_play_game_number": current_game_num,
            "current_self_play_game_steps": 0,
            "buffer_size": self.experience_queue.qsize(),
        }
        self.stats_aggregator.record_step(recording_step)
        logger.debug(
            f"{self.log_prefix} Game {current_game_num} started. Buffer size: {recording_step['buffer_size']}"
        )

        while not game.is_over() and game_steps < self.max_game_steps:
            if self.stop_event.is_set():
                logger.info(
                    f"{self.log_prefix} Stop event set during game {current_game_num}. Aborting."
                )
                self._update_render_state(
                    game, {"status": "Stopped", "game_steps": game_steps}
                )
                return None

            current_time = time.monotonic()
            if (
                current_time - self.last_intermediate_stats_time
                > self.INTERMEDIATE_STATS_INTERVAL_SEC
            ):
                self._update_render_state(
                    game, {"status": "Running", "game_steps": game_steps}
                )
                recording_step = {
                    "current_self_play_game_number": current_game_num,
                    "current_self_play_game_steps": game_steps,
                    "buffer_size": self.experience_queue.qsize(),
                }
                self.stats_aggregator.record_step(recording_step)
                self.last_intermediate_stats_time = current_time

            mcts_start_time = time.monotonic()
            self.agent.eval()
            root_node, mcts_stats = self.mcts.run_simulations(
                root_state=game, num_simulations=self.mcts_config.NUM_SIMULATIONS
            )
            mcts_duration = time.monotonic() - mcts_start_time
            mcts_stats["game_steps"] = game_steps
            mcts_stats["status"] = "Running (MCTS)"
            logger.debug(
                f"{self.log_prefix} Game {current_game_num} Step {game_steps}: MCTS took {mcts_duration:.4f}s"
            )

            self._update_render_state(game, mcts_stats)

            temperature = self._get_temperature(game_steps)
            policy_target = self.mcts.get_policy_target(root_node, temperature)
            game_data.append((current_state_features, policy_target, 1))

            action = self.mcts.choose_action(root_node, temperature)
            if action == -1:
                logger.error(
                    f"{self.log_prefix} MCTS failed to choose an action. Aborting game."
                )
                game.game_over = True
                break

            step_start_time = time.monotonic()
            _, done = game.step(action)
            step_duration = time.monotonic() - step_start_time
            logger.debug(
                f"{self.log_prefix} Game {current_game_num} Step {game_steps}: Game step took {step_duration:.4f}s"
            )

            current_state_features = game.get_state()
            game_steps += 1

            step_stats_for_aggregator = {
                "mcts_sim_time": mcts_stats.get("mcts_total_duration", 0.0),
                "mcts_nn_time": mcts_stats.get("total_nn_prediction_time", 0.0),
                "mcts_nodes_explored": mcts_stats.get("nodes_created", 0),
                "mcts_avg_depth": mcts_stats.get("avg_leaf_depth", 0.0),
                "buffer_size": self.experience_queue.qsize(),
            }
            self.stats_aggregator.record_step(step_stats_for_aggregator)

        status = (
            "Finished (Max Steps)"
            if game_steps >= self.max_game_steps and not game.is_over()
            else "Finished"
        )
        self._update_render_state(game, {"status": status, "game_steps": game_steps})

        if self.stop_event.is_set():
            logger.info(
                f"{self.log_prefix} Stop event set after game {current_game_num} finished. Not processing."
            )
            return None

        final_outcome = game.get_outcome()
        processed_data: ProcessedExperienceBatch = [
            (state, policy, final_outcome * player)
            for state, policy, player in game_data
        ]

        game_duration = time.monotonic() - start_time
        logger.info(
            f"{self.log_prefix} Game {current_game_num} finished in {game_duration:.2f}s "
            f"({game_steps} steps). Outcome: {final_outcome}, Score: {game.game_score}. "
            f"Queueing {len(processed_data)} experiences."
        )

        current_global_step = self.stats_aggregator.storage.current_global_step
        final_state_for_best = game.get_state()
        self.stats_aggregator.record_episode(
            episode_outcome=final_outcome,
            episode_length=game_steps,
            episode_num=current_game_num,
            global_step=current_global_step,
            game_score=game.game_score,
            triangles_cleared=game.triangles_cleared_this_episode,
            game_state_for_best=final_state_for_best,
        )
        return processed_data

    def run(self):
        """Main loop for the self-play worker."""
        logger.info(f"{self.log_prefix} Starting run loop.")
        while not self.stop_event.is_set():
            try:
                processed_data = self._play_one_game()
                if processed_data is None:
                    if self.stop_event.is_set():
                        break
                    else:
                        logger.warning(
                            f"{self.log_prefix} Game play returned None without stop signal. Continuing."
                        )
                        time.sleep(0.5)
                        continue

                if processed_data:
                    try:
                        q_put_start = time.monotonic()
                        self.experience_queue.put(processed_data, timeout=5.0)
                        q_put_duration = time.monotonic() - q_put_start
                        logger.debug(
                            f"{self.log_prefix} Added game data to queue (qsize: {self.experience_queue.qsize()}) in {q_put_duration:.4f}s."
                        )
                    except queue.Full:
                        logger.warning(
                            f"{self.log_prefix} Experience queue full after waiting. Discarding game data."
                        )
                        time.sleep(0.1)
                    except Exception as q_err:
                        logger.error(
                            f"{self.log_prefix} Error putting data in queue: {q_err}"
                        )
            except Exception as e:
                logger.critical(
                    f"{self.log_prefix} CRITICAL ERROR in run loop: {e}", exc_info=True
                )
                with self._current_game_state_lock:
                    self._current_render_state_dict = None
                    self._last_stats = {
                        "status": "Error",
                        "game_steps": 0,
                        "game_score": 0,
                        "available_shapes_data": [],
                    }
                time.sleep(5.0)

        logger.info(f"{self.log_prefix} Run loop finished.")
        with self._current_game_state_lock:
            self._current_render_state_dict = None
            self._last_stats = {
                "status": "Stopped",
                "game_steps": 0,
                "game_score": 0,
                "available_shapes_data": [],
            }


File: workers\training_worker.py
# File: workers/training_worker.py
import threading
import time
import queue
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
import logging

from config import TrainConfig
from utils.types import StateType, ActionType

if TYPE_CHECKING:
    from agent.alphazero_net import AlphaZeroNet
    from stats.aggregator import StatsAggregator
    from torch.optim.lr_scheduler import _LRScheduler  # Import for type hint

ExperienceData = Tuple[StateType, Dict[ActionType, float], float]
ProcessedExperienceBatch = List[ExperienceData]
logger = logging.getLogger(__name__)


class TrainingWorker(threading.Thread):
    """Samples experience and trains the neural network."""

    def __init__(
        self,
        agent: "AlphaZeroNet",
        optimizer: optim.Optimizer,
        scheduler: Optional["_LRScheduler"],  # Add scheduler parameter
        experience_queue: queue.Queue,
        stats_aggregator: "StatsAggregator",
        stop_event: threading.Event,
        train_config: TrainConfig,
        device: torch.device,
    ):
        super().__init__(daemon=True, name="TrainingWorker")
        self.agent = agent
        self.optimizer = optimizer
        self.scheduler = scheduler  # Store scheduler
        self.experience_queue = experience_queue
        self.stats_aggregator = stats_aggregator
        self.stop_event = stop_event
        self.train_config = train_config
        self.device = device
        self.log_prefix = "[TrainingWorker]"
        # Initialize steps_done from aggregator *after* potential checkpoint load
        self.steps_done = self.stats_aggregator.storage.current_global_step
        logger.info(
            f"{self.log_prefix} Initialized. Device: {self.device}. Initial Step: {self.steps_done}"
        )
        logger.info(
            f"{self.log_prefix} Config: Batch={self.train_config.BATCH_SIZE}, LR={self.train_config.LEARNING_RATE}, MinBuffer={self.train_config.MIN_BUFFER_SIZE_TO_TRAIN}"
        )
        if self.scheduler:
            logger.info(
                f"{self.log_prefix} LR Scheduler Type: {type(self.scheduler).__name__}"
            )
        else:
            logger.info(f"{self.log_prefix} LR Scheduler: DISABLED")

    def get_init_args(self) -> Dict[str, Any]:
        """Returns arguments needed to re-initialize the thread."""
        return {
            "agent": self.agent,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,  # Include scheduler
            "experience_queue": self.experience_queue,
            "stats_aggregator": self.stats_aggregator,
            "stop_event": self.stop_event,
            "train_config": self.train_config,
            "device": self.device,
        }

    def _prepare_batch(
        self, batch_data: ProcessedExperienceBatch
    ) -> Optional[Tuple[StateType, torch.Tensor, torch.Tensor]]:
        """Converts a list of experience tuples into batched tensors."""
        try:
            if (
                not batch_data
                or not isinstance(batch_data[0], tuple)
                or len(batch_data[0]) != 3
            ):
                logger.error(f"{self.log_prefix} Invalid batch data structure (outer).")
                return None
            if not isinstance(batch_data[0][0], dict):
                logger.error(
                    f"{self.log_prefix} Invalid batch data structure (state dict)."
                )
                return None

            states = {key: [] for key in batch_data[0][0].keys()}
            policy_targets, value_targets = [], []
            valid_items = 0

            for item in batch_data:
                if not isinstance(item, tuple) or len(item) != 3:
                    logger.warning(
                        f"{self.log_prefix} Skipping invalid item in batch (wrong structure)."
                    )
                    continue
                state_dict, policy_dict, outcome = item
                if not isinstance(state_dict, dict) or not isinstance(
                    policy_dict, dict
                ):
                    logger.warning(
                        f"{self.log_prefix} Skipping invalid item in batch (wrong inner types)."
                    )
                    continue
                if not (isinstance(outcome, (float, int)) and np.isfinite(outcome)):
                    logger.warning(
                        f"{self.log_prefix} Skipping invalid item in batch (invalid outcome: {outcome})."
                    )
                    continue

                temp_state, valid_state = {}, True
                for key, value in state_dict.items():
                    if key in states and isinstance(value, np.ndarray):
                        temp_state[key] = value
                    else:
                        logger.warning(
                            f"{self.log_prefix} Skipping invalid item in batch (invalid state key/value: {key}, type: {type(value)})."
                        )
                        valid_state = False
                        break
                if not valid_state:
                    continue

                policy_array = np.zeros(self.agent.env_cfg.ACTION_DIM, dtype=np.float32)
                policy_sum = 0.0
                valid_policy_entries = 0
                for action, prob in policy_dict.items():
                    if (
                        isinstance(action, int)
                        and 0 <= action < self.agent.env_cfg.ACTION_DIM
                        and isinstance(prob, (float, int))
                        and np.isfinite(prob)
                        and prob >= 0
                    ):
                        policy_array[action] = prob
                        policy_sum += prob
                        valid_policy_entries += 1

                if valid_policy_entries > 0 and not np.isclose(
                    policy_sum, 1.0, atol=1e-4
                ):
                    logger.warning(
                        f"{self.log_prefix} Policy target sum is {policy_sum:.4f}, expected ~1.0. Using as is."
                    )

                for key in states.keys():
                    states[key].append(temp_state[key])
                policy_targets.append(policy_array)
                value_targets.append(outcome)
                valid_items += 1

            if valid_items == 0:
                logger.error(f"{self.log_prefix} No valid items found in the batch.")
                return None

            batched_states = {
                k: torch.from_numpy(np.stack(v)).to(self.device)
                for k, v in states.items()
            }
            batched_policy = torch.from_numpy(np.stack(policy_targets)).to(self.device)
            batched_value = (
                torch.tensor(value_targets, dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device)
            )

            if (
                batched_policy.shape[0] != valid_items
                or batched_value.shape[0] != valid_items
            ):
                logger.error(
                    f"{self.log_prefix} Shape mismatch after stacking tensors."
                )
                return None

            return batched_states, batched_policy, batched_value

        except Exception as e:
            logger.error(f"{self.log_prefix} Error preparing batch: {e}", exc_info=True)
            return None

    def _perform_training_step(
        self, batch_data: ProcessedExperienceBatch
    ) -> Optional[Dict[str, float]]:
        """Performs a single training step."""
        prep_start = time.monotonic()
        prepared_batch = self._prepare_batch(batch_data)
        prep_duration = time.monotonic() - prep_start
        if prepared_batch is None:
            logger.warning(
                f"{self.log_prefix} Failed to prepare batch (took {prep_duration:.4f}s). Skipping step."
            )
            return None
        batch_states, batch_policy_targets, batch_value_targets = prepared_batch
        logger.info(  # Changed to debug
            f"{self.log_prefix} Batch preparation took {prep_duration:.4f}s."
        )

        try:
            step_start_time = time.monotonic()
            self.agent.train()
            self.optimizer.zero_grad()

            policy_logits, value_preds = self.agent(batch_states)

            if (
                policy_logits.shape[0] != batch_policy_targets.shape[0]
                or value_preds.shape[0] != batch_value_targets.shape[0]
            ):
                logger.error(
                    f"{self.log_prefix} Batch size mismatch after forward pass! Skipping. "
                    f"Logits: {policy_logits.shape}, Targets: {batch_policy_targets.shape}, "
                    f"Values: {value_preds.shape}, Targets: {batch_value_targets.shape}"
                )
                return None
            if policy_logits.shape[1] != batch_policy_targets.shape[1]:
                logger.error(
                    f"{self.log_prefix} Action dim mismatch after forward pass! Skipping. "
                    f"Logits: {policy_logits.shape[1]}, Targets: {batch_policy_targets.shape[1]}"
                )
                return None

            # Policy Loss: Cross-entropy between predicted policy logits and MCTS policy target
            log_policy_preds = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(
                batch_policy_targets * log_policy_preds, dim=1
            ).mean()

            # Value Loss: Mean Squared Error between predicted value and game outcome
            value_loss = F.mse_loss(value_preds, batch_value_targets)

            # Total Loss
            total_loss = (
                self.train_config.POLICY_LOSS_WEIGHT * policy_loss
                + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            )

            total_loss.backward()
            self.optimizer.step()

            # Step the scheduler if it exists
            if self.scheduler:
                self.scheduler.step()

            step_duration = time.monotonic() - step_start_time
            logger.info(  # Changed to debug
                f"{self.log_prefix} Training step took {step_duration:.4f}s."
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            return {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "update_time": step_duration,
                "lr": current_lr,  # Include learning rate
            }
        except Exception as e:
            logger.critical(
                f"{self.log_prefix} CRITICAL ERROR during training step {self.steps_done}: {e}",
                exc_info=True,
            )
            return None

    def run(self):
        """Main training loop."""
        logger.info(f"{self.log_prefix} Starting run loop.")
        # Ensure steps_done is correctly initialized *after* potential checkpoint load
        self.steps_done = self.stats_aggregator.storage.current_global_step
        logger.info(
            f"{self.log_prefix} Starting training from Global Step: {self.steps_done}"
        )

        last_buffer_update_time = 0
        buffer_update_interval = 1.0  # Log buffer size roughly every second if waiting

        while not self.stop_event.is_set():
            buffer_size = self.experience_queue.qsize()

            # Wait if buffer is not large enough
            if buffer_size < self.train_config.MIN_BUFFER_SIZE_TO_TRAIN:
                if time.time() - last_buffer_update_time > buffer_update_interval:
                    # Record buffer size periodically while waiting
                    self.stats_aggregator.record_step({"buffer_size": buffer_size})
                    last_buffer_update_time = time.time()
                    logger.info(
                        f"{self.log_prefix} Waiting for buffer... Size: {buffer_size}/{self.train_config.MIN_BUFFER_SIZE_TO_TRAIN}"
                    )
                time.sleep(0.1)  # Short sleep while waiting
                continue

            # Buffer is ready, start training iteration
            logger.info(  # Changed to debug
                f"{self.log_prefix} Starting training iteration. Buffer size: {buffer_size}"
            )
            steps_this_iter, iter_policy_loss, iter_value_loss = 0, 0.0, 0.0
            iter_start_time = time.monotonic()

            for _ in range(self.train_config.NUM_TRAINING_STEPS_PER_ITER):
                if self.stop_event.is_set():
                    break  # Exit inner loop if stop event is set

                batch_data_list: Optional[ProcessedExperienceBatch] = None
                try:
                    q_get_start = time.monotonic()
                    # Get a batch of experience from the queue
                    batch_data_list = self.experience_queue.get(timeout=1.0)
                    q_get_duration = time.monotonic() - q_get_start
                    logger.info(  # Changed to debug
                        f"{self.log_prefix} Queue get (batch size {len(batch_data_list) if batch_data_list else 0}) took {q_get_duration:.4f}s."
                    )
                except queue.Empty:
                    logger.warning(
                        f"{self.log_prefix} Queue empty during training iteration, waiting..."
                    )
                    time.sleep(0.1)
                    break  # Exit inner loop, will check buffer size again
                except Exception as e:
                    logger.error(
                        f"{self.log_prefix} Error getting data from queue: {e}",
                        exc_info=True,
                    )
                    break  # Exit inner loop on error

                if not batch_data_list:
                    continue  # Skip if queue returned None or empty list

                # Use the actual batch size received, up to the configured max
                actual_batch_size = min(
                    len(batch_data_list), self.train_config.BATCH_SIZE
                )
                if actual_batch_size < 1:  # Should not happen if queue get succeeded
                    continue

                # Perform one training step
                step_result = self._perform_training_step(
                    batch_data_list[:actual_batch_size]  # Slice to actual batch size
                )
                if step_result is None:
                    logger.warning(
                        f"{self.log_prefix} Training step failed, ending iteration early."
                    )
                    break  # Exit inner loop if step failed

                # Update counters and aggregate losses
                self.steps_done += 1
                steps_this_iter += 1
                iter_policy_loss += step_result["policy_loss"]
                iter_value_loss += step_result["value_loss"]

                # Record step statistics
                step_stats = {
                    "global_step": self.steps_done,  # Pass the incremented step
                    "buffer_size": self.experience_queue.qsize(),  # Get current size
                    "training_steps_performed": self.steps_done,  # Track total steps
                    **step_result,  # Include losses, time, lr from step result
                }
                self.stats_aggregator.record_step(step_stats)

            # Log iteration summary
            iter_duration = time.monotonic() - iter_start_time
            if steps_this_iter > 0:
                avg_p = iter_policy_loss / steps_this_iter
                avg_v = iter_value_loss / steps_this_iter
                logger.info(
                    f"{self.log_prefix} Iteration complete. Steps: {steps_this_iter}, "
                    f"Duration: {iter_duration:.2f}s, Avg P.Loss: {avg_p:.4f}, Avg V.Loss: {avg_v:.4f}"
                )
            else:
                logger.info(
                    f"{self.log_prefix} Iteration finished with 0 steps performed (Duration: {iter_duration:.2f}s)."
                )

            # Small sleep to yield control if needed
            time.sleep(0.01)

        logger.info(f"{self.log_prefix} Run loop finished.")


File: workers\__init__.py
# File: workers/__init__.py
# This file makes the 'workers' directory a Python package.

from .self_play_worker import SelfPlayWorker
from .training_worker import TrainingWorker  # Added TrainingWorker

__all__ = ["SelfPlayWorker", "TrainingWorker"]


