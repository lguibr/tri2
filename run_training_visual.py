# File: run_training_visual.py
import sys
import os
import logging
import traceback
import threading
import queue
import time
import pygame
import torch
import mlflow
import ray
from typing import Optional, Dict, Any, List, TextIO

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Updated imports
from src import config, utils, nn, data
from src.rl import TrainingOrchestrator, ExperienceBuffer, Trainer
from src.config import MCTSConfig  # Import Pydantic MCTSConfig from central location
from src import visualization
from src import environment
from src.stats import StatsCollectorActor

# --- Configuration ---
# CHANGE 1: Set log level back to INFO
LOG_LEVEL = logging.INFO
# Keep logger name retrieval before basicConfig
logger = logging.getLogger(__name__)  # Get logger instance

# Basic config setup - will be refined after config objects are loaded
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # Start with stdout
    force=True,  # Override existing handlers
)
# Initial log message
logger.info(f"Set main process root logger level to {logging.getLevelName(LOG_LEVEL)}")


# --- Tee Class for redirecting stdout/stderr ---
class Tee:
    """Helper class to duplicate stream output to multiple targets."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, message: str):
        for stream in self.streams:
            try:
                stream.write(message)
            except Exception as e:
                # Avoid errors within the Tee itself causing loops
                print(
                    f"Tee Error writing to stream {stream}: {e}", file=sys.__stderr__
                )  # Use original stderr
        self.flush()  # Flush after each write

    def flush(self):
        for stream in self.streams:
            try:
                if hasattr(stream, "flush"):
                    stream.flush()
            except Exception as e:
                print(f"Tee Error flushing stream {stream}: {e}", file=sys.__stderr__)

    def isatty(self) -> bool:
        # Return True if any underlying stream is a TTY (e.g., console)
        # This helps libraries that check isatty() for color output etc.
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


# --- End Tee Class ---


visual_state_queue: queue.Queue[Optional[Dict[int, Any]]] = queue.Queue(maxsize=5)


def training_thread_func(orchestrator: TrainingOrchestrator):
    """Function to run the training orchestrator in a separate thread."""
    try:
        logger.info("Training thread started.")
        orchestrator.run_training_loop()
        logger.info("Training thread finished.")
    except Exception as e:
        logger.critical(f"Error in training thread: {e}", exc_info=True)
        if orchestrator:
            orchestrator.training_exception = e
    finally:
        try:
            while not visual_state_queue.empty():
                try:
                    visual_state_queue.get_nowait()
                except queue.Empty:
                    break
            visual_state_queue.put(None, timeout=1.0)
        except queue.Full:
            logger.error(
                "Visual queue still full after clearing attempt during shutdown."
            )
        except Exception as e_q:
            logger.error(
                f"Error putting None signal into visual queue during thread shutdown: {e_q}"
            )


if __name__ == "__main__":
    main_thread_exception = None
    train_thread = None
    orchestrator = None
    mlflow_run_active = False
    ray_initialized = False
    error_logged_in_except_block = False
    stats_collector_actor = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    file_handler = None  # Initialize file_handler

    try:
        # --- Initialize Ray ---
        # Keep log_to_driver=True so worker logs also appear in the main log file
        ray.init(logging_level=logging.WARNING, log_to_driver=True)
        ray_initialized = True
        logger.info(f"Ray initialized. Cluster resources: {ray.cluster_resources()}")

        # --- Initialize Configurations ---
        train_config = config.TrainConfig()
        env_config = config.EnvConfig()
        model_config = config.ModelConfig()
        mcts_config = MCTSConfig()
        persist_config = config.PersistenceConfig()
        vis_config = config.VisConfig()

        # --- Configuration Overrides ---
        persist_config.RUN_NAME = train_config.RUN_NAME
        # Example: train_config.LOAD_CHECKPOINT_PATH = ".alphatriangle_data/runs/<PREVIOUS_RUN_NAME>/checkpoints/latest.pkl"

        # --- Setup File Logging ---
        run_base_dir = persist_config.get_run_base_dir()
        log_dir = os.path.join(run_base_dir, persist_config.LOG_DIR_NAME)
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{train_config.RUN_NAME}_visual.log")

        # Reconfigure logging to include the file handler
        file_handler = logging.FileHandler(
            log_file_path, mode="w"
        )  # 'w' to overwrite each run
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Get the root logger and add the file handler
        # Keep the existing StreamHandler for console output
        root_logger = logging.getLogger()
        root_logger.setLevel(LOG_LEVEL)  # Ensure root logger level is INFO
        # Add file handler if not already present (force=True might remove it)
        root_logger.addHandler(file_handler)

        logger.info(
            f"Logging {logging.getLevelName(LOG_LEVEL)} and higher messages to: {log_file_path}"
        )

        # --- Redirect stdout/stderr using Tee ---
        # Ensure file_handler.stream is available and open
        if file_handler and hasattr(file_handler, "stream") and file_handler.stream:
            sys.stdout = Tee(original_stdout, file_handler.stream)
            sys.stderr = Tee(original_stderr, file_handler.stream)
            print("--- Stdout/Stderr redirected to console and log file ---")
            logger.info("Stdout/Stderr redirected to console and log file.")
        else:
            logger.error(
                "Could not redirect stdout/stderr: File handler stream not available."
            )
        # --- End Redirection ---

        # --- Setup MLflow Tracking ---
        mlflow_abs_path = persist_config.get_mlflow_abs_path()
        os.makedirs(mlflow_abs_path, exist_ok=True)
        logger.info(f"Ensured MLflow directory exists: {mlflow_abs_path}")
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}")

        experiment_name = config.APP_NAME
        mlflow.set_experiment(experiment_name)
        logger.info(f"Set MLflow experiment to: {experiment_name}")

        config.print_config_info_and_validate(mcts_config)

        # --- Setup ---
        utils.set_random_seeds(train_config.RANDOM_SEED)
        device = utils.get_device(train_config.DEVICE)

        # --- Start MLflow Run ---
        mlflow.start_run(run_name=train_config.RUN_NAME)
        mlflow_run_active = True
        logger.info(f"MLflow Run started (ID: {mlflow.active_run().info.run_id}).")

        # --- Initialize Components ---
        # Change: Increase max_history for StatsCollectorActor
        stats_collector_actor = StatsCollectorActor.remote(max_history=100_000)
        neural_net = nn.NeuralNetwork(model_config, env_config, train_config, device)
        buffer = ExperienceBuffer(train_config)
        trainer = Trainer(neural_net, train_config, env_config)
        data_manager = data.DataManager(persist_config, train_config)

        orchestrator = TrainingOrchestrator(
            nn=neural_net,
            buffer=buffer,
            trainer=trainer,
            data_manager=data_manager,
            stats_collector_actor=stats_collector_actor,
            train_config=train_config,
            env_config=env_config,
            mcts_config=mcts_config,
            persist_config=persist_config,
            visual_state_queue=visual_state_queue,
        )

        # --- Start Training Thread ---
        train_thread = threading.Thread(
            target=training_thread_func, args=(orchestrator,), daemon=True
        )
        train_thread.start()
        logger.info("Training thread launched.")

        # --- Initialize Visualization ---
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode(
            (vis_config.SCREEN_WIDTH, vis_config.SCREEN_HEIGHT), pygame.RESIZABLE
        )
        pygame.display.set_caption(
            f"{config.APP_NAME} - Training Visual Mode ({train_config.RUN_NAME})"
        )
        clock = pygame.time.Clock()
        fonts = visualization.load_fonts()
        game_renderer = visualization.GameRenderer(
            screen, vis_config, env_config, fonts, stats_collector_actor
        )
        current_worker_states: Dict[int, environment.GameState] = {}
        global_stats_for_hud: Dict[str, Any] = {}
        has_received_states = False
        has_received_worker_states = False

        # --- Visualization Loop (Main Thread) ---
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.VIDEORESIZE:
                    try:
                        w, h = max(640, event.w), max(480, event.h)
                        screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                        game_renderer.screen = screen
                        game_renderer.layout_rects = None
                    except pygame.error as e:
                        logger.error(f"Error resizing window: {e}")

            # --- Process Visual Queue ---
            try:
                new_state_obj = visual_state_queue.get(timeout=0.05)
                if new_state_obj is None:
                    if train_thread and not train_thread.is_alive():
                        running = False
                        logger.info("Received exit signal from training thread.")
                elif isinstance(new_state_obj, dict):
                    has_received_states = True
                    global_stats_for_hud = new_state_obj.pop(-1, {})
                    worker_states_in_dict = {
                        k: v
                        for k, v in new_state_obj.items()
                        if isinstance(k, int)
                        and k >= 0
                        and isinstance(v, environment.GameState)
                    }
                    if worker_states_in_dict:
                        has_received_worker_states = True
                        current_worker_states = worker_states_in_dict
                else:
                    logger.warning(
                        f"Received unexpected item from visual queue: {type(new_state_obj)}"
                    )
            except queue.Empty:
                pass
            except Exception as q_get_err:
                logger.error(f"Error getting from visual queue: {q_get_err}")
                time.sleep(0.1)

            # --- Rendering Logic ---
            screen.fill(visualization.colors.DARK_GRAY)

            if has_received_states:
                try:
                    game_renderer.render(current_worker_states, global_stats_for_hud)
                except Exception as render_err:
                    logger.error(f"Error during rendering: {render_err}", exc_info=True)
                    err_font = fonts.get("help")
                    if err_font:
                        err_surf = err_font.render(
                            f"Render Error: {render_err}",
                            True,
                            visualization.colors.RED,
                        )
                        screen.blit(err_surf, (10, screen.get_height() // 2))

                if not has_received_worker_states:
                    if fonts.get("help"):
                        wait_font = fonts["help"]
                        wait_surf = wait_font.render(
                            "Waiting for worker states...",
                            True,
                            visualization.colors.LIGHT_GRAY,
                        )
                        layout_rects = game_renderer._calculate_layout()
                        worker_area = (
                            layout_rects.get("worker_grid")
                            if layout_rects
                            else screen.get_rect()
                        )
                        wait_rect = wait_surf.get_rect(
                            centerx=worker_area.centerx, top=worker_area.top + 20
                        )
                        screen.blit(wait_surf, wait_rect)

            else:
                if fonts.get("help"):
                    wait_font = fonts["help"]
                    wait_surf = wait_font.render(
                        "Waiting for first game states from workers...",
                        True,
                        visualization.colors.LIGHT_GRAY,
                    )
                    wait_rect = wait_surf.get_rect(
                        center=(screen.get_width() // 2, screen.get_height() // 2)
                    )
                    screen.blit(wait_surf, wait_rect)

            pygame.display.flip()

            # --- Check Training Thread Status ---
            if train_thread and not train_thread.is_alive() and running:
                logger.warning("Training thread terminated.")
                if orchestrator and orchestrator.training_exception:
                    logger.error(
                        f"Training thread terminated due to exception: {orchestrator.training_exception}"
                    )
                    main_thread_exception = orchestrator.training_exception
                running = False

            clock.tick(vis_config.FPS)

    except Exception as e:
        logger.critical(
            f"An unhandled error occurred in visual training script (main thread): {e}"
        )
        traceback.print_exc()  # This will now go to console AND log file
        main_thread_exception = e
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "FAILED")
                mlflow.log_param("error_message", f"MainThread: {str(e)}")
                error_logged_in_except_block = True
            except Exception as mlf_err:
                logger.error(f"Failed to log main thread error to MLflow: {mlf_err}")

    finally:
        # --- Restore stdout/stderr ---
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print("--- Restored stdout/stderr ---")
        # --- End Restore ---

        logger.info("Initiating shutdown sequence...")
        if orchestrator:
            orchestrator.request_stop()

        if train_thread and train_thread.is_alive():
            logger.info("Waiting for training thread to join...")
            train_thread.join(timeout=15.0)
            if train_thread.is_alive():
                logger.error("Training thread did not exit gracefully within timeout.")

        if orchestrator:
            logger.info("Attempting to save final training state...")
            orchestrator.save_final_state()
            orchestrator._final_cleanup()

        final_status = "INTERRUPTED"
        final_error = None
        exit_code = 1

        if main_thread_exception:
            final_status = "FAILED"
            final_error = main_thread_exception
        elif orchestrator and orchestrator.training_exception:
            final_status = "FAILED"
            final_error = orchestrator.training_exception
        elif orchestrator and orchestrator.training_complete:
            final_status = "COMPLETED"
            exit_code = 0

        logger.info(f"Final Training Status: {final_status}")

        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", final_status)
                if final_error and not error_logged_in_except_block:
                    mlflow.log_param("error_message", str(final_error))
                mlflow.end_run()
                logger.info("MLflow Run ended.")
            except Exception as mlf_end_err:
                logger.error(f"Error ending MLflow run: {mlf_end_err}")

        if ray_initialized:
            ray.shutdown()
            logger.info("Ray shut down.")

        pygame.quit()
        logger.info("Visual training script finished.")
        # Close the file handler explicitly
        if file_handler:
            try:
                # Ensure all buffered output is written
                file_handler.flush()
                file_handler.close()
                # Remove handler AFTER closing stream
                root_logger.removeHandler(file_handler)
            except Exception as e_close:
                # Use original stderr for this final print
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        sys.exit(exit_code)
