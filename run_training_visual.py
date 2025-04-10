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
from typing import Optional, Dict, Any, List

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Updated imports
from src import config, utils, nn, data
from src.rl import TrainingOrchestrator, ExperienceBuffer, Trainer
from src.mcts import MCTSConfig # Import Pydantic MCTSConfig
from src import visualization
from src import environment
from src.stats import StatsCollectorActor

# --- Configuration ---
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)
logger.info(f"Set main process root logger level to {logging.getLevelName(LOG_LEVEL)}")

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
                try: visual_state_queue.get_nowait()
                except queue.Empty: break
            visual_state_queue.put(None, timeout=1.0)
        except queue.Full: logger.error("Visual queue still full after clearing attempt during shutdown.")
        except Exception as e_q: logger.error(f"Error putting None signal into visual queue during thread shutdown: {e_q}")


if __name__ == "__main__":
    main_thread_exception = None
    train_thread = None
    orchestrator = None
    mlflow_run_active = False
    ray_initialized = False
    error_logged_in_except_block = False
    stats_collector_actor = None

    try:
        # --- Initialize Ray ---
        ray.init(logging_level=logging.WARNING, log_to_driver=True)
        ray_initialized = True
        logger.info(f"Ray initialized. Cluster resources: {ray.cluster_resources()}")

        # --- Initialize Configurations ---
        train_config = config.TrainConfig()
        env_config = config.EnvConfig()
        model_config = config.ModelConfig()
        mcts_config = MCTSConfig() # Instantiate Pydantic model
        persist_config = config.PersistenceConfig()
        vis_config = config.VisConfig()

        # --- Configuration Overrides ---
        persist_config.RUN_NAME = train_config.RUN_NAME
        # Example: train_config.LOAD_CHECKPOINT_PATH = ".alphatriangle_data/runs/<PREVIOUS_RUN_NAME>/checkpoints/latest.pkl"

        # --- Set MLflow Tracking URI ---
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        os.makedirs(os.path.dirname(mlflow_tracking_uri.replace("file:", "")), exist_ok=True)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}")

        config.print_config_info_and_validate(mcts_config) # Pass Pydantic instance

        # --- Setup ---
        utils.set_random_seeds(train_config.RANDOM_SEED)
        device = utils.get_device(train_config.DEVICE)

        # --- Start MLflow Run ---
        mlflow.start_run(run_name=train_config.RUN_NAME)
        mlflow_run_active = True
        logger.info(f"MLflow Run started (ID: {mlflow.active_run().info.run_id}).")

        # --- Initialize Components ---
        stats_collector_actor = StatsCollectorActor.remote(max_history=1000)
        neural_net = nn.NeuralNetwork(model_config, env_config, train_config, device)
        buffer = ExperienceBuffer(train_config)
        trainer = Trainer(neural_net, train_config, env_config)
        # Pass train_config to DataManager for auto-resume logic
        data_manager = data.DataManager(persist_config, train_config)

        # Initialize Orchestrator (it will handle loading state internally)
        orchestrator = TrainingOrchestrator(
            nn=neural_net,
            buffer=buffer,
            trainer=trainer,
            data_manager=data_manager,
            stats_collector_actor=stats_collector_actor,
            train_config=train_config,
            env_config=env_config,
            mcts_config=mcts_config, # Pass Pydantic instance
            persist_config=persist_config,
            visual_state_queue=visual_state_queue,
        )

        # --- Start Training Thread ---
        train_thread = threading.Thread(target=training_thread_func, args=(orchestrator,), daemon=True)
        train_thread.start()
        logger.info("Training thread launched.")

        # --- Initialize Visualization ---
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((vis_config.SCREEN_WIDTH, vis_config.SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption(f"{config.APP_NAME} - Training Visual Mode ({train_config.RUN_NAME})")
        clock = pygame.time.Clock()
        fonts = visualization.load_fonts()
        # Pass stats_collector_actor handle to GameRenderer
        game_renderer = visualization.GameRenderer(screen, vis_config, env_config, fonts, stats_collector_actor)
        current_worker_states: Dict[int, environment.GameState] = {}
        global_stats_for_hud: Dict[str, Any] = {}
        has_received_states = False # Flag: Have we received *any* dict from the queue?
        has_received_worker_states = False # Flag: Have we received a dict with actual worker states?

        # --- Visualization Loop (Main Thread) ---
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False
                if event.type == pygame.VIDEORESIZE:
                    try:
                        w, h = max(640, event.w), max(480, event.h)
                        screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                        game_renderer.screen = screen
                        game_renderer.layout_rects = None # Force layout recalc
                    except pygame.error as e: logger.error(f"Error resizing window: {e}")

            # --- Process Visual Queue ---
            try:
                new_state_obj = visual_state_queue.get(timeout=0.05)
                if new_state_obj is None:
                    if train_thread and not train_thread.is_alive():
                        running = False
                        logger.info("Received exit signal from training thread.")
                elif isinstance(new_state_obj, dict):
                    has_received_states = True # Received something
                    global_stats_for_hud = new_state_obj.pop(-1, {}) # Extract global stats
                    # Check if any worker states (keys >= 0) are present
                    worker_states_in_dict = {k: v for k, v in new_state_obj.items() if isinstance(k, int) and k >= 0 and isinstance(v, environment.GameState)}
                    if worker_states_in_dict:
                        has_received_worker_states = True # Received actual worker states
                        current_worker_states = worker_states_in_dict
                    # If only global stats were received, current_worker_states remains empty/unchanged
                else:
                    logger.warning(f"Received unexpected item from visual queue: {type(new_state_obj)}")
            except queue.Empty: pass
            except Exception as q_get_err:
                logger.error(f"Error getting from visual queue: {q_get_err}")
                time.sleep(0.1)

            # --- Rendering Logic ---
            screen.fill(visualization.colors.DARK_GRAY)

            if has_received_states:
                # We have received at least global stats, render HUD/Plots/Progress
                try:
                    # Pass current_worker_states (might be empty) and global_stats
                    game_renderer.render(current_worker_states, global_stats_for_hud)
                except Exception as render_err:
                    logger.error(f"Error during rendering: {render_err}", exc_info=True)
                    err_font = fonts.get("help")
                    if err_font:
                        err_surf = err_font.render(f"Render Error: {render_err}", True, visualization.colors.RED)
                        screen.blit(err_surf, (10, screen.get_height() // 2))

                # Show specific message if we only have global stats so far
                if not has_received_worker_states:
                    if fonts.get("help"):
                        wait_font = fonts["help"]
                        wait_surf = wait_font.render("Waiting for worker states...", True, visualization.colors.LIGHT_GRAY)
                        # Position message somewhere sensible, e.g., top-center of worker area
                        layout_rects = game_renderer._calculate_layout()
                        worker_area = layout_rects.get("worker_grid") if layout_rects else screen.get_rect()
                        wait_rect = wait_surf.get_rect(centerx=worker_area.centerx, top=worker_area.top + 20)
                        screen.blit(wait_surf, wait_rect)

            else:
                # Still waiting for the very first dictionary from the queue
                if fonts.get("help"):
                    wait_font = fonts["help"]
                    wait_surf = wait_font.render("Waiting for first game states from workers...", True, visualization.colors.LIGHT_GRAY)
                    wait_rect = wait_surf.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
                    screen.blit(wait_surf, wait_rect)

            pygame.display.flip()

            # --- Check Training Thread Status ---
            if train_thread and not train_thread.is_alive() and running:
                logger.warning("Training thread terminated.")
                if orchestrator and orchestrator.training_exception:
                    logger.error(f"Training thread terminated due to exception: {orchestrator.training_exception}")
                    main_thread_exception = orchestrator.training_exception
                running = False

            clock.tick(vis_config.FPS)

    except Exception as e:
        logger.critical(f"An unhandled error occurred in visual training script (main thread): {e}")
        traceback.print_exc()
        main_thread_exception = e
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "FAILED")
                mlflow.log_param("error_message", f"MainThread: {str(e)}")
                error_logged_in_except_block = True
            except Exception as mlf_err: logger.error(f"Failed to log main thread error to MLflow: {mlf_err}")

    finally:
        logger.info("Initiating shutdown sequence...")
        if orchestrator:
            orchestrator.request_stop()

        if train_thread and train_thread.is_alive():
            logger.info("Waiting for training thread to join...")
            train_thread.join(timeout=15.0)
            if train_thread.is_alive():
                logger.error("Training thread did not exit gracefully within timeout.")

        # --- Force Save Final State (after thread join/timeout) ---
        if orchestrator:
             logger.info("Attempting to save final training state...")
             orchestrator.save_final_state() # Calls DataManager internally
             # --- Perform Orchestrator Cleanup (kills actors) ---
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
        sys.exit(exit_code)