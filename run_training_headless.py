# File: run_training_headless.py
import sys
import os
import logging
import traceback
import torch
import mlflow
import ray

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Updated imports
from src import config, utils, nn, data
from src.rl import TrainingOrchestrator, ExperienceBuffer, Trainer
from src.mcts import MCTSConfig  # Import Pydantic MCTSConfig
from src.stats import StatsCollectorActor

# --- Configuration ---
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    orchestrator = None
    mlflow_run_active = False
    ray_initialized = False
    stats_collector_actor = None

    try:
        # --- Initialize Ray ---
        ray.init(logging_level=logging.WARNING)
        ray_initialized = True
        logger.info(f"Ray initialized. Cluster resources: {ray.cluster_resources()}")

        # --- Initialize Configurations ---
        train_config = config.TrainConfig()
        env_config = config.EnvConfig()
        model_config = config.ModelConfig()
        mcts_config = MCTSConfig()  # Instantiate Pydantic model
        persist_config = config.PersistenceConfig()

        # --- Configuration Overrides ---
        persist_config.RUN_NAME = train_config.RUN_NAME  # Ensure RUN_NAME is consistent
        # Example: train_config.LOAD_CHECKPOINT_PATH = ".alphatriangle_data/runs/<PREVIOUS_RUN_NAME>/checkpoints/latest.pkl"

        # --- Setup MLflow Tracking ---
        # Get the absolute OS path first
        mlflow_abs_path = persist_config.get_mlflow_abs_path()
        # Create the directory using the OS path
        os.makedirs(mlflow_abs_path, exist_ok=True)
        logger.info(f"Ensured MLflow directory exists: {mlflow_abs_path}")
        # Get the correctly formatted file URI
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}")

        # --- Set Experiment (creates if not exists) ---
        experiment_name = config.APP_NAME  # Use app name from config
        mlflow.set_experiment(experiment_name)
        logger.info(f"Set MLflow experiment to: {experiment_name}")

        config.print_config_info_and_validate(mcts_config)  # Pass Pydantic instance

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
            mcts_config=mcts_config,  # Pass Pydantic instance
            persist_config=persist_config,
            visual_state_queue=None,  # No visualization
        )

        # --- Run Training ---
        logger.info("Starting headless training...")
        orchestrator.run_training_loop()

    except Exception as e:
        logger.critical(f"An unhandled error occurred during headless training: {e}")
        traceback.print_exc()
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "FAILED")
                mlflow.log_param("error_message", str(e))
            except Exception as mlf_err:
                logger.error(f"Failed to log error status to MLflow: {mlf_err}")
        sys.exit(1)

    finally:
        final_status = "INTERRUPTED"
        error_msg = ""
        exit_code = 1

        if orchestrator:
            if orchestrator.training_exception:
                final_status = "FAILED"
                error_msg = str(orchestrator.training_exception)
            elif orchestrator.training_complete:
                final_status = "COMPLETED"
                exit_code = 0

            # --- Force Save Final State ---
            logger.info("Attempting to save final training state...")
            orchestrator.save_final_state()  # Calls DataManager internally

            # --- Perform Orchestrator Cleanup (kills actors) ---
            orchestrator._final_cleanup()
        elif "e" in locals() and isinstance(e, Exception):
            final_status = "FAILED"
            error_msg = str(e)

        logger.info(f"Final Training Status: {final_status}")

        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", final_status)
                if error_msg:
                    mlflow.log_param("error_message", error_msg)
                mlflow.end_run()
                logger.info("MLflow Run ended.")
            except Exception as mlf_end_err:
                logger.error(f"Error ending MLflow run: {mlf_end_err}")

        if ray_initialized:
            ray.shutdown()
            logger.info("Ray shut down.")

        logger.info("Headless training script finished.")
        sys.exit(exit_code)
