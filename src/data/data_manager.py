# File: src/data/data_manager.py
import os
import shutil
import logging
import glob
import cloudpickle
import torch
import ray
import re
import json
import mlflow
import numpy as np  # Import numpy
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any, List
from collections import deque
from pydantic import ValidationError

# Import Pydantic models and Experience type
from .schemas import CheckpointData, BufferData, LoadedTrainingState
from src.utils.types import Experience

if TYPE_CHECKING:
    from src.nn import NeuralNetwork
    from src.rl.core.buffer import ExperienceBuffer  # Import buffer type
    from src.config import PersistenceConfig, TrainConfig, MCTSConfig
    from src.stats import StatsCollectorActor
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages loading and saving of training artifacts using Pydantic schemas
    and cloudpickle for serialization. Handles MLflow artifact logging.
    """

    def __init__(
        self, persist_config: "PersistenceConfig", train_config: "TrainConfig"
    ):
        self.persist_config = persist_config
        self.train_config = train_config
        if (
            not self.persist_config.RUN_NAME
            or self.persist_config.RUN_NAME == "default_run"
        ):
            logger.warning("DataManager RUN_NAME not set. Using default/current value.")
        os.makedirs(self.persist_config.ROOT_DATA_DIR, exist_ok=True)
        self._update_paths()
        self._create_directories()
        logger.info(
            f"DataManager initialized. Current Run Name: {self.persist_config.RUN_NAME}. Run directory: {self.run_base_dir}"
        )

    def _update_paths(self):
        """Updates paths based on the current RUN_NAME."""
        self.run_base_dir = self.persist_config.get_run_base_dir()
        self.checkpoint_dir = os.path.join(
            self.run_base_dir, self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        )
        self.buffer_dir = os.path.join(
            self.run_base_dir, self.persist_config.BUFFER_SAVE_DIR_NAME
        )
        self.log_dir = os.path.join(self.run_base_dir, self.persist_config.LOG_DIR_NAME)
        self.config_path = os.path.join(
            self.run_base_dir, self.persist_config.CONFIG_FILENAME
        )

    def _create_directories(self):
        """Creates necessary temporary directories for the current run."""
        os.makedirs(self.run_base_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if self.persist_config.SAVE_BUFFER:
            os.makedirs(self.buffer_dir, exist_ok=True)

    def get_checkpoint_path(
        self,
        run_name: Optional[str] = None,
        step: Optional[int] = None,
        is_latest: bool = False,
        is_best: bool = False,
        is_final: bool = False,
    ) -> str:
        """Constructs the path for a checkpoint file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = self.persist_config.get_run_base_dir(target_run_name)
        checkpoint_dir = os.path.join(
            base_dir, self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        )
        if is_latest:
            filename = self.persist_config.LATEST_CHECKPOINT_FILENAME
        elif is_best:
            filename = self.persist_config.BEST_CHECKPOINT_FILENAME
        elif is_final and step is not None:
            filename = f"checkpoint_final_step_{step}.pkl"
        elif step is not None:
            filename = f"checkpoint_step_{step}.pkl"
        else:
            filename = self.persist_config.LATEST_CHECKPOINT_FILENAME
        base, _ = os.path.splitext(filename)
        filename_pkl = base + ".pkl"
        return os.path.join(checkpoint_dir, filename_pkl)

    def get_buffer_path(
        self,
        run_name: Optional[str] = None,
        step: Optional[int] = None,
        is_final: bool = False,
    ) -> str:
        """Constructs the path for the replay buffer file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = self.persist_config.get_run_base_dir(target_run_name)
        buffer_dir = os.path.join(base_dir, self.persist_config.BUFFER_SAVE_DIR_NAME)
        if is_final and step is not None:
            filename = f"buffer_final_step_{step}.pkl"
        elif step is not None and self.persist_config.BUFFER_SAVE_FREQ_STEPS > 0:
            filename = f"buffer_step_{step}.pkl"
        else:
            filename = self.persist_config.BUFFER_FILENAME
        return os.path.join(buffer_dir, filename)

    def find_latest_run_dir(self, current_run_name: str) -> Optional[str]:
        """Finds the most recent *previous* run directory."""
        runs_root_dir = os.path.join(
            self.persist_config.ROOT_DATA_DIR, self.persist_config.RUNS_DIR_NAME
        )
        run_prefix = "train_"
        try:
            if not os.path.exists(runs_root_dir):
                return None
            potential_dirs = [
                d
                for d in os.listdir(runs_root_dir)
                if os.path.isdir(os.path.join(runs_root_dir, d))
                and d.startswith(run_prefix)
                and d != current_run_name
            ]
            if not potential_dirs:
                return None

            def extract_timestamp(dir_name):
                match = re.search(r"(\d{8}_\d{6})", dir_name)
                return match.group(1) if match else "0"

            potential_dirs.sort(key=extract_timestamp, reverse=True)
            return potential_dirs[0]
        except Exception as e:
            logger.error(f"Error finding latest run directory: {e}", exc_info=True)
            return None

    def _determine_checkpoint_to_load(self) -> Optional[str]:
        """Determines the absolute path of the checkpoint file to load."""
        load_path_config = self.train_config.LOAD_CHECKPOINT_PATH
        auto_resume = self.train_config.AUTO_RESUME_LATEST
        current_run_name = self.persist_config.RUN_NAME
        checkpoint_to_load = None
        if load_path_config and os.path.exists(load_path_config):
            checkpoint_to_load = os.path.abspath(load_path_config)
            logger.info(f"Using specified checkpoint path: {checkpoint_to_load}")
        elif auto_resume:
            latest_run_name = self.find_latest_run_dir(current_run_name)
            if latest_run_name:
                potential_latest_path = self.get_checkpoint_path(
                    run_name=latest_run_name, is_latest=True
                )
                if os.path.exists(potential_latest_path):
                    checkpoint_to_load = os.path.abspath(potential_latest_path)
                    logger.info(
                        f"Auto-resuming from latest checkpoint: {checkpoint_to_load}"
                    )
        if not checkpoint_to_load:
            logger.info("No checkpoint found to load. Starting training from scratch.")
        return checkpoint_to_load

    def _determine_buffer_to_load(
        self, checkpoint_path: Optional[str]
    ) -> Optional[str]:
        """Determines the buffer file path to load."""
        if self.train_config.LOAD_BUFFER_PATH and os.path.exists(
            self.train_config.LOAD_BUFFER_PATH
        ):
            logger.info(
                f"Using specified buffer path: {self.train_config.LOAD_BUFFER_PATH}"
            )
            return os.path.abspath(self.train_config.LOAD_BUFFER_PATH)
        if checkpoint_path:
            try:
                run_name_loaded_from = os.path.basename(
                    os.path.dirname(os.path.dirname(checkpoint_path))
                )
                potential_buffer_path = self.get_buffer_path(
                    run_name=run_name_loaded_from
                )
                if os.path.exists(potential_buffer_path):
                    logger.info(
                        f"Derived buffer path from checkpoint run: {potential_buffer_path}"
                    )
                    return os.path.abspath(potential_buffer_path)
            except Exception as e:
                logger.warning(
                    f"Could not derive run name from checkpoint path {checkpoint_path}: {e}"
                )
        logger.info("No suitable buffer file found to load.")
        return None

    def load_initial_state(self) -> LoadedTrainingState:
        """
        Loads the initial training state using Pydantic models for validation.
        Returns a LoadedTrainingState object containing the deserialized data.
        """
        loaded_state = LoadedTrainingState()  # Initialize empty Pydantic model
        checkpoint_to_load = self._determine_checkpoint_to_load()

        if checkpoint_to_load:
            logger.info(f"Loading checkpoint: {checkpoint_to_load}")
            try:
                with open(checkpoint_to_load, "rb") as f:
                    loaded_checkpoint_model = cloudpickle.load(f)
                if isinstance(loaded_checkpoint_model, CheckpointData):
                    loaded_state.checkpoint_data = loaded_checkpoint_model
                    logger.info(
                        f"Checkpoint loaded and validated (Run: {loaded_state.checkpoint_data.run_name}, Step: {loaded_state.checkpoint_data.global_step})"
                    )
                else:
                    logger.error(
                        f"Loaded checkpoint file {checkpoint_to_load} did not contain a CheckpointData object (type: {type(loaded_checkpoint_model)})."
                    )
                    checkpoint_to_load = None
            except ValidationError as e:
                logger.error(
                    f"Pydantic validation failed for checkpoint {checkpoint_to_load}: {e}",
                    exc_info=True,
                )
                checkpoint_to_load = None
            except Exception as e:
                logger.error(
                    f"Error loading/validating checkpoint from {checkpoint_to_load}: {e}",
                    exc_info=True,
                )
                checkpoint_to_load = None

        if self.persist_config.SAVE_BUFFER:
            buffer_to_load = self._determine_buffer_to_load(checkpoint_to_load)
            if buffer_to_load:
                logger.info(f"Loading buffer: {buffer_to_load}")
                try:
                    with open(buffer_to_load, "rb") as f:
                        loaded_buffer_model = cloudpickle.load(f)
                    if isinstance(loaded_buffer_model, BufferData):
                        loaded_state.buffer_data = loaded_buffer_model
                        logger.info(
                            f"Buffer loaded and validated. Size: {len(loaded_state.buffer_data.buffer_list)}"
                        )
                    else:
                        logger.error(
                            f"Loaded buffer file {buffer_to_load} did not contain a BufferData object (type: {type(loaded_buffer_model)})."
                        )
                except ValidationError as e:
                    logger.error(
                        f"Pydantic validation failed for buffer {buffer_to_load}: {e}",
                        exc_info=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load/validate experience buffer from {buffer_to_load}: {e}",
                        exc_info=True,
                    )

        if not loaded_state.checkpoint_data and not loaded_state.buffer_data:
            logger.info("No checkpoint or buffer loaded. Starting fresh.")

        return loaded_state

    def save_training_state(
        self,
        nn: "NeuralNetwork",
        optimizer: "Optimizer",
        stats_collector_actor: "StatsCollectorActor",
        buffer: "ExperienceBuffer",  # Use the buffer type hint
        global_step: int,
        episodes_played: int,
        total_simulations_run: int,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Saves the training state using Pydantic models and cloudpickle."""
        run_name = self.persist_config.RUN_NAME
        logger.info(
            f"Saving training state for run '{run_name}' at step {global_step}. Final={is_final}, Best={is_best}"
        )

        stats_collector_state = {}
        if stats_collector_actor:
            try:
                stats_state_ref = stats_collector_actor.get_state.remote()
                stats_collector_state = ray.get(stats_state_ref, timeout=5.0)
            except Exception as e:
                logger.error(
                    f"Error fetching state from StatsCollectorActor for saving: {e}",
                    exc_info=True,
                )

        optimizer_state_cpu = {}
        try:
            optimizer_state_dict = optimizer.state_dict()

            def move_to_cpu(item):
                if isinstance(item, torch.Tensor):
                    return item.cpu()
                elif isinstance(item, dict):
                    return {k: move_to_cpu(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [move_to_cpu(elem) for elem in item]
                else:
                    return item

            optimizer_state_cpu = move_to_cpu(optimizer_state_dict)
        except Exception as e:
            logger.error(f"Could not prepare optimizer state for saving: {e}")

        try:
            checkpoint_data = CheckpointData(
                run_name=run_name,
                global_step=global_step,
                episodes_played=episodes_played,
                total_simulations_run=total_simulations_run,
                model_config_dict=nn.model_config.model_dump(),
                env_config_dict=nn.env_config.model_dump(),
                model_state_dict=nn.get_weights(),
                optimizer_state_dict=optimizer_state_cpu,
                stats_collector_state=stats_collector_state,
            )
        except ValidationError as e:
            logger.error(f"Failed to create CheckpointData model: {e}", exc_info=True)
            return

        step_checkpoint_path = self.get_checkpoint_path(
            run_name=run_name, step=global_step, is_final=is_final
        )
        saved_checkpoint_path = None
        try:
            os.makedirs(os.path.dirname(step_checkpoint_path), exist_ok=True)
            with open(step_checkpoint_path, "wb") as f:
                cloudpickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint temporarily saved to {step_checkpoint_path}")
            saved_checkpoint_path = step_checkpoint_path
            latest_path = self.get_checkpoint_path(run_name=run_name, is_latest=True)
            best_path = self.get_checkpoint_path(run_name=run_name, is_best=True)
            try:
                shutil.copy2(step_checkpoint_path, latest_path)
            except Exception as e:
                logger.error(f"Failed to update latest checkpoint link: {e}")
            if is_best:
                try:
                    shutil.copy2(step_checkpoint_path, best_path)
                    logger.info(f"Updated best checkpoint link to step {global_step}")
                except Exception as e:
                    logger.error(f"Failed to update best checkpoint link: {e}")
        except Exception as e:
            logger.error(
                f"Failed to save checkpoint file to {step_checkpoint_path}: {e}",
                exc_info=True,
            )

        saved_buffer_path = None
        if self.persist_config.SAVE_BUFFER:
            buffer_path = self.get_buffer_path(
                run_name=run_name, step=global_step, is_final=is_final
            )
            default_buffer_path = self.get_buffer_path(run_name=run_name)
            try:
                # --- FIX: Access buffer data correctly based on PER ---
                if buffer.use_per:
                    # For PER, save the data stored in the SumTree leaves
                    # Need to handle potential empty slots if buffer not full
                    buffer_list = [
                        buffer.tree.data[i]
                        for i in range(buffer.tree.n_entries)
                        if buffer.tree.data[i] != 0
                    ]
                else:
                    # For uniform buffer, use the deque
                    buffer_list = list(buffer.buffer)
                # --- END FIX ---

                buffer_data = BufferData(buffer_list=buffer_list)
                os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
                with open(buffer_path, "wb") as f:
                    cloudpickle.dump(buffer_data, f)
                logger.info(f"Experience buffer temporarily saved to {buffer_path}")
                saved_buffer_path = buffer_path
                try:
                    with open(default_buffer_path, "wb") as f_default:
                        cloudpickle.dump(buffer_data, f_default)
                    logger.debug(f"Updated default buffer file: {default_buffer_path}")
                except Exception as e_default:
                    logger.error(
                        f"Error updating default buffer file {default_buffer_path}: {e_default}"
                    )
            except ValidationError as e:
                logger.error(f"Failed to create BufferData model: {e}", exc_info=True)
            except Exception as e:
                logger.error(
                    f"Error saving experience buffer to {buffer_path}: {e}",
                    exc_info=True,
                )

        self._log_artifacts(saved_checkpoint_path, saved_buffer_path, run_name, is_best)

    def _log_artifacts(
        self,
        checkpoint_path: Optional[str],
        buffer_path: Optional[str],
        run_name: str,
        is_best: bool,
    ):
        """Logs saved checkpoint and buffer files to MLflow."""
        try:
            if checkpoint_path and os.path.exists(checkpoint_path):
                ckpt_artifact_path = self.persist_config.CHECKPOINT_SAVE_DIR_NAME
                mlflow.log_artifact(checkpoint_path, artifact_path=ckpt_artifact_path)
                latest_path = self.get_checkpoint_path(
                    run_name=run_name, is_latest=True
                )
                if os.path.exists(latest_path):
                    mlflow.log_artifact(latest_path, artifact_path=ckpt_artifact_path)
                if is_best:
                    best_path = self.get_checkpoint_path(
                        run_name=run_name, is_best=True
                    )
                    if os.path.exists(best_path):
                        mlflow.log_artifact(best_path, artifact_path=ckpt_artifact_path)
                logger.info(
                    f"Logged checkpoint artifacts to MLflow path: {ckpt_artifact_path}"
                )
            if buffer_path and os.path.exists(buffer_path):
                buffer_artifact_path = self.persist_config.BUFFER_SAVE_DIR_NAME
                mlflow.log_artifact(buffer_path, artifact_path=buffer_artifact_path)
                default_buffer_path = self.get_buffer_path(run_name=run_name)
                if os.path.exists(default_buffer_path):
                    mlflow.log_artifact(
                        default_buffer_path, artifact_path=buffer_artifact_path
                    )
                logger.info(
                    f"Logged buffer artifacts to MLflow path: {buffer_artifact_path}"
                )
        except Exception as e:
            logger.error(f"Failed to log artifacts to MLflow: {e}", exc_info=True)

    def save_run_config(self, configs: Dict[str, Any]):
        """Saves the combined configuration dictionary as a JSON artifact."""
        try:
            config_path = self.config_path
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:

                def default_serializer(obj):
                    if isinstance(obj, (torch.Tensor, np.ndarray)):
                        return "<tensor/array>"
                    if isinstance(obj, deque):
                        return list(obj)
                    try:
                        return str(obj)
                    except:
                        return "<not serializable>"

                json.dump(configs, f, indent=4, default=default_serializer)
            mlflow.log_artifact(config_path, artifact_path="config")
            logger.info("Logged combined config JSON to MLflow.")
        except Exception as e:
            logger.error(f"Failed to save/log run config JSON: {e}", exc_info=True)
