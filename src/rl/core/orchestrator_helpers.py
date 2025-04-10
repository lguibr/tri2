# File: src/rl/core/orchestrator_helpers.py
import logging
import os
import json
import mlflow
import queue
import numpy as np
import ray
import glob
import re
from collections import deque
from typing import TYPE_CHECKING, Optional, Dict, Any

# --- Package Imports ---
from src.environment import GameState
# Import SelfPlayResult Pydantic model from local rl types
from ..types import SelfPlayResult # Updated import
from src.utils.types import StatsCollectorData

if TYPE_CHECKING:
    # Use TYPE_CHECKING to avoid runtime circular import if helpers need the orchestrator type
    from .orchestrator import TrainingOrchestrator

logger = logging.getLogger(__name__)

# --- MLflow Logging ---

def log_configs_to_mlflow(orchestrator: "TrainingOrchestrator"):
    """Logs configuration parameters and saves config JSON to MLflow."""
    try:
        from src.config import APP_NAME

        mlflow.log_param("APP_NAME", APP_NAME)
        # Use model_dump() for Pydantic models
        mlflow.log_params(orchestrator.train_config.model_dump())
        mlflow.log_params(orchestrator.env_config.model_dump())
        mlflow.log_params(orchestrator.nn.model_config.model_dump())
        mlflow.log_params(orchestrator.mcts_config.model_dump()) # Use model_dump
        # Exclude computed field from persistence config logging if desired
        persist_params = orchestrator.persist_config.model_dump(exclude={'MLFLOW_TRACKING_URI'})
        mlflow.log_params(persist_params)

        logger.info("Logged configuration parameters to MLflow.")

        # Log configs as JSON artifact using DataManager
        all_configs = {
            "train_config": orchestrator.train_config.model_dump(),
            "env_config": orchestrator.env_config.model_dump(),
            "model_config": orchestrator.nn.model_config.model_dump(),
            "mcts_config": orchestrator.mcts_config.model_dump(), # Use model_dump
            "persist_config": orchestrator.persist_config.model_dump(),
        }
        orchestrator.data_manager.save_run_config(all_configs)

    except Exception as e:
        logger.error(f"Failed to log parameters/configs to MLflow: {e}", exc_info=True)


def log_metrics_to_mlflow(metrics: dict, step: int):
    """Logs a dictionary of metrics to MLflow."""
    try:
        numeric_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.number)) and np.isfinite(v):
                numeric_metrics[k] = v
            else: logger.debug(f"Skipping non-numeric metric for MLflow: {k}={v} (type: {type(v)})")
        if numeric_metrics: mlflow.log_metrics(numeric_metrics, step=step)
    except Exception as e: logger.error(f"Failed to log metrics to MLflow: {e}")


# --- State Loading/Saving ---
# REMOVED - Handled by DataManager

# --- Self-Play Result Processing ---

def process_self_play_result(
    orchestrator: "TrainingOrchestrator", result: SelfPlayResult, worker_id: int
):
    """Processes the result (SelfPlayResult Pydantic model) from a completed self-play episode."""
    # Access data using Pydantic model attributes
    if result.episode_experiences:
        episode_total_sims = 0
        episode_root_visits = []
        episode_tree_depths = []
        for exp_state, _, _ in result.episode_experiences:
            if hasattr(exp_state, "display_stats") and exp_state.display_stats:
                episode_total_sims += exp_state.display_stats.get("mcts_simulations", 0)
                episode_root_visits.append(exp_state.display_stats.get("mcts_root_visits", 0))
                episode_tree_depths.append(exp_state.display_stats.get("mcts_tree_depth", 0))

        orchestrator.buffer.add_batch(result.episode_experiences)
        if orchestrator.buffer_fill_progress:
            orchestrator.buffer_fill_progress.set_current_steps(len(orchestrator.buffer))
        orchestrator.episodes_played += 1
        orchestrator.total_simulations_run += episode_total_sims
        _log_self_play_results_async(orchestrator, result.final_score, result.episode_steps, episode_root_visits, episode_tree_depths, worker_id)
    elif not orchestrator.stop_requested.is_set():
        logger.warning(f"Self-play episode from worker {worker_id} produced no experiences.")


def _log_self_play_results_async(
    orchestrator: "TrainingOrchestrator", final_score: float, episode_steps: int,
    root_visits: list, tree_depths: list, worker_id: int
):
    """Logs self-play results asynchronously."""
    avg_root_visits = np.mean(root_visits) if root_visits else 0
    avg_tree_depth = np.mean(tree_depths) if tree_depths else 0
    episode_num = orchestrator.episodes_played
    global_step = orchestrator.global_step
    buffer_size = len(orchestrator.buffer)
    total_sims = orchestrator.total_simulations_run
    buffer_fill_perc = (orchestrator.buffer_fill_progress.get_progress() * 100) if orchestrator.buffer_fill_progress else 0.0

    logger.info(
        f"[W{worker_id}] Ep {episode_num} ({episode_steps} steps, Score: {final_score:.2f}, "
        f"Visits: {avg_root_visits:.1f}, Depth: {avg_tree_depth:.1f}). Buffer: {buffer_size}"
    )

    if orchestrator.stats_collector_actor:
        stats_batch = {
            "SelfPlay/Episode_Score": (final_score, episode_num), "SelfPlay/Episode_Length": (episode_steps, episode_num),
            "MCTS/Avg_Root_Visits": (avg_root_visits, episode_num), "MCTS/Avg_Tree_Depth": (avg_tree_depth, episode_num),
            "Buffer/Size": (buffer_size, global_step), "Progress/Total_Simulations": (total_sims, global_step),
            "Buffer/Fill_Percent": (buffer_fill_perc, global_step),
        }
        orchestrator.stats_collector_actor.log_batch.remote(stats_batch)

    mlflow.log_metric("SelfPlay/Episode_Score", final_score, step=episode_num)
    mlflow.log_metric("SelfPlay/Episode_Length", episode_steps, step=episode_num)
    mlflow.log_metric("MCTS/Avg_Root_Visits", avg_root_visits, step=episode_num)
    mlflow.log_metric("MCTS/Avg_Tree_Depth", avg_tree_depth, step=episode_num)
    mlflow.log_metric("Progress/Episodes_Played", episode_num, step=global_step)
    mlflow.log_metric("Progress/Total_Simulations", total_sims, step=global_step)
    mlflow.log_metric("Buffer/Size", buffer_size, step=global_step)
    mlflow.log_metric("Buffer/Fill_Percent", buffer_fill_perc, step=global_step)


# --- Training Step ---

def run_training_step(orchestrator: "TrainingOrchestrator") -> bool:
    """Runs one step of neural network training."""
    batch = orchestrator.buffer.sample(orchestrator.train_config.BATCH_SIZE)
    if not batch: return False
    loss_info = orchestrator.trainer.train_step(batch)
    if loss_info:
        orchestrator.global_step += 1
        if orchestrator.train_step_progress:
            orchestrator.train_step_progress.set_current_steps(orchestrator.global_step)
        _log_training_results_async(orchestrator, loss_info)
        if orchestrator.global_step % 50 == 0:
            logger.info(f"Step {orchestrator.global_step}: P Loss={loss_info['policy_loss']:.4f}, V Loss={loss_info['value_loss']:.4f}, Ent={loss_info['entropy']:.4f}")
        return True
    else:
        logger.warning(f"Training step {orchestrator.global_step + 1} failed (loss_info is None).")
        return False


def _log_training_results_async(orchestrator: "TrainingOrchestrator", loss_info: dict):
    """Logs training results asynchronously."""
    current_lr = orchestrator.trainer.get_current_lr()
    step = orchestrator.global_step
    train_step_perc = (orchestrator.train_step_progress.get_progress() * 100) if orchestrator.train_step_progress else 0.0

    if orchestrator.stats_collector_actor:
        stats_batch = {
            "Loss/Total": (loss_info["total_loss"], step), "Loss/Policy": (loss_info["policy_loss"], step),
            "Loss/Value": (loss_info["value_loss"], step), "Loss/Entropy": (loss_info["entropy"], step),
            "LearningRate": (current_lr, step), "Progress/Train_Step_Percent": (train_step_perc, step),
        }
        orchestrator.stats_collector_actor.log_batch.remote(stats_batch)

    mlflow_metrics = {
        "Loss/Total": loss_info["total_loss"], "Loss/Policy": loss_info["policy_loss"],
        "Loss/Value": loss_info["value_loss"], "Loss/Entropy": loss_info["entropy"],
        "LearningRate": current_lr, "Progress/Train_Step_Percent": train_step_perc,
    }
    log_metrics_to_mlflow(mlflow_metrics, step=step)