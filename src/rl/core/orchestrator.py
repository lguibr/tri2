# File: src/rl/core/orchestrator.py
import logging
import time
import threading
import queue
import ray
import torch
from typing import Optional, Dict, Any, List
from collections import deque

# --- Package Imports ---
# Change: Import MCTSConfig from the central config location
from src.config import (
    TrainConfig,
    EnvConfig,
    PersistenceConfig,
    ModelConfig,
    MCTSConfig,
)
from src.nn import NeuralNetwork

# Import DataManager and the Pydantic schema for loaded state
from src.data import DataManager, LoadedTrainingState
from src.environment import GameState
from src.utils import format_eta, get_device

# Import Experience from utils, SelfPlayResult from local rl types
from src.utils.types import StatsCollectorData, Experience
from ..types import SelfPlayResult  # Corrected import location
from src.visualization.ui import ProgressBar
from src.stats import StatsCollectorActor

# Other RL Components
from .buffer import ExperienceBuffer, SumTree  # Import SumTree here
from .trainer import Trainer
from .visual_state_actor import VisualStateActor
from ..self_play.worker import SelfPlayWorker  # Keep worker import

# Import helper functions (logging only now)
from . import orchestrator_helpers as helpers

logger = logging.getLogger(__name__)

LARGE_STEP_COUNT = 10_000_000
VISUAL_UPDATE_INTERVAL = 0.2
STATS_FETCH_INTERVAL = 0.5


class TrainingOrchestrator:
    """
    Manages the overall training loop, coordinating parallel self-play via Ray actors
    and centralized training. Uses DataManager for loading/saving.
    Integrates Prioritized Experience Replay (PER).
    """

    def __init__(
        self,
        nn: NeuralNetwork,
        buffer: ExperienceBuffer,
        trainer: Trainer,
        data_manager: DataManager,
        stats_collector_actor: StatsCollectorActor,
        train_config: TrainConfig,
        env_config: EnvConfig,
        mcts_config: MCTSConfig,  # Use Pydantic MCTSConfig from central location
        persist_config: PersistenceConfig,
        visual_state_queue: Optional[queue.Queue[Optional[Dict[int, Any]]]] = None,
    ):
        self.nn = nn
        self.buffer = buffer  # Should be PER-enabled buffer if config.USE_PER
        self.trainer = trainer
        self.data_manager = data_manager
        self.stats_collector_actor = stats_collector_actor
        self.train_config = train_config
        self.env_config = env_config
        self.mcts_config = mcts_config  # Store Pydantic instance
        self.persist_config = persist_config
        self.visual_state_queue = visual_state_queue

        self.device = nn.device
        self.global_step = 0
        self.episodes_played = 0
        self.total_simulations_run = 0
        self.best_eval_score = -float(
            "inf"
        )  # TODO: Implement evaluation logic if needed
        self.start_time = time.time()
        self.stop_requested = threading.Event()
        self.training_complete = False
        self.target_steps_reached = False
        self.training_exception: Optional[Exception] = None
        self.last_visual_update_time = 0.0
        self.last_stats_fetch_time = 0.0
        self.latest_stats_data: StatsCollectorData = {}

        self.train_step_progress: Optional[ProgressBar] = None
        self.buffer_fill_progress: Optional[ProgressBar] = None

        self.visual_state_actor = VisualStateActor.remote()
        self.workers: List[ray.actor.ActorHandle] = []
        self.worker_tasks: Dict[ray.ObjectRef, int] = {}

        self._load_and_initialize_state()
        self._initialize_workers()
        self._initialize_progress_bars()
        # No need to update workers here, initial weights passed during init
        helpers.log_configs_to_mlflow(self)
        self._update_visual_states()
        logger.info("Orchestrator initialized.")

    def _load_and_initialize_state(self):
        """Loads initial state using DataManager and applies it."""
        logger.info("Loading initial training state...")
        loaded_state: LoadedTrainingState = self.data_manager.load_initial_state()

        if loaded_state.checkpoint_data:
            cp_data = loaded_state.checkpoint_data
            logger.info(
                f"Applying loaded checkpoint data (Run: {cp_data.run_name}, Step: {cp_data.global_step})"
            )

            # Config validation (optional, basic check)
            if cp_data.model_config_dict != self.nn.model_config.model_dump():
                logger.warning("Loaded ModelConfig differs from current config!")
            if cp_data.env_config_dict != self.env_config.model_dump():
                logger.warning("Loaded EnvConfig differs from current config!")

            if cp_data.model_state_dict:
                self.nn.set_weights(cp_data.model_state_dict)
            else:
                logger.warning("No model state dictionary found in checkpoint.")

            if cp_data.optimizer_state_dict:
                try:
                    self.trainer.optimizer.load_state_dict(cp_data.optimizer_state_dict)
                    for state in self.trainer.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.nn.device)
                    logger.info("Optimizer state loaded and moved to device.")
                except Exception as opt_load_err:
                    logger.error(
                        f"Could not load optimizer state: {opt_load_err}. Optimizer might reset."
                    )
            else:
                logger.warning("No optimizer state found in checkpoint.")

            if cp_data.stats_collector_state and self.stats_collector_actor:
                try:
                    set_state_ref = self.stats_collector_actor.set_state.remote(
                        cp_data.stats_collector_state
                    )
                    ray.get(set_state_ref, timeout=5.0)
                    logger.info("StatsCollectorActor state restored.")
                except Exception as e:
                    logger.error(
                        f"Error restoring StatsCollectorActor state: {e}", exc_info=True
                    )
            elif not cp_data.stats_collector_state:
                logger.warning("No stats_collector_state found in checkpoint.")

            self.global_step = cp_data.global_step
            self.episodes_played = cp_data.episodes_played
            self.total_simulations_run = cp_data.total_simulations_run

        else:
            logger.info("No checkpoint data loaded. Starting fresh.")
            self.trainer.optimizer.zero_grad(set_to_none=True)
            if self.stats_collector_actor:
                try:
                    ray.get(self.stats_collector_actor.clear.remote())
                except Exception as e:
                    logger.error(f"Failed to clear stats actor state: {e}")

        # Apply Buffer state - Handles both PER and uniform buffer loading
        if loaded_state.buffer_data:
            if self.train_config.USE_PER:
                # Rebuild SumTree from loaded list (assuming BufferData stores list)
                logger.info("Rebuilding PER SumTree from loaded buffer data...")
                # Ensure buffer object has 'tree' attribute before assigning
                if not hasattr(self.buffer, "tree") or self.buffer.tree is None:
                    self.buffer.tree = SumTree(
                        self.buffer.capacity
                    )  # Initialize if missing
                else:
                    # Re-initialize tree to clear old state before loading
                    self.buffer.tree = SumTree(self.buffer.capacity)

                # Add items with max priority initially when loading
                max_p = 1.0  # Start with max priority 1 when loading
                for exp in loaded_state.buffer_data.buffer_list:
                    self.buffer.tree.add(max_p, exp)
                # Note: Priorities will be updated after the first training steps
                logger.info(
                    f"PER buffer loaded and rebuilt. Size: {len(self.buffer)}, Capacity: {self.buffer.capacity}"
                )
            else:
                # Load into deque for uniform buffer
                self.buffer.buffer = deque(
                    loaded_state.buffer_data.buffer_list, maxlen=self.buffer.capacity
                )
                logger.info(
                    f"Uniform experience buffer loaded. Size: {len(self.buffer.buffer)}, Capacity: {self.buffer.capacity}"
                )
        else:
            logger.info("No buffer data loaded.")

        self.nn.model.train()  # Ensure model is in training mode after loading

    def _initialize_workers(self):
        """Creates the pool of SelfPlayWorker Ray actors."""
        logger.info(
            f"Initializing {self.train_config.NUM_SELF_PLAY_WORKERS} self-play workers..."
        )
        initial_weights = self.nn.get_weights()
        weights_ref = ray.put(initial_weights)

        for i in range(self.train_config.NUM_SELF_PLAY_WORKERS):
            worker = SelfPlayWorker.options(num_cpus=1).remote(
                actor_id=i,
                env_config=self.env_config,
                mcts_config=self.mcts_config,  # Pass the unified config
                model_config=self.nn.model_config,
                train_config=self.train_config,
                initial_weights=weights_ref,
                seed=self.train_config.RANDOM_SEED + i,
                worker_device_str=self.train_config.WORKER_DEVICE,
                visual_state_actor_handle=self.visual_state_actor,
            )
            self.workers.append(worker)
        logger.info("Self-play workers initialized.")

    def _update_worker_networks(self):
        """Sends the latest network weights to all workers using ray.put."""
        if not self.workers:
            return
        logger.debug("Updating worker networks...")
        current_weights = self.nn.get_weights()
        weights_ref = ray.put(current_weights)
        update_tasks = [
            worker.set_weights.remote(weights_ref) for worker in self.workers if worker
        ]
        if not update_tasks:
            return
        try:
            ray.get(update_tasks, timeout=15.0)
            logger.debug("Worker networks updated.")
        except ray.exceptions.RayActorError as e:
            logger.error(
                f"A worker actor failed during weight update: {e}", exc_info=True
            )
        except ray.exceptions.GetTimeoutError:
            logger.error("Timeout waiting for workers to update weights.")
        except Exception as e:
            logger.error(
                f"Unexpected error updating worker networks: {e}", exc_info=True
            )

    def _initialize_progress_bars(self):
        """Initializes progress bars after state has been loaded."""
        train_total_steps = self.train_config.MAX_TRAINING_STEPS or LARGE_STEP_COUNT
        self.train_step_progress = ProgressBar(
            "Training Steps",
            train_total_steps,
            start_time=self.start_time,
            initial_steps=self.global_step,
        )
        self.buffer_fill_progress = ProgressBar(
            "Buffer Fill",
            self.train_config.BUFFER_CAPACITY,
            start_time=self.start_time,
            initial_steps=len(self.buffer),
        )
        if self.global_step > 0:
            current_time = time.time()
            self.train_step_progress.start_time = current_time
            self.buffer_fill_progress.start_time = current_time
            logger.info("Reset progress bar start time due to loading checkpoint.")

    def _fetch_latest_stats(self):
        """Fetches the latest stats data from the actor for ETA calculation."""
        current_time = time.time()
        if current_time - self.last_stats_fetch_time < STATS_FETCH_INTERVAL:
            return
        self.last_stats_fetch_time = current_time
        if self.stats_collector_actor:
            try:
                data_ref = self.stats_collector_actor.get_data.remote()
                self.latest_stats_data = ray.get(data_ref, timeout=1.0)
            except Exception as e:
                logger.warning(f"Failed to fetch latest stats for ETA: {e}")

    def _log_progress_eta(self):
        """Logs progress and ETA."""
        if self.global_step % 20 != 0 and not self.target_steps_reached:
            return
        if not self.train_step_progress:
            return

        elapsed_time = time.time() - self.train_step_progress.start_time
        steps_since_load = self.global_step - self.train_step_progress.initial_steps
        steps_per_sec = steps_since_load / elapsed_time if elapsed_time > 1 else 0
        target_steps = self.train_config.MAX_TRAINING_STEPS
        target_steps_str = str(target_steps) if target_steps else "inf"
        eta_str = (
            format_eta(self.train_step_progress.get_eta_seconds())
            if not self.target_steps_reached
            else "N/A (Target Reached)"
        )
        progress_str = f"Step {self.global_step}/{target_steps_str}"
        if self.target_steps_reached:
            progress_str += (
                f" (TARGET REACHED +{self.global_step - (target_steps or 0)} extra)"
            )
        buffer_fill_perc = (
            (len(self.buffer) / self.buffer.capacity) * 100
            if self.buffer.capacity > 0
            else 0.0
        )
        total_sims_str = (
            f"{self.total_simulations_run / 1e6:.2f}M"
            if self.total_simulations_run >= 1e6
            else (
                f"{self.total_simulations_run / 1e3:.1f}k"
                if self.total_simulations_run >= 1000
                else str(self.total_simulations_run)
            )
        )
        num_pending_tasks = len(self.worker_tasks)
        logger.info(
            f"Progress: {progress_str}, Episodes: {self.episodes_played}, Total Sims: {total_sims_str}, "
            f"Buffer: {len(self.buffer)}/{self.buffer.capacity} ({buffer_fill_perc:.1f}%), "
            f"Pending Tasks: {num_pending_tasks}, Speed: {steps_per_sec:.2f} steps/sec, ETA: {eta_str}"
        )

    def request_stop(self):
        """Signals the training loop and workers to stop gracefully."""
        if not self.stop_requested.is_set():
            logger.info("Stop requested.")
            self.stop_requested.set()

    def _update_visual_states(self):
        """Fetches latest states and stats, puts them on the visual queue."""
        if not self.visual_state_queue or not self.visual_state_actor:
            return
        current_time = time.time()
        if current_time - self.last_visual_update_time < VISUAL_UPDATE_INTERVAL:
            return
        self.last_visual_update_time = current_time
        try:
            states_ref = self.visual_state_actor.get_all_states.remote()
            stats_ref = (
                self.stats_collector_actor.get_data.remote()
                if self.stats_collector_actor
                else None
            )
            refs_to_get = [states_ref] + ([stats_ref] if stats_ref else [])
            results = ray.get(refs_to_get, timeout=1.0)
            combined_states = results[0]
            stats_data = results[1] if stats_ref else {}

            global_stats_for_vis = {
                "global_step": self.global_step,
                "target_steps_reached": self.target_steps_reached,
                "total_episodes": self.episodes_played,
                "total_simulations": self.total_simulations_run,
                "train_progress": self.train_step_progress,
                "buffer_progress": self.buffer_fill_progress,
                "stats_data": stats_data,
                "num_workers": len(self.workers),
                "pending_tasks": len(self.worker_tasks),
            }
            combined_states[-1] = global_stats_for_vis

            if len(combined_states) <= 1:
                return  # No worker states
            try:
                while self.visual_state_queue.full():
                    self.visual_state_queue.get_nowait()
                self.visual_state_queue.put_nowait(combined_states)
            except queue.Full:
                logger.warning("Visual state queue full, dropping state dictionary.")
            except Exception as qe:
                logger.error(f"Error putting state dict in visual queue: {qe}")
        except Exception as e:
            logger.warning(f"Error getting states/stats for visualization: {e}")

    def save_final_state(self):
        """Saves the final training state using DataManager."""
        logger.info("Saving final training state...")
        try:
            self.data_manager.save_training_state(
                nn=self.nn,
                optimizer=self.trainer.optimizer,
                stats_collector_actor=self.stats_collector_actor,
                buffer=self.buffer,
                global_step=self.global_step,
                episodes_played=self.episodes_played,
                total_simulations_run=self.total_simulations_run,
                is_final=True,
            )
        except Exception as e_save:
            logger.error(f"Failed to save final training state: {e_save}")

    def _final_cleanup(self):
        """Performs final cleanup of Ray actors and signals visualizer."""
        end_time = time.time()
        logger.info(
            f"Training loop finished after {format_eta(end_time - self.start_time)}."
        )
        logger.info("Terminating Ray workers and helper actors...")
        actors_to_kill = self.workers + [
            self.visual_state_actor,
            self.stats_collector_actor,
        ]
        for actor in actors_to_kill:
            if actor:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as kill_e:
                    logger.warning(f"Error killing actor: {kill_e}")
        self.workers = []
        self.visual_state_actor = None
        self.stats_collector_actor = None
        self.worker_tasks = {}
        logger.info("Ray actors terminated.")
        if self.visual_state_queue:
            logger.info("Signaling visualizer thread to stop.")
            try:
                self.visual_state_queue.put(None, timeout=1.0)
            except Exception as qe:
                logger.error(f"Error signaling visual queue on exit: {qe}")

    def run_training_loop(self):
        """Main training loop coordinating parallel self-play and training."""
        logger.info(
            f"Starting training loop... Target steps: {self.train_config.MAX_TRAINING_STEPS or 'Infinite'}"
        )
        self.start_time = time.time()  # Reset start time after loading

        try:
            # Initial launch of tasks on all workers
            for i, worker in enumerate(self.workers):
                if worker:
                    task_ref = worker.run_episode.remote()
                    self.worker_tasks[task_ref] = i

            while not self.stop_requested.is_set():
                if (
                    not self.target_steps_reached
                    and self.train_config.MAX_TRAINING_STEPS
                    and self.global_step >= self.train_config.MAX_TRAINING_STEPS
                ):
                    logger.info(
                        f"Reached target training steps ({self.train_config.MAX_TRAINING_STEPS})."
                    )
                    self.target_steps_reached = True

                # --- Training Step (only if target not reached) ---
                if self.buffer.is_ready() and not self.target_steps_reached:
                    # Use helper which now handles PER priority updates internally
                    trained_this_cycle = helpers.run_training_step(self)
                    if trained_this_cycle and (
                        self.global_step % self.train_config.WORKER_UPDATE_FREQ_STEPS
                        == 0
                    ):
                        self._update_worker_networks()

                if self.stop_requested.is_set():
                    break

                # --- Collect Self-Play Results ---
                wait_timeout = 0.1 if self.buffer.is_ready() else 0.5
                ready_refs, _ = ray.wait(
                    list(self.worker_tasks.keys()), num_returns=1, timeout=wait_timeout
                )

                if ready_refs:
                    for ref in ready_refs:
                        worker_idx = self.worker_tasks.pop(ref, -1)
                        if worker_idx == -1:
                            continue
                        try:
                            result: SelfPlayResult = ray.get(ref)
                            helpers.process_self_play_result(self, result, worker_idx)
                        except ray.exceptions.RayActorError as e:
                            logger.error(
                                f"Worker {worker_idx} failed: {e}", exc_info=True
                            )
                            if worker_idx < len(self.workers):
                                self.workers[worker_idx] = None  # Mark worker as failed
                        except Exception as e:
                            logger.error(
                                f"Error processing result from worker {worker_idx}: {e}",
                                exc_info=True,
                            )
                            if worker_idx < len(self.workers):
                                self.workers[worker_idx] = None  # Mark worker as failed
                            continue
                        # Relaunch task only if worker is still valid and not stopping
                        if (
                            not self.stop_requested.is_set()
                            and worker_idx < len(self.workers)
                            and self.workers[worker_idx]
                        ):
                            new_task_ref = self.workers[worker_idx].run_episode.remote()
                            self.worker_tasks[new_task_ref] = worker_idx
                if self.stop_requested.is_set():
                    break

                self._update_visual_states()

                # --- Periodic Checkpointing (only if target not reached) ---
                if (
                    not self.target_steps_reached
                    and self.global_step > 0
                    and self.global_step % self.train_config.CHECKPOINT_SAVE_FREQ_STEPS
                    == 0
                ):
                    self.data_manager.save_training_state(
                        nn=self.nn,
                        optimizer=self.trainer.optimizer,
                        stats_collector_actor=self.stats_collector_actor,
                        buffer=self.buffer,
                        global_step=self.global_step,
                        episodes_played=self.episodes_played,
                        total_simulations_run=self.total_simulations_run,
                    )

                self._log_progress_eta()
                if not ready_refs and not self.buffer.is_ready():
                    time.sleep(0.05)

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received. Stopping training gracefully.")
            self.request_stop()
        except Exception as e:
            logger.critical(f"Unhandled exception in training loop: {e}", exc_info=True)
            self.training_exception = e
            self.request_stop()
        finally:
            if self.training_exception:
                self.training_complete = False
            elif self.stop_requested.is_set():
                self.training_complete = self.target_steps_reached
            else:
                self.training_complete = self.target_steps_reached
            # Cleanup is called externally by the script that ran the orchestrator
