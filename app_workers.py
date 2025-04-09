# File: app_workers.py
import threading
import queue
import time
from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Any
import logging
import multiprocessing as mp
import ray
import asyncio
import torch

# Import Ray actor classes
from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker

# Import Actor Handles for type hinting
if TYPE_CHECKING:
    LogicAppState = Any
    SelfPlayWorkerHandle = ray.actor.ActorHandle
    TrainingWorkerHandle = ray.actor.ActorHandle
    AgentPredictorHandle = ray.actor.ActorHandle
    StatsAggregatorHandle = ray.actor.ActorHandle
    from ray.util.queue import Queue as RayQueue

logger = logging.getLogger(__name__)


class AppWorkerManager:
    """Manages the creation, starting, and stopping of Ray worker actors."""

    DEFAULT_KILL_TIMEOUT = 5.0

    def __init__(self, app: "LogicAppState"):
        self.app = app
        self.self_play_worker_actors: List["SelfPlayWorkerHandle"] = []
        self.training_worker_actor: Optional["TrainingWorkerHandle"] = None
        self.agent_predictor_actor: Optional["AgentPredictorHandle"] = None
        self._workers_running = False
        logger.info("[AppWorkerManager] Initialized for Ray Actors.")

    def initialize_actors(self):
        """Initializes Ray worker actors (SelfPlay, Training). Does NOT start their loops."""
        logger.info("[AppWorkerManager] Initializing worker actors...")
        if not self.app.agent_predictor:
            logger.error(
                "[AppWorkerManager] ERROR: AgentPredictor actor not initialized in AppInitializer."
            )
            self.app.set_state(self.app.app_state.ERROR)
            self.app.set_status("Worker Init Failed: Missing AgentPredictor")
            return
        if not self.app.stats_aggregator:
            logger.error(
                "[AppWorkerManager] ERROR: StatsAggregator actor handle not initialized in AppInitializer."
            )
            self.app.set_state(self.app.app_state.ERROR)
            self.app.set_status("Worker Init Failed: Missing StatsAggregator")
            return

        self.agent_predictor_actor = self.app.agent_predictor

        self._init_self_play_actors()
        self._init_training_actor()

        num_sp = len(self.self_play_worker_actors)
        num_tr = 1 if self.training_worker_actor else 0
        logger.info(
            f"Worker actors initialized ({num_sp} Self-Play, {num_tr} Training)."
        )

    def _init_self_play_actors(self):
        """Creates SelfPlayWorker Ray actors."""
        self.self_play_worker_actors = []
        num_sp_workers = self.app.train_config_instance.NUM_SELF_PLAY_WORKERS
        logger.info(f"Initializing {num_sp_workers} SelfPlayWorker actor(s)...")
        for i in range(num_sp_workers):
            try:
                actor = SelfPlayWorker.remote(
                    worker_id=i,
                    agent_predictor=self.agent_predictor_actor,
                    mcts_config=self.app.mcts_config,
                    env_config=self.app.env_config,
                    experience_queue=self.app.experience_queue,
                    stats_aggregator=self.app.stats_aggregator,
                    max_game_steps=None,
                )
                self.self_play_worker_actors.append(actor)
                logger.info(f"  SelfPlayWorker-{i} actor created.")
            except Exception as e:
                logger.error(
                    f"  ERROR creating SelfPlayWorker-{i} actor: {e}", exc_info=True
                )

    def _init_training_actor(self):
        """Creates the TrainingWorker Ray actor."""
        logger.info("Initializing TrainingWorker actor...")
        if not self.app.optimizer or not self.app.train_config_instance:
            logger.error(
                "[AppWorkerManager] ERROR: Optimizer or TrainConfig missing for TrainingWorker init."
            )
            return

        optimizer_cls = type(self.app.optimizer)
        optimizer_kwargs = self.app.optimizer.defaults

        scheduler_cls = type(self.app.scheduler) if self.app.scheduler else None
        scheduler_kwargs = {}
        if self.app.scheduler and hasattr(self.app.scheduler, "state_dict"):
            sd = self.app.scheduler.state_dict()
            if isinstance(
                self.app.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
            ):
                scheduler_kwargs = {
                    "T_max": sd.get("T_max", 1000),
                    "eta_min": sd.get("eta_min", 0),
                }
            else:
                logger.warning(
                    f"Cannot automatically determine kwargs for scheduler type {scheduler_cls}. Scheduler might not be correctly re-initialized in actor."
                )
                scheduler_cls = None

        try:
            actor = TrainingWorker.remote(
                agent_predictor=self.agent_predictor_actor,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                scheduler_cls=scheduler_cls,
                scheduler_kwargs=scheduler_kwargs,
                experience_queue=self.app.experience_queue,
                stats_aggregator=self.app.stats_aggregator,
                train_config=self.app.train_config_instance,
            )
            self.training_worker_actor = actor
            logger.info("  TrainingWorker actor created.")
        except Exception as e:
            logger.error(f"  ERROR creating TrainingWorker actor: {e}", exc_info=True)

    def get_active_worker_counts(self) -> Dict[str, int]:
        """Returns the count of initialized worker actors."""
        sp_count = len(self.self_play_worker_actors)
        tr_count = 1 if self.training_worker_actor else 0
        return {"SelfPlay": sp_count, "Training": tr_count}

    def is_any_worker_running(self) -> bool:
        """Checks the internal flag indicating if workers have been started."""
        return self._workers_running

    async def get_worker_render_data_async(
        self, max_envs: int
    ) -> List[Optional[Dict[str, Any]]]:
        """Retrieves render data from active self-play actors asynchronously."""
        if not self.self_play_worker_actors:
            return [None] * max_envs

        tasks = []
        num_to_fetch = min(len(self.self_play_worker_actors), max_envs)
        for i in range(num_to_fetch):
            actor = self.self_play_worker_actors[i]
            tasks.append(actor.get_current_render_data.remote())

        render_data_list: List[Optional[Dict[str, Any]]] = []
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting render data from worker {i}: {result}")
                    render_data_list.append(None)
                else:
                    render_data_list.append(result)
        except Exception as e:
            logger.error(f"Error gathering render data: {e}")
            render_data_list = [None] * num_to_fetch

        while len(render_data_list) < max_envs:
            render_data_list.append(None)
        return render_data_list

    def get_worker_render_data(self, max_envs: int) -> List[Optional[Dict[str, Any]]]:
        """Synchronous wrapper for get_worker_render_data_async."""
        if not self.self_play_worker_actors:
            return [None] * max_envs

        refs = []
        num_to_fetch = min(len(self.self_play_worker_actors), max_envs)
        for i in range(num_to_fetch):
            actor = self.self_play_worker_actors[i]
            refs.append(actor.get_current_render_data.remote())

        render_data_list: List[Optional[Dict[str, Any]]] = []
        try:
            results = ray.get(refs)
            render_data_list.extend(results)
        except Exception as e:
            logger.error(f"Error getting render data via ray.get: {e}")
            render_data_list = [None] * num_to_fetch

        while len(render_data_list) < max_envs:
            render_data_list.append(None)
        return render_data_list

    def start_all_workers(self):
        """Starts the main loops of all initialized worker actors."""
        if self._workers_running:
            logger.warning("[AppWorkerManager] Workers already started.")
            return
        if not self.self_play_worker_actors and not self.training_worker_actor:
            logger.error("[AppWorkerManager] No worker actors initialized to start.")
            return

        logger.info("[AppWorkerManager] Starting all worker actor loops...")
        self._workers_running = True

        for i, actor in enumerate(self.self_play_worker_actors):
            try:
                actor.run_loop.remote()
                logger.info(f"  SelfPlayWorker-{i} actor loop started.")
            except Exception as e:
                logger.error(f"  ERROR starting SelfPlayWorker-{i} actor loop: {e}")

        if self.training_worker_actor:
            try:
                self.training_worker_actor.run_loop.remote()
                logger.info("  TrainingWorker actor loop started.")
            except Exception as e:
                logger.error(f"  ERROR starting TrainingWorker actor loop: {e}")

        if self.is_any_worker_running():
            self.app.set_status("Running AlphaZero")
            num_sp = len(self.self_play_worker_actors)
            num_tr = 1 if self.training_worker_actor else 0
            logger.info(
                f"[AppWorkerManager] Worker loops started ({num_sp} SP, {num_tr} TR)."
            )

    def stop_all_workers(self, timeout: float = DEFAULT_KILL_TIMEOUT):
        """Signals all worker actors to stop and attempts to terminate them."""
        if (
            not self._workers_running
            and not self.self_play_worker_actors
            and not self.training_worker_actor
        ):
            logger.info("[AppWorkerManager] No workers running or initialized to stop.")
            return

        logger.info("[AppWorkerManager] Stopping ALL worker actors...")
        self._workers_running = False

        actors_to_stop: List[ray.actor.ActorHandle] = []
        actors_to_stop.extend(self.self_play_worker_actors)
        if self.training_worker_actor:
            actors_to_stop.append(self.training_worker_actor)

        if not actors_to_stop:
            logger.info("[AppWorkerManager] No active actor handles found to stop.")
            return

        logger.info(
            f"[AppWorkerManager] Sending stop signal to {len(actors_to_stop)} actors..."
        )
        for actor in actors_to_stop:
            try:
                actor.stop.remote()
            except Exception as e:
                logger.warning(f"Error sending stop signal to actor {actor}: {e}")

        time.sleep(0.5)

        logger.info(f"[AppWorkerManager] Killing actors...")
        for actor in actors_to_stop:
            try:
                ray.kill(actor, no_restart=True)
                logger.info(f"  Killed actor {actor}.")
            except Exception as e:
                logger.error(f"  Error killing actor {actor}: {e}")

        self.self_play_worker_actors = []
        self.training_worker_actor = None

        self._clear_experience_queue()

        logger.info("[AppWorkerManager] All worker actors stopped/killed.")
        self.app.set_status("Ready")

    def _clear_experience_queue(self):
        """Safely clears the experience queue (assuming Ray Queue)."""
        logger.info("[AppWorkerManager] Clearing experience queue...")
        # Check if it's a RayQueue instance (which acts as a handle)
        if hasattr(self.app, "experience_queue") and isinstance(
            self.app.experience_queue, ray.util.queue.Queue
        ):
            try:
                # Call qsize() directly, it returns an ObjectRef
                qsize_ref = self.app.experience_queue.qsize()
                qsize = ray.get(qsize_ref)  # Use ray.get() to resolve the ObjectRef
                logger.info(
                    f"[AppWorkerManager] Experience queue size before potential drain: {qsize}"
                )
                # Optional drain logic can be added here if needed
                # Example: Drain items if size is large
                # if qsize > 100:
                #     logger.info("[AppWorkerManager] Draining experience queue...")
                #     while qsize > 0:
                #         try:
                #             # Use get_nowait_batch to drain efficiently
                #             items_ref = self.app.experience_queue.get_nowait_batch(100)
                #             items = ray.get(items_ref)
                #             if not items: break
                #             qsize_ref = self.app.experience_queue.qsize()
                #             qsize = ray.get(qsize_ref)
                #         except ray.exceptions.RayActorError: # Handle queue actor potentially gone
                #             logger.warning("[AppWorkerManager] Queue actor error during drain.")
                #             break
                #         except Exception as drain_e:
                #             logger.error(f"Error draining queue: {drain_e}")
                #             break
                #     logger.info("[AppWorkerManager] Experience queue drained.")

            except Exception as e:
                logger.error(f"Error accessing Ray queue size: {e}")
        else:
            logger.warning(
                "[AppWorkerManager] Experience queue not found or not a Ray Queue during clearing."
            )
