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
