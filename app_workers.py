# File: app_workers.py
import threading
import queue
import time
import traceback
from typing import TYPE_CHECKING, Optional

from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppWorkerManager:
    """Manages the creation, starting, and stopping of worker threads."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Keep references to the worker *instances* from AppInitializer
        self.self_play_worker_thread: Optional[SelfPlayWorker] = None
        self.training_worker_thread: Optional[TrainingWorker] = None
        print("[AppWorkerManager] Initialized.")

    def is_self_play_running(self) -> bool:
        """Checks if the self-play worker thread is active."""
        return (
            self.self_play_worker_thread is not None
            and self.self_play_worker_thread.is_alive()
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

    def start_all_workers(self):
        """Starts both worker threads if they are initialized and not running."""
        if self.is_any_worker_running():
            print("[AppWorkerManager] Workers already running.")
            return

        # Check required components
        if (
            not self.app.initializer.agent
            or not self.app.initializer.mcts
            or not self.app.initializer.stats_aggregator
            or not self.app.initializer.optimizer
        ):
            print(
                "[AppWorkerManager] ERROR: Cannot start workers, core RL components missing."
            )
            self.app.app_state = self.app.app_state.ERROR
            self.app.status = "Worker Init Failed"
            return

        # Check worker instances
        if (
            not self.app.initializer.self_play_worker
            or not self.app.initializer.training_worker
        ):
            print(
                "[AppWorkerManager] ERROR: Workers not initialized in AppInitializer."
            )
            self.app.app_state = self.app.app_state.ERROR
            self.app.status = "Worker Init Failed"
            return

        print("[AppWorkerManager] Starting all worker threads...")
        self.app.stop_event.clear()  # Clear stop event before starting

        # Start Self-Play
        self.self_play_worker_thread = self.app.initializer.self_play_worker
        if self.self_play_worker_thread:
            # Need to create a new thread instance if the old one was joined
            if not self.self_play_worker_thread.is_alive():
                self.self_play_worker_thread = SelfPlayWorker(
                    **self.self_play_worker_thread.get_init_args()
                )  # Recreate
                self.app.initializer.self_play_worker = (
                    self.self_play_worker_thread
                )  # Update initializer ref
            self.self_play_worker_thread.start()
            print("SelfPlayWorker thread started.")
        else:
            print(
                "[AppWorkerManager] ERROR: SelfPlayWorker instance is None during start."
            )

        # Start Training
        self.training_worker_thread = self.app.initializer.training_worker
        if self.training_worker_thread:
            if not self.training_worker_thread.is_alive():
                self.training_worker_thread = TrainingWorker(
                    **self.training_worker_thread.get_init_args()
                )  # Recreate
                self.app.initializer.training_worker = (
                    self.training_worker_thread
                )  # Update initializer ref
            self.training_worker_thread.start()
            print("TrainingWorker thread started.")
        else:
            print(
                "[AppWorkerManager] ERROR: TrainingWorker instance is None during start."
            )

        if self.is_any_worker_running():
            self.app.status = "Running AlphaZero"

    def stop_all_workers(self, join_timeout: float = 5.0):
        """Signals ALL worker threads to stop and waits for them to join."""
        if not self.is_any_worker_running():
            # print("[AppWorkerManager] No workers running to stop.")
            return

        print("[AppWorkerManager] Stopping ALL worker threads...")
        self.app.stop_event.set()  # Signal stop

        threads_to_join = []
        if self.self_play_worker_thread and self.self_play_worker_thread.is_alive():
            threads_to_join.append(("SelfPlayWorker", self.self_play_worker_thread))
        if self.training_worker_thread and self.training_worker_thread.is_alive():
            threads_to_join.append(("TrainingWorker", self.training_worker_thread))

        start_join_time = time.time()
        for name, thread in threads_to_join:
            remaining_timeout = max(0.1, join_timeout - (time.time() - start_join_time))
            print(
                f"[AppWorkerManager] Joining {name} (timeout: {remaining_timeout:.1f}s)..."
            )
            thread.join(timeout=remaining_timeout)
            if thread.is_alive():
                print(
                    f"[AppWorkerManager] WARNING: {name} thread did not join cleanly."
                )
            else:
                print(f"[AppWorkerManager] {name} joined.")

        # Clear references after joining
        self.self_play_worker_thread = None
        self.training_worker_thread = None

        # Clear experience queue after stopping workers
        print("[AppWorkerManager] Clearing experience queue...")
        cleared_count = 0
        while not self.app.experience_queue.empty():
            try:
                self.app.experience_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error clearing queue item: {e}")
                break
        print(
            f"[AppWorkerManager] Cleared {cleared_count} items from experience queue."
        )

        print("[AppWorkerManager] All worker threads stopped.")
        # Keep stop_event set after stopping all workers? Or clear it?
        # Let's clear it so they can be restarted individually if needed later.
        # self.app.stop_event.clear() # Decide if this should be cleared
