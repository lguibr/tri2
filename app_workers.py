import threading
import queue
import time
from typing import TYPE_CHECKING, Optional, List, Dict

from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppWorkerManager:
    """Manages the creation, starting, and stopping of worker threads."""

    def __init__(self, app: "MainApp"):
        self.app = app
        # Keep references to the worker *instances* from AppInitializer
        self.self_play_worker_threads: List[SelfPlayWorker] = []  # Now a list
        self.training_worker_thread: Optional[TrainingWorker] = None
        print("[AppWorkerManager] Initialized.")

    def get_active_worker_counts(self) -> Dict[str, int]:
        """Returns the count of currently active workers by type."""
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

    def start_all_workers(self):
        """Starts all initialized worker threads if they are not already running."""
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

        # Check worker instances from initializer
        if (
            not self.app.initializer.self_play_workers
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

        # --- Start Self-Play Workers ---
        self.self_play_worker_threads = []  # Reset the list of active threads
        for i, worker_instance in enumerate(self.app.initializer.self_play_workers):
            if worker_instance:
                # Need to create a new thread instance if the old one was joined
                if not worker_instance.is_alive():
                    try:
                        # Recreate worker with original args
                        recreated_worker = SelfPlayWorker(
                            **worker_instance.get_init_args()
                        )
                        self.app.initializer.self_play_workers[i] = (
                            recreated_worker  # Update initializer ref
                        )
                        worker_to_start = recreated_worker
                        print(f"  Recreated SelfPlayWorker-{i}.")
                    except Exception as e:
                        print(f"  ERROR recreating SelfPlayWorker-{i}: {e}")
                        continue  # Skip starting this worker
                else:
                    worker_to_start = worker_instance  # Start existing instance

                self.self_play_worker_threads.append(
                    worker_to_start
                )  # Add to active list
                worker_to_start.start()
                print(f"  SelfPlayWorker-{i} thread started.")
            else:
                print(
                    f"[AppWorkerManager] ERROR: SelfPlayWorker instance {i} is None during start."
                )

        # --- Start Training Worker ---
        self.training_worker_thread = self.app.initializer.training_worker
        if self.training_worker_thread:
            if not self.training_worker_thread.is_alive():
                try:
                    # Recreate worker with original args
                    recreated_worker = TrainingWorker(
                        **self.training_worker_thread.get_init_args()
                    )
                    self.app.initializer.training_worker = (
                        recreated_worker  # Update initializer ref
                    )
                    self.training_worker_thread = recreated_worker
                    print("  Recreated TrainingWorker.")
                except Exception as e:
                    print(f"  ERROR recreating TrainingWorker: {e}")
                    self.training_worker_thread = None  # Failed to recreate

            if self.training_worker_thread:  # Check again if recreation was successful
                self.training_worker_thread.start()
                print("  TrainingWorker thread started.")
        else:
            print(
                "[AppWorkerManager] ERROR: TrainingWorker instance is None during start."
            )

        if self.is_any_worker_running():
            self.app.status = "Running AlphaZero"
            num_sp = len(self.self_play_worker_threads)
            num_tr = 1 if self.is_training_running() else 0
            print(f"[AppWorkerManager] Workers started ({num_sp} SP, {num_tr} TR).")

    def stop_all_workers(self, join_timeout: float = 5.0):
        """Signals ALL worker threads to stop and waits for them to join."""
        if not self.is_any_worker_running():
            return

        print("[AppWorkerManager] Stopping ALL worker threads...")
        self.app.stop_event.set()  # Signal stop

        threads_to_join: List[Tuple[str, threading.Thread]] = []

        # Add active self-play workers
        for i, worker in enumerate(self.self_play_worker_threads):
            if worker and worker.is_alive():
                threads_to_join.append((f"SelfPlayWorker-{i}", worker))

        # Add active training worker
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
        self.self_play_worker_threads = []
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
