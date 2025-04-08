# File: app_workers.py
import threading
import queue
import time
from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Any

from workers.self_play_worker import SelfPlayWorker
from workers.training_worker import TrainingWorker
from environment.game_state import GameState

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppWorkerManager:
    """Manages the creation, starting, and stopping of worker threads."""

    def __init__(self, app: "MainApp"):
        self.app = app
        self.self_play_worker_threads: List[SelfPlayWorker] = []
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

    # Renamed method
    def get_worker_render_data(self, max_envs: int) -> List[Optional[Dict[str, Any]]]:
        """
        Retrieves render data (state copy and stats) from active self-play workers.
        Returns a list of dictionaries or None.
        """
        render_data_list: List[Optional[Dict[str, Any]]] = []
        count = 0
        worker_instances = self.app.initializer.self_play_workers
        for worker in worker_instances:
            if count >= max_envs:
                break
            if (
                worker
                and worker.is_alive()
                and hasattr(worker, "get_current_render_data")
            ):
                try:
                    data = worker.get_current_render_data()
                    render_data_list.append(data)
                except Exception as e:
                    print(
                        f"Error getting render data from worker {worker.worker_id}: {e}"
                    )
                    render_data_list.append(None)
                count += 1
            else:
                render_data_list.append(None)
                count += 1

        while count < max_envs:
            render_data_list.append(None)
            count += 1

        return render_data_list

    def start_all_workers(self):
        """Starts all initialized worker threads if they are not already running."""
        if self.is_any_worker_running():
            print("[AppWorkerManager] Workers already running.")
            return

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
        self.app.stop_event.clear()

        self.self_play_worker_threads = []
        for i, worker_instance in enumerate(self.app.initializer.self_play_workers):
            if worker_instance:
                if not worker_instance.is_alive():
                    try:
                        recreated_worker = SelfPlayWorker(
                            **worker_instance.get_init_args()
                        )
                        self.app.initializer.self_play_workers[i] = recreated_worker
                        worker_to_start = recreated_worker
                        print(f"  Recreated SelfPlayWorker-{i}.")
                    except Exception as e:
                        print(f"  ERROR recreating SelfPlayWorker-{i}: {e}")
                        continue
                else:
                    worker_to_start = worker_instance

                self.self_play_worker_threads.append(worker_to_start)
                worker_to_start.start()
                print(f"  SelfPlayWorker-{i} thread started.")
            else:
                print(
                    f"[AppWorkerManager] ERROR: SelfPlayWorker instance {i} is None during start."
                )

        self.training_worker_thread = self.app.initializer.training_worker
        if self.training_worker_thread:
            if not self.training_worker_thread.is_alive():
                try:
                    recreated_worker = TrainingWorker(
                        **self.training_worker_thread.get_init_args()
                    )
                    self.app.initializer.training_worker = recreated_worker
                    self.training_worker_thread = recreated_worker
                    print("  Recreated TrainingWorker.")
                except Exception as e:
                    print(f"  ERROR recreating TrainingWorker: {e}")
                    self.training_worker_thread = None

            if self.training_worker_thread:
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
        if not self.is_any_worker_running() and not any(
            w and not w.is_alive() for w in self.app.initializer.self_play_workers
        ):
            if (
                not self.app.initializer.self_play_workers
                and not self.app.initializer.training_worker
            ):
                print("[AppWorkerManager] No workers initialized to stop.")
                return
            elif not self.is_any_worker_running():
                print("[AppWorkerManager] No workers currently running to stop.")
                return

        print("[AppWorkerManager] Stopping ALL worker threads...")
        self.app.stop_event.set()

        threads_to_join: List[Tuple[str, threading.Thread]] = []

        for i, worker in enumerate(self.app.initializer.self_play_workers):
            if worker and worker.is_alive():
                threads_to_join.append((f"SelfPlayWorker-{i}", worker))

        if (
            self.app.initializer.training_worker
            and self.app.initializer.training_worker.is_alive()
        ):
            threads_to_join.append(
                ("TrainingWorker", self.app.initializer.training_worker)
            )

        start_join_time = time.time()
        if not threads_to_join:
            print("[AppWorkerManager] No active threads found to join.")
        else:
            for name, thread in threads_to_join:
                remaining_timeout = max(
                    0.1, join_timeout - (time.time() - start_join_time)
                )
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

        self.self_play_worker_threads = []
        self.training_worker_thread = None

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
