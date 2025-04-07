# File: app_workers.py
import threading
import queue
import time
import traceback
from typing import TYPE_CHECKING

from workers import EnvironmentRunner, TrainingWorker

if TYPE_CHECKING:
    from main_pygame import MainApp


class AppWorkerManager:
    """Manages the creation, starting, and stopping of worker threads."""

    def __init__(self, app: "MainApp"):
        self.app = app
        self.env_runner_thread: Optional[EnvironmentRunner] = None
        self.training_worker_thread: Optional[TrainingWorker] = None

    def start_worker_threads(self):
        """Creates and starts the environment runner and training worker threads."""
        if (
            not self.app.initializer.rollout_collector
            or not self.app.initializer.agent
            or not self.app.initializer.stats_recorder
            or not self.app.initializer.checkpoint_manager
        ):
            print("ERROR: Cannot start workers, core components not initialized.")
            self.app.app_state = self.app.app_state.ERROR
            self.app.status = "Worker Init Failed"
            return

        print("Starting worker threads...")
        self.app.stop_event.clear()
        # Keep pause_event set initially, toggle run will clear it

        # Environment Runner Thread
        self.env_runner_thread = EnvironmentRunner(
            collector=self.app.initializer.rollout_collector,
            experience_queue=self.app.experience_queue,
            action_queue=None,  # Not used in this PPO setup
            stop_event=self.app.stop_event,
            pause_event=self.app.pause_event,
            num_steps_per_rollout=self.app.ppo_config.NUM_STEPS_PER_ROLLOUT,
            stats_recorder=self.app.initializer.stats_recorder,
        )
        self.env_runner_thread.start()

        # Training Worker Thread
        self.training_worker_thread = TrainingWorker(
            agent=self.app.initializer.agent,
            experience_queue=self.app.experience_queue,
            stop_event=self.app.stop_event,
            pause_event=self.app.pause_event,
            stats_recorder=self.app.initializer.stats_recorder,
            ppo_config=self.app.ppo_config,
            device=self.app.device,
            checkpoint_manager=self.app.initializer.checkpoint_manager,
        )
        self.training_worker_thread.start()
        print("Worker threads started.")

    def stop_worker_threads(self):
        """Signals worker threads to stop and waits for them to join."""
        if self.app.stop_event.is_set():
            return

        print("Stopping worker threads...")
        self.app.stop_event.set()
        self.app.pause_event.clear()  # Ensure threads aren't stuck paused

        join_timeout = 5.0
        if self.env_runner_thread and self.env_runner_thread.is_alive():
            self.env_runner_thread.join(timeout=join_timeout)
            if self.env_runner_thread.is_alive():
                print("EnvRunner thread did not join cleanly.")
        if self.training_worker_thread and self.training_worker_thread.is_alive():
            self.training_worker_thread.join(timeout=join_timeout)
            if self.training_worker_thread.is_alive():
                print("TrainingWorker thread did not join cleanly.")

        while not self.app.experience_queue.empty():
            try:
                self.app.experience_queue.get_nowait()
            except queue.Empty:
                break

        self.env_runner_thread = None
        self.training_worker_thread = None
        print("Worker threads stopped.")
