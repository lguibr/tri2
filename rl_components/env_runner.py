# rl_components/env_runner.py
import time
import logging
import threading
from config import Config
from rl_components.rollout_collector import RolloutCollector

logger = logging.getLogger(__name__)


class EnvRunner:
    """
    Runs the environment interaction loop in a separate thread.
    Collects rollouts using the RolloutCollector.
    """

    def __init__(
        self,
        collector: RolloutCollector,
        stop_event: threading.Event,
        pause_event: threading.Event,
        config: Config,
    ):
        self.collector = collector
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.config = config
        self.num_steps = config.ppo_config.NUM_STEPS
        self.num_envs = config.ppo_config.NUM_ENVS
        self.total_steps_collected = 0
        self._was_paused = True  # Assume starts paused as per main_pygame logic

    def run(self):
        """The main loop for the environment runner thread."""
        logger.info(f"[{self.__class__.__name__}] Starting environment runner loop.")
        loop_count = 0
        try:
            while not self.stop_event.is_set():
                # --- Pause Handling ---
                if self.pause_event.is_set():
                    if not self._was_paused:
                        logger.info(f"[{self.__class__.__name__}] Paused.")
                        self._was_paused = True
                    # Wait efficiently without busy-looping, checking stop_event periodically
                    self.pause_event.wait(timeout=0.1)
                    continue  # Re-check stop_event and pause_event

                # --- Resume Logging ---
                if self._was_paused:
                    logger.info(f"[{self.__class__.__name__}] Resumed.")
                    self._was_paused = False

                # --- Active Work ---
                logger.debug(
                    f"[{self.__class__.__name__} Loop {loop_count}] Running rollout collection step."
                )
                start_time = time.time()

                # Collect one step across all environments
                step_data = self.collector.collect_step()
                self.total_steps_collected += self.num_envs

                # Optional: Add a small sleep to prevent high CPU usage if collection is too fast
                # time.sleep(0.001)

                end_time = time.time()
                logger.debug(
                    f"[{self.__class__.__name__} Loop {loop_count}] Rollout step finished in {end_time - start_time:.4f}s."
                )
                loop_count += 1

        except Exception as e:
            logger.critical(
                f"[{self.__class__.__name__}] Exception in run loop: {e}", exc_info=True
            )
            self.stop_event.set()  # Signal other threads to stop on critical error
        finally:
            logger.info(
                f"[{self.__class__.__name__}] Exiting run loop. Total steps collected by this runner: {self.total_steps_collected}"
            )
