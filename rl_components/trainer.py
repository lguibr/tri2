# rl_components/trainer.py
import time
import logging
import threading
import torch
from config import Config
from rl_components.ppo_agent import PPOAgent
from rl_components.rollout_buffer import RolloutStorage
from rl_components.stats_recorder import StatsAggregator
from rl_components.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the PPO training process in a separate thread.
    """

    def __init__(
        self,
        agent: PPOAgent,
        storage: RolloutStorage,
        stats_aggregator: StatsAggregator,
        checkpoint_manager: CheckpointManager,
        stop_event: threading.Event,
        pause_event: threading.Event,
        config: Config,
        device: torch.device,
    ):
        self.agent = agent
        self.storage = storage
        self.stats_aggregator = stats_aggregator
        self.checkpoint_manager = checkpoint_manager
        self.stop_event = stop_event
        self.pause_event = pause_event  # Event to pause training
        self.config = config
        self.device = device

        self.ppo_config = config.ppo_config
        self.run_config = config.run_config
        self.checkpoint_config = config.checkpoint_config
        self.log_config = config.log_config

        self.total_timesteps = config.ppo_config.TOTAL_TIMESTEPS
        self.num_steps = config.ppo_config.NUM_STEPS
        self.num_envs = config.ppo_config.NUM_ENVS
        self.ppo_epochs = config.ppo_config.PPO_EPOCHS
        self.num_minibatches = config.ppo_config.NUM_MINIBATCHES
        self.batch_size = self.num_steps * self.num_envs
        self.minibatch_size = self.batch_size // self.num_minibatches

        self.current_step = (
            self.stats_aggregator.global_step
        )  # Start from loaded step if checkpoint was loaded
        self.current_rollout = (
            self.stats_aggregator.completed_rollouts
        )  # Start from loaded rollout count

        self._was_paused = True  # Assume starts paused as per main_pygame logic
        self._rollout_ready_logged = (
            False  # Track if "waiting for rollout" has been logged
        )

        logger.info(
            f"[{self.__class__.__name__}] Initialized. Starting at Step: {self.current_step}, Rollout: {self.current_rollout}"
        )
        logger.info(
            f"[{self.__class__.__name__}] Training Target: {self.total_timesteps} steps."
        )
        logger.info(
            f"[{self.__class__.__name__}] Batch Size: {self.batch_size}, Minibatch Size: {self.minibatch_size} ({self.num_minibatches} minibatches)"
        )

    def run(self):
        """The main loop for the trainer thread."""
        logger.info(f"[{self.__class__.__name__}] Starting training worker loop.")
        loop_count = 0
        try:
            while (
                self.current_step < self.total_timesteps
                and not self.stop_event.is_set()
            ):
                loop_start_time = time.time()
                # --- Pause Handling ---
                if self.pause_event.is_set():
                    if not self._was_paused:
                        logger.info(f"[{self.__class__.__name__}] Paused.")
                        self._was_paused = True
                    # Wait efficiently, checking stop_event periodically
                    self.pause_event.wait(timeout=0.1)
                    continue  # Re-check stop/pause events

                # --- Resume Logging ---
                if self._was_paused:
                    logger.info(f"[{self.__class__.__name__}] Resumed.")
                    self._was_paused = False
                    self._rollout_ready_logged = False  # Reset log flag on resume

                # --- Active Work ---
                logger.debug(
                    f"[{self.__class__.__name__} Loop {loop_count}] Checking for completed rollout..."
                )

                # Wait for the storage to be ready (filled by EnvRunner)
                if not self.storage.ready_for_update():
                    if not self._rollout_ready_logged:
                        logger.debug(
                            f"[{self.__class__.__name__} Loop {loop_count}] Waiting for rollout data (Current: {self.storage.step}/{self.num_steps})..."
                        )
                        self._rollout_ready_logged = True
                    time.sleep(0.01)  # Avoid busy-waiting
                    continue

                # --- Rollout Ready ---
                logger.debug(
                    f"[{self.__class__.__name__} Loop {loop_count}] Rollout data ready. Starting training update."
                )
                self._rollout_ready_logged = False  # Reset log flag
                train_start_time = time.time()

                # Perform PPO update
                update_metrics = self.agent.update(self.storage)

                train_end_time = time.time()
                logger.debug(
                    f"[{self.__class__.__name__} Loop {loop_count}] PPO update finished in {train_end_time - train_start_time:.4f}s."
                )

                # Update global step count and rollout count
                steps_in_rollout = self.num_steps * self.num_envs
                self.current_step += steps_in_rollout
                self.current_rollout += 1

                # Log metrics via StatsAggregator
                self.stats_aggregator.record_training_metrics(
                    update_metrics, self.current_step, self.current_rollout
                )

                # Checkpointing
                if (
                    self.current_rollout % self.checkpoint_config.SAVE_FREQ_ROLLOUTS
                    == 0
                ):
                    logger.info(
                        f"[{self.__class__.__name__}] Reached checkpoint interval (Rollout {self.current_rollout}). Saving checkpoint..."
                    )
                    save_start_time = time.time()
                    self.checkpoint_manager.save_checkpoint(
                        global_step=self.current_step,
                        completed_rollouts=self.current_rollout,
                    )
                    save_end_time = time.time()
                    logger.info(
                        f"[{self.__class__.__name__}] Checkpoint saved in {save_end_time - save_start_time:.3f}s."
                    )

                # Prepare storage for the next rollout
                self.storage.after_update()
                logger.debug(
                    f"[{self.__class__.__name__} Loop {loop_count}] Storage reset for next rollout."
                )

                loop_end_time = time.time()
                logger.debug(
                    f"[{self.__class__.__name__} Loop {loop_count}] Full training loop iteration took {loop_end_time - loop_start_time:.4f}s."
                )
                loop_count += 1

        except Exception as e:
            logger.critical(
                f"[{self.__class__.__name__}] Exception in run loop: {e}", exc_info=True
            )
            self.stop_event.set()  # Signal other threads to stop
        finally:
            if self.current_step >= self.total_timesteps:
                logger.info(
                    f"[{self.__class__.__name__}] Reached target timesteps ({self.total_timesteps})."
                )
            if self.stop_event.is_set():
                logger.info(f"[{self.__class__.__name__}] Stop event received.")

            # Attempt a final save if configured and stopping normally
            if (
                self.checkpoint_config.SAVE_ON_EXIT and not self.pause_event.is_set()
            ):  # Don't save if just paused
                logger.info(
                    f"[{self.__class__.__name__}] Attempting final checkpoint save on exit..."
                )
                self.checkpoint_manager.save_checkpoint(
                    global_step=self.current_step,
                    completed_rollouts=self.current_rollout,
                    is_final=True,
                )

            logger.info(
                f"[{self.__class__.__name__}] Exiting run loop. Final Step: {self.current_step}, Final Rollout: {self.current_rollout}"
            )
