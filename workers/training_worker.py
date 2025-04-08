# File: workers/training_worker.py
import threading
import time
import queue
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
import logging

from config import TrainConfig
from utils.types import StateType, ActionType

if TYPE_CHECKING:
    from agent.alphazero_net import AlphaZeroNet
    from stats.aggregator import StatsAggregator
    from torch.optim.lr_scheduler import _LRScheduler  # Import for type hint

ExperienceData = Tuple[StateType, Dict[ActionType, float], float]
ProcessedExperienceBatch = List[ExperienceData]
logger = logging.getLogger(__name__)


class TrainingWorker(threading.Thread):
    """Samples experience and trains the neural network."""

    def __init__(
        self,
        agent: "AlphaZeroNet",
        optimizer: optim.Optimizer,
        scheduler: Optional["_LRScheduler"],  # Add scheduler parameter
        experience_queue: queue.Queue,
        stats_aggregator: "StatsAggregator",
        stop_event: threading.Event,
        train_config: TrainConfig,
        device: torch.device,
    ):
        super().__init__(daemon=True, name="TrainingWorker")
        self.agent = agent
        self.optimizer = optimizer
        self.scheduler = scheduler  # Store scheduler
        self.experience_queue = experience_queue
        self.stats_aggregator = stats_aggregator
        self.stop_event = stop_event
        self.train_config = train_config
        self.device = device
        self.log_prefix = "[TrainingWorker]"
        # Initialize steps_done from aggregator *after* potential checkpoint load
        self.steps_done = self.stats_aggregator.storage.current_global_step
        logger.info(
            f"{self.log_prefix} Initialized. Device: {self.device}. Initial Step: {self.steps_done}"
        )
        logger.info(
            f"{self.log_prefix} Config: Batch={self.train_config.BATCH_SIZE}, LR={self.train_config.LEARNING_RATE}, MinBuffer={self.train_config.MIN_BUFFER_SIZE_TO_TRAIN}"
        )
        if self.scheduler:
            logger.info(
                f"{self.log_prefix} LR Scheduler Type: {type(self.scheduler).__name__}"
            )
        else:
            logger.info(f"{self.log_prefix} LR Scheduler: DISABLED")

    def get_init_args(self) -> Dict[str, Any]:
        """Returns arguments needed to re-initialize the thread."""
        return {
            "agent": self.agent,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,  # Include scheduler
            "experience_queue": self.experience_queue,
            "stats_aggregator": self.stats_aggregator,
            "stop_event": self.stop_event,
            "train_config": self.train_config,
            "device": self.device,
        }

    def _prepare_batch(
        self, batch_data: ProcessedExperienceBatch
    ) -> Optional[Tuple[StateType, torch.Tensor, torch.Tensor]]:
        """Converts a list of experience tuples into batched tensors."""
        try:
            if (
                not batch_data
                or not isinstance(batch_data[0], tuple)
                or len(batch_data[0]) != 3
            ):
                logger.error(f"{self.log_prefix} Invalid batch data structure (outer).")
                return None
            if not isinstance(batch_data[0][0], dict):
                logger.error(
                    f"{self.log_prefix} Invalid batch data structure (state dict)."
                )
                return None

            states = {key: [] for key in batch_data[0][0].keys()}
            policy_targets, value_targets = [], []
            valid_items = 0

            for item in batch_data:
                if not isinstance(item, tuple) or len(item) != 3:
                    logger.warning(
                        f"{self.log_prefix} Skipping invalid item in batch (wrong structure)."
                    )
                    continue
                state_dict, policy_dict, outcome = item
                if not isinstance(state_dict, dict) or not isinstance(
                    policy_dict, dict
                ):
                    logger.warning(
                        f"{self.log_prefix} Skipping invalid item in batch (wrong inner types)."
                    )
                    continue
                if not (isinstance(outcome, (float, int)) and np.isfinite(outcome)):
                    logger.warning(
                        f"{self.log_prefix} Skipping invalid item in batch (invalid outcome: {outcome})."
                    )
                    continue

                temp_state, valid_state = {}, True
                for key, value in state_dict.items():
                    if key in states and isinstance(value, np.ndarray):
                        temp_state[key] = value
                    else:
                        logger.warning(
                            f"{self.log_prefix} Skipping invalid item in batch (invalid state key/value: {key}, type: {type(value)})."
                        )
                        valid_state = False
                        break
                if not valid_state:
                    continue

                policy_array = np.zeros(self.agent.env_cfg.ACTION_DIM, dtype=np.float32)
                policy_sum = 0.0
                valid_policy_entries = 0
                for action, prob in policy_dict.items():
                    if (
                        isinstance(action, int)
                        and 0 <= action < self.agent.env_cfg.ACTION_DIM
                        and isinstance(prob, (float, int))
                        and np.isfinite(prob)
                        and prob >= 0
                    ):
                        policy_array[action] = prob
                        policy_sum += prob
                        valid_policy_entries += 1

                if valid_policy_entries > 0 and not np.isclose(
                    policy_sum, 1.0, atol=1e-4
                ):
                    logger.warning(
                        f"{self.log_prefix} Policy target sum is {policy_sum:.4f}, expected ~1.0. Using as is."
                    )

                for key in states.keys():
                    states[key].append(temp_state[key])
                policy_targets.append(policy_array)
                value_targets.append(outcome)
                valid_items += 1

            if valid_items == 0:
                logger.error(f"{self.log_prefix} No valid items found in the batch.")
                return None

            batched_states = {
                k: torch.from_numpy(np.stack(v)).to(self.device)
                for k, v in states.items()
            }
            batched_policy = torch.from_numpy(np.stack(policy_targets)).to(self.device)
            batched_value = (
                torch.tensor(value_targets, dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device)
            )

            if (
                batched_policy.shape[0] != valid_items
                or batched_value.shape[0] != valid_items
            ):
                logger.error(
                    f"{self.log_prefix} Shape mismatch after stacking tensors."
                )
                return None

            return batched_states, batched_policy, batched_value

        except Exception as e:
            logger.error(f"{self.log_prefix} Error preparing batch: {e}", exc_info=True)
            return None

    def _perform_training_step(
        self, batch_data: ProcessedExperienceBatch
    ) -> Optional[Dict[str, float]]:
        """Performs a single training step."""
        prep_start = time.monotonic()
        prepared_batch = self._prepare_batch(batch_data)
        prep_duration = time.monotonic() - prep_start
        if prepared_batch is None:
            logger.warning(
                f"{self.log_prefix} Failed to prepare batch (took {prep_duration:.4f}s). Skipping step."
            )
            return None
        batch_states, batch_policy_targets, batch_value_targets = prepared_batch
        logger.info(  # Changed to debug
            f"{self.log_prefix} Batch preparation took {prep_duration:.4f}s."
        )

        try:
            step_start_time = time.monotonic()
            self.agent.train()
            self.optimizer.zero_grad()

            policy_logits, value_preds = self.agent(batch_states)

            if (
                policy_logits.shape[0] != batch_policy_targets.shape[0]
                or value_preds.shape[0] != batch_value_targets.shape[0]
            ):
                logger.error(
                    f"{self.log_prefix} Batch size mismatch after forward pass! Skipping. "
                    f"Logits: {policy_logits.shape}, Targets: {batch_policy_targets.shape}, "
                    f"Values: {value_preds.shape}, Targets: {batch_value_targets.shape}"
                )
                return None
            if policy_logits.shape[1] != batch_policy_targets.shape[1]:
                logger.error(
                    f"{self.log_prefix} Action dim mismatch after forward pass! Skipping. "
                    f"Logits: {policy_logits.shape[1]}, Targets: {batch_policy_targets.shape[1]}"
                )
                return None

            # Policy Loss: Cross-entropy between predicted policy logits and MCTS policy target
            log_policy_preds = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(
                batch_policy_targets * log_policy_preds, dim=1
            ).mean()

            # Value Loss: Mean Squared Error between predicted value and game outcome
            value_loss = F.mse_loss(value_preds, batch_value_targets)

            # Total Loss
            total_loss = (
                self.train_config.POLICY_LOSS_WEIGHT * policy_loss
                + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            )

            total_loss.backward()
            self.optimizer.step()

            # Step the scheduler if it exists
            if self.scheduler:
                self.scheduler.step()

            step_duration = time.monotonic() - step_start_time
            logger.info(  # Changed to debug
                f"{self.log_prefix} Training step took {step_duration:.4f}s."
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            return {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "update_time": step_duration,
                "lr": current_lr,  # Include learning rate
            }
        except Exception as e:
            logger.critical(
                f"{self.log_prefix} CRITICAL ERROR during training step {self.steps_done}: {e}",
                exc_info=True,
            )
            return None

    def run(self):
        """Main training loop."""
        logger.info(f"{self.log_prefix} Starting run loop.")
        # Ensure steps_done is correctly initialized *after* potential checkpoint load
        self.steps_done = self.stats_aggregator.storage.current_global_step
        logger.info(
            f"{self.log_prefix} Starting training from Global Step: {self.steps_done}"
        )

        last_buffer_update_time = 0
        buffer_update_interval = 1.0  # Log buffer size roughly every second if waiting

        while not self.stop_event.is_set():
            buffer_size = self.experience_queue.qsize()

            # Wait if buffer is not large enough
            if buffer_size < self.train_config.MIN_BUFFER_SIZE_TO_TRAIN:
                if time.time() - last_buffer_update_time > buffer_update_interval:
                    # Record buffer size periodically while waiting
                    self.stats_aggregator.record_step({"buffer_size": buffer_size})
                    last_buffer_update_time = time.time()
                    logger.info(
                        f"{self.log_prefix} Waiting for buffer... Size: {buffer_size}/{self.train_config.MIN_BUFFER_SIZE_TO_TRAIN}"
                    )
                time.sleep(0.1)  # Short sleep while waiting
                continue

            # Buffer is ready, start training iteration
            logger.info(  # Changed to debug
                f"{self.log_prefix} Starting training iteration. Buffer size: {buffer_size}"
            )
            steps_this_iter, iter_policy_loss, iter_value_loss = 0, 0.0, 0.0
            iter_start_time = time.monotonic()

            for _ in range(self.train_config.NUM_TRAINING_STEPS_PER_ITER):
                if self.stop_event.is_set():
                    break  # Exit inner loop if stop event is set

                batch_data_list: Optional[ProcessedExperienceBatch] = None
                try:
                    q_get_start = time.monotonic()
                    # Get a batch of experience from the queue
                    batch_data_list = self.experience_queue.get(timeout=1.0)
                    q_get_duration = time.monotonic() - q_get_start
                    logger.info(  # Changed to debug
                        f"{self.log_prefix} Queue get (batch size {len(batch_data_list) if batch_data_list else 0}) took {q_get_duration:.4f}s."
                    )
                except queue.Empty:
                    logger.warning(
                        f"{self.log_prefix} Queue empty during training iteration, waiting..."
                    )
                    time.sleep(0.1)
                    break  # Exit inner loop, will check buffer size again
                except Exception as e:
                    logger.error(
                        f"{self.log_prefix} Error getting data from queue: {e}",
                        exc_info=True,
                    )
                    break  # Exit inner loop on error

                if not batch_data_list:
                    continue  # Skip if queue returned None or empty list

                # Use the actual batch size received, up to the configured max
                actual_batch_size = min(
                    len(batch_data_list), self.train_config.BATCH_SIZE
                )
                if actual_batch_size < 1:  # Should not happen if queue get succeeded
                    continue

                # Perform one training step
                step_result = self._perform_training_step(
                    batch_data_list[:actual_batch_size]  # Slice to actual batch size
                )
                if step_result is None:
                    logger.warning(
                        f"{self.log_prefix} Training step failed, ending iteration early."
                    )
                    break  # Exit inner loop if step failed

                # Update counters and aggregate losses
                self.steps_done += 1
                steps_this_iter += 1
                iter_policy_loss += step_result["policy_loss"]
                iter_value_loss += step_result["value_loss"]

                # Record step statistics
                step_stats = {
                    "global_step": self.steps_done,  # Pass the incremented step
                    "buffer_size": self.experience_queue.qsize(),  # Get current size
                    "training_steps_performed": self.steps_done,  # Track total steps
                    **step_result,  # Include losses, time, lr from step result
                }
                self.stats_aggregator.record_step(step_stats)

            # Log iteration summary
            iter_duration = time.monotonic() - iter_start_time
            if steps_this_iter > 0:
                avg_p = iter_policy_loss / steps_this_iter
                avg_v = iter_value_loss / steps_this_iter
                logger.info(
                    f"{self.log_prefix} Iteration complete. Steps: {steps_this_iter}, "
                    f"Duration: {iter_duration:.2f}s, Avg P.Loss: {avg_p:.4f}, Avg V.Loss: {avg_v:.4f}"
                )
            else:
                logger.info(
                    f"{self.log_prefix} Iteration finished with 0 steps performed (Duration: {iter_duration:.2f}s)."
                )

            # Small sleep to yield control if needed
            time.sleep(0.01)

        logger.info(f"{self.log_prefix} Run loop finished.")
