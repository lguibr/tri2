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
import multiprocessing as mp
import ray
import asyncio

from config import TrainConfig
from utils.types import StateType, ActionType

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import _LRScheduler
    from ray.util.queue import Queue as RayQueue

    AgentPredictorHandle = ray.actor.ActorHandle
    StatsAggregatorHandle = ray.actor.ActorHandle

ExperienceData = Tuple[StateType, Dict[ActionType, float], float]
ProcessedExperienceBatch = List[ExperienceData]
logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class TrainingWorker:
    """Samples experience and trains the neural network (Ray Actor)."""

    def __init__(
        self,
        agent_predictor: "AgentPredictorHandle",
        optimizer_cls: type,
        optimizer_kwargs: dict,
        scheduler_cls: Optional[type],
        scheduler_kwargs: Optional[dict],
        experience_queue: "RayQueue",
        stats_aggregator: "StatsAggregatorHandle",
        train_config: TrainConfig,
    ):
        self.agent_predictor = agent_predictor
        self.experience_queue = experience_queue
        self.stats_aggregator = stats_aggregator
        self.train_config = train_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_prefix = "[TrainingWorker]"

        from config import EnvConfig, ModelConfig
        from agent.alphazero_net import AlphaZeroNet

        self.local_agent = AlphaZeroNet(EnvConfig(), ModelConfig.Network()).to(
            self.device
        )

        self.optimizer = optimizer_cls(
            self.local_agent.parameters(), **optimizer_kwargs
        )
        self.scheduler = None
        if scheduler_cls and scheduler_kwargs:
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_kwargs)

        self.steps_done = 0
        self._stop_requested = False

        logger.info(
            f"{self.log_prefix} Initialized as Ray Actor. Device: {self.device}."
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

    async def _get_initial_state(self):
        """Asynchronously fetches initial weights and global step."""
        try:
            weights_ref = self.agent_predictor.get_weights.remote()
            step_ref = self.stats_aggregator.get_current_global_step.remote()
            initial_weights, initial_step = await asyncio.gather(weights_ref, step_ref)
            self.local_agent.load_state_dict(initial_weights)
            self.steps_done = initial_step
            logger.info(
                f"{self.log_prefix} Initial weights loaded. Initial global step: {self.steps_done}"
            )
        except Exception as e:
            logger.error(
                f"{self.log_prefix} Failed to get initial state: {e}. Starting from scratch."
            )
            self.steps_done = 0

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
                        if key == "death_mask" and key not in states:
                            logger.warning(
                                f"{self.log_prefix} State dict missing expected 'death_mask' key initially."
                            )
                        elif key != "death_mask":
                            logger.warning(
                                f"{self.log_prefix} Skipping invalid item in batch (invalid state key/value: {key}, type: {type(value)})."
                            )
                            valid_state = False
                            break
                        else:
                            temp_state[key] = value
                if not valid_state:
                    continue

                policy_array = np.zeros(
                    self.local_agent.env_cfg.ACTION_DIM, dtype=np.float32
                )
                policy_sum = 0.0
                valid_policy_entries = 0
                for action, prob in policy_dict.items():
                    if (
                        isinstance(action, int)
                        and 0 <= action < self.local_agent.env_cfg.ACTION_DIM
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

    async def _perform_training_step(
        self, batch_data: ProcessedExperienceBatch
    ) -> Optional[Dict[str, float]]:
        """Performs a single training step (async for stats)."""
        prep_start = time.monotonic()
        prepared_batch = self._prepare_batch(batch_data)
        prep_duration = time.monotonic() - prep_start
        if prepared_batch is None:
            logger.warning(
                f"{self.log_prefix} Failed to prepare batch (took {prep_duration:.4f}s). Skipping step."
            )
            return None
        batch_states, batch_policy_targets, batch_value_targets = prepared_batch
        logger.debug(f"{self.log_prefix} Batch preparation took {prep_duration:.4f}s.")

        try:
            step_start_time = time.monotonic()
            self.local_agent.train()
            self.optimizer.zero_grad()
            policy_logits, value_preds = self.local_agent(batch_states)

            if (
                policy_logits.shape[0] != batch_policy_targets.shape[0]
                or value_preds.shape[0] != batch_value_targets.shape[0]
            ):
                logger.error(
                    f"{self.log_prefix} Batch size mismatch after forward pass! Skipping. Logits: {policy_logits.shape}, Targets: {batch_policy_targets.shape}, Values: {value_preds.shape}, Targets: {batch_value_targets.shape}"
                )
                return None
            if policy_logits.shape[1] != batch_policy_targets.shape[1]:
                logger.error(
                    f"{self.log_prefix} Action dim mismatch after forward pass! Skipping. Logits: {policy_logits.shape[1]}, Targets: {batch_policy_targets.shape[1]}"
                )
                return None

            log_policy_preds = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(
                batch_policy_targets * log_policy_preds, dim=1
            ).mean()
            value_loss = F.mse_loss(value_preds, batch_value_targets)
            total_loss = (
                self.train_config.POLICY_LOSS_WEIGHT * policy_loss
                + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            )

            total_loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            step_duration = time.monotonic() - step_start_time
            logger.debug(f"{self.log_prefix} Training step took {step_duration:.4f}s.")
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.steps_done % 10 == 0:
                weights = self.local_agent.state_dict()
                self.agent_predictor.set_weights.remote(weights)
                logger.debug(
                    f"{self.log_prefix} Sent updated weights to AgentPredictor."
                )

            return {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "update_time": step_duration,
                "lr": current_lr,
            }
        except Exception as e:
            logger.critical(
                f"{self.log_prefix} CRITICAL ERROR during training step {self.steps_done}: {e}",
                exc_info=True,
            )
            return None

    async def run_loop(self):
        """Main training loop (async)."""
        logger.info(f"{self.log_prefix} Starting run loop.")
        await self._get_initial_state()
        logger.info(
            f"{self.log_prefix} Starting training from Global Step: {self.steps_done}"
        )

        last_buffer_update_time = 0
        buffer_update_interval = 1.0

        while not self._stop_requested:
            # Call qsize() directly, assume it returns int
            buffer_size = self.experience_queue.qsize()

            if buffer_size < self.train_config.MIN_BUFFER_SIZE_TO_TRAIN:
                if time.time() - last_buffer_update_time > buffer_update_interval:
                    self.stats_aggregator.record_step.remote(
                        {"buffer_size": buffer_size}
                    )
                    last_buffer_update_time = time.time()
                    logger.info(
                        f"{self.log_prefix} Waiting for buffer... Size: {buffer_size}/{self.train_config.MIN_BUFFER_SIZE_TO_TRAIN}"
                    )
                await asyncio.sleep(0.1)
                continue

            logger.debug(
                f"{self.log_prefix} Starting training iteration. Buffer size: {buffer_size}"
            )
            steps_this_iter, iter_policy_loss, iter_value_loss = 0, 0.0, 0.0
            iter_start_time = time.monotonic()

            for _ in range(self.train_config.NUM_TRAINING_STEPS_PER_ITER):
                if self._stop_requested:
                    break

                batch_data_list: Optional[ProcessedExperienceBatch] = None
                try:
                    q_get_start = time.monotonic()
                    # Use get_async for getting data
                    batch_data_list = await self.experience_queue.get_async(timeout=1.0)
                    q_get_duration = time.monotonic() - q_get_start
                    logger.debug(
                        f"{self.log_prefix} Queue get (batch size {len(batch_data_list) if batch_data_list else 0}) took {q_get_duration:.4f}s."
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"{self.log_prefix} Queue empty during training iteration, waiting..."
                    )
                    await asyncio.sleep(0.1)
                    break
                except Exception as e:
                    logger.error(
                        f"{self.log_prefix} Error getting data from queue: {e}",
                        exc_info=True,
                    )
                    break

                if not batch_data_list:
                    continue

                actual_batch_size = min(
                    len(batch_data_list), self.train_config.BATCH_SIZE
                )
                if actual_batch_size < 1:
                    continue

                step_result = await self._perform_training_step(
                    batch_data_list[:actual_batch_size]
                )
                if step_result is None:
                    logger.warning(
                        f"{self.log_prefix} Training step failed, ending iteration early."
                    )
                    break

                self.steps_done += 1
                steps_this_iter += 1
                iter_policy_loss += step_result["policy_loss"]
                iter_value_loss += step_result["value_loss"]

                # Call qsize() directly, assume it returns int
                current_buffer_size = self.experience_queue.qsize()
                step_stats = {
                    "global_step": self.steps_done,
                    "buffer_size": current_buffer_size,
                    "training_steps_performed": self.steps_done,
                    **step_result,
                }
                self.stats_aggregator.record_step.remote(step_stats)

            iter_duration = time.monotonic() - iter_start_time
            if steps_this_iter > 0:
                avg_p = iter_policy_loss / steps_this_iter
                avg_v = iter_value_loss / steps_this_iter
                logger.info(
                    f"{self.log_prefix} Iteration complete. Steps: {steps_this_iter}, Duration: {iter_duration:.2f}s, Avg P.Loss: {avg_p:.4f}, Avg V.Loss: {avg_v:.4f}"
                )
            else:
                logger.info(
                    f"{self.log_prefix} Iteration finished with 0 steps performed (Duration: {iter_duration:.2f}s)."
                )

            await asyncio.sleep(0.01)

        logger.info(f"{self.log_prefix} Run loop finished.")

    def stop(self):
        """Signals the actor to stop gracefully."""
        logger.info(f"{self.log_prefix} Stop requested.")
        self._stop_requested = True

    def health_check(self):
        """Ray health check method."""
        return "OK"
