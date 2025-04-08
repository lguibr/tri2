# File: workers/training_worker.py
import threading
import time
import queue
import traceback
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional

from config import TrainConfig
from utils.types import StateType, ActionType
from utils.helpers import ensure_numpy

if TYPE_CHECKING:
    from agent.alphazero_net import AlphaZeroNet
    from stats.aggregator import StatsAggregator

ExperienceData = Tuple[StateType, Dict[ActionType, float], float]


class TrainingWorker(threading.Thread):
    """
    Worker thread that samples experience from a queue and trains the neural network.
    """

    def __init__(
        self,
        agent: "AlphaZeroNet",
        optimizer: optim.Optimizer,
        experience_queue: queue.Queue,
        stats_aggregator: "StatsAggregator",
        stop_event: threading.Event,
        train_config: TrainConfig,
        device: torch.device,
    ):
        super().__init__(daemon=True, name="TrainingWorker")
        self.agent = agent
        self.optimizer = optimizer
        self.experience_queue = experience_queue
        self.stats_aggregator = stats_aggregator
        self.stop_event = stop_event
        self.train_config = train_config
        self.device = device

        self.steps_done = (
            0  # Internal counter for training steps performed by this worker
        )
        print(f"[TrainingWorker] Initialized. Device: {self.device}")
        print(
            f"[TrainingWorker] Config: Batch={self.train_config.BATCH_SIZE}, LR={self.train_config.LEARNING_RATE}, MinBuffer={self.train_config.MIN_BUFFER_SIZE_TO_TRAIN}"
        )

    def get_init_args(self) -> Dict[str, Any]:
        """Returns arguments needed to re-initialize the thread."""
        return {
            "agent": self.agent,
            "optimizer": self.optimizer,
            "experience_queue": self.experience_queue,
            "stats_aggregator": self.stats_aggregator,
            "stop_event": self.stop_event,
            "train_config": self.train_config,
            "device": self.device,
        }

    def _prepare_batch(
        self, batch_data: List[ExperienceData]
    ) -> Optional[Tuple[StateType, torch.Tensor, torch.Tensor]]:
        """Converts a list of experience tuples into batched tensors."""
        try:
            states = {key: [] for key in batch_data[0][0].keys()}
            policy_targets = []
            value_targets = []

            for state_dict, policy_dict, outcome in batch_data:
                for key, value in state_dict.items():
                    states[key].append(value)
                policy_array = np.zeros(self.agent.env_cfg.ACTION_DIM, dtype=np.float32)
                policy_sum = sum(policy_dict.values())
                if policy_sum > 1e-6:
                    for action, prob in policy_dict.items():
                        if 0 <= action < self.agent.env_cfg.ACTION_DIM:
                            policy_array[action] = prob / policy_sum
                policy_targets.append(policy_array)
                value_targets.append(outcome)

            batched_states = {
                key: torch.from_numpy(np.stack(value_list)).to(self.device)
                for key, value_list in states.items()
            }
            batched_policy_targets = torch.from_numpy(np.stack(policy_targets)).to(
                self.device
            )
            batched_value_targets = (
                torch.tensor(value_targets, dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device)
            )

            return batched_states, batched_policy_targets, batched_value_targets

        except Exception as e:
            print(f"[TrainingWorker] Error preparing batch: {e}")
            traceback.print_exc()
            return None

    def run(self):
        """Main training loop."""
        print(f"[TrainingWorker] Starting run loop.")
        last_log_time = time.time()
        # Initialize internal step counter from aggregator's global step
        self.steps_done = self.stats_aggregator.storage.current_global_step

        while not self.stop_event.is_set():
            buffer_size = self.experience_queue.qsize()

            if buffer_size < self.train_config.MIN_BUFFER_SIZE_TO_TRAIN:
                time.sleep(1.0)
                continue

            steps_this_iter = 0
            iter_policy_loss = 0.0
            iter_value_loss = 0.0

            for _ in range(self.train_config.NUM_TRAINING_STEPS_PER_ITER):
                if self.stop_event.is_set():
                    break

                batch_data: List[ExperienceData] = []
                try:
                    actual_batch_size = min(
                        self.train_config.BATCH_SIZE, self.experience_queue.qsize()
                    )
                    if actual_batch_size < self.train_config.BATCH_SIZE // 2:
                        break

                    for _ in range(actual_batch_size):
                        experience = self.experience_queue.get(timeout=0.1)
                        batch_data.append(experience)
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"[TrainingWorker] Error getting data from queue: {e}")
                    break

                if not batch_data:
                    continue

                prepared_batch = self._prepare_batch(batch_data)
                if prepared_batch is None:
                    continue

                batch_states, batch_policy_targets, batch_value_targets = prepared_batch

                try:
                    start_step_time = time.time()
                    self.agent.train()
                    self.optimizer.zero_grad()

                    policy_logits, value_preds = self.agent(batch_states)

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

                    self.steps_done += 1  # Increment internal counter
                    steps_this_iter += 1
                    iter_policy_loss += policy_loss.item()
                    iter_value_loss += value_loss.item()
                    step_duration = time.time() - start_step_time

                    current_lr = self.optimizer.param_groups[0]["lr"]
                    step_stats = {
                        "global_step": self.steps_done,  # Report the incremented step count
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "lr": current_lr,
                        "buffer_size": self.experience_queue.qsize(),
                        "update_time": step_duration,
                    }
                    # This call will update the aggregator's internal global_step
                    self.stats_aggregator.record_step(step_stats)

                except Exception as e:
                    print(
                        f"[TrainingWorker] CRITICAL ERROR during training step {self.steps_done}: {e}"
                    )
                    traceback.print_exc()
                    time.sleep(5)
                    break

            time.sleep(0.01)

        print(f"[TrainingWorker] Run loop finished.")
