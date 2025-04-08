# File: stats/tb_scalar_logger.py
# File: stats/tb_scalar_logger.py
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from typing import Dict, Any, Optional

from .aggregator import StatsAggregator


class TBScalarLogger:
    """Handles logging scalar values to TensorBoard."""

    def __init__(self, writer: Optional[SummaryWriter], lock: threading.Lock):
        self.writer = writer
        self._lock = lock

    def log_episode_scalars(
        self,
        g_step: int,
        episode_score: float,  # This is the game outcome (-1, 0, 1)
        episode_length: int,
        episode_num: int,
        game_score: Optional[int],
        triangles_cleared: Optional[int],
        update_info: Dict[str, Any],
        aggregator: StatsAggregator,
    ):
        """Logs scalars related to a completed episode."""
        if not self.writer:
            return
        with self._lock:
            try:
                # Log game outcome instead of RL score
                self.writer.add_scalar("Episode/Outcome", episode_score, g_step)
                self.writer.add_scalar("Episode/Length", episode_length, g_step)
                if game_score is not None:
                    self.writer.add_scalar("Episode/Game Score", game_score, g_step)
                if triangles_cleared is not None:
                    self.writer.add_scalar(
                        "Episode/Triangles Cleared", triangles_cleared, g_step
                    )
                self.writer.add_scalar("Progress/Total Episodes", episode_num, g_step)

                # RL Score removed
                # if update_info.get("new_best_rl"):
                #     self.writer.add_scalar(
                #         "Best Performance/RL Score",
                #         aggregator.storage.best_score,
                #         g_step,
                #     )
                if update_info.get("new_best_game"):
                    self.writer.add_scalar(
                        "Best Performance/Game Score",
                        aggregator.storage.best_game_score,
                        g_step,
                    )
            except Exception as e:
                print(f"Error writing episode scalars to TensorBoard: {e}")

    def log_step_scalars(
        self,
        g_step: int,
        step_data: Dict[str, Any],
        update_info: Dict[str, Any],
        aggregator: StatsAggregator,
    ):
        """Logs scalars related to a training/environment step."""
        if not self.writer:
            return
        with self._lock:
            try:
                scalar_map = {
                    "policy_loss": "Loss/Policy Loss",
                    "value_loss": "Loss/Value Loss",
                    # "avg_max_q": "Debug/Avg Max Q", # Removed PPO/DQN specific stat
                    # "beta": "Debug/Beta", # Removed PER specific stat
                    "buffer_size": "Debug/Buffer Size",
                    "lr": "Train/Learning Rate",
                    # "epsilon": "Train/Epsilon", # Removed Epsilon-greedy specific stat
                    "update_time": "Performance/Update Time",
                    # "step_time": "Performance/Total Step Time", # Less relevant for AZ training step
                    "cpu_usage": "Resource/CPU Usage (%)",
                    "memory_usage": "Resource/Memory Usage (%)",
                    "gpu_memory_usage_percent": "Resource/GPU Memory Usage (%)",
                }
                for key, tag in scalar_map.items():
                    if key in step_data and step_data[key] is not None:
                        self.writer.add_scalar(tag, step_data[key], g_step)

                if update_info.get("new_best_value_loss"):
                    self.writer.add_scalar(
                        "Best Performance/Value Loss",
                        aggregator.storage.best_value_loss,
                        g_step,
                    )
                if update_info.get("new_best_policy_loss"):
                    self.writer.add_scalar(
                        "Best Performance/Policy Loss",
                        aggregator.storage.best_policy_loss,
                        g_step,
                    )
            except Exception as e:
                print(f"Error writing step scalars to TensorBoard: {e}")
