import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from typing import Dict, Any, Optional

from .aggregator import StatsAggregator  # For type hinting


class TBScalarLogger:
    """Handles logging scalar values to TensorBoard."""

    def __init__(self, writer: Optional[SummaryWriter], lock: threading.Lock):
        self.writer = writer
        self._lock = lock

    def log_episode_scalars(
        self,
        g_step: int,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        game_score: Optional[int],
        triangles_cleared: Optional[int],
        update_info: Dict[str, Any],
        aggregator: StatsAggregator,  # Pass aggregator for best values
    ):
        """Logs scalars related to a completed episode."""
        if not self.writer:
            return
        with self._lock:
            try:
                self.writer.add_scalar("Episode/Score", episode_score, g_step)
                self.writer.add_scalar("Episode/Length", episode_length, g_step)
                if game_score is not None:
                    self.writer.add_scalar("Episode/Game Score", game_score, g_step)
                if triangles_cleared is not None:
                    self.writer.add_scalar(
                        "Episode/Triangles Cleared", triangles_cleared, g_step
                    )
                self.writer.add_scalar("Progress/Total Episodes", episode_num, g_step)

                # --- Corrected Access ---
                if update_info.get("new_best_rl"):
                    self.writer.add_scalar(
                        "Best Performance/RL Score",
                        aggregator.storage.best_score,
                        g_step,
                    )
                if update_info.get("new_best_game"):
                    self.writer.add_scalar(
                        "Best Performance/Game Score",
                        aggregator.storage.best_game_score,
                        g_step,
                    )
                # --- End Correction ---
            except Exception as e:
                print(f"Error writing episode scalars to TensorBoard: {e}")

    def log_step_scalars(
        self,
        g_step: int,
        step_data: Dict[str, Any],
        update_info: Dict[str, Any],
        aggregator: StatsAggregator,  # Pass aggregator for best values
    ):
        """Logs scalars related to a training/environment step."""
        if not self.writer:
            return
        with self._lock:
            try:
                scalar_map = {
                    "policy_loss": "Loss/Policy Loss",
                    "value_loss": "Loss/Value Loss",
                    "entropy": "Loss/Entropy",
                    "grad_norm": "Debug/Grad Norm",
                    "avg_max_q": "Debug/Avg Max Q",
                    "beta": "Debug/Beta",
                    "buffer_size": "Debug/Buffer Size",
                    "lr": "Train/Learning Rate",
                    "epsilon": "Train/Epsilon",
                    "sps_collection": "Performance/SPS Collection",  # Collection SPS
                    "update_steps_per_second": "Performance/SPS Update (Overall)",  # Overall SPS
                    "minibatch_update_sps": "Performance/SPS Update (Minibatch)",  # NEW: Minibatch SPS
                    "update_time": "Performance/Update Time",
                    "step_time": "Performance/Total Step Time",  # Might be less useful now
                    "cpu_usage": "Resource/CPU Usage (%)",
                    "memory_usage": "Resource/Memory Usage (%)",
                    "gpu_memory_usage_percent": "Resource/GPU Memory Usage (%)",
                }
                for key, tag in scalar_map.items():
                    if key in step_data and step_data[key] is not None:
                        self.writer.add_scalar(tag, step_data[key], g_step)

                # --- Corrected Access ---
                if update_info.get("new_best_loss"):
                    self.writer.add_scalar(
                        "Best Performance/Loss",
                        aggregator.storage.best_value_loss,
                        g_step,
                    )
                # --- End Correction ---
            except Exception as e:
                print(f"Error writing step scalars to TensorBoard: {e}")
