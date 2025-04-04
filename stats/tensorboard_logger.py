# File: stats/tensorboard_logger.py
import time
import os
import numpy as np
import torch
import torchvision  # For image processing if needed
from collections import deque
from typing import Deque, List, Dict, Any, Optional, Union

try:
    from torch.utils.tensorboard import SummaryWriter

    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from .stats_recorder import StatsRecorderBase
from .simple_stats_recorder import (
    SimpleStatsRecorder,
)  # Use SimpleRecorder for averaging logic
from config import TensorBoardConfig  # Import TB config for flags


class TensorBoardStatsRecorder(StatsRecorderBase):
    """
    Records statistics and logs them to TensorBoard, including scalars,
    histograms, hyperparameters, and optionally images and model graph.
    """

    def __init__(
        self,
        log_dir: str,
        hparam_dict: Dict[str, Any],  # Pass config dictionary here
        model_for_graph: Optional[torch.nn.Module] = None,
        dummy_input_for_graph: Optional[torch.Tensor] = None,
        console_log_interval: int = 50_000,
        avg_window: int = 500,
    ):
        if not _TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard not found. Please install it: pip install tensorboard"
            )

        self.log_dir = log_dir
        self.config = TensorBoardConfig  # Store config for flags
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[TensorBoardStatsRecorder] Initializing. Log directory: {self.log_dir}")
        print(
            f"[TensorBoardStatsRecorder] Histograms: {self.config.LOG_HISTOGRAMS}, Images: {self.config.LOG_IMAGES}"
        )

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.hparams = hparam_dict  # Store hparams for potential final logging

        # Use SimpleStatsRecorder internally for averages needed by console/UI
        self.simple_recorder = SimpleStatsRecorder(
            console_log_interval=console_log_interval, avg_window=avg_window
        )
        self.last_histogram_log_step = (
            -self.config.HISTOGRAM_LOG_FREQ
        )  # Ensure initial log
        self.last_image_log_step = -self.config.IMAGE_LOG_FREQ  # Ensure initial log

        # --- Log Hyperparameters and Graph at Init ---
        self.record_hparams(self.hparams, {})  # Log hparams immediately, metrics later

        if model_for_graph is not None and dummy_input_for_graph is not None:
            self.record_graph(model_for_graph, dummy_input_for_graph)
        # --- End Init Logging ---

    def record_episode(
        self,
        episode_score: float,  # RL Score
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        lines_cleared: Optional[int] = None,
    ):
        """Record episode stats and log episode-level metrics to TensorBoard."""
        # Update internal simple recorder for averages
        self.simple_recorder.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            lines_cleared,
        )

        # Log episode metrics directly to TensorBoard
        if global_step is not None:
            self.writer.add_scalar("Episode/RL Score", episode_score, global_step)
            self.writer.add_scalar("Episode/Length", episode_length, global_step)
            if game_score is not None:
                self.writer.add_scalar("Episode/Game Score", game_score, global_step)
            if lines_cleared is not None:
                self.writer.add_scalar(
                    "Episode/Lines Cleared", lines_cleared, global_step
                )

            # Log rolling averages at episode end as well (provides smoother curves)
            summary = self.simple_recorder.get_summary(global_step)
            self.writer.add_scalar(
                "Rollout/Avg RL Score (Window)", summary["avg_score_100"], global_step
            )
            self.writer.add_scalar(
                "Rollout/Avg Game Score (Window)",
                summary["avg_game_score_100"],
                global_step,
            )
            self.writer.add_scalar(
                "Rollout/Avg Length (Window)", summary["avg_length_100"], global_step
            )
            self.writer.add_scalar(
                "Rollout/Avg Lines Cleared (Window)",
                summary["avg_lines_cleared_100"],
                global_step,
            )

    def record_step(self, step_data: Dict[str, Any]):
        """Record step/batch stats and log them to TensorBoard."""
        # Update internal simple recorder
        self.simple_recorder.record_step(step_data)

        global_step = step_data.get("global_step")
        if global_step is None:
            global_step = self.simple_recorder.current_global_step
            if global_step is None:
                print("Warning: Cannot log step to TensorBoard without global_step.")
                return  # Cannot log without step count

        # --- Log Scalars (Always) ---
        if "loss" in step_data:
            self.writer.add_scalar("Train/Loss", step_data["loss"], global_step)
        if "grad_norm" in step_data:
            self.writer.add_scalar(
                "Train/Gradient Norm", step_data["grad_norm"], global_step
            )
        if "avg_max_q" in step_data:
            self.writer.add_scalar(
                "Train/Avg Max Q (Batch)", step_data["avg_max_q"], global_step
            )  # Log batch avg Q
        if "beta" in step_data:
            self.writer.add_scalar("Train/PER Beta", step_data["beta"], global_step)
        if "buffer_size" in step_data:
            self.writer.add_scalar(
                "Info/Buffer Size", step_data["buffer_size"], global_step
            )
        if "step_time" in step_data:
            self.writer.add_scalar(
                "Info/Step Time (sec)", step_data["step_time"], global_step
            )

        # --- Log Histograms (Periodically) ---
        if (
            self.config.LOG_HISTOGRAMS
            and global_step
            >= self.last_histogram_log_step + self.config.HISTOGRAM_LOG_FREQ
        ):
            if "batch_q_values" in step_data:
                self.record_histogram(
                    "Train/Batch Q-Values", step_data["batch_q_values"], global_step
                )
            if "batch_td_errors" in step_data:
                self.record_histogram(
                    "Train/Batch TD Errors", step_data["batch_td_errors"], global_step
                )
            if "step_rewards_batch" in step_data:
                self.record_histogram(
                    "Rollout/Step Rewards Batch",
                    step_data["step_rewards_batch"],
                    global_step,
                )
            if "action_batch" in step_data:
                self.record_histogram(
                    "Rollout/Action Selection Batch",
                    step_data["action_batch"],
                    global_step,
                )
            # Add weight/gradient histograms here if implemented later
            self.last_histogram_log_step = global_step

        # --- Log Images (Periodically) ---
        if (
            self.config.LOG_IMAGES
            and global_step >= self.last_image_log_step + self.config.IMAGE_LOG_FREQ
        ):
            if "env_image" in step_data:
                # Assuming env_image is a single image tensor/numpy array
                self.record_image(
                    "Environment/Sample State", step_data["env_image"], global_step
                )
                self.last_image_log_step = global_step

        # Trigger console log check
        self.log_summary(global_step)

    # --- New Logging Methods Implementation ---
    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Logs histogram data if enabled."""
        if not self.config.LOG_HISTOGRAMS:
            return
        try:
            # Ensure numpy array for add_histogram
            if isinstance(values, torch.Tensor):
                values_np = values.detach().cpu().numpy()
            elif isinstance(values, list):
                values_np = np.array(values)
            else:
                values_np = values  # Assume it's already numpy

            if values_np is not None and values_np.size > 0:  # Check if empty
                # Add some basic stats along with histogram
                self.writer.add_scalar(f"{tag}/Mean", np.mean(values_np), global_step)
                self.writer.add_scalar(f"{tag}/StdDev", np.std(values_np), global_step)
                self.writer.add_histogram(tag, values_np, global_step)
            # else: print(f"Debug: Skipping empty histogram {tag}") # Optional debug
        except Exception as e:
            print(f"Error logging histogram '{tag}': {e}")

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Logs image data if enabled."""
        if not self.config.LOG_IMAGES:
            return
        try:
            # add_image expects Tensor [C, H, W] or [N, C, H, W]
            # or numpy array [H, W, C]
            if isinstance(image, np.ndarray):
                # Assuming HWC format from pygame surface usually
                self.writer.add_image(tag, image, global_step, dataformats="HWC")
            elif isinstance(image, torch.Tensor):
                # Check tensor format and potentially permute/squeeze
                img_tensor = image.detach().cpu()
                if img_tensor.ndim == 4:
                    img_tensor = img_tensor.squeeze(0)  # Remove batch dim if present
                # Assuming CHW is the most likely tensor format
                self.writer.add_image(tag, img_tensor, global_step, dataformats="CHW")
            else:
                print(f"Warning: Unsupported image type for tag '{tag}': {type(image)}")

        except Exception as e:
            print(f"Error logging image '{tag}': {e}")

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Logs hyperparameters. Typically called once at the start."""
        try:
            # Filter hparam_dict for scalar types suitable for add_hparams
            filtered_hparams = {
                k: v
                for k, v in hparam_dict.items()
                if isinstance(v, (int, float, str, bool))
            }
            # Metric dict is usually empty at start, filled at end of training
            self.writer.add_hparams(filtered_hparams, metric_dict)
            print(
                f"[TensorBoardStatsRecorder] Logged {len(filtered_hparams)} hyperparameters."
            )
        except Exception as e:
            # add_hparams can be finicky, log error but don't crash
            print(f"Warning: Error logging hparams: {e}")

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[torch.Tensor] = None
    ):
        """Logs the model graph."""
        if input_to_model is None:
            print("Warning: Cannot log model graph without dummy input.")
            return
        try:
            # Ensure model and input are on the same device
            device = next(model.parameters()).device
            input_to_model = input_to_model.to(device)
            # Check if model has 'reset_noise' and call if exists (for NoisyNets consistency during trace)
            if hasattr(model, "reset_noise"):
                model.reset_noise()
            # Trace the graph
            self.writer.add_graph(model, input_to_model=input_to_model, verbose=False)
            print("[TensorBoardStatsRecorder] Logged model graph.")
        except Exception as e:
            # Graph logging can fail (e.g., dynamic ops). Log error but continue.
            print(f"Warning: Error logging model graph: {e}")
            import traceback

            traceback.print_exc()  # Print stack trace for debugging graph issues

    # --- End New Logging Methods ---

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        # Delegate to simple recorder for UI/console summary
        return self.simple_recorder.get_summary(current_global_step)

    def log_summary(self, global_step: int):
        # Delegate console logging
        self.simple_recorder.log_summary(global_step)

        # Log averaged training stats to TB periodically (using console freq)
        # Check if the console log interval was *just* met
        log_interval = self.simple_recorder.console_log_interval
        if (
            log_interval > 0
            and global_step > 0
            and (global_step - self.simple_recorder.last_log_step < log_interval)
            and (global_step // log_interval > (global_step - 1) // log_interval)
        ):  # Check if boundary crossed

            summary = self.simple_recorder.get_summary(global_step)  # Get fresh summary
            self.writer.add_scalar(
                "Train/Avg Loss (Window)", summary["avg_loss_100"], global_step
            )
            self.writer.add_scalar(
                "Train/Avg Grad Norm (Window)", summary["avg_grad_100"], global_step
            )
            self.writer.add_scalar(
                "Train/Avg Max Q (Window)", summary["avg_max_q_100"], global_step
            )
            self.writer.add_scalar(
                "Info/Steps Per Second", summary["steps_per_second"], global_step
            )

    def close(self):
        """Closes the SummaryWriter and logs final metrics."""
        print("[TensorBoardStatsRecorder] Closing SummaryWriter...")
        if self.writer:
            try:
                # --- Log final metrics with hparams ---
                final_summary = self.simple_recorder.get_summary(
                    self.simple_recorder.current_global_step
                )
                final_metrics = {
                    "hparam/final_avg_score_100": final_summary.get("avg_score_100", 0),
                    "hparam/best_rl_score": final_summary.get("best_score", 0),
                    "hparam/best_game_score": final_summary.get("best_game_score", 0),
                    "hparam/total_episodes": final_summary.get("total_episodes", 0),
                    "hparam/final_global_step": self.simple_recorder.current_global_step,
                }
                # Filter hparams again just before logging
                filtered_hparams = {
                    k: v
                    for k, v in self.hparams.items()
                    if isinstance(v, (int, float, str, bool))
                }
                # Re-log hparams with final metrics (TensorBoard groups runs based on hparams)
                self.writer.add_hparams(filtered_hparams, final_metrics)
                print("[TensorBoardStatsRecorder] Logged final metrics with hparams.")
                # --- End final metrics ---

                self.writer.flush()
                self.writer.close()
            except Exception as e:
                print(f"Error closing TensorBoard writer: {e}")
        self.simple_recorder.close()  # Close internal recorder
        print("[TensorBoardStatsRecorder] Closed.")
