# File: stats/tensorboard_logger.py
import time
import traceback
from typing import Deque, Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from .simple_stats_recorder import SimpleStatsRecorder
from config import TensorBoardConfig, StatsConfig
from utils.helpers import ensure_numpy


class TensorBoardStatsRecorder(StatsRecorderBase):
    """
    Records statistics to TensorBoard and optionally logs to console.
    Uses a StatsAggregator for data storage and a SimpleStatsRecorder for console logging.
    """

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_recorder: SimpleStatsRecorder,
        log_dir: str,
        hparam_dict: Optional[Dict[str, Any]] = None,
        model_for_graph: Optional[torch.nn.Module] = None,
        dummy_input_for_graph: Optional[Any] = None,
        histogram_log_interval: int = TensorBoardConfig.HISTOGRAM_LOG_FREQ,
        image_log_interval: int = TensorBoardConfig.IMAGE_LOG_FREQ,
        shape_q_log_interval: int = TensorBoardConfig.SHAPE_Q_LOG_FREQ,
    ):
        self.aggregator = aggregator
        self.console_recorder = console_recorder
        self.log_dir = log_dir
        self.histogram_log_interval = (
            max(1, histogram_log_interval) if histogram_log_interval > 0 else -1
        )
        self.image_log_interval = (
            max(1, image_log_interval) if image_log_interval > 0 else -1
        )
        self.shape_q_log_interval = (
            max(1, shape_q_log_interval) if shape_q_log_interval > 0 else -1
        )

        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"[TensorBoard] Initialized. Logging to: {self.log_dir}")
        except Exception as e:
            print(f"FATAL: Failed to initialize TensorBoard SummaryWriter: {e}")
            raise e

        if hparam_dict:
            self.record_hparams(hparam_dict, {})

        if model_for_graph:
            self.record_graph(model_for_graph, dummy_input_for_graph)

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        lines_cleared: Optional[int] = None,
    ):
        """Passes data to aggregator/console and logs scalars to TensorBoard."""
        self.console_recorder.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            lines_cleared,
        )
        current_step = (
            global_step
            if global_step is not None
            else self.aggregator.current_global_step
        )
        try:
            self.writer.add_scalar("Episode/RL Score", episode_score, current_step)
            self.writer.add_scalar("Episode/Length", episode_length, current_step)
            if game_score is not None:
                self.writer.add_scalar("Episode/Game Score", game_score, current_step)
            if lines_cleared is not None:
                self.writer.add_scalar(
                    "Episode/Lines Cleared", lines_cleared, current_step
                )
        except Exception as e:
            print(f"Warning: TensorBoard episode scalar logging failed: {e}")

    def record_step(self, step_data: Dict[str, Any]):
        """Passes data to aggregator/console and logs scalars/histograms to TensorBoard."""
        self.console_recorder.record_step(step_data)
        g_step = step_data.get("global_step", self.aggregator.current_global_step)

        # Log scalars
        scalars_to_log = {
            "loss": "Train/Loss",
            "grad_norm": "Train/Gradient Norm",
            "avg_max_q": "Train/Avg Max Q",
            "beta": "Train/PER Beta",
            "buffer_size": "Info/Buffer Size",
            "lr": "Train/Learning Rate",
            "epsilon": "Train/Epsilon",
            "steps_per_second": "Info/Steps Per Second",
        }
        for key, tag in scalars_to_log.items():
            if key in step_data and step_data[key] is not None:
                try:
                    self.writer.add_scalar(tag, step_data[key], g_step)
                except Exception as e:
                    print(f"Warning: TensorBoard scalar logging failed for {tag}: {e}")

        # Log histograms periodically
        if (
            self.histogram_log_interval > 0
            and g_step % self.histogram_log_interval == 0
        ):
            histograms_to_log = {
                "step_rewards_batch": "Batch/Step Rewards",
                "action_batch": "Batch/Actions Taken",
                "batch_td_errors": "Train/TD Errors",
                "chosen_shape_slot_batch": "Batch/Chosen Shape Slot",
                "shape_slot_max_q_batch": "Batch/Shape Slot Max Q",
            }
            for key, tag in histograms_to_log.items():
                if key in step_data and step_data[key] is not None:
                    self.record_histogram(tag, step_data[key], g_step)

        # Log specific Q-value histograms
        if (
            self.shape_q_log_interval > 0
            and g_step % self.shape_q_log_interval == 0
            and "batch_q_values_actions_taken" in step_data
            and step_data["batch_q_values_actions_taken"] is not None
        ):
            self.record_histogram(
                "Batch/Q Values (Actions Taken)",
                step_data["batch_q_values_actions_taken"],
                g_step,
            )

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Logs histogram data to TensorBoard."""
        if self.histogram_log_interval <= 0:
            return
        try:
            # Ensure numpy array for consistency
            values_np = ensure_numpy(values)
            if values_np.size > 0:
                self.writer.add_histogram(tag, values_np, global_step)
        except Exception as e:
            print(f"Warning: TensorBoard histogram logging failed for {tag}: {e}")

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Logs image data to TensorBoard."""
        if self.image_log_interval <= 0:
            return
        try:
            self.writer.add_image(tag, image, global_step, dataformats="CHW")
        except Exception as e:
            print(f"Warning: TensorBoard image logging failed for {tag}: {e}")

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Logs hyperparameters and final metrics."""
        try:
            # Filter out non-scalar/string hparams if necessary
            filtered_hparams = {
                k: v
                for k, v in hparam_dict.items()
                if isinstance(v, (int, float, str, bool))
            }
            self.writer.add_hparams(filtered_hparams, metric_dict)
        except Exception as e:
            print(f"Warning: TensorBoard hparam logging failed: {e}")

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[Any] = None
    ):
        """Logs the model graph to TensorBoard."""
        if input_to_model is None:
            print("[TensorBoard] Skipping graph logging: No dummy input provided.")
            return
        try:
            # Ensure model and input are on the same device (CPU recommended for graph)
            model.cpu()
            if isinstance(input_to_model, torch.Tensor):
                input_to_model = input_to_model.cpu()
            elif isinstance(input_to_model, tuple):
                input_to_model = tuple(t.cpu() for t in input_to_model)

            # Temporarily ignore specific warnings during graph tracing
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", category=torch.jit.TracerWarning
                )  # Common warning
                warnings.simplefilter("ignore", category=UserWarning)
                self.writer.add_graph(model, input_to_model)
            print("[TensorBoard] Model graph logged.")
        except Exception as e:
            print(f"Warning: TensorBoard graph logging failed: {e}")
            traceback.print_exc()
        finally:
            # Ensure model is moved back to original device if needed
            # (Handled by caller, typically)
            pass

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Retrieves summary statistics from the aggregator."""
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        """Retrieves plot data deques from the aggregator."""
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        """Triggers console logging via the composed SimpleStatsRecorder."""
        self.console_recorder.log_summary(global_step)

    def close(self):
        """Closes the TensorBoard writer and the console recorder."""
        print("[TensorBoard] Closing SummaryWriter...")
        try:
            self.writer.flush()
            self.writer.close()
        except Exception as e:
            print(f"Error closing TensorBoard writer: {e}")
        self.console_recorder.close()
        print("[TensorBoard] Closed.")
