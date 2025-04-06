import time
import traceback
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io
import PIL.Image

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from .simple_stats_recorder import SimpleStatsRecorder
from config import (
    TensorBoardConfig,
    EnvConfig,
    RNNConfig,
    VisConfig,
    StatsConfig,
)

# Import ActorCriticNetwork for type hinting graph model
from agent.networks.agent_network import ActorCriticNetwork


class TensorBoardStatsRecorder(StatsRecorderBase):
    """
    Logs aggregated statistics, histograms, images, and hyperparameters to TensorBoard.
    Uses a SimpleStatsRecorder for console logging and a StatsAggregator for data handling.
    """

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_recorder: SimpleStatsRecorder,
        log_dir: str,
        hparam_dict: Optional[Dict[str, Any]] = None,
        model_for_graph: Optional[ActorCriticNetwork] = None,
        dummy_input_for_graph: Optional[
            Tuple
        ] = None,  # Use generic Tuple for dummy input
        histogram_log_interval: int = TensorBoardConfig.HISTOGRAM_LOG_FREQ,
        image_log_interval: int = TensorBoardConfig.IMAGE_LOG_FREQ,
        env_config: Optional[EnvConfig] = None,  # Keep for context
        rnn_config: Optional[RNNConfig] = None,  # Keep for context
    ):
        self.aggregator = aggregator
        self.console_recorder = console_recorder
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.hparam_dict = hparam_dict if hparam_dict else {}
        self.histogram_log_interval = (
            max(1, histogram_log_interval) if histogram_log_interval > 0 else -1
        )
        self.image_log_interval = (
            max(1, image_log_interval) if image_log_interval > 0 else -1
        )
        self.last_histogram_log_step = -1
        self.last_image_log_step = -1
        self.env_config = env_config
        self.rnn_config = rnn_config
        self.vis_config = VisConfig()  # Needed for image rendering utils potentially
        self.summary_avg_window = self.aggregator.summary_avg_window

        self.rollouts_since_last_tb_log = 0

        print(f"[TensorBoardStatsRecorder] Initialized. Logging to: {self.log_dir}")
        print(f"  Histogram Log Interval: {self.histogram_log_interval} rollouts")
        print(f"  Image Log Interval: {self.image_log_interval} rollouts")
        print(f"  Summary Avg Window: {self.summary_avg_window}")

        if model_for_graph and dummy_input_for_graph:
            self.record_graph(model_for_graph, dummy_input_for_graph)
        else:
            print(
                "[TensorBoardStatsRecorder] Model graph logging skipped (model or dummy input not provided)."
            )

        if self.hparam_dict:
            self._log_hparams_initial()

    def _log_hparams_initial(self):
        """Logs hyperparameters at the beginning of the run."""
        try:
            # Define initial metrics to avoid errors if no episodes complete before closing
            initial_metrics = {
                "hparam/final_best_rl_score": -float("inf"),
                "hparam/final_best_game_score": -float("inf"),
                "hparam/final_best_loss": float("inf"),
                "hparam/final_total_episodes": 0,
            }
            # Filter hparams to only include loggable types
            filtered_hparams = {
                k: v
                for k, v in self.hparam_dict.items()
                if isinstance(v, (int, float, str, bool, torch.Tensor))
            }
            self.writer.add_hparams(filtered_hparams, initial_metrics, run_name=".")
            print("[TensorBoardStatsRecorder] Hyperparameters logged.")
        except Exception as e:
            print(f"Error logging initial hyperparameters: {e}")
            traceback.print_exc()

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,  # Use triangles_cleared
    ):
        """Records episode stats to TensorBoard and delegates to console recorder."""
        # Call aggregator first to update internal state and get update_info
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,  # Pass renamed parameter
        )
        g_step = (
            global_step
            if global_step is not None
            else self.aggregator.current_global_step
        )

        # Log scalar values for episode stats
        self.writer.add_scalar("Episode/Score", episode_score, g_step)
        self.writer.add_scalar("Episode/Length", episode_length, g_step)
        if game_score is not None:
            self.writer.add_scalar("Episode/Game Score", game_score, g_step)
        if triangles_cleared is not None:
            self.writer.add_scalar(
                "Episode/Triangles Cleared", triangles_cleared, g_step
            )
        self.writer.add_scalar("Progress/Total Episodes", episode_num, g_step)

        # Log best scores if they were updated
        if update_info.get("new_best_rl"):
            self.writer.add_scalar(
                "Best Performance/RL Score", self.aggregator.best_score, g_step
            )
        if update_info.get("new_best_game"):
            self.writer.add_scalar(
                "Best Performance/Game Score", self.aggregator.best_game_score, g_step
            )

        # Delegate to console recorder AFTER aggregator and TB logging
        self.console_recorder.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,  # Pass renamed parameter
        )

    def record_step(self, step_data: Dict[str, Any]):
        """Records step stats to TensorBoard and delegates to console recorder."""
        # Call aggregator first
        update_info = self.aggregator.record_step(step_data)
        g_step = step_data.get("global_step", self.aggregator.current_global_step)

        # Log scalar values from step_data
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
            "sps_collection": "Performance/SPS Collection",
            "update_time": "Performance/Update Time",
            "step_time": "Performance/Total Step Time",
        }
        for key, tag in scalar_map.items():
            if key in step_data and step_data[key] is not None:
                self.writer.add_scalar(tag, step_data[key], g_step)

        # Log total SPS if step_time is available
        if "step_time" in step_data and step_data["step_time"] > 1e-9:
            num_steps = step_data.get("num_steps_processed", 1)
            sps_total = num_steps / step_data["step_time"]
            self.writer.add_scalar("Performance/SPS Total", sps_total, g_step)

        # Log best loss if updated
        if update_info.get("new_best_loss"):
            self.writer.add_scalar(
                "Best Performance/Loss", self.aggregator.best_value_loss, g_step
            )

        # Delegate to console recorder AFTER aggregator and TB logging
        self.console_recorder.record_step(step_data)

        # Use internal counter for TB logging frequency
        if "policy_loss" in step_data:
            self.rollouts_since_last_tb_log += 1

            # Check histogram logging
            if (
                self.histogram_log_interval > 0
                and self.rollouts_since_last_tb_log >= self.histogram_log_interval
            ):
                if g_step > self.last_histogram_log_step:
                    self.last_histogram_log_step = g_step

            # Check image logging
            if (
                self.image_log_interval > 0
                and self.rollouts_since_last_tb_log >= self.image_log_interval
            ):
                if g_step > self.last_image_log_step:
                    self.last_image_log_step = g_step

            if (
                self.histogram_log_interval > 0
                and self.rollouts_since_last_tb_log >= self.histogram_log_interval
            ) or (
                self.image_log_interval > 0
                and self.rollouts_since_last_tb_log >= self.image_log_interval
            ):
                if (
                    g_step == self.last_histogram_log_step
                    or g_step == self.last_image_log_step
                ):
                    self.rollouts_since_last_tb_log = 0

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Records a histogram to TensorBoard."""
        try:
            self.writer.add_histogram(tag, values, global_step)
        except Exception as e:
            print(f"Error logging histogram '{tag}': {e}")

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Records an image to TensorBoard, ensuring correct data format."""
        try:
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[-1] in [1, 3, 4]:
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
                elif image.ndim == 2:
                    image_tensor = torch.from_numpy(image).unsqueeze(0)
                else:
                    image_tensor = torch.from_numpy(image)
            elif isinstance(image, torch.Tensor):
                if image.ndim == 3 and image.shape[0] not in [1, 3, 4]:
                    if image.shape[-1] in [1, 3, 4]:
                        image_tensor = image.permute(2, 0, 1)
                    else:
                        image_tensor = image
                elif image.ndim == 2:
                    image_tensor = image.unsqueeze(0)
                else:
                    image_tensor = image
            else:
                print(f"Warning: Unsupported image type for tag '{tag}': {type(image)}")
                return

            self.writer.add_image(tag, image_tensor, global_step, dataformats="CHW")
        except Exception as e:
            print(f"Error logging image '{tag}': {e}")

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Records final hyperparameters and metrics."""
        try:
            filtered_hparams = {
                k: v
                for k, v in hparam_dict.items()
                if isinstance(v, (int, float, str, bool, torch.Tensor))
            }
            filtered_metrics = {
                k: v for k, v in metric_dict.items() if isinstance(v, (int, float))
            }
            self.writer.add_hparams(filtered_hparams, filtered_metrics, run_name=".")
            print("[TensorBoardStatsRecorder] Final hparams and metrics logged.")
        except Exception as e:
            print(f"Error logging final hyperparameters/metrics: {e}")
            traceback.print_exc()

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[Any] = None
    ):
        """Records the model graph to TensorBoard."""
        if input_to_model is None:
            print("Warning: Cannot record graph without dummy input.")
            return
        try:
            original_device = next(model.parameters()).device
            model.cpu()
            dummy_input_cpu: Any
            if isinstance(input_to_model, tuple):
                dummy_input_cpu_list = []
                for item in input_to_model:
                    if isinstance(item, torch.Tensor):
                        dummy_input_cpu_list.append(item.cpu())
                    elif isinstance(item, tuple):
                        dummy_input_cpu_list.append(
                            tuple(
                                t.cpu() if isinstance(t, torch.Tensor) else t
                                for t in item
                            )
                        )
                    else:
                        dummy_input_cpu_list.append(item)
                dummy_input_cpu = tuple(dummy_input_cpu_list)
            elif isinstance(input_to_model, torch.Tensor):
                dummy_input_cpu = input_to_model.cpu()
            else:
                dummy_input_cpu = input_to_model

            self.writer.add_graph(model, dummy_input_cpu, verbose=False)
            print("[TensorBoardStatsRecorder] Model graph logged.")
            model.to(original_device)
        except Exception as e:
            print(f"Error logging model graph: {e}. Graph logging can be tricky.")
            traceback.print_exc()
            try:
                model.to(original_device)
            except:
                pass

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """Gets the summary dictionary from the aggregator."""
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        """Gets the plot data deques from the aggregator."""
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        """Delegates console logging to the SimpleStatsRecorder."""
        self.console_recorder.log_summary(global_step)

    def close(self):
        """Closes the TensorBoard writer and logs final hparams."""
        print("[TensorBoardStatsRecorder] Closing writer...")
        try:
            final_summary = self.get_summary(self.aggregator.current_global_step)
            final_metrics = {
                "hparam/final_best_rl_score": final_summary.get(
                    "best_score", -float("inf")
                ),
                "hparam/final_best_game_score": final_summary.get(
                    "best_game_score", -float("inf")
                ),
                "hparam/final_best_loss": final_summary.get("best_loss", float("inf")),
                "hparam/final_total_episodes": final_summary.get("total_episodes", 0),
            }
            if self.hparam_dict:
                self.record_hparams(self.hparam_dict, final_metrics)
            else:
                print(
                    "[TensorBoardStatsRecorder] Skipping final hparam logging (hparam_dict not set)."
                )

            self.writer.flush()
            self.writer.close()
            print("[TensorBoardStatsRecorder] Writer closed.")
        except Exception as e:
            print(f"Error during TensorBoard writer close: {e}")
            traceback.print_exc()
        self.console_recorder.close()