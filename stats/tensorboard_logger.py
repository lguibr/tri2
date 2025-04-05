# File: stats/tensorboard_logger.py
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
        dummy_input_for_graph: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        histogram_log_interval: int = TensorBoardConfig.HISTOGRAM_LOG_FREQ,
        image_log_interval: int = TensorBoardConfig.IMAGE_LOG_FREQ,
        env_config: Optional[EnvConfig] = None,
        rnn_config: Optional[RNNConfig] = None,
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
        self.vis_config = VisConfig()
        self.summary_avg_window = self.aggregator.summary_avg_window

        print(f"[TensorBoardStatsRecorder] Initialized. Logging to: {self.log_dir}")
        print(f"  Histogram Log Interval: {self.histogram_log_interval}")
        print(f"  Image Log Interval: {self.image_log_interval}")
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
            initial_metrics = {
                "hparam/initial_best_rl_score": -float("inf"),
                "hparam/initial_best_game_score": -float("inf"),
                "hparam/initial_best_loss": float("inf"),
            }
            filtered_hparams = {
                k: v
                for k, v in self.hparam_dict.items()
                if isinstance(v, (int, float, str, bool, torch.Tensor))
            }
            self.writer.add_hparams(filtered_hparams, initial_metrics, run_name=".")
            print("[TensorBoardStatsRecorder] Hyperparameters logged.")
        except Exception as e:
            print(f"Error logging initial hyperparameters: {e}")

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        lines_cleared: Optional[int] = None,
    ):
        # --- MODIFICATION: Call aggregator first to get update_info ---
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            lines_cleared,
        )
        # --- END MODIFICATION ---

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
        if lines_cleared is not None:
            self.writer.add_scalar("Episode/Lines Cleared", lines_cleared, g_step)
        self.writer.add_scalar("Progress/Total Episodes", episode_num, g_step)

        # Log best scores if they were updated (using update_info from aggregator)
        if update_info.get("new_best_rl"):
            self.writer.add_scalar(
                "Best Performance/RL Score", self.aggregator.best_score, g_step
            )
            # Optional: Keep console print here or rely on console_recorder's print
            # print(f"--- 🏆 TB Logged New Best RL: {self.aggregator.best_score:.2f} at Step ~{g_step/1e6:.1f}M (Prev: {self.aggregator.previous_best_score:.2f}) ---")

        if update_info.get("new_best_game"):
            self.writer.add_scalar(
                "Best Performance/Game Score", self.aggregator.best_game_score, g_step
            )
            # Optional: Keep console print here or rely on console_recorder's print
            # print(f"--- 🎮 TB Logged New Best Game: {self.aggregator.best_game_score:.0f} at Step ~{g_step/1e6:.1f}M (Prev: {self.aggregator.previous_best_game_score:.0f}) ---")

        # --- MODIFICATION: Delegate to console recorder AFTER aggregator ---
        # Pass the original arguments to the console recorder
        self.console_recorder.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,  # Pass original global_step
            game_score,
            lines_cleared,
        )
        # --- END MODIFICATION ---

    def record_step(self, step_data: Dict[str, Any]):
        # --- MODIFICATION: Call aggregator first to get update_info ---
        update_info = self.aggregator.record_step(step_data)
        # --- END MODIFICATION ---

        g_step = step_data.get("global_step", self.aggregator.current_global_step)

        # Log scalar values from step_data
        if "policy_loss" in step_data:
            self.writer.add_scalar("Loss/Policy Loss", step_data["policy_loss"], g_step)
        if "value_loss" in step_data:
            self.writer.add_scalar("Loss/Value Loss", step_data["value_loss"], g_step)
        if "entropy" in step_data:
            self.writer.add_scalar("Loss/Entropy", step_data["entropy"], g_step)
        if "grad_norm" in step_data:
            self.writer.add_scalar("Debug/Grad Norm", step_data["grad_norm"], g_step)
        if "avg_max_q" in step_data:
            self.writer.add_scalar("Debug/Avg Max Q", step_data["avg_max_q"], g_step)
        if "beta" in step_data:
            self.writer.add_scalar("Debug/Beta", step_data["beta"], g_step)
        if "buffer_size" in step_data:
            self.writer.add_scalar(
                "Debug/Buffer Size", step_data["buffer_size"], g_step
            )
        if "lr" in step_data:
            self.writer.add_scalar("Train/Learning Rate", step_data["lr"], g_step)
        if "epsilon" in step_data:
            self.writer.add_scalar("Train/Epsilon", step_data["epsilon"], g_step)
        if "sps_collection" in step_data:
            self.writer.add_scalar(
                "Performance/SPS Collection", step_data["sps_collection"], g_step
            )
        if "update_time" in step_data:
            self.writer.add_scalar(
                "Performance/Update Time", step_data["update_time"], g_step
            )
        if "step_time" in step_data:
            self.writer.add_scalar(
                "Performance/Total Step Time", step_data["step_time"], g_step
            )
            if step_data["step_time"] > 1e-9:
                num_steps = step_data.get("num_steps_processed", 1)
                sps_total = num_steps / step_data["step_time"]
                self.writer.add_scalar("Performance/SPS Total", sps_total, g_step)

        # Log best loss if updated (using update_info from aggregator)
        if update_info.get("new_best_loss"):
            self.writer.add_scalar(
                "Best Performance/Loss", self.aggregator.best_value_loss, g_step
            )
            # Optional: Keep console print here or rely on console_recorder's print
            # print(f"---📉 TB Logged New Best Loss: {self.aggregator.best_value_loss:.4f} at Step ~{g_step/1e6:.1f}M (Prev: {self.aggregator.previous_best_value_loss:.4f}) ---")

        # --- MODIFICATION: Delegate to console recorder AFTER aggregator ---
        # Pass the original step_data dictionary
        self.console_recorder.record_step(step_data)
        # --- END MODIFICATION ---

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        if self.histogram_log_interval <= 0:
            return
        # Log only at specified intervals based on global_step
        # Use modulo for periodic logging relative to start
        if (
            global_step
            // (
                self.aggregator.num_envs
                * self.aggregator.ppo_config.NUM_STEPS_PER_ROLLOUT
            )
            % self.histogram_log_interval
            == 0
        ):
            # Check if we already logged for this update cycle step
            if global_step > self.last_histogram_log_step:
                try:
                    self.writer.add_histogram(tag, values, global_step)
                    self.last_histogram_log_step = global_step
                except Exception as e:
                    print(f"Error logging histogram '{tag}': {e}")

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        if self.image_log_interval <= 0:
            return
        # Log only at specified intervals based on global_step
        if (
            global_step
            // (
                self.aggregator.num_envs
                * self.aggregator.ppo_config.NUM_STEPS_PER_ROLLOUT
            )
            % self.image_log_interval
            == 0
        ):
            if global_step > self.last_image_log_step:
                try:
                    # Ensure image has channel-first format (C, H, W) or (N, C, H, W)
                    if isinstance(image, np.ndarray):
                        if image.ndim == 3 and image.shape[-1] in [
                            1,
                            3,
                            4,
                        ]:  # HWC -> CHW
                            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
                        elif image.ndim == 2:  # HW -> CHW (add channel dim)
                            image_tensor = torch.from_numpy(image).unsqueeze(0)
                        else:  # Assume CHW or NCHW
                            image_tensor = torch.from_numpy(image)
                    elif isinstance(image, torch.Tensor):
                        if image.ndim == 3 and image.shape[0] not in [
                            1,
                            3,
                            4,
                        ]:  # HWC? -> CHW
                            if image.shape[-1] in [1, 3, 4]:
                                image_tensor = image.permute(2, 0, 1)
                            else:  # Assume CHW
                                image_tensor = image
                        elif image.ndim == 2:  # HW -> CHW
                            image_tensor = image.unsqueeze(0)
                        else:  # Assume CHW or NCHW
                            image_tensor = image
                    else:
                        print(
                            f"Warning: Unsupported image type for tag '{tag}': {type(image)}"
                        )
                        return

                    self.writer.add_image(
                        tag, image_tensor, global_step, dataformats="CHW"
                    )
                    self.last_image_log_step = global_step
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

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[Any] = None
    ):
        """Records the model graph."""
        if input_to_model is None:
            print("Warning: Cannot record graph without dummy input.")
            return
        try:
            model.cpu()
            if isinstance(input_to_model, tuple):
                dummy_input_cpu = tuple(
                    t.cpu() for t in input_to_model if isinstance(t, torch.Tensor)
                )
            elif isinstance(input_to_model, torch.Tensor):
                dummy_input_cpu = input_to_model.cpu()
            else:
                dummy_input_cpu = input_to_model

            self.writer.add_graph(model, dummy_input_cpu, verbose=False)
            print("[TensorBoardStatsRecorder] Model graph logged.")
            # Move model back to original device
            if hasattr(model, "device"):  # Check if model has device attr
                model.to(model.device)
            elif self.env_config:  # Fallback to general device
                model.to(DEVICE)

        except Exception as e:
            print(f"Error logging model graph: {e}. Graph logging can be tricky.")

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        # Delegate to console logger
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
            # Ensure hparam_dict exists before logging
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
        # Close console recorder as well
        self.console_recorder.close()
