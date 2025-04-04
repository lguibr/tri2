# File: stats/tensorboard_logger.py
import time
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List, Tuple, Callable
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
import traceback  # Import traceback

from .simple_stats_recorder import SimpleStatsRecorder
from config import TensorBoardConfig  # Import specific config

# Filter known TensorBoard warnings
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="torch.utils.tensorboard"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.utils.tensorboard"
)


class TensorBoardStatsRecorder(SimpleStatsRecorder):
    """
    Extends SimpleStatsRecorder to log metrics, histograms, images,
    hparams, and model graph to TensorBoard.
    """

    def __init__(
        self,
        log_dir: str,
        hparam_dict: Dict[str, Any],
        model_for_graph: Optional[torch.nn.Module] = None,
        dummy_input_for_graph: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        ] = None,  # Accept tuple
        console_log_interval: int = 50_000,
        avg_window: int = 100,
        histogram_log_interval: int = 10_000,
        image_log_interval: int = -1,  # Disabled by default
    ):
        super().__init__(console_log_interval, avg_window)
        self.log_dir = log_dir
        self.histogram_log_interval = (
            max(1, histogram_log_interval) if histogram_log_interval > 0 else -1
        )
        self.image_log_interval = (
            max(1, image_log_interval) if image_log_interval > 0 else -1
        )
        self.last_histogram_log_step = -self.histogram_log_interval  # Ensure first log
        self.last_image_log_step = -self.image_log_interval  # Ensure first log

        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"[TensorBoard] Writer initialized. Logging to: {self.log_dir}")
            # Log hyperparameters immediately
            self.record_hparams(
                hparam_dict, {}
            )  # Log hparams with empty metrics initially
            # Log graph if model provided
            if model_for_graph:
                self.record_graph(model_for_graph, dummy_input_for_graph)

        except Exception as e:
            print(f"FATAL: Failed to initialize TensorBoard SummaryWriter: {e}")
            print("TensorBoard logging will be disabled.")
            traceback.print_exc()
            self.writer = None  # Disable writer on error

    def record_episode(self, *args, **kwargs):
        super().record_episode(*args, **kwargs)
        # Log episode metrics to TensorBoard
        if self.writer:
            global_step = kwargs.get("global_step", self.current_global_step)
            if "episode_score" in kwargs:
                self.writer.add_scalar(
                    "Episode/RL Score", kwargs["episode_score"], global_step
                )
            if "episode_length" in kwargs:
                self.writer.add_scalar(
                    "Episode/Length", kwargs["episode_length"], global_step
                )
            if "game_score" in kwargs and kwargs["game_score"] is not None:
                self.writer.add_scalar(
                    "Episode/Game Score", kwargs["game_score"], global_step
                )
            if "lines_cleared" in kwargs and kwargs["lines_cleared"] is not None:
                self.writer.add_scalar(
                    "Episode/Lines Cleared", kwargs["lines_cleared"], global_step
                )

            # Log running averages
            summary = self.get_summary(global_step)
            self.writer.add_scalar(
                "Averages/Avg RL Score (Window)",
                summary["avg_score_window"],
                global_step,
            )
            self.writer.add_scalar(
                "Averages/Avg Ep Length (Window)",
                summary["avg_length_window"],
                global_step,
            )
            self.writer.add_scalar(
                "Averages/Avg Game Score (Window)",
                summary["avg_game_score_window"],
                global_step,
            )

    def record_step(self, step_data: Dict[str, Any]):
        super().record_step(step_data)  # Updates internal deques and logs console
        if not self.writer:
            return

        g_step = step_data.get("global_step", self.current_global_step)

        # Log scalar metrics from step_data
        if "loss" in step_data and step_data["loss"] is not None:
            self.writer.add_scalar("Train/Loss", step_data["loss"], g_step)
        if "avg_max_q" in step_data and step_data["avg_max_q"] is not None:
            self.writer.add_scalar("Train/Avg Max Q", step_data["avg_max_q"], g_step)
        if "grad_norm" in step_data and step_data["grad_norm"] is not None:
            self.writer.add_scalar(
                "Train/Gradient Norm", step_data["grad_norm"], g_step
            )
        if "beta" in step_data and step_data["beta"] is not None:
            self.writer.add_scalar("Train/PER Beta", step_data["beta"], g_step)
        if "lr" in step_data and step_data["lr"] is not None:
            self.writer.add_scalar("Train/Learning Rate", step_data["lr"], g_step)
        if "steps_per_second" in step_data:  # Log current SPS
            self.writer.add_scalar(
                "Performance/Steps Per Second", step_data["steps_per_second"], g_step
            )
        if "buffer_size" in step_data:
            self.writer.add_scalar(
                "Performance/Buffer Size", step_data["buffer_size"], g_step
            )

        # Log histograms periodically
        if (
            self.histogram_log_interval > 0
            and g_step >= self.last_histogram_log_step + self.histogram_log_interval
        ):
            if "batch_q_values" in step_data:
                self.record_histogram(
                    "Histograms/Batch Q-Values", step_data["batch_q_values"], g_step
                )
            if "batch_td_errors" in step_data:
                self.record_histogram(
                    "Histograms/Batch TD-Errors", step_data["batch_td_errors"], g_step
                )
            if "step_rewards_batch" in step_data:
                self.record_histogram(
                    "Histograms/Step Rewards (Batch)",
                    step_data["step_rewards_batch"],
                    g_step,
                )
            if "action_batch" in step_data:
                self.record_histogram(
                    "Histograms/Actions (Batch)", step_data["action_batch"], g_step
                )
            self.last_histogram_log_step = g_step

        # Note: Image logging is handled by Trainer calling record_image directly

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        if not self.writer:
            return
        try:
            # Ensure values are suitable for add_histogram (numpy array or tensor)
            if isinstance(values, list):
                values_np = np.array(values)
            elif isinstance(values, torch.Tensor):
                values_np = values.detach().cpu().numpy()
            else:
                values_np = values  # Assume numpy array

            if values_np is not None and values_np.size > 0:
                self.writer.add_histogram(tag, values_np, global_step)
            # else: print(f"Warning: Skipping histogram '{tag}' due to empty/None values.")
        except Exception as e:
            print(f"Warning: Failed to log histogram '{tag}': {e}")
            # traceback.print_exc() # Optional: for more detail

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        if not self.writer or self.image_log_interval <= 0:
            return
        # Check frequency (already done by caller in Trainer._maybe_log_image)
        # if global_step < self.last_image_log_step + self.image_log_interval: return

        try:
            # Ensure image has CHW format for add_image
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] in [1, 3, 4]:  # HWC
                    img_tensor = torch.from_numpy(image).permute(2, 0, 1)
                elif image.ndim == 3 and image.shape[0] in [1, 3, 4]:  # CHW
                    img_tensor = torch.from_numpy(image)
                elif image.ndim == 2:  # Grayscale HW -> add channel dim 1HW
                    img_tensor = torch.from_numpy(image).unsqueeze(0)
                else:
                    print(
                        f"Warning: Unsupported image shape for TB logging: {image.shape}"
                    )
                    return
            elif isinstance(image, torch.Tensor):
                if image.ndim == 3 and image.shape[0] in [1, 3, 4]:  # CHW
                    img_tensor = image
                elif (
                    image.ndim == 4
                    and image.shape[0] == 1
                    and image.shape[1] in [1, 3, 4]
                ):  # BCHW
                    img_tensor = image.squeeze(0)
                # Add more checks if needed (e.g., for HWC tensors)
                else:
                    print(
                        f"Warning: Unsupported image tensor shape for TB logging: {image.shape}"
                    )
                    return
            else:
                print(f"Warning: Unsupported image type for TB logging: {type(image)}")
                return

            self.writer.add_image(tag, img_tensor, global_step)
            self.last_image_log_step = global_step  # Update last log step
        except Exception as e:
            print(f"Warning: Failed to log image '{tag}': {e}")
            # traceback.print_exc()

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        if not self.writer:
            return
        try:
            # Sanitize hparam_dict: TensorBoard only supports bool, str, float, int, None
            sanitized_hparams = {}
            for k, v in hparam_dict.items():
                if isinstance(v, (bool, str, float, int)) or v is None:
                    sanitized_hparams[k] = v
                elif isinstance(
                    v, (list, tuple, dict, torch.device)
                ):  # Convert complex types to string
                    sanitized_hparams[k] = str(v)
                # Add more specific conversions if needed

            # Ensure metric_dict values are simple types
            sanitized_metrics = {}
            for k, v in metric_dict.items():
                if isinstance(v, (float, int)):
                    sanitized_metrics[k] = v

            # Use "/" for grouping in TensorBoard HParams UI
            formatted_hparams = {
                k.replace(".", "/"): v for k, v in sanitized_hparams.items()
            }

            self.writer.add_hparams(formatted_hparams, sanitized_metrics)
            print("[TensorBoard] Hyperparameters logged.")
        except Exception as e:
            print(f"Warning: Failed to log hyperparameters: {e}")
            # traceback.print_exc()

    # --- MODIFIED: record_graph handles tuple input ---
    def record_graph(
        self,
        model: torch.nn.Module,
        input_to_model: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    ):
        """Record the model graph."""
        if not self.writer:
            return
        print("[TensorBoard] Attempting to log model graph...")
        try:
            # Ensure model is on CPU for graph logging if it isn't already
            original_device = next(model.parameters()).device
            model.cpu()

            if input_to_model is not None:
                # Ensure dummy input is also on CPU
                if isinstance(input_to_model, torch.Tensor):
                    input_cpu = input_to_model.cpu()
                    self.writer.add_graph(model, input_to_model=input_cpu)
                    print("[TensorBoard] Model graph logged (single input).")
                elif isinstance(input_to_model, tuple):
                    input_cpu_tuple = tuple(
                        t.cpu() for t in input_to_model if isinstance(t, torch.Tensor)
                    )
                    # add_graph expects a tuple of inputs if the model's forward takes multiple args
                    self.writer.add_graph(model, input_to_model=input_cpu_tuple)
                    print("[TensorBoard] Model graph logged (tuple input).")
                else:
                    print(
                        "Warning: Unsupported type for dummy_input_for_graph. Skipping graph input."
                    )
                    self.writer.add_graph(model)
                    print("[TensorBoard] Model graph logged (no input).")
            else:
                # Log graph without dummy input (less informative)
                self.writer.add_graph(model)
                print("[TensorBoard] Model graph logged (no input).")

            # Move model back to original device
            model.to(original_device)

        except Exception as e:
            print(f"Warning: Failed to log model graph: {e}")
            traceback.print_exc()
            # Ensure model is moved back even if logging fails
            try:
                model.to(original_device)
            except:
                pass

    # --- END MODIFIED ---

    def close(self):
        """Closes the TensorBoard writer."""
        super().close()  # Call parent's close (which is no-op)
        if self.writer:
            try:
                # Log final metrics before closing?
                # summary = self.get_summary(self.current_global_step)
                # final_metrics = {
                #     "hparam/final_avg_score": summary.get("avg_score_window", 0),
                #     "hparam/best_score": summary.get("best_score", -float('inf')),
                # }
                # self.record_hparams({}, final_metrics) # Log only metrics

                self.writer.flush()
                self.writer.close()
                print("[TensorBoard] Writer closed.")
            except Exception as e:
                print(f"Error closing TensorBoard writer: {e}")
            self.writer = None
