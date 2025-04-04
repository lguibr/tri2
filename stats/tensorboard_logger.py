# File: stats/tensorboard_logger.py
import time
from typing import Dict, Any, Optional, Union, List, Callable
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import traceback

from .simple_stats_recorder import SimpleStatsRecorder
from config import TensorBoardConfig, StatsConfig  # Import configs


class TensorBoardStatsRecorder(SimpleStatsRecorder):
    """
    Extends SimpleStatsRecorder to log data to TensorBoard.
    Uses the SimpleStatsRecorder for in-memory averaging and summary generation.
    """

    def __init__(
        self,
        log_dir: str,
        hparam_dict: Optional[Dict[str, Any]] = None,
        model_for_graph: Optional[torch.nn.Module] = None,
        dummy_input_for_graph: Optional[torch.Tensor] = None,
        console_log_interval: int = StatsConfig.CONSOLE_LOG_FREQ,  # Use config default
        avg_window: int = StatsConfig.STATS_AVG_WINDOW,  # Use config default
        histogram_log_interval: int = TensorBoardConfig.HISTOGRAM_LOG_FREQ,  # Use config default
        image_log_interval: int = TensorBoardConfig.IMAGE_LOG_FREQ,  # Use config default
        # --- REMOVED: notification_callback ---
        # notification_callback: Optional[Callable[[str], None]] = None,
    ):
        # --- MODIFIED: Pass None for notification_callback to parent ---
        super().__init__(
            console_log_interval=console_log_interval,
            avg_window=avg_window,
            # notification_callback=notification_callback, # Removed
        )
        # --- END MODIFIED ---

        self.log_dir = log_dir
        self.histogram_log_interval = (
            max(1, histogram_log_interval) if histogram_log_interval > 0 else -1
        )
        self.image_log_interval = (
            max(1, image_log_interval) if image_log_interval > 0 else -1
        )
        self.last_histogram_log_step = (
            -self.histogram_log_interval
        )  # Ensure first log happens
        self.last_image_log_step = -self.image_log_interval  # Ensure first log happens

        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(
                f"[TB Logger] TensorBoard writer initialized. Log directory: {self.log_dir}"
            )
        except Exception as e:
            print(
                f"FATAL: Could not initialize TensorBoard SummaryWriter at {self.log_dir}: {e}"
            )
            traceback.print_exc()
            raise e  # Re-raise to prevent execution without logging

        # Log hyperparameters if provided
        if hparam_dict:
            self.record_hparams(hparam_dict, {})  # Log hparams early, metrics later

        # Log model graph if provided
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
        # Call parent first to update best scores, messages, and print to console
        super().record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            lines_cleared,
        )

        # Log episode stats to TensorBoard
        step = global_step if global_step is not None else self.current_global_step
        self.writer.add_scalar("Episode/RL Score", episode_score, step)
        self.writer.add_scalar("Episode/Length", episode_length, step)
        if game_score is not None:
            self.writer.add_scalar("Episode/Game Score", game_score, step)
        if lines_cleared is not None:
            self.writer.add_scalar("Episode/Lines Cleared", lines_cleared, step)
        self.writer.add_scalar("Progress/Total Episodes", episode_num, step)

        # Log best scores achieved so far
        if self.best_score > -float("inf"):
            self.writer.add_scalar("Best Performance/RL Score", self.best_score, step)
        if self.best_game_score > -float("inf"):
            self.writer.add_scalar(
                "Best Performance/Game Score", self.best_game_score, step
            )

    def record_step(self, step_data: Dict[str, Any]):
        # Call parent first to update deques, current values, and log to console
        super().record_step(step_data)

        g_step = step_data.get("global_step", self.current_global_step)

        # Log scalar values present in step_data
        if "loss" in step_data and step_data["loss"] is not None:
            self.writer.add_scalar("Train/Loss", step_data["loss"], g_step)
        if "grad_norm" in step_data and step_data["grad_norm"] is not None:
            self.writer.add_scalar(
                "Train/Gradient Norm", step_data["grad_norm"], g_step
            )
        if "avg_max_q" in step_data and step_data["avg_max_q"] is not None:
            self.writer.add_scalar("Train/Avg Max Q", step_data["avg_max_q"], g_step)
        if "beta" in step_data and step_data["beta"] is not None:
            self.writer.add_scalar("Buffer/PER Beta", step_data["beta"], g_step)
        if "buffer_size" in step_data and step_data["buffer_size"] is not None:
            self.writer.add_scalar("Buffer/Size", step_data["buffer_size"], g_step)
            self.writer.add_scalar(
                "Buffer/Fill Percentage",
                (step_data["buffer_size"] / max(1, self.avg_window))
                * 100,  # Assuming avg_window is buffer capacity - FIX THIS if needed
                g_step,
            )  # TODO: Need actual buffer capacity here
        if "lr" in step_data and step_data["lr"] is not None:
            self.writer.add_scalar("Train/Learning Rate", step_data["lr"], g_step)
        if (
            "steps_per_second" in step_data
            and step_data["steps_per_second"] is not None
        ):
            self.writer.add_scalar(
                "Performance/Steps Per Second", step_data["steps_per_second"], g_step
            )
        elif "step_time" in step_data and step_data["step_time"] > 1e-6:
            num_steps = step_data.get("num_steps_processed", 1)
            sps = num_steps / step_data["step_time"]
            self.writer.add_scalar("Performance/Steps Per Second", sps, g_step)

        # Log histograms periodically
        if (
            self.histogram_log_interval > 0
            and g_step >= self.last_histogram_log_step + self.histogram_log_interval
        ):
            if "step_rewards_batch" in step_data:
                self.record_histogram(
                    "Batch/Step Rewards", step_data["step_rewards_batch"], g_step
                )
            if "action_batch" in step_data:
                self.record_histogram(
                    "Batch/Actions Taken", step_data["action_batch"], g_step
                )
            if "batch_q_values" in step_data:
                self.record_histogram(
                    "Batch/Q Values (Online Net)", step_data["batch_q_values"], g_step
                )
            if "batch_td_errors" in step_data:
                self.record_histogram(
                    "Batch/TD Errors", step_data["batch_td_errors"], g_step
                )
            self.last_histogram_log_step = g_step

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Records a histogram to TensorBoard."""
        if self.histogram_log_interval <= 0:
            return  # Skip if disabled

        try:
            # Ensure values are suitable for add_histogram (numpy array or tensor)
            if isinstance(values, list):
                values_np = np.array(values)
            elif isinstance(values, torch.Tensor):
                values_np = values.detach().cpu().numpy()
            else:
                values_np = values  # Assume it's already a numpy array

            # Filter out NaNs or Infs which cause errors
            values_np = values_np[np.isfinite(values_np)]

            if values_np.size > 0:  # Only log if there's valid data
                self.writer.add_histogram(tag, values_np, global_step)
            # else: print(f"Warning: Skipping histogram '{tag}' at step {global_step} due to no finite values.")

        except Exception as e:
            print(f"Error logging histogram '{tag}' at step {global_step}: {e}")
            # traceback.print_exc() # Uncomment for detailed debugging

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Records an image to TensorBoard."""
        if self.image_log_interval <= 0:
            return  # Skip if disabled

        # Log image periodically (check done in Trainer._maybe_log_image)
        try:
            # add_image expects CHW format
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] in [1, 3, 4]:  # HWC
                    img_tensor = torch.from_numpy(image).permute(2, 0, 1)
                elif image.ndim == 2:  # HW (grayscale) -> CHW
                    img_tensor = torch.from_numpy(image).unsqueeze(0)
                else:
                    print(f"Warning: Unsupported image shape for TB: {image.shape}")
                    return
            elif isinstance(image, torch.Tensor):
                if image.ndim == 3 and image.shape[0] in [1, 3, 4]:  # CHW
                    img_tensor = image
                elif image.ndim == 2:  # HW -> CHW
                    img_tensor = image.unsqueeze(0)
                else:
                    print(
                        f"Warning: Unsupported image tensor shape for TB: {image.shape}"
                    )
                    return
            else:
                print(f"Warning: Unsupported image type for TB: {type(image)}")
                return

            self.writer.add_image(tag, img_tensor, global_step)

        except Exception as e:
            print(f"Error logging image '{tag}' at step {global_step}: {e}")
            # traceback.print_exc()

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Records hyperparameters and final metrics."""
        try:
            # Sanitize hparams: TensorBoard prefers simple types (str, bool, int, float)
            sanitized_hparams = {}
            for k, v in hparam_dict.items():
                if isinstance(v, (str, bool, int, float)):
                    sanitized_hparams[k] = v
                elif isinstance(v, (list, tuple, dict)):
                    try:
                        sanitized_hparams[k] = str(v)  # Convert complex types to string
                    except Exception:
                        sanitized_hparams[k] = "ConversionError"
                elif v is None:
                    sanitized_hparams[k] = "None"
                else:
                    sanitized_hparams[k] = str(v)  # Fallback to string conversion

            # Sanitize metrics (ensure they are numeric)
            sanitized_metrics = {}
            for k, v in metric_dict.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    sanitized_metrics[k] = v
                # else: Skip non-finite or non-numeric metrics

            self.writer.add_hparams(sanitized_hparams, sanitized_metrics)
            print("[TB Logger] Hyperparameters logged to TensorBoard.")
        except Exception as e:
            print(f"Error logging hparams: {e}")
            # traceback.print_exc()

    def record_graph(
        self, model: torch.nn.Module, input_to_model: Optional[torch.Tensor] = None
    ):
        """Records the model graph to TensorBoard."""
        if input_to_model is None:
            print("[TB Logger] Skipping graph logging: No dummy input provided.")
            return
        try:
            # Ensure model and input are on CPU for graph logging if needed
            # (Assuming model_for_graph is already on CPU from init)
            device_backup = next(model.parameters()).device
            model.cpu()
            input_cpu = input_to_model.cpu()

            # Check input shape compatibility before logging
            # This requires knowing the expected input shape of the model
            # expected_shape = (1, model.input_dim) # Example, get actual dim
            # if input_cpu.shape != expected_shape:
            #    print(f"Warning: Dummy input shape {input_cpu.shape} might not match model expectation.")

            self.writer.add_graph(model, input_cpu, verbose=False)
            print("[TB Logger] Model graph logged to TensorBoard.")
            model.to(device_backup)  # Move model back to original device
        except Exception as e:
            print(f"Error logging model graph: {e}")
            # traceback.print_exc() # Very verbose, enable if needed

    def log_summary(self, global_step: int):
        """Logs averaged statistics to TensorBoard."""
        # Console logging is handled by the parent class's log_summary
        # This method logs the *averaged* stats to TensorBoard scalars

        summary = self.get_summary(global_step)

        # Log averaged values
        self.writer.add_scalar(
            "Averages/RL Score (Window)",
            summary.get("avg_score_window", 0.0),
            global_step,
        )
        self.writer.add_scalar(
            "Averages/Episode Length (Window)",
            summary.get("avg_length_window", 0.0),
            global_step,
        )
        self.writer.add_scalar(
            "Averages/Loss (Window)", summary.get("avg_loss_window", 0.0), global_step
        )
        self.writer.add_scalar(
            "Averages/Avg Max Q (Window)",
            summary.get("avg_max_q_window", 0.0),
            global_step,
        )
        self.writer.add_scalar(
            "Averages/Game Score (Window)",
            summary.get("avg_game_score_window", 0.0),
            global_step,
        )
        self.writer.add_scalar(
            "Averages/Lines Cleared (Window)",
            summary.get("avg_lines_cleared_window", 0.0),
            global_step,
        )
        self.writer.add_scalar(
            "Averages/SPS (Window)", summary.get("avg_sps_window", 0.0), global_step
        )
        self.writer.add_scalar(
            "Averages/Learning Rate (Window)",
            summary.get("avg_lr_window", 0.0),
            global_step,
        )

    def close(self):
        """Closes the TensorBoard writer and the parent recorder."""
        print("[TB Logger] Closing TensorBoard writer...")
        try:
            # Log final metrics before closing?
            # final_metrics = {"hparam/final_best_rl_score": self.best_score, ...}
            # self.record_hparams(self.hparam_dict, final_metrics) # Need to store hparam_dict
            self.writer.flush()
            self.writer.close()
            print("[TB Logger] TensorBoard writer closed.")
        except Exception as e:
            print(f"Error closing TensorBoard writer: {e}")
        super().close()  # Call parent's close method
