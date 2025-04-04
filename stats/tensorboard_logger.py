# File: stats/tensorboard_logger.py
import time
import warnings
from collections import deque
from typing import Deque, Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .simple_stats_recorder import SimpleStatsRecorder
from config import TensorBoardConfig, StatsConfig, DEVICE


class TensorBoardStatsRecorder(SimpleStatsRecorder):
    """
    Extends SimpleStatsRecorder to log data to TensorBoard.
    Inherits in-memory storage and console logging from SimpleStatsRecorder.
    """

    def __init__(
        self,
        log_dir: str,
        hparam_dict: Dict[str, Any],
        model_for_graph: Optional[torch.nn.Module] = None,
        dummy_input_for_graph: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # Expect tuple now
        console_log_interval: int = 50_000,
        avg_window: int = StatsConfig.STATS_AVG_WINDOW,  # Use from config
        histogram_log_interval: int = 10_000,
        image_log_interval: int = 50_000,
        # --- NEW: Pass shape Q log freq ---
        shape_q_log_interval: int = TensorBoardConfig.SHAPE_Q_LOG_FREQ,
        # --- END NEW ---
        flush_secs: int = 120,
    ):
        super().__init__(console_log_interval, avg_window)
        print(f"[TB Logger] Initializing TensorBoard writer in: {log_dir}")
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self.histogram_log_interval = (
            max(1, histogram_log_interval) if histogram_log_interval > 0 else -1
        )
        self.image_log_interval = (
            max(1, image_log_interval) if image_log_interval > 0 else -1
        )
        # --- NEW: Store shape Q log freq ---
        self.shape_q_log_interval = (
            max(1, shape_q_log_interval) if shape_q_log_interval > 0 else -1
        )
        # --- END NEW ---
        self.last_histogram_log_step = (
            -self.histogram_log_interval
        )  # Log on first opportunity
        self.last_image_log_step = -self.image_log_interval  # Log on first opportunity
        # --- NEW: Track last shape Q log step ---
        self.last_shape_q_log_step = -self.shape_q_log_interval
        # --- END NEW ---

        # Log hyperparameters immediately
        self.record_hparams(hparam_dict, {})  # Log hparams without metrics initially

        # Log model graph if provided
        if model_for_graph and dummy_input_for_graph:
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
        # Use parent method for in-memory tracking and best score updates
        super().record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            lines_cleared,
        )

        # Log episode metrics to TensorBoard
        step = global_step if global_step is not None else self.current_global_step
        if step > 0:  # Avoid logging at step 0 if not meaningful
            self.writer.add_scalar("Episode/RL Score", episode_score, step)
            self.writer.add_scalar("Episode/Length", episode_length, step)
            if game_score is not None:
                self.writer.add_scalar("Episode/Game Score", game_score, step)
            if lines_cleared is not None:
                self.writer.add_scalar("Episode/Lines Cleared", lines_cleared, step)

            # Log averages as well
            avg_score = np.mean(self.episode_scores) if self.episode_scores else 0.0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
            avg_game_score = np.mean(self.game_scores) if self.game_scores else 0.0

            self.writer.add_scalar(
                f"Episode/Avg RL Score ({self.avg_window})", avg_score, step
            )
            self.writer.add_scalar(
                f"Episode/Avg Length ({self.avg_window})", avg_length, step
            )
            self.writer.add_scalar(
                f"Episode/Avg Game Score ({self.avg_window})", avg_game_score, step
            )

    def record_step(self, step_data: Dict[str, Any]):
        # Use parent method for in-memory tracking, console logging trigger, etc.
        super().record_step(step_data)

        g_step = step_data.get("global_step", self.current_global_step)
        if g_step <= 0:
            return  # Don't log step 0 data

        # Log scalar values present in step_data
        scalar_map = {
            "loss": "Train/Loss",
            "grad_norm": "Train/Gradient Norm",
            "avg_max_q": "Train/Avg Max Q (batch)",
            "beta": "Buffer/PER Beta",
            "buffer_size": "Buffer/Size",
            "steps_per_second": "Perf/Steps Per Second",
            "lr": "Train/Learning Rate",
            "epsilon": "Train/Epsilon",  # Keep logging even if noisy
        }
        for key, tag in scalar_map.items():
            if key in step_data and step_data[key] is not None:
                try:
                    # Ensure value is float/int before logging
                    value = float(step_data[key])
                    if not np.isnan(value) and not np.isinf(value):
                        self.writer.add_scalar(tag, value, g_step)
                except (ValueError, TypeError):
                    print(
                        f"Warning: Could not convert value for '{key}' to float for TB scalar logging."
                    )

        # Log Histograms periodically
        if (
            self.histogram_log_interval > 0
            and g_step >= self.last_histogram_log_step + self.histogram_log_interval
        ):
            # Log standard histograms
            hist_map = {
                "step_rewards_batch": "Rewards/Step Reward Distribution",
                "action_batch": "Actions/Chosen Action Index Distribution",
                "batch_q_values_actions_taken": "Q-Values/Q for Chosen Actions (Batch)",  # From compute_loss via agent
                "batch_td_errors": "Train/TD Error Distribution (Batch)",
            }
            for key, tag in hist_map.items():
                if key in step_data and step_data[key] is not None:
                    self.record_histogram(tag, step_data[key], g_step)

            # --- NEW: Log Shape Selection Histograms ---
            if TensorBoardConfig.LOG_SHAPE_PLACEMENT_Q_VALUES:
                shape_hist_map = {
                    "chosen_shape_slot_batch": "Actions/Chosen Shape Slot Distribution",
                    "shape_slot_max_q_batch": "Q-Values/Max Q per Shape Slot (Batch)",
                }
                for key, tag in shape_hist_map.items():
                    if key in step_data and step_data[key] is not None:
                        self.record_histogram(tag, step_data[key], g_step)
            # --- END NEW ---

            self.last_histogram_log_step = g_step

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Records a histogram to TensorBoard."""
        if not TensorBoardConfig.LOG_HISTOGRAMS:
            return

        # Convert to numpy if it's a tensor
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        # Ensure numpy array for processing
        if not isinstance(values, np.ndarray):
            try:
                values = np.array(values)
            except Exception as e:
                print(
                    f"Warning: Could not convert values for histogram '{tag}' to numpy array: {e}"
                )
                return

        # Filter out NaN/inf values which cause TensorBoard errors
        values = values[np.isfinite(values)]

        if values.size > 0:
            try:
                self.writer.add_histogram(tag, values, global_step)
            except ValueError as e:
                print(
                    f"Warning: TensorBoard add_histogram failed for '{tag}' (likely empty or invalid data after filtering): {e}"
                )
            except Exception as e:
                print(
                    f"ERROR: Unexpected error during TensorBoard add_histogram for '{tag}': {e}"
                )
        # else:
        #     print(f"Debug: Skipping empty histogram for '{tag}' at step {global_step}")

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Records an image to TensorBoard."""
        if not TensorBoardConfig.LOG_IMAGES or self.image_log_interval <= 0:
            return

        # Check frequency handled by caller (Trainer._maybe_log_image)
        try:
            # Assuming image is CHW or HWC (numpy)
            self.writer.add_image(
                tag,
                image,
                global_step,
                dataformats=(
                    "CHW"
                    if isinstance(image, torch.Tensor) and image.ndim == 3
                    else "HWC"
                ),
            )
        except Exception as e:
            print(f"Error logging image '{tag}' to TensorBoard: {e}")
            traceback.print_exc()  # Print full traceback for image errors

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Records hyperparameters and metrics to TensorBoard HParams tab."""
        try:
            # Sanitize hparam_dict: TB only supports bool, string, float, int, None
            sanitized_hparams = {}
            for k, v in hparam_dict.items():
                if isinstance(v, (bool, str, float, int)) or v is None:
                    sanitized_hparams[k] = v
                else:
                    # Attempt to convert common types (like torch devices)
                    if isinstance(v, torch.device):
                        sanitized_hparams[k] = str(v)
                    # Add other conversions if needed
                    # Else, convert to string as fallback
                    else:
                        sanitized_hparams[k] = str(v)

            # Sanitize metric_dict: Ensure values are numeric
            sanitized_metrics = {}
            for k, v in metric_dict.items():
                if isinstance(v, (float, int)) and not np.isnan(v) and not np.isinf(v):
                    sanitized_metrics[k] = v
                # else: Skip non-numeric metrics

            self.writer.add_hparams(sanitized_hparams, sanitized_metrics)
            print(f"[TB Logger] Recorded hyperparameters.")
        except Exception as e:
            print(f"Error recording hyperparameters to TensorBoard: {e}")

    def record_graph(
        self,
        model: torch.nn.Module,
        input_to_model: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # Expect tuple
    ):
        """Records the model graph to TensorBoard."""
        if input_to_model is None:
            print("Warning: Cannot record graph without dummy input.")
            return
        if not isinstance(input_to_model, tuple) or len(input_to_model) != 2:
            print(
                f"Warning: record_graph expects a tuple of (grid_tensor, shape_tensor), got {type(input_to_model)}. Skipping graph."
            )
            return

        try:
            # Ensure model and input are on CPU for graph logging
            model.cpu()
            dummy_grid_cpu, dummy_shapes_cpu = input_to_model
            dummy_grid_cpu = dummy_grid_cpu.cpu()
            dummy_shapes_cpu = dummy_shapes_cpu.cpu()

            # Use add_graph with tuple input
            with warnings.catch_warnings():  # Suppress ONNX warnings if they occur
                warnings.simplefilter("ignore")
                self.writer.add_graph(model, (dummy_grid_cpu, dummy_shapes_cpu))
            print("[TB Logger] Recorded model graph.")
        except Exception as e:
            print(f"Error recording model graph to TensorBoard: {e}")
            # traceback.print_exc() # Optional: print traceback for debugging
        finally:
            # Move model back to its original device (important!)
            model.to(DEVICE)

    def close(self):
        """Closes the TensorBoard SummaryWriter."""
        print("[TB Logger] Closing TensorBoard writer...")
        if self.writer:
            try:
                # Log final best metrics to HParams before closing
                final_metrics = {
                    "hparam/best_rl_score": (
                        self.best_score if self.best_score > -float("inf") else 0.0
                    ),
                    "hparam/best_game_score": (
                        self.best_game_score
                        if self.best_game_score > -float("inf")
                        else 0.0
                    ),
                    "hparam/best_loss": (
                        self.best_loss if self.best_loss < float("inf") else 0.0
                    ),
                    "hparam/total_episodes": float(self.total_episodes),
                }
                # Make sure hparams were logged initially
                # self.record_hparams({}, final_metrics) # Re-logging hparams might duplicate, just flush

                self.writer.flush()
                self.writer.close()
                print("[TB Logger] TensorBoard writer closed.")
            except Exception as e:
                print(f"Error closing TensorBoard writer: {e}")
        super().close()  # Call parent close if needed
