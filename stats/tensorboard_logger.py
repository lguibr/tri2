# File: stats/tensorboard_logger.py
import time
import traceback
from collections import deque
from typing import (
    Deque,
    Dict,
    Any,
    Optional,
    Union,
    List,
    Tuple,
    TYPE_CHECKING,
)  # Added TYPE_CHECKING
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import threading

from .stats_recorder import StatsRecorderBase
from .aggregator import StatsAggregator
from .simple_stats_recorder import SimpleStatsRecorder
from config import (
    TensorBoardConfig,
    EnvConfig,
    RNNConfig,
    TransformerConfig,
)
from agent.alphazero_net import AlphaZeroNet

from .tb_log_utils import format_image_for_tb
from .tb_scalar_logger import TBScalarLogger
from .tb_histogram_logger import TBHistogramLogger
from .tb_image_logger import TBImageLogger
from .tb_hparam_logger import TBHparamLogger

if TYPE_CHECKING:
    from environment.game_state import GameState  # Import for type hinting


class TensorBoardStatsRecorder(StatsRecorderBase):
    """
    Logs aggregated statistics, histograms, images, and hyperparameters to TensorBoard. Thread-safe.
    Uses a SimpleStatsRecorder for console logging and a StatsAggregator for data handling.
    Delegates specific logging tasks to helper classes.
    """

    def __init__(
        self,
        aggregator: StatsAggregator,
        console_recorder: SimpleStatsRecorder,
        log_dir: str,
        hparam_dict: Optional[Dict[str, Any]] = None,
        model_for_graph: Optional[AlphaZeroNet] = None,
        dummy_input_for_graph: Optional[Dict[str, torch.Tensor]] = None,
        histogram_log_interval: int = TensorBoardConfig.HISTOGRAM_LOG_FREQ,
        image_log_interval: int = TensorBoardConfig.IMAGE_LOG_FREQ,
        env_config: Optional[EnvConfig] = None,
        rnn_config: Optional[RNNConfig] = None,
        transformer_config: Optional[TransformerConfig] = None,
    ):
        self.aggregator = aggregator
        self.console_recorder = console_recorder
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None
        self._lock = threading.Lock()

        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"[TensorBoardStatsRecorder] Initialized. Logging to: {self.log_dir}")

            self.scalar_logger = TBScalarLogger(self.writer, self._lock)
            self.histogram_logger = TBHistogramLogger(
                self.writer, self._lock, histogram_log_interval
            )
            self.image_logger = TBImageLogger(
                self.writer, self._lock, image_log_interval
            )
            self.hparam_logger = TBHparamLogger(self.writer, self._lock, hparam_dict)

            if model_for_graph and dummy_input_for_graph:
                self.record_graph(model_for_graph, dummy_input_for_graph)
            else:
                print("[TensorBoardStatsRecorder] Model graph logging skipped.")

            self.hparam_logger.log_initial_hparams()

        except Exception as e:
            print(f"FATAL: Error initializing TensorBoard SummaryWriter: {e}")
            traceback.print_exc()
            self.writer = None
            self.scalar_logger = None
            self.histogram_logger = None
            self.image_logger = None
            self.hparam_logger = None

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
        game_score: Optional[int] = None,
        triangles_cleared: Optional[int] = None,
        game_state_for_best: Optional["GameState"] = None,  # Added optional GameState
    ):
        """Records episode stats to TensorBoard and delegates to console recorder."""
        update_info = self.aggregator.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
            game_state_for_best,  # Pass GameState down
        )
        g_step = (
            global_step
            if global_step is not None
            else getattr(self.aggregator.storage, "current_global_step", 0)
        )

        if self.scalar_logger:
            self.scalar_logger.log_episode_scalars(
                g_step,
                episode_score,
                episode_length,
                episode_num,
                game_score,
                triangles_cleared,
                update_info,
                self.aggregator,
            )

        self.console_recorder.record_episode(
            episode_score,
            episode_length,
            episode_num,
            global_step,
            game_score,
            triangles_cleared,
            game_state_for_best,  # Pass GameState down to console recorder
        )

    def record_step(self, step_data: Dict[str, Any]):
        """Records step stats (e.g., NN training step) to TensorBoard and console."""
        update_info = self.aggregator.record_step(step_data)
        g_step = step_data.get(
            "global_step",
            getattr(self.aggregator.storage, "current_global_step", 0),
        )

        if self.scalar_logger:
            self.scalar_logger.log_step_scalars(
                g_step, step_data, update_info, self.aggregator
            )

        if "policy_loss" in step_data or "value_loss" in step_data:
            if self.histogram_logger:
                self.histogram_logger.increment_rollout_counter()
                if self.histogram_logger.should_log(g_step):
                    self.histogram_logger.reset_rollout_counter()
            if self.image_logger:
                self.image_logger.increment_rollout_counter()
                if self.image_logger.should_log(g_step):
                    self.image_logger.reset_rollout_counter()

        self.console_recorder.record_step(step_data)

    def record_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor, List[float]],
        global_step: int,
    ):
        """Records a histogram to TensorBoard using the helper."""
        if self.histogram_logger:
            self.histogram_logger.log_histogram(tag, values, global_step)

    def record_image(
        self, tag: str, image: Union[np.ndarray, torch.Tensor], global_step: int
    ):
        """Records an image to TensorBoard using the helper."""
        if self.image_logger:
            self.image_logger.log_image(tag, image, global_step)

    def record_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]):
        """Records final hyperparameters and metrics using the helper."""
        if self.hparam_logger:
            self.hparam_logger.log_final_hparams(hparam_dict, metric_dict)

    def record_graph(
        self,
        model: AlphaZeroNet,
        input_to_model: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Records the model graph to TensorBoard."""
        if not self.writer:
            return
        if input_to_model is None:
            print("Warning: Cannot record graph without dummy input.")
            return
        with self._lock:
            try:
                original_device = next(iter(model.parameters())).device
                model.cpu()
                dummy_input_cpu = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in input_to_model.items()
                }
                self.writer.add_graph(
                    model, input_to_model=dummy_input_cpu, verbose=False
                )
                print("[TensorBoardStatsRecorder] Model graph logged.")
                model.to(original_device)
            except Exception as e:
                print(f"Error logging model graph: {e}.")
                traceback.print_exc()
                try:
                    model.to(original_device)
                except Exception:
                    pass

    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        return self.aggregator.get_summary(current_global_step)

    def get_plot_data(self) -> Dict[str, Deque]:
        return self.aggregator.get_plot_data()

    def log_summary(self, global_step: int):
        self.console_recorder.log_summary(global_step)

    def close(self, is_cleanup: bool = False):
        """Closes the TensorBoard writer and logs final hparams unless cleaning up."""
        print(f"[TensorBoardStatsRecorder] Close called (is_cleanup={is_cleanup})...")
        if not self.writer:
            print(
                "[TensorBoardStatsRecorder] Writer was not initialized or already closed."
            )
            self.console_recorder.close(is_cleanup=is_cleanup)
            return

        with self._lock:
            print("[TensorBoardStatsRecorder] Acquired lock for closing.")
            try:
                if not is_cleanup and self.hparam_logger:
                    print("[TensorBoardStatsRecorder] Logging final hparams...")
                    final_step = getattr(
                        self.aggregator.storage, "current_global_step", 0
                    )
                    final_summary = self.get_summary(final_step)
                    self.hparam_logger.log_final_hparams_from_summary(final_summary)
                    print("[TensorBoardStatsRecorder] Final hparams logged.")
                elif is_cleanup:
                    print(
                        "[TensorBoardStatsRecorder] Skipping final hparams logging due to cleanup."
                    )

                print("[TensorBoardStatsRecorder] Flushing writer...")
                self.writer.flush()
                print("[TensorBoardStatsRecorder] Writer flushed.")
                print("[TensorBoardStatsRecorder] Closing writer...")
                self.writer.close()
                self.writer = None
                print("[TensorBoardStatsRecorder] Writer closed successfully.")
            except Exception as e:
                print(f"[TensorBoardStatsRecorder] Error during writer close: {e}")
                traceback.print_exc()
            finally:
                print(
                    "[TensorBoardStatsRecorder] Releasing lock after closing attempt."
                )

        self.console_recorder.close(is_cleanup=is_cleanup)
        print("[TensorBoardStatsRecorder] Close method finished.")
