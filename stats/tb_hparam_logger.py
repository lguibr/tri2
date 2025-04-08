# File: stats/tb_hparam_logger.py
import torch
from torch.utils.tensorboard import SummaryWriter
import threading
from typing import Dict, Any, Optional
import traceback


class TBHparamLogger:
    """Handles logging hyperparameters and final metrics to TensorBoard."""

    def __init__(
        self,
        writer: Optional[SummaryWriter],
        lock: threading.Lock,
        hparam_dict: Optional[Dict[str, Any]],
    ):
        self.writer = writer
        self._lock = lock
        self.hparam_dict = hparam_dict if hparam_dict else {}
        self.initial_hparams_logged = False

    def _filter_hparams(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
        """Filters hyperparameters to types supported by TensorBoard."""
        return {
            k: v
            for k, v in hparams.items()
            if isinstance(v, (int, float, str, bool, torch.Tensor))
        }

    def log_initial_hparams(self):
        """Logs hyperparameters at the beginning of the run."""
        if not self.writer or not self.hparam_dict or self.initial_hparams_logged:
            return
        with self._lock:
            try:
                initial_metrics = {
                    "hparam/final_best_rl_score": -float("inf"),
                    "hparam/final_best_game_score": -float("inf"),
                    "hparam/final_best_value_loss": float("inf"),
                    "hparam/final_best_policy_loss": float("inf"),
                    "hparam/final_total_episodes": 0,
                }
                filtered_hparams = self._filter_hparams(self.hparam_dict)
                self.writer.add_hparams(filtered_hparams, initial_metrics, run_name=".")
                self.initial_hparams_logged = True
                print("[TensorBoardStatsRecorder] Hyperparameters logged.")
            except Exception as e:
                print(f"Error logging initial hyperparameters: {e}")
                traceback.print_exc()

    def log_final_hparams(
        self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, Any]
    ):
        """Logs final hyperparameters and metrics."""
        if not self.writer:
            return
        with self._lock:
            try:
                filtered_hparams = self._filter_hparams(hparam_dict)
                filtered_metrics = {
                    k: v for k, v in metric_dict.items() if isinstance(v, (int, float))
                }
                self.writer.add_hparams(
                    filtered_hparams, filtered_metrics, run_name="."
                )
                print("[TensorBoardStatsRecorder] Final hparams and metrics logged.")
            except Exception as e:
                print(f"Error logging final hyperparameters/metrics: {e}")
                traceback.print_exc()

    def log_final_hparams_from_summary(self, final_summary: Dict[str, Any]):
        """Logs final hparams using the stored hparam_dict and metrics from summary."""
        if not self.hparam_dict:
            print(
                "[TensorBoardStatsRecorder] Skipping final hparam logging (hparam_dict not set)."
            )
            return
        final_metrics = {
            "hparam/final_best_rl_score": final_summary.get(
                "best_score", -float("inf")
            ),
            "hparam/final_best_game_score": final_summary.get(
                "best_game_score", -float("inf")
            ),
            "hparam/final_best_value_loss": final_summary.get(
                "best_value_loss", float("inf")
            ),
            "hparam/final_best_policy_loss": final_summary.get(
                "best_policy_loss", float("inf")
            ),
            "hparam/final_total_episodes": final_summary.get("total_episodes", 0),
        }
        self.log_final_hparams(self.hparam_dict, final_metrics)
