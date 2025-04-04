# File: stats/wandb_logger.py
# <<< NEW FILE >>>
import time
import wandb
from collections import deque
from typing import Deque, Dict, Any, Optional
import numpy as np
from stats.stats_recorder import StatsRecorderBase, SimpleStatsRecorder


class WandbStatsRecorder(SimpleStatsRecorder):
    """
    Extends SimpleStatsRecorder to log summaries to Weights & Biases.
    Inherits in-memory averaging logic.
    """

    def __init__(
        self,
        console_log_interval: int = 0,
        avg_window: int = 500,
        wandb_log_interval: int = 10_000,
    ):
        # Initialize SimpleStatsRecorder for averaging
        super().__init__(
            console_log_interval=console_log_interval, avg_window=avg_window
        )

        self.wandb_log_interval = max(1, wandb_log_interval)
        self.last_wandb_log_step = 0

        if not wandb.run:
            print(
                "Warning: WandbStatsRecorder initialized but wandb.run is None. WandB logging disabled."
            )
            self.wandb_enabled = False
        else:
            print(
                f"[WandbStatsRecorder] Initialized. WandB Log Interval: {self.wandb_log_interval} steps. Avg Window: {self.avg_window}"
            )
            self.wandb_enabled = True
            # Define metrics for WandB step axis (optional but good practice)
            try:
                wandb.define_metric("global_step")
                # Define other metrics to use global_step as x-axis
                metrics_to_define = [
                    f"avg_score_{self.avg_window}",
                    f"avg_length_{self.avg_window}",
                    f"avg_loss_{self.avg_window}",
                    f"avg_grad_{self.avg_window}",
                    f"avg_max_q_{self.avg_window}",
                    f"avg_game_score_{self.avg_window}",
                    f"avg_lines_cleared_{self.avg_window}",
                    "steps_per_second",
                    "beta",
                    "buffer_size",
                    "total_episodes",
                    "best_score",
                    "best_game_score",
                    "total_lines_cleared",
                ]
                for metric in metrics_to_define:
                    wandb.define_metric(metric, step_metric="global_step")
            except Exception as e:
                print(f"Warning: Failed to define WandB metrics: {e}")

    def log_summary(self, global_step: int):
        """Logs summary stats to console (if enabled) and WandB."""
        # Call parent's log_summary for console logging (if interval > 0)
        super().log_summary(global_step)

        # Log to WandB if enabled and interval is met
        if (
            self.wandb_enabled
            and wandb.run
            and (global_step >= self.last_wandb_log_step + self.wandb_log_interval)
        ):
            summary = self.get_summary(global_step)  # Get averaged stats

            # Prepare data for wandb.log, maybe prefix keys for clarity
            log_data = {
                f"stats/{k}": v for k, v in summary.items() if k != "global_step"
            }  # Exclude global_step from dict itself
            log_data["global_step"] = global_step  # Add global_step for the x-axis

            try:
                wandb.log(
                    log_data, step=global_step
                )  # Use global_step as the step argument
                self.last_wandb_log_step = global_step  # Update last log step for WandB
            except Exception as e:
                print(f"Error logging to WandB: {e}")

    def close(self):
        """Finishes the WandB run."""
        if self.wandb_enabled and wandb.run:
            try:
                print("[WandbStatsRecorder] Finishing WandB run...")
                wandb.finish()
                print("[WandbStatsRecorder] WandB run finished.")
            except Exception as e:
                print(f"Error finishing WandB run: {e}")
        else:
            print("[WandbStatsRecorder] Closed (WandB was not active).")
