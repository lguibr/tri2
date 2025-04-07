# File: workers.py
# File: workers.py
import threading
import queue
import time
import traceback
from typing import Optional, Dict, Any, Tuple  # Added Tuple
from collections import defaultdict

import torch
import numpy as np

from config import PPOConfig
from agent.ppo_agent import PPOAgent
from training.rollout_collector import RolloutCollector
from stats.stats_recorder import StatsRecorderBase
from training.checkpoint_manager import CheckpointManager


class EnvironmentRunner(threading.Thread):
    """Worker thread for collecting experience from environments."""

    def __init__(
        self,
        collector: RolloutCollector,
        experience_queue: queue.Queue,
        action_queue: Optional[queue.Queue],  # Keep optional for flexibility
        stop_event: threading.Event,
        pause_event: threading.Event,
        num_steps_per_rollout: int,  # Keep for reference, but loop uses storage size
        stats_recorder: StatsRecorderBase,
        name="EnvRunner",
    ):
        super().__init__(name=name, daemon=True)
        self.collector = collector
        self.experience_queue = experience_queue
        self.action_queue = action_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        # self.num_steps_per_rollout = num_steps_per_rollout # Use storage size directly
        self.stats_recorder = stats_recorder
        # Global step is now managed within the collector/aggregator
        print(f"[{self.name}] Initialized.")

    def run(self):
        print(f"[{self.name}] Starting environment runner loop.")
        rollout_count = 0
        loop_iter = 0
        try:
            while not self.stop_event.is_set():
                loop_iter += 1
                # --- Pause Handling ---
                pause_check_iter = 0
                while self.pause_event.is_set():
                    if self.stop_event.is_set():
                        break
                    # if pause_check_iter % 50 == 0:  # Log pause less frequently
                    #     print(f"[{self.name} Loop {loop_iter}] Paused (is_set={self.pause_event.is_set()})...")
                    time.sleep(0.1)  # Wait a bit before re-checking pause_event
                    pause_check_iter += 1
                if self.stop_event.is_set():  # Check stop event after pause loop
                    break
                # --- End Pause Handling ---

                steps_collected_in_rollout = 0
                rollout_start_time = time.time()
                current_global_step = 0
                if hasattr(self.collector.stats_recorder, "aggregator"):
                    current_global_step = getattr(
                        self.collector.stats_recorder.aggregator.storage,
                        "current_global_step",
                        0,
                    )

                # --- Modified Loop Condition ---
                # Loop while the storage is not full
                while (
                    self.collector.rollout_storage.step
                    < self.collector.rollout_storage.num_steps
                ):
                    # Check for pause/stop *before* collecting the step
                    if self.stop_event.is_set() or self.pause_event.is_set():
                        break  # Exit inner loop, will re-check pause at outer loop start

                    # Pass the *current* global step to collect_one_step for potential use
                    steps_this_iter = self.collector.collect_one_step(
                        current_global_step
                    )
                    steps_collected_in_rollout += steps_this_iter
                    time.sleep(
                        0.0001
                    )  # Small sleep to prevent tight loop if collection is very fast
                # --- End Modified Loop Condition ---

                # Check if the loop was broken by pause/stop before processing rollout
                if self.stop_event.is_set() or self.pause_event.is_set():
                    continue  # Go back to outer loop to handle pause/stop

                # --- Check if storage is actually full after loop ---
                # This ensures we only process complete rollouts
                if (
                    self.collector.rollout_storage.step
                    == self.collector.rollout_storage.num_steps
                ):
                    if steps_collected_in_rollout > 0:
                        rollout_duration = time.time() - rollout_start_time
                        sps = steps_collected_in_rollout / max(1e-6, rollout_duration)

                        self.collector.compute_advantages_for_storage()
                        rollout_data_cpu = (
                            self.collector.rollout_storage.get_data_for_update()
                        )

                        if rollout_data_cpu:
                            # --- Check stop event before blocking put ---
                            if self.stop_event.is_set():
                                break
                            # --- End check ---
                            try:
                                # Reduced timeout
                                self.experience_queue.put(
                                    rollout_data_cpu, block=True, timeout=1.0
                                )
                            except queue.Full:
                                print(
                                    f"[{self.name}] WARNING: Experience queue full after 1s timeout. Discarding rollout."
                                )
                            except Exception as q_err:
                                print(
                                    f"[{self.name}] ERROR putting data onto queue: {q_err}"
                                )
                                traceback.print_exc()
                        else:
                            print(
                                f"[{self.name}] No rollout data generated, skipping queue put."
                            )

                        self.collector.rollout_storage.after_update()
                        rollout_count += 1
                    else:
                        print(
                            f"[{self.name}] Warning: Storage full but no steps collected in rollout?"
                        )
                # else: # DEBUG REMOVED
                # print(f"[{self.name}] Rollout loop finished prematurely (step={self.collector.rollout_storage.step}/{self.collector.rollout_storage.num_steps}). Likely paused/stopped.")

                time.sleep(0.001)  # Small sleep between rollouts

        except Exception as e:
            print(f"[{self.name}] CRITICAL ERROR in environment runner loop: {e}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            print(f"[{self.name}] Environment runner loop finished.")


class TrainingWorker(threading.Thread):
    """Worker thread for performing agent updates."""

    def __init__(
        self,
        agent: PPOAgent,
        experience_queue: queue.Queue,
        stop_event: threading.Event,
        pause_event: threading.Event,
        stats_recorder: StatsRecorderBase,
        ppo_config: PPOConfig,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        name="Trainer",
    ):
        super().__init__(name=name, daemon=True)
        self.agent = agent
        self.experience_queue = experience_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.stats_recorder = stats_recorder
        self.ppo_config = ppo_config
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        # Global step is managed by aggregator, worker just reads/uses it
        self.rollouts_processed = 0

        # --- State for Update Progress Tracking ---
        self._progress_lock = threading.Lock()
        self.current_update_epoch = 0
        self.total_update_epochs = self.ppo_config.PPO_EPOCHS
        self.current_minibatch_in_epoch = 0
        self.total_minibatches_in_epoch = 0
        self.update_start_time = 0.0
        # --- End Update Progress State ---

        print(f"[{self.name}] Initialized.")

    def get_update_progress_details(self) -> Dict[str, Any]:
        """Returns detailed progress information for the current update phase (thread-safe)."""
        with self._progress_lock:
            overall_progress = 0.0
            epoch_progress = 0.0
            if self.total_update_epochs > 0 and self.total_minibatches_in_epoch > 0:
                total_minibatches_overall = (
                    self.total_update_epochs * self.total_minibatches_in_epoch
                )
                minibatches_done_overall = max(
                    0,
                    (self.current_update_epoch - 1) * self.total_minibatches_in_epoch
                    + self.current_minibatch_in_epoch,
                )
                overall_progress = min(
                    1.0, minibatches_done_overall / max(1, total_minibatches_overall)
                )
                epoch_progress = min(
                    1.0,
                    self.current_minibatch_in_epoch
                    / max(1, self.total_minibatches_in_epoch),
                )

            return {
                "overall_progress": overall_progress,
                "epoch_progress": epoch_progress,
                "current_epoch": self.current_update_epoch,
                "total_epochs": self.total_update_epochs,
                "phase": "Updating Agent",  # Assume this is only called when updating
                "update_start_time": self.update_start_time,
                "num_minibatches_per_epoch": self.total_minibatches_in_epoch,
                "current_minibatch_index": self.current_minibatch_in_epoch,
            }

    def run(self):
        print(f"[{self.name}] Starting training worker loop.")
        update_count = 0
        loop_iter = 0
        try:
            while not self.stop_event.is_set():
                loop_iter += 1
                # --- Pause Handling ---
                pause_check_iter = 0
                while self.pause_event.is_set():
                    if self.stop_event.is_set():
                        break
                    # if pause_check_iter % 50 == 0:  # Log pause less frequently
                    #     print(f"[{self.name} Loop {loop_iter}] Paused (is_set={self.pause_event.is_set()})...")
                    time.sleep(0.1)  # Wait a bit before re-checking pause_event
                    pause_check_iter += 1
                if self.stop_event.is_set():  # Check stop event after pause loop
                    break
                # --- End Pause Handling ---

                try:
                    # Use a timeout to allow checking stop/pause events periodically
                    rollout_data_cpu = self.experience_queue.get(
                        block=True, timeout=0.1
                    )
                except queue.Empty:
                    continue  # Go back to check stop/pause events
                except Exception as q_err:
                    print(f"[{self.name}] ERROR getting data from queue: {q_err}")
                    traceback.print_exc()
                    time.sleep(0.1)
                    continue

                # --- Check stop event after getting data ---
                if self.stop_event.is_set():
                    break
                # --- End check ---

                # --- Reset Progress Tracking for New Update ---
                with self._progress_lock:
                    self.update_start_time = time.time()
                    self.current_update_epoch = 0
                    self.current_minibatch_in_epoch = 0
                    self.total_minibatches_in_epoch = 0  # Will be calculated below
                # --- End Reset Progress ---

                update_count += 1
                print(f"\n[{self.name}] Starting Update #{update_count}...")

                # Check if 'actions' key exists and has data
                if (
                    "actions" not in rollout_data_cpu
                    or rollout_data_cpu["actions"] is None
                ):
                    print(
                        f"[{self.name}] Warning: Received rollout data missing 'actions'. Skipping update."
                    )
                    continue
                num_samples = rollout_data_cpu["actions"].shape[0]
                if num_samples == 0:
                    print(
                        f"[{self.name}] Warning: Received empty rollout data. Skipping update."
                    )
                    continue

                # --- PPO Update Loop ---
                update_error_occurred = False  # Flag to track errors within update
                all_minibatch_metrics = defaultdict(list)  # Accumulate metrics

                # Normalize advantages (should already be done, but can be done here too)
                advantages = rollout_data_cpu["advantages"]
                if advantages is not None and advantages.numel() > 0:
                    rollout_data_cpu["advantages"] = (
                        advantages - advantages.mean()
                    ) / (advantages.std() + 1e-8)
                else:
                    print(
                        f"[{self.name}] Warning: Advantages are None or empty. Skipping normalization and update."
                    )
                    continue

                # Move data to device (handle potential errors)
                rollout_data_device = None
                try:
                    rollout_data_device = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in rollout_data_cpu.items()
                        if isinstance(v, torch.Tensor)
                    }
                    # Handle non-tensor data (like initial_lstm_state)
                    if (
                        "initial_lstm_state" in rollout_data_cpu
                        and rollout_data_cpu["initial_lstm_state"] is not None
                    ):
                        h, c = rollout_data_cpu["initial_lstm_state"]
                        rollout_data_device["initial_lstm_state"] = (
                            h.to(self.device, non_blocking=True),
                            c.to(self.device, non_blocking=True),
                        )
                except Exception as move_err:
                    print(
                        f"[{self.name}] Error moving rollout data to device {self.device}: {move_err}"
                    )
                    traceback.print_exc()
                    update_error_occurred = True  # Mark error occurred

                if not update_error_occurred:
                    for epoch in range(self.ppo_config.PPO_EPOCHS):
                        with self._progress_lock:
                            self.current_update_epoch = epoch + 1
                            self.current_minibatch_in_epoch = 0
                            # Calculate total minibatches for this epoch (can vary slightly if num_samples not divisible)
                            self.total_minibatches_in_epoch = (
                                num_samples + self.ppo_config.MINIBATCH_SIZE - 1
                            ) // self.ppo_config.MINIBATCH_SIZE

                        # --- Epoch Start Log ---
                        print(
                            f"  [{self.name}] Starting Epoch {epoch+1}/{self.ppo_config.PPO_EPOCHS}..."
                        )
                        # --- End Epoch Start Log ---

                        # Check pause/stop before starting epoch
                        if self.stop_event.is_set() or self.pause_event.is_set():
                            update_error_occurred = (
                                True  # Treat pause as needing to stop the update cycle
                            )
                            break
                        if (
                            update_error_occurred
                        ):  # Check if error occurred in previous epoch
                            break

                        indices = np.arange(num_samples)
                        np.random.shuffle(indices)
                        num_minibatches_this_epoch = 0

                        for start_idx in range(
                            0, num_samples, self.ppo_config.MINIBATCH_SIZE
                        ):
                            with self._progress_lock:
                                self.current_minibatch_in_epoch = (
                                    num_minibatches_this_epoch + 1
                                )

                            # Check pause/stop before processing minibatch
                            if self.stop_event.is_set() or self.pause_event.is_set():
                                update_error_occurred = True  # Treat pause as needing to stop the update cycle
                                break
                            if (
                                update_error_occurred
                            ):  # Check if error occurred in previous minibatch
                                break

                            end_idx = start_idx + self.ppo_config.MINIBATCH_SIZE
                            minibatch_indices = indices[start_idx:end_idx]
                            minibatch_size = len(minibatch_indices)

                            if minibatch_size < 2:  # Avoid tiny batches
                                continue

                            try:
                                # --- Time the minibatch update ---
                                mb_start_time = time.time()

                                # Select minibatch data (already on device)
                                minibatch_device = {
                                    key: rollout_data_device[key][minibatch_indices]
                                    for key in [
                                        "obs_grid",
                                        "obs_shapes",
                                        "obs_availability",
                                        "obs_explicit_features",
                                        "actions",
                                        "log_probs",
                                        "returns",
                                        "advantages",
                                    ]
                                    if key in rollout_data_device  # Check key exists
                                }

                                # Perform update using agent's method (which includes lock)
                                minibatch_metrics = self.agent.update_minibatch(
                                    minibatch_device
                                )

                                # --- Calculate and log minibatch SPS ---
                                mb_duration = time.time() - mb_start_time
                                minibatch_sps = minibatch_size / max(1e-9, mb_duration)
                                minibatch_metrics["minibatch_update_sps"] = (
                                    minibatch_sps
                                )
                                # --- End minibatch SPS calculation ---

                                # --- Accumulate minibatch metrics ---
                                for k, v in minibatch_metrics.items():
                                    all_minibatch_metrics[k].append(v)
                                # --- End accumulate ---

                                num_minibatches_this_epoch += 1

                                # --- Minibatch Progress Log (less frequent) ---
                                if (
                                    num_minibatches_this_epoch
                                    % max(1, self.total_minibatches_in_epoch // 4)
                                    == 0
                                ):
                                    print(
                                        f"    [{self.name}] Epoch {epoch+1}: Minibatch {num_minibatches_this_epoch}/{self.total_minibatches_in_epoch} done."
                                    )
                                # --- End Minibatch Progress Log ---

                            except KeyError as ke:
                                print(
                                    f"[{self.name}] ERROR: Missing key in minibatch data: {ke}"
                                )
                                update_error_occurred = True
                            except Exception as update_err:
                                print(
                                    f"[{self.name}] CRITICAL ERROR during agent.update_minibatch: {update_err}"
                                )
                                traceback.print_exc()
                                update_error_occurred = True
                            finally:
                                # Clean up minibatch tensors if needed (though data is sliced from larger tensor)
                                del minibatch_device
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()  # Maybe call less often?
                        # End minibatch loop
                        if update_error_occurred:  # Break epoch loop if error occurred
                            break
                        # --- Epoch End Log ---
                        print(
                            f"  [{self.name}] Epoch {epoch+1} finished ({num_minibatches_this_epoch} minibatches)."
                        )
                        # --- End Epoch End Log ---
                    # End epoch loop

                # --- End PPO Update Loop ---

                del rollout_data_device
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                # --- Log overall update stats AFTER the update cycle ---
                if not update_error_occurred:
                    update_duration = (
                        time.time() - self.update_start_time
                    )  # Use tracked start time

                    # --- Update Global Step in Aggregator ---
                    steps_this_update = num_samples
                    new_global_step = 0
                    if hasattr(self.stats_recorder, "aggregator"):
                        with self.stats_recorder.aggregator._lock:
                            self.stats_recorder.aggregator.storage.current_global_step += (
                                steps_this_update
                            )
                            new_global_step = (
                                self.stats_recorder.aggregator.storage.current_global_step
                            )
                    # --- End Update Global Step ---

                    # Calculate Overall Update SPS
                    overall_update_sps = num_samples / max(1e-6, update_duration)

                    # --- Prepare aggregated metrics for logging ---
                    step_record_data = {
                        "update_time": update_duration,
                        "lr": self.agent.optimizer.param_groups[0]["lr"],
                        "global_step": new_global_step,  # Log against the NEW step
                        "training_target_step": self.checkpoint_manager.training_target_step,
                        "update_steps_per_second": overall_update_sps,  # Log the overall SPS here
                    }
                    # Add mean of accumulated minibatch metrics
                    for k, v_list in all_minibatch_metrics.items():
                        if v_list:
                            step_record_data[k] = np.mean(v_list)
                    # --- End prepare aggregated metrics ---

                    # --- Log aggregated step metrics ---
                    self.stats_recorder.record_step(step_record_data)
                    # --- End log aggregated step metrics ---

                    # Increment rollout counter for histogram/image logging frequency checks
                    if (
                        hasattr(self.stats_recorder, "histogram_logger")
                        and self.stats_recorder.histogram_logger
                    ):
                        self.stats_recorder.histogram_logger.increment_rollout_counter()
                        # Reset counter only if logging actually happened (checked internally by logger)
                        if self.stats_recorder.histogram_logger.should_log(
                            new_global_step
                        ):
                            self.stats_recorder.histogram_logger.reset_rollout_counter()
                    if (
                        hasattr(self.stats_recorder, "image_logger")
                        and self.stats_recorder.image_logger
                    ):
                        self.stats_recorder.image_logger.increment_rollout_counter()
                        # Reset counter only if logging actually happened
                        if self.stats_recorder.image_logger.should_log(new_global_step):
                            self.stats_recorder.image_logger.reset_rollout_counter()

                    self.rollouts_processed += 1
                    print(
                        f"[{self.name}] Update #{update_count} finished in {update_duration:.2f}s (Overall SPS: {overall_update_sps:.0f})."
                    )

                elif update_error_occurred:
                    print(
                        f"[{self.name}] Update #{update_count} skipped or interrupted due to error/pause/stop."
                    )
                else:
                    print(
                        f"[{self.name}] Update #{update_count} skipped (no minibatches processed)."
                    )

                # Reset progress tracking after update finishes or errors out
                with self._progress_lock:
                    self.current_update_epoch = 0
                    self.current_minibatch_in_epoch = 0
                    self.total_minibatches_in_epoch = 0
                    self.update_start_time = 0.0

                time.sleep(0.001)  # Small sleep between updates

        except Exception as e:
            print(f"[{self.name}] CRITICAL ERROR in training worker loop: {e}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            print(f"[{self.name}] Training worker loop finished.")
