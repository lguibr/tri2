# File: training/checkpoint_manager.py
# File: training/checkpoint_manager.py
import os
import torch
import traceback
import re
import time
from typing import Optional, Dict, Any, Tuple
import pickle  # Import pickle for the specific error type

from agent.ppo_agent import PPOAgent
from utils.running_mean_std import RunningMeanStd
from stats.aggregator import StatsAggregator


def find_latest_run_and_checkpoint(
    base_checkpoint_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Finds the latest run directory and the latest checkpoint within it.
    Returns (run_id, checkpoint_path) or (None, None).
    """
    latest_run_id = None
    latest_run_mtime = 0

    if not os.path.isdir(base_checkpoint_dir):
        print(
            f"[CheckpointFinder] Base checkpoint directory not found: {base_checkpoint_dir}"
        )
        return None, None

    # 1. Find the latest run directory
    try:
        for item in os.listdir(base_checkpoint_dir):
            item_path = os.path.join(base_checkpoint_dir, item)
            if os.path.isdir(item_path) and item.startswith("run_"):
                try:
                    mtime = os.path.getmtime(item_path)
                    if mtime > latest_run_mtime:
                        latest_run_mtime = mtime
                        latest_run_id = item
                except OSError:
                    continue  # Ignore directories we can't access
    except OSError as e:
        print(
            f"[CheckpointFinder] Error listing base checkpoint directory {base_checkpoint_dir}: {e}"
        )
        return None, None

    if latest_run_id is None:
        print(f"[CheckpointFinder] No run directories found in {base_checkpoint_dir}.")
        return None, None

    latest_run_dir = os.path.join(base_checkpoint_dir, latest_run_id)
    print(f"[CheckpointFinder] Identified latest run directory: {latest_run_dir}")

    # 2. Find the latest checkpoint within that directory
    latest_checkpoint_path = find_latest_checkpoint_in_dir(latest_run_dir)

    if latest_checkpoint_path:
        print(
            f"[CheckpointFinder] Found latest checkpoint in run '{latest_run_id}': {os.path.basename(latest_checkpoint_path)}"
        )
        return latest_run_id, latest_checkpoint_path
    else:
        print(
            f"[CheckpointFinder] No valid checkpoints found in the latest run directory: {latest_run_dir}"
        )
        return latest_run_id, None  # Return run_id even if no checkpoint found in it


def find_latest_checkpoint_in_dir(checkpoint_dir: str) -> Optional[str]:
    """
    Finds the latest checkpoint file (step_*.pth or FINAL_*.pth) in a specific directory.
    Prioritizes FINAL checkpoint if it exists.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = []
    final_checkpoint = None
    step_pattern = re.compile(r"step_(\d+)_agent_state\.pth")

    try:
        for filename in os.listdir(checkpoint_dir):
            full_path = os.path.join(checkpoint_dir, filename)
            if not os.path.isfile(full_path):
                continue

            if filename == "FINAL_agent_state.pth":
                final_checkpoint = full_path
                # Prioritize FINAL checkpoint
                return final_checkpoint  # Return immediately if FINAL found
            else:
                match = step_pattern.match(filename)
                if match:
                    step = int(match.group(1))
                    checkpoints.append((step, full_path))

    except OSError as e:
        print(f"[CheckpointFinder] Error listing directory {checkpoint_dir}: {e}")
        return None

    if not checkpoints:
        return None

    # Sort by step number (descending) and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


class CheckpointManager:
    """Handles loading and saving of agent states, observation normalization, and stats."""

    def __init__(
        self,
        agent: PPOAgent,
        stats_aggregator: StatsAggregator,
        base_checkpoint_dir: str,  # Use base directory
        run_checkpoint_dir: str,  # Current run's checkpoint dir for saving
        load_checkpoint_path_config: Optional[str],  # Explicit path from config
        device: torch.device,
        obs_rms_dict: Optional[Dict[str, RunningMeanStd]] = None,
    ):
        self.agent = agent
        self.stats_aggregator = stats_aggregator
        self.base_checkpoint_dir = base_checkpoint_dir
        self.run_checkpoint_dir = (
            run_checkpoint_dir  # Directory for *saving* checkpoints for the current run
        )
        self.device = device
        self.obs_rms_dict = obs_rms_dict if obs_rms_dict else {}

        self.global_step = 0
        self.episode_count = 0
        # training_target_step will be loaded from checkpoint or set by Trainer
        self.training_target_step = 0

        # --- Determine Checkpoint to Load ---
        self.run_id_to_load_from: Optional[str] = None
        self.checkpoint_path_to_load: Optional[str] = None

        if load_checkpoint_path_config:
            # Priority 1: Explicit path from config
            print(
                f"[CheckpointManager] Using explicit checkpoint path from config: {load_checkpoint_path_config}"
            )
            if os.path.isfile(load_checkpoint_path_config):
                self.checkpoint_path_to_load = load_checkpoint_path_config
                # Try to extract run_id from the path (best effort)
                try:
                    # Assumes path structure like .../checkpoints/run_XXXX/step_YYY.pth
                    parent_dir = os.path.dirname(load_checkpoint_path_config)
                    self.run_id_to_load_from = os.path.basename(parent_dir)
                    if not self.run_id_to_load_from.startswith("run_"):
                        self.run_id_to_load_from = None  # Invalid format
                except Exception:
                    self.run_id_to_load_from = None
                if self.run_id_to_load_from:
                    print(
                        f"[CheckpointManager] Extracted run_id '{self.run_id_to_load_from}' from explicit path."
                    )
                else:
                    print(
                        "[CheckpointManager] Could not determine run_id from explicit path."
                    )
            else:
                print(
                    f"[CheckpointManager] WARNING: Explicit checkpoint path not found: {load_checkpoint_path_config}. Starting fresh."
                )
        else:
            # Priority 2: Auto-resume from the overall latest run
            print(
                f"[CheckpointManager] No explicit checkpoint path. Searching for latest run in: {self.base_checkpoint_dir}"
            )
            latest_run_id, latest_checkpoint_path = find_latest_run_and_checkpoint(
                self.base_checkpoint_dir
            )
            if latest_run_id and latest_checkpoint_path:
                print(
                    f"[CheckpointManager] Found latest run '{latest_run_id}' with checkpoint: {os.path.basename(latest_checkpoint_path)}"
                )
                self.run_id_to_load_from = latest_run_id
                self.checkpoint_path_to_load = latest_checkpoint_path
            elif latest_run_id:
                print(
                    f"[CheckpointManager] Found latest run directory '{latest_run_id}' but no valid checkpoint inside. Starting fresh."
                )
            else:
                print(
                    f"[CheckpointManager] No previous runs found in {self.base_checkpoint_dir}. Starting fresh."
                )

        # Note: Actual loading happens in the load_checkpoint method, called externally.

    def get_run_id_to_load_from(self) -> Optional[str]:
        """Returns the run_id determined during initialization, if any."""
        return self.run_id_to_load_from

    def get_checkpoint_path_to_load(self) -> Optional[str]:
        """Returns the checkpoint path determined during initialization, if any."""
        return self.checkpoint_path_to_load

    def load_checkpoint(self):
        """
        Loads agent state, observation normalization, and stats aggregator state
        using the path determined during initialization.
        """
        if not self.checkpoint_path_to_load:
            print(
                "[CheckpointManager] No checkpoint path specified for loading. Skipping load."
            )
            # Ensure aggregator episode count matches if starting fresh
            self.stats_aggregator.total_episodes = self.episode_count
            return

        if not os.path.isfile(self.checkpoint_path_to_load):
            print(
                f"[CheckpointManager] LOAD ERROR: Checkpoint file not found: {self.checkpoint_path_to_load}"
            )
            self._reset_all_states()  # Reset state if specified file doesn't exist
            return

        print(
            f"[CheckpointManager] Loading checkpoint from: {self.checkpoint_path_to_load}"
        )
        try:
            # Load checkpoint onto the correct device immediately
            checkpoint = torch.load(
                self.checkpoint_path_to_load,
                map_location=self.device,
                weights_only=False,
            )

            # --- Load Agent State ---
            if "agent_state_dict" in checkpoint:
                self.agent.load_state_dict(checkpoint["agent_state_dict"])
                print("  -> Agent state loaded successfully.")
            else:
                print(
                    "  -> WARNING: 'agent_state_dict' key missing. Agent state NOT loaded."
                )

            # --- Load Global Step ---
            self.global_step = checkpoint.get("global_step", 0)
            print(f"  -> Loaded Global Step: {self.global_step}")

            # --- Load Stats Aggregator State ---
            if "stats_aggregator_state_dict" in checkpoint and self.stats_aggregator:
                try:
                    self.stats_aggregator.load_state_dict(
                        checkpoint["stats_aggregator_state_dict"]
                    )
                    print("  -> Stats Aggregator state loaded successfully.")
                    # Overwrite episode_count with the one from the aggregator for consistency
                    self.episode_count = self.stats_aggregator.total_episodes
                    # Load the training target step from the aggregator state
                    self.training_target_step = (
                        self.stats_aggregator.training_target_step
                    )
                    # Log the loaded start time
                    loaded_start_time = self.stats_aggregator.start_time
                    print(
                        f"  -> Loaded Run Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(loaded_start_time))}"
                    )
                    print(
                        f"  -> Loaded Training Target Step: {self.training_target_step}"
                    )
                except Exception as stats_err:
                    print(
                        f"  -> ERROR loading Stats Aggregator state: {stats_err}. Stats reset."
                    )
                    self._reset_aggregator_state()  # Reset only aggregator on specific failure
                    self.episode_count = 0  # Reset episode count too
                    self.training_target_step = 0  # Reset target step
            elif self.stats_aggregator:
                print(
                    "  -> WARNING: 'stats_aggregator_state_dict' not found. Stats Aggregator reset."
                )
                self._reset_aggregator_state()
                self.episode_count = 0  # Reset episode count too
                self.training_target_step = 0  # Reset target step
            else:  # Fallback if no aggregator exists (should not happen with current setup)
                self.episode_count = checkpoint.get("episode_count", 0)
                self.training_target_step = checkpoint.get("training_target_step", 0)

            print(
                f"  -> Resuming from Step: {self.global_step}, Ep: {self.episode_count}"
            )

            # --- Load Observation RMS State ---
            if "obs_rms_state_dict" in checkpoint and self.obs_rms_dict:
                rms_state_dict = checkpoint["obs_rms_state_dict"]
                loaded_keys = set()
                for key, rms_instance in self.obs_rms_dict.items():
                    if key in rms_state_dict:
                        try:
                            rms_data = rms_state_dict[key]
                            # Ensure data is numpy before loading into RMS
                            if isinstance(rms_data.get("mean"), torch.Tensor):
                                rms_data["mean"] = rms_data["mean"].cpu().numpy()
                            if isinstance(rms_data.get("var"), torch.Tensor):
                                rms_data["var"] = rms_data["var"].cpu().numpy()
                            rms_instance.load_state_dict(rms_data)
                            loaded_keys.add(key)
                            print(f"  -> Loaded Obs RMS for '{key}'")
                        except Exception as rms_load_err:
                            print(
                                f"  -> ERROR loading Obs RMS for '{key}': {rms_load_err}. RMS for this key reset."
                            )
                            rms_instance.reset()
                    else:
                        print(
                            f"  -> WARNING: Obs RMS state for '{key}' not found in checkpoint. Using initial RMS."
                        )
                        rms_instance.reset()

                extra_keys = set(rms_state_dict.keys()) - loaded_keys
                if extra_keys:
                    print(
                        f"  -> WARNING: Checkpoint contains unused Obs RMS keys: {extra_keys}"
                    )
            elif self.obs_rms_dict:
                print(
                    "  -> WARNING: Obs RMS state dict ('obs_rms_state_dict') not found. Using initial RMS for all keys."
                )
                for rms in self.obs_rms_dict.values():
                    rms.reset()

            print("[CheckpointManager] Checkpoint loading finished.")

        except pickle.UnpicklingError as e:  # Catch the specific error
            print(
                f"  -> ERROR loading checkpoint (UnpicklingError): {e}. This often happens with PyTorch version changes or corrupted files. State reset."
            )
            traceback.print_exc()
            self._reset_all_states()
        except KeyError as e:
            print(
                f"  -> ERROR loading checkpoint: Missing key '{e}'. Check compatibility. State reset."
            )
            traceback.print_exc()
            self._reset_all_states()
        except Exception as e:
            print(
                f"  -> ERROR loading checkpoint ('{e}'). Check compatibility. State reset."
            )
            traceback.print_exc()
            self._reset_all_states()

    def _reset_aggregator_state(self):
        """Helper to reset only the stats aggregator state."""
        if self.stats_aggregator:
            # Preserve avg_windows and plot_window during reset
            avg_windows = self.stats_aggregator.avg_windows
            plot_window = self.stats_aggregator.plot_window
            self.stats_aggregator.__init__(
                avg_windows=avg_windows,
                plot_window=plot_window,
            )
            self.stats_aggregator.total_episodes = 0
            self.stats_aggregator.training_target_step = 0  # Reset target

    def _reset_all_states(self):
        """Helper to reset all managed states on critical load failure."""
        print("[CheckpointManager] Resetting all managed states due to load failure.")
        self.global_step = 0
        self.episode_count = 0
        self.training_target_step = 0  # Reset target
        if self.obs_rms_dict:
            for rms in self.obs_rms_dict.values():
                rms.reset()
        self._reset_aggregator_state()

    def save_checkpoint(
        self,
        global_step: int,
        episode_count: int,
        training_target_step: int,  # Add target step parameter
        is_final: bool = False,
    ):
        """Saves agent, observation normalization, and stats aggregator state to the current run's directory."""
        prefix = "FINAL" if is_final else f"step_{global_step}"
        # Use the run_checkpoint_dir specific to the current run for saving
        save_dir = self.run_checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{prefix}_agent_state.pth"
        full_save_path = os.path.join(save_dir, filename)

        print(f"[CheckpointManager] Saving checkpoint ({prefix}) to {save_dir}...")
        try:
            agent_save_data = self.agent.get_state_dict()

            obs_rms_save_data = {}
            if self.obs_rms_dict:
                for key, rms_instance in self.obs_rms_dict.items():
                    rms_state = rms_instance.state_dict()
                    # Ensure RMS state is saved as numpy arrays
                    if isinstance(rms_state.get("mean"), torch.Tensor):
                        rms_state["mean"] = rms_state["mean"].cpu().numpy()
                    if isinstance(rms_state.get("var"), torch.Tensor):
                        rms_state["var"] = rms_state["var"].cpu().numpy()
                    obs_rms_save_data[key] = rms_state

            stats_aggregator_save_data = {}
            if self.stats_aggregator:
                # Ensure aggregator has the latest target step before saving its state
                self.stats_aggregator.training_target_step = training_target_step
                stats_aggregator_save_data = self.stats_aggregator.state_dict()
                # Ensure episode count in checkpoint matches aggregator's count
                episode_count = self.stats_aggregator.total_episodes

            # Combine into a single checkpoint dictionary
            checkpoint_data = {
                "global_step": global_step,
                "episode_count": episode_count,  # Use count from aggregator if available
                "training_target_step": training_target_step,  # Save target step
                "agent_state_dict": agent_save_data,
                "obs_rms_state_dict": obs_rms_save_data,
                "stats_aggregator_state_dict": stats_aggregator_save_data,  # Add stats state
            }

            # Use a temporary file and rename for atomicity
            temp_save_path = full_save_path + ".tmp"
            torch.save(checkpoint_data, temp_save_path)
            os.replace(temp_save_path, full_save_path)  # Atomic rename

            print(f"  -> Checkpoint saved: {filename}")
        except Exception as e:
            print(f"  -> ERROR saving checkpoint: {e}")
            traceback.print_exc()
            # Clean up temporary file if saving failed
            if os.path.exists(temp_save_path):
                try:
                    os.remove(temp_save_path)
                except OSError:
                    pass

    def get_initial_state(self) -> Tuple[int, int]:
        """Returns the initial global step and episode count after potential loading."""
        # Episode count is now primarily sourced from the loaded stats aggregator
        return self.global_step, self.episode_count
