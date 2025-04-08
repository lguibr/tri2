# File: training/checkpoint_manager.py
import os
import torch
import torch.optim as optim  # Added optimizer import
import traceback
import re
import time
from typing import Optional, Tuple, Any, Dict
import pickle

from stats.aggregator import StatsAggregator
from agent.alphazero_net import AlphaZeroNet


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
                    continue
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
        return latest_run_id, None


def find_latest_checkpoint_in_dir(checkpoint_dir: str) -> Optional[str]:
    """
    Finds the latest checkpoint file (step_*_alphazero_nn.pth or FINAL_alphazero_nn.pth)
    in a specific directory. Prioritizes FINAL checkpoint if it exists.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = []
    final_checkpoint = None
    step_pattern = re.compile(r"step_(\d+)_alphazero_nn\.pth")
    final_filename = "FINAL_alphazero_nn.pth"

    try:
        for filename in os.listdir(checkpoint_dir):
            full_path = os.path.join(checkpoint_dir, filename)
            if not os.path.isfile(full_path):
                continue

            if filename == final_filename:
                final_checkpoint = full_path
            else:
                match = step_pattern.match(filename)
                if match:
                    step = int(match.group(1))
                    checkpoints.append((step, full_path))

    except OSError as e:
        print(f"[CheckpointFinder] Error listing directory {checkpoint_dir}: {e}")
        return None

    if final_checkpoint:
        try:
            final_mtime = os.path.getmtime(final_checkpoint)
            newer_step_checkpoints = [
                cp for step, cp in checkpoints if os.path.getmtime(cp) > final_mtime
            ]
            if not newer_step_checkpoints:
                print(f"[CheckpointFinder] Using FINAL checkpoint: {final_checkpoint}")
                return final_checkpoint
        except OSError:
            pass

    if not checkpoints:
        return final_checkpoint

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


class CheckpointManager:
    """Handles loading and saving of agent, optimizer, and stats states."""

    def __init__(
        self,
        agent: Optional[AlphaZeroNet],
        optimizer: Optional[optim.Optimizer],  # Added optimizer
        stats_aggregator: Optional[StatsAggregator],
        base_checkpoint_dir: str,
        run_checkpoint_dir: str,
        load_checkpoint_path_config: Optional[str],
        device: torch.device,
    ):
        self.agent = agent
        self.optimizer = optimizer  # Store optimizer
        self.stats_aggregator = stats_aggregator
        self.base_checkpoint_dir = base_checkpoint_dir
        self.run_checkpoint_dir = run_checkpoint_dir
        self.device = device

        self.global_step = 0
        self.episode_count = 0
        self.training_target_step = 0

        self.run_id_to_load_from: Optional[str] = None
        self.checkpoint_path_to_load: Optional[str] = None

        if load_checkpoint_path_config:
            print(
                f"[CheckpointManager] Using explicit checkpoint path from config: {load_checkpoint_path_config}"
            )
            if os.path.isfile(load_checkpoint_path_config):
                self.checkpoint_path_to_load = load_checkpoint_path_config
                try:
                    parent_dir = os.path.dirname(load_checkpoint_path_config)
                    self.run_id_to_load_from = os.path.basename(parent_dir)
                    if not self.run_id_to_load_from.startswith("run_"):
                        self.run_id_to_load_from = None
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

        if self.stats_aggregator:
            self.stats_aggregator.storage.training_target_step = (
                self.training_target_step
            )

    def get_run_id_to_load_from(self) -> Optional[str]:
        return self.run_id_to_load_from

    def get_checkpoint_path_to_load(self) -> Optional[str]:
        return self.checkpoint_path_to_load

    def load_checkpoint(self):
        """Loads agent, optimizer, and stats aggregator state."""
        if not self.checkpoint_path_to_load:
            print(
                "[CheckpointManager] No checkpoint path specified for loading. Skipping load."
            )
            self._reset_all_states()
            return

        if not os.path.isfile(self.checkpoint_path_to_load):
            print(
                f"[CheckpointManager] LOAD ERROR: Checkpoint file not found: {self.checkpoint_path_to_load}"
            )
            self._reset_all_states()
            return

        print(
            f"[CheckpointManager] Loading checkpoint from: {self.checkpoint_path_to_load}"
        )
        loaded_target_step = None
        agent_load_successful = False
        optimizer_load_successful = False
        try:
            checkpoint = torch.load(
                self.checkpoint_path_to_load,
                map_location=self.device,
                weights_only=False,
            )

            # --- Load Agent State ---
            if "agent_state_dict" in checkpoint:
                if self.agent:
                    try:
                        self.agent.load_state_dict(checkpoint["agent_state_dict"])
                        agent_load_successful = True
                        print("  -> Agent state loaded successfully.")
                    except Exception as agent_load_err:
                        print(
                            f"  -> ERROR loading Agent state: {agent_load_err}. Agent state may be inconsistent."
                        )
                        agent_load_successful = False
                else:
                    print(
                        "  -> WARNING: Agent not initialized, cannot load agent state dict."
                    )
            else:
                print(
                    "  -> WARNING: 'agent_state_dict' key missing. Agent state NOT loaded."
                )
            # --- End Load Agent State ---

            # --- Load Optimizer State ---
            if "optimizer_state_dict" in checkpoint:
                if self.optimizer:
                    try:
                        self.optimizer.load_state_dict(
                            checkpoint["optimizer_state_dict"]
                        )
                        # Move optimizer state to the correct device
                        for state in self.optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(self.device)
                        optimizer_load_successful = True
                        print("  -> Optimizer state loaded successfully.")
                    except Exception as opt_load_err:
                        print(
                            f"  -> ERROR loading Optimizer state: {opt_load_err}. Optimizer state reset."
                        )
                        # Consider resetting optimizer if load fails
                        # self.optimizer.state = defaultdict(dict) # Or re-initialize
                        optimizer_load_successful = False
                else:
                    print(
                        "  -> WARNING: Optimizer not initialized, cannot load optimizer state dict."
                    )
            else:
                print(
                    "  -> WARNING: 'optimizer_state_dict' key missing. Optimizer state NOT loaded."
                )
            # --- End Load Optimizer State ---

            self.global_step = checkpoint.get("global_step", 0)
            print(f"  -> Loaded Global Step: {self.global_step}")

            # --- Load Stats Aggregator State ---
            if "stats_aggregator_state_dict" in checkpoint and self.stats_aggregator:
                try:
                    self.stats_aggregator.load_state_dict(
                        checkpoint["stats_aggregator_state_dict"]
                    )
                    print("  -> Stats Aggregator state loaded successfully.")
                    self.episode_count = self.stats_aggregator.storage.total_episodes
                    loaded_target_step = getattr(
                        self.stats_aggregator.storage, "training_target_step", None
                    )
                    if loaded_target_step is not None:
                        print(
                            f"  -> Loaded Training Target Step from Stats: {loaded_target_step}"
                        )
                    else:
                        print(
                            "  -> WARNING: 'training_target_step' not found in loaded stats."
                        )
                    loaded_start_time = self.stats_aggregator.storage.start_time
                    print(
                        f"  -> Loaded Run Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(loaded_start_time))}"
                    )
                except Exception as stats_err:
                    print(
                        f"  -> ERROR loading Stats Aggregator state: {stats_err}. Stats reset."
                    )
                    self._reset_aggregator_state()
                    self.episode_count = 0
            elif self.stats_aggregator:
                print(
                    "  -> WARNING: 'stats_aggregator_state_dict' not found. Stats Aggregator reset."
                )
                self._reset_aggregator_state()
                self.episode_count = 0
            else:
                self.episode_count = checkpoint.get("episode_count", 0)
                loaded_target_step = checkpoint.get("training_target_step", None)
                if loaded_target_step is not None:
                    print(
                        f"  -> Loaded Training Target Step from Checkpoint (fallback): {loaded_target_step}"
                    )
            # --- End Load Stats Aggregator State ---

            print(
                f"  -> Resuming from Step: {self.global_step}, Ep: {self.episode_count}"
            )

            if loaded_target_step is not None:
                self.training_target_step = loaded_target_step
                print(
                    f"[CheckpointManager] Using loaded Training Target Step: {self.training_target_step}"
                )
            else:
                self.training_target_step = 0
                print(
                    "[CheckpointManager] WARNING: No training target step found in checkpoint or stats. Setting target to 0."
                )

            if self.stats_aggregator:
                self.stats_aggregator.storage.training_target_step = (
                    self.training_target_step
                )

            print("[CheckpointManager] Checkpoint loading finished.")

        except pickle.UnpicklingError as e:
            print(f"  -> ERROR loading checkpoint (UnpicklingError): {e}. State reset.")
            traceback.print_exc()
            self._reset_all_states()
        except KeyError as e:
            print(f"  -> ERROR loading checkpoint: Missing key '{e}'. State reset.")
            traceback.print_exc()
            self._reset_all_states()
        except Exception as e:
            print(f"  -> ERROR loading checkpoint ('{e}'). State reset.")
            traceback.print_exc()
            self._reset_all_states()

        if not agent_load_successful:
            print("[CheckpointManager] Agent load was unsuccessful.")
        if not optimizer_load_successful:
            print("[CheckpointManager] Optimizer load was unsuccessful.")

        print(
            f"[CheckpointManager] Final Training Target Step set to: {self.training_target_step}"
        )

    def _reset_aggregator_state(self):
        """Helper to reset only the stats aggregator state."""
        if self.stats_aggregator:
            avg_windows = self.stats_aggregator.avg_windows
            plot_window = self.stats_aggregator.plot_window
            self.stats_aggregator.__init__(
                avg_windows=avg_windows, plot_window=plot_window
            )
            self.stats_aggregator.storage.training_target_step = (
                self.training_target_step
            )
            self.stats_aggregator.storage.total_episodes = 0

    def _reset_all_states(self):
        """Helper to reset all managed states on critical load failure."""
        print("[CheckpointManager] Resetting all managed states due to load failure.")
        self.global_step = 0
        self.episode_count = 0
        self.training_target_step = 0
        # Reset optimizer state if it exists
        if self.optimizer:
            self.optimizer.state = {}  # Clear optimizer state
            print("  -> Optimizer state reset.")
        self._reset_aggregator_state()

    def save_checkpoint(
        self,
        global_step: int,
        episode_count: int,
        training_target_step: int,
        is_final: bool = False,
    ):
        """Saves agent, optimizer, and stats aggregator state."""
        prefix = "FINAL" if is_final else f"step_{global_step}"
        save_dir = self.run_checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{prefix}_alphazero_nn.pth"
        full_save_path = os.path.join(save_dir, filename)

        print(f"[CheckpointManager] Saving checkpoint ({prefix}) to {save_dir}...")
        temp_save_path = full_save_path + ".tmp"
        try:
            agent_save_data = {}
            if self.agent:
                agent_save_data = self.agent.state_dict()
            else:
                print("  -> WARNING: Agent not initialized, saving empty agent state.")

            optimizer_save_data = {}
            if self.optimizer:
                optimizer_save_data = self.optimizer.state_dict()
            else:
                print(
                    "  -> WARNING: Optimizer not initialized, saving empty optimizer state."
                )

            stats_aggregator_save_data = {}
            aggregator_episode_count = episode_count
            aggregator_target_step = training_target_step
            if self.stats_aggregator:
                self.stats_aggregator.storage.training_target_step = (
                    training_target_step
                )
                stats_aggregator_save_data = self.stats_aggregator.state_dict()
                aggregator_episode_count = self.stats_aggregator.storage.total_episodes
                aggregator_target_step = (
                    self.stats_aggregator.storage.training_target_step
                )

            checkpoint_data = {
                "global_step": global_step,
                "episode_count": aggregator_episode_count,
                "training_target_step": aggregator_target_step,
                "agent_state_dict": agent_save_data,
                "optimizer_state_dict": optimizer_save_data,  # Save optimizer state
                "stats_aggregator_state_dict": stats_aggregator_save_data,
            }

            torch.save(checkpoint_data, temp_save_path)
            os.replace(temp_save_path, full_save_path)
            print(f"  -> Checkpoint saved: {filename}")
        except Exception as e:
            print(f"  -> ERROR saving checkpoint: {e}")
            traceback.print_exc()
            if os.path.exists(temp_save_path):
                try:
                    os.remove(temp_save_path)
                except OSError:
                    pass

    def get_initial_state(self) -> Tuple[int, int]:
        """Returns the initial global step and episode count after potential loading."""
        return self.global_step, self.episode_count
