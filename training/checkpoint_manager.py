import os
import torch
import torch.optim as optim
import traceback
import re
import time
from typing import Optional, Tuple, Any, Dict
import pickle

from stats.aggregator import StatsAggregator
from agent.alphazero_net import AlphaZeroNet

# Import scheduler base class for type hinting
from torch.optim.lr_scheduler import _LRScheduler


# --- Checkpoint Finding Logic ---
def find_latest_run_and_checkpoint(
    base_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Finds the latest run directory and the latest checkpoint within it."""
    latest_run_id, latest_run_mtime = None, 0
    if not os.path.isdir(base_dir):
        return None, None
    try:
        for item in os.listdir(base_dir):
            path = os.path.join(base_dir, item)
            if os.path.isdir(path) and item.startswith("run_"):
                try:
                    mtime = os.path.getmtime(path)
                    if mtime > latest_run_mtime:
                        latest_run_mtime, latest_run_id = mtime, item
                except OSError:
                    continue
    except OSError as e:
        print(f"[CheckpointFinder] Error listing {base_dir}: {e}")
        return None, None

    if latest_run_id is None:
        print(f"[CheckpointFinder] No runs found in {base_dir}.")
        return None, None
    latest_run_dir = os.path.join(base_dir, latest_run_id)
    print(f"[CheckpointFinder] Latest run directory: {latest_run_dir}")
    latest_checkpoint = find_latest_checkpoint_in_dir(latest_run_dir)
    if latest_checkpoint:
        print(
            f"[CheckpointFinder] Found checkpoint: {os.path.basename(latest_checkpoint)}"
        )
    else:
        print(f"[CheckpointFinder] No valid checkpoints found in {latest_run_dir}")
    return latest_run_id, latest_checkpoint


def find_latest_checkpoint_in_dir(ckpt_dir: str) -> Optional[str]:
    """Finds the latest checkpoint file in a specific directory."""
    if not os.path.isdir(ckpt_dir):
        return None
    checkpoints, final_ckpt = [], None
    step_pattern = re.compile(r"step_(\d+)_alphazero_nn\.pth")
    final_name = "FINAL_alphazero_nn.pth"
    try:
        for fname in os.listdir(ckpt_dir):
            fpath = os.path.join(ckpt_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if fname == final_name:
                final_ckpt = fpath
            else:
                match = step_pattern.match(fname)
                checkpoints.append((int(match.group(1)), fpath)) if match else None
    except OSError as e:
        print(f"[CheckpointFinder] Error listing {ckpt_dir}: {e}")
        return None

    if final_ckpt:
        try:
            final_mtime = os.path.getmtime(final_ckpt)
            if not any(os.path.getmtime(cp) > final_mtime for _, cp in checkpoints):
                return final_ckpt
        except OSError:
            pass
    if not checkpoints:
        return final_ckpt
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


# --- Checkpoint Manager Class ---
class CheckpointManager:
    """Handles loading and saving of agent, optimizer, scheduler, and stats states."""

    def __init__(
        self,
        agent: Optional[AlphaZeroNet],
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[_LRScheduler],  # Add scheduler
        stats_aggregator: Optional[StatsAggregator],
        base_checkpoint_dir: str,
        run_checkpoint_dir: str,
        load_checkpoint_path_config: Optional[str],
        device: torch.device,
    ):
        self.agent = agent
        self.optimizer = optimizer
        self.scheduler = scheduler  # Store scheduler
        self.stats_aggregator = stats_aggregator
        self.base_checkpoint_dir = base_checkpoint_dir
        self.run_checkpoint_dir = run_checkpoint_dir
        self.device = device
        self.global_step = 0
        self.episode_count = 0
        self.training_target_step = 0
        self.run_id_to_load_from, self.checkpoint_path_to_load = (
            self._determine_checkpoint_to_load(load_checkpoint_path_config)
        )
        if self.stats_aggregator:
            self.stats_aggregator.storage.training_target_step = (
                self.training_target_step
            )

    def _determine_checkpoint_to_load(
        self, config_path: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Determines which checkpoint to load based on config or latest run."""
        if config_path:
            print(f"[CheckpointManager] Using explicit checkpoint path: {config_path}")
            if os.path.isfile(config_path):
                run_id = None
                try:
                    run_id = (
                        os.path.basename(os.path.dirname(config_path))
                        if os.path.basename(os.path.dirname(config_path)).startswith(
                            "run_"
                        )
                        else None
                    )
                except Exception:
                    pass
                print(
                    f"[CheckpointManager] Extracted run_id '{run_id}' from path."
                    if run_id
                    else "[CheckpointManager] Could not determine run_id from path."
                )
                return run_id, config_path
            else:
                print(
                    f"[CheckpointManager] WARNING: Explicit path not found: {config_path}. Starting fresh."
                )
                return None, None
        else:
            print(
                f"[CheckpointManager] Searching for latest run in: {self.base_checkpoint_dir}"
            )
            run_id, ckpt_path = find_latest_run_and_checkpoint(self.base_checkpoint_dir)
            if run_id and ckpt_path:
                print(
                    f"[CheckpointManager] Found latest run '{run_id}' with checkpoint."
                )
            elif run_id:
                print(
                    f"[CheckpointManager] Found latest run '{run_id}' but no checkpoint. Starting fresh."
                )
            else:
                print(f"[CheckpointManager] No previous runs found. Starting fresh.")
            return run_id, ckpt_path

    def get_run_id_to_load_from(self) -> Optional[str]:
        return self.run_id_to_load_from

    def get_checkpoint_path_to_load(self) -> Optional[str]:
        return self.checkpoint_path_to_load

    def load_checkpoint(self):
        """Loads agent, optimizer, scheduler, and stats aggregator state."""
        if not self.checkpoint_path_to_load or not os.path.isfile(
            self.checkpoint_path_to_load
        ):
            print(
                f"[CheckpointManager] Checkpoint not found or not specified: {self.checkpoint_path_to_load}. Skipping load."
            )
            self._reset_all_states()
            return
        print(f"[CheckpointManager] Loading checkpoint: {self.checkpoint_path_to_load}")
        try:
            checkpoint = torch.load(
                self.checkpoint_path_to_load,
                map_location=self.device,
                weights_only=False,  # Set to False to load optimizer/scheduler states
            )
            agent_ok = self._load_agent_state(checkpoint)
            opt_ok = self._load_optimizer_state(checkpoint)
            sched_ok = self._load_scheduler_state(checkpoint)  # Load scheduler
            stats_ok, loaded_target = self._load_stats_state(checkpoint)
            self.global_step = checkpoint.get("global_step", 0)
            print(f"  -> Loaded Global Step: {self.global_step}")
            if stats_ok:
                self.episode_count = self.stats_aggregator.storage.total_episodes
            else:
                self.episode_count = checkpoint.get("episode_count", 0)
            print(
                f"  -> Resuming from Step: {self.global_step}, Ep: {self.episode_count}"
            )
            self.training_target_step = (
                loaded_target
                if loaded_target is not None
                else checkpoint.get("training_target_step", 0)
            )
            if self.stats_aggregator:
                self.stats_aggregator.storage.training_target_step = (
                    self.training_target_step
                )
            print("[CheckpointManager] Checkpoint loading finished.")
            if not agent_ok:
                print("[CheckpointManager] Agent load was unsuccessful.")
            if not opt_ok:
                print("[CheckpointManager] Optimizer load was unsuccessful.")
            if not sched_ok:
                print("[CheckpointManager] Scheduler load was unsuccessful.")
            if not stats_ok:
                print("[CheckpointManager] Stats load was unsuccessful.")
        except (pickle.UnpicklingError, KeyError, Exception) as e:
            print(f"  -> ERROR loading checkpoint ('{e}'). State reset.")
            traceback.print_exc()
            self._reset_all_states()
        print(
            f"[CheckpointManager] Final Training Target Step set to: {self.training_target_step}"
        )

    def _load_agent_state(self, checkpoint: Dict[str, Any]) -> bool:
        """Loads the agent state dictionary."""
        if "agent_state_dict" not in checkpoint:
            print("  -> WARNING: 'agent_state_dict' missing.")
            return False
        if not self.agent:
            print("  -> WARNING: Agent not initialized.")
            return False
        try:
            self.agent.load_state_dict(checkpoint["agent_state_dict"])
            print("  -> Agent state loaded.")
            return True
        except Exception as e:
            print(f"  -> ERROR loading Agent state: {e}.")
            return False

    def _load_optimizer_state(self, checkpoint: Dict[str, Any]) -> bool:
        """Loads the optimizer state dictionary."""
        if "optimizer_state_dict" not in checkpoint:
            print("  -> WARNING: 'optimizer_state_dict' missing.")
            return False
        if not self.optimizer:
            print("  -> WARNING: Optimizer not initialized.")
            return False
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Move optimizer state to the correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print("  -> Optimizer state loaded.")
            return True
        except Exception as e:
            print(f"  -> ERROR loading Optimizer state: {e}.")
            return False

    def _load_scheduler_state(self, checkpoint: Dict[str, Any]) -> bool:
        """Loads the scheduler state dictionary."""
        if "scheduler_state_dict" not in checkpoint:
            # This is not necessarily a warning if scheduler wasn't used before
            print("  -> INFO: 'scheduler_state_dict' missing (may be expected).")
            return False
        if not self.scheduler:
            print(
                "  -> INFO: Scheduler not initialized for current run, skipping load."
            )
            return False
        try:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("  -> Scheduler state loaded.")
            return True
        except Exception as e:
            print(f"  -> ERROR loading Scheduler state: {e}.")
            return False

    def _load_stats_state(
        self, checkpoint: Dict[str, Any]
    ) -> Tuple[bool, Optional[int]]:
        """Loads the stats aggregator state."""
        loaded_target = None
        if "stats_aggregator_state_dict" not in checkpoint:
            print("  -> WARNING: 'stats_aggregator_state_dict' missing.")
            return False, loaded_target
        if not self.stats_aggregator:
            print("  -> WARNING: Stats Aggregator not initialized.")
            return False, loaded_target
        try:
            self.stats_aggregator.load_state_dict(
                checkpoint["stats_aggregator_state_dict"]
            )
            loaded_target = getattr(
                self.stats_aggregator.storage, "training_target_step", None
            )
            start_time = self.stats_aggregator.storage.start_time
            print("  -> Stats Aggregator state loaded.")
            if loaded_target is not None:
                print(f"  -> Loaded Training Target Step from Stats: {loaded_target}")
            print(
                f"  -> Loaded Run Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
            )
            return True, loaded_target
        except Exception as e:
            print(f"  -> ERROR loading Stats Aggregator state: {e}.")
            self._reset_aggregator_state()
            return False, loaded_target

    def _reset_aggregator_state(self):
        """Resets only the stats aggregator state."""
        if self.stats_aggregator:
            self.stats_aggregator.__init__(
                avg_windows=self.stats_aggregator.avg_windows,
                plot_window=self.stats_aggregator.plot_window,
            )
            self.stats_aggregator.storage.training_target_step = (
                self.training_target_step
            )
            self.stats_aggregator.storage.total_episodes = 0

    def _reset_all_states(self):
        """Resets all managed states on critical load failure."""
        print("[CheckpointManager] Resetting all managed states due to load failure.")
        self.global_step = 0
        self.episode_count = 0
        self.training_target_step = 0
        if self.optimizer:
            # Reset optimizer state
            self.optimizer.state = {}
            # Re-initialize scheduler if it exists, as its state depends on optimizer
            if self.scheduler:
                # Assuming CosineAnnealingLR, re-init with same params
                # This might need adjustment if using other schedulers
                try:
                    self.scheduler = type(self.scheduler)(
                        self.optimizer, **self.scheduler.state_dict()
                    )
                    print("  -> Scheduler re-initialized after optimizer reset.")
                except Exception as e:
                    print(f"  -> WARNING: Failed to re-initialize scheduler: {e}")
                    self.scheduler = None  # Fallback
            print("  -> Optimizer state reset.")
        self._reset_aggregator_state()

    def save_checkpoint(
        self,
        global_step: int,
        episode_count: int,
        training_target_step: int,
        is_final: bool = False,
    ):
        """Saves agent, optimizer, scheduler, and stats aggregator state."""
        prefix = "FINAL" if is_final else f"step_{global_step}"
        save_dir = self.run_checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{prefix}_alphazero_nn.pth"
        full_path = os.path.join(save_dir, filename)
        print(f"[CheckpointManager] Saving checkpoint ({prefix}) to {save_dir}...")
        temp_path = full_path + ".tmp"
        try:
            agent_sd = self.agent.state_dict() if self.agent else {}
            opt_sd = self.optimizer.state_dict() if self.optimizer else {}
            sched_sd = (
                self.scheduler.state_dict() if self.scheduler else {}
            )  # Get scheduler state
            stats_sd = {}
            agg_ep_count = episode_count
            agg_target_step = training_target_step
            if self.stats_aggregator:
                self.stats_aggregator.storage.training_target_step = (
                    training_target_step
                )
                stats_sd = self.stats_aggregator.state_dict()
                agg_ep_count = self.stats_aggregator.storage.total_episodes
                agg_target_step = self.stats_aggregator.storage.training_target_step

            checkpoint_data = {
                "global_step": global_step,
                "episode_count": agg_ep_count,
                "training_target_step": agg_target_step,
                "agent_state_dict": agent_sd,
                "optimizer_state_dict": opt_sd,
                "scheduler_state_dict": sched_sd,  # Save scheduler state
                "stats_aggregator_state_dict": stats_sd,
            }
            torch.save(checkpoint_data, temp_path)
            os.replace(temp_path, full_path)
            print(f"  -> Checkpoint saved: {filename}")
        except Exception as e:
            print(f"  -> ERROR saving checkpoint: {e}")
            traceback.print_exc()
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def get_initial_state(self) -> Tuple[int, int]:
        """Returns the initial global step and episode count after potential loading."""
        return self.global_step, self.episode_count
