import os
import torch
import traceback
from typing import Optional, Dict, Any, Tuple

from agent.ppo_agent import PPOAgent

from utils.running_mean_std import RunningMeanStd


class CheckpointManager:
    """Handles loading and saving of agent states and observation normalization stats."""

    def __init__(
        self,
        agent: PPOAgent,
        model_save_path: str,
        load_checkpoint_path: Optional[str],
        device: torch.device,
        obs_rms_dict: Optional[Dict[str, RunningMeanStd]] = None,
    ):
        self.agent = agent
        self.model_save_path = model_save_path
        self.device = device
        self.obs_rms_dict = obs_rms_dict if obs_rms_dict else {}

        self.global_step = 0
        self.episode_count = 0

        if load_checkpoint_path:
            self.load_checkpoint(load_checkpoint_path)
        else:
            print("[CheckpointManager] No checkpoint specified, starting fresh.")

    def load_checkpoint(self, path_to_load: str):
        """Loads agent state and observation normalization stats from a checkpoint file."""
        if not os.path.isfile(path_to_load):
            print(
                f"[CheckpointManager] LOAD WARNING: Checkpoint not found: {path_to_load}"
            )
            return
        print(f"[CheckpointManager] Loading checkpoint from: {path_to_load}")
        try:
            checkpoint = torch.load(path_to_load, map_location=self.device)

            # Load agent state
            if "agent_state_dict" in checkpoint:
                self.agent.load_state_dict(checkpoint["agent_state_dict"])
            else:
                print(
                    "  -> Agent state dict not found at 'agent_state_dict', attempting legacy load..."
                )
                self.agent.load_state_dict(
                    checkpoint
                )  # May raise error if incompatible

            # Load global step and episode count
            self.global_step = checkpoint.get("global_step", 0)
            self.episode_count = checkpoint.get("episode_count", 0)
            print(
                f"  -> Resuming from Step: {self.global_step}, Ep: {self.episode_count}"
            )

            if "obs_rms_state_dict" in checkpoint and self.obs_rms_dict:
                rms_state_dict = checkpoint["obs_rms_state_dict"]
                loaded_keys = set()
                for key, rms_instance in self.obs_rms_dict.items():
                    if key in rms_state_dict:
                        rms_instance.load_state_dict(rms_state_dict[key])
                        loaded_keys.add(key)
                        print(f"  -> Loaded Obs RMS for '{key}'")
                    else:
                        print(
                            f"  -> WARNING: Obs RMS state for '{key}' not found in checkpoint."
                        )
                # Check for extra keys in checkpoint not in current config
                extra_keys = set(rms_state_dict.keys()) - loaded_keys
                if extra_keys:
                    print(
                        f"  -> WARNING: Checkpoint contains unused Obs RMS keys: {extra_keys}"
                    )
            elif self.obs_rms_dict:
                print(
                    "  -> WARNING: Obs RMS state dict not found in checkpoint, using initial RMS."
                )

        except KeyError as e:
            print(
                f"  -> ERROR loading checkpoint: Missing key '{e}'. Check compatibility."
            )
            self.global_step = 0
            self.episode_count = 0
            for rms in self.obs_rms_dict.values():
                rms.reset()  # Reset RMS on critical load failure
        except Exception as e:
            print(f"  -> ERROR loading checkpoint ('{e}'). Check compatibility.")
            traceback.print_exc()
            self.global_step = 0
            self.episode_count = 0
            for rms in self.obs_rms_dict.values():
                rms.reset()

    def save_checkpoint(
        self, global_step: int, episode_count: int, is_final: bool = False
    ):
        """Saves agent state and observation normalization stats to a checkpoint file."""
        prefix = "FINAL" if is_final else f"step_{global_step}"
        save_dir = os.path.dirname(self.model_save_path)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{prefix}_agent_state.pth"
        full_save_path = os.path.join(save_dir, filename)  # Use specific filename

        print(f"[CheckpointManager] Saving checkpoint ({prefix})...")
        try:
            agent_save_data = self.agent.get_state_dict()

            obs_rms_save_data = {}
            if self.obs_rms_dict:
                for key, rms_instance in self.obs_rms_dict.items():
                    obs_rms_save_data[key] = rms_instance.state_dict()

            # Combine into a single checkpoint dictionary
            checkpoint_data = {
                "global_step": global_step,
                "episode_count": episode_count,
                "agent_state_dict": agent_save_data,
                "obs_rms_state_dict": obs_rms_save_data,  # Add RMS state
            }

            torch.save(checkpoint_data, full_save_path)  # Save to specific file
            # Optionally update the main save path symlink or keep track of latest
            # For simplicity, we just save with a step/final prefix now.
            print(f"  -> Checkpoint saved: {filename}")
        except Exception as e:
            print(f"  -> ERROR saving checkpoint: {e}")
            traceback.print_exc()

    def get_initial_state(self) -> Tuple[int, int]:
        """Returns the initial global step and episode count after potential loading."""
        return self.global_step, self.episode_count
