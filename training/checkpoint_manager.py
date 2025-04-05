# File: training/checkpoint_manager.py
import os
import torch
import traceback
from typing import Optional, Dict, Any, Tuple

from agent.ppo_agent import PPOAgent


class CheckpointManager:
    """Handles loading and saving of PPO agent states."""

    def __init__(
        self,
        agent: PPOAgent,
        model_save_path: str,
        load_checkpoint_path: Optional[str],
        device: torch.device,
    ):
        self.agent = agent
        self.model_save_path = model_save_path
        self.device = device

        self.global_step = 0
        self.episode_count = 0

        if load_checkpoint_path:
            self.load_agent_checkpoint(load_checkpoint_path)
        else:
            print(
                "[CheckpointManager-PPO] No agent checkpoint specified, starting fresh."
            )

    def load_agent_checkpoint(self, path_to_load: str):
        if not os.path.isfile(path_to_load):
            print(
                f"[CheckpointManager-PPO] LOAD WARNING: Agent ckpt not found: {path_to_load}"
            )
            return
        print(f"[CheckpointManager-PPO] Loading agent checkpoint from: {path_to_load}")
        try:
            checkpoint = torch.load(path_to_load, map_location=self.device)
            self.agent.load_state_dict(checkpoint)

            self.global_step = checkpoint.get("global_step", 0)
            self.episode_count = checkpoint.get("episode_count", 0)
            print(
                f"  -> Resuming from Step: {self.global_step}, Ep: {self.episode_count}"
            )
        except KeyError as e:
            print(
                f"  -> ERROR loading agent checkpoint: Missing key '{e}'. Check compatibility."
            )
            self.global_step = 0
            self.episode_count = 0
        except Exception as e:
            print(f"  -> ERROR loading agent checkpoint ('{e}'). Check compatibility.")
            traceback.print_exc()
            self.global_step = 0
            self.episode_count = 0

    def save_checkpoint(
        self, global_step: int, episode_count: int, is_final: bool = False
    ):
        prefix = "FINAL" if is_final else f"step_{global_step}"
        save_dir = os.path.dirname(self.model_save_path)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[CheckpointManager-PPO] Saving agent checkpoint ({prefix})...")
        try:
            agent_save_data = self.agent.get_state_dict()
            agent_save_data["global_step"] = global_step
            agent_save_data["episode_count"] = episode_count
            torch.save(agent_save_data, self.model_save_path)
            print(
                f"  -> Agent checkpoint saved: {os.path.basename(self.model_save_path)}"
            )
        except Exception as e:
            print(f"  -> ERROR saving agent checkpoint: {e}")
            traceback.print_exc()

    def get_initial_state(self) -> Tuple[int, int]:
        return self.global_step, self.episode_count
