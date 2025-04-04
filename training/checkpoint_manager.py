# File: training/checkpoint_manager.py
import os
import torch
import traceback
from typing import Optional, Dict, Any, Tuple

from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from agent.replay_buffer.buffer_utils import create_replay_buffer
from config import BufferConfig, DQNConfig


class CheckpointManager:
    """Handles loading and saving of agent and buffer states."""

    def __init__(
        self,
        agent: DQNAgent,
        buffer: ReplayBufferBase,
        model_save_path: str,
        buffer_save_path: str,
        load_checkpoint_path: Optional[str],
        load_buffer_path: Optional[str],
        buffer_config: BufferConfig,
        dqn_config: DQNConfig,
        device: torch.device,
    ):
        self.agent = agent
        self.buffer = buffer
        self.model_save_path = model_save_path
        self.buffer_save_path = buffer_save_path
        self.buffer_config = buffer_config
        self.dqn_config = dqn_config
        self.device = device

        self.global_step = 0
        self.episode_count = 0

        if load_checkpoint_path:
            self.load_agent_checkpoint(load_checkpoint_path)
        else:
            print("[CheckpointManager] No agent checkpoint specified, starting fresh.")

        if load_buffer_path:
            self.load_buffer_state(load_buffer_path)
        else:
            print("[CheckpointManager] No buffer specified, starting empty.")
            if self.buffer is None:
                self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)

    def load_agent_checkpoint(self, path_to_load: str):
        if not os.path.isfile(path_to_load):
            print(
                f"[CheckpointManager] LOAD WARNING: Agent ckpt not found: {path_to_load}"
            )
            return
        print(f"[CheckpointManager] Loading agent checkpoint from: {path_to_load}")
        try:
            checkpoint = torch.load(path_to_load, map_location=self.device)
            self.agent.load_state_dict(checkpoint)
            self.global_step = checkpoint.get("global_step", 0)
            self.episode_count = checkpoint.get("episode_count", 0)
            print(
                f"  -> Resuming from Step: {self.global_step}, Ep: {self.episode_count}"
            )
        except Exception as e:
            print(f"  -> ERROR loading agent checkpoint ('{e}'). Check compatibility.")
            traceback.print_exc()
            self.global_step = 0
            self.episode_count = 0

    def load_buffer_state(self, path_to_load: str):
        if not os.path.isfile(path_to_load):
            print(
                f"[CheckpointManager] LOAD WARNING: Buffer file not found: {path_to_load}"
            )
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
            return
        print(f"[CheckpointManager] Loading buffer state from: {path_to_load}")
        try:
            # Recreate buffer first to ensure config match
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
            self.buffer.load_state(path_to_load)
            print(f"  -> Buffer loaded. Size: {len(self.buffer)}")
        except Exception as e:
            print(f"  -> ERROR loading buffer state ('{e}'). Starting empty.")
            traceback.print_exc()
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)

    def save_checkpoint(
        self, global_step: int, episode_count: int, is_final: bool = False
    ):
        prefix = "FINAL" if is_final else f"step_{global_step}"
        save_dir = os.path.dirname(self.model_save_path)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[CheckpointManager] Saving agent checkpoint ({prefix})...")
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

        print(f"[CheckpointManager] Saving buffer state ({prefix})...")
        try:
            if hasattr(self.buffer, "flush_pending"):
                self.buffer.flush_pending()
            self.buffer.save_state(self.buffer_save_path)
            print(
                f"  -> Buffer state saved: {os.path.basename(self.buffer_save_path)} (Size: {len(self.buffer)})"
            )
        except Exception as e:
            print(f"  -> ERROR saving buffer state: {e}")
            traceback.print_exc()

    def get_initial_state(self) -> Tuple[int, int]:
        return self.global_step, self.episode_count

    def get_buffer(self) -> ReplayBufferBase:
        return self.buffer
