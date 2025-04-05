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
        self.buffer = buffer  # Store initial reference
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

        # --- Load or create buffer ---
        if load_buffer_path:
            self.load_buffer_state(load_buffer_path)
        else:
            print("[CheckpointManager] No buffer specified, creating new empty buffer.")
            # Create a new buffer if none was loaded
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)

        # --- Ensure buffer is initialized ---
        if self.buffer is None:
            print(
                "[CheckpointManager] ERROR: Buffer is None after initialization/loading attempt. Creating empty."
            )
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)

    def load_agent_checkpoint(self, path_to_load: str):
        # Logic remains the same
        if not os.path.isfile(path_to_load):
            print(
                f"[CheckpointManager] LOAD WARNING: Agent ckpt not found: {path_to_load}"
            )
            return
        print(f"[CheckpointManager] Loading agent checkpoint from: {path_to_load}")
        try:
            # Load checkpoint to the specified device
            checkpoint = torch.load(path_to_load, map_location=self.device)
            # --- Pass buffer_config to agent's load_state_dict ---
            self.agent.load_state_dict(checkpoint)
            # --- Get state from checkpoint AFTER loading agent ---
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

    def load_buffer_state(self, path_to_load: str):
        # Logic remains the same, but recreate buffer first
        if not os.path.isfile(path_to_load):
            print(
                f"[CheckpointManager] LOAD WARNING: Buffer file not found: {path_to_load}"
            )
            # Create new buffer if load path invalid
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
            return

        print(
            f"[CheckpointManager] Recreating buffer structure before loading state from: {path_to_load}"
        )
        try:
            # Create buffer instance based on config FIRST
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)
            # Then load the state into the newly created instance
            self.buffer.load_state(path_to_load)
            print(f"  -> Buffer loaded. Size: {len(self.buffer)}")
        except Exception as e:
            print(f"  -> ERROR loading buffer state ('{e}'). Creating empty buffer.")
            traceback.print_exc()
            # Create new buffer on error
            self.buffer = create_replay_buffer(self.buffer_config, self.dqn_config)

    def save_checkpoint(
        self, global_step: int, episode_count: int, is_final: bool = False
    ):
        # Logic remains the same for agent
        prefix = "FINAL" if is_final else f"step_{global_step}"
        save_dir = os.path.dirname(self.model_save_path)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[CheckpointManager] Saving agent checkpoint ({prefix})...")
        try:
            agent_save_data = self.agent.get_state_dict()
            # Add step/episode info directly to the agent state dict
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
            # --- MODIFIED: Flush pending transitions before saving buffer ---
            if hasattr(self.buffer, "flush_pending"):
                print("  -> Flushing pending buffer transitions before saving...")
                self.buffer.flush_pending()
            # --- END MODIFIED ---
            self.buffer.save_state(self.buffer_save_path)
            print(
                f"  -> Buffer state saved: {os.path.basename(self.buffer_save_path)} (Size: {len(self.buffer)})"
            )
        except Exception as e:
            print(f"  -> ERROR saving buffer state: {e}")
            traceback.print_exc()

    def get_initial_state(self) -> Tuple[int, int]:
        # Logic remains the same
        return self.global_step, self.episode_count

    def get_buffer(self) -> ReplayBufferBase:
        # Logic remains the same
        if self.buffer is None:
            raise RuntimeError(
                "Buffer accessed before initialization in CheckpointManager."
            )
        return self.buffer
