import torch
import os
import time
from typing import Optional

DEVICE: Optional[torch.device] = None

RANDOM_SEED = 42
RUN_ID = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
BASE_CHECKPOINT_DIR = "checkpoints"
BASE_LOG_DIR = "logs"

# Reduced total training steps
TOTAL_TRAINING_STEPS = 20_000_000

RUN_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, RUN_ID)
RUN_LOG_DIR = os.path.join(BASE_LOG_DIR, "tensorboard", RUN_ID)

MODEL_SAVE_PATH = os.path.join(RUN_CHECKPOINT_DIR, "ppo_agent_state.pth")


def set_device(device: torch.device):
    """Sets the global DEVICE variable."""
    global DEVICE
    DEVICE = device
    print(f"[Config] Global DEVICE set to: {DEVICE}")
