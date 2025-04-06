# File: config/general.py
# File: config/general.py
import torch
import os
import time
from typing import Optional
# --- REMOVED: from utils.helpers import get_device ---

# --- MODIFIED: Define DEVICE as placeholder initially ---
DEVICE: Optional[torch.device] = None
# --- END MODIFIED ---

RANDOM_SEED = 42
RUN_ID = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
BASE_CHECKPOINT_DIR = "checkpoints"
BASE_LOG_DIR = "logs"

TOTAL_TRAINING_STEPS = 500_000_000  # 500 Million steps

RUN_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, RUN_ID)
RUN_LOG_DIR = os.path.join(BASE_LOG_DIR, "tensorboard", RUN_ID)

MODEL_SAVE_PATH = os.path.join(RUN_CHECKPOINT_DIR, "ppo_agent_state.pth")


def set_device(device: torch.device):
    """Sets the global DEVICE variable."""
    global DEVICE
    DEVICE = device
    # Update dependent configs if necessary (though direct usage is preferred)
    # Example: If other configs directly reference config.general.DEVICE at import time,
    # this won't update them. It's better to pass the device where needed.
    print(f"[Config] Global DEVICE set to: {DEVICE}")
