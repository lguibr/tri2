# File: config/general.py
import torch
import os
import time
from utils.helpers import get_device

DEVICE = get_device()
RANDOM_SEED = 42
RUN_ID = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
BASE_CHECKPOINT_DIR = "checkpoints"
BASE_LOG_DIR = "logs"

TOTAL_TRAINING_STEPS = 10_000_000

RUN_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, RUN_ID)
RUN_LOG_DIR = os.path.join(BASE_LOG_DIR, "tensorboard", RUN_ID)

MODEL_SAVE_PATH = os.path.join(
    RUN_CHECKPOINT_DIR, "ppo_agent_state.pth"
)  # Renamed file
