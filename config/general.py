# File: config/general.py
# --- General Constants and Paths ---
import torch
import os
import time
from utils.helpers import get_device

# --- General ---
DEVICE = get_device()
RANDOM_SEED = 42
RUN_ID = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
BASE_CHECKPOINT_DIR = "checkpoints"
BASE_LOG_DIR = "logs"

# --- Define Total Steps Here ---
# This makes it available for default LR scheduler T_max
TOTAL_TRAINING_STEPS = 10_000_000

# --- Derived Paths (using RUN_ID) ---
RUN_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, RUN_ID)
RUN_LOG_DIR = os.path.join(BASE_LOG_DIR, "tensorboard", RUN_ID)

BUFFER_SAVE_PATH = os.path.join(RUN_CHECKPOINT_DIR, "replay_buffer_state.pkl")
MODEL_SAVE_PATH = os.path.join(RUN_CHECKPOINT_DIR, "dqn_agent_state.pth")

# --- REMOVED Assign derived paths to Config classes ---
# from .core import TensorBoardConfig # <<< REMOVE THIS IMPORT
# TensorBoardConfig.LOG_DIR = RUN_LOG_DIR # <<< REMOVE THIS ASSIGNMENT
# --- END REMOVED ---
