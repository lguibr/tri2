# File: config/validation.py
import os, torch
from .core import (
    EnvConfig,
    DQNConfig,
    TrainConfig,
    BufferConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    VisConfig,
)
from .general import (
    RUN_ID,
    DEVICE,
    MODEL_SAVE_PATH,
    BUFFER_SAVE_PATH,
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,
    TOTAL_TRAINING_STEPS,
)


def print_config_info_and_validate():
    print("-" * 70)
    print(f"RUN ID: {RUN_ID}")
    print(f"Log Directory: {RUN_LOG_DIR}")
    print(f"Checkpoint Directory: {RUN_CHECKPOINT_DIR}")
    print(f"Device: {DEVICE}")
    print(
        f"TB Logging: Histograms={'ON' if TensorBoardConfig.LOG_HISTOGRAMS else 'OFF'}, Images={'ON' if TensorBoardConfig.LOG_IMAGES else 'OFF'}"
    )
    if EnvConfig.GRID_FEATURES_PER_CELL != 3:
        print(
            "Warning: Network assumes 3 features per cell (Occupied, Is_Up, Is_Death)."
        )
    if TrainConfig.LOAD_CHECKPOINT_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD CHECKPOINT from: {TrainConfig.LOAD_CHECKPOINT_PATH} ***\n*** Ensure ckpt matches current Model/DQN Config (distributional, scheduler). ***\n"
            + "*" * 70
        )
    else:
        print("--- Starting training from scratch (no checkpoint specified). ---")
    if TrainConfig.LOAD_BUFFER_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD BUFFER from: {TrainConfig.LOAD_BUFFER_PATH} ***\n*** Ensure buffer matches current Buffer Config (PER, N-Step). ***\n"
            + "*" * 70
        )
    else:
        print("--- Starting with an empty replay buffer (no buffer specified). ---")
    print(f"--- Using Noisy Nets: {DQNConfig.USE_NOISY_NETS} ---")
    print(
        f"--- Using Distributional (C51): {DQNConfig.USE_DISTRIBUTIONAL} (Atoms: {DQNConfig.NUM_ATOMS}, Vmin: {DQNConfig.V_MIN}, Vmax: {DQNConfig.V_MAX}) ---"
    )
    print(
        f"--- Using LR Scheduler: {DQNConfig.USE_LR_SCHEDULER}"
        + (
            f" (CosineAnnealingLR, T_max={DQNConfig.LR_SCHEDULER_T_MAX}, eta_min={DQNConfig.LR_SCHEDULER_ETA_MIN})"
            if DQNConfig.USE_LR_SCHEDULER
            else ""
        )
        + " ---"
    )
    print(
        f"Config: Env=(R={EnvConfig.ROWS}, C={EnvConfig.COLS}), StateDim={EnvConfig.STATE_DIM}, ActionDim={EnvConfig.ACTION_DIM}"
    )
    cnn_str = str(ModelConfig.Network.CONV_CHANNELS).replace(" ", "")
    mlp_str = str(ModelConfig.Network.COMBINED_FC_DIMS).replace(" ", "")
    print(
        f"Network: CNN={cnn_str}, ShapeMLP={ModelConfig.Network.SHAPE_MLP_HIDDEN_DIM}, Fusion={mlp_str}, Dueling={DQNConfig.USE_DUELING}"
    )
    print(
        f"Training: NUM_ENVS={EnvConfig.NUM_ENVS}, TOTAL_STEPS={TOTAL_TRAINING_STEPS/1e6:.1f}M, BUFFER={BufferConfig.REPLAY_BUFFER_SIZE/1e6:.1f}M, BATCH={TrainConfig.BATCH_SIZE}"
    )
    print(
        f"Buffer: PER={BufferConfig.USE_PER}, N-Step={BufferConfig.N_STEP if BufferConfig.USE_N_STEP else 'N/A'}"
    )
    print(
        f"Stats: AVG_WINDOW={StatsConfig.STATS_AVG_WINDOW}, Console Log Freq={StatsConfig.CONSOLE_LOG_FREQ}"
    )
    if EnvConfig.NUM_ENVS >= 1024:
        print(
            "*" * 70
            + f"\n*** Warning: NUM_ENVS={EnvConfig.NUM_ENVS}. Monitor system resources. ***"
            + (
                "\n*** Using MPS device. Performance varies. Force CPU via env var if needed. ***"
                if DEVICE.type == "mps"
                else ""
            )
            + "\n"
            + "*" * 70
        )
    print(
        f"--- Rendering {VisConfig.NUM_ENVS_TO_RENDER if VisConfig.NUM_ENVS_TO_RENDER > 0 else 'ALL'} of {EnvConfig.NUM_ENVS} environments ---"
    )
    print("-" * 70)
