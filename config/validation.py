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
    # --- MODIFIED: Instantiate EnvConfig ---
    env_config_instance = EnvConfig()
    # --- END MODIFIED ---

    print("-" * 70)
    print(f"RUN ID: {RUN_ID}")
    print(f"Log Directory: {RUN_LOG_DIR}")
    print(f"Checkpoint Directory: {RUN_CHECKPOINT_DIR}")
    print(f"Device: {DEVICE}")
    print(
        f"TB Logging: Histograms={'ON' if TensorBoardConfig.LOG_HISTOGRAMS else 'OFF'}, Images={'ON' if TensorBoardConfig.LOG_IMAGES else 'OFF'}"
    )
    # --- MODIFIED: Check GRID_FEATURES_PER_CELL on instance ---
    if env_config_instance.GRID_FEATURES_PER_CELL != 3:
        # --- END MODIFIED ---
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
    # --- MODIFIED: Print state shapes and action dim from instance ---
    print(
        f"Config: Env=(R={env_config_instance.ROWS}, C={env_config_instance.COLS}), "
        f"GridState={env_config_instance.GRID_STATE_SHAPE}, "
        f"ShapeState={env_config_instance.SHAPE_STATE_DIM}, "
        f"ActionDim={env_config_instance.ACTION_DIM}"
    )
    # --- END MODIFIED ---
    cnn_str = str(ModelConfig.Network.CONV_CHANNELS).replace(" ", "")
    mlp_str = str(ModelConfig.Network.COMBINED_FC_DIMS).replace(" ", "")
    # --- MODIFIED: Print shape MLP dims from config ---
    shape_mlp_cfg_str = str(ModelConfig.Network.SHAPE_FEATURE_MLP_DIMS).replace(" ", "")
    print(
        f"Network: CNN={cnn_str}, ShapeEmb={ModelConfig.Network.SHAPE_EMBEDDING_DIM}, ShapeMLP={shape_mlp_cfg_str}, Fusion={mlp_str}, Dueling={DQNConfig.USE_DUELING}"
    )
    # --- END MODIFIED ---
    # --- MODIFIED: Access NUM_ENVS from instance ---
    print(
        f"Training: NUM_ENVS={env_config_instance.NUM_ENVS}, TOTAL_STEPS={TOTAL_TRAINING_STEPS/1e6:.1f}M, BUFFER={BufferConfig.REPLAY_BUFFER_SIZE/1e6:.1f}M, BATCH={TrainConfig.BATCH_SIZE}"
    )
    # --- END MODIFIED ---
    print(
        f"Buffer: PER={BufferConfig.USE_PER}, N-Step={BufferConfig.N_STEP if BufferConfig.USE_N_STEP else 'N/A'}"
    )
    print(
        f"Stats: AVG_WINDOW={StatsConfig.STATS_AVG_WINDOW}, Console Log Freq={StatsConfig.CONSOLE_LOG_FREQ}"
    )
    # --- MODIFIED: Access NUM_ENVS from instance ---
    if env_config_instance.NUM_ENVS >= 1024:
        # --- END MODIFIED ---
        print(
            "*" * 70
            + f"\n*** Warning: NUM_ENVS={env_config_instance.NUM_ENVS}. Monitor system resources. ***"
            + (
                "\n*** Using MPS device. Performance varies. Force CPU via env var if needed. ***"
                if DEVICE.type == "mps"
                else ""
            )
            + "\n"
            + "*" * 70
        )
    # --- MODIFIED: Access NUM_ENVS from instance ---
    print(
        f"--- Rendering {VisConfig.NUM_ENVS_TO_RENDER if VisConfig.NUM_ENVS_TO_RENDER > 0 else 'ALL'} of {env_config_instance.NUM_ENVS} environments ---"
    )
    # --- END MODIFIED ---
    print("-" * 70)
