# File: config/validation.py
from .core import (
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    VisConfig,
    TransformerConfig,
)
from .general import (
    DEVICE,
    # Removed TOTAL_TRAINING_STEPS
    get_run_id,
    get_run_log_dir,
    get_run_checkpoint_dir,
    get_model_save_path,
)


def print_config_info_and_validate():
    env_config_instance = EnvConfig()
    rnn_config_instance = RNNConfig()
    transformer_config_instance = TransformerConfig()
    vis_config_instance = VisConfig()
    train_config_instance = TrainConfig()

    run_id = get_run_id()
    run_log_dir = get_run_log_dir()
    run_checkpoint_dir = get_run_checkpoint_dir()

    print("-" * 70)
    print(f"RUN ID: {run_id}")
    print(f"Log Directory: {run_log_dir}")
    print(f"Checkpoint Directory: {run_checkpoint_dir}")
    print(f"Device: {DEVICE}")
    print(
        f"TB Logging: Histograms={'ON' if TensorBoardConfig.LOG_HISTOGRAMS else 'OFF'}, "
        f"Images={'ON' if TensorBoardConfig.LOG_IMAGES else 'OFF'}"
    )

    if train_config_instance.LOAD_CHECKPOINT_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD CHECKPOINT specified: {train_config_instance.LOAD_CHECKPOINT_PATH} ***\n"
            "*** CheckpointManager will attempt to load this path (e.g., NN weights). ***\n"
            + "*" * 70
        )
    else:
        print(
            "--- No explicit checkpoint path. CheckpointManager will attempt auto-resume if applicable. ---"
        )

    print("--- Training Algorithm: AlphaZero (MCTS + NN) ---")  # Updated description

    # Removed PPO specific prints

    print(
        f"--- Using RNN: {rnn_config_instance.USE_RNN}"
        + (
            f" (LSTM Hidden: {rnn_config_instance.LSTM_HIDDEN_SIZE}, Layers: {rnn_config_instance.LSTM_NUM_LAYERS})"
            if rnn_config_instance.USE_RNN
            else ""
        )
        + " ---"
    )
    print(
        f"--- Using Transformer: {transformer_config_instance.USE_TRANSFORMER}"
        + (
            f" (d_model={transformer_config_instance.TRANSFORMER_D_MODEL}, nhead={transformer_config_instance.TRANSFORMER_NHEAD}, layers={transformer_config_instance.TRANSFORMER_NUM_LAYERS})"
            if transformer_config_instance.USE_TRANSFORMER
            else ""
        )
        + " ---"
    )
    # Removed ObsNorm print

    print(
        f"Config: Env=(R={env_config_instance.ROWS}, C={env_config_instance.COLS}), "
        f"GridState={env_config_instance.GRID_STATE_SHAPE}, "
        f"ShapeState={env_config_instance.SHAPE_STATE_DIM}, "
        f"ActionDim={env_config_instance.ACTION_DIM}"
    )
    cnn_str = str(ModelConfig.Network.CONV_CHANNELS).replace(" ", "")
    mlp_str = str(ModelConfig.Network.COMBINED_FC_DIMS).replace(" ", "")
    shape_mlp_cfg_str = str(ModelConfig.Network.SHAPE_FEATURE_MLP_DIMS).replace(" ", "")
    print(
        f"Network Base: CNN={cnn_str}, ShapeMLP={shape_mlp_cfg_str}, Fusion={mlp_str}"
    )  # Adapted description

    print(f"Training: NUM_ENVS={env_config_instance.NUM_ENVS}")  # Removed total steps
    print(
        f"Stats: AVG_WINDOWS={StatsConfig.STATS_AVG_WINDOW}, Console Log Freq={StatsConfig.CONSOLE_LOG_FREQ} (episodes/updates)"  # Adapted freq description
    )

    if env_config_instance.NUM_ENVS >= 1024:  # Keep warning, though NUM_ENVS is now 1
        device_type = DEVICE.type if DEVICE else "UNKNOWN"
        print(
            "*" * 70
            + f"\n*** Warning: NUM_ENVS={env_config_instance.NUM_ENVS}. Monitor system resources. ***"
            + (
                "\n*** Using MPS device. Performance varies. Force CPU via env var if needed. ***"
                if device_type == "mps"
                else ""
            )
            + "\n"
            + "*" * 70
        )
    print(
        f"--- Rendering {VisConfig.NUM_ENVS_TO_RENDER if VisConfig.NUM_ENVS_TO_RENDER > 0 else 'ALL'} of {env_config_instance.NUM_ENVS} environments ---"
    )
    print("-" * 70)
