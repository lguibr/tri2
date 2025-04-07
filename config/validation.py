# File: config/validation.py
from .core import (
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    VisConfig,
    ObsNormConfig,
    TransformerConfig,
)
from .general import (
    DEVICE,
    TOTAL_TRAINING_STEPS,
    # Import getters instead of direct constants
    get_run_id,
    get_run_log_dir,
    get_run_checkpoint_dir,
    get_model_save_path,  # Keep if needed, or remove if only used in trainer
)


def print_config_info_and_validate():
    env_config_instance = EnvConfig()
    ppo_config_instance = PPOConfig()
    rnn_config_instance = RNNConfig()
    transformer_config_instance = TransformerConfig()
    obs_norm_config_instance = ObsNormConfig()
    vis_config_instance = VisConfig()
    train_config_instance = TrainConfig()  # Instantiate to access LOAD_CHECKPOINT_PATH

    # Use getter functions for dynamic paths
    run_id = get_run_id()
    run_log_dir = get_run_log_dir()
    run_checkpoint_dir = get_run_checkpoint_dir()

    print("-" * 70)
    print(f"RUN ID: {run_id}")  # Use variable
    print(f"Log Directory: {run_log_dir}")  # Use variable
    print(f"Checkpoint Directory: {run_checkpoint_dir}")  # Use variable
    print(f"Device: {DEVICE}")
    print(
        f"TB Logging: Histograms={'ON' if TensorBoardConfig.LOG_HISTOGRAMS else 'OFF'}, "
        f"Images={'ON' if TensorBoardConfig.LOG_IMAGES else 'OFF'}"
    )

    # Use the instance to check the config value
    if train_config_instance.LOAD_CHECKPOINT_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD CHECKPOINT specified: {train_config_instance.LOAD_CHECKPOINT_PATH} ***\n"
            "*** CheckpointManager will attempt to load this path. ***\n" + "*" * 70
        )
    else:
        print(
            "--- No explicit checkpoint path. CheckpointManager will attempt auto-resume. ---"
        )

    print("--- Pre-training DISABLED ---")

    print(f"--- Using PPO Algorithm ---")
    print(f"    Rollout Steps: {ppo_config_instance.NUM_STEPS_PER_ROLLOUT}")
    print(f"    PPO Epochs: {ppo_config_instance.PPO_EPOCHS}")
    print(
        f"    Minibatches: {ppo_config_instance.NUM_MINIBATCHES} (Size: {ppo_config_instance.MINIBATCH_SIZE})"
    )
    print(f"    Clip Param: {ppo_config_instance.CLIP_PARAM}")
    print(f"    GAE Lambda: {ppo_config_instance.GAE_LAMBDA}")
    print(
        f"    Value Coef: {ppo_config_instance.VALUE_LOSS_COEF}, Entropy Coef: {ppo_config_instance.ENTROPY_COEF}"
    )

    lr_schedule_str = ""
    if ppo_config_instance.USE_LR_SCHEDULER:
        schedule_type = getattr(ppo_config_instance, "LR_SCHEDULE_TYPE", "linear")
        if schedule_type == "linear":
            end_fraction = getattr(ppo_config_instance, "LR_LINEAR_END_FRACTION", 0.0)
            lr_schedule_str = f" (Linear Decay to {end_fraction * 100}%)"
        elif schedule_type == "cosine":
            min_factor = getattr(ppo_config_instance, "LR_COSINE_MIN_FACTOR", 0.01)
            lr_schedule_str = f" (Cosine Decay to {min_factor * 100}%)"
        else:
            lr_schedule_str = f" (Unknown Schedule: {schedule_type})"

    print(
        f"--- Using LR Scheduler: {ppo_config_instance.USE_LR_SCHEDULER}"
        + lr_schedule_str
        + " ---"
    )

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
    print(
        f"--- Using Obs Normalization: {obs_norm_config_instance.ENABLE_OBS_NORMALIZATION}"
        + (
            f" (Grid:{obs_norm_config_instance.NORMALIZE_GRID}, Shapes:{obs_norm_config_instance.NORMALIZE_SHAPES}, Avail:{obs_norm_config_instance.NORMALIZE_AVAILABILITY}, Explicit:{obs_norm_config_instance.NORMALIZE_EXPLICIT_FEATURES}, Clip:{obs_norm_config_instance.OBS_CLIP})"
            if obs_norm_config_instance.ENABLE_OBS_NORMALIZATION
            else ""
        )
        + " ---"
    )

    print(
        f"Config: Env=(R={env_config_instance.ROWS}, C={env_config_instance.COLS}), "
        f"GridState={env_config_instance.GRID_STATE_SHAPE}, "
        f"ShapeState={env_config_instance.SHAPE_STATE_DIM}, "
        f"ActionDim={env_config_instance.ACTION_DIM}"
    )
    cnn_str = str(ModelConfig.Network.CONV_CHANNELS).replace(" ", "")
    mlp_str = str(ModelConfig.Network.COMBINED_FC_DIMS).replace(" ", "")
    shape_mlp_cfg_str = str(ModelConfig.Network.SHAPE_FEATURE_MLP_DIMS).replace(" ", "")
    print(f"Network: CNN={cnn_str}, ShapeMLP={shape_mlp_cfg_str}, Fusion={mlp_str}")

    print(
        f"Training: NUM_ENVS={env_config_instance.NUM_ENVS}, TOTAL_STEPS={TOTAL_TRAINING_STEPS/1e6:.1f}M"
    )
    print(
        f"Stats: AVG_WINDOWS={StatsConfig.STATS_AVG_WINDOW}, Console Log Freq={StatsConfig.CONSOLE_LOG_FREQ} (rollouts)"
    )

    if env_config_instance.NUM_ENVS >= 1024:
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
