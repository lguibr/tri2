# File: config/validation.py
from .core import (
    EnvConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    VisConfig,
    TransformerConfig,
    MCTSConfig,
)
from .general import (
    DEVICE,
    get_run_id,
    get_run_log_dir,
    get_run_checkpoint_dir,
)


def print_config_info_and_validate():
    env_config_instance = EnvConfig()
    rnn_config_instance = RNNConfig()
    transformer_config_instance = TransformerConfig()
    train_config_instance = TrainConfig()
    mcts_config_instance = MCTSConfig()
    vis_config_instance = VisConfig()  # Get instance for NUM_ENVS_TO_RENDER
    stats_config_instance = StatsConfig()  # Get instance for logging frequency

    run_id = get_run_id()
    run_log_dir = get_run_log_dir()
    run_checkpoint_dir = get_run_checkpoint_dir()

    print("-" * 70)
    print(f"RUN ID: {run_id}")
    print(f"Log Directory: {run_log_dir}")
    print(f"Checkpoint Directory: {run_checkpoint_dir}")
    print(f"Device: {DEVICE}")

    if train_config_instance.LOAD_CHECKPOINT_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD CHECKPOINT specified: {train_config_instance.LOAD_CHECKPOINT_PATH} ***\n"
            "*** CheckpointManager will attempt to load this path (NN weights, Optimizer, Stats). ***\n"
            + "*" * 70
        )
    else:
        print(
            "--- No explicit checkpoint path. CheckpointManager will attempt auto-resume if applicable. ---"
        )

    print("--- Training Algorithm: AlphaZero (MCTS + NN) ---")

    if rnn_config_instance.USE_RNN:
        print(
            f"--- Warning: RNN configured ON ({rnn_config_instance.LSTM_HIDDEN_SIZE}, {rnn_config_instance.LSTM_NUM_LAYERS}) but not used by AlphaZeroNet ---"
        )
    if transformer_config_instance.USE_TRANSFORMER:
        print(
            f"--- Warning: Transformer configured ON ({transformer_config_instance.TRANSFORMER_D_MODEL}, {transformer_config_instance.TRANSFORMER_NHEAD}, {transformer_config_instance.TRANSFORMER_NUM_LAYERS}) but not used by AlphaZeroNet ---"
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
    dropout = ModelConfig.Network.DROPOUT_FC
    print(
        f"Network Base: CNN={cnn_str}, ShapeMLP={shape_mlp_cfg_str}, Fusion={mlp_str}, Dropout={dropout}"
    )

    print(
        f"MCTS: Sims={mcts_config_instance.NUM_SIMULATIONS}, "
        f"PUCT_C={mcts_config_instance.PUCT_C:.2f}, "
        f"Temp={mcts_config_instance.TEMPERATURE_INITIAL:.2f}->{mcts_config_instance.TEMPERATURE_FINAL:.2f} (Steps: {mcts_config_instance.TEMPERATURE_ANNEAL_STEPS}), "
        f"Dirichlet(α={mcts_config_instance.DIRICHLET_ALPHA:.2f}, ε={mcts_config_instance.DIRICHLET_EPSILON:.2f})"
    )

    scheduler_info = "DISABLED"
    if train_config_instance.USE_LR_SCHEDULER:
        scheduler_info = f"{train_config_instance.SCHEDULER_TYPE} (T_max={train_config_instance.SCHEDULER_T_MAX:,}, eta_min={train_config_instance.SCHEDULER_ETA_MIN:.1e})"
    print(
        f"Training: Batch={train_config_instance.BATCH_SIZE}, LR={train_config_instance.LEARNING_RATE:.1e}, "
        f"WD={train_config_instance.WEIGHT_DECAY:.1e}, Scheduler={scheduler_info}"
    )
    print(
        f"Buffer: Capacity={train_config_instance.BUFFER_CAPACITY:,}, MinSize={train_config_instance.MIN_BUFFER_SIZE_TO_TRAIN:,}, Steps/Iter={train_config_instance.NUM_TRAINING_STEPS_PER_ITER}"
    )
    print(
        f"Workers: Self-Play={train_config_instance.NUM_SELF_PLAY_WORKERS}, Training=1"
    )
    print(
        f"Stats: AVG_WINDOWS={stats_config_instance.STATS_AVG_WINDOW}, Console Log Freq={stats_config_instance.CONSOLE_LOG_FREQ} (updates/episodes)"
    )

    render_info = (
        f"Live Self-Play Workers (Max: {vis_config_instance.NUM_ENVS_TO_RENDER})"
    )
    print(f"--- Rendering {render_info} in Game Area ---")
    print("-" * 70)
