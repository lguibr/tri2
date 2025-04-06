import os, torch
from .core import (
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    VisConfig,
    DemoConfig,
)
from .general import (
    RUN_ID,
    DEVICE,
    MODEL_SAVE_PATH,
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,
    TOTAL_TRAINING_STEPS,
)


def print_config_info_and_validate():
    env_config_instance = EnvConfig()
    ppo_config_instance = PPOConfig()
    rnn_config_instance = RNNConfig()

    print("-" * 70)
    print(f"RUN ID: {RUN_ID}")
    print(f"Log Directory: {RUN_LOG_DIR}")
    print(f"Checkpoint Directory: {RUN_CHECKPOINT_DIR}")
    print(f"Device: {DEVICE}")
    print(
        f"TB Logging: Histograms={'ON' if TensorBoardConfig.LOG_HISTOGRAMS else 'OFF'}, "
        f"Images={'ON' if TensorBoardConfig.LOG_IMAGES else 'OFF'}"
    )

    if TrainConfig.LOAD_CHECKPOINT_PATH:
        print(
            "*" * 70
            + f"\n*** Warning: LOAD CHECKPOINT from: {TrainConfig.LOAD_CHECKPOINT_PATH} ***\n"
            "*** Ensure ckpt matches current Model/PPO/RNN Config. ***\n" + "*" * 70
        )
    else:
        print("--- Starting training from scratch (no checkpoint specified). ---")

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
        f"--- Using LR Scheduler: {ppo_config_instance.USE_LR_SCHEDULER}"
        + (
            f" (Linear Decay to {ppo_config_instance.LR_SCHEDULER_END_FRACTION * 100}%)"
            if ppo_config_instance.USE_LR_SCHEDULER
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
    print(
        f"--- Rendering {VisConfig.NUM_ENVS_TO_RENDER if VisConfig.NUM_ENVS_TO_RENDER > 0 else 'ALL'} of {env_config_instance.NUM_ENVS} environments ---"
    )
    print("-" * 70)
