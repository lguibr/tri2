# File: init/rl_components_ppo.py
import traceback
import numpy as np
import torch
from typing import List, Optional, Dict, Any

from config import (
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    TensorBoardConfig,
    ObsNormConfig,
    TransformerConfig,
    get_run_log_dir,  # Import getter for TB log dir
)
from environment.game_state import GameState, StateType
from agent.ppo_agent import PPOAgent
from training.trainer import Trainer
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregator
from stats.simple_stats_recorder import SimpleStatsRecorder
from stats.tensorboard_logger import TensorBoardStatsRecorder
from utils.running_mean_std import RunningMeanStd


def initialize_envs(num_envs: int, env_config: EnvConfig) -> List[GameState]:
    """Initializes parallel game environments and performs basic state checks."""
    print(f"Initializing {num_envs} game environments...")
    try:
        envs = [GameState() for _ in range(num_envs)]
        s_test_dict = envs[0].reset()
        if not isinstance(s_test_dict, dict):
            raise TypeError("Env reset did not return a dict.")
        if "grid" not in s_test_dict:
            raise KeyError("State dict missing 'grid'")
        grid_state = s_test_dict["grid"]
        expected_grid_shape = env_config.GRID_STATE_SHAPE
        if (
            not isinstance(grid_state, np.ndarray)
            or grid_state.shape != expected_grid_shape
        ):
            raise ValueError(
                f"Grid shape mismatch! Env:{grid_state.shape}, Cfg:{expected_grid_shape}"
            )
        if "shapes" not in s_test_dict:
            raise KeyError("State dict missing 'shapes'")
        shape_state = s_test_dict["shapes"]
        expected_shape_feature_shape = (env_config.SHAPE_STATE_DIM,)
        if (
            not isinstance(shape_state, np.ndarray)
            or shape_state.shape != expected_shape_feature_shape
        ):
            raise ValueError(
                f"Shape feature shape mismatch! Env:{shape_state.shape}, Cfg:{expected_shape_feature_shape}"
            )
        if "shape_availability" not in s_test_dict:
            raise KeyError("State dict missing 'shape_availability'")
        availability_state = s_test_dict["shape_availability"]
        expected_availability_shape = (env_config.SHAPE_AVAILABILITY_DIM,)
        if (
            not isinstance(availability_state, np.ndarray)
            or availability_state.shape != expected_availability_shape
        ):
            raise ValueError(
                f"Shape availability shape mismatch! Env:{availability_state.shape}, Cfg:{expected_availability_shape}"
            )
        if "explicit_features" not in s_test_dict:
            raise KeyError("State dict missing 'explicit_features'")
        explicit_features_state = s_test_dict["explicit_features"]
        expected_explicit_features_shape = (env_config.EXPLICIT_FEATURES_DIM,)
        if (
            not isinstance(explicit_features_state, np.ndarray)
            or explicit_features_state.shape != expected_explicit_features_shape
        ):
            raise ValueError(
                f"Explicit features shape mismatch! Env:{explicit_features_state.shape}, Cfg:{expected_explicit_features_shape}"
            )
        print("Initial state shape checks PASSED.")
        print(f"Successfully initialized {num_envs} environments.")
        return envs
    except Exception as e:
        print(f"FATAL ERROR during env init: {e}")
        traceback.print_exc()
        raise e


def initialize_agent(
    model_config: ModelConfig,
    ppo_config: PPOConfig,
    rnn_config: RNNConfig,
    env_config: EnvConfig,
    transformer_config: TransformerConfig,
    device: torch.device,
) -> PPOAgent:
    """Initializes the PPO agent."""
    print("Initializing PPO Agent...")
    agent = PPOAgent(
        model_config=model_config,
        ppo_config=ppo_config,
        rnn_config=rnn_config,
        env_config=env_config,
        transformer_config=transformer_config,
        device=device,
    )
    print("PPO Agent initialized.")
    return agent


def initialize_stats_recorder(
    stats_config: StatsConfig,
    tb_config: TensorBoardConfig,
    config_dict: Dict[str, Any],
    agent: Optional[PPOAgent],
    env_config: EnvConfig,
    rnn_config: RNNConfig,
    transformer_config: TransformerConfig,
    is_reinit: bool = False,
) -> StatsRecorderBase:
    """Initializes the statistics recording components (Aggregator, Console, TensorBoard)."""
    print(f"Initializing Statistics Components... Re-init: {is_reinit}")
    stats_aggregator = StatsAggregator(
        avg_windows=stats_config.STATS_AVG_WINDOW,
        plot_window=stats_config.PLOT_DATA_WINDOW,
    )
    console_recorder = SimpleStatsRecorder(
        aggregator=stats_aggregator,
        console_log_interval=stats_config.CONSOLE_LOG_FREQ,
    )
    model_for_graph_cpu = None
    dummy_input_tuple = None
    print("[Stats Init] Model graph logging DISABLED.")
    # Use the getter for the log directory
    current_run_log_dir = get_run_log_dir()
    print(f"Using TensorBoard Logger (Log Dir: {current_run_log_dir})")
    try:
        tb_recorder = TensorBoardStatsRecorder(
            aggregator=stats_aggregator,
            console_recorder=console_recorder,
            log_dir=current_run_log_dir,  # Pass the dynamically obtained log dir
            hparam_dict=(config_dict if not is_reinit else None),
            model_for_graph=model_for_graph_cpu,
            dummy_input_for_graph=dummy_input_tuple,
            histogram_log_interval=(
                tb_config.HISTOGRAM_LOG_FREQ if tb_config.LOG_HISTOGRAMS else -1
            ),
            image_log_interval=(
                tb_config.IMAGE_LOG_FREQ if tb_config.LOG_IMAGES else -1
            ),
            env_config=env_config,
            rnn_config=rnn_config,
        )
        print("Statistics Components initialized successfully.")
        return tb_recorder
    except Exception as e:
        print(f"FATAL: Error initializing TensorBoardStatsRecorder: {e}. Exiting.")
        traceback.print_exc()
        raise e


def initialize_trainer(
    envs: List[GameState],
    agent: PPOAgent,
    stats_recorder: StatsRecorderBase,
    env_config: EnvConfig,
    ppo_config: PPOConfig,
    rnn_config: RNNConfig,
    train_config: TrainConfig,
    model_config: ModelConfig,
    obs_norm_config: ObsNormConfig,
    transformer_config: TransformerConfig,
    device: torch.device,
    # model_save_path: str, # Removed parameter
    load_checkpoint_path: Optional[str],
) -> Trainer:
    """Initializes the PPO Trainer."""
    print("Initializing PPO Trainer...")
    trainer = Trainer(
        envs=envs,
        agent=agent,
        stats_recorder=stats_recorder,
        env_config=env_config,
        ppo_config=ppo_config,
        rnn_config=rnn_config,
        train_config=train_config,
        model_config=model_config,
        obs_norm_config=obs_norm_config,
        transformer_config=transformer_config,
        # model_save_path=model_save_path, # Removed argument
        load_checkpoint_path=load_checkpoint_path,
        device=device,
    )
    print("PPO Trainer initialization finished.")
    return trainer
