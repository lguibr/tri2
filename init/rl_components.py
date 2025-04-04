# File: init/rl_components.py
import sys
import traceback
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Callable

# Import configurations
from config import (
    EnvConfig,
    DQNConfig,
    TrainConfig,
    BufferConfig,
    ModelConfig,
    StatsConfig,
    RewardConfig,
    TensorBoardConfig,
    DEVICE,
    BUFFER_SAVE_PATH,
    MODEL_SAVE_PATH,
    get_config_dict,
)

# Import core components
try:
    from environment.game_state import GameState, StateType
except ImportError as e:
    print(f"Error importing environment: {e}")
    sys.exit(1)
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from agent.replay_buffer.buffer_utils import create_replay_buffer
from training.trainer import Trainer

# --- MODIFIED IMPORTS ---
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregator
from stats.simple_stats_recorder import SimpleStatsRecorder
from stats.tensorboard_logger import TensorBoardStatsRecorder

# --- END MODIFIED ---


def initialize_envs(num_envs: int, env_config: EnvConfig) -> List[GameState]:
    print(f"Initializing {num_envs} game environments...")
    try:
        envs = [GameState() for _ in range(num_envs)]
        s_test_dict = envs[0].reset()

        if not isinstance(s_test_dict, dict):
            raise TypeError("Env reset did not return a dictionary state.")
        if "grid" not in s_test_dict:
            raise KeyError("State dict missing 'grid'")
        grid_state = s_test_dict["grid"]
        expected_grid_shape = env_config.GRID_STATE_SHAPE
        if (
            not isinstance(grid_state, np.ndarray)
            or grid_state.shape != expected_grid_shape
        ):
            raise ValueError(
                f"Initial grid state shape mismatch! Env:{grid_state.shape}, Cfg:{expected_grid_shape}"
            )
        print(f"Initial grid state shape check PASSED: {grid_state.shape}")

        if "shapes" not in s_test_dict:
            raise KeyError("State dict missing 'shapes'")
        shape_state = s_test_dict["shapes"]
        expected_shape_shape = (
            env_config.NUM_SHAPE_SLOTS,
            env_config.SHAPE_FEATURES_PER_SHAPE,
        )
        if (
            not isinstance(shape_state, np.ndarray)
            or shape_state.shape != expected_shape_shape
        ):
            raise ValueError(
                f"Initial shape state shape mismatch! Env:{shape_state.shape}, Cfg:{expected_shape_shape}"
            )
        print(f"Initial shape state shape check PASSED: {shape_state.shape}")

        valid_acts_init = envs[0].valid_actions()
        if valid_acts_init:
            _, _ = envs[0].step(valid_acts_init[0])
        else:
            print(
                "Warning: No valid actions available after initial reset for testing step()."
            )

        print(f"Successfully initialized {num_envs} environments.")
        return envs
    except Exception as e:
        print(f"FATAL ERROR during env init: {e}")
        traceback.print_exc()
        raise e


def initialize_agent_buffer(
    model_config: ModelConfig,
    dqn_config: DQNConfig,
    env_config: EnvConfig,
    buffer_config: BufferConfig,
) -> Tuple[DQNAgent, ReplayBufferBase]:
    print("Initializing Agent and Buffer...")
    agent = DQNAgent(config=model_config, dqn_config=dqn_config, env_config=env_config)
    buffer = create_replay_buffer(config=buffer_config, dqn_config=dqn_config)
    print("Agent and Buffer initialized.")
    return agent, buffer


# --- MODIFIED: initialize_stats_recorder ---
def initialize_stats_recorder(
    stats_config: StatsConfig,
    tb_config: TensorBoardConfig,
    config_dict: Dict[str, Any],
    agent: Optional[DQNAgent],
    env_config: EnvConfig,
) -> StatsRecorderBase:
    """
    Creates the StatsAggregator, SimpleStatsRecorder, and TensorBoardStatsRecorder.
    Returns the final TensorBoardStatsRecorder instance.
    """
    print(f"Initializing Statistics Components...")

    # 1. Create the Aggregator
    stats_aggregator = StatsAggregator(
        avg_window=stats_config.STATS_AVG_WINDOW,
        plot_window=stats_config.PLOT_DATA_WINDOW,
    )

    # 2. Create the Console Logger (uses the aggregator)
    console_recorder = SimpleStatsRecorder(
        aggregator=stats_aggregator,
        console_log_interval=stats_config.CONSOLE_LOG_FREQ,
    )

    # 3. Prepare for TensorBoard Logger (Graph, HParams)
    dummy_grid_cpu = None
    dummy_shapes_cpu = None
    model_for_graph_cpu = None
    if agent and agent.online_net:
        try:
            expected_grid_shape = env_config.GRID_STATE_SHAPE
            dummy_grid_np = np.zeros(expected_grid_shape, dtype=np.float32)
            dummy_shapes_np = np.zeros(
                (env_config.NUM_SHAPE_SLOTS, env_config.SHAPE_FEATURES_PER_SHAPE),
                dtype=np.float32,
            )
            dummy_grid_cpu = torch.tensor(dummy_grid_np, device="cpu").unsqueeze(0)
            dummy_shapes_cpu = torch.tensor(dummy_shapes_np, device="cpu").unsqueeze(0)

            if not hasattr(agent, "dqn_config") or not hasattr(
                agent.online_net, "config"
            ):
                raise AttributeError(
                    "Agent or network missing required config attributes."
                )

            model_for_graph_cpu = type(agent.online_net)(
                env_config=env_config,
                action_dim=env_config.ACTION_DIM,
                model_config=agent.online_net.config,
                dqn_config=agent.dqn_config,
                dueling=agent.online_net.dueling,
                use_noisy=agent.online_net.use_noisy,
            ).to("cpu")
            model_for_graph_cpu.load_state_dict(agent.online_net.state_dict())
            model_for_graph_cpu.eval()
            print("[Stats Init] Prepared model copy and dummy input on CPU for graph.")
        except Exception as e:
            print(f"Warning: Failed to prepare model/input for graph logging: {e}")
            traceback.print_exc()
            dummy_grid_cpu = None
            dummy_shapes_cpu = None
            model_for_graph_cpu = None

    # 4. Create the TensorBoard Logger (uses aggregator and console logger)
    print(f"Using TensorBoard Logger (Log Dir: {tb_config.LOG_DIR})")
    try:
        dummy_input_tuple = (
            (dummy_grid_cpu, dummy_shapes_cpu)
            if dummy_grid_cpu is not None and dummy_shapes_cpu is not None
            else None
        )
        tb_recorder = TensorBoardStatsRecorder(
            aggregator=stats_aggregator,
            console_recorder=console_recorder,
            log_dir=tb_config.LOG_DIR,
            hparam_dict=config_dict,
            model_for_graph=model_for_graph_cpu,
            dummy_input_for_graph=dummy_input_tuple,
            histogram_log_interval=(
                tb_config.HISTOGRAM_LOG_FREQ if tb_config.LOG_HISTOGRAMS else -1
            ),
            image_log_interval=(
                tb_config.IMAGE_LOG_FREQ if tb_config.LOG_IMAGES else -1
            ),
            shape_q_log_interval=(
                tb_config.SHAPE_Q_LOG_FREQ
                if tb_config.LOG_SHAPE_PLACEMENT_Q_VALUES
                else -1
            ),
        )
        print("Statistics Components initialized successfully.")
        return tb_recorder  # Return the main recorder instance
    except Exception as e:
        print(f"FATAL: Error initializing TensorBoardStatsRecorder: {e}. Exiting.")
        traceback.print_exc()
        raise e


# --- END MODIFIED ---


def initialize_trainer(
    envs: List[GameState],
    agent: DQNAgent,
    buffer: ReplayBufferBase,
    stats_recorder: StatsRecorderBase,  # This is now the TB recorder instance
    env_config: EnvConfig,
    dqn_config: DQNConfig,
    train_config: TrainConfig,
    buffer_config: BufferConfig,
    model_config: ModelConfig,
) -> Trainer:
    print("Initializing Trainer...")
    trainer = Trainer(
        envs=envs,
        agent=agent,
        buffer=buffer,
        stats_recorder=stats_recorder,
        env_config=env_config,
        dqn_config=dqn_config,
        train_config=train_config,
        buffer_config=buffer_config,
        model_config=model_config,
        model_save_path=MODEL_SAVE_PATH,
        buffer_save_path=BUFFER_SAVE_PATH,
        load_checkpoint_path=train_config.LOAD_CHECKPOINT_PATH,
        load_buffer_path=train_config.LOAD_BUFFER_PATH,
    )
    print("Trainer initialization finished.")
    return trainer
