# File: init/rl_components_ppo.py
import sys
import traceback
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Callable

from config import (
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TrainConfig,
    ModelConfig,
    StatsConfig,
    RewardConfig,
    TensorBoardConfig,
    DEVICE,
    MODEL_SAVE_PATH,
    get_config_dict,
)

try:
    from environment.game_state import GameState, StateType
except ImportError as e:
    print(f"Error importing environment: {e}")
    sys.exit(1)

from agent.ppo_agent import PPOAgent
from training.trainer import Trainer
from stats.stats_recorder import StatsRecorderBase
from stats.aggregator import StatsAggregator
from stats.simple_stats_recorder import SimpleStatsRecorder
from stats.tensorboard_logger import TensorBoardStatsRecorder


def initialize_envs(num_envs: int, env_config: EnvConfig) -> List[GameState]:
    """Initializes the specified number of game environments."""
    print(f"Initializing {num_envs} game environments...")
    try:
        envs = [GameState() for _ in range(num_envs)]
        # Basic validation on the first environment
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

        # Test step with a valid action if available
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


def initialize_agent(
    model_config: ModelConfig,
    ppo_config: PPOConfig,
    rnn_config: RNNConfig,
    env_config: EnvConfig,
) -> PPOAgent:
    """Initializes the PPO Agent."""
    print("Initializing PPO Agent...")
    agent = PPOAgent(
        model_config=model_config,
        ppo_config=ppo_config,
        rnn_config=rnn_config,
        env_config=env_config,
    )
    print("PPO Agent initialized.")
    return agent


# --- MODIFIED: Added is_reinit flag ---
def initialize_stats_recorder(
    stats_config: StatsConfig,
    tb_config: TensorBoardConfig,
    config_dict: Dict[str, Any],
    agent: Optional[PPOAgent],
    env_config: EnvConfig,
    rnn_config: RNNConfig,
    is_reinit: bool = False,  # Added flag
) -> StatsRecorderBase:
    """Initializes the statistics recording components."""
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

    # --- MODIFIED: Only prepare graph model/input on initial setup ---
    if not is_reinit and agent and agent.network:
        print("[Stats Init] Preparing model copy and dummy input for graph...")
        try:
            # Prepare dummy input on CPU
            expected_grid_shape = env_config.GRID_STATE_SHAPE
            dummy_grid_np = np.zeros(expected_grid_shape, dtype=np.float32)
            dummy_shapes_np = np.zeros(env_config.SHAPE_STATE_DIM, dtype=np.float32)
            # Add batch and sequence dimensions if RNN is used (B=1, T=1)
            batch_dim = 1
            seq_dim = 1 if rnn_config.USE_RNN else 0
            grid_dims = ([batch_dim, seq_dim] if seq_dim else [batch_dim]) + list(
                expected_grid_shape
            )
            shape_dims = ([batch_dim, seq_dim] if seq_dim else [batch_dim]) + [
                env_config.SHAPE_STATE_DIM
            ]

            dummy_grid_cpu = torch.tensor(dummy_grid_np).reshape(grid_dims).to("cpu")
            dummy_shapes_cpu = (
                torch.tensor(dummy_shapes_np).reshape(shape_dims).to("cpu")
            )

            # Create a copy of the network on CPU for graph tracing
            # Ensure the network type is correctly inferred or passed
            model_for_graph_cpu = type(agent.network)(
                env_config=env_config,
                action_dim=env_config.ACTION_DIM,
                model_config=agent.network.config,  # Access config from the agent's network instance
                rnn_config=rnn_config,
            ).to("cpu")

            model_for_graph_cpu.load_state_dict(agent.network.state_dict())
            model_for_graph_cpu.eval()

            # Prepare dummy input tuple (grid, shapes) - hidden state is optional for graph
            dummy_input_tuple = (dummy_grid_cpu, dummy_shapes_cpu)

            print("[Stats Init] Prepared model copy and dummy input on CPU for graph.")
        except Exception as e:
            print(f"Warning: Failed to prepare model/input for graph logging: {e}")
            traceback.print_exc()
            model_for_graph_cpu, dummy_input_tuple = None, None
    elif is_reinit:
        print("[Stats Init] Skipping graph model preparation during re-initialization.")
    # --- END MODIFIED ---

    print(f"Using TensorBoard Logger (Log Dir: {tb_config.LOG_DIR})")
    try:
        # Pass potentially None model/input to constructor
        tb_recorder = TensorBoardStatsRecorder(
            aggregator=stats_aggregator,
            console_recorder=console_recorder,
            log_dir=tb_config.LOG_DIR,
            hparam_dict=config_dict if not is_reinit else None,  # Log hparams only once
            model_for_graph=model_for_graph_cpu,  # Will be None if is_reinit
            dummy_input_for_graph=dummy_input_tuple,  # Will be None if is_reinit
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


# --- END MODIFIED ---


def initialize_trainer(
    envs: List[GameState],
    agent: PPOAgent,
    stats_recorder: StatsRecorderBase,
    env_config: EnvConfig,
    ppo_config: PPOConfig,
    rnn_config: RNNConfig,
    train_config: TrainConfig,
    model_config: ModelConfig,
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
        model_save_path=MODEL_SAVE_PATH,
        load_checkpoint_path=train_config.LOAD_CHECKPOINT_PATH,
    )
    print("PPO Trainer initialization finished.")
    return trainer
