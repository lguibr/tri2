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
    from environment.game_state import GameState
except ImportError as e:
    print(f"Error importing environment: {e}")
    sys.exit(1)
from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from agent.replay_buffer.buffer_utils import create_replay_buffer
from training.trainer import Trainer
from stats.stats_recorder import StatsRecorderBase
from stats.tensorboard_logger import TensorBoardStatsRecorder
from utils.helpers import ensure_numpy


def initialize_envs(
    num_envs: int, env_config: EnvConfig
) -> List[GameState]:  # Unchanged
    print(f"Initializing {num_envs} game environments...")
    try:
        envs = [GameState() for _ in range(num_envs)]
        s_test = envs[0].reset()
        s_np = ensure_numpy(s_test)
        if s_np.shape[0] != env_config.STATE_DIM:
            raise ValueError(
                f"FATAL: State dim mismatch! Env:{s_np.shape[0]}, Cfg:{env_config.STATE_DIM}"
            )
        _ = envs[0].valid_actions()
        _, _ = envs[0].step(0)
        print(f"Successfully initialized {num_envs} environments.")
        return envs
    except Exception as e:
        print(f"FATAL ERROR during env init: {e}")
        traceback.print_exc()
        raise e


def initialize_agent_buffer(  # Unchanged
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


def initialize_stats_recorder(
    stats_config: StatsConfig,
    tb_config: TensorBoardConfig,
    config_dict: Dict[str, Any],
    agent: Optional[DQNAgent],
    env_config: EnvConfig,
    # --- MODIFIED: Removed notification_callback parameter ---
    # notification_callback: Optional[Callable[[str], None]] = None,
) -> StatsRecorderBase:
    """Creates the TensorBoard recorder, logs graph (on CPU) and hparams."""
    print(f"Initializing Stats Recorder (TensorBoard)...")
    avg_window = stats_config.STATS_AVG_WINDOW
    console_log_freq = stats_config.CONSOLE_LOG_FREQ

    dummy_input_cpu = None
    model_for_graph_cpu = None
    if agent and agent.online_net:
        try:
            dummy_state = np.zeros((1, env_config.STATE_DIM), dtype=np.float32)
            dummy_input_cpu = torch.tensor(dummy_state, device="cpu")

            if not hasattr(agent, "dqn_config"):
                raise AttributeError(
                    "DQNAgent instance is missing 'dqn_config' attribute needed for graph logging."
                )

            model_for_graph_cpu = type(agent.online_net)(
                state_dim=env_config.STATE_DIM,
                action_dim=env_config.ACTION_DIM,
                config=agent.online_net.config,
                env_config=agent.online_net.env_config,
                dqn_config=agent.dqn_config,
                dueling=agent.online_net.dueling,
                use_noisy=agent.online_net.use_noisy,
            ).to("cpu")

            model_for_graph_cpu.load_state_dict(agent.online_net.state_dict())
            model_for_graph_cpu.eval()
            print(
                "[Stats Init] Prepared model copy and dummy input on CPU for graph logging."
            )
        except AttributeError as ae:
            print(
                f"Warning: Attribute error preparing model/input for graph logging: {ae}. Check AgentNetwork init."
            )
            dummy_input_cpu = None
            model_for_graph_cpu = None
        except Exception as e:
            print(f"Warning: Failed to prepare model/input for graph logging: {e}")
            traceback.print_exc()
            dummy_input_cpu = None
            model_for_graph_cpu = None

    print(f"Using TensorBoard Logger (Log Dir: {tb_config.LOG_DIR})")
    try:
        # --- MODIFIED: Removed notification_callback argument from the call ---
        stats_recorder = TensorBoardStatsRecorder(
            log_dir=tb_config.LOG_DIR,
            hparam_dict=config_dict,
            model_for_graph=model_for_graph_cpu,
            dummy_input_for_graph=dummy_input_cpu,
            console_log_interval=console_log_freq,
            avg_window=avg_window,
            histogram_log_interval=tb_config.HISTOGRAM_LOG_FREQ,
            image_log_interval=tb_config.IMAGE_LOG_FREQ if tb_config.LOG_IMAGES else -1,
            # notification_callback=notification_callback, # <<< REMOVED THIS LINE
        )
        # --- END MODIFIED ---
        print("Stats Recorder initialized.")
        return stats_recorder
    except Exception as e:
        print(f"FATAL: Error initializing TensorBoardStatsRecorder: {e}. Exiting.")
        traceback.print_exc()
        raise e


def initialize_trainer(
    envs: List[GameState],
    agent: DQNAgent,
    buffer: ReplayBufferBase,
    stats_recorder: StatsRecorderBase,
    env_config: EnvConfig,
    dqn_config: DQNConfig,
    train_config: TrainConfig,
    buffer_config: BufferConfig,
    model_config: ModelConfig,
    # --- MODIFIED: Removed notification_callback parameter ---
    # notification_callback: Optional[Callable[[str], None]] = None,
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
        # Trainer itself doesn't directly use the callback
        # notification_callback=notification_callback # <<< REMOVED THIS LINE (already commented, but confirming)
    )
    print("Trainer initialization finished.")
    return trainer
