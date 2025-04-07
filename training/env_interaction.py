# File: training/env_interaction.py
import numpy as np
import torch
import traceback
from typing import List, Tuple, Optional

from environment.game_state import GameState, StateType
from config import EnvConfig
from agent.ppo_agent import PPOAgent  # For type hinting


def update_raw_obs_from_state_dict(
    env_index: int,
    state_dict: StateType,
    raw_obs_grid_cpu: np.ndarray,
    raw_obs_shapes_cpu: np.ndarray,
    raw_obs_availability_cpu: np.ndarray,
    raw_obs_explicit_features_cpu: np.ndarray,
):
    """Updates the CPU RAW observation buffers for a given environment index."""
    raw_obs_grid_cpu[env_index] = state_dict["grid"]
    raw_obs_shapes_cpu[env_index] = state_dict["shapes"]
    raw_obs_availability_cpu[env_index] = state_dict["shape_availability"]
    raw_obs_explicit_features_cpu[env_index] = state_dict["explicit_features"]


def reset_env(
    env: GameState,
    episode_scores: np.ndarray,
    episode_lengths: np.ndarray,
    episode_game_scores: np.ndarray,
    episode_triangles_cleared: np.ndarray,
    env_index: int,
    env_config: EnvConfig,  # Pass env_config for dummy state shape
) -> StateType:
    """Resets a single environment and returns its initial state dict."""
    try:
        state_dict = env.reset()
        episode_scores[env_index] = 0.0
        episode_lengths[env_index] = 0
        episode_game_scores[env_index] = 0
        episode_triangles_cleared[env_index] = 0
        return state_dict
    except Exception as e:
        print(f"ERROR resetting env {env_index}: {e}")
        traceback.print_exc()
        # Return a dummy state matching expected structure
        dummy_state: StateType = {
            "grid": np.zeros(env_config.GRID_STATE_SHAPE, dtype=np.float32),
            "shapes": np.zeros(env_config.SHAPE_STATE_DIM, dtype=np.float32),
            "shape_availability": np.zeros(
                env_config.SHAPE_AVAILABILITY_DIM, dtype=np.float32
            ),
            "explicit_features": np.zeros(
                env_config.EXPLICIT_FEATURES_DIM, dtype=np.float32
            ),
        }
        # Mark as done implicitly by returning dummy state? Or handle done flag separately?
        # Let's assume the caller handles the done flag based on the exception.
        return dummy_state


def reset_all_envs(
    envs: List[GameState],
    num_envs: int,
    raw_obs_grid_cpu: np.ndarray,
    raw_obs_shapes_cpu: np.ndarray,
    raw_obs_availability_cpu: np.ndarray,
    raw_obs_explicit_features_cpu: np.ndarray,
    dones_cpu: np.ndarray,
    episode_scores: np.ndarray,
    episode_lengths: np.ndarray,
    episode_game_scores: np.ndarray,
    episode_triangles_cleared: np.ndarray,
    env_config: EnvConfig,  # Pass env_config
):
    """Resets all environments and updates initial raw observations."""
    print(f"[EnvInteraction] Resetting {num_envs} environments...")
    initial_states = [None] * num_envs
    for i in range(num_envs):
        try:
            initial_states[i] = envs[i].reset()
            episode_scores[i] = 0.0
            episode_lengths[i] = 0
            episode_game_scores[i] = 0
            episode_triangles_cleared[i] = 0
            dones_cpu[i] = False
        except Exception as e:
            print(f"ERROR resetting env {i}: {e}")
            traceback.print_exc()
            initial_states[i] = {
                "grid": np.zeros(env_config.GRID_STATE_SHAPE, dtype=np.float32),
                "shapes": np.zeros(env_config.SHAPE_STATE_DIM, dtype=np.float32),
                "shape_availability": np.zeros(
                    env_config.SHAPE_AVAILABILITY_DIM, dtype=np.float32
                ),
                "explicit_features": np.zeros(
                    env_config.EXPLICIT_FEATURES_DIM, dtype=np.float32
                ),
            }
            dones_cpu[i] = True  # Mark as done if reset failed

    for i in range(num_envs):
        if initial_states[i] is not None:
            update_raw_obs_from_state_dict(
                i,
                initial_states[i],
                raw_obs_grid_cpu,
                raw_obs_shapes_cpu,
                raw_obs_availability_cpu,
                raw_obs_explicit_features_cpu,
            )
    print("[EnvInteraction] Environments reset.")


def reset_rnn_state_for_env(
    env_index: int,
    use_rnn: bool,
    current_lstm_state_device: Optional[Tuple[torch.Tensor, torch.Tensor]],
    agent: PPOAgent,  # Pass agent to get initial state
):
    """Resets the hidden state for a specific environment index if RNN is used."""
    if use_rnn and current_lstm_state_device is not None:
        reset_h, reset_c = agent.get_initial_hidden_state(1)
        if reset_h is not None and reset_c is not None:
            current_lstm_state_device[0][:, env_index : env_index + 1, :] = reset_h
            current_lstm_state_device[1][:, env_index : env_index + 1, :] = reset_c
