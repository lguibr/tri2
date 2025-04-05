# File: agent/model_factory.py
import torch.nn as nn
from config import ModelConfig, EnvConfig, PPOConfig, RNNConfig
from typing import Type

from agent.networks.agent_network import ActorCriticNetwork


def create_network(
    env_config: EnvConfig,
    action_dim: int,
    model_config: ModelConfig,
    rnn_config: RNNConfig,
) -> nn.Module:
    """Creates the ActorCriticNetwork based on configuration."""
    print(f"[ModelFactory] Creating ActorCriticNetwork (RNN: {rnn_config.USE_RNN})")
    return ActorCriticNetwork(
        env_config=env_config,
        action_dim=action_dim,
        model_config=model_config.Network,
        rnn_config=rnn_config,
    )
