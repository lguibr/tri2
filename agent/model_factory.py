# File: agent/model_factory.py
# (Largely unchanged, just cleaner print)
import torch.nn as nn
from config import ModelConfig, EnvConfig, DQNConfig
from typing import Type

from agent.networks.agent_network import AgentNetwork


def create_network(
    state_dim: int,
    action_dim: int,
    model_config: ModelConfig,
    dqn_config: DQNConfig,
) -> nn.Module:
    """Creates the AgentNetwork based on configuration."""

    print(
        f"[ModelFactory] Creating AgentNetwork (Dueling: {dqn_config.USE_DUELING}, NoisyNets Heads: {dqn_config.USE_NOISY_NETS})"
    )

    # Pass the specific sub-config ModelConfig.Network
    return AgentNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        config=model_config.Network,  # Pass the Network sub-config
        env_config=EnvConfig,  # AgentNetwork needs EnvConfig
        dueling=dqn_config.USE_DUELING,
        use_noisy=dqn_config.USE_NOISY_NETS,
        dqn_config=dqn_config,
    )
