# File: agent/model_factory.py
import torch.nn as nn
from config import ModelConfig, EnvConfig, DQNConfig
from typing import Type

from agent.networks.agent_network import AgentNetwork


def create_network(
    # --- MODIFIED: Pass EnvConfig instance ---
    env_config: EnvConfig,
    # --- END MODIFIED ---
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
        # --- MODIFIED: Pass EnvConfig instance ---
        env_config=env_config,
        # --- END MODIFIED ---
        action_dim=action_dim,
        model_config=model_config.Network,  # Pass the Network sub-config
        dqn_config=dqn_config,
        dueling=dqn_config.USE_DUELING,
        use_noisy=dqn_config.USE_NOISY_NETS,
    )
