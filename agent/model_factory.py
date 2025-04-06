import torch
import torch.nn as nn

from config import ModelConfig, EnvConfig, RNNConfig, TransformerConfig
from agent.networks.agent_network import ActorCriticNetwork


def create_network(
    env_config: EnvConfig,
    action_dim: int,
    model_config: ModelConfig,
    rnn_config: RNNConfig,
    transformer_config: TransformerConfig,
    device: torch.device,
) -> nn.Module:
    """Creates the ActorCriticNetwork based on configuration."""
    print(
        f"[ModelFactory] Creating ActorCriticNetwork (RNN: {rnn_config.USE_RNN}, Transformer: {transformer_config.USE_TRANSFORMER})"
    )
    return ActorCriticNetwork(
        env_config=env_config,
        action_dim=action_dim,
        model_config=model_config.Network,
        rnn_config=rnn_config,
        transformer_config=transformer_config,
        device=device,
    )
