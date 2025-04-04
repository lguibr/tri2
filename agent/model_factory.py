# File: agent/model_factory.py
import torch.nn as nn
from config import (
    ModelConfig,
    EnvConfig,
)

# Import all network types
from agent.networks.transformer_net import TransformerNet
from agent.networks.conv2d_net import Conv2DNet
from agent.networks.lstm_net import LSTMNet
from agent.networks.mixed_net import MixedNet


def create_network(state_dim: int, action_dim: int, config: ModelConfig) -> nn.Module:
    """Creates the neural network based on config settings."""
    model_type = config.MODEL_TYPE
    dueling = config.USE_DUELING
    print(f"[ModelFactory] Creating model: {model_type}, Dueling: {dueling}")

    if model_type == "transformer":
        # Ensure Transformer sub-config exists
        if not hasattr(config, "Transformer"):
            raise AttributeError(
                "ModelConfig missing 'Transformer' sub-configuration for MODEL_TYPE='transformer'"
            )
        return TransformerNet(state_dim, action_dim, config.Transformer, dueling)

    elif model_type == "conv2d":
        # Ensure Conv2D sub-config exists
        if not hasattr(config, "Conv2D"):
            raise AttributeError(
                "ModelConfig missing 'Conv2D' sub-configuration for MODEL_TYPE='conv2d'"
            )
        # Ensure H/W are set, potentially sourcing from EnvConfig
        if not hasattr(config.Conv2D, "HEIGHT") or not hasattr(config.Conv2D, "WIDTH"):
            print(
                "Warning: Conv2D config missing HEIGHT/WIDTH. Using EnvConfig values."
            )
            config.Conv2D.HEIGHT = getattr(config.Conv2D, "HEIGHT", EnvConfig.ROWS)
            config.Conv2D.WIDTH = getattr(config.Conv2D, "WIDTH", EnvConfig.COLS)
        return Conv2DNet(state_dim, action_dim, config.Conv2D, dueling)

    elif model_type == "lstm":
        # Ensure LSTM sub-config exists
        if not hasattr(config, "LSTM"):
            raise AttributeError(
                "ModelConfig missing 'LSTM' sub-configuration for MODEL_TYPE='lstm'"
            )
        return LSTMNet(state_dim, action_dim, config.LSTM, dueling)

    elif model_type == "mixed":
        # Ensure Mixed sub-config exists
        if not hasattr(config, "Mixed"):
            raise AttributeError(
                "ModelConfig missing 'Mixed' sub-configuration for MODEL_TYPE='mixed'"
            )
        # Ensure H/W are set, potentially sourcing from EnvConfig
        if not hasattr(config.Mixed, "HEIGHT") or not hasattr(config.Mixed, "WIDTH"):
            print("Warning: Mixed config missing HEIGHT/WIDTH. Using EnvConfig values.")
            config.Mixed.HEIGHT = getattr(config.Mixed, "HEIGHT", EnvConfig.ROWS)
            config.Mixed.WIDTH = getattr(config.Mixed, "WIDTH", EnvConfig.COLS)
        # <<< Pass EnvConfig to MixedNet to help parse state_dim >>>
        return MixedNet(
            state_dim, action_dim, config.Mixed, EnvConfig, dueling
        )  # Pass EnvConfig

    else:
        raise ValueError(f"Unknown MODEL_TYPE specified in config: {model_type}")
