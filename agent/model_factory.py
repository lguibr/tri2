import torch.nn as nn
from config import (
    ModelConfig,
    EnvConfig,
) 
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
        return TransformerNet(state_dim, action_dim, config.Transformer, dueling)
    elif model_type == "conv2d":
        # Conv2D needs Height/Width which might depend on EnvConfig, ensure config has it
        if not hasattr(config.Conv2D, "HEIGHT") or not hasattr(config.Conv2D, "WIDTH"):
            print(
                "Warning: Conv2D config missing HEIGHT/WIDTH. Attempting to use EnvConfig."
            )
            # This is slightly hacky, better to ensure H/W are in ModelConfig.Conv2D directly
            config.Conv2D.HEIGHT = getattr(config.Conv2D, "HEIGHT", EnvConfig.ROWS)
            config.Conv2D.WIDTH = getattr(config.Conv2D, "WIDTH", EnvConfig.COLS)
        return Conv2DNet(state_dim, action_dim, config.Conv2D, dueling)
    elif model_type == "lstm":
        return LSTMNet(state_dim, action_dim, config.LSTM, dueling)
    elif model_type == "mixed":
        if not hasattr(config.Mixed, "HEIGHT") or not hasattr(config.Mixed, "WIDTH"):
            print(
                "Warning: Mixed config missing HEIGHT/WIDTH. Attempting to use EnvConfig."
            )
            config.Mixed.HEIGHT = getattr(config.Mixed, "HEIGHT", EnvConfig.ROWS)
            config.Mixed.WIDTH = getattr(config.Mixed, "WIDTH", EnvConfig.COLS)
        return MixedNet(state_dim, action_dim, config.Mixed, dueling)
    else:
        raise ValueError(f"Unknown MODEL_TYPE={model_type}")
