import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import ModelConfig
from typing import Tuple


class MixedNet(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, config: ModelConfig.Mixed, dueling: bool
    ):
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        self.height = config.HEIGHT
        self.width = config.WIDTH
        self.target_flat_dim = self.height * self.width

        print(
            f"[MixedNet] Init: Reshaping flat {state_dim} -> C=1, H={self.height}, W={self.width}"
        )
        if self.target_flat_dim != state_dim:
            print(
                f"Warning: state_dim {state_dim} != H*W {self.target_flat_dim}. Input formatting will pad/truncate."
            )

        # Conv Layers
        channels = [1] + config.CONV_CHANNELS
        conv_layers = []
        current_channels = 1
        h, w = self.height, self.width
        for out_channels in config.CONV_CHANNELS:
            conv_layers.append(
                nn.Conv2d(
                    current_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            )
            if config.USE_BN:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = out_channels
            h = (h + 2 * 0 - 2) // 2 + 1
            w = (w + 2 * 0 - 2) // 2 + 1
        self.conv_base = nn.Sequential(*conv_layers)
        conv_out_size = self._get_conv_out_size((1, self.height, self.width))
        print(f"[MixedNet] Flattened conv output size: {conv_out_size}")

        # LSTM Layer
        self.lstm_hidden_dim = config.LSTM_HIDDEN_DIM
        self.lstm = nn.LSTM(conv_out_size, self.lstm_hidden_dim, batch_first=True)

        # FC Layers
        fc_input_dim = self.lstm_hidden_dim
        if self.dueling:
            self.value_head = self._build_fc_layers(fc_input_dim, 1, config)
            self.advantage_head = self._build_fc_layers(
                fc_input_dim, action_dim, config
            )
        else:
            self.output_head = self._build_fc_layers(fc_input_dim, action_dim, config)

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv_base(dummy_input)
            return int(np.prod(output.size()[1:]))

    def _build_fc_layers(
        self, input_dim: int, output_dim: int, config: ModelConfig.Mixed
    ) -> nn.Sequential:
        layers = []
        current_dim = input_dim
        fc_hidden_dim = config.FC_DIM
        for _ in range(config.NUM_FC_LAYERS - 1):
            layers.append(nn.Linear(current_dim, fc_hidden_dim))
            if config.USE_BN:
                layers.append(nn.BatchNorm1d(fc_hidden_dim))
            layers.append(nn.ReLU())
            if config.DROPOUT > 0:
                layers.append(nn.Dropout(config.DROPOUT))
            current_dim = fc_hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def _format_input(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        current_dim = x.size(1)
        if current_dim != self.target_flat_dim:
            if current_dim > self.target_flat_dim:
                x = x[:, : self.target_flat_dim]
            else:
                x = F.pad(x, (0, self.target_flat_dim - current_dim))
        return x.view(bsz, 1, self.height, self.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self._format_input(x)
        x = self.conv_base(x)
        x = x.view(batch_size, -1)

        x = x.unsqueeze(1)  # Treat as sequence of length 1 for LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_hidden = hn[-1]

        if self.dueling:
            value = self.value_head(lstm_hidden)
            advantage = self.advantage_head(lstm_hidden)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_head(lstm_hidden)
        return q_values
