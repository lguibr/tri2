import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import ModelConfig
from typing import Tuple


class Conv2DNet(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, config: ModelConfig.Conv2D, dueling: bool
    ):
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        # <<< Use H/W from config object directly >>>
        self.height = config.HEIGHT
        self.width = config.WIDTH
        self.target_flat_dim = self.height * self.width

        print(
            f"[Conv2DNet] Init: Reshaping flat {state_dim} -> C=1, H={self.height}, W={self.width}"
        )
        if self.target_flat_dim != state_dim:
            print(
                f"Warning: state_dim {state_dim} != H*W {self.target_flat_dim}. Input formatting will pad/truncate."
            )

        channels = [1] + config.CHANNELS
        conv_layers = []
        current_channels = 1
        h, w = self.height, self.width
        for out_channels in config.CHANNELS:
            conv_layers.append(
                nn.Conv2d(
                    current_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = out_channels
            # Update effective H/W after pooling
            h = (
                h + 2 * 0 - 2
            ) // 2 + 1  # Integer division for pooling size calculation
            w = (w + 2 * 0 - 2) // 2 + 1
        self.conv_base = nn.Sequential(*conv_layers)

        conv_out_size = self._get_conv_out_size((1, self.height, self.width))
        print(f"[Conv2DNet] Flattened conv output size: {conv_out_size}")

        fc_hidden_dim = config.FC_DIM
        if self.dueling:
            self.value_head = nn.Sequential(
                nn.Linear(conv_out_size, fc_hidden_dim),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(conv_out_size, fc_hidden_dim),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, action_dim),
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(conv_out_size, fc_hidden_dim),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, action_dim),
            )

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv_base(dummy_input)
            return int(np.prod(output.size()[1:]))

    def _format_input(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        current_dim = x.size(1)
        if current_dim != self.target_flat_dim:
            if current_dim > self.target_flat_dim:
                x = x[:, : self.target_flat_dim]
            else:
                pad = self.target_flat_dim - current_dim
                x = F.pad(x, (0, pad))
        return x.view(bsz, 1, self.height, self.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._format_input(x)
        x = self.conv_base(x)
        x = x.view(x.size(0), -1)

        if self.dueling:
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_head(x)
        return q_values
