# File: agent/networks/agent_network.py
# (No structural changes, cleanup comments)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from config import ModelConfig, EnvConfig
from typing import Tuple, List, Type

from .noisy_layer import NoisyLinear


class AgentNetwork(nn.Module):
    """
    Agent Network: CNN (Grid) + MLP (Shape) -> Fused MLP -> Dueling Heads (Noisy optional).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: ModelConfig.Network,  # The specific network sub-config
        env_config: EnvConfig,
        dueling: bool,
        use_noisy: bool,  # Use NoisyLinear in final heads?
    ):
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        self.env_config = env_config
        self.config = config
        self.use_noisy = use_noisy

        # Calculate expected feature dimensions from config
        self.grid_h = env_config.ROWS
        self.grid_w = env_config.COLS
        self.grid_feat_per_cell = env_config.GRID_FEATURES_PER_CELL
        self.expected_grid_flat_dim = (
            self.grid_h * self.grid_w * self.grid_feat_per_cell
        )

        self.num_shape_slots = env_config.NUM_SHAPE_SLOTS
        self.shape_feat_per_shape = env_config.SHAPE_FEATURES_PER_SHAPE
        self.expected_shape_flat_dim = self.num_shape_slots * self.shape_feat_per_shape

        self.expected_total_dim = (
            self.expected_grid_flat_dim + self.expected_shape_flat_dim
        )
        if state_dim != self.expected_total_dim:
            raise ValueError(
                f"AgentNetwork init: State dimension mismatch! "
                f"Input state_dim ({state_dim}) != calculated expected_total_dim ({self.expected_total_dim}). "
                f"Grid={self.expected_grid_flat_dim}, Shape={self.expected_shape_flat_dim}"
            )

        print(f"[AgentNetwork] Initializing (Noisy Heads: {self.use_noisy}):")
        print(
            f"  Input Dim: {state_dim} (Grid: {self.expected_grid_flat_dim}, Shape: {self.expected_shape_flat_dim})"
        )

        # --- 1. CNN Branch (Grid Features) ---
        conv_layers: List[nn.Module] = []
        current_channels = self.grid_feat_per_cell
        h, w = self.grid_h, self.grid_w
        for i, out_channels in enumerate(config.CONV_CHANNELS):
            conv_layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=config.CONV_KERNEL_SIZE,
                    stride=config.CONV_STRIDE,
                    padding=config.CONV_PADDING,
                    bias=not config.USE_BATCHNORM_CONV,
                )
            )
            if config.USE_BATCHNORM_CONV:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(config.CONV_ACTIVATION())
            conv_layers.append(
                nn.MaxPool2d(
                    kernel_size=config.POOL_KERNEL_SIZE, stride=config.POOL_STRIDE
                )
            )
            current_channels = out_channels
            # Calculate output size after conv and pool (for info print)
            h = (
                h + 2 * config.CONV_PADDING - config.CONV_KERNEL_SIZE
            ) // config.CONV_STRIDE + 1
            w = (
                w + 2 * config.CONV_PADDING - config.CONV_KERNEL_SIZE
            ) // config.CONV_STRIDE + 1
            h = (h - config.POOL_KERNEL_SIZE) // config.POOL_STRIDE + 1
            w = (w - config.POOL_KERNEL_SIZE) // config.POOL_STRIDE + 1

        self.conv_base = nn.Sequential(*conv_layers)
        self.conv_out_size = self._get_conv_out_size(
            (self.grid_feat_per_cell, self.grid_h, self.grid_w)
        )
        print(
            f"  CNN Output Dim (HxWxC): ({h}x{w}x{current_channels}) -> Flattened: {self.conv_out_size}"
        )

        # --- 2. Shape Feature Branch (MLP) ---
        shape_mlp_layers: List[nn.Module] = []
        shape_mlp_layers.append(
            nn.Linear(self.expected_shape_flat_dim, config.SHAPE_MLP_HIDDEN_DIM)
        )
        shape_mlp_layers.append(config.SHAPE_MLP_ACTIVATION())
        self.shape_mlp = nn.Sequential(*shape_mlp_layers)
        shape_mlp_out_dim = config.SHAPE_MLP_HIDDEN_DIM
        print(f"  Shape MLP Output Dim: {shape_mlp_out_dim}")

        # --- 3. Combined Feature Fusion (MLP) ---
        combined_features_dim = self.conv_out_size + shape_mlp_out_dim
        print(
            f"  Combined Features Dim (CNN_flat + Shape_MLP): {combined_features_dim}"
        )

        fusion_layers: List[nn.Module] = []
        current_fusion_dim = combined_features_dim
        fusion_linear_layer_class = nn.Linear  # Use standard Linear for fusion part
        for i, hidden_dim in enumerate(config.COMBINED_FC_DIMS):
            fusion_layers.append(
                fusion_linear_layer_class(current_fusion_dim, hidden_dim)
            )
            if config.USE_BATCHNORM_FC:
                fusion_layers.append(nn.BatchNorm1d(hidden_dim))
            fusion_layers.append(config.COMBINED_ACTIVATION())
            if config.DROPOUT_FC > 0:
                fusion_layers.append(nn.Dropout(config.DROPOUT_FC))
            current_fusion_dim = hidden_dim

        self.fusion_mlp = nn.Sequential(*fusion_layers)
        head_input_dim = current_fusion_dim
        print(f"  Fusion MLP Output Dim (Input to Heads): {head_input_dim}")

        # --- 4. Final Output Head(s) ---
        head_linear_layer_class = NoisyLinear if self.use_noisy else nn.Linear

        if self.dueling:
            self.value_head = nn.Sequential(head_linear_layer_class(head_input_dim, 1))
            self.advantage_head = nn.Sequential(
                head_linear_layer_class(head_input_dim, action_dim)
            )
            print(f"  Using Dueling Heads ({head_linear_layer_class.__name__})")
        else:
            self.output_head = nn.Sequential(
                head_linear_layer_class(head_input_dim, action_dim)
            )
            print(f"  Using Single Output Head ({head_linear_layer_class.__name__})")

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        """Helper to calculate the flattened output size of the conv base."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)  # Batch size 1
            output = self.conv_base(dummy_input)
            return int(np.prod(output.size()[1:]))  # Flattened size (C*H*W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.expected_total_dim:
            raise ValueError(
                f"AgentNetwork forward: Invalid input shape {x.shape}. Expected [B, {self.expected_total_dim}]."
            )
        batch_size = x.size(0)

        # Split input features
        grid_features_flat = x[:, : self.expected_grid_flat_dim]
        shape_features_flat = x[:, self.expected_grid_flat_dim :]

        # Process Grid Features (CNN)
        grid_features_reshaped = grid_features_flat.view(
            batch_size, self.grid_feat_per_cell, self.grid_h, self.grid_w
        )
        conv_output = self.conv_base(grid_features_reshaped)
        conv_output_flat = conv_output.view(batch_size, -1)

        # Process Shape Features (MLP)
        shape_output = self.shape_mlp(shape_features_flat)

        # Feature Fusion
        combined_features = torch.cat((conv_output_flat, shape_output), dim=1)
        fused_output = self.fusion_mlp(combined_features)

        # Output Heads
        if self.dueling:
            value = self.value_head(fused_output)
            advantage = self.advantage_head(fused_output)
            q_values = value + (
                advantage - advantage.mean(dim=1, keepdim=True)
            )  # Combine V + (A - mean(A))
        else:
            q_values = self.output_head(fused_output)

        return q_values

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers within the network."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
