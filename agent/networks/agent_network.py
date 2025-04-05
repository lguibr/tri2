# File: agent/networks/agent_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from config import ModelConfig, EnvConfig, DQNConfig, DEVICE
from typing import Tuple, List, Type, Optional

from .noisy_layer import NoisyLinear


class AgentNetwork(nn.Module):
    """
    Agent Network: CNN+MLP -> Fusion -> Dueling Heads -> Optional Distributional Output.
    Accepts separate grid and shape tensors as input.
    Uses NoisyLinear layers in output heads if configured.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        action_dim: int,
        model_config: ModelConfig.Network,
        dqn_config: DQNConfig,
        dueling: bool,
        use_noisy: bool,  # This flag controls NoisyLinear usage
    ):
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        self.env_config = env_config
        self.config = model_config
        self.use_noisy = use_noisy  # Store the flag
        self.use_distributional = dqn_config.USE_DISTRIBUTIONAL
        self.num_atoms = dqn_config.NUM_ATOMS
        self.device = DEVICE

        print(f"[AgentNetwork] Target device set to: {self.device}")
        print(
            f"[AgentNetwork] Distributional C51: {self.use_distributional} ({self.num_atoms} atoms)"
        )

        self.grid_c, self.grid_h, self.grid_w = self.env_config.GRID_STATE_SHAPE
        self.shape_feat_dim = self.env_config.SHAPE_STATE_DIM
        self.num_shape_slots = self.env_config.NUM_SHAPE_SLOTS
        self.shape_feat_per_slot = self.env_config.SHAPE_FEATURES_PER_SHAPE

        print(f"[AgentNetwork] Initializing (Noisy Heads: {self.use_noisy}):")
        print(f"  Input Grid Shape: [B, {self.grid_c}, {self.grid_h}, {self.grid_w}]")
        print(
            f"  Input Shape Features Dim: {self.shape_feat_dim} ({self.num_shape_slots} slots x {self.shape_feat_per_slot} features)"
        )

        self.conv_base, conv_out_h, conv_out_w, conv_out_c = self._build_cnn_branch()
        self.conv_out_size = self._get_conv_out_size(
            (self.grid_c, self.grid_h, self.grid_w)
        )
        print(
            f"  CNN Output Dim (HxWxC): ({conv_out_h}x{conv_out_w}x{conv_out_c}) -> Flat: {self.conv_out_size}"
        )

        self.shape_mlp, self.shape_mlp_out_dim = self._build_shape_mlp_branch()
        print(f"  Shape MLP Output Dim: {self.shape_mlp_out_dim}")

        combined_features_dim = self.conv_out_size + self.shape_mlp_out_dim
        print(f"  Combined Features Dim: {combined_features_dim}")

        # --- Modify Fusion MLP if needed (keeping it simple for now) ---
        # If Noisy Nets are used, potentially the last fusion layer could also be noisy.
        # Keeping fusion MLP with standard Linear layers unless specifically requested otherwise.
        self.fusion_mlp, self.head_input_dim = self._build_fusion_mlp_branch(
            combined_features_dim
        )
        print(f"  Fusion MLP Output Dim (Input to Heads): {self.head_input_dim}")

        # --- Build heads using NoisyLinear conditionally ---
        self._build_output_heads()
        head_type = NoisyLinear if self.use_noisy else nn.Linear  # Determine head type
        output_type = "Distributional" if self.use_distributional else "Q-Value"
        print(
            f"  Using {'Dueling' if self.dueling else 'Single'} Heads ({head_type.__name__}), Output: {output_type} [{self.action_dim * (self.num_atoms if self.use_distributional else 1)} units]"
        )

        if hasattr(self.conv_base, "0") and hasattr(self.conv_base[0], "weight"):
            print(
                f"[AgentNetwork] Final check - conv_base device: {next(self.conv_base.parameters()).device}"
            )
        else:
            print(
                "[AgentNetwork] Final check - conv_base seems empty or has no weights."
            )

    def _build_cnn_branch(self) -> Tuple[nn.Sequential, int, int, int]:
        conv_layers: List[nn.Module] = []
        current_channels = self.grid_c
        h, w = self.grid_h, self.grid_w
        cfg = self.config
        print(
            f"  Building CNN (Input Channels: {current_channels}) on device: {self.device}"
        )
        for i, out_channels in enumerate(cfg.CONV_CHANNELS):
            conv_layer = nn.Conv2d(
                current_channels,
                out_channels,
                kernel_size=cfg.CONV_KERNEL_SIZE,
                stride=cfg.CONV_STRIDE,
                padding=cfg.CONV_PADDING,
                bias=not cfg.USE_BATCHNORM_CONV,
            ).to(self.device)
            conv_layers.append(conv_layer)
            if cfg.USE_BATCHNORM_CONV:
                conv_layers.append(nn.BatchNorm2d(out_channels).to(self.device))
            conv_layers.append(cfg.CONV_ACTIVATION())
            current_channels = out_channels
            h = (h + 2 * cfg.CONV_PADDING - cfg.CONV_KERNEL_SIZE) // cfg.CONV_STRIDE + 1
            w = (w + 2 * cfg.CONV_PADDING - cfg.CONV_KERNEL_SIZE) // cfg.CONV_STRIDE + 1

        cnn_module = nn.Sequential(*conv_layers)
        if len(cnn_module) > 0 and hasattr(cnn_module[0], "weight"):
            print(f"    CNN Layer 0 device after build: {cnn_module[0].weight.device}")
        return cnn_module, h, w, current_channels

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        self.conv_base.to(self.device)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape, device=self.device)
            self.conv_base.eval()
            try:
                output = self.conv_base(dummy_input)
                out_size = int(np.prod(output.size()[1:]))
            except Exception as e:
                print(f"Error calculating conv output size: {e}")
                print(f"Input shape to CNN: {dummy_input.shape}")
                out_size = 1
            return out_size

    def _build_shape_mlp_branch(self) -> Tuple[nn.Sequential, int]:
        shape_mlp_layers: List[nn.Module] = []
        current_dim = self.env_config.SHAPE_STATE_DIM
        cfg = self.config
        for hidden_dim in cfg.SHAPE_FEATURE_MLP_DIMS:
            lin_layer = nn.Linear(current_dim, hidden_dim).to(self.device)
            shape_mlp_layers.append(lin_layer)
            shape_mlp_layers.append(cfg.SHAPE_MLP_ACTIVATION())
            current_dim = hidden_dim
        return nn.Sequential(*shape_mlp_layers), current_dim

    def _build_fusion_mlp_branch(self, input_dim: int) -> Tuple[nn.Sequential, int]:
        fusion_layers: List[nn.Module] = []
        current_fusion_dim = input_dim
        cfg = self.config
        fusion_linear_layer_class = nn.Linear  # Keep fusion linear for now
        for i, hidden_dim in enumerate(cfg.COMBINED_FC_DIMS):
            linear_layer = fusion_linear_layer_class(
                current_fusion_dim, hidden_dim, bias=not cfg.USE_BATCHNORM_FC
            ).to(self.device)
            fusion_layers.append(linear_layer)
            if cfg.USE_BATCHNORM_FC:
                fusion_layers.append(nn.BatchNorm1d(hidden_dim).to(self.device))
            fusion_layers.append(cfg.COMBINED_ACTIVATION())
            if cfg.DROPOUT_FC > 0:
                fusion_layers.append(nn.Dropout(cfg.DROPOUT_FC).to(self.device))
            current_fusion_dim = hidden_dim
        return nn.Sequential(*fusion_layers), current_fusion_dim

    def _build_output_heads(self):
        # Use NoisyLinear if self.use_noisy is True
        head_linear_layer_class = NoisyLinear if self.use_noisy else nn.Linear
        output_units_per_stream = (
            self.action_dim * self.num_atoms
            if self.use_distributional
            else self.action_dim
        )
        value_units = self.num_atoms if self.use_distributional else 1

        if self.dueling:
            self.value_head = head_linear_layer_class(
                self.head_input_dim, value_units
            ).to(self.device)
            self.advantage_head = head_linear_layer_class(
                self.head_input_dim, output_units_per_stream
            ).to(self.device)
        else:
            self.output_head = head_linear_layer_class(
                self.head_input_dim, output_units_per_stream
            ).to(self.device)

    def forward(
        self, grid_tensor: torch.Tensor, shape_tensor: torch.Tensor
    ) -> torch.Tensor:
        model_device = next(self.parameters()).device
        if grid_tensor.device != model_device:
            grid_tensor = grid_tensor.to(model_device)
        if shape_tensor.device != model_device:
            shape_tensor = shape_tensor.to(model_device)

        expected_grid_shape = (self.grid_c, self.grid_h, self.grid_w)
        if grid_tensor.ndim != 4 or grid_tensor.shape[1:] != expected_grid_shape:
            raise ValueError(
                f"AgentNetwork forward: Invalid grid_tensor shape {grid_tensor.shape}. Expected [B, {self.grid_c}, {self.grid_h}, {self.grid_w}]."
            )

        batch_size = grid_tensor.size(0)
        expected_shape_flat_dim = self.num_shape_slots * self.shape_feat_per_slot
        if shape_tensor.ndim == 3 and shape_tensor.shape[1:] == (
            self.num_shape_slots,
            self.shape_feat_per_slot,
        ):
            shape_tensor_flat = shape_tensor.view(batch_size, -1)
        elif (
            shape_tensor.ndim == 2 and shape_tensor.shape[1] == expected_shape_flat_dim
        ):
            shape_tensor_flat = shape_tensor
        else:
            raise ValueError(
                f"AgentNetwork forward: Invalid shape_tensor shape {shape_tensor.shape}. Expected [B, {self.num_shape_slots}, {self.shape_feat_per_slot}] or [B, {expected_shape_flat_dim}]."
            )

        conv_output = self.conv_base(grid_tensor)
        conv_output_flat = conv_output.view(batch_size, -1)
        shape_output = self.shape_mlp(shape_tensor_flat)
        combined_features = torch.cat((conv_output_flat, shape_output), dim=1)
        fused_output = self.fusion_mlp(combined_features)

        if self.dueling:
            value = self.value_head(fused_output)
            advantage = self.advantage_head(fused_output)
            if self.use_distributional:
                value = value.view(batch_size, 1, self.num_atoms)
                advantage = advantage.view(batch_size, self.action_dim, self.num_atoms)
                adv_mean = advantage.mean(dim=1, keepdim=True)
                dist_logits = value + (advantage - adv_mean)
            else:
                adv_mean = advantage.mean(dim=1, keepdim=True)
                dist_logits = value + (advantage - adv_mean)
        else:
            dist_logits = self.output_head(fused_output)
            if self.use_distributional:
                dist_logits = dist_logits.view(
                    batch_size, self.action_dim, self.num_atoms
                )

        return dist_logits

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers within the network."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
