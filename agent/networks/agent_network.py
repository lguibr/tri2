# File: agent/networks/agent_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# --- Import specific configs needed ---
from config import ModelConfig, EnvConfig, DQNConfig, DEVICE

# ---
from typing import Tuple, List, Type

from .noisy_layer import NoisyLinear


class AgentNetwork(nn.Module):
    """
    Agent Network: CNN+MLP -> Fusion -> Dueling Heads -> Optional Distributional Output.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: ModelConfig.Network,
        env_config: EnvConfig,
        dqn_config: DQNConfig,  # Pass full DQNConfig
        dueling: bool,
        use_noisy: bool,
    ):
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        self.env_config = env_config
        self.config = config
        self.use_noisy = use_noisy
        # --- Store Distributional Config ---
        self.use_distributional = dqn_config.USE_DISTRIBUTIONAL
        self.num_atoms = dqn_config.NUM_ATOMS
        # ---
        self.device = DEVICE
        print(f"[AgentNetwork] Target device set to: {self.device}")
        print(
            f"[AgentNetwork] Distributional C51: {self.use_distributional} ({self.num_atoms} atoms)"
        )

        self._calculate_dims()
        if state_dim != self.expected_total_dim:
            raise ValueError(
                f"AgentNetwork init: State dim mismatch! Input:{state_dim} != Expected:{self.expected_total_dim}"
            )

        print(f"[AgentNetwork] Initializing (Noisy Heads: {self.use_noisy}):")
        print(
            f"  Input Dim: {state_dim} (Grid: {self.expected_grid_flat_dim}, Shape: {self.expected_shape_flat_dim})"
        )

        # Build network branches
        self.conv_base, conv_out_h, conv_out_w, conv_out_c = self._build_cnn_branch()
        self.conv_out_size = self._get_conv_out_size(
            (self.grid_feat_per_cell, self.grid_h, self.grid_w)
        )
        print(
            f"  CNN Output Dim (HxWxC): ({conv_out_h}x{conv_out_w}x{conv_out_c}) -> Flat: {self.conv_out_size}"
        )

        self.shape_mlp, self.shape_mlp_out_dim = self._build_shape_mlp_branch()
        print(f"  Shape MLP Output Dim: {self.shape_mlp_out_dim}")

        combined_features_dim = self.conv_out_size + self.shape_mlp_out_dim
        print(f"  Combined Features Dim: {combined_features_dim}")

        self.fusion_mlp, self.head_input_dim = self._build_fusion_mlp_branch(
            combined_features_dim
        )
        print(f"  Fusion MLP Output Dim (Input to Heads): {self.head_input_dim}")

        # --- Modified Head Building ---
        self._build_output_heads()
        head_type = NoisyLinear if self.use_noisy else nn.Linear
        output_type = "Distributional" if self.use_distributional else "Q-Value"
        print(
            f"  Using {'Dueling' if self.dueling else 'Single'} Heads ({head_type.__name__}), Output: {output_type}"
        )
        # ---
        print(
            f"[AgentNetwork] Final check - conv_base device: {next(self.conv_base.parameters()).device}"
        )

    def _calculate_dims(self):
        self.grid_h = self.env_config.ROWS
        self.grid_w = self.env_config.COLS
        self.grid_feat_per_cell = self.env_config.GRID_FEATURES_PER_CELL
        self.expected_grid_flat_dim = (
            self.grid_h * self.grid_w * self.grid_feat_per_cell
        )
        self.num_shape_slots = self.env_config.NUM_SHAPE_SLOTS
        self.shape_feat_per_shape = self.env_config.SHAPE_FEATURES_PER_SHAPE
        self.expected_shape_flat_dim = self.num_shape_slots * self.shape_feat_per_shape
        self.expected_total_dim = (
            self.expected_grid_flat_dim + self.expected_shape_flat_dim
        )

    def _build_cnn_branch(self) -> Tuple[nn.Sequential, int, int, int]:
        conv_layers: List[nn.Module] = []
        current_channels = self.grid_feat_per_cell
        h, w = self.grid_h, self.grid_w
        cfg = self.config
        print(f"  Building CNN on device: {self.device}")
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
            pool_layer = nn.MaxPool2d(
                kernel_size=cfg.POOL_KERNEL_SIZE, stride=cfg.POOL_STRIDE
            ).to(self.device)
            conv_layers.append(pool_layer)
            current_channels = out_channels
            h = (h + 2 * cfg.CONV_PADDING - cfg.CONV_KERNEL_SIZE) // cfg.CONV_STRIDE + 1
            w = (w + 2 * cfg.CONV_PADDING - cfg.CONV_KERNEL_SIZE) // cfg.CONV_STRIDE + 1
            h = (h - cfg.POOL_KERNEL_SIZE) // cfg.POOL_STRIDE + 1
            w = (w - cfg.POOL_KERNEL_SIZE) // cfg.POOL_STRIDE + 1
        cnn_module = nn.Sequential(*conv_layers)
        if len(cnn_module) > 0 and hasattr(cnn_module[0], "weight"):
            print(f"    CNN Layer 0 device after build: {cnn_module[0].weight.device}")
        return cnn_module, h, w, current_channels

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape, device=self.device)
            self.conv_base.eval()
            output = self.conv_base(dummy_input)
            # self.conv_base.train() # Not needed, forward handles mode
            return int(np.prod(output.size()[1:]))

    def _build_shape_mlp_branch(self) -> Tuple[nn.Sequential, int]:
        shape_mlp_layers: List[nn.Module] = []
        lin1 = nn.Linear(
            self.expected_shape_flat_dim, self.config.SHAPE_MLP_HIDDEN_DIM
        ).to(self.device)
        shape_mlp_layers.append(lin1)
        shape_mlp_layers.append(self.config.SHAPE_MLP_ACTIVATION())
        return nn.Sequential(*shape_mlp_layers), self.config.SHAPE_MLP_HIDDEN_DIM

    def _build_fusion_mlp_branch(self, input_dim: int) -> Tuple[nn.Sequential, int]:
        fusion_layers: List[nn.Module] = []
        current_fusion_dim = input_dim
        cfg = self.config
        fusion_linear_layer_class = nn.Linear
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
        """Builds the final heads, adjusting output size for distributional RL."""
        head_linear_layer_class = NoisyLinear if self.use_noisy else nn.Linear
        # --- Determine output units based on distributional flag ---
        output_units_per_stream = (
            self.action_dim * self.num_atoms
            if self.use_distributional
            else self.action_dim
        )
        value_units = self.num_atoms if self.use_distributional else 1
        # ---

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns Q-values or distributions (logits)."""
        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)
        if x.dim() != 2 or x.size(1) != self.expected_total_dim:
            raise ValueError(
                f"AgentNetwork forward: Invalid input shape {x.shape}. Expected [B, {self.expected_total_dim}]."
            )
        batch_size = x.size(0)

        # Split and process features
        grid_features_flat = x[:, : self.expected_grid_flat_dim]
        shape_features_flat = x[:, self.expected_grid_flat_dim :]
        grid_features_reshaped = grid_features_flat.view(
            batch_size, self.grid_feat_per_cell, self.grid_h, self.grid_w
        )
        conv_output = self.conv_base(grid_features_reshaped)
        conv_output_flat = conv_output.view(batch_size, -1)
        shape_output = self.shape_mlp(shape_features_flat)
        combined_features = torch.cat((conv_output_flat, shape_output), dim=1)
        fused_output = self.fusion_mlp(combined_features)

        # Output Heads
        if self.dueling:
            value = self.value_head(fused_output)  # Shape: [B, 1] or [B, N_ATOMS]
            advantage = self.advantage_head(
                fused_output
            )  # Shape: [B, A] or [B, A*N_ATOMS]

            if self.use_distributional:
                # Reshape for distributional: value[B, N], advantage[B, A, N]
                value = value.view(batch_size, 1, self.num_atoms)  # Add action dim
                advantage = advantage.view(batch_size, self.action_dim, self.num_atoms)
                # Combine using mean subtraction only on advantage stream
                adv_mean = advantage.mean(
                    dim=1, keepdim=True
                )  # Mean over actions [B, 1, N]
                dist_logits = value + (advantage - adv_mean)  # Shape: [B, A, N]
                # Apply Softmax over atoms dimension later (in agent or loss)
            else:
                # Standard Dueling Q-value combination
                adv_mean = advantage.mean(
                    dim=1, keepdim=True
                )  # Mean over actions [B, 1]
                dist_logits = value + (advantage - adv_mean)  # Shape: [B, A] (Q-values)

        else:  # Non-Dueling
            dist_logits = self.output_head(
                fused_output
            )  # Shape: [B, A] or [B, A*N_ATOMS]
            if self.use_distributional:
                # Reshape for distributional: [B, A, N]
                dist_logits = dist_logits.view(
                    batch_size, self.action_dim, self.num_atoms
                )

        # --- Return logits; softmax is applied in agent/loss function ---
        return dist_logits

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers within the network."""
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
