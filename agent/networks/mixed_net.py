# File: agent/networks/mixed_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math  # For sqrt in embedding scaling
from config import ModelConfig, EnvConfig
from typing import Tuple, List

# --- Optional: Helper Positional Encoding Class (if not using learned) ---
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5): # max_len=2 for [CLS, Features]
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(1, max_len, d_model) # Batch dim first
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x shape: [Batch, SeqLen, Dim]
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)


class MixedNet(nn.Module):
    """
    Network combining Convolutional (Grid) + MLP (Shape) + Transformer (Combined).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: ModelConfig.Mixed,  # Specific Mixed config
        env_config: EnvConfig,
        dueling: bool,
    ):
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        self.env_config = env_config
        self.config = config  # Store sub-config

        # --- Calculate expected feature dimensions ---
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
                f"MixedNet init: Mismatch! state_dim ({state_dim}) != calculated expected_total_dim ({self.expected_total_dim})."
            )

        print(f"[MixedNet] Initializing:")
        print(
            f"  - Input Dim: {state_dim} (Grid: {self.expected_grid_flat_dim}, Shape: {self.expected_shape_flat_dim})"
        )

        # --- 1. Convolutional Branch (Grid Features) ---
        conv_layers = []
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
            # Update effective H/W estimate after pooling
            h = h // config.POOL_STRIDE
            w = w // config.POOL_STRIDE
        self.conv_base = nn.Sequential(*conv_layers)
        self.conv_out_size = self._get_conv_out_size(
            (self.grid_feat_per_cell, self.grid_h, self.grid_w)
        )
        print(f"  - Conv Branch Output Dim: {self.conv_out_size}")

        # --- 2. Shape Feature Branch (MLP) ---
        shape_mlp_layers = []
        shape_mlp_layers.append(
            nn.Linear(self.expected_shape_flat_dim, config.SHAPE_MLP_HIDDEN_DIM)
        )
        shape_mlp_layers.append(config.SHAPE_MLP_ACTIVATION())
        self.shape_mlp = nn.Sequential(*shape_mlp_layers)
        shape_mlp_out_dim = config.SHAPE_MLP_HIDDEN_DIM
        print(f"  - Shape MLP Output Dim: {shape_mlp_out_dim}")

        # --- 3. Combined Feature Embedding ---
        combined_features_dim = self.conv_out_size + shape_mlp_out_dim
        self.d_model = config.HDIM  # Transformer hidden dimension
        # Embed the concatenated features into the transformer dimension
        self.combined_embedding = nn.Linear(combined_features_dim, self.d_model)
        print(
            f"  - Combined Features Dim: {combined_features_dim} -> Embed Dim (d_model): {self.d_model}"
        )

        # --- 4. Transformer Branch ---
        # Learnable CLS token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)  # Initialize CLS token

        # Positional Encoding/Embedding for the sequence ([CLS], EmbeddedFeatures) - Length 2
        self.pos_dropout = nn.Dropout(p=config.TRANSFORMER_DROPOUT)
        if config.USE_LEARNED_POS_EMBEDDING:
            # Learned embedding for 2 positions (CLS, Features)
            self.pos_embedding = nn.Parameter(torch.zeros(1, 2, self.d_model))
            nn.init.normal_(
                self.pos_embedding, std=0.02
            )  # Initialize learned pos embedding
            print("  - Using Learned Positional Embedding")
        # else:
        #      # Use fixed sinusoidal embedding (optional, requires PositionalEncoding class)
        #      self.pos_embedding = PositionalEncoding(self.d_model, config.TRANSFORMER_DROPOUT, max_len=2)
        #      print("  - Using Sinusoidal Positional Encoding")

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.TRANSFORMER_HEADS,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=config.TRANSFORMER_DROPOUT,
            activation=F.gelu,  # Use GELU activation, common in transformers
            batch_first=True,  # Expect input as [Batch, Seq, Feature]
            norm_first=True,  # Apply LayerNorm before attention/FFN (Pre-LN)
        )
        encoder_norm = nn.LayerNorm(self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.TRANSFORMER_LAYERS,
            norm=encoder_norm,
        )
        print(
            f"  - Transformer Layers: {config.TRANSFORMER_LAYERS}, Heads: {config.TRANSFORMER_HEADS}"
        )

        # --- 5. Final Output Head(s) ---
        # Input dimension is d_model (from the CLS token output)
        fc_input_dim = self.d_model
        print(f"  - Input to Final FC Head: {fc_input_dim}")
        if self.dueling:
            self.value_head = self._build_fc_layers(fc_input_dim, 1, config)
            self.advantage_head = self._build_fc_layers(
                fc_input_dim, action_dim, config
            )
            print(f"  - Using Dueling Heads")
        else:
            self.output_head = self._build_fc_layers(fc_input_dim, action_dim, config)
            print(f"  - Using Single Output Head")

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        """Helper to determine flattened size after conv layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv_base(dummy_input)
            return int(np.prod(output.size()[1:]))

    def _build_fc_layers(
        self, input_dim: int, output_dim: int, config: ModelConfig.Mixed
    ) -> nn.Sequential:
        """Helper to build the final fully connected layers"""
        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in config.COMBINED_FC_DIMS:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if config.USE_BATCHNORM_FC:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(config.COMBINED_ACTIVATION())
            if config.DROPOUT_FC > 0:
                layers.append(nn.Dropout(config.DROPOUT_FC))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes flattened state through Conv, MLP, combines, embeds,
        passes through Transformer, and finally through output heads.
        Input x shape: [batch_size, state_dim]
        """
        if x.dim() != 2 or x.size(1) != self.expected_total_dim:
            raise ValueError(
                f"MixedNet forward: Invalid input shape {x.shape}, expected [B, {self.expected_total_dim}]"
            )

        batch_size = x.size(0)

        # --- Split input ---
        grid_features_flat = x[:, : self.expected_grid_flat_dim]
        shape_features_flat = x[:, self.expected_grid_flat_dim :]

        # --- Branch Processing ---
        grid_features_reshaped = grid_features_flat.view(
            batch_size, self.grid_feat_per_cell, self.grid_h, self.grid_w
        )
        conv_output = self.conv_base(grid_features_reshaped)
        conv_output_flat = conv_output.view(batch_size, -1)
        shape_output = self.shape_mlp(shape_features_flat)

        # --- Combine and Embed ---
        combined_features = torch.cat((conv_output_flat, shape_output), dim=1)
        # Embed into transformer dimension, maybe scale like ViT?
        embedded_features = self.combined_embedding(combined_features) * math.sqrt(
            self.d_model
        )  # Apply embedding

        # --- Prepare Transformer Input ---
        # Prepend CLS token: [B, 1, D] + [B, 1, D] -> [B, 2, D]
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # Expand CLS token to batch size
        transformer_sequence = torch.cat(
            (cls_tokens, embedded_features.unsqueeze(1)), dim=1
        )

        # Add positional encoding/embedding
        if self.config.USE_LEARNED_POS_EMBEDDING:
            transformer_sequence = (
                transformer_sequence + self.pos_embedding
            )  # Add learned pos embedding
        # else:
        #      transformer_sequence = self.pos_embedding(transformer_sequence) # Apply fixed pos encoding module
        transformer_sequence = self.pos_dropout(transformer_sequence)

        # --- Pass through Transformer ---
        # TODO: Add padding mask here if sequence length becomes variable
        transformer_output = self.transformer_encoder(
            transformer_sequence
        )  # Shape: [B, 2, D]

        # --- Extract CLS Token Output ---
        # Use the output corresponding to the CLS token for classification/regression head
        cls_output = transformer_output[:, 0]  # Shape: [B, D]

        # --- Final Head(s) ---
        if self.dueling:
            value = self.value_head(cls_output)
            advantage = self.advantage_head(cls_output)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_head(cls_output)

        return q_values
