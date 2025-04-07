# File: agent/networks/agent_network.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional

from config import (
    ModelConfig,
    EnvConfig,
    RNNConfig,
    TransformerConfig,
)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network: CNN -> Spatial Transformer -> Fusion MLP -> Optional LSTM -> Actor/Critic Heads.
    Handles both single step (eval) and sequence (RNN/Transformer training) inputs.
    Includes shape availability and explicit features.
    Uses SiLU activation by default based on ModelConfig.
    Applies Transformer attention directly to spatial CNN features.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        action_dim: int,
        model_config: ModelConfig.Network,
        rnn_config: RNNConfig,
        transformer_config: TransformerConfig,
        device: torch.device,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.env_config = env_config
        self.config = model_config
        self.rnn_config = rnn_config
        self.transformer_config = transformer_config
        self.device = device

        print(f"[ActorCriticNetwork] Target device set to: {self.device}")
        print(f"[ActorCriticNetwork] Using RNN: {self.rnn_config.USE_RNN}")
        print(
            f"[ActorCriticNetwork] Using Transformer: {self.transformer_config.USE_TRANSFORMER}"
        )
        print(
            f"[ActorCriticNetwork] Using Activation: {self.config.CONV_ACTIVATION.__name__}"
        )

        self.grid_c, self.grid_h, self.grid_w = self.env_config.GRID_STATE_SHAPE
        self.shape_feat_dim = self.env_config.SHAPE_STATE_DIM
        self.shape_availability_dim = self.env_config.SHAPE_AVAILABILITY_DIM
        self.explicit_features_dim = self.env_config.EXPLICIT_FEATURES_DIM

        # --- Feature Extractors ---
        self.conv_base, self.conv_out_h, self.conv_out_w, self.conv_out_c = (
            self._build_cnn_branch()
        )
        # self.conv_out_size = self._get_conv_out_size( # No longer needed as we don't flatten immediately
        #     (self.grid_c, self.grid_h, self.grid_w)
        # )
        print(
            f"  CNN Output Dim (HxWxC): ({self.conv_out_h}x{self.conv_out_w}x{self.conv_out_c})"
        )

        self.shape_mlp, self.shape_mlp_out_dim = self._build_shape_mlp_branch()
        print(f"  Shape Feature MLP Output Dim: {self.shape_mlp_out_dim}")

        # --- Optional Transformer Layer (Applied to Spatial CNN Features) ---
        self.transformer_encoder = None
        self.pos_embedding = None
        self.patch_projection = None
        transformer_output_dim = 0  # Will be set if transformer is used

        if self.transformer_config.USE_TRANSFORMER:
            if (
                self.transformer_config.TRANSFORMER_D_MODEL
                % self.transformer_config.TRANSFORMER_NHEAD
                != 0
            ):
                raise ValueError(
                    f"TRANSFORMER_D_MODEL ({self.transformer_config.TRANSFORMER_D_MODEL}) must be divisible by TRANSFORMER_NHEAD ({self.transformer_config.TRANSFORMER_NHEAD})"
                )

            # Linear projection from CNN output channels to Transformer dimension
            self.patch_projection = nn.Linear(
                self.conv_out_c, self.transformer_config.TRANSFORMER_D_MODEL
            ).to(self.device)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.transformer_config.TRANSFORMER_D_MODEL,
                nhead=self.transformer_config.TRANSFORMER_NHEAD,
                dim_feedforward=self.transformer_config.TRANSFORMER_DIM_FEEDFORWARD,
                dropout=self.transformer_config.TRANSFORMER_DROPOUT,
                activation=self.transformer_config.TRANSFORMER_ACTIVATION,
                batch_first=True,  # Input shape (Batch, Seq, Feature)
                device=self.device,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=self.transformer_config.TRANSFORMER_NUM_LAYERS
            ).to(self.device)

            # Positional embedding for the flattened spatial features
            self.num_patches = self.conv_out_h * self.conv_out_w
            self.pos_embedding = nn.Parameter(
                torch.randn(
                    1, self.num_patches, self.transformer_config.TRANSFORMER_D_MODEL
                )
            ).to(self.device)

            transformer_output_dim = (
                self.transformer_config.TRANSFORMER_D_MODEL
            )  # Output dim after pooling/CLS
            print(
                f"  Spatial Transformer Added (Input Patches: {self.num_patches}, Proj: {self.conv_out_c}->{transformer_output_dim}, d_model={transformer_output_dim}, nhead={self.transformer_config.TRANSFORMER_NHEAD}, layers={self.transformer_config.TRANSFORMER_NUM_LAYERS})"
            )
        else:
            # If no transformer, the input to fusion is the flattened CNN output
            transformer_output_dim = self.conv_out_h * self.conv_out_w * self.conv_out_c
            print(
                f"  Spatial Transformer DISABLED. Using flattened CNN output ({transformer_output_dim})."
            )

        # --- Fusion MLP ---
        # Input dim depends on whether Transformer was used
        combined_features_dim = (
            transformer_output_dim  # Output of Transformer (pooled) or flattened CNN
            + self.shape_mlp_out_dim
            + self.shape_availability_dim
            + self.explicit_features_dim
        )
        print(
            f"  Combined Features Dim (Spatial Features + Shape MLP + Avail + Explicit): {combined_features_dim}"
        )

        self.fusion_mlp, self.fusion_output_dim = self._build_fusion_mlp_branch(
            combined_features_dim
        )
        print(f"  Fusion MLP Output Dim: {self.fusion_output_dim}")

        # --- Optional LSTM Layer ---
        self.lstm_layer = None
        self.lstm_hidden_size = 0
        # Input to LSTM is output of Fusion MLP
        lstm_input_dim = self.fusion_output_dim
        # Input to heads is output of LSTM (or Fusion MLP if no LSTM)
        head_input_dim = lstm_input_dim

        if self.rnn_config.USE_RNN:
            self.lstm_hidden_size = self.rnn_config.LSTM_HIDDEN_SIZE
            self.lstm_layer = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.rnn_config.LSTM_NUM_LAYERS,
                batch_first=True,
            ).to(self.device)
            print(
                f"  LSTM Layer Added (Input: {lstm_input_dim}, Hidden: {self.lstm_hidden_size}, Layers: {self.rnn_config.LSTM_NUM_LAYERS})"
            )
            head_input_dim = self.lstm_hidden_size  # Heads take LSTM output

        # --- Actor/Critic Heads ---
        self.actor_head = nn.Linear(head_input_dim, self.action_dim).to(self.device)
        self.critic_head = nn.Linear(head_input_dim, 1).to(self.device)
        print(f"  Head Input Dim: {head_input_dim}")
        print(f"  Actor Head Output Dim: {self.action_dim}")
        print(f"  Critic Head Output Dim: 1")

        self._init_head_weights()

    def _init_head_weights(self):
        """Initializes Actor/Critic head weights."""
        print("  Initializing Actor/Critic heads using Xavier Uniform.")
        # Small gain for actor output layer
        nn.init.xavier_uniform_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0)
        # Standard gain for critic output layer
        nn.init.xavier_uniform_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0)

    def _build_cnn_branch(self) -> Tuple[nn.Sequential, int, int, int]:
        """Builds the CNN feature extractor for the grid."""
        conv_layers: List[nn.Module] = []
        current_channels = self.grid_c
        h, w = self.grid_h, self.grid_w
        cfg = self.config
        for i, out_channels in enumerate(cfg.CONV_CHANNELS):
            conv_layer = nn.Conv2d(
                current_channels,
                out_channels,
                kernel_size=cfg.CONV_KERNEL_SIZE,
                stride=cfg.CONV_STRIDE,
                padding=cfg.CONV_PADDING,
                bias=not cfg.USE_BATCHNORM_CONV,  # No bias if using BatchNorm
            ).to(self.device)
            conv_layers.append(conv_layer)
            if cfg.USE_BATCHNORM_CONV:
                conv_layers.append(nn.BatchNorm2d(out_channels).to(self.device))
            conv_layers.append(
                cfg.CONV_ACTIVATION()
            )  # Use activation from config (e.g., SiLU)
            current_channels = out_channels
            # Calculate output dimensions after conv
            h = (h + 2 * cfg.CONV_PADDING - cfg.CONV_KERNEL_SIZE) // cfg.CONV_STRIDE + 1
            w = (w + 2 * cfg.CONV_PADDING - cfg.CONV_KERNEL_SIZE) // cfg.CONV_STRIDE + 1
        return nn.Sequential(*conv_layers), h, w, current_channels

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        """Calculates the flattened output size of the CNN."""
        # This might still be useful for validation or if Transformer is disabled
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape, device=self.device)
            output = self.conv_base(dummy_input)
            return int(np.prod(output.size()[1:]))

    def _build_shape_mlp_branch(self) -> Tuple[nn.Sequential, int]:
        """Builds the MLP for processing shape features."""
        shape_mlp_layers: List[nn.Module] = []
        current_dim = self.env_config.SHAPE_STATE_DIM
        cfg = self.config
        for hidden_dim in cfg.SHAPE_FEATURE_MLP_DIMS:
            lin_layer = nn.Linear(current_dim, hidden_dim).to(self.device)
            shape_mlp_layers.append(lin_layer)
            shape_mlp_layers.append(
                cfg.SHAPE_MLP_ACTIVATION()
            )  # Use activation from config
            current_dim = hidden_dim
        if not cfg.SHAPE_FEATURE_MLP_DIMS:
            return (
                nn.Identity(),
                current_dim,
            )  # Return Identity if no MLP layers defined
        return nn.Sequential(*shape_mlp_layers), current_dim

    def _build_fusion_mlp_branch(self, input_dim: int) -> Tuple[nn.Sequential, int]:
        """Builds the MLP that fuses all features before RNN/Transformer/Heads."""
        fusion_layers: List[nn.Module] = []
        current_fusion_dim = input_dim
        cfg = self.config
        for i, hidden_dim in enumerate(cfg.COMBINED_FC_DIMS):
            linear_layer = nn.Linear(
                current_fusion_dim, hidden_dim, bias=not cfg.USE_BATCHNORM_FC
            ).to(self.device)
            fusion_layers.append(linear_layer)
            if cfg.USE_BATCHNORM_FC:
                # Use BatchNorm1d for MLP layers
                fusion_layers.append(nn.BatchNorm1d(hidden_dim).to(self.device))
            fusion_layers.append(
                cfg.COMBINED_ACTIVATION()
            )  # Use activation from config
            if cfg.DROPOUT_FC > 0:
                fusion_layers.append(nn.Dropout(cfg.DROPOUT_FC).to(self.device))
            current_fusion_dim = hidden_dim
        return nn.Sequential(*fusion_layers), current_fusion_dim

    def forward(
        self,
        grid_tensor: torch.Tensor,
        shape_feature_tensor: torch.Tensor,
        shape_availability_tensor: torch.Tensor,
        explicit_features_tensor: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        padding_mask: Optional[
            torch.Tensor
        ] = None,  # Shape: (batch_size, seq_len) - Not used with spatial transformer
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the network. Handles single step and sequence inputs.
        Applies Transformer to spatial CNN features if enabled.
        """
        # --- Ensure inputs are on the correct device ---
        model_device = self.device
        grid_tensor = grid_tensor.to(model_device)
        shape_feature_tensor = shape_feature_tensor.to(model_device)
        shape_availability_tensor = shape_availability_tensor.to(model_device)
        explicit_features_tensor = explicit_features_tensor.to(model_device)
        if hidden_state is not None:
            hidden_state = (
                hidden_state[0].to(model_device),
                hidden_state[1].to(model_device),
            )
        # padding_mask is not used in this spatial transformer setup

        # --- Determine input shape and flatten if necessary ---
        is_sequence_input = grid_tensor.ndim == 5
        initial_batch_size = grid_tensor.shape[0]
        seq_len = grid_tensor.shape[1] if is_sequence_input else 1
        num_samples = initial_batch_size * seq_len  # Total samples to process

        # Reshape inputs to (N, Features) where N = B*T for processing by CNN/MLP
        grid_input_flat = grid_tensor.reshape(
            num_samples, *self.env_config.GRID_STATE_SHAPE
        )
        shape_feature_input_flat = shape_feature_tensor.reshape(
            num_samples, self.env_config.SHAPE_STATE_DIM
        )
        shape_availability_input_flat = shape_availability_tensor.reshape(
            num_samples, self.env_config.SHAPE_AVAILABILITY_DIM
        )
        explicit_features_input_flat = explicit_features_tensor.reshape(
            num_samples, self.env_config.EXPLICIT_FEATURES_DIM
        )

        # --- Feature Extraction ---
        # CNN processing
        conv_features_spatial = self.conv_base(grid_input_flat)  # Shape: (N, C, H', W')

        # Shape MLP processing
        shape_features_flat = self.shape_mlp(
            shape_feature_input_flat
        )  # Shape: (N, ShapeMLPDim)

        # --- Optional Spatial Transformer ---
        spatial_features_processed = None
        if (
            self.transformer_config.USE_TRANSFORMER
            and self.transformer_encoder is not None
            and self.patch_projection is not None
            and self.pos_embedding is not None
        ):
            # Reshape spatial features for Transformer: (N, C, H', W') -> (N, H'*W', C)
            n_samples, c_out, h_out, w_out = conv_features_spatial.shape
            spatial_flat_patches = conv_features_spatial.flatten(2).permute(
                0, 2, 1
            )  # Shape: (N, H'*W', C)

            # Project patches to Transformer dimension
            projected_patches = self.patch_projection(
                spatial_flat_patches
            )  # Shape: (N, NumPatches, D_model)

            # Add positional embedding
            transformer_input = (
                projected_patches + self.pos_embedding
            )  # Broadcasting (1, NumPatches, D_model)

            # Apply Transformer encoder
            # Note: padding_mask is not typically used with spatial features like this
            transformer_output = self.transformer_encoder(
                transformer_input
            )  # Shape: (N, NumPatches, D_model)

            # Pool the transformer output (e.g., mean pooling)
            spatial_features_processed = transformer_output.mean(
                dim=1
            )  # Shape: (N, D_model)

        else:
            # If no transformer, flatten the CNN output
            spatial_features_processed = conv_features_spatial.view(
                num_samples, -1
            )  # Shape: (N, C*H'*W')

        # --- Feature Fusion ---
        combined_features_flat = torch.cat(
            (
                spatial_features_processed,  # Output from Transformer or flattened CNN
                shape_features_flat,
                shape_availability_input_flat,
                explicit_features_input_flat,
            ),
            dim=1,
        )
        fused_features_flat = self.fusion_mlp(
            combined_features_flat
        )  # Shape: (N, FusionDim)

        # --- Optional LSTM ---
        next_hidden_state = hidden_state  # Pass incoming state through
        head_input_features = (
            fused_features_flat  # Input to heads starts as output of Fusion MLP
        )

        if self.rnn_config.USE_RNN and self.lstm_layer is not None:
            # Reshape to (B, T, Dim) for LSTM
            lstm_input = fused_features_flat.view(initial_batch_size, seq_len, -1)
            # Apply LSTM
            lstm_output_seq, next_hidden_state_tuple = self.lstm_layer(
                lstm_input, hidden_state
            )
            # Flatten output for heads
            head_input_features = lstm_output_seq.reshape(
                num_samples, -1
            )  # Shape: (N, LSTMDim)
            # Update the hidden state to be returned
            next_hidden_state = next_hidden_state_tuple

        # --- Actor/Critic Heads ---
        policy_logits_flat = self.actor_head(head_input_features)
        value_flat = self.critic_head(head_input_features)

        # --- Reshape outputs back to sequence if necessary ---
        if is_sequence_input:
            policy_logits = policy_logits_flat.view(initial_batch_size, seq_len, -1)
            value = value_flat.view(initial_batch_size, seq_len, -1)
        else:
            policy_logits = policy_logits_flat
            value = value_flat

        return policy_logits, value, next_hidden_state

    def get_initial_hidden_state(
        self, batch_size: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the initial hidden state for the LSTM layer."""
        if not self.rnn_config.USE_RNN or self.lstm_layer is None:
            return None
        model_device = self.device
        num_layers = self.rnn_config.LSTM_NUM_LAYERS
        hidden_size = self.rnn_config.LSTM_HIDDEN_SIZE
        # Shape: (num_layers, batch_size, hidden_size)
        h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=model_device)
        c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=model_device)
        return (h_0, c_0)
