# File: agent/networks/agent_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from config import ModelConfig, EnvConfig, PPOConfig, RNNConfig, DEVICE
from typing import Tuple, List, Type, Optional


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network: CNN+MLP -> Fusion -> Optional LSTM -> Actor/Critic Heads.
    Handles both single step (eval) and sequence (RNN training) inputs.
    Now includes shape availability and explicit features.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        action_dim: int,
        model_config: ModelConfig.Network,
        rnn_config: RNNConfig,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.env_config = env_config
        self.config = model_config
        self.rnn_config = rnn_config
        self.device = DEVICE

        print(f"[ActorCriticNetwork] Target device set to: {self.device}")
        print(f"[ActorCriticNetwork] Using RNN: {self.rnn_config.USE_RNN}")

        self.grid_c, self.grid_h, self.grid_w = self.env_config.GRID_STATE_SHAPE
        self.shape_feat_dim = self.env_config.SHAPE_STATE_DIM
        self.shape_availability_dim = self.env_config.SHAPE_AVAILABILITY_DIM
        # --- UPDATED: Read the potentially larger explicit features dimension ---
        self.explicit_features_dim = self.env_config.EXPLICIT_FEATURES_DIM
        # --- END UPDATED ---
        self.num_shape_slots = self.env_config.NUM_SHAPE_SLOTS
        self.shape_feat_per_slot = self.env_config.SHAPE_FEATURES_PER_SHAPE

        print(f"[ActorCriticNetwork] Initializing:")
        print(f"  Input Grid Shape: [B, {self.grid_c}, {self.grid_h}, {self.grid_w}]")
        print(f"  Input Shape Features Dim: {self.shape_feat_dim}")
        print(f"  Input Shape Availability Dim: {self.shape_availability_dim}")
        # --- UPDATED: Log the correct explicit features dim ---
        print(f"  Input Explicit Features Dim: {self.explicit_features_dim}")
        # --- END UPDATED ---

        self.conv_base, conv_out_h, conv_out_w, conv_out_c = self._build_cnn_branch()
        self.conv_out_size = self._get_conv_out_size(
            (self.grid_c, self.grid_h, self.grid_w)
        )
        print(
            f"  CNN Output Dim (HxWxC): ({conv_out_h}x{conv_out_w}x{conv_out_c}) -> Flat: {self.conv_out_size}"
        )

        self.shape_mlp, self.shape_mlp_out_dim = self._build_shape_mlp_branch()
        print(f"  Shape Feature MLP Output Dim: {self.shape_mlp_out_dim}")

        # --- MODIFIED: Calculate combined dim including availability and the *new* explicit features dim ---
        combined_features_dim = (
            self.conv_out_size
            + self.shape_mlp_out_dim
            + self.shape_availability_dim
            + self.explicit_features_dim  # Use the updated dimension
        )
        # --- END MODIFIED ---
        print(
            f"  Combined Features Dim (CNN + Shape MLP + Availability + Explicit): {combined_features_dim}"
        )

        self.fusion_mlp, self.fusion_output_dim = self._build_fusion_mlp_branch(
            combined_features_dim  # Pass the updated combined dimension
        )
        print(f"  Fusion MLP Output Dim: {self.fusion_output_dim}")

        self.lstm_layer = None
        self.lstm_hidden_size = 0
        if self.rnn_config.USE_RNN:
            self.lstm_hidden_size = self.rnn_config.LSTM_HIDDEN_SIZE
            self.lstm_layer = nn.LSTM(
                input_size=self.fusion_output_dim,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.rnn_config.LSTM_NUM_LAYERS,
                batch_first=True,
            ).to(self.device)
            print(f"  LSTM Layer Added (Hidden Size: {self.lstm_hidden_size})")
            head_input_dim = self.lstm_hidden_size
        else:
            head_input_dim = self.fusion_output_dim

        self.actor_head = nn.Linear(head_input_dim, self.action_dim).to(self.device)
        self.critic_head = nn.Linear(head_input_dim, 1).to(self.device)
        print(f"  Actor Head Output Dim: {self.action_dim}")
        print(f"  Critic Head Output Dim: 1")

        self._init_head_weights()

    def _init_head_weights(self):
        print("  Initializing Actor/Critic heads using Xavier Uniform.")
        actor_gain = nn.init.calculate_gain("linear")
        critic_gain = nn.init.calculate_gain("linear")
        nn.init.xavier_uniform_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0)
        nn.init.xavier_uniform_(self.critic_head.weight, gain=critic_gain)
        nn.init.constant_(self.critic_head.bias, 0)

    def _build_cnn_branch(self) -> Tuple[nn.Sequential, int, int, int]:
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
                bias=not cfg.USE_BATCHNORM_CONV,
            ).to(self.device)
            conv_layers.append(conv_layer)
            if cfg.USE_BATCHNORM_CONV:
                conv_layers.append(nn.BatchNorm2d(out_channels).to(self.device))
            conv_layers.append(cfg.CONV_ACTIVATION())
            current_channels = out_channels
            h = (h + 2 * cfg.CONV_PADDING - cfg.CONV_KERNEL_SIZE) // cfg.CONV_STRIDE + 1
            w = (w + 2 * cfg.CONV_PADDING - cfg.CONV_KERNEL_SIZE) // cfg.CONV_STRIDE + 1
        return nn.Sequential(*conv_layers), h, w, current_channels

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape, device=self.device)
            self.conv_base.eval()
            output = self.conv_base(dummy_input)
            self.conv_base.train()
            return int(np.prod(output.size()[1:]))

    def _build_shape_mlp_branch(self) -> Tuple[nn.Sequential, int]:
        shape_mlp_layers: List[nn.Module] = []
        current_dim = self.env_config.SHAPE_STATE_DIM
        cfg = self.config
        for hidden_dim in cfg.SHAPE_FEATURE_MLP_DIMS:
            lin_layer = nn.Linear(current_dim, hidden_dim).to(self.device)
            shape_mlp_layers.append(lin_layer)
            shape_mlp_layers.append(cfg.SHAPE_MLP_ACTIVATION())
            current_dim = hidden_dim
        if not cfg.SHAPE_FEATURE_MLP_DIMS:
            return nn.Identity(), current_dim
        return nn.Sequential(*shape_mlp_layers), current_dim

    def _build_fusion_mlp_branch(self, input_dim: int) -> Tuple[nn.Sequential, int]:
        fusion_layers: List[nn.Module] = []
        current_fusion_dim = input_dim
        cfg = self.config
        for i, hidden_dim in enumerate(cfg.COMBINED_FC_DIMS):
            linear_layer = nn.Linear(
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

    def forward(
        self,
        grid_tensor: torch.Tensor,
        shape_feature_tensor: torch.Tensor,
        shape_availability_tensor: torch.Tensor,
        # --- Input signature remains the same ---
        explicit_features_tensor: torch.Tensor,
        # --- End Input ---
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        model_device = next(self.parameters()).device
        grid_tensor = grid_tensor.to(model_device)
        shape_feature_tensor = shape_feature_tensor.to(model_device)
        shape_availability_tensor = shape_availability_tensor.to(model_device)
        # --- Move explicit features tensor to device ---
        explicit_features_tensor = explicit_features_tensor.to(model_device)
        # --- End Move ---
        if hidden_state:
            hidden_state = (
                hidden_state[0].to(model_device),
                hidden_state[1].to(model_device),
            )

        is_sequence = self.rnn_config.USE_RNN and grid_tensor.ndim == 5
        initial_batch_size = grid_tensor.shape[0]
        seq_len = grid_tensor.shape[1] if is_sequence else 1
        num_samples = initial_batch_size * seq_len

        # Reshape inputs to [B*T or B, ...]
        grid_input_flat = grid_tensor.reshape(
            num_samples, *self.env_config.GRID_STATE_SHAPE
        )
        shape_feature_input_flat = shape_feature_tensor.reshape(
            num_samples, self.env_config.SHAPE_STATE_DIM
        )
        shape_availability_input_flat = shape_availability_tensor.reshape(
            num_samples, self.env_config.SHAPE_AVAILABILITY_DIM
        )
        # --- Reshape explicit features tensor (using the updated dimension from config) ---
        explicit_features_input_flat = explicit_features_tensor.reshape(
            num_samples, self.env_config.EXPLICIT_FEATURES_DIM
        )
        # --- End Reshape ---

        conv_output = self.conv_base(grid_input_flat)
        conv_output_flat = conv_output.view(num_samples, -1)
        shape_mlp_output = self.shape_mlp(shape_feature_input_flat)

        # --- Combine all feature streams (logic remains the same, dimensions handled by reshape) ---
        combined_features = torch.cat(
            (
                conv_output_flat,
                shape_mlp_output,
                shape_availability_input_flat,
                explicit_features_input_flat,  # Add explicit features
            ),
            dim=1,
        )
        # --- End Combine ---

        fused_output = self.fusion_mlp(combined_features)

        next_hidden_state = hidden_state
        if self.rnn_config.USE_RNN and self.lstm_layer is not None:
            lstm_input = fused_output.view(
                initial_batch_size, seq_len, self.fusion_output_dim
            )
            lstm_output, next_hidden_state = self.lstm_layer(lstm_input, hidden_state)
            head_input = lstm_output.contiguous().view(num_samples, -1)
        else:
            head_input = fused_output

        policy_logits = self.actor_head(head_input)
        value = self.critic_head(head_input)

        # Reshape back if input was sequence
        if is_sequence:
            policy_logits = policy_logits.view(initial_batch_size, seq_len, -1)
            value = value.view(initial_batch_size, seq_len, -1)

        return policy_logits, value, next_hidden_state

    def get_initial_hidden_state(
        self, batch_size: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.rnn_config.USE_RNN or self.lstm_layer is None:
            return None
        model_device = next(self.parameters()).device
        num_layers = self.rnn_config.LSTM_NUM_LAYERS
        hidden_size = self.rnn_config.LSTM_HIDDEN_SIZE
        h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=model_device)
        c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=model_device)
        return (h_0, c_0)
