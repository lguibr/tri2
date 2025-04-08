import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List  # Added List
import numpy as np  # Added numpy

from config import ModelConfig, EnvConfig
from utils.types import StateType, ActionType


class ResidualBlock(nn.Module):
    """Basic Residual Block for CNN."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """
    Neural Network for AlphaZero.
    Takes game state features and outputs policy logits and a value estimate.
    Includes methods for single and batched predictions.
    """

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        model_config: Optional[ModelConfig.Network] = None,
    ):
        super().__init__()
        self.env_cfg = env_config if env_config else EnvConfig()
        self.model_cfg = model_config if model_config else ModelConfig.Network()

        # --- Input Processing Layers ---
        grid_input_channels = self.env_cfg.GRID_STATE_SHAPE[0]
        conv_channels = self.model_cfg.CONV_CHANNELS
        current_channels = grid_input_channels
        conv_layers = []
        for out_channels in conv_channels:
            conv_layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=self.model_cfg.CONV_KERNEL_SIZE,
                    stride=self.model_cfg.CONV_STRIDE,
                    padding=self.model_cfg.CONV_PADDING,
                    bias=not self.model_cfg.USE_BATCHNORM_CONV,
                )
            )
            if self.model_cfg.USE_BATCHNORM_CONV:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(self.model_cfg.CONV_ACTIVATION())
            # Add ResidualBlock *after* activation
            conv_layers.append(ResidualBlock(out_channels))
            current_channels = out_channels
        self.conv_backbone = nn.Sequential(*conv_layers)

        conv_output_size = self._calculate_conv_output_size(
            (grid_input_channels, self.env_cfg.ROWS, self.env_cfg.COLS)
        )

        shape_input_dim = self.env_cfg.SHAPE_STATE_DIM
        shape_mlp_dims = self.model_cfg.SHAPE_FEATURE_MLP_DIMS
        shape_layers = []
        current_shape_dim = shape_input_dim
        for dim in shape_mlp_dims:
            shape_layers.append(nn.Linear(current_shape_dim, dim))
            shape_layers.append(self.model_cfg.SHAPE_MLP_ACTIVATION())
            current_shape_dim = dim
        self.shape_mlp = nn.Sequential(*shape_layers)
        shape_output_dim = current_shape_dim if shape_mlp_dims else shape_input_dim

        other_features_dim = (
            self.env_cfg.SHAPE_AVAILABILITY_DIM + self.env_cfg.EXPLICIT_FEATURES_DIM
        )

        combined_input_dim = conv_output_size + shape_output_dim + other_features_dim
        combined_fc_dims = self.model_cfg.COMBINED_FC_DIMS
        fusion_layers = []
        current_combined_dim = combined_input_dim
        for dim in combined_fc_dims:
            fusion_layers.append(
                nn.Linear(
                    current_combined_dim, dim, bias=not self.model_cfg.USE_BATCHNORM_FC
                )
            )
            if self.model_cfg.USE_BATCHNORM_FC:
                fusion_layers.append(nn.BatchNorm1d(dim))
            fusion_layers.append(self.model_cfg.COMBINED_ACTIVATION())
            if self.model_cfg.DROPOUT_FC > 0:
                fusion_layers.append(nn.Dropout(self.model_cfg.DROPOUT_FC))
            current_combined_dim = dim
        self.fusion_mlp = nn.Sequential(*fusion_layers)
        fusion_output_dim = current_combined_dim

        self.policy_head = nn.Linear(fusion_output_dim, self.env_cfg.ACTION_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh()
        )

    def _calculate_conv_output_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Helper to calculate the flattened output size of the conv backbone."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.conv_backbone(dummy_input)
            return int(torch.flatten(output, 1).shape[1])

    def forward(
        self, state: StateType  # Expects Tensors in the dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        Assumes input state dictionary contains tensors.
        """
        grid = state["grid"]
        shapes = state["shapes"]
        shape_availability = state["shape_availability"]
        explicit_features = state["explicit_features"]

        conv_out = self.conv_backbone(grid)
        flat_conv_out = torch.flatten(conv_out, 1)

        if self.model_cfg.SHAPE_FEATURE_MLP_DIMS:
            shape_out = self.shape_mlp(shapes)
        else:
            shape_out = shapes

        other_features = torch.cat([shape_availability, explicit_features], dim=-1)
        combined_features = torch.cat([flat_conv_out, shape_out, other_features], dim=1)

        fused_out = self.fusion_mlp(combined_features)

        policy_logits = self.policy_head(fused_out)
        value = self.value_head(fused_out)

        return policy_logits, value

    def _prepare_state_batch(self, state_numpy_list: List[StateType]) -> StateType:
        """Converts a list of numpy state dicts to a batched tensor state dict."""
        device = next(self.parameters()).device
        # Initialize structure to hold batched data for each state component
        batched_tensors = {key: [] for key in state_numpy_list[0].keys()}

        for state_numpy in state_numpy_list:
            for key, value in state_numpy.items():
                if key in batched_tensors:
                    batched_tensors[key].append(value)
                else:
                    # This shouldn't happen if all state dicts have the same keys
                    print(
                        f"Warning: Key {key} not found in initial state dict during batching."
                    )

        # Convert lists of numpy arrays to batched tensors
        final_batched_states = {
            k: torch.from_numpy(np.stack(v)).to(device)
            for k, v in batched_tensors.items()
        }
        return final_batched_states

    def predict_batch(
        self,
        state_numpy_list: List[
            StateType
        ],  # Expects list of numpy arrays from GameState
    ) -> Tuple[List[Dict[ActionType, float]], List[float]]:
        """
        Performs batched prediction for MCTS.
        Takes a list of state dictionaries (numpy arrays), converts to a batch tensor,
        runs inference, applies softmax, and returns lists of policy dicts and scalar values.
        """
        if not state_numpy_list:
            return [], []

        # Prepare the batch of states
        batched_state_tensors = self._prepare_state_batch(state_numpy_list)

        self.eval()
        with torch.no_grad():
            policy_logits_batch, value_batch = self.forward(batched_state_tensors)

        # Process results back into lists
        policy_probs_batch = F.softmax(policy_logits_batch, dim=-1).cpu().numpy()
        values = value_batch.squeeze(-1).cpu().numpy().tolist()  # Squeeze value output

        policy_dicts = []
        for policy_probs in policy_probs_batch:
            policy_dicts.append({i: float(prob) for i, prob in enumerate(policy_probs)})

        return policy_dicts, values

    def predict(
        self, state_numpy: StateType  # Expects numpy arrays from GameState
    ) -> Tuple[Dict[ActionType, float], float]:
        """
        Convenience method for single prediction (e.g., initial root prediction).
        Uses predict_batch internally.
        """
        policy_list, value_list = self.predict_batch([state_numpy])
        return policy_list[0], value_list[0]

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns the model's state dictionary."""
        return self.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the model's state dictionary."""
        super().load_state_dict(state_dict)
