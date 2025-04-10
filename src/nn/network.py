# File: src/nn/network.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Mapping, Dict, Any

# Import necessary components
from src.config import ModelConfig, EnvConfig, TrainConfig
from src.environment import GameState
from src.utils.types import ActionType, PolicyValueOutput, StateType
from .model import AlphaTriangleNet
from src.features import extract_state_features

logger = logging.getLogger(__name__)


class NeuralNetwork:
    """Wrapper for the PyTorch model providing evaluation and state management."""

    def __init__(
        self,
        model_config: ModelConfig,
        env_config: EnvConfig,
        train_config: TrainConfig,
        device: torch.device,
    ):
        self.model_config = model_config
        self.env_config = env_config
        self.train_config = train_config
        self.device = device
        self.model = AlphaTriangleNet(model_config, env_config).to(device)
        self.action_dim = env_config.ACTION_DIM

    def _state_to_tensors(self, state: GameState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from GameState and converts them to tensors."""
        state_dict: StateType = extract_state_features(state, self.model_config)
        grid_tensor = torch.from_numpy(state_dict["grid"]).unsqueeze(0).to(self.device)
        other_features_tensor = (
            torch.from_numpy(state_dict["other_features"]).unsqueeze(0).to(self.device)
        )
        return grid_tensor, other_features_tensor

    def _batch_states_to_tensors(
        self, states: List[GameState]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from a batch of GameStates and converts to batched tensors."""
        if not states:
            grid_shape = (0, self.model_config.GRID_INPUT_CHANNELS, self.env_config.ROWS, self.env_config.COLS)
            other_shape = (0, self.model_config.OTHER_NN_INPUT_FEATURES_DIM)
            return torch.empty(grid_shape, device=self.device), torch.empty(other_shape, device=self.device)

        batch_grid = []
        batch_other = []
        for state in states:
            state_dict: StateType = extract_state_features(state, self.model_config)
            batch_grid.append(state_dict["grid"])
            batch_other.append(state_dict["other_features"])

        grid_tensor = torch.from_numpy(np.stack(batch_grid)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(batch_other)).to(self.device)
        return grid_tensor, other_features_tensor

    @torch.inference_mode()
    def evaluate(self, state: GameState) -> PolicyValueOutput:
        """Evaluates a single state by extracting features and running the model."""
        self.model.eval()
        grid_tensor, other_features_tensor = self._state_to_tensors(state)
        policy_logits, value = self.model(grid_tensor, other_features_tensor)
        policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value_scalar = value.squeeze(0).item()
        action_policy: Mapping[ActionType, float] = dict(enumerate(policy_probs))
        return action_policy, value_scalar

    @torch.inference_mode()
    def evaluate_batch(self, states: List[GameState]) -> List[PolicyValueOutput]:
        """Evaluates a batch of states by extracting features and running the model."""
        if not states: return []
        self.model.eval()
        grid_tensor, other_features_tensor = self._batch_states_to_tensors(states)
        policy_logits, value = self.model(grid_tensor, other_features_tensor)
        policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()
        values = value.squeeze(1).cpu().numpy()
        results: List[PolicyValueOutput] = [
            (dict(enumerate(policy_probs[i])), values[i]) for i in range(len(states))
        ]
        return results

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Returns the model's state dictionary, moved to CPU."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Loads the model's state dictionary from the provided weights."""
        try:
            # Ensure weights are loaded to the correct device for this NN instance
            weights_on_device = {k: v.to(self.device) for k, v in weights.items()}
            self.model.load_state_dict(weights_on_device)
            self.model.eval() # Ensure model is in eval mode after loading weights
            logger.debug("NN weights set successfully.")
        except Exception as e:
            logger.error(f"Error setting weights on NN instance: {e}", exc_info=True)
            raise
