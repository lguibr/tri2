# File: src/rl/core/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

# --- Package Imports ---
from src.config import TrainConfig, EnvConfig, ModelConfig
from src.nn import NeuralNetwork
from src.environment import GameState

# Use core types - ExperienceBatch contains GameState now
from src.utils.types import ExperienceBatch, PolicyTargetMapping, ActionType, StateType
from src.features import extract_state_features

logger = logging.getLogger(__name__)


class Trainer:
    """Handles the neural network training process, including loss calculation and optimizer steps."""

    def __init__(
        self,
        nn_interface: NeuralNetwork,
        train_config: TrainConfig,
        env_config: EnvConfig,
    ):
        self.nn = nn_interface
        self.model = nn_interface.model
        self.train_config = train_config
        self.env_config = env_config
        self.model_config = nn_interface.model_config
        self.device = nn_interface.device
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)

    def _create_optimizer(self) -> optim.Optimizer:
        """Creates the optimizer based on TrainConfig."""
        lr = self.train_config.LEARNING_RATE
        wd = self.train_config.WEIGHT_DECAY
        params = self.model.parameters()
        opt_type = self.train_config.OPTIMIZER_TYPE.lower()
        logger.info(f"Creating optimizer: {opt_type}, LR: {lr}, WD: {wd}")
        if opt_type == "adam": return optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_type == "adamw": return optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_type == "sgd": return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        else: raise ValueError(f"Unsupported optimizer type: {self.train_config.OPTIMIZER_TYPE}")

    def _create_scheduler(self, optimizer: optim.Optimizer) -> Optional[_LRScheduler]:
        """Creates the learning rate scheduler based on TrainConfig."""
        scheduler_type = self.train_config.LR_SCHEDULER_TYPE
        if not scheduler_type or scheduler_type.lower() == "none":
            logger.info("No LR scheduler configured.")
            return None
        scheduler_type = scheduler_type.lower()
        logger.info(f"Creating LR scheduler: {scheduler_type}")
        if scheduler_type == "steplr":
            step_size = getattr(self.train_config, "LR_SCHEDULER_STEP_SIZE", 100000)
            gamma = getattr(self.train_config, "LR_SCHEDULER_GAMMA", 0.1)
            logger.info(f"  StepLR params: step_size={step_size}, gamma={gamma}")
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "cosineannealinglr":
            t_max = self.train_config.LR_SCHEDULER_T_MAX
            eta_min = self.train_config.LR_SCHEDULER_ETA_MIN
            logger.info(f"  CosineAnnealingLR params: T_max={t_max}, eta_min={eta_min}")
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        else: raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _prepare_batch(
        self, batch: ExperienceBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts features from GameStates and converts experiences into tensors."""
        batch_size = len(batch)
        grids = []
        other_features = []
        value_targets = []
        policy_target_tensor = torch.zeros((batch_size, self.env_config.ACTION_DIM), dtype=torch.float32, device=self.device)

        for i, (game_state, policy_target_map, value_target) in enumerate(batch):
            state_dict: StateType = extract_state_features(game_state, self.model_config)
            grids.append(state_dict["grid"])
            other_features.append(state_dict["other_features"])
            value_targets.append(value_target)
            for action, prob in policy_target_map.items():
                if 0 <= action < self.env_config.ACTION_DIM:
                    policy_target_tensor[i, action] = prob
                else: logger.warning(f"Action {action} out of bounds in policy target map for sample {i}.")

        grid_tensor = torch.from_numpy(np.stack(grids)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(other_features)).to(self.device)
        value_target_tensor = torch.tensor(value_targets, dtype=torch.float32, device=self.device).unsqueeze(1)

        expected_other_dim = self.model_config.OTHER_NN_INPUT_FEATURES_DIM
        if batch_size > 0 and other_features_tensor.shape[1] != expected_other_dim:
            raise ValueError(f"Unexpected other_features tensor shape: {other_features_tensor.shape}, expected dim {expected_other_dim}")

        return grid_tensor, other_features_tensor, policy_target_tensor, value_target_tensor

    def train_step(self, batch: ExperienceBatch) -> Optional[Dict[str, float]]:
        """Performs a single training step on the given batch."""
        if not batch:
            logger.warning("train_step called with empty batch.")
            return None

        self.model.train()
        try:
            grid_t, other_t, policy_target_t, value_target_t = self._prepare_batch(batch)
        except Exception as e:
            logger.error(f"Error preparing batch for training: {e}", exc_info=True)
            return None

        self.optimizer.zero_grad()
        policy_logits, value_pred = self.model(grid_t, other_t)

        value_loss = F.mse_loss(value_pred, value_target_t)
        policy_loss = F.cross_entropy(policy_logits, policy_target_t)

        entropy = 0.0
        entropy_loss = 0.0
        if self.train_config.ENTROPY_BONUS_WEIGHT > 0:
            policy_probs = F.softmax(policy_logits, dim=1)
            entropy_term = -torch.sum(policy_probs * torch.log(policy_probs + 1e-9), dim=1)
            entropy = entropy_term.mean().item()
            entropy_loss = -self.train_config.ENTROPY_BONUS_WEIGHT * entropy_term.mean()

        total_loss = (
            self.train_config.POLICY_LOSS_WEIGHT * policy_loss
            + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            + entropy_loss
        )

        total_loss.backward()

        if self.train_config.GRADIENT_CLIP_VALUE is not None and self.train_config.GRADIENT_CLIP_VALUE > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.GRADIENT_CLIP_VALUE)

        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy,
        }

    def get_current_lr(self) -> float:
        """Returns the current learning rate from the optimizer."""
        try: return self.optimizer.param_groups[0]["lr"]
        except (IndexError, KeyError):
            logger.warning("Could not retrieve learning rate from optimizer.")
            return 0.0