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

# Removed GameState import as it's no longer needed for batch prep

# Use core types - ExperienceBatch contains StateType now
# Import PERBatchSample for the sample result type
from src.utils.types import (
    ExperienceBatch,
    PolicyTargetMapping,
    ActionType,
    StateType,
    PERBatchSample,
)

# Removed feature extractor import as features are pre-extracted
# from src.features import extract_state_features

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
        if opt_type == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_type == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.train_config.OPTIMIZER_TYPE}"
            )

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
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "cosineannealinglr":
            t_max = self.train_config.LR_SCHEDULER_T_MAX
            eta_min = self.train_config.LR_SCHEDULER_ETA_MIN
            if t_max is None:
                logger.warning(
                    "LR_SCHEDULER_T_MAX is None for CosineAnnealingLR. Scheduler might not work as expected."
                )
                t_max = self.train_config.MAX_TRAINING_STEPS or 1_000_000
            logger.info(f"  CosineAnnealingLR params: T_max={t_max}, eta_min={eta_min}")
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _prepare_batch(
        self, batch: ExperienceBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts a batch of experiences (containing StateType) into tensors.
        No feature extraction needed here anymore.
        """
        batch_size = len(batch)
        grids = []
        other_features = []
        value_targets = []
        policy_target_tensor = torch.zeros(
            (batch_size, self.env_config.ACTION_DIM),
            dtype=torch.float32,
            device=self.device,
        )

        for i, (state_features, policy_target_map, value_target) in enumerate(batch):
            # Directly use the stored features
            grids.append(state_features["grid"])
            other_features.append(state_features["other_features"])
            value_targets.append(value_target)
            for action, prob in policy_target_map.items():
                if 0 <= action < self.env_config.ACTION_DIM:
                    policy_target_tensor[i, action] = prob
                else:
                    logger.warning(
                        f"Action {action} out of bounds in policy target map for sample {i}."
                    )

        grid_tensor = torch.from_numpy(np.stack(grids)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(other_features)).to(
            self.device
        )
        value_target_tensor = torch.tensor(
            value_targets, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        expected_other_dim = self.model_config.OTHER_NN_INPUT_FEATURES_DIM
        if batch_size > 0 and other_features_tensor.shape[1] != expected_other_dim:
            raise ValueError(
                f"Unexpected other_features tensor shape: {other_features_tensor.shape}, expected dim {expected_other_dim}"
            )

        return (
            grid_tensor,
            other_features_tensor,
            policy_target_tensor,
            value_target_tensor,
        )

    def train_step(
        self, per_sample: PERBatchSample
    ) -> Optional[Tuple[Dict[str, float], np.ndarray]]:
        """
        Performs a single training step on the given batch from PER buffer.
        Returns loss info dictionary and TD errors for priority updates.
        """
        batch = per_sample["batch"]
        indices = per_sample["indices"]  # Tree indices for priority updates
        is_weights = per_sample["weights"]  # Importance sampling weights

        if not batch:
            logger.warning("train_step called with empty batch.")
            return None

        self.model.train()
        try:
            grid_t, other_t, policy_target_t, value_target_t = self._prepare_batch(
                batch
            )
            # Ensure is_weights_t has shape [batch_size] for element-wise multiplication with policy loss
            is_weights_t = torch.from_numpy(is_weights).to(self.device)
        except Exception as e:
            logger.error(f"Error preparing batch for training: {e}", exc_info=True)
            return None

        self.optimizer.zero_grad()
        policy_logits, value_pred = self.model(grid_t, other_t)

        # --- Calculate Losses ---
        # Value Loss (MSE) - Calculate element-wise before applying weights
        value_loss_elementwise = F.mse_loss(
            value_pred, value_target_t, reduction="none"
        )
        # Apply importance sampling weights (needs shape [batch_size, 1] to broadcast with [batch_size, 1])
        value_loss = (value_loss_elementwise * is_weights_t.unsqueeze(1)).mean()

        # Policy Loss (Cross-Entropy for soft targets)
        # Calculate -sum(target_probs * log(predicted_probs)) element-wise
        log_probs = F.log_softmax(policy_logits, dim=1)
        # Ensure policy_target_t has no invalid values (e.g., NaN from MCTS?)
        policy_target_t = torch.nan_to_num(policy_target_t, nan=0.0)
        # Element-wise cross-entropy: sum over action dimension
        policy_loss_elementwise = -torch.sum(policy_target_t * log_probs, dim=1)
        # Apply importance sampling weights (shape [batch_size])
        policy_loss = (policy_loss_elementwise * is_weights_t).mean()

        # Entropy Bonus (optional)
        entropy = 0.0
        entropy_loss = 0.0
        if self.train_config.ENTROPY_BONUS_WEIGHT > 0:
            policy_probs = F.softmax(policy_logits, dim=1)
            entropy_term = -torch.sum(
                policy_probs * torch.log(policy_probs + 1e-9), dim=1
            )
            entropy = entropy_term.mean().item()  # Log average entropy
            # Apply entropy bonus loss (negative entropy encourages exploration)
            # Should this loss term be weighted by IS weights? Typically not.
            entropy_loss = -self.train_config.ENTROPY_BONUS_WEIGHT * entropy_term.mean()

        # Total Loss
        total_loss = (
            self.train_config.POLICY_LOSS_WEIGHT * policy_loss
            + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            + entropy_loss
        )

        # --- Backpropagation & Optimization ---
        total_loss.backward()

        if (
            self.train_config.GRADIENT_CLIP_VALUE is not None
            and self.train_config.GRADIENT_CLIP_VALUE > 0
        ):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.GRADIENT_CLIP_VALUE
            )

        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        # --- Calculate TD Errors for PER Update ---
        # Use the element-wise value loss (absolute difference) as the TD error proxy
        td_errors = value_loss_elementwise.squeeze(1).detach().cpu().numpy()

        loss_info = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),  # Weighted policy loss
            "value_loss": value_loss.item(),  # Weighted value loss
            "entropy": entropy,
            "mean_td_error": np.mean(np.abs(td_errors)),  # Log mean absolute TD error
        }

        return loss_info, td_errors

    def get_current_lr(self) -> float:
        """Returns the current learning rate from the optimizer."""
        try:
            return self.optimizer.param_groups[0]["lr"]
        except (IndexError, KeyError):
            logger.warning("Could not retrieve learning rate from optimizer.")
            return 0.0
