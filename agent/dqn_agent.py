# File: agent/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import traceback
import copy
from typing import Tuple, List, Dict, Any, Optional, Union

from config import (
    ModelConfig,
    EnvConfig,
    DQNConfig,
    DEVICE,
    TensorBoardConfig,
    BufferConfig,
)
from environment.game_state import StateType
from utils.types import ActionType, AgentStateDict, NumpyBatch, NumpyNStepBatch
from agent.model_factory import create_network
from agent.networks.noisy_layer import NoisyLinear
from .action_selector import ActionSelector
from .loss_calculator import LossCalculator


class DQNAgent:
    """DQN Agent orchestrating network, action selection, and loss calculation."""

    def __init__(
        self,
        config: ModelConfig,
        dqn_config: DQNConfig,
        env_config: EnvConfig,
        buffer_config: BufferConfig,
    ):
        print("[DQNAgent] Initializing...")
        self.device = DEVICE
        self.env_config = env_config
        self.dqn_config = dqn_config
        self.tb_config = TensorBoardConfig()
        self.buffer_config = buffer_config
        self.action_dim = env_config.ACTION_DIM

        self.online_net = create_network(
            env_config=self.env_config,
            action_dim=self.action_dim,
            model_config=config,
            dqn_config=dqn_config,
        ).to(self.device)
        self.target_net = create_network(
            env_config=self.env_config,
            action_dim=self.action_dim,
            model_config=config,
            dqn_config=dqn_config,
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.online_net.parameters(),
            lr=dqn_config.LEARNING_RATE,
            eps=dqn_config.ADAM_EPS,
            weight_decay=1e-5,
        )
        self.scheduler = self._initialize_scheduler(dqn_config)

        self.action_selector = ActionSelector(
            self.online_net, self.env_config, self.dqn_config, self.device
        )
        self.loss_calculator = LossCalculator(
            self.online_net,
            self.target_net,
            self.env_config,
            self.dqn_config,
            self.tb_config,
            self.device,
        )

        self._print_init_info(dqn_config)

    def _initialize_scheduler(
        self, dqn_config: DQNConfig
    ) -> Optional[CosineAnnealingLR]:
        if dqn_config.USE_LR_SCHEDULER:
            print(
                f"[DQNAgent] Using CosineAnnealingLR scheduler (T_max={dqn_config.LR_SCHEDULER_T_MAX}, eta_min={dqn_config.LR_SCHEDULER_ETA_MIN})"
            )
            return CosineAnnealingLR(
                self.optimizer,
                T_max=dqn_config.LR_SCHEDULER_T_MAX,
                eta_min=dqn_config.LR_SCHEDULER_ETA_MIN,
            )
        return None

    def _print_init_info(self, dqn_config: DQNConfig):
        print(f"[DQNAgent] Using Device: {self.device}")
        print(f"[DQNAgent] Online Network: {type(self.online_net).__name__}")
        print(f"[DQNAgent] Using Double DQN: {dqn_config.USE_DOUBLE_DQN}")
        print(f"[DQNAgent] Using Dueling: {dqn_config.USE_DUELING}")
        print(f"[DQNAgent] Using Noisy Nets: {dqn_config.USE_NOISY_NETS}")
        print(f"[DQNAgent] Using Distributional (C51): {dqn_config.USE_DISTRIBUTIONAL}")
        total_params = sum(
            p.numel() for p in self.online_net.parameters() if p.requires_grad
        )
        print(f"[DQNAgent] Trainable Parameters: {total_params / 1e6:.2f} M")

    def select_action(
        self, state: StateType, epsilon: float, valid_actions_indices: List[ActionType]
    ) -> ActionType:
        self.online_net.to(self.device)
        # --- Epsilon is ignored by ActionSelector if USE_NOISY_NETS is true ---
        return self.action_selector.select_action(state, epsilon, valid_actions_indices)

    def compute_loss(
        self,
        batch: Union[NumpyBatch, NumpyNStepBatch],
        is_weights: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        is_n_step = self.buffer_config.USE_N_STEP and self.buffer_config.N_STEP > 1
        # --- Set networks to train() for Noisy Nets before loss calculation ---
        if self.dqn_config.USE_NOISY_NETS:
            self.online_net.train()
            self.target_net.train()  # Target also uses noise during loss calculation
        else:
            self.online_net.train()
            self.target_net.eval()  # Standard eval mode for target net
        loss, td_errors = self.loss_calculator.compute_loss(
            batch, is_n_step, is_weights
        )
        # --- Restore target net eval mode if not noisy ---
        if not self.dqn_config.USE_NOISY_NETS:
            self.target_net.eval()
        return loss, td_errors

    def update(self, loss: torch.Tensor) -> Optional[float]:
        grad_norm = None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.online_net.train()  # Keep online net in train mode

        if self.dqn_config.GRADIENT_CLIP_NORM > 0:
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.online_net.parameters(),
                    max_norm=self.dqn_config.GRADIENT_CLIP_NORM,
                    error_if_nonfinite=True,
                ).item()
            except RuntimeError as clip_err:
                print(
                    f"ERROR: Gradient clipping failed: {clip_err}. Skipping optimizer step."
                )
                self.optimizer.zero_grad(set_to_none=True)
                return None
            except Exception as clip_err:
                print(f"Warning: Unexpected error during gradient clipping: {clip_err}")
                grad_norm = None

        try:
            self.optimizer.step()
        except Exception as optim_err:
            print(f"ERROR: Optimizer step failed: {optim_err}")
            traceback.print_exc()
            return None

        if self.scheduler:
            self.scheduler.step()

        # --- Reset noise after optimizer step ---
        if self.dqn_config.USE_NOISY_NETS:
            self.online_net.reset_noise()
            self.target_net.reset_noise()
        # --- END Reset Noise ---

        return grad_norm

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Standard eval mode after update

    def get_state_dict(self) -> AgentStateDict:
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        self.online_net.cpu()
        self.target_net.cpu()

        cpu_optimizer = optim.AdamW(
            self.online_net.parameters(),
            lr=self.dqn_config.LEARNING_RATE,
            eps=self.dqn_config.ADAM_EPS,
            weight_decay=1e-5,
        )
        cpu_optimizer.load_state_dict(self.optimizer.state_dict())
        optimizer_state_dict_cpu = cpu_optimizer.state_dict()
        del cpu_optimizer

        state = {
            "online_net_state_dict": self.online_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": optimizer_state_dict_cpu,
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
        }

        self.online_net.to(self.device)
        self.target_net.to(self.device)
        self.online_net.train()
        self.target_net.eval()

        return state

    def load_state_dict(self, state_dict: AgentStateDict):
        print(f"[DQNAgent] Loading state dict. Target device: {self.device}")
        try:
            self.online_net.load_state_dict(
                state_dict["online_net_state_dict"], strict=False
            )
            if "target_net_state_dict" in state_dict:
                self.target_net.load_state_dict(
                    state_dict["target_net_state_dict"], strict=False
                )
            else:
                print("Warning: Target net state missing, copying from online.")
                self.target_net.load_state_dict(self.online_net.state_dict())

            self.online_net.to(self.device)
            self.target_net.to(self.device)

            if "optimizer_state_dict" in state_dict:
                self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                print("[DQNAgent] Optimizer state loaded.")
            else:
                print("Warning: Optimizer state not found. Resetting optimizer.")
                self._reset_optimizer()

            if (
                self.scheduler
                and "scheduler_state_dict" in state_dict
                and state_dict["scheduler_state_dict"] is not None
            ):
                try:
                    self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
                    print("[DQNAgent] Scheduler state loaded.")
                except Exception as e:
                    print(
                        f"Warning: Could not load LR scheduler state ({e}). Resetting."
                    )
                    self.scheduler = self._initialize_scheduler(self.dqn_config)
            elif self.scheduler:
                print("Warning: LR Scheduler state not found. Resetting.")
                self.scheduler = self._initialize_scheduler(self.dqn_config)

            self.online_net.train()
            self.target_net.eval()
            print("[DQNAgent] load_state_dict complete.")

        except Exception as e:
            print(f"CRITICAL ERROR during DQNAgent.load_state_dict: {e}")
            traceback.print_exc()
            print("Attempting to reset agent state after load failure.")
            self._reset_optimizer()
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.online_net.train()
            self.target_net.eval()

    def _reset_optimizer(self):
        self.optimizer = optim.AdamW(
            self.online_net.parameters(),
            lr=self.dqn_config.LEARNING_RATE,
            eps=self.dqn_config.ADAM_EPS,
            weight_decay=1e-5,
        )

    def get_last_avg_max_q(self) -> float:
        return self.loss_calculator.get_last_avg_max_q()

    def get_last_shape_selection_info(
        self,
    ) -> Tuple[Optional[int], Optional[List[float]], Optional[List[float]]]:
        return self.action_selector.get_last_shape_selection_info()

    def get_last_batch_q_values_for_actions(self) -> Optional[np.ndarray]:
        return self.loss_calculator.get_last_batch_q_values_for_actions()
