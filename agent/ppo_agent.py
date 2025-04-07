# File: agent/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import traceback
import threading
from typing import Tuple, List, Dict, Any, Optional, Union

from config import (
    ModelConfig,
    EnvConfig,
    PPOConfig,
    RNNConfig,
    TransformerConfig,
    TensorBoardConfig,
)
from environment.game_state import StateType  # Keep StateType for single action method
from utils.types import ActionType, AgentStateDict
from agent.model_factory import create_network


class PPOAgent:
    """
    PPO Agent orchestrating network, action selection, and updates.
    Assumes observations received are ALREADY NORMALIZED if ObsNorm is enabled.
    Includes locks for thread safety.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        ppo_config: PPOConfig,
        rnn_config: RNNConfig,
        env_config: EnvConfig,
        transformer_config: TransformerConfig,
        device: torch.device,
    ):
        self.device = device
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.transformer_config = transformer_config
        self.tb_config = TensorBoardConfig()
        self.action_dim = env_config.ACTION_DIM
        self._lock = threading.Lock()

        self.network = create_network(
            env_config=self.env_config,
            action_dim=self.action_dim,
            model_config=model_config,
            rnn_config=self.rnn_config,
            transformer_config=self.transformer_config,
            device=self.device,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=ppo_config.LEARNING_RATE,
            eps=ppo_config.ADAM_EPS,
        )
        self._print_init_info()

    def _print_init_info(self):
        """Logs basic agent configuration on initialization."""
        print(f"[PPOAgent] Using Device: {self.device}")
        print(f"[PPOAgent] Network: {type(self.network).__name__}")
        print(f"[PPOAgent] Using RNN: {self.rnn_config.USE_RNN}")
        print(
            f"[PPOAgent] Using Transformer: {self.transformer_config.USE_TRANSFORMER}"
        )
        total_params = sum(
            p.numel() for p in self.network.parameters() if p.requires_grad
        )
        print(f"[PPOAgent] Trainable Parameters: {total_params / 1e6:.2f} M")

    @torch.no_grad()
    def select_action(
        self,
        state: StateType,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
        valid_actions_indices: Optional[List[ActionType]] = None,
    ) -> Tuple[ActionType, float, float, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Selects an action based on the (potentially normalized) state. (Less used now)"""
        with self._lock:
            self.network.eval()
            grid_t = (
                torch.from_numpy(state["grid"]).float().unsqueeze(0).to(self.device)
            )
            shapes_t = (
                torch.from_numpy(state["shapes"]).float().unsqueeze(0).to(self.device)
            )
            availability_t = (
                torch.from_numpy(state["shape_availability"])
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
            explicit_features_t = (
                torch.from_numpy(state["explicit_features"])
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

            needs_sequence_dim = (
                self.rnn_config.USE_RNN or self.transformer_config.USE_TRANSFORMER
            )
            if needs_sequence_dim:
                grid_t, shapes_t, availability_t, explicit_features_t = (
                    t.unsqueeze(1)
                    for t in [grid_t, shapes_t, availability_t, explicit_features_t]
                )
                if hidden_state:
                    hidden_state = (
                        hidden_state[0][:, 0:1, :].contiguous(),
                        hidden_state[1][:, 0:1, :].contiguous(),
                    )

            policy_logits, value, next_hidden_state = self.network(
                grid_t,
                shapes_t,
                availability_t,
                explicit_features_t,
                hidden_state,
                padding_mask=None,
            )

            if needs_sequence_dim:
                policy_logits, value = policy_logits.squeeze(1), value.squeeze(1)
            policy_logits = torch.nan_to_num(policy_logits.squeeze(0), nan=-1e9)

            if valid_actions_indices is not None:
                mask = torch.full_like(policy_logits, -float("inf"))
                valid_indices_in_bounds = [
                    idx
                    for idx in valid_actions_indices
                    if 0 <= idx < policy_logits.shape[0]
                ]
                if valid_indices_in_bounds:
                    mask[valid_indices_in_bounds] = 0
                    policy_logits += mask

            if torch.all(policy_logits == -float("inf")):
                action, action_log_prob = 0, torch.tensor(-1e9, device=self.device)
            else:
                distribution = Categorical(logits=policy_logits)
                action_tensor = (
                    distribution.mode if deterministic else distribution.sample()
                )
                action_log_prob = distribution.log_prob(action_tensor)
                action = action_tensor.item()
                if (
                    valid_actions_indices is not None
                    and action not in valid_actions_indices
                ):
                    if valid_indices_in_bounds:
                        action = np.random.choice(valid_indices_in_bounds)
                        action_log_prob = distribution.log_prob(
                            torch.tensor(action, device=self.device)
                        )
                    else:
                        action, action_log_prob = 0, torch.tensor(
                            -1e9, device=self.device
                        )

            return (
                action,
                action_log_prob.item(),
                value.squeeze().item(),
                next_hidden_state,
            )

    @torch.no_grad()
    def select_action_batch(
        self,
        grid_batch: torch.Tensor,
        shape_batch: torch.Tensor,
        availability_batch: torch.Tensor,
        explicit_features_batch: torch.Tensor,
        hidden_state_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
        valid_actions_lists: List[Optional[List[ActionType]]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Selects actions for a batch of (potentially normalized) states. Thread-safe."""
        with self._lock:
            self.network.eval()
            batch_size = grid_batch.shape[0]

            needs_sequence_dim = (
                self.rnn_config.USE_RNN or self.transformer_config.USE_TRANSFORMER
            )
            if needs_sequence_dim:
                grid_batch, shape_batch, availability_batch, explicit_features_batch = (
                    t.unsqueeze(1)
                    for t in [
                        grid_batch,
                        shape_batch,
                        availability_batch,
                        explicit_features_batch,
                    ]
                )

            policy_logits, value, next_hidden_batch = self.network(
                grid_batch,
                shape_batch,
                availability_batch,
                explicit_features_batch,
                hidden_state_batch,
                padding_mask=None,
            )

            if needs_sequence_dim:
                policy_logits, value = policy_logits.squeeze(1), value.squeeze(1)
            policy_logits = torch.nan_to_num(policy_logits, nan=-1e9)

            mask = torch.full_like(policy_logits, -float("inf"))
            any_valid = False
            for i in range(batch_size):
                valid_actions = valid_actions_lists[i]
                if valid_actions:
                    valid_indices_in_bounds = [
                        idx for idx in valid_actions if 0 <= idx < self.action_dim
                    ]
                    if valid_indices_in_bounds:
                        mask[i, valid_indices_in_bounds] = 0
                        any_valid = True

            if any_valid:
                policy_logits += mask
            all_masked_rows = torch.all(policy_logits == -float("inf"), dim=1)
            policy_logits[all_masked_rows] = 0.0

            distribution = Categorical(logits=policy_logits)
            actions_tensor = distribution.sample()
            action_log_probs = distribution.log_prob(actions_tensor)
            actions_tensor[all_masked_rows] = 0
            action_log_probs[all_masked_rows] = -1e9
            value = value.squeeze(-1) if value.ndim > 1 else value

            return actions_tensor, action_log_probs, value, next_hidden_batch

    def evaluate_actions(
        self,
        grid_tensor: torch.Tensor,
        shape_feature_tensor: torch.Tensor,
        shape_availability_tensor: torch.Tensor,
        explicit_features_tensor: torch.Tensor,
        actions: torch.Tensor,
        initial_lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dones_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates actions for loss calculation. Assumes called within update lock."""
        self.network.train()
        padding_mask = None
        policy_logits, value, _ = self.network(
            grid_tensor,
            shape_feature_tensor,
            shape_availability_tensor,
            explicit_features_tensor,
            initial_lstm_state,
            padding_mask=padding_mask,
        )
        policy_logits = torch.nan_to_num(policy_logits, nan=-1e9)
        distribution = Categorical(logits=policy_logits)
        action_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        value = value.squeeze(-1)
        return action_log_probs, value, entropy

    def update_minibatch(
        self, minibatch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Performs a PPO update step on a SINGLE minibatch. Thread-safe."""
        with self._lock:
            self.network.train()
            mb_obs_grid = minibatch_data["obs_grid"]
            mb_obs_shapes = minibatch_data["obs_shapes"]
            mb_obs_availability = minibatch_data["obs_availability"]
            mb_obs_explicit_features = minibatch_data["obs_explicit_features"]
            mb_actions = minibatch_data["actions"]
            mb_old_log_probs = minibatch_data["log_probs"]
            mb_returns = minibatch_data["returns"]
            mb_advantages = minibatch_data["advantages"]

            new_log_probs, predicted_values, entropy = self.evaluate_actions(
                mb_obs_grid,
                mb_obs_shapes,
                mb_obs_availability,
                mb_obs_explicit_features,
                mb_actions,
                None,
                None,
            )

            logratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(logratio)
            surr1 = ratio * mb_advantages
            surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - self.ppo_config.CLIP_PARAM,
                    1.0 + self.ppo_config.CLIP_PARAM,
                )
                * mb_advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(predicted_values, mb_returns)
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + self.ppo_config.VALUE_LOSS_COEF * value_loss
                + self.ppo_config.ENTROPY_COEF * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm_val = None
            if self.ppo_config.MAX_GRAD_NORM > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.ppo_config.MAX_GRAD_NORM
                )
                grad_norm_val = grad_norm.item()
            self.optimizer.step()

            metrics = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": -entropy_loss.item(),
            }
            if grad_norm_val is not None:
                metrics["grad_norm"] = grad_norm_val
            return metrics

    def get_state_dict(self) -> AgentStateDict:
        """Returns the agent's state dictionary for checkpointing. Thread-safe."""
        with self._lock:
            original_device = next(self.network.parameters()).device
            self.network.cpu()
            state = {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            self.network.to(original_device)
            return state

    def load_state_dict(self, state_dict: AgentStateDict):
        """Loads the agent's state from a dictionary. Thread-safe."""
        with self._lock:
            print(f"[PPOAgent] Loading state dict. Target device: {self.device}")
            network_loaded_successfully = False
            try:
                # Attempt to load network state with strict=False first
                try:
                    self.network.load_state_dict(
                        state_dict["network_state_dict"], strict=False
                    )
                    print(
                        "[PPOAgent] Network state loaded (strict=False). Mismatched/missing keys ignored."
                    )
                    network_loaded_successfully = True
                except RuntimeError as e:
                    # If strict=False still fails, it's likely a more fundamental issue
                    print(
                        f"Warning: Network load failed even with strict=False: {e}. Network will remain initialized."
                    )
                    # Optionally, re-initialize the network completely here if desired
                    # self.network = create_network(...)
                except KeyError:
                    print(
                        "Warning: 'network_state_dict' key missing in checkpoint. Network will remain initialized."
                    )
                except Exception as e:
                    print(
                        f"Warning: Unexpected error loading network state: {e}. Network will remain initialized."
                    )
                    traceback.print_exc()

                self.network.to(self.device)

                # Always try to load optimizer state, but re-initialize if it fails or network load failed
                if "optimizer_state_dict" in state_dict and network_loaded_successfully:
                    try:
                        # Re-create optimizer linked to the potentially partially loaded network
                        self.optimizer = optim.AdamW(
                            self.network.parameters(),
                            lr=self.ppo_config.LEARNING_RATE,
                            eps=self.ppo_config.ADAM_EPS,
                        )
                        self.optimizer.load_state_dict(
                            state_dict["optimizer_state_dict"]
                        )
                        # Move optimizer state to the correct device
                        for state in self.optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(self.device)
                        print("[PPOAgent] Optimizer state loaded and moved to device.")
                    except Exception as e:
                        print(
                            f"Warning: Could not load optimizer state ({e}). Re-initializing optimizer."
                        )
                        self.optimizer = optim.AdamW(
                            self.network.parameters(),
                            lr=self.ppo_config.LEARNING_RATE,
                            eps=self.ppo_config.ADAM_EPS,
                        )
                else:
                    if not network_loaded_successfully:
                        print(
                            "[PPOAgent] Network load failed or incomplete, re-initializing optimizer."
                        )
                    else:
                        print(
                            "[PPOAgent] Optimizer state not found in checkpoint. Re-initializing optimizer."
                        )
                    self.optimizer = optim.AdamW(
                        self.network.parameters(),
                        lr=self.ppo_config.LEARNING_RATE,
                        eps=self.ppo_config.ADAM_EPS,
                    )
                print("[PPOAgent] load_state_dict finished.")
            except Exception as e:
                print(f"CRITICAL ERROR during PPOAgent.load_state_dict: {e}")
                traceback.print_exc()
                # Ensure optimizer is re-initialized on critical failure
                print(
                    "[PPOAgent] Re-initializing optimizer due to critical load error."
                )
                self.optimizer = optim.AdamW(
                    self.network.parameters(),
                    lr=self.ppo_config.LEARNING_RATE,
                    eps=self.ppo_config.ADAM_EPS,
                )

    def get_initial_hidden_state(
        self, num_envs: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Gets the initial hidden state for the LSTM, if used. Thread-safe."""
        if not self.rnn_config.USE_RNN:
            return None
        # Ensure network is on the correct device before calling its method
        # self.network.to(self.device) # <<< REMOVED THIS LINE
        # Call the network's method directly (network should already be on self.device)
        return self.network.get_initial_hidden_state(num_envs)
