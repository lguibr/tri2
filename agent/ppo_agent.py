# File: agent/ppo_agent.py
# (No changes needed in this file for this specific fix,
# the evaluate_actions method expects initial_hidden_state as a tuple (h,c)
# which it will now receive correctly from the updated RolloutStorage)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import traceback
from typing import Tuple, List, Dict, Any, Optional, Union

from config import (
    ModelConfig,
    EnvConfig,
    PPOConfig,
    RNNConfig,
    DEVICE,
    TensorBoardConfig,
    TOTAL_TRAINING_STEPS,
)
from environment.game_state import StateType
from utils.types import ActionType, AgentStateDict
from agent.model_factory import create_network
from agent.networks.agent_network import ActorCriticNetwork


class PPOAgent:
    """PPO Agent orchestrating network, action selection, and updates."""

    def __init__(
        self,
        model_config: ModelConfig,
        ppo_config: PPOConfig,
        rnn_config: RNNConfig,
        env_config: EnvConfig,
    ):
        print("[PPOAgent] Initializing...")
        self.device = DEVICE
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.rnn_config = rnn_config
        self.tb_config = TensorBoardConfig()
        self.action_dim = env_config.ACTION_DIM

        self.network = create_network(
            env_config=self.env_config,
            action_dim=self.action_dim,
            model_config=model_config,
            rnn_config=self.rnn_config,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=ppo_config.LEARNING_RATE,
            eps=ppo_config.ADAM_EPS,
        )

        self._print_init_info()

    def _print_init_info(self):
        print(f"[PPOAgent] Using Device: {self.device}")
        print(f"[PPOAgent] Network: {type(self.network).__name__}")
        print(f"[PPOAgent] Using RNN: {self.rnn_config.USE_RNN}")
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
        """
        Selects action for a SINGLE environment. Use select_action_batch for multiple.
        Returns action, log_prob, value, next_hidden.
        """
        self.network.eval()

        grid_np = state["grid"]
        shapes_np = state["shapes"]
        grid_t = torch.from_numpy(grid_np).float().unsqueeze(0).to(self.device)
        shapes_t = (
            torch.from_numpy(shapes_np.flatten()).float().unsqueeze(0).to(self.device)
        )

        if self.rnn_config.USE_RNN:
            grid_t = grid_t.unsqueeze(1)
            shapes_t = shapes_t.unsqueeze(1)
            if hidden_state:
                hidden_state = (hidden_state[0][:, 0:1, :], hidden_state[1][:, 0:1, :])

        policy_logits, value, next_hidden_state = self.network(
            grid_t, shapes_t, hidden_state
        )

        if self.rnn_config.USE_RNN:
            policy_logits = policy_logits.squeeze(1)
            value = value.squeeze(1)

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
            return 0, -1e9, value.squeeze().item(), next_hidden_state

        distribution = Categorical(logits=policy_logits)

        action_tensor = distribution.mode if deterministic else distribution.sample()
        action_log_prob = distribution.log_prob(action_tensor)
        action = action_tensor.item()

        if valid_actions_indices is not None and action not in valid_actions_indices:
            if valid_indices_in_bounds:
                action = np.random.choice(valid_indices_in_bounds)
                action_log_prob = distribution.log_prob(
                    torch.tensor(action, device=self.device)
                )
            else:
                action = 0
                action_log_prob = torch.tensor(-1e9, device=self.device)

        return action, action_log_prob.item(), value.squeeze().item(), next_hidden_state

    @torch.no_grad()
    def select_action_batch(
        self,
        grid_batch: torch.Tensor,
        shape_batch: torch.Tensor,
        hidden_state_batch: Optional[Tuple[torch.Tensor, torch.Tensor]],
        valid_actions_lists: List[Optional[List[ActionType]]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        self.network.eval()
        batch_size = grid_batch.shape[0]

        grid_batch = grid_batch.to(self.device)
        shape_batch = shape_batch.to(self.device)
        if hidden_state_batch:
            hidden_state_batch = (
                hidden_state_batch[0].to(self.device),
                hidden_state_batch[1].to(self.device),
            )

        if self.rnn_config.USE_RNN:
            grid_batch = grid_batch.unsqueeze(1)
            shape_batch = shape_batch.unsqueeze(1)

        policy_logits, value, next_hidden_batch = self.network(
            grid_batch, shape_batch, hidden_state_batch
        )

        if self.rnn_config.USE_RNN:
            policy_logits = policy_logits.squeeze(1)
            value = value.squeeze(1)

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

        return actions_tensor, action_log_probs, value.squeeze(-1), next_hidden_batch

    def evaluate_actions(
        self,
        grid_tensor: torch.Tensor,
        shape_tensor: torch.Tensor,
        actions: torch.Tensor,
        # MODIFIED: Renamed for clarity
        initial_lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dones_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.network.train()
        is_sequence = self.rnn_config.USE_RNN and grid_tensor.ndim == 5

        if is_sequence:
            batch_size = grid_tensor.shape[0]
            seq_len = grid_tensor.shape[1]

            # Use provided initial LSTM state or get default if None
            if initial_lstm_state is None:
                current_hidden_state = self.network.get_initial_hidden_state(batch_size)
            else:
                # Ensure the provided tuple components are on the correct device
                current_hidden_state = (
                    initial_lstm_state[0].to(self.device),
                    initial_lstm_state[1].to(self.device),
                )

            policy_logits, value, _ = self.network(
                grid_tensor, shape_tensor, current_hidden_state  # Pass tuple (h, c)
            )

            policy_logits = policy_logits.view(batch_size * seq_len, -1)
            value = value.view(batch_size * seq_len, -1)
            actions = actions.view(-1)

        else:
            policy_logits, value, _ = self.network(
                grid_tensor, shape_tensor, hidden_state=None
            )

        policy_logits = torch.nan_to_num(policy_logits, nan=-1e9)
        distribution = Categorical(logits=policy_logits)

        action_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        value = value.squeeze(-1)

        return action_log_probs, value, entropy

    def update(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        self.network.train()

        obs_grid_flat = rollout_data["obs_grid"]
        obs_shapes_flat = rollout_data["obs_shapes"]
        actions_flat = rollout_data["actions"]
        old_log_probs_flat = rollout_data["log_probs"]
        returns_flat = rollout_data["returns"]
        advantages_flat = rollout_data["advantages"]

        # MODIFIED: Get the initial LSTM state tuple (h, c)
        initial_lstm_state_tuple = rollout_data.get("initial_lstm_state", None)
        dones_seq = rollout_data.get("dones", None)

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        num_samples = actions_flat.shape[0]
        batch_size = self.ppo_config.MINIBATCH_SIZE
        num_envs = self.env_config.NUM_ENVS
        seq_len = self.ppo_config.NUM_STEPS_PER_ROLLOUT

        indices = np.arange(num_samples)

        for _ in range(self.ppo_config.PPO_EPOCHS):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = start_idx + batch_size
                minibatch_indices = indices[start_idx:end_idx]

                mb_obs_grid = obs_grid_flat[minibatch_indices]
                mb_obs_shapes = obs_shapes_flat[minibatch_indices]
                mb_actions = actions_flat[minibatch_indices]
                mb_old_log_probs = old_log_probs_flat[minibatch_indices]
                mb_returns = returns_flat[minibatch_indices]
                mb_advantages = advantages_flat[minibatch_indices]

                eval_grid = mb_obs_grid
                eval_shapes = mb_obs_shapes
                eval_actions = mb_actions
                # MODIFIED: Pass the initial LSTM state tuple for the whole sequence
                eval_hidden = initial_lstm_state_tuple
                eval_dones = dones_seq  # Pass sequence dones

                if self.rnn_config.USE_RNN:
                    # Reshape flat data to sequences for evaluation
                    # Assuming evaluate_actions can handle this reshaping internally for now,
                    # or that we pass the full sequence data. Let's assume we evaluate full sequence.
                    eval_grid = obs_grid_flat.view(
                        num_envs, seq_len, *obs_grid_flat.shape[1:]
                    )
                    eval_shapes = obs_shapes_flat.view(num_envs, seq_len, -1)
                    eval_actions = actions_flat.view(num_envs, seq_len)

                    new_log_probs_seq, predicted_values_seq, entropy_seq = (
                        self.evaluate_actions(
                            eval_grid,
                            eval_shapes,
                            eval_actions,
                            eval_hidden,
                            eval_dones,
                        )
                    )
                    # Get the minibatch portion from the full sequence evaluation
                    new_log_probs = new_log_probs_seq.view(-1)[minibatch_indices]
                    predicted_values = predicted_values_seq.view(-1)[minibatch_indices]
                    entropy = entropy_seq.view(-1)[minibatch_indices]

                else:  # Not using RNN
                    # Pass None for hidden state and dones
                    new_log_probs, predicted_values, entropy = self.evaluate_actions(
                        eval_grid, eval_shapes, eval_actions, None, None
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
                if self.ppo_config.MAX_GRAD_NORM > 0:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.ppo_config.MAX_GRAD_NORM
                    )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        avg_policy_loss = total_policy_loss / max(1, num_updates)
        avg_value_loss = total_value_loss / max(1, num_updates)
        avg_entropy = total_entropy / max(1, num_updates)

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
        }

    def get_state_dict(self) -> AgentStateDict:
        original_device = next(self.network.parameters()).device
        self.network.cpu()
        state = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        self.network.to(original_device)
        return state

    def load_state_dict(self, state_dict: AgentStateDict):
        print(f"[PPOAgent] Loading state dict. Target device: {self.device}")
        try:
            self.network.load_state_dict(state_dict["network_state_dict"])
            self.network.to(self.device)
            print("[PPOAgent] Network state loaded.")

            if "optimizer_state_dict" in state_dict:
                try:
                    self.optimizer = optim.AdamW(
                        self.network.parameters(),
                        lr=self.ppo_config.LEARNING_RATE,
                        eps=self.ppo_config.ADAM_EPS,
                    )
                    self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
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
                print(
                    "[PPOAgent] Optimizer state not found. Re-initializing optimizer."
                )
                self.optimizer = optim.AdamW(
                    self.network.parameters(),
                    lr=self.ppo_config.LEARNING_RATE,
                    eps=self.ppo_config.ADAM_EPS,
                )

            print("[PPOAgent] load_state_dict complete.")

        except Exception as e:
            print(f"CRITICAL ERROR during PPOAgent.load_state_dict: {e}")
            traceback.print_exc()

    def get_initial_hidden_state(
        self, num_envs: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.rnn_config.USE_RNN:
            return None
        # Returns tuple (h_0, c_0)
        return self.network.get_initial_hidden_state(num_envs)
