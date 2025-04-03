import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig
from typing import Tuple


class LSTMNet(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, config: ModelConfig.LSTM, dueling: bool
    ):
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        self.hidden_dim = config.HIDDEN_DIM
        self.num_layers = config.NUM_LAYERS

        # Input embedding (treat state_dim as seq_len, feature_dim=1)
        self.emb = nn.Linear(1, self.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Output heads
        fc_hidden_dim = config.FC_DIM
        if self.dueling:
            self.value_head = nn.Sequential(
                nn.Linear(self.hidden_dim, fc_hidden_dim),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(self.hidden_dim, fc_hidden_dim),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, action_dim),
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(self.hidden_dim, fc_hidden_dim),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, action_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: [B, state_dim]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, state_dim, 1]

        # Embedding
        x = F.relu(self.emb(x))  # [B, state_dim, hidden_dim]

        # LSTM
        # Output: (batch, seq_len, hidden_dim)
        # hn: (num_layers, batch, hidden_dim)
        # cn: (num_layers, batch, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x)

        # Use the hidden state of the last layer from the last time step
        # hn is shape [num_layers, batch, hidden_dim], get last layer -> hn[-1]
        last_hidden = hn[-1]  # Shape: [B, hidden_dim]

        # Output
        if self.dueling:
            value = self.value_head(last_hidden)
            advantage = self.advantage_head(last_hidden)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_head(last_hidden)
        return q_values
