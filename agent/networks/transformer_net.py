# File: agent/networks/transformer_net.py
# <<< No changes needed based on request, assuming implementation is correct >>>
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import ModelConfig
from typing import Optional


class PosEnc(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_encoding = self.pe[: x.size(1)].transpose(0, 1)
        x = x + pos_encoding
        return self.dropout(x)


class TransformerNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # <<< MODIFIED >>> Type hint specific Transformer config
        config: ModelConfig.Transformer,
        dueling: bool,
    ):
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim
        self.d_model = config.HDIM

        # Assuming state_dim is the flattened sequence length, need to infer structure
        # Option 1: Assume fixed feature dim = 1, seq_len = state_dim
        # Option 2: Define feature dim explicitly?
        # Let's stick with Option 1 for now as per original structure
        self.seq_len = state_dim
        self.input_feature_dim = 1
        self.emb = nn.Linear(self.input_feature_dim, self.d_model)

        self.pos_encoder = PosEnc(
            self.d_model, config.DROPOUT, max_len=self.seq_len + 10
        )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.HEADS,
            dim_feedforward=self.d_model * 4,
            dropout=config.DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        encoder_norm = nn.LayerNorm(self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, config.LAYERS, norm=encoder_norm
        )

        fc_hidden_dim = self.d_model // 2

        if self.dueling:
            self.value_head = nn.Sequential(
                nn.Linear(self.d_model, fc_hidden_dim),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(self.d_model, fc_hidden_dim),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, action_dim),
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(self.d_model, fc_hidden_dim),
                nn.ReLU(),
                nn.Linear(fc_hidden_dim, action_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: [batch_size, state_dim]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Reshape to [B, seq_len, features=1]

        # Ensure sequence length matches expected (handle potential mismatches if necessary)
        current_seq_len = x.size(1)
        if current_seq_len != self.seq_len:
            # This indicates a mismatch between GameState.get_state() and EnvConfig.STATE_DIM
            # It's better to fix the source than pad/truncate here, as padding might harm learning.
            # For robustness, let's pad/truncate but print a clear warning.
            print(
                f"CRITICAL WARNING [TransformerNet]: Input seq len {current_seq_len} != expected {self.seq_len}. Check state generation/config."
            )
            if current_seq_len < self.seq_len:
                pad_size = self.seq_len - current_seq_len
                x = F.pad(x, (0, 0, 0, pad_size))  # Pad seq dim (dim 1)
            else:
                x = x[:, : self.seq_len, :]  # Truncate seq dim

        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        # TODO: Add padding mask if using variable length sequences
        memory = self.transformer_encoder(x)
        pooled_output = memory.mean(dim=1)  # Mean pooling

        if self.dueling:
            value = self.value_head(pooled_output)
            advantage = self.advantage_head(pooled_output)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_head(pooled_output)

        return q_values
