# File: agent/networks/noisy_layer.py
# (No changes needed, already clean)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for Noisy Network (Factorised Gaussian Noise).
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable weights and biases (mean parameters)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learnable noise parameters (standard deviation parameters)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Non-learnable noise buffers
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()  # Initial noise generation

    def reset_parameters(self):
        """Initialize mean and std parameters."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)

        # Initialize sigma parameters (std dev)
        nn.init.constant_(
            self.weight_sigma, self.std_init / math.sqrt(self.in_features)
        )
        nn.init.constant_(self.bias_sigma, self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Generate new noise samples using Factorised Gaussian noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Outer product for weight noise, direct sample for bias noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate noise tensor with sign-sqrt transformation."""
        x = torch.randn(size, device=self.weight_mu.device)  # Noise on same device
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters if training, mean parameters otherwise."""
        if self.training:
            # Sample noise is implicitly used via weight_epsilon, bias_epsilon
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            # Reset noise *after* use in forward pass for next iteration?
            # Or reset in train() method? Resetting in train() is common.
        else:
            # Use mean parameters during evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def train(self, mode: bool = True):
        """Override train mode to reset noise when entering training."""
        if self.training is False and mode is True:  # If switching from eval to train
            self.reset_noise()
        super().train(mode)
