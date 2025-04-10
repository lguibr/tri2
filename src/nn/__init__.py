# File: src/nn/__init__.py
"""
Neural Network module for the AlphaTriangle agent.
Contains the model definition and a wrapper for inference and training interface.
"""
from .model import AlphaTriangleNet
from .network import NeuralNetwork

__all__ = [
    "AlphaTriangleNet",  # The PyTorch nn.Module class
    "NeuralNetwork",  # Wrapper class providing evaluate() and train() methods
]
