# File: src/environment/__init__.py
"""
Environment module defining the game rules, state, actions, and logic.
This module is now independent of feature extraction for the NN.
"""
# Core components
from .core.game_state import GameState
from .core.action_codec import encode_action, decode_action

# Grid related components
from .grid.grid_data import GridData

# Removed: from .grid.triangle import Triangle
from .grid import logic as GridLogic  # Expose grid logic functions via a namespace

# Shape related components
# Removed: from .shapes.shape import Shape
from .shapes import logic as ShapeLogic  # Expose shape logic functions via a namespace

# Game Logic components (Actions, Step)
from .logic.actions import get_valid_actions
from .logic.step import execute_placement, calculate_reward

# Configuration (often needed alongside environment components)
from src.config import EnvConfig


__all__ = [
    # Core
    "GameState",
    "encode_action",
    "decode_action",
    # Grid
    "GridData",
    # "Triangle", # Removed
    "GridLogic",
    # Shapes
    # "Shape", # Removed
    "ShapeLogic",
    # Logic
    "get_valid_actions",
    "execute_placement",
    "calculate_reward",
    # Config
    "EnvConfig",
]
