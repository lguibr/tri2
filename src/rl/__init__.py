# File: src/rl/__init__.py
"""
Reinforcement Learning (RL) module.
Contains the core components for training an agent using self-play and MCTS.
"""

# Core RL classes
from .core.orchestrator import TrainingOrchestrator
from .core.trainer import Trainer
from .core.buffer import ExperienceBuffer

# Self-play functionality (now using Ray actor)
from .self_play.worker import SelfPlayWorker
# Import Pydantic model for result type hint from local types module
from .types import SelfPlayResult # Updated import

__all__ = [
    "TrainingOrchestrator",
    "Trainer",
    "ExperienceBuffer",
    "SelfPlayWorker",  # Export the actor class
    "SelfPlayResult",  # Export the Pydantic result type
]