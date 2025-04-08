# File: workers/__init__.py
# This file makes the 'workers' directory a Python package.

from .self_play_worker import SelfPlayWorker
from .training_worker import TrainingWorker  # Added TrainingWorker

__all__ = ["SelfPlayWorker", "TrainingWorker"]
