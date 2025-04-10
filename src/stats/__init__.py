# File: src/stats/__init__.py
"""
Statistics collection and plotting module.
"""
# Removed: from .collector import StatsCollector
from .collector import StatsCollectorActor # Import the new actor
from .plotter import Plotter
from . import plot_utils  # Expose plot_utils if needed externally
from src.utils.types import StatsCollectorData  # Import type alias

__all__ = [
    # Removed: "StatsCollector",
    "StatsCollectorActor", # Export the actor
    "StatsCollectorData",  # Export type alias
    "Plotter",
    "plot_utils",
]
