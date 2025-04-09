from .stats_recorder import StatsRecorderBase

# from .aggregator import StatsAggregator # Original class name removed/renamed
from .aggregator import StatsAggregatorActor  # Import the Actor class
from .simple_stats_recorder import SimpleStatsRecorder


__all__ = [
    "StatsRecorderBase",
    # "StatsAggregator", # Remove old export
    "StatsAggregatorActor",  # Export the Actor class
    "SimpleStatsRecorder",
]
