# File: app_state.py
from enum import Enum, auto


class AppState(Enum):
    INITIALIZING = "Initializing"
    MAIN_MENU = "MainMenu"
    PLAYING = "Playing"  # Demo Mode
    DEBUG = "Debug"
    MCTS_VISUALIZE = "MCTS Visualize"  # New state for MCTS view
    CLEANUP_CONFIRM = "Confirm Cleanup"
    CLEANING = "Cleaning"
    ERROR = "Error"
    UNKNOWN = "Unknown"
