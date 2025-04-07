# File: app_state.py
from enum import Enum, auto


class AppState(Enum):
    INITIALIZING = "Initializing"
    MAIN_MENU = "MainMenu"
    PLAYING = "Playing"  # Demo Mode
    DEBUG = "Debug"
    CLEANUP_CONFIRM = "Confirm Cleanup"  # Not used directly, handled by flag
    CLEANING = "Cleaning"  # Intermediate state during cleanup
    ERROR = "Error"
    UNKNOWN = "Unknown"  # Fallback
