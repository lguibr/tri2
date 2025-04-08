from enum import Enum


class AppState(Enum):
    INITIALIZING = "Initializing"
    MAIN_MENU = "MainMenu"
    PLAYING = "Playing" 
    DEBUG = "Debug"
    CLEANUP_CONFIRM = "Confirm Cleanup"
    CLEANING = "Cleaning"
    ERROR = "Error"
    UNKNOWN = "Unknown"
