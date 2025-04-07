# File: app_setup.py
import os
import pygame
from typing import Tuple, Dict, Any

from config import (
    VisConfig,
    get_run_checkpoint_dir,  # Use getter
    get_run_log_dir,  # Use getter
    get_config_dict,
    print_config_info_and_validate,
)


def initialize_pygame(
    vis_config: VisConfig,
) -> Tuple[pygame.Surface, pygame.time.Clock]:
    """Initializes Pygame, sets up the screen and clock."""
    print("Initializing Pygame...")
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (vis_config.SCREEN_WIDTH, vis_config.SCREEN_HEIGHT), pygame.RESIZABLE
    )
    pygame.display.set_caption("TriCrack PPO")
    clock = pygame.time.Clock()
    print("Pygame initialized.")
    return screen, clock


def initialize_directories():
    """Creates necessary runtime directories using dynamic paths."""
    run_checkpoint_dir = get_run_checkpoint_dir()
    run_log_dir = get_run_log_dir()
    # Note: Console log dir might be the same as run_log_dir,
    # but creating it separately ensures it exists if structure changes.
    # console_log_dir = get_console_log_dir() # Already created in main

    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    # os.makedirs(console_log_dir, exist_ok=True) # Already created in main
    print(f"Ensured directories exist: {run_checkpoint_dir}, {run_log_dir}")


def load_and_validate_configs() -> Dict[str, Any]:
    """Loads all config classes and returns the combined config dictionary."""
    config_dict = get_config_dict()
    print_config_info_and_validate()
    return config_dict
