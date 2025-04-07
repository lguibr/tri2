# File: app_setup.py
import os
import pygame
from typing import Tuple, Dict, Any

from config import (
    VisConfig,
    get_run_checkpoint_dir,
    get_run_log_dir,
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
    # Console log dir is created within main_pygame before logger init

    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"Ensured directories exist: {run_checkpoint_dir}, {run_log_dir}")


def load_and_validate_configs() -> Dict[str, Any]:
    """Loads all config classes and returns the combined config dictionary."""
    config_dict = get_config_dict()
    print_config_info_and_validate()
    return config_dict
