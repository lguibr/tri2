# File: app_setup.py
import os
import pygame
from typing import Tuple, Dict, Any

from config import (
    VisConfig,
    EnvConfig,
    DQNConfig,
    TrainConfig,
    BufferConfig,
    ModelConfig,
    StatsConfig,
    RewardConfig,
    TensorBoardConfig,
    DemoConfig,
    RUN_CHECKPOINT_DIR,
    RUN_LOG_DIR,
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
    pygame.display.set_caption("TriCrack DQN")
    clock = pygame.time.Clock()
    print("Pygame initialized.")
    return screen, clock


def initialize_directories():
    """Creates necessary runtime directories."""
    os.makedirs(RUN_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    print(f"Ensured directories exist: {RUN_CHECKPOINT_DIR}, {RUN_LOG_DIR}")


def load_and_validate_configs() -> Dict[str, Any]:
    """Loads all config classes and returns the combined config dictionary."""
    # Instantiating config classes implicitly loads defaults
    # The get_config_dict function retrieves their values
    config_dict = get_config_dict()
    print_config_info_and_validate()
    return config_dict
