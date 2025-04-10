# File: src/interaction/event_processor.py
# Change: Updated imports for visualization.
import pygame
import logging
from typing import TYPE_CHECKING, Generator, Any

# Import specific modules/classes needed
from src import visualization  # Import top-level vis module

if TYPE_CHECKING:
    # Use specific type hint from visualization.core
    from src.visualization.core.visualizer import Visualizer

logger = logging.getLogger(__name__)


def process_pygame_events(
    visualizer: "Visualizer",
) -> Generator[pygame.event.Event, Any, bool]:
    """
    Processes basic Pygame events like QUIT, ESCAPE, VIDEORESIZE.
    Yields other events for mode-specific handlers.
    Returns False via StopIteration value if the application should quit, True otherwise.
    """
    should_quit = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            logger.info("Received QUIT event.")
            should_quit = True
            break
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            logger.info("Received ESCAPE key press.")
            should_quit = True
            break
        if event.type == pygame.VIDEORESIZE:
            try:
                w, h = max(320, event.w), max(240, event.h)
                # Update screen directly on the visualizer instance
                visualizer.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                visualizer.layout_rects = (
                    None  # Force layout recalculation on visualizer
                )
                logger.info(f"Window resized to {w}x{h}")
            except pygame.error as e:
                logger.error(f"Error resizing window: {e}")
            yield event  # Yield resize event for potential further handling if needed
        else:
            yield event  # Yield other events

    return not should_quit  # Return True to continue, False to quit
