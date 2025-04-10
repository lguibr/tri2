# File: src/interaction/input_handler.py
# Change: Updated imports for environment, visualization.
import pygame
import logging
from typing import TYPE_CHECKING

# Use relative imports within interaction module
from . import event_processor, play_mode_handler, debug_mode_handler

# Import specific modules/classes needed from other packages
from src import environment
from src import visualization

logger = logging.getLogger(__name__)


class InputHandler:
    """Handles user input, delegating to mode-specific handlers."""

    def __init__(
        self,
        game_state: environment.GameState,  # Use specific type hint
        visualizer: visualization.Visualizer,  # Use specific type hint
        mode: str,
        env_config: environment.EnvConfig,  # Use specific type hint
    ):
        self.game_state = game_state
        self.visualizer = visualizer
        self.mode = mode
        self.env_config = env_config

    def handle_input(self) -> bool:
        """Processes Pygame events and updates state based on mode. Returns False to quit."""
        mouse_pos = pygame.mouse.get_pos()

        # Reset hover states before processing events
        if self.mode == "debug":
            self.game_state.debug_highlight_pos = None
        # Reset snapped position before hover update
        self.game_state.demo_snapped_position = None

        running = True
        # Use the generator to process events
        # Pass the visualizer instance for resize handling
        event_generator = event_processor.process_pygame_events(self.visualizer)
        try:
            while True:
                event = next(event_generator)  # Get next event yielded by processor
                # Pass yielded events to mode-specific handlers
                if self.mode == "play":
                    # Pass necessary components to the handler function
                    play_mode_handler.handle_play_click(
                        event, mouse_pos, self.game_state, self.visualizer
                    )
                elif self.mode == "debug":
                    debug_mode_handler.handle_debug_click(
                        event, mouse_pos, self.game_state, self.visualizer
                    )
        except StopIteration as e:
            # Generator finished, return value indicates if we should continue
            running = e.value

        # Update hover effects after processing all events for this frame
        if running:  # Only update hover if not quitting
            if self.mode == "play":
                # Pass necessary components to the hover update function
                play_mode_handler.update_play_hover(
                    mouse_pos, self.game_state, self.visualizer
                )
            elif self.mode == "debug":
                debug_mode_handler.update_debug_hover(
                    mouse_pos, self.game_state, self.visualizer
                )
            # No else needed, snapped_position was reset earlier

        return running  # Return whether to continue the main loop
