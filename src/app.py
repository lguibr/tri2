# File: src/app.py
# Change: Updated imports for visualization, environment, interaction.
import pygame
import sys
import logging
from . import config, utils

# Updated imports
from . import visualization
from . import environment
from . import interaction

logger = logging.getLogger(__name__)


class Application:
    """Main application integrating visualization and interaction."""

    def __init__(self, mode: str = "play"):
        self.vis_config = config.VisConfig()
        self.env_config = config.EnvConfig()
        self.mode = mode

        pygame.init()
        pygame.font.init()
        self.screen = self._setup_screen()
        self.clock = pygame.time.Clock()
        # Use load_fonts from the visualization module
        self.fonts = visualization.load_fonts()

        # Game state is now managed by the specific run script (interactive or training)
        # For interactive modes, we still need it here.
        if self.mode in ["play", "debug"]:
            # Use GameState from the environment module
            self.game_state = environment.GameState(self.env_config)
            # Use Visualizer from the visualization module
            self.visualizer = visualization.Visualizer(
                self.screen, self.vis_config, self.env_config, self.fonts
            )
            # Use InputHandler from the interaction module
            self.input_handler = interaction.InputHandler(
                self.game_state, self.visualizer, self.mode, self.env_config
            )
        # Removed 'training_visual' mode handling from App, as it's managed by run_training_visual.py
        # If this App class were reused for visual training display, it would need setup.
        else:
            raise ValueError(f"Unsupported application mode: {self.mode}")

        self.running = True

    def _setup_screen(self) -> pygame.Surface:
        """Initializes the Pygame screen."""
        screen = pygame.display.set_mode(
            (self.vis_config.SCREEN_WIDTH, self.vis_config.SCREEN_HEIGHT),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption(f"{config.APP_NAME} - {self.mode.capitalize()} Mode")
        return screen

    def run(self):
        """Main application loop."""
        logger.info(f"Starting application in {self.mode} mode.")
        while self.running:
            dt = self.clock.tick(self.vis_config.FPS) / 1000.0  # dt not used currently

            # Process input (if handler exists) and update state
            if self.input_handler:
                self.running = self.input_handler.handle_input()
                if not self.running:
                    break
            else:
                # Basic event handling if no specific input handler is setup (shouldn't happen with current modes)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.type == pygame.VIDEORESIZE:
                        # Basic resize handling if visualizer exists (e.g., if app were reused)
                        if self.visualizer:
                            try:
                                w, h = max(320, event.w), max(240, event.h)
                                # Update screen directly on visualizer
                                self.visualizer.screen = pygame.display.set_mode(
                                    (w, h), pygame.RESIZABLE
                                )
                                self.visualizer.layout_rects = None  # Force recalc
                            except pygame.error as e:
                                logger.error(f"Error resizing window: {e}")

                if not self.running:
                    break

            # Render current state (if applicable)
            if self.mode in ["play", "debug"] and self.visualizer and self.game_state:
                # Visualizer handles rendering the game state
                self.visualizer.render(self.game_state, self.mode)
                pygame.display.flip()
            # Removed training_visual rendering logic here

        logger.info("Application loop finished.")
        pygame.quit()
