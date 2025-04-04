# File: ui/renderer.py
import pygame
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque

from config import VisConfig, EnvConfig, TensorBoardConfig, DemoConfig
from environment.game_state import GameState
from .panels import LeftPanelRenderer, GameAreaRenderer
from .overlays import OverlayRenderer
from .tooltips import TooltipRenderer
from .plotter import Plotter
from .demo_renderer import DemoRenderer  # Import the new demo renderer


class UIRenderer:
    """Orchestrates rendering of all UI components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = Plotter()
        self.left_panel = LeftPanelRenderer(screen, vis_config, self.plotter)
        self.game_area = GameAreaRenderer(screen, vis_config)
        self.overlays = OverlayRenderer(screen, vis_config)
        self.tooltips = TooltipRenderer(screen, vis_config)
        self.demo_config = DemoConfig()  # Need demo config for demo renderer
        # --- NEW: Instantiate DemoRenderer ---
        self.demo_renderer = DemoRenderer(
            screen, vis_config, self.demo_config, self.game_area
        )
        # --- END NEW ---
        self.last_plot_update_time = 0

    def check_hover(self, mouse_pos: Tuple[int, int], app_state: str):
        """Passes hover check to the tooltip renderer."""
        if app_state == "MainMenu":
            self.tooltips.update_rects_and_texts(
                self.left_panel.get_stat_rects(), self.left_panel.get_tooltip_texts()
            )
            self.tooltips.check_hover(mouse_pos)
        else:
            # Disable tooltips in other states for now
            self.tooltips.hovered_stat_key = None
            self.tooltips.stat_rects.clear()

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0

    def render_all(
        self,
        app_state: str,
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_confirmation_active: bool,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        demo_env: Optional[GameState] = None,
    ):
        """Renders UI based on the application state."""
        try:
            # 1. Render main content based on state (WITHOUT flipping)
            if app_state == "MainMenu":
                self._render_main_menu(
                    is_training,
                    status,
                    stats_summary,
                    buffer_capacity,
                    envs,
                    num_envs,
                    env_config,
                    cleanup_message,
                    last_cleanup_message_time,
                    tensorboard_log_dir,
                    plot_data,
                )
            elif app_state == "Playing":
                # --- MODIFIED: Call DemoRenderer ---
                if demo_env:
                    self.demo_renderer.render(demo_env, env_config)
                else:
                    # Render error if demo_env is missing
                    print("Error: Attempting to render demo mode without demo_env.")
                    self._render_simple_message("Demo Env Error!", VisConfig.RED)
                # --- END MODIFIED ---
            elif app_state == "Initializing":
                self._render_initializing_screen(status)
            elif app_state == "Error":
                self._render_error_screen(status)

            # 2. Render Overlays ON TOP if needed
            # Cleanup confirmation overlay takes precedence
            if cleanup_confirmation_active and app_state != "Error":
                self.overlays.render_cleanup_confirmation()
            # Render transient status message if cleanup overlay is NOT active
            elif not cleanup_confirmation_active:
                self.overlays.render_status_message(
                    cleanup_message, last_cleanup_message_time
                )

            # 3. Render Tooltip (only in MainMenu and if cleanup not active)
            if app_state == "MainMenu" and not cleanup_confirmation_active:
                # Tooltip rects/texts are updated during check_hover or implicitly by left_panel.render
                # We just need to call render_tooltip here.
                self.tooltips.render_tooltip()

            # 4. Final Flip
            pygame.display.flip()

        except pygame.error as e:
            # Handle specific pygame errors if needed
            print(f"Pygame rendering error in render_all: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"Unexpected critical rendering error in render_all: {e}")
            traceback.print_exc()
            # Attempt to render a basic error screen on critical failure
            try:
                self._render_simple_message("Critical Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception:
                pass  # Ignore errors during error rendering

    def _render_main_menu(
        self,
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_message: str,  # No longer need cleanup_confirmation_active here
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
    ):
        """Renders the main training dashboard view. Does NOT flip display."""
        self.screen.fill(VisConfig.BLACK)  # Clear screen

        # Render Main Panels (Left Panel updates tooltip rects internally now)
        self.left_panel.render(
            is_training,
            status,
            stats_summary,
            buffer_capacity,
            tensorboard_log_dir,
            plot_data,
            app_state="MainMenu",  # Pass state for conditional rendering
        )
        self.game_area.render(envs, num_envs, env_config)

        # Status message and tooltips are handled by render_all after this

    def _render_initializing_screen(
        self, status_message: str = "Initializing RL Components..."
    ):
        """Renders the initializing screen with a status message."""
        self._render_simple_message(status_message, VisConfig.WHITE)

    def _render_error_screen(self, status_message: str):
        """Renders the error screen. Does NOT flip display."""
        try:
            self.screen.fill((40, 0, 0))  # Dark red background
            font_title = pygame.font.SysFont(None, 70)
            font_msg = pygame.font.SysFont(None, 30)

            title_surf = font_title.render("APPLICATION ERROR", True, VisConfig.RED)
            title_rect = title_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() // 3)
            )

            msg_surf = font_msg.render(
                f"Status: {status_message}", True, VisConfig.YELLOW
            )
            msg_rect = msg_surf.get_rect(
                center=(self.screen.get_width() // 2, title_rect.bottom + 30)
            )

            exit_surf = font_msg.render(
                "Press ESC or close window to exit.", True, VisConfig.WHITE
            )
            exit_rect = exit_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() * 0.8)
            )

            self.screen.blit(title_surf, title_rect)
            self.screen.blit(msg_surf, msg_rect)
            self.screen.blit(exit_surf, exit_rect)

        except Exception as e:
            print(f"Error rendering error screen: {e}")
            # Fallback to simple message
            self._render_simple_message(f"Error State: {status_message}", VisConfig.RED)

    def _render_simple_message(self, message: str, color: Tuple[int, int, int]):
        """Renders a simple centered text message."""
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)
            text_surf = font.render(message, True, color)
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
        except Exception as e:
            print(f"Error rendering simple message '{message}': {e}")
