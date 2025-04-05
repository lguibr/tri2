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
from .demo_renderer import DemoRenderer


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
        self.demo_config = DemoConfig()
        self.demo_renderer = DemoRenderer(
            screen, vis_config, self.demo_config, self.game_area
        )
        self.last_plot_update_time = 0

    def check_hover(self, mouse_pos: Tuple[int, int], app_state: str):
        """Passes hover check to the tooltip renderer."""
        if app_state == "MainMenu":
            self.tooltips.update_rects_and_texts(
                self.left_panel.get_stat_rects(), self.left_panel.get_tooltip_texts()
            )
            self.tooltips.check_hover(mouse_pos)
        else:
            self.tooltips.hovered_stat_key = None
            self.tooltips.stat_rects.clear()

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0

    def render_all(
        self,
        app_state: str,
        is_running: bool,  # Renamed from is_training
        status: str,
        stats_summary: Dict[str, Any],
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
            if app_state == "MainMenu":
                self._render_main_menu(
                    is_running,
                    status,
                    stats_summary,
                    envs,
                    num_envs,
                    env_config,
                    cleanup_message,
                    last_cleanup_message_time,
                    tensorboard_log_dir,
                    plot_data,
                )
            elif app_state == "Playing":
                if demo_env:
                    self.demo_renderer.render(demo_env, env_config)
                else:
                    print("Error: Attempting to render demo mode without demo_env.")
                    self._render_simple_message("Demo Env Error!", VisConfig.RED)
            elif app_state == "Initializing":
                self._render_initializing_screen(status)
            elif app_state == "Error":
                self._render_error_screen(status)

            if cleanup_confirmation_active and app_state != "Error":
                self.overlays.render_cleanup_confirmation()
            elif not cleanup_confirmation_active:
                self.overlays.render_status_message(
                    cleanup_message, last_cleanup_message_time
                )

            if app_state == "MainMenu" and not cleanup_confirmation_active:
                self.tooltips.render_tooltip()

            pygame.display.flip()

        except pygame.error as e:
            print(f"Pygame rendering error in render_all: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"Unexpected critical rendering error in render_all: {e}")
            traceback.print_exc()
            try:
                self._render_simple_message("Critical Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception:
                pass

    def _render_main_menu(
        self,
        is_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
    ):
        """Renders the main training dashboard view."""
        self.screen.fill(VisConfig.BLACK)

        self.left_panel.render(
            is_running,
            status,
            stats_summary,
            tensorboard_log_dir,
            plot_data,
            app_state="MainMenu",
        )
        self.game_area.render(envs, num_envs, env_config)

    def _render_initializing_screen(
        self, status_message: str = "Initializing RL Components..."
    ):
        """Renders the initializing screen with a status message."""
        self._render_simple_message(status_message, VisConfig.WHITE)

    def _render_error_screen(self, status_message: str):
        """Renders the error screen."""
        try:
            self.screen.fill((40, 0, 0))
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
