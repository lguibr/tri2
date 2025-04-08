# File: ui/renderer.py
import pygame
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque
import logging

from config import VisConfig, EnvConfig, DemoConfig
from environment.game_state import GameState
from .panels import LeftPanelRenderer, GameAreaRenderer
from .overlays import OverlayRenderer
from .plotter import Plotter
from .demo_renderer import DemoRenderer
from .input_handler import InputHandler
from app_state import AppState

logger = logging.getLogger(__name__)


class UIRenderer:
    """Orchestrates rendering of all UI components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = Plotter()
        self.left_panel = LeftPanelRenderer(screen, vis_config, self.plotter)
        self.game_area = GameAreaRenderer(screen, vis_config)
        self.overlays = OverlayRenderer(screen, vis_config)
        self.demo_config = DemoConfig()
        self.demo_renderer = DemoRenderer(
            screen, vis_config, self.demo_config, self.game_area
        )
        self.last_plot_update_time = 0

    def set_input_handler(self, input_handler: InputHandler):
        """Sets the InputHandler reference after it's initialized."""
        self.left_panel.input_handler = input_handler
        if hasattr(self.left_panel, "button_status_renderer"):
            self.left_panel.button_status_renderer.input_handler_ref = input_handler
            # Pass app reference if needed by button renderer
            if hasattr(self.left_panel.button_status_renderer, "app_ref"):
                self.left_panel.button_status_renderer.app_ref = input_handler.app_ref

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0
        # Also clear cached surfaces in game area
        self.game_area.best_state_surface_cache = None
        self.game_area.placeholder_surface_cache = None
        logger.info("[Renderer] Forced redraw triggered.")

    def render_all(self, **kwargs):  # Use kwargs for flexibility
        """Renders UI based on the application state."""
        try:
            app_state_str = kwargs.get("app_state", AppState.UNKNOWN.value)
            current_app_state = (
                AppState(app_state_str)
                if app_state_str in AppState._value2member_map_
                else AppState.UNKNOWN
            )

            if current_app_state == AppState.MAIN_MENU:
                self._render_main_menu(**kwargs)
            elif current_app_state == AppState.PLAYING:
                self._render_demo_mode(
                    kwargs.get("demo_env"), kwargs.get("env_config"), is_debug=False
                )
            elif current_app_state == AppState.DEBUG:
                self._render_demo_mode(
                    kwargs.get("demo_env"), kwargs.get("env_config"), is_debug=True
                )
            elif current_app_state == AppState.INITIALIZING:
                self._render_initializing_screen(
                    kwargs.get("status", "Initializing...")
                )
            elif current_app_state == AppState.ERROR:
                self._render_error_screen(kwargs.get("status", "Unknown Error"))

            # Render overlays on top
            if (
                kwargs.get("cleanup_confirmation_active")
                and current_app_state != AppState.ERROR
            ):
                self.overlays.render_cleanup_confirmation()
            elif not kwargs.get("cleanup_confirmation_active"):
                self.overlays.render_status_message(
                    kwargs.get("cleanup_message", ""),
                    kwargs.get("last_cleanup_message_time", 0.0),
                )

            pygame.display.flip()

        except pygame.error as e:
            logger.error(f"Pygame rendering error in render_all: {e}")
        except Exception as e:
            logger.critical(
                f"Unexpected critical rendering error in render_all: {e}", exc_info=True
            )
            try:
                self._render_simple_message("Critical Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception:
                pass

    def _render_main_menu(self, **kwargs):
        """Renders the main dashboard view."""
        self.screen.fill(VisConfig.BLACK)
        current_width, current_height = self.screen.get_size()
        lp_width, ga_width = self._calculate_panel_widths(current_width)

        # Render Left Panel
        self.left_panel.render(
            panel_width=lp_width,
            is_process_running=kwargs.get("is_process_running", False),
            status=kwargs.get("status", ""),
            stats_summary=kwargs.get("stats_summary", {}),
            plot_data=kwargs.get("plot_data", {}),
            app_state=kwargs.get("app_state", ""),
            update_progress_details=kwargs.get("update_progress_details", {}),
            agent_param_count=kwargs.get("agent_param_count", 0),
            worker_counts=kwargs.get("worker_counts", {}),
        )

        # Render Game Area Panel
        if ga_width > 0:
            self.game_area.render(
                panel_width=ga_width,
                panel_x_offset=lp_width,
                is_running=kwargs.get(
                    "is_process_running", False
                ),  # Pass running state
                envs=kwargs.get("envs", []),
                num_envs=kwargs.get("num_envs", 0),
                env_config=kwargs.get("env_config"),
                best_game_state_data=kwargs.get("best_game_state_data"),
                stats_summary=kwargs.get("stats_summary", {}),
            )

    def _calculate_panel_widths(self, current_width: int) -> Tuple[int, int]:
        """Calculates the widths for the left and game area panels."""
        left_panel_ratio = max(0.1, min(0.9, self.vis_config.LEFT_PANEL_RATIO))
        lp_width = int(current_width * left_panel_ratio)
        ga_width = current_width - lp_width
        min_lp_width = 300
        if lp_width < min_lp_width and current_width > min_lp_width:
            lp_width = min_lp_width
            ga_width = max(0, current_width - lp_width)
        elif current_width <= min_lp_width:
            lp_width = current_width
            ga_width = 0
        return lp_width, ga_width

    def _render_demo_mode(
        self,
        demo_env: Optional[GameState],
        env_config: Optional[EnvConfig],
        is_debug: bool,
    ):
        """Renders the demo or debug mode."""
        if demo_env and env_config:
            self.demo_renderer.render(demo_env, env_config, is_debug=is_debug)
        else:
            mode = "Debug" if is_debug else "Demo"
            self._render_simple_message(f"{mode} Env Error!", VisConfig.RED)

    def _render_initializing_screen(self, status_message: str = "Initializing..."):
        self._render_simple_message(status_message, VisConfig.WHITE)

    def _render_error_screen(self, status_message: str):
        """Renders the error screen."""
        try:
            self.screen.fill((40, 0, 0))
            font_title = pygame.font.SysFont(None, 70)
            font_msg = pygame.font.SysFont(None, 30)
            title_surf = font_title.render("APPLICATION ERROR", True, VisConfig.RED)
            msg_surf = font_msg.render(
                f"Status: {status_message}", True, VisConfig.YELLOW
            )
            exit_surf = font_msg.render(
                "Press ESC or close window to exit.", True, VisConfig.WHITE
            )

            title_rect = title_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() // 3)
            )
            msg_rect = msg_surf.get_rect(
                center=(self.screen.get_width() // 2, title_rect.bottom + 30)
            )
            exit_rect = exit_surf.get_rect(
                center=(self.screen.get_width() // 2, self.screen.get_height() * 0.8)
            )

            self.screen.blit(title_surf, title_rect)
            self.screen.blit(msg_surf, msg_rect)
            self.screen.blit(exit_surf, exit_rect)
        except Exception as e:
            logger.error(f"Error rendering error screen: {e}")
            self._render_simple_message(f"Error State: {status_message}", VisConfig.RED)

    def _render_simple_message(self, message: str, color: Tuple[int, int, int]):
        """Renders a simple centered message."""
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)
            text_surf = font.render(message, True, color)
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
        except Exception as e:
            logger.error(f"Error rendering simple message '{message}': {e}")
