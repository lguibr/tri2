# File: ui/renderer.py
import pygame
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque

from config import VisConfig, EnvConfig, DemoConfig
from environment.game_state import GameState
from .panels import LeftPanelRenderer, GameAreaRenderer
from .overlays import OverlayRenderer

# Removed TooltipRenderer import
from .plotter import Plotter
from .demo_renderer import DemoRenderer
from .input_handler import InputHandler
from app_state import AppState  # Import Enum


class UIRenderer:
    """Orchestrates rendering of all UI components."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = Plotter()
        self.left_panel = LeftPanelRenderer(screen, vis_config, self.plotter)
        self.game_area = GameAreaRenderer(screen, vis_config)
        self.overlays = OverlayRenderer(screen, vis_config)
        # Removed TooltipRenderer instance
        self.demo_config = DemoConfig()
        self.demo_renderer = DemoRenderer(
            screen, vis_config, self.demo_config, self.game_area
        )
        self.last_plot_update_time = 0

    def set_input_handler(self, input_handler: InputHandler):
        """Sets the InputHandler reference after it's initialized."""
        self.left_panel.input_handler = input_handler
        # Also set the reference in the button renderer
        if hasattr(self.left_panel, "button_status_renderer"):
            self.left_panel.button_status_renderer.input_handler_ref = input_handler

    def check_hover(self, mouse_pos: Tuple[int, int], app_state_str: str):
        """Passes hover check to the tooltip renderer."""
        # Removed tooltip hover check logic
        pass

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0

    def render_all(
        self,
        app_state: str,  # Keep as string for now, internal logic uses Enum
        is_process_running: bool,
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
        update_progress_details: Dict[str, Any] = {},
        agent_param_count: int = 0,
        worker_counts: Dict[str, int] = {
            "env_runners": 0,
            "trainers": 0,
        },  # Added worker_counts
    ):
        """Renders UI based on the application state."""
        try:
            # Convert string state back to Enum for comparison if needed, or keep using string
            current_app_state = (
                AppState(app_state)
                if app_state in AppState._value2member_map_
                else AppState.UNKNOWN
            )

            if current_app_state == AppState.MAIN_MENU:
                self._render_main_menu(
                    is_process_running=is_process_running,
                    status=status,
                    stats_summary=stats_summary,
                    envs=envs,
                    num_envs=num_envs,
                    env_config=env_config,
                    cleanup_message=cleanup_message,
                    last_cleanup_message_time=last_cleanup_message_time,
                    tensorboard_log_dir=tensorboard_log_dir,
                    plot_data=plot_data,
                    update_progress_details=update_progress_details,
                    app_state=app_state,
                    agent_param_count=agent_param_count,
                    worker_counts=worker_counts,  # Pass worker counts
                )
            elif current_app_state == AppState.PLAYING:
                if demo_env:
                    self.demo_renderer.render(demo_env, env_config, is_debug=False)
                else:
                    self._render_simple_message("Demo Env Error!", VisConfig.RED)
            elif current_app_state == AppState.DEBUG:
                if demo_env:
                    self.demo_renderer.render(demo_env, env_config, is_debug=True)
                else:
                    self._render_simple_message("Debug Env Error!", VisConfig.RED)
            elif current_app_state == AppState.INITIALIZING:
                self._render_initializing_screen(status)
            elif current_app_state == AppState.ERROR:
                self._render_error_screen(status)

            if cleanup_confirmation_active and current_app_state != AppState.ERROR:
                self.overlays.render_cleanup_confirmation()
            elif not cleanup_confirmation_active:
                self.overlays.render_status_message(
                    cleanup_message, last_cleanup_message_time
                )

            # Removed tooltip rendering call
            # if (
            #     current_app_state == AppState.MAIN_MENU
            #     and not cleanup_confirmation_active
            # ):
            #     self.tooltips.render_tooltip()

            pygame.display.flip()

        except pygame.error as e:
            print(f"Pygame rendering error in render_all: {e}")
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
        is_process_running: bool,
        status: str,
        stats_summary: Dict[str, Any],
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        update_progress_details: Dict[str, Any],
        app_state: str,
        agent_param_count: int,
        worker_counts: Dict[str, int],  # Added worker_counts
    ):
        """Renders the main training dashboard view."""
        self.screen.fill(VisConfig.BLACK)
        current_width, current_height = self.screen.get_size()
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

        self.left_panel.render(
            panel_width=lp_width,
            is_process_running=is_process_running,
            status=status,
            stats_summary=stats_summary,
            tensorboard_log_dir=tensorboard_log_dir,
            plot_data=plot_data,
            app_state=app_state,
            update_progress_details=update_progress_details,
            agent_param_count=agent_param_count,
            worker_counts=worker_counts,  # Pass worker counts
        )
        self.game_area.render(
            envs=envs,
            num_envs=num_envs,
            env_config=env_config,
            panel_width=ga_width,
            panel_x_offset=lp_width,
        )

    def _render_initializing_screen(
        self, status_message: str = "Initializing RL Components..."
    ):
        self._render_simple_message(status_message, VisConfig.WHITE)

    def _render_error_screen(self, status_message: str):
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
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)
            text_surf = font.render(message, True, color)
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
        except Exception as e:
            print(f"Error rendering simple message '{message}': {e}")
