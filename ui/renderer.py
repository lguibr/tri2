# File: ui/renderer.py
import pygame
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque
import logging

from config import VisConfig, EnvConfig, DemoConfig

# from environment.game_state import GameState # No longer needed directly
from .panels import LeftPanelRenderer, GameAreaRenderer
from .overlays import OverlayRenderer
from .plotter import Plotter
from .demo_renderer import DemoRenderer
from .input_handler import InputHandler
from app_state import AppState

logger = logging.getLogger(__name__)


class UIRenderer:
    """Orchestrates rendering of all UI components based on data received from the logic process."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        # Plotter is specific to the UI process
        self.plotter = Plotter()
        # Sub-renderers are initialized here
        self.left_panel = LeftPanelRenderer(screen, vis_config, self.plotter)
        self.game_area = GameAreaRenderer(screen, vis_config)
        self.overlays = OverlayRenderer(screen, vis_config)
        self.demo_config = DemoConfig()
        # DemoRenderer needs access to GameAreaRenderer's fonts/methods if shared
        self.demo_renderer = DemoRenderer(
            screen, vis_config, self.demo_config, self.game_area
        )
        self.input_handler_ref: Optional[InputHandler] = None

    def set_input_handler(self, input_handler: InputHandler):
        """Sets the InputHandler reference for components that need it (e.g., buttons)."""
        self.input_handler_ref = input_handler
        self.left_panel.input_handler = input_handler
        # Pass references down if needed
        if hasattr(self.left_panel, "button_status_renderer"):
            self.left_panel.button_status_renderer.input_handler_ref = input_handler
            # Pass app_ref if needed (though app_ref is less relevant now)
            # self.left_panel.button_status_renderer.app_ref = input_handler.app_ref

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0
        # Clear caches that might depend on screen size or old data
        self.game_area.best_state_surface_cache = None
        self.game_area.placeholder_surface_cache = None
        # Maybe force re-calc of button rects? (Handled by input handler resize)
        logger.info("[Renderer] Forced redraw triggered.")

    def render_all(self, **render_data: Dict[str, Any]):
        """
        Renders UI based on the application state dictionary received from the logic process.
        """
        try:
            app_state_str = render_data.get("app_state", AppState.UNKNOWN.value)
            current_app_state = (
                AppState(app_state_str)
                if app_state_str in AppState._value2member_map_
                else AppState.UNKNOWN
            )

            # --- Main Rendering Logic ---
            if current_app_state == AppState.MAIN_MENU:
                self._render_main_menu(**render_data)
            elif current_app_state == AppState.PLAYING:
                self._render_demo_mode(is_debug=False, **render_data)
            elif current_app_state == AppState.DEBUG:
                self._render_demo_mode(is_debug=True, **render_data)
            elif current_app_state == AppState.INITIALIZING:
                # Use status from render_data
                self._render_initializing_screen(
                    render_data.get("status", "Initializing...")
                )
            elif current_app_state == AppState.ERROR:
                self._render_error_screen(render_data.get("status", "Unknown Error"))
            else:  # Handle other potential states or default view
                self._render_simple_message(f"State: {app_state_str}", VisConfig.WHITE)

            # Render overlays on top (e.g., cleanup confirmation)
            if (
                render_data.get("cleanup_confirmation_active")
                and current_app_state != AppState.ERROR
            ):
                self.overlays.render_cleanup_confirmation()
            elif not render_data.get("cleanup_confirmation_active"):
                # Render temporary status messages (like 'Cleanup complete')
                self.overlays.render_status_message(
                    render_data.get("cleanup_message", ""),
                    render_data.get("last_cleanup_message_time", 0.0),
                )

            pygame.display.flip()  # Flip the display once after all rendering

        except pygame.error as e:
            logger.error(f"Pygame rendering error in render_all: {e}", exc_info=True)
            try:
                self._render_simple_message("Pygame Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception:
                pass
        except Exception as e:
            logger.critical(
                f"Unexpected critical rendering error in render_all: {e}", exc_info=True
            )
            try:
                self._render_simple_message("Critical Render Error!", VisConfig.RED)
                pygame.display.flip()
            except Exception:
                pass

    def _render_main_menu(self, **render_data: Dict[str, Any]):
        """Renders the main dashboard view with live worker envs."""
        self.screen.fill(VisConfig.BLACK)
        current_width, current_height = self.screen.get_size()
        lp_width, ga_width = self._calculate_panel_widths(current_width)

        # --- Render Left Panel ---
        # Pass only the necessary data from render_data
        self.left_panel.render(
            panel_width=lp_width,
            is_process_running=render_data.get("is_process_running", False),
            status=render_data.get("status", ""),
            stats_summary=render_data.get("stats_summary", {}),
            plot_data=render_data.get("plot_data", {}),
            app_state=render_data.get("app_state", ""),
            update_progress_details={},  # Maybe remove if not used
            agent_param_count=render_data.get("agent_param_count", 0),
            worker_counts=render_data.get("worker_counts", {}),
        )

        # --- Render Game Area Panel ---
        if ga_width > 0:
            # Recreate minimalist EnvConfig for rendering if needed
            # Or pass necessary values directly
            # env_config_render = EnvConfig() # Avoid creating full config if possible
            env_config_render_dict = {
                "ROWS": render_data.get("env_config_rows", 8),
                "COLS": render_data.get("env_config_cols", 15),
                # Add other needed EnvConfig values if GameAreaRenderer uses them
            }
            self.game_area.render(
                panel_width=ga_width,
                panel_x_offset=lp_width,
                worker_render_data=render_data.get("worker_render_data", []),
                num_envs=render_data.get("num_envs", 0),
                env_config=env_config_render_dict,  # Pass dict or minimalist object
                best_game_state_data=render_data.get("best_game_state_data"),
            )

    def _calculate_panel_widths(self, current_width: int) -> Tuple[int, int]:
        """Calculates the widths for the left and game area panels."""
        left_panel_ratio = max(0.2, min(0.8, self.vis_config.LEFT_PANEL_RATIO))
        lp_width = int(current_width * left_panel_ratio)
        ga_width = current_width - lp_width
        min_lp_width = 400
        if lp_width < min_lp_width and current_width > min_lp_width:
            lp_width = min_lp_width
            ga_width = max(0, current_width - lp_width)
        elif current_width <= min_lp_width:
            lp_width = current_width
            ga_width = 0
        return lp_width, ga_width

    def _render_demo_mode(self, is_debug: bool, **render_data: Dict[str, Any]):
        """Renders the demo or debug mode using data from the queue."""
        if not self.demo_renderer:
            self._render_simple_message("Demo Renderer Missing!", VisConfig.RED)
            return

        # Reconstruct a temporary GameState or pass data directly
        # Passing data directly avoids creating GameState in UI process
        # Requires DemoRenderer to accept data dictionary instead of GameState object
        demo_env_state = render_data.get("demo_env_state")
        if demo_env_state:
            # DemoRenderer needs to be adapted to use this data
            self.demo_renderer.render(
                demo_env_data=render_data,  # Pass the relevant subset
                env_config=None,  # Pass necessary config values instead
                is_debug=is_debug,
            )
        else:
            mode = "Debug" if is_debug else "Demo"
            self._render_simple_message(f"{mode} Env Data Missing!", VisConfig.RED)

    def _render_initializing_screen(self, status_message: str = "Initializing..."):
        """Renders a simple initializing message."""
        self._render_simple_message(status_message, VisConfig.WHITE)

    def _render_error_screen(self, status_message: str):
        """Renders the error screen."""
        try:
            self.screen.fill((40, 0, 0))
            font_title = pygame.font.SysFont(None, 70)
            font_msg = pygame.font.SysFont(None, 30)
            if not font_title or not font_msg:
                font_title = pygame.font.Font(None, 70)
                font_msg = pygame.font.Font(None, 30)

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
            logger.error(f"Error rendering error screen itself: {e}")
            self._render_simple_message(f"Error State: {status_message}", VisConfig.RED)

    def _render_simple_message(self, message: str, color: Tuple[int, int, int]):
        """Renders a simple centered message."""
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)
            if not font:
                font = pygame.font.Font(None, 50)
            text_surf = font.render(message, True, color)
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
        except Exception as e:
            logger.error(f"Error rendering simple message '{message}': {e}")
