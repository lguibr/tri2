# File: ui/renderer.py
import pygame
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque

from config import VisConfig, EnvConfig, TensorBoardConfig
from environment.game_state import GameState
from .panels import LeftPanelRenderer, GameAreaRenderer
from .overlays import OverlayRenderer
from .tooltips import TooltipRenderer
from .plotter import Plotter


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

        self.last_plot_update_time = 0
        # --- REMOVED: State for toasts ---
        # self.active_toast_messages: List[Tuple[str, float]] = []
        # self.toast_duration = 4.0
        # --- END REMOVED ---

    def check_hover(self, mouse_pos: Tuple[int, int]):
        """Passes hover check to the tooltip renderer."""
        # Update tooltip renderer with the latest clickable areas from panels
        self.tooltips.update_rects_and_texts(
            self.left_panel.get_stat_rects(), self.left_panel.get_tooltip_texts()
        )
        self.tooltips.check_hover(mouse_pos)

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0  # Reset plot timer to force update

    # --- REMOVED: Method to add toasts ---
    # def add_toast(self, message: str):
    #     pass # No longer used
    # --- END REMOVED ---

    # --- REMOVED: Method to clear expired toasts ---
    # def _update_toasts(self):
    #     pass # No longer used
    # --- END REMOVED ---

    def render_all(
        self,
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
    ):
        """Renders all UI components in the correct order."""
        try:
            self.screen.fill(VisConfig.BLACK)  # Clear screen

            # --- REMOVED: Update active toasts ---
            # self._update_toasts()
            # --- END REMOVED ---

            # 1. Render Main Panels
            self.left_panel.render(
                is_training,
                status,
                stats_summary,
                buffer_capacity,
                tensorboard_log_dir,
                plot_data,
            )
            self.game_area.render(envs, num_envs, env_config)

            # 2. Render Overlays (if active)
            message_active = False
            # Cleanup confirmation takes priority
            if cleanup_confirmation_active:
                self.overlays.render_cleanup_confirmation()
            else:
                # Render status message (e.g., after cleanup) if cleanup confirm is NOT active
                message_active = self.overlays.render_status_message(
                    cleanup_message, last_cleanup_message_time
                )
                # --- REMOVED: Render toasts ---
                # self.overlays.render_toast_notifications(...) # Removed
                # --- END REMOVED ---

            # 3. Render Tooltip (if no blocking overlay is active)
            if not cleanup_confirmation_active and not message_active:
                # Update tooltips again *after* panels are drawn to ensure rects are current
                self.tooltips.update_rects_and_texts(
                    self.left_panel.get_stat_rects(),
                    self.left_panel.get_tooltip_texts(),
                )
                self.tooltips.render_tooltip()

            pygame.display.flip()  # Update the full display

        except pygame.error as e:
            # Handle specific Pygame errors gracefully
            if "video system not initialized" in str(e):
                print("Error: Pygame video system not initialized. Exiting render.")
                # Consider exiting or re-initializing Pygame here if appropriate
            elif "Invalid subsurface rectangle" in str(e):
                print(f"Warning: Invalid subsurface rectangle during rendering: {e}")
                # This might happen during resize or if env rendering fails
            else:
                print(f"Pygame rendering error: {e}")
                traceback.print_exc()  # Print details for other Pygame errors
        except Exception as e:
            print(f"Unexpected critical rendering error: {e}")
            traceback.print_exc()
