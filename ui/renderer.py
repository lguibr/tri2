# File: ui/renderer.py
import pygame
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple, Deque

# --- MODIFIED: Import DemoConfig if created ---
from config import (
    VisConfig,
    EnvConfig,
    TensorBoardConfig,
    DemoConfig,
)  # Added DemoConfig
from environment.game_state import GameState
from environment.shape import Shape  # Import Shape for preview type hint
from environment.triangle import Triangle  # Import Triangle for preview type hint
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
        # --- NEW: Add DemoConfig instance ---
        self.demo_config = DemoConfig()  # Instantiate if created
        # --- END NEW ---
        self._init_demo_fonts()  # NEW

    def _init_demo_fonts(self):
        """Initialize fonts specific to demo mode."""
        try:
            self.demo_hud_font = pygame.font.SysFont(
                None, self.demo_config.HUD_FONT_SIZE
            )
            self.demo_help_font = pygame.font.SysFont(
                None, self.demo_config.HELP_FONT_SIZE
            )
        except Exception as e:
            print(f"Warning: SysFont error for demo fonts: {e}. Using default.")
            self.demo_hud_font = pygame.font.Font(None, self.demo_config.HUD_FONT_SIZE)
            self.demo_help_font = pygame.font.Font(
                None, self.demo_config.HELP_FONT_SIZE
            )

    def check_hover(self, mouse_pos: Tuple[int, int], app_state: str):  # Pass app_state
        """Passes hover check to the tooltip renderer."""
        # Only update tooltips if in MainMenu
        if app_state == "MainMenu":
            self.tooltips.update_rects_and_texts(
                self.left_panel.get_stat_rects(), self.left_panel.get_tooltip_texts()
            )
            self.tooltips.check_hover(mouse_pos)
        else:
            # Clear any active tooltips when not in main menu
            self.tooltips.hovered_stat_key = None
            self.tooltips.stat_rects.clear()  # Also clear rects

    def force_redraw(self):
        """Forces components like the plotter to redraw on the next frame."""
        self.plotter.last_plot_update_time = 0  # Reset plot timer to force update

    def render_all(
        self,
        app_state: str,  # NEW: Determine render mode
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
        demo_env: Optional[GameState] = None,  # NEW: Add demo_env
    ):
        """Renders UI based on the application state."""
        try:
            if app_state == "MainMenu":
                self._render_main_menu(
                    is_training,
                    status,
                    stats_summary,
                    buffer_capacity,
                    envs,
                    num_envs,
                    env_config,
                    cleanup_confirmation_active,
                    cleanup_message,
                    last_cleanup_message_time,
                    tensorboard_log_dir,
                    plot_data,
                )
            elif app_state == "Playing":
                if demo_env:
                    self._render_demo_mode(demo_env, env_config)
                else:
                    print("Error: Attempting to render demo mode without demo_env.")
                    # Optionally render an error message screen
                    self.screen.fill(VisConfig.BLACK)
                    err_font = pygame.font.SysFont(None, 50)
                    err_surf = err_font.render("Demo Env Error!", True, VisConfig.RED)
                    self.screen.blit(
                        err_surf,
                        err_surf.get_rect(center=self.screen.get_rect().center),
                    )
                    pygame.display.flip()

            elif app_state == "Initializing":
                self._render_initializing_screen()  # Optional: Add a loading screen

            elif app_state == "Error":
                self._render_error_screen(
                    status
                )  # Render a specific error state screen

            # Overlays like cleanup confirmation can appear over any state (except maybe Error?)
            if cleanup_confirmation_active and app_state != "Error":
                self.overlays.render_cleanup_confirmation()

            # Status message (after cleanup) can appear over MainMenu
            if app_state == "MainMenu" and not cleanup_confirmation_active:
                self.overlays.render_status_message(
                    cleanup_message, last_cleanup_message_time
                )

            # Tooltips are only handled and rendered within _render_main_menu now

        except pygame.error as e:
            if "video system not initialized" in str(e):
                print("Error: Pygame video system not initialized. Exiting render.")
            elif "Invalid subsurface rectangle" in str(e):
                print(f"Warning: Invalid subsurface rectangle during rendering: {e}")
            else:
                print(f"Pygame rendering error: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"Unexpected critical rendering error in render_all: {e}")
            traceback.print_exc()

    def _render_main_menu(
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
        """Renders the main training dashboard view."""
        self.screen.fill(VisConfig.BLACK)  # Clear screen

        # 1. Render Main Panels
        self.left_panel.render(
            is_training,
            status,
            stats_summary,
            buffer_capacity,
            tensorboard_log_dir,
            plot_data,
            app_state="MainMenu",  # Pass state
        )
        self.game_area.render(envs, num_envs, env_config)

        # 2. Render Overlays (excluding cleanup and status message, handled globally or in render_all)

        # 3. Render Tooltip (only if cleanup confirm is NOT active and no other modal overlay is shown)
        if not cleanup_confirmation_active:
            # Update tooltips again *after* panels are drawn to ensure rects are current
            self.tooltips.update_rects_and_texts(
                self.left_panel.get_stat_rects(),
                self.left_panel.get_tooltip_texts(),
            )
            self.tooltips.render_tooltip()

        pygame.display.flip()

    def _render_demo_mode(self, demo_env: GameState, env_config: EnvConfig):
        """Renders the single-player interactive demo mode."""
        self.screen.fill(self.demo_config.BACKGROUND_COLOR)
        sw, sh = self.screen.get_size()

        # --- Game Area ---
        # Make game area large, e.g., 85% of height, centered horizontally
        game_h = int(sh * 0.85)
        aspect_ratio = (env_config.COLS * 0.75 + 0.25) / max(1, env_config.ROWS)
        game_w = int(game_h * aspect_ratio)

        # Ensure minimum width/height for game area
        min_dim = 100
        game_w = max(min_dim, game_w)
        game_h = max(
            min_dim, int(game_w / aspect_ratio) if aspect_ratio > 0 else min_dim
        )

        # Center the game area, considering potential minimum size adjustments
        game_x = max(0, (sw - game_w) // 2)
        game_y = max(0, int(sh * 0.05))

        # Ensure game area fits on screen after adjustments
        game_w = min(game_w, sw - game_x)
        game_h = min(game_h, sh - game_y - 30)  # Leave space for HUD/Help

        game_rect = pygame.Rect(game_x, game_y, game_w, game_h)

        if game_w > 10 and game_h > 10:
            # Calculate cell size for the single env
            tri_cell_w, tri_cell_h = self.game_area._calculate_triangle_size(
                game_w, game_h, env_config
            )
            # Render the env using a subsurface
            try:
                # Ensure subsurface rect is valid
                game_rect_clipped = game_rect.clip(self.screen.get_rect())
                if game_rect_clipped.width <= 0 or game_rect_clipped.height <= 0:
                    raise ValueError(
                        "Calculated demo game subsurface has zero dimension"
                    )

                game_surf = self.screen.subsurface(game_rect_clipped)
                # Render the core game grid/pieces
                self.game_area._render_single_env(
                    game_surf, demo_env, int(tri_cell_w), int(tri_cell_h)
                )
                # --- Render Placement Preview ---
                self._render_placement_preview(
                    game_surf, demo_env, int(tri_cell_w), int(tri_cell_h)
                )
                # Render shape previews slightly differently (maybe larger)
                self._render_demo_shape_previews(game_surf, demo_env)

            except ValueError as e:
                print(
                    f"Error creating subsurface for demo game ({game_rect_clipped}): {e}"
                )
                pygame.draw.rect(self.screen, VisConfig.RED, game_rect, 1)
            except Exception as render_e:
                print(f"Error rendering demo game area: {render_e}")
                traceback.print_exc()
                pygame.draw.rect(self.screen, VisConfig.RED, game_rect, 1)
        else:
            # Render "Too small" message if needed
            self.game_area._render_too_small_message(game_rect, game_w, game_h)

        # --- HUD ---
        hud_y = game_rect.bottom + 10
        score_text = f"Score: {demo_env.game_score} | Lines: {demo_env.lines_cleared_this_episode}"
        score_surf = self.demo_hud_font.render(score_text, True, VisConfig.WHITE)
        score_rect = score_surf.get_rect(midtop=(sw // 2, hud_y))
        self.screen.blit(score_surf, score_rect)

        # --- Help Text ---
        help_surf = self.demo_help_font.render(
            self.demo_config.HELP_TEXT, True, VisConfig.LIGHTG
        )
        help_rect = help_surf.get_rect(centerx=sw // 2, bottom=sh - 10)
        self.screen.blit(help_surf, help_rect)

        pygame.display.flip()

    def _render_placement_preview(
        self, surf: pygame.Surface, env: GameState, cell_w: int, cell_h: int
    ):
        """Renders a ghost preview of the currently selected shape at the target location."""
        if cell_w <= 0 or cell_h <= 0:
            return  # Cannot render if cell size is invalid

        shp, rr, cc = env.get_current_selection_info()
        if shp is None:
            return

        is_valid = env.grid.can_place(shp, rr, cc)
        preview_color = (
            self.demo_config.PREVIEW_COLOR
            if is_valid
            else self.demo_config.INVALID_PREVIEW_COLOR
        )

        for dr, dc, up in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if env.grid.valid(nr, nc):
                # Create a temporary Triangle to get points
                temp_tri = Triangle(row=nr, col=nc, is_up=up)
                try:
                    pts = temp_tri.get_points(ox=0, oy=0, cw=cell_w, ch=cell_h)
                    # Create a temporary surface with alpha for transparency
                    temp_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                    pygame.draw.polygon(temp_surface, preview_color, pts)
                    surf.blit(temp_surface, (0, 0))
                except Exception as e:
                    print(f"Warning: Error rendering preview tri ({nr},{nc}): {e}")

    def _render_demo_shape_previews(self, surf: pygame.Surface, env: GameState):
        """Renders shape previews for demo mode, highlighting the selected one."""
        available_shapes = env.get_shapes()  # Gets only non-None shapes
        # No need to render if no shapes are available, although this shouldn't happen in normal gameplay
        # if not available_shapes:
        #     return

        all_slots = env.shapes  # Get the list including None
        selected_shape_obj = (
            all_slots[env.demo_selected_shape_idx]
            if 0 <= env.demo_selected_shape_idx < len(all_slots)
            else None
        )

        # Place previews vertically on the right side
        num_slots = env.env_config.NUM_SHAPE_SLOTS
        if num_slots <= 0:
            return

        preview_area_w = surf.get_width() * 0.2  # Allocate space on the right
        preview_area_h = surf.get_height()
        preview_area_x = surf.get_width() - preview_area_w
        preview_area_y = 0

        preview_h = max(20, preview_area_h / num_slots)  # Min height per preview
        preview_w = max(20, preview_area_w)
        preview_padding = 5

        for i in range(num_slots):
            shp = all_slots[i] if i < len(all_slots) else None
            current_preview_y = preview_area_y + i * preview_h
            # Define the rect for this preview slot
            preview_rect = pygame.Rect(
                preview_area_x + preview_padding,
                current_preview_y + preview_padding,
                preview_w - 2 * preview_padding,
                preview_h - 2 * preview_padding,
            )

            # Clip the preview rect to the surface bounds
            clipped_preview_rect = preview_rect.clip(surf.get_rect())
            if clipped_preview_rect.width <= 0 or clipped_preview_rect.height <= 0:
                continue  # Skip rendering if preview area is off-screen or too small

            # Highlight background if selected
            if shp is not None and shp == selected_shape_obj:
                pygame.draw.rect(
                    surf,
                    self.demo_config.SELECTED_SHAPE_HIGHLIGHT_COLOR,
                    clipped_preview_rect,
                    2,
                    border_radius=3,
                )
            else:
                # Optional: Draw a faint border for empty/unselected slots
                pygame.draw.rect(
                    surf, VisConfig.GRAY, clipped_preview_rect, 1, border_radius=3
                )

            if shp is not None:
                # Render the shape centered in the preview rect
                try:
                    # Create a subsurface for drawing the shape, makes clipping easier
                    shape_sub_surf = surf.subsurface(clipped_preview_rect)
                    shape_surf_rect = (
                        shape_sub_surf.get_rect()
                    )  # Rect relative to the subsurface (topleft is 0,0)

                    # Calculate appropriate cell size to fit the shape within the subsurface
                    min_r, min_c, max_r, max_c = shp.bbox()
                    shape_h_cells = max(1, max_r - min_r + 1)
                    shape_w_cells = max(1, max_c - min_c + 1)
                    cell_size = 1  # Default cell size
                    if shape_h_cells > 0 and shape_w_cells > 0:
                        cell_size_w = shape_surf_rect.width / max(
                            1, (shape_w_cells * 0.75 + 0.25)
                        )
                        cell_size_h = shape_surf_rect.height / max(1, shape_h_cells)
                        cell_size = max(
                            2, min(cell_size_w, cell_size_h)
                        )  # Use smaller dimension scale, ensure > 0

                    # Create a temporary surface to draw the shape onto, then blit to subsurface
                    temp_shape_surf = pygame.Surface(
                        shape_surf_rect.size, pygame.SRCALPHA
                    )
                    temp_shape_surf.fill((0, 0, 0, 0))  # Transparent background

                    self.game_area._render_single_shape(
                        temp_shape_surf, shp, int(cell_size)
                    )

                    # Blit the temporary surface onto the clipped subsurface
                    shape_sub_surf.blit(temp_shape_surf, (0, 0))

                except ValueError as sub_err:
                    print(f"Error creating shape preview subsurface {i}: {sub_err}")
                    pygame.draw.rect(surf, VisConfig.RED, clipped_preview_rect, 1)
                except Exception as e:
                    print(f"Error rendering demo shape preview {i}: {e}")
                    pygame.draw.rect(surf, VisConfig.RED, clipped_preview_rect, 1)

    def _render_initializing_screen(self):
        """Renders a simple 'Initializing...' screen."""
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)
            text_surf = font.render(
                "Initializing RL Components...", True, VisConfig.WHITE
            )
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
            pygame.display.flip()  # Update screen
        except Exception as e:
            print(f"Error rendering initializing screen: {e}")

    def _render_error_screen(self, status_message: str):
        """Renders a screen indicating an error state."""
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
            pygame.display.flip()
        except Exception as e:
            print(f"Error rendering error screen: {e}")
