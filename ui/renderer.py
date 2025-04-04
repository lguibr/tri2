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
import math


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
        self.demo_config = DemoConfig()
        self._init_demo_fonts()

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
            print(f"Warning: SysFont error for demo fonts: {e}. Using default.")  # LOG
            self.demo_hud_font = pygame.font.Font(None, self.demo_config.HUD_FONT_SIZE)
            self.demo_help_font = pygame.font.Font(
                None, self.demo_config.HELP_FONT_SIZE
            )

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
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_confirmation_active: bool,  # Receive the flag
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        demo_env: Optional[GameState] = None,
    ):
        """Renders UI based on the application state."""
        # LOGGING: Check received flag value
        # print(f"[UIRenderer::render_all] Received cleanup_confirmation_active = {cleanup_confirmation_active}, app_state = {app_state}") # LOG (Can be spammy)

        try:
            # Render main content based on state
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
                    print(
                        "Error: Attempting to render demo mode without demo_env."
                    )  # LOG
                    self.screen.fill(VisConfig.BLACK)
                    err_font = pygame.font.SysFont(None, 50)
                    err_surf = err_font.render("Demo Env Error!", True, VisConfig.RED)
                    self.screen.blit(
                        err_surf,
                        err_surf.get_rect(center=self.screen.get_rect().center),
                    )
                    pygame.display.flip()
            elif app_state == "Initializing":
                self._render_initializing_screen()
            elif app_state == "Error":
                self._render_error_screen(status)

            # --- Check condition for drawing cleanup overlay ---
            # This should draw ON TOP of the state-specific rendering above
            overlay_condition_met = cleanup_confirmation_active and app_state != "Error"
            # print(f"[UIRenderer::render_all] Checking overlay condition: cleanup_active={cleanup_confirmation_active}, app_state='{app_state}', condition_met={overlay_condition_met}") # LOG

            if overlay_condition_met:
                print(
                    "[UIRenderer::render_all] Condition met, calling render_cleanup_confirmation."
                )  # LOG
                self.overlays.render_cleanup_confirmation()  # <--- Check if this is called

            # Flip should happen *after* potential overlay drawing if not handled state-specifically
            # If _render_main_menu handles flip, this might double-flip or hide overlay briefly.
            # Let's let _render_main_menu handle its own flip, and add one here for other states.
            if app_state != "MainMenu":
                pygame.display.flip()

        except pygame.error as e:
            if "video system not initialized" in str(e):
                print("Error: Pygame video system not initialized.")  # LOG
            elif "Invalid subsurface rectangle" in str(e):
                print(f"Warning: Invalid subsurface rectangle: {e}")  # LOG
            else:
                print(f"Pygame rendering error: {e}")
                traceback.print_exc()  # LOG
        except Exception as e:
            print(f"Unexpected critical rendering error in render_all: {e}")
            traceback.print_exc()  # LOG

    def _render_main_menu(
        self,
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_confirmation_active: bool,  # Receive flag
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
            app_state="MainMenu",
        )
        self.game_area.render(envs, num_envs, env_config)

        # 2. Render Status Message (if not showing confirmation)
        if not cleanup_confirmation_active:
            self.overlays.render_status_message(
                cleanup_message, last_cleanup_message_time
            )

        # 3. Render Tooltip (if not showing confirmation)
        if not cleanup_confirmation_active:
            self.tooltips.update_rects_and_texts(
                self.left_panel.get_stat_rects(), self.left_panel.get_tooltip_texts()
            )
            self.tooltips.render_tooltip()

        # 4. Flip display *after* all main menu elements are drawn
        # Note: Cleanup overlay will be drawn *after* this flip in render_all if active.
        pygame.display.flip()

    def _render_demo_mode(self, demo_env: GameState, env_config: EnvConfig):
        """Renders the single-player interactive demo mode."""
        self.screen.fill(self.demo_config.BACKGROUND_COLOR)
        sw, sh = self.screen.get_size()
        padding = 30
        hud_height = 60
        max_game_h = sh - 2 * padding - hud_height
        max_game_w = sw - 2 * padding

        if max_game_h <= 0 or max_game_w <= 0:
            self._render_too_small_message(self.screen.get_rect(), sw, sh)
            pygame.display.flip()
            return

        aspect_ratio = (env_config.COLS * 0.75 + 0.25) / max(1, env_config.ROWS)
        game_w = max_game_w
        game_h = game_w / aspect_ratio if aspect_ratio > 0 else max_game_h
        if game_h > max_game_h:
            game_h = max_game_h
            game_w = game_h * aspect_ratio
        game_w = math.floor(min(game_w, max_game_w))
        game_h = math.floor(min(game_h, max_game_h))
        game_x = (sw - game_w) // 2
        game_y = padding
        game_rect = pygame.Rect(game_x, game_y, game_w, game_h)
        clipped_game_rect = game_rect.clip(self.screen.get_rect())

        if clipped_game_rect.width > 10 and clipped_game_rect.height > 10:
            try:
                game_surf = self.screen.subsurface(clipped_game_rect)
                self.game_area._render_single_env_grid(game_surf, demo_env, env_config)

                # Render Placement Preview
                preview_tri_cell_w, preview_tri_cell_h = (
                    self._calculate_demo_triangle_size(
                        clipped_game_rect.width, clipped_game_rect.height, env_config
                    )
                )
                if preview_tri_cell_w > 0 and preview_tri_cell_h > 0:
                    padding_grid = self.vis_config.ENV_GRID_PADDING
                    drawable_w = max(0, clipped_game_rect.width - 2 * padding_grid)
                    drawable_h = max(0, clipped_game_rect.height - 2 * padding_grid)
                    grid_cols_eff_width = env_config.COLS * 0.75 + 0.25
                    scale_w = drawable_w / max(1, grid_cols_eff_width)
                    scale_h = drawable_h / max(1, env_config.ROWS)
                    final_scale = min(scale_w, scale_h)
                    final_grid_pixel_w = max(1, grid_cols_eff_width * final_scale)
                    final_grid_pixel_h = max(1, env_config.ROWS * final_scale)
                    grid_ox = padding_grid + (drawable_w - final_grid_pixel_w) / 2
                    grid_oy = padding_grid + (drawable_h - final_grid_pixel_h) / 2
                    self._render_placement_preview(
                        game_surf,
                        demo_env,
                        preview_tri_cell_w,
                        preview_tri_cell_h,
                        grid_ox,
                        grid_oy,
                    )

                # Render shape previews
                preview_area_w = min(150, sw - clipped_game_rect.right - padding)
                if preview_area_w > 20:
                    preview_area_rect = pygame.Rect(
                        clipped_game_rect.right + padding // 2,
                        clipped_game_rect.top,
                        preview_area_w,
                        clipped_game_rect.height,
                    )
                    clipped_preview_area_rect = preview_area_rect.clip(
                        self.screen.get_rect()
                    )
                    if (
                        clipped_preview_area_rect.width > 0
                        and clipped_preview_area_rect.height > 0
                    ):
                        preview_area_surf = self.screen.subsurface(
                            clipped_preview_area_rect
                        )
                        self._render_demo_shape_previews(preview_area_surf, demo_env)
            except ValueError as e:
                print(f"Error subsurface demo game ({clipped_game_rect}): {e}")
                pygame.draw.rect(
                    self.screen, VisConfig.RED, clipped_game_rect, 1
                )  # LOG
            except Exception as render_e:
                print(f"Error rendering demo game area: {render_e}")
                traceback.print_exc()
                pygame.draw.rect(
                    self.screen, VisConfig.RED, clipped_game_rect, 1
                )  # LOG
        else:
            self._render_too_small_message(game_rect, game_w, game_h)

        # HUD
        hud_y = game_rect.bottom + 10
        score_text = f"Score: {demo_env.game_score} | Lines: {demo_env.lines_cleared_this_episode}"
        try:
            score_surf = self.demo_hud_font.render(score_text, True, VisConfig.WHITE)
            score_rect = score_surf.get_rect(midtop=(sw // 2, hud_y))
            self.screen.blit(score_surf, score_rect)
        except Exception as e:
            print(f"HUD render error: {e}")  # LOG

        # Help Text
        try:
            help_surf = self.demo_help_font.render(
                self.demo_config.HELP_TEXT, True, VisConfig.LIGHTG
            )
            help_rect = help_surf.get_rect(centerx=sw // 2, bottom=sh - 10)
            self.screen.blit(help_surf, help_rect)
        except Exception as e:
            print(f"Help render error: {e}")  # LOG

        # Don't flip here, let render_all handle it

    # --- Helper methods (_calculate_demo_triangle_size, _render_placement_preview, _render_demo_shape_previews, _render_too_small_message, _render_initializing_screen, _render_error_screen) remain unchanged ---
    def _calculate_demo_triangle_size(self, surf_w, surf_h, env_config):
        padding = self.vis_config.ENV_GRID_PADDING
        drawable_w = max(1, surf_w - 2 * padding)
        drawable_h = max(1, surf_h - 2 * padding)
        grid_rows = env_config.ROWS
        grid_cols_eff_width = env_config.COLS * 0.75 + 0.25
        if grid_rows <= 0 or grid_cols_eff_width <= 0:
            return 0, 0
        scale_w = drawable_w / grid_cols_eff_width
        scale_h = drawable_h / grid_rows
        final_scale = min(scale_w, scale_h)
        if final_scale <= 0:
            return 0, 0
        tri_cell_w = max(1, final_scale)
        tri_cell_h = max(1, final_scale)
        return int(tri_cell_w), int(tri_cell_h)

    def _render_placement_preview(
        self,
        surf: pygame.Surface,
        env: GameState,
        cell_w: int,
        cell_h: int,
        offset_x: float,
        offset_y: float,
    ):
        if cell_w <= 0 or cell_h <= 0:
            return
        shp, rr, cc = env.get_current_selection_info()
        if shp is None:
            return
        is_valid = env.grid.can_place(shp, rr, cc)
        preview_color = (
            self.demo_config.PREVIEW_COLOR
            if is_valid
            else self.demo_config.INVALID_PREVIEW_COLOR
        )
        temp_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        for dr, dc, up in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if env.grid.valid(nr, nc) and not env.grid.triangles[nr][nc].is_death:
                temp_tri = env.grid.triangles[nr][
                    nc
                ]  # Use actual triangle for orientation
                try:
                    pts = temp_tri.get_points(
                        ox=offset_x, oy=offset_y, cw=cell_w, ch=cell_h
                    )
                    pygame.draw.polygon(temp_surface, preview_color, pts)
                except Exception:
                    pass  # Reduce spam
        surf.blit(temp_surface, (0, 0))

    def _render_demo_shape_previews(self, surf: pygame.Surface, env: GameState):
        surf.fill((25, 25, 25))
        all_slots = env.shapes
        selected_shape_obj = (
            all_slots[env.demo_selected_shape_idx]
            if 0 <= env.demo_selected_shape_idx < len(all_slots)
            else None
        )
        num_slots = env.env_config.NUM_SHAPE_SLOTS
        surf_w, surf_h = surf.get_size()
        preview_padding = 5
        if num_slots <= 0:
            return
        preview_h = max(20, (surf_h - (num_slots + 1) * preview_padding) / num_slots)
        preview_w = max(20, surf_w - 2 * preview_padding)
        current_preview_y = preview_padding
        for i in range(num_slots):
            shp = all_slots[i] if i < len(all_slots) else None
            preview_rect = pygame.Rect(
                preview_padding, current_preview_y, preview_w, preview_h
            )
            clipped_preview_rect = preview_rect.clip(surf.get_rect())
            if clipped_preview_rect.width <= 0 or clipped_preview_rect.height <= 0:
                current_preview_y += preview_h + preview_padding
                continue
            bg_color = (40, 40, 40)
            border_color = VisConfig.GRAY
            border_width = 1
            if shp is not None and shp == selected_shape_obj:
                border_color = self.demo_config.SELECTED_SHAPE_HIGHLIGHT_COLOR
                border_width = 2
            pygame.draw.rect(surf, bg_color, clipped_preview_rect, border_radius=3)
            pygame.draw.rect(
                surf, border_color, clipped_preview_rect, border_width, border_radius=3
            )
            if shp is not None:
                try:
                    shape_sub_surf = surf.subsurface(clipped_preview_rect)
                    inner_padding = 2
                    shape_render_area_rect = pygame.Rect(
                        inner_padding,
                        inner_padding,
                        clipped_preview_rect.width - 2 * inner_padding,
                        clipped_preview_rect.height - 2 * inner_padding,
                    )
                    if (
                        shape_render_area_rect.width > 0
                        and shape_render_area_rect.height > 0
                    ):
                        shape_render_surf = shape_sub_surf.subsurface(
                            shape_render_area_rect
                        )
                        min_r, min_c, max_r, max_c = shp.bbox()
                        shape_h = max(1, max_r - min_r + 1)
                        shape_w_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)
                        scale_h = shape_render_surf.get_height() / shape_h
                        scale_w = shape_render_surf.get_width() / shape_w_eff
                        cell_size = max(1, min(scale_h, scale_w))
                        self.game_area._render_single_shape(
                            shape_render_surf, shp, int(cell_size)
                        )
                except ValueError as sub_err:
                    print(f"Error subsurface shape preview {i}: {sub_err}")
                    pygame.draw.rect(
                        surf, VisConfig.RED, clipped_preview_rect, 1
                    )  # LOG
                except Exception as e:
                    print(f"Error rendering demo shape preview {i}: {e}")
                    pygame.draw.rect(
                        surf, VisConfig.RED, clipped_preview_rect, 1
                    )  # LOG
            current_preview_y += preview_h + preview_padding

    def _render_too_small_message(self, area_rect: pygame.Rect, w: int, h: int):
        try:
            font = pygame.font.SysFont(None, 24)
            err_surf = font.render(f"Area Too Small ({w}x{h})", True, VisConfig.GRAY)
            target_rect = err_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(err_surf, target_rect)
        except Exception as e:
            print(f"Error rendering 'too small' message: {e}")  # LOG

    def _render_initializing_screen(self):
        try:
            self.screen.fill(VisConfig.BLACK)
            font = pygame.font.SysFont(None, 50)
            text_surf = font.render(
                "Initializing RL Components...", True, VisConfig.WHITE
            )
            text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text_surf, text_rect)
            pygame.display.flip()
        except Exception as e:
            print(f"Error rendering initializing screen: {e}")  # LOG

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
            pygame.display.flip()
        except Exception as e:
            print(f"Error rendering error screen: {e}")  # LOG
