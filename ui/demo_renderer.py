import pygame
import math
import traceback
from typing import Tuple, Dict

from config import (
    VisConfig,
    EnvConfig,
    DemoConfig,
    RED,
    BLUE,
    WHITE,
    LIGHTG,
    GRAY,
    DARK_RED,
    YELLOW,
    BLACK,
)
from config.constants import LINE_CLEAR_FLASH_COLOR, GAME_OVER_FLASH_COLOR

from environment.game_state import GameState
from environment.triangle import Triangle
from .panels.game_area import GameAreaRenderer


class DemoRenderer:
    """Handles rendering specifically for the interactive Demo Mode."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer
        self._init_demo_fonts()
        self.overlay_font = self.game_area_renderer.fonts.get(
            "env_overlay", pygame.font.Font(None, 36)
        )
        self.invalid_placement_color = (
            0,
            0,
            0,
            150,
        )
        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

    def _init_demo_fonts(self):
        """Initializes fonts used in demo mode."""
        try:
            self.demo_hud_font = pygame.font.SysFont(
                None, self.demo_config.HUD_FONT_SIZE
            )
            self.demo_help_font = pygame.font.SysFont(
                None, self.demo_config.HELP_FONT_SIZE
            )
            if not hasattr(
                self.game_area_renderer, "fonts"
            ) or not self.game_area_renderer.fonts.get("ui"):
                self.game_area_renderer._init_fonts()
        except Exception as e:
            print(f"Warning: SysFont error for demo fonts: {e}. Using default.")
            self.demo_hud_font = pygame.font.Font(None, self.demo_config.HUD_FONT_SIZE)
            self.demo_help_font = pygame.font.Font(
                None, self.demo_config.HELP_FONT_SIZE
            )

    def render(
        self, demo_env: GameState, env_config: EnvConfig, is_debug: bool = False
    ):
        """Renders the entire demo/debug mode screen."""
        if not demo_env:
            print("Error: DemoRenderer called with demo_env=None")
            return

        bg_color = self._determine_background_color(demo_env)
        self.screen.fill(bg_color)

        screen_width, screen_height = self.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30

        game_rect, clipped_game_rect = self._calculate_game_area_rect(
            screen_width, screen_height, padding, hud_height, help_height, env_config
        )

        if clipped_game_rect.width > 10 and clipped_game_rect.height > 10:
            self._render_game_area(
                demo_env, env_config, clipped_game_rect, bg_color, is_debug
            )
        else:
            self._render_too_small_message("Demo Area Too Small", clipped_game_rect)

        if not is_debug:
            self.shape_preview_rects = self._render_shape_previews_area(
                demo_env, screen_width, clipped_game_rect, padding
            )
        else:
            self.shape_preview_rects.clear()

        self._render_hud(demo_env, screen_width, game_rect.bottom + 10, is_debug)
        self._render_help_text(screen_width, screen_height, is_debug)

    def _determine_background_color(self, demo_env: GameState) -> Tuple[int, int, int]:
        """Determines the background color based on game state."""
        if demo_env.is_line_clearing():
            return LINE_CLEAR_FLASH_COLOR
        if demo_env.is_game_over_flashing():
            return GAME_OVER_FLASH_COLOR
        if demo_env.is_over():
            return DARK_RED
        if demo_env.is_frozen():
            return (30, 30, 100)
        return self.demo_config.BACKGROUND_COLOR

    def _calculate_game_area_rect(
        self,
        screen_width: int,
        screen_height: int,
        padding: int,
        hud_height: int,
        help_height: int,
        env_config: EnvConfig,
    ) -> Tuple[pygame.Rect, pygame.Rect]:
        """Calculates the main game area rectangle, maintaining aspect ratio."""
        max_game_h = screen_height - 2 * padding - hud_height - help_height
        max_game_w = screen_width - 2 * padding
        aspect_ratio = (env_config.COLS * 0.75 + 0.25) / max(1, env_config.ROWS)

        game_w = max_game_w
        game_h = game_w / aspect_ratio if aspect_ratio > 0 else max_game_h
        if game_h > max_game_h:
            game_h = max_game_h
            game_w = game_h * aspect_ratio

        game_w = math.floor(min(game_w, max_game_w))
        game_h = math.floor(min(game_h, max_game_h))
        game_x = (screen_width - game_w) // 2
        game_y = padding
        game_rect = pygame.Rect(game_x, game_y, game_w, game_h)
        clipped_game_rect = game_rect.clip(self.screen.get_rect())
        return game_rect, clipped_game_rect

    def _render_game_area(
        self,
        demo_env: GameState,
        env_config: EnvConfig,
        clipped_game_rect: pygame.Rect,
        bg_color: Tuple[int, int, int],
        is_debug: bool,
    ):
        """Renders the central game grid and placement preview."""
        try:
            game_surf = self.screen.subsurface(clipped_game_rect)
            game_surf.fill(bg_color)

            # Render the grid itself
            self.game_area_renderer._render_single_env_grid(
                game_surf, demo_env, env_config
            )

            if not is_debug:
                tri_cell_w, tri_cell_h = self._calculate_demo_triangle_size(
                    clipped_game_rect.width, clipped_game_rect.height, env_config
                )
                if tri_cell_w > 0 and tri_cell_h > 0:
                    grid_ox, grid_oy = self._calculate_grid_offset(
                        clipped_game_rect.width, clipped_game_rect.height, env_config
                    )
                    self._render_dragged_shape(
                        game_surf,
                        demo_env,
                        tri_cell_w,
                        tri_cell_h,
                        grid_ox,
                        grid_oy,
                        clipped_game_rect.topleft, 
                    )

            if demo_env.is_over():
                self._render_demo_overlay_text(game_surf, "GAME OVER", RED)
            elif demo_env.is_line_clearing():
                tris = demo_env.last_triangles_cleared
                score = demo_env.last_line_clear_score
                clear_msg = f"{tris} Triangles Cleared! (+{score:.2f} pts)"
                self._render_demo_overlay_text(game_surf, clear_msg, BLUE)

        except ValueError as e:
            print(f"Error subsurface demo game ({clipped_game_rect}): {e}")
            pygame.draw.rect(self.screen, RED, clipped_game_rect, 1)
        except Exception as render_e:
            print(f"Error rendering demo game area: {render_e}")
            traceback.print_exc()
            pygame.draw.rect(self.screen, RED, clipped_game_rect, 1)

    def _render_shape_previews_area(
        self,
        demo_env: GameState,
        screen_width: int,
        clipped_game_rect: pygame.Rect,
        padding: int,
    ) -> Dict[int, pygame.Rect]:
        """Renders the shape preview area. Returns dict of preview rects."""
        preview_rects: Dict[int, pygame.Rect] = {}
        preview_area_w = min(150, screen_width - clipped_game_rect.right - padding // 2)
        if preview_area_w <= 20:
            return preview_rects

        preview_area_rect = pygame.Rect(
            clipped_game_rect.right + padding // 2,
            clipped_game_rect.top,
            preview_area_w,
            clipped_game_rect.height,
        )
        clipped_preview_area_rect = preview_area_rect.clip(self.screen.get_rect())
        if (
            clipped_preview_area_rect.width <= 0
            or clipped_preview_area_rect.height <= 0
        ):
            return preview_rects

        try:
            preview_area_surf = self.screen.subsurface(clipped_preview_area_rect)
            # Store the calculated rects relative to the screen
            preview_rects = self._render_demo_shape_previews(
                preview_area_surf, demo_env, preview_area_rect.topleft
            )
        except ValueError as e:
            print(f"Error subsurface demo shape preview area: {e}")
            pygame.draw.rect(self.screen, RED, clipped_preview_area_rect, 1)
        except Exception as e:
            print(f"Error rendering demo shape previews: {e}")
            traceback.print_exc()
        return preview_rects  # Return the calculated rects


    def _render_hud(
        self, demo_env: GameState, screen_width: int, hud_y: int, is_debug: bool
    ):
        """Renders the score and triangles cleared HUD."""
        if is_debug:
            hud_text = (
                f"DEBUG MODE | Tris Cleared: {demo_env.triangles_cleared_this_episode}"
            )
        else:
            hud_text = f"Score: {demo_env.game_score} | Tris Cleared: {demo_env.triangles_cleared_this_episode}"
        try:
            hud_surf = self.demo_hud_font.render(hud_text, True, WHITE)
            hud_rect = hud_surf.get_rect(midtop=(screen_width // 2, hud_y))
            self.screen.blit(hud_surf, hud_rect)
        except Exception as e:
            print(f"HUD render error: {e}")

    def _render_help_text(self, screen_width: int, screen_height: int, is_debug: bool):
        """Renders the control help text at the bottom."""
        if is_debug:
            help_text = self.demo_config.DEBUG_HELP_TEXT
        else:
            help_text = (
                "[Click Preview]=Select/Deselect | [Click Grid]=Place | [ESC]=Exit"
            )
        try:
            help_surf = self.demo_help_font.render(help_text, True, LIGHTG)
            help_rect = help_surf.get_rect(
                centerx=screen_width // 2, bottom=screen_height - 10
            )
            self.screen.blit(help_surf, help_rect)
        except Exception as e:
            print(f"Help render error: {e}")

    def _render_demo_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders centered overlay text (e.g., GAME OVER)."""
        try:
            # Adjust font size dynamically based on surface width and text length
            max_width = surf.get_width() * 0.9
            font_size = 36
            overlay_font = self.overlay_font  # Start with default
            # Attempt to recreate font with adjusted size
            try:
                overlay_font = pygame.font.SysFont(None, font_size)
            except Exception:
                overlay_font = pygame.font.Font(None, font_size)  # Fallback

            text_surf = overlay_font.render(text, True, WHITE)
            while text_surf.get_width() > max_width and font_size > 10:
                font_size -= 2
                try:
                    overlay_font = pygame.font.SysFont(None, font_size)
                except Exception:
                    overlay_font = pygame.font.Font(None, font_size)
                text_surf = overlay_font.render(text, True, WHITE)

            # Add background color with alpha
            bg_color_rgba = (color[0] // 2, color[1] // 2, color[2] // 2, 220)
            text_surf_with_bg = overlay_font.render(text, True, WHITE, bg_color_rgba)

            text_rect = text_surf_with_bg.get_rect(center=surf.get_rect().center)
            surf.blit(text_surf_with_bg, text_rect)
        except Exception as e:
            print(f"Error rendering demo overlay text '{text}': {e}")

    def _calculate_demo_triangle_size(
        self, surf_w: int, surf_h: int, env_config: EnvConfig
    ) -> Tuple[int, int]:
        """Calculates the size of triangles for rendering within the demo area."""
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
        tri_cell_size = max(1, int(final_scale))
        return (
            tri_cell_size,
            tri_cell_size,
        )

    def _calculate_grid_offset(
        self, surf_w: int, surf_h: int, env_config: EnvConfig
    ) -> Tuple[float, float]:
        """Calculates the top-left offset for centering the grid rendering."""
        padding = self.vis_config.ENV_GRID_PADDING
        drawable_w = max(1, surf_w - 2 * padding)
        drawable_h = max(1, surf_h - 2 * padding)
        grid_rows = env_config.ROWS
        grid_cols_eff_width = env_config.COLS * 0.75 + 0.25
        if grid_rows <= 0 or grid_cols_eff_width <= 0:
            return float(padding), float(padding)

        scale_w = drawable_w / grid_cols_eff_width
        scale_h = drawable_h / grid_rows
        final_scale = min(scale_w, scale_h)
        final_grid_pixel_w = max(1, grid_cols_eff_width * final_scale)
        final_grid_pixel_h = max(1, grid_rows * final_scale)
        grid_ox = padding + (drawable_w - final_grid_pixel_w) / 2
        grid_oy = padding + (drawable_h - final_grid_pixel_h) / 2
        return grid_ox, grid_oy

    def _render_dragged_shape(
        self,
        surf: pygame.Surface,
        env: GameState,
        cell_w: int,
        cell_h: int,
        grid_offset_x: float,
        grid_offset_y: float,
        game_area_offset: Tuple[
            int, int
        ],  # Top-left of the game area surface relative to screen
    ):
        """Renders the shape being dragged, either snapped or following the mouse."""
        if cell_w <= 0 or cell_h <= 0:
            return
        dragged_shape, snapped_pos = env.get_dragged_shape_info()
        if dragged_shape is None:
            return

        is_valid_placement = snapped_pos is not None
        preview_alpha = 150
        preview_color_rgba = self.invalid_placement_color
        if is_valid_placement:
            shape_rgb = dragged_shape.color
            preview_color_rgba = (
                shape_rgb[0],
                shape_rgb[1],
                shape_rgb[2],
                preview_alpha,
            )
        else:
            # Use a different color/alpha if not snapped/invalid
            preview_color_rgba = (50, 50, 50, 100)

        temp_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))

        # Determine the reference point (top-left of where the shape's (0,0) relative coord would be)
        ref_x, ref_y = 0, 0
        if snapped_pos:
            # Render at snapped grid position
            snap_r, snap_c = snapped_pos
            ref_x = grid_offset_x + snap_c * (cell_w * 0.75)
            ref_y = grid_offset_y + snap_r * cell_h
        else:
            # Render centered on mouse cursor
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Adjust mouse pos relative to the game_surf
            mouse_x -= game_area_offset[0]
            mouse_y -= game_area_offset[1]

            # Calculate shape dimensions in pixels to center it
            min_r, min_c, max_r, max_c = dragged_shape.bbox()
            shape_h_cells = max_r - min_r + 1
            shape_w_cells_eff = (max_c - min_c + 1) * 0.75 + 0.25
            shape_pixel_w = shape_w_cells_eff * cell_w
            shape_pixel_h = shape_h_cells * cell_h

            # Offset mouse position to center the shape bbox
            # The reference point needs to be adjusted based on the shape's min_r/min_c
            ref_x = mouse_x - (shape_pixel_w / 2) - (min_c * cell_w * 0.75)
            ref_y = mouse_y - (shape_pixel_h / 2) - (min_r * cell_h)

        # Draw the triangles relative to the reference point
        for dr, dc, is_up in dragged_shape.triangles:
            # Calculate triangle top-left corner based on its relative position
            tri_x = ref_x + dc * (cell_w * 0.75)
            tri_y = ref_y + dr * cell_h

            # Use Triangle class logic to get points relative to this corner
            temp_tri = Triangle(0, 0, is_up)  # Dummy triangle for point calculation
            try:
                points = temp_tri.get_points(ox=tri_x, oy=tri_y, cw=cell_w, ch=cell_h)
                pygame.draw.polygon(temp_surface, preview_color_rgba, points)
                # Optionally draw border
                # pygame.draw.polygon(temp_surface, (200, 200, 200, 180), points, 1)
            except Exception:
                pass  # Ignore drawing errors for single triangles

        surf.blit(temp_surface, (0, 0))


    def _render_demo_shape_previews(
        self, surf: pygame.Surface, env: GameState, area_topleft: Tuple[int, int]
    ) -> Dict[int, pygame.Rect]:
        """Renders the small previews of available shapes. Returns dict of screen rects."""
        calculated_rects: Dict[int, pygame.Rect] = {}  # Store rects relative to screen
        surf.fill((25, 25, 25))
        all_slots = env.shapes
        selected_idx = env.demo_selected_shape_idx  # Highlight based on selection
        dragged_idx = env.demo_dragged_shape_idx  # Could also highlight dragged

        num_slots = env.env_config.NUM_SHAPE_SLOTS
        surf_w, surf_h = surf.get_size()
        preview_padding = 5

        if num_slots <= 0:
            return calculated_rects

        preview_h = max(20, (surf_h - (num_slots + 1) * preview_padding) / num_slots)
        preview_w = max(20, surf_w - 2 * preview_padding)
        current_preview_y = preview_padding

        for i in range(num_slots):
            shape_in_slot = all_slots[i] if i < len(all_slots) else None
            # Rect relative to the preview surface
            preview_rect_local = pygame.Rect(
                preview_padding, current_preview_y, preview_w, preview_h
            )
            # Rect relative to the main screen
            preview_rect_screen = preview_rect_local.move(area_topleft)
            calculated_rects[i] = preview_rect_screen  # Store screen rect

            clipped_preview_rect = preview_rect_local.clip(surf.get_rect())
            if clipped_preview_rect.width <= 0 or clipped_preview_rect.height <= 0:
                current_preview_y += preview_h + preview_padding
                continue

            bg_color = (40, 40, 40)
            border_color = GRAY
            border_width = 1
            # Highlight if selected OR dragged
            if i == selected_idx and shape_in_slot is not None:
                border_color = self.demo_config.SELECTED_SHAPE_HIGHLIGHT_COLOR
                border_width = 3  # Make highlight thicker
            elif i == dragged_idx:  # Optional: different highlight for dragged
                border_color = (100, 100, 255)  # Lighter blue
                border_width = 2

            pygame.draw.rect(surf, bg_color, clipped_preview_rect, border_radius=3)
            pygame.draw.rect(
                surf, border_color, clipped_preview_rect, border_width, border_radius=3
            )

            if shape_in_slot is not None:
                self._render_single_shape_in_preview_box(
                    surf, shape_in_slot, preview_rect_local, clipped_preview_rect
                )

            current_preview_y += preview_h + preview_padding
        return calculated_rects


    def _render_single_shape_in_preview_box(
        self,
        surf: pygame.Surface,
        shape_obj,
        preview_rect: pygame.Rect,
        clipped_preview_rect: pygame.Rect,
    ):
        """Renders a single shape scaled to fit within its preview box."""
        try:
            inner_padding = 2
            shape_render_area_rect = pygame.Rect(
                inner_padding,
                inner_padding,
                clipped_preview_rect.width - 2 * inner_padding,
                clipped_preview_rect.height - 2 * inner_padding,
            )
            if shape_render_area_rect.width <= 0 or shape_render_area_rect.height <= 0:
                return

            # Render onto a temporary surface relative to the preview_rect
            temp_surf = pygame.Surface(shape_render_area_rect.size, pygame.SRCALPHA)
            temp_surf.fill((0, 0, 0, 0))  # Transparent background

            min_r, min_c, max_r, max_c = shape_obj.bbox()
            shape_h_cells = max(1, max_r - min_r + 1)
            shape_w_cells_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)
            scale_h = shape_render_area_rect.height / shape_h_cells
            scale_w = shape_render_area_rect.width / shape_w_cells_eff
            cell_size = max(1, min(scale_h, scale_w))

            # Render shape centered on the temporary surface
            self.game_area_renderer._render_single_shape(
                temp_surf, shape_obj, int(cell_size)
            )

            # Blit the temporary surface onto the main preview surface
            surf.blit(
                temp_surf, shape_render_area_rect.move(preview_rect.topleft).topleft
            )

        except ValueError as sub_err:
            print(f"Error subsurface shape preview: {sub_err}")
            pygame.draw.rect(surf, RED, clipped_preview_rect, 1)
        except Exception as e:
            print(f"Error rendering demo shape preview: {e}")
            pygame.draw.rect(surf, RED, clipped_preview_rect, 1)

    def _render_too_small_message(self, text: str, area_rect: pygame.Rect):
        """Renders a message indicating the area is too small."""
        try:
            font = self.game_area_renderer.fonts.get("ui", pygame.font.Font(None, 24))
            err_surf = font.render(text, True, GRAY)
            target_rect = err_surf.get_rect(center=area_rect.center)
            self.screen.blit(err_surf, target_rect)
        except Exception as e:
            print(f"Error rendering 'too small' message: {e}")

    def get_shape_preview_rects(self) -> Dict[int, pygame.Rect]:
        """Returns the dictionary of screen-relative shape preview rects."""
        return self.shape_preview_rects.copy()
