# File: ui/demo_components/grid_renderer.py
import pygame
import math
import traceback
from typing import Tuple

from config import (
    VisConfig,
    EnvConfig,
    DemoConfig,
    RED,
    BLUE,
    WHITE,
    GRAY,
    BLACK,
)  # Added WHITE, GRAY, BLACK
from config.constants import LINE_CLEAR_FLASH_COLOR, GAME_OVER_FLASH_COLOR
from environment.game_state import GameState
from environment.triangle import Triangle
from ui.panels.game_area import GameAreaRenderer  # Import for base rendering


class DemoGridRenderer:
    """Renders the main game grid area for Demo/Debug mode."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,  # Use fonts/methods from GameAreaRenderer
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer  # Store reference
        self.overlay_font = self.game_area_renderer.fonts.get(
            "env_overlay", pygame.font.Font(None, 36)
        )
        self.invalid_placement_color = (0, 0, 0, 150)

    def calculate_game_area_rect(
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

    def render_game_area(
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

            # Calculate grid rendering parameters
            tri_cell_w, tri_cell_h = self.calculate_demo_triangle_size(
                clipped_game_rect.width, clipped_game_rect.height, env_config
            )
            if tri_cell_w > 0 and tri_cell_h > 0:
                grid_ox, grid_oy = self.calculate_grid_offset(
                    clipped_game_rect.width, clipped_game_rect.height, env_config
                )

                # Render the grid directly using GameState object
                self._render_demo_grid_from_gamestate(
                    game_surf, demo_env, tri_cell_w, tri_cell_h, grid_ox, grid_oy
                )

                # Render dragged shape if not in debug mode
                if not is_debug:
                    self._render_dragged_shape(
                        game_surf,
                        demo_env,
                        tri_cell_w,
                        tri_cell_h,
                        grid_ox,
                        grid_oy,
                        clipped_game_rect.topleft,
                    )

            # Render overlays (Game Over, Line Clear) - using HUD renderer's logic if needed
            if demo_env.is_over():
                self._render_demo_overlay_text(game_surf, "GAME OVER", RED)
            elif demo_env.is_line_clearing() and demo_env.last_line_clear_info:
                lines, tris, score = demo_env.last_line_clear_info
                line_str = "Line" if lines == 1 else "Lines"
                clear_msg = (
                    f"{lines} {line_str} Cleared! ({tris} Tris, +{score:.2f} pts)"
                )
                self._render_demo_overlay_text(game_surf, clear_msg, BLUE)

        except ValueError as e:
            print(f"Error subsurface demo game ({clipped_game_rect}): {e}")
            pygame.draw.rect(self.screen, RED, clipped_game_rect, 1)
        except Exception as render_e:
            print(f"Error rendering demo game area: {render_e}")
            traceback.print_exc()
            pygame.draw.rect(self.screen, RED, clipped_game_rect, 1)

    def _render_demo_grid_from_gamestate(
        self,
        surf: pygame.Surface,
        env: GameState,
        cell_w: int,
        cell_h: int,
        grid_offset_x: float,
        grid_offset_y: float,
    ):
        """Renders the grid directly from the GameState object."""
        grid = env.grid
        for r in range(grid.rows):
            for c in range(grid.cols):
                if not (
                    0 <= r < len(grid.triangles) and 0 <= c < len(grid.triangles[r])
                ):
                    continue
                t = grid.triangles[r][c]
                if t.is_death:
                    continue  # Don't draw death cells

                try:
                    pts = t.get_points(
                        ox=grid_offset_x, oy=grid_offset_y, cw=cell_w, ch=cell_h
                    )
                    color = self.vis_config.LIGHTG  # Default empty color
                    if t.is_occupied:
                        color = (
                            t.color if t.color else self.vis_config.RED
                        )  # Use shape color or fallback

                    # Highlight cleared triangles
                    is_highlighted = (
                        env.is_highlighting_cleared()
                        and (r, c) in env.cleared_triangles_coords
                    )
                    if is_highlighted:
                        highlight_color = self.vis_config.LINE_CLEAR_HIGHLIGHT_COLOR
                        # Draw slightly larger background polygon
                        center_x = sum(p[0] for p in pts) / 3
                        center_y = sum(p[1] for p in pts) / 3
                        scale_factor = 1.2
                        highlight_pts = [
                            (
                                center_x + (p[0] - center_x) * scale_factor,
                                center_y + (p[1] - center_y) * scale_factor,
                            )
                            for p in pts
                        ]
                        pygame.draw.polygon(
                            surf,
                            (
                                highlight_color[0],
                                highlight_color[1],
                                highlight_color[2],
                            ),
                            highlight_pts,
                        )  # Use RGB for solid bg

                    # Draw the main triangle
                    pygame.draw.polygon(surf, color, pts)
                    # Draw border
                    pygame.draw.polygon(surf, self.vis_config.GRAY, pts, 1)

                except Exception as tri_err:
                    # logger.info(f"Error drawing triangle ({r},{c}): {tri_err}") # Use debug level
                    pass  # Ignore errors for single triangles

    def calculate_demo_triangle_size(
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
        return tri_cell_size, tri_cell_size

    def calculate_grid_offset(
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
        game_area_offset: Tuple[int, int],
    ):
        """Renders the shape being dragged, either snapped or following the mouse."""
        if cell_w <= 0 or cell_h <= 0:
            return
        dragged_shape, snapped_pos = env.get_dragged_shape_info()
        if dragged_shape is None:
            return

        is_valid_placement = snapped_pos is not None
        preview_alpha = 150
        if is_valid_placement:
            shape_rgb = dragged_shape.color
            preview_color_rgba = (
                shape_rgb[0],
                shape_rgb[1],
                shape_rgb[2],
                preview_alpha,
            )
        else:
            preview_color_rgba = (50, 50, 50, 100)  # Invalid placement color

        temp_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))

        ref_x, ref_y = 0, 0
        if snapped_pos:
            snap_r, snap_c = snapped_pos
            ref_x = grid_offset_x + snap_c * (cell_w * 0.75)
            ref_y = grid_offset_y + snap_r * cell_h
        else:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            mouse_x -= game_area_offset[0]
            mouse_y -= game_area_offset[1]
            min_r, min_c, max_r, max_c = dragged_shape.bbox()
            shape_h_cells = max_r - min_r + 1
            shape_w_cells_eff = (max_c - min_c + 1) * 0.75 + 0.25
            shape_pixel_w = shape_w_cells_eff * cell_w
            shape_pixel_h = shape_h_cells * cell_h
            ref_x = mouse_x - (shape_pixel_w / 2) - (min_c * cell_w * 0.75)
            ref_y = mouse_y - (shape_pixel_h / 2) - (min_r * cell_h)

        for dr, dc, is_up in dragged_shape.triangles:
            tri_x = ref_x + dc * (cell_w * 0.75)
            tri_y = ref_y + dr * cell_h
            temp_tri = Triangle(0, 0, is_up)
            try:
                points = temp_tri.get_points(ox=tri_x, oy=tri_y, cw=cell_w, ch=cell_h)
                pygame.draw.polygon(temp_surface, preview_color_rgba, points)
                # Optionally draw border for preview
                # pygame.draw.polygon(temp_surface, (200, 200, 200, 100), points, 1)
            except Exception:
                pass

        surf.blit(temp_surface, (0, 0))

    def _render_demo_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders centered overlay text (e.g., GAME OVER)."""
        try:
            font = self.overlay_font  # Use the font initialized for overlays
            if not font:
                return

            max_w = surf.get_width() * 0.9
            original_size = font.get_height()
            current_size = original_size

            surf_txt = font.render(text, True, WHITE)
            while surf_txt.get_width() > max_w and current_size > 8:
                current_size -= 2
                try:
                    font = pygame.font.SysFont(None, current_size)
                except:
                    font = pygame.font.Font(None, current_size)
                surf_txt = font.render(text, True, WHITE)

            bg_rgba = (color[0] // 2, color[1] // 2, color[2] // 2, 220)
            surf_bg = font.render(text, True, WHITE, bg_rgba)
            rect = surf_bg.get_rect(center=surf.get_rect().center)
            surf.blit(surf_bg, rect)
        except Exception as e:
            print(f"Error rendering overlay '{text}': {e}")
