# File: ui/demo_renderer.py
import pygame
import math
import traceback
from typing import Optional, Tuple

from config import VisConfig, EnvConfig, DemoConfig
from environment.game_state import GameState
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
        self.overlay_font = self.game_area_renderer.fonts.get("env_overlay")
        if not self.overlay_font:
            print("Warning: DemoRenderer could not get overlay font. Using default.")
            self.overlay_font = pygame.font.Font(None, 36)
        # --- NEW: Define invalid placement color ---
        self.invalid_placement_color = (0, 0, 0, 150)  # Transparent Gray
        # --- END NEW ---

    def _init_demo_fonts(self):
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

    def render(self, demo_env: GameState, env_config: EnvConfig):
        if not demo_env:
            print("Error: DemoRenderer called with demo_env=None")
            return

        bg_color = self._determine_background_color(demo_env)
        self.screen.fill(bg_color)

        sw, sh = self.screen.get_size()
        padding = 30
        hud_height = 60
        help_height = 30
        max_game_h = sh - 2 * padding - hud_height - help_height
        max_game_w = sw - 2 * padding

        if max_game_h <= 0 or max_game_w <= 0:
            self._render_too_small_message(
                "Demo Area Too Small", self.screen.get_rect()
            )
            return

        game_rect, clipped_game_rect = self._calculate_game_area_rect(
            sw, sh, padding, hud_height, help_height, env_config
        )

        if clipped_game_rect.width > 10 and clipped_game_rect.height > 10:
            self._render_game_area(demo_env, env_config, clipped_game_rect, bg_color)
        else:
            self._render_too_small_message("Demo Area Too Small", clipped_game_rect)

        self._render_shape_previews_area(demo_env, sw, clipped_game_rect, padding)
        self._render_hud(demo_env, sw, game_rect.bottom + 10)
        self._render_help_text(sw, sh)

    def _determine_background_color(self, demo_env: GameState) -> Tuple[int, int, int]:
        if demo_env.is_line_clearing():
            return VisConfig.LINE_CLEAR_FLASH_COLOR
        elif demo_env.is_game_over_flashing():
            return VisConfig.GAME_OVER_FLASH_COLOR
        elif demo_env.is_over():
            return VisConfig.DARK_RED
        elif demo_env.is_frozen():
            return (30, 30, 100)
        else:
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
    ):
        try:
            game_surf = self.screen.subsurface(clipped_game_rect)
            game_surf.fill(bg_color)

            self.game_area_renderer._render_single_env_grid(
                game_surf, demo_env, env_config
            )

            preview_tri_cell_w, preview_tri_cell_h = self._calculate_demo_triangle_size(
                clipped_game_rect.width, clipped_game_rect.height, env_config
            )
            if preview_tri_cell_w > 0 and preview_tri_cell_h > 0:
                grid_ox, grid_oy = self._calculate_grid_offset(
                    clipped_game_rect.width, clipped_game_rect.height, env_config
                )
                self._render_placement_preview(
                    game_surf,
                    demo_env,
                    preview_tri_cell_w,
                    preview_tri_cell_h,
                    grid_ox,
                    grid_oy,
                )

            if demo_env.is_over():
                self._render_demo_overlay_text(game_surf, "GAME OVER", VisConfig.RED)
            elif demo_env.is_line_clearing():
                self._render_demo_overlay_text(game_surf, "Line Clear!", VisConfig.BLUE)

        except ValueError as e:
            print(f"Error subsurface demo game ({clipped_game_rect}): {e}")
            pygame.draw.rect(self.screen, VisConfig.RED, clipped_game_rect, 1)
        except Exception as render_e:
            print(f"Error rendering demo game area: {render_e}")
            traceback.print_exc()
            pygame.draw.rect(self.screen, VisConfig.RED, clipped_game_rect, 1)

    def _render_shape_previews_area(
        self,
        demo_env: GameState,
        screen_width: int,
        clipped_game_rect: pygame.Rect,
        padding: int,
    ):
        preview_area_w = min(150, screen_width - clipped_game_rect.right - padding // 2)
        if preview_area_w > 20:
            preview_area_rect = pygame.Rect(
                clipped_game_rect.right + padding // 2,
                clipped_game_rect.top,
                preview_area_w,
                clipped_game_rect.height,
            )
            clipped_preview_area_rect = preview_area_rect.clip(self.screen.get_rect())
            if (
                clipped_preview_area_rect.width > 0
                and clipped_preview_area_rect.height > 0
            ):
                try:
                    preview_area_surf = self.screen.subsurface(
                        clipped_preview_area_rect
                    )
                    self._render_demo_shape_previews(preview_area_surf, demo_env)
                except ValueError as e:
                    print(f"Error subsurface demo shape preview area: {e}")
                    pygame.draw.rect(
                        self.screen, VisConfig.RED, clipped_preview_area_rect, 1
                    )
                except Exception as e:
                    print(f"Error rendering demo shape previews: {e}")
                    traceback.print_exc()

    def _render_hud(self, demo_env: GameState, screen_width: int, hud_y: int):
        score_text = f"Score: {demo_env.game_score} | Lines: {demo_env.lines_cleared_this_episode}"
        try:
            score_surf = self.demo_hud_font.render(score_text, True, VisConfig.WHITE)
            score_rect = score_surf.get_rect(midtop=(screen_width // 2, hud_y))
            self.screen.blit(score_surf, score_rect)
        except Exception as e:
            print(f"HUD render error: {e}")

    def _render_help_text(self, screen_width: int, screen_height: int):
        try:
            help_surf = self.demo_help_font.render(
                self.demo_config.HELP_TEXT, True, VisConfig.LIGHTG
            )
            help_rect = help_surf.get_rect(
                centerx=screen_width // 2, bottom=screen_height - 10
            )
            self.screen.blit(help_surf, help_rect)
        except Exception as e:
            print(f"Help render error: {e}")

    def _render_demo_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        try:
            if not self.overlay_font:
                print("Error: Overlay font not available for demo overlay.")
                return
            text_surf = self.overlay_font.render(
                text,
                True,
                VisConfig.WHITE,
                (color[0] // 2, color[1] // 2, color[2] // 2, 220),
            )
            text_rect = text_surf.get_rect(center=surf.get_rect().center)
            surf.blit(text_surf, text_rect)
        except Exception as e:
            print(f"Error rendering demo overlay text '{text}': {e}")

    def _calculate_demo_triangle_size(
        self, surf_w: int, surf_h: int, env_config: EnvConfig
    ) -> Tuple[int, int]:
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

    def _calculate_grid_offset(
        self, surf_w: int, surf_h: int, env_config: EnvConfig
    ) -> Tuple[float, float]:
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

        preview_alpha = 150
        if is_valid:
            shape_rgb = shp.color
            preview_color_to_use = (
                shape_rgb[0],
                shape_rgb[1],
                shape_rgb[2],
                preview_alpha,
            )
        else:
            # --- MODIFIED: Use the defined gray color for invalid placement ---
            preview_color_to_use = self.invalid_placement_color
            # --- END MODIFIED ---

        temp_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))

        for dr, dc, up in shp.triangles:
            nr, nc = rr + dr, cc + dc
            if (
                env.grid.valid(nr, nc)
                and 0 <= nr < len(env.grid.triangles)
                and 0 <= nc < len(env.grid.triangles[nr])
                and not env.grid.triangles[nr][nc].is_death
            ):
                temp_tri = env.grid.triangles[nr][nc]
                try:
                    pts = temp_tri.get_points(
                        ox=offset_x, oy=offset_y, cw=cell_w, ch=cell_h
                    )
                    pygame.draw.polygon(temp_surface, preview_color_to_use, pts)
                except Exception as e:
                    pass

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
                self._render_single_shape_in_preview_box(
                    surf, shp, preview_rect, clipped_preview_rect
                )

            current_preview_y += preview_h + preview_padding

    def _render_single_shape_in_preview_box(
        self,
        surf: pygame.Surface,
        shp,
        preview_rect: pygame.Rect,
        clipped_preview_rect: pygame.Rect,
    ):
        try:
            inner_padding = 2
            shape_render_area_rect = pygame.Rect(
                inner_padding,
                inner_padding,
                clipped_preview_rect.width - 2 * inner_padding,
                clipped_preview_rect.height - 2 * inner_padding,
            )
            if shape_render_area_rect.width > 0 and shape_render_area_rect.height > 0:
                sub_surf_x = preview_rect.left + shape_render_area_rect.left
                sub_surf_y = preview_rect.top + shape_render_area_rect.top
                shape_sub_surf = surf.subsurface(
                    sub_surf_x,
                    sub_surf_y,
                    shape_render_area_rect.width,
                    shape_render_area_rect.height,
                )
                min_r, min_c, max_r, max_c = shp.bbox()
                shape_h = max(1, max_r - min_r + 1)
                shape_w_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)

                scale_h = shape_render_area_rect.height / shape_h
                scale_w = shape_render_area_rect.width / shape_w_eff
                cell_size = max(1, min(scale_h, scale_w))

                self.game_area_renderer._render_single_shape(
                    shape_sub_surf, shp, int(cell_size)
                )
        except ValueError as sub_err:
            print(f"Error subsurface shape preview: {sub_err}")
            pygame.draw.rect(surf, VisConfig.RED, clipped_preview_rect, 1)
        except Exception as e:
            print(f"Error rendering demo shape preview: {e}")
            pygame.draw.rect(surf, VisConfig.RED, clipped_preview_rect, 1)

    def _render_too_small_message(self, text: str, area_rect: pygame.Rect):
        try:
            font = self.game_area_renderer.fonts.get("ui") or pygame.font.SysFont(
                None, 24
            )
            err_surf = font.render(text, True, VisConfig.GRAY)
            target_rect = err_surf.get_rect(center=area_rect.center)
            self.screen.blit(err_surf, target_rect)
        except Exception as e:
            print(f"Error rendering 'too small' message: {e}")
