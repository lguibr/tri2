# File: ui/demo_renderer.py
import pygame
import math
import traceback
from typing import Optional, Tuple

from config import VisConfig, EnvConfig, DemoConfig
from environment.game_state import GameState
from .panels.game_area import GameAreaRenderer  # Need this for grid rendering


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
        self.game_area_renderer = (
            game_area_renderer  # Use existing grid/shape render logic
        )
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
            # Ensure the game_area_renderer fonts are also loaded if needed here
            if not hasattr(
                self.game_area_renderer, "fonts"
            ) or not self.game_area_renderer.fonts.get("ui"):
                self.game_area_renderer._init_fonts()  # Ensure base fonts are loaded
        except Exception as e:
            print(f"Warning: SysFont error for demo fonts: {e}. Using default.")
            self.demo_hud_font = pygame.font.Font(None, self.demo_config.HUD_FONT_SIZE)
            self.demo_help_font = pygame.font.Font(
                None, self.demo_config.HELP_FONT_SIZE
            )

    def render(self, demo_env: GameState, env_config: EnvConfig):
        """Renders the single-player interactive demo mode. Does NOT flip display."""
        if not demo_env:
            print("Error: DemoRenderer called with demo_env=None")
            # Optionally render an error message here
            return

        self.screen.fill(self.demo_config.BACKGROUND_COLOR)
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

        # 1. Render Game Area (Grid + Preview)
        if clipped_game_rect.width > 10 and clipped_game_rect.height > 10:
            try:
                game_surf = self.screen.subsurface(clipped_game_rect)
                # Render the main grid using GameAreaRenderer's method
                self.game_area_renderer._render_single_env_grid(
                    game_surf, demo_env, env_config
                )

                # Render Placement Preview on top of the grid
                preview_tri_cell_w, preview_tri_cell_h = (
                    self._calculate_demo_triangle_size(
                        clipped_game_rect.width, clipped_game_rect.height, env_config
                    )
                )
                if preview_tri_cell_w > 0 and preview_tri_cell_h > 0:
                    # Calculate offset needed for _render_placement_preview based on scaling
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
                        game_surf,  # Draw on the game surface
                        demo_env,
                        preview_tri_cell_w,
                        preview_tri_cell_h,
                        grid_ox,
                        grid_oy,
                    )

            except ValueError as e:
                print(f"Error subsurface demo game ({clipped_game_rect}): {e}")
                pygame.draw.rect(self.screen, VisConfig.RED, clipped_game_rect, 1)
            except Exception as render_e:
                print(f"Error rendering demo game area: {render_e}")
                traceback.print_exc()
                pygame.draw.rect(self.screen, VisConfig.RED, clipped_game_rect, 1)
        else:
            self._render_too_small_message("Demo Area Too Small", clipped_game_rect)

        # 2. Render Shape Previews (Right Side)
        preview_area_w = min(150, sw - clipped_game_rect.right - padding // 2)
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

        # 3. Render HUD (Below Game Area)
        hud_y = game_rect.bottom + 10
        score_text = f"Score: {demo_env.game_score} | Lines: {demo_env.lines_cleared_this_episode}"
        try:
            score_surf = self.demo_hud_font.render(score_text, True, VisConfig.WHITE)
            score_rect = score_surf.get_rect(midtop=(sw // 2, hud_y))
            self.screen.blit(score_surf, score_rect)
        except Exception as e:
            print(f"HUD render error: {e}")

        # 4. Render Help Text (Bottom)
        try:
            help_surf = self.demo_help_font.render(
                self.demo_config.HELP_TEXT, True, VisConfig.LIGHTG
            )
            help_rect = help_surf.get_rect(centerx=sw // 2, bottom=sh - 10)
            self.screen.blit(help_surf, help_rect)
        except Exception as e:
            print(f"Help render error: {e}")

    def _calculate_demo_triangle_size(
        self, surf_w, surf_h, env_config: EnvConfig
    ) -> Tuple[int, int]:
        """Calculates the cell width/height for triangles within a given surface."""
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
        # Use integer scaling for simplicity in drawing
        tri_cell_size = max(1, int(final_scale))
        return tri_cell_size, tri_cell_size

    def _render_placement_preview(
        self,
        surf: pygame.Surface,
        env: GameState,
        cell_w: int,
        cell_h: int,
        offset_x: float,
        offset_y: float,
    ):
        """Renders the preview of where the selected shape will be placed."""
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

        # Create a temporary surface with alpha for drawing the preview
        temp_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))  # Transparent

        for dr, dc, up in shp.triangles:
            nr, nc = rr + dr, cc + dc
            # Check if the target triangle exists and is not a death cell
            if (
                env.grid.valid(nr, nc)
                and 0 <= nr < len(env.grid.triangles)
                and 0 <= nc < len(env.grid.triangles[nr])
                and not env.grid.triangles[nr][nc].is_death
            ):
                # Use the actual triangle from the grid to get correct orientation for points
                temp_tri = env.grid.triangles[nr][nc]
                try:
                    pts = temp_tri.get_points(
                        ox=offset_x, oy=offset_y, cw=cell_w, ch=cell_h
                    )
                    pygame.draw.polygon(temp_surface, preview_color, pts)
                except Exception as e:
                    # Reduce log spam for minor render errors during preview
                    # print(f"Error rendering preview tri ({nr},{nc}): {e}")
                    pass

        # Blit the transparent preview surface onto the main game surface
        surf.blit(temp_surface, (0, 0))

    def _render_demo_shape_previews(self, surf: pygame.Surface, env: GameState):
        """Renders the small previews of available shapes in the side area."""
        surf.fill((25, 25, 25))  # Background for shape preview area
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

        # Calculate layout: Vertical list of previews
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

            # Determine background and border based on selection
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

            # Render the shape itself inside the preview box
            if shp is not None:
                try:
                    # Create a subsurface for the shape rendering area inside the box
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
                        # Use subsurface from the main surf, not clipped_preview_rect surface
                        shape_sub_surf = surf.subsurface(
                            preview_rect.left + shape_render_area_rect.left,
                            preview_rect.top + shape_render_area_rect.top,
                            shape_render_area_rect.width,
                            shape_render_area_rect.height,
                        )
                        # Calculate cell size to fit the shape
                        min_r, min_c, max_r, max_c = shp.bbox()
                        shape_h = max(1, max_r - min_r + 1)
                        shape_w_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)
                        scale_h = shape_render_surf.get_height() / shape_h
                        scale_w = shape_render_surf.get_width() / shape_w_eff
                        cell_size = max(1, min(scale_h, scale_w))
                        # Render the shape using game_area_renderer's helper
                        self.game_area_renderer._render_single_shape(
                            shape_render_surf, shp, int(cell_size)
                        )
                except ValueError as sub_err:
                    print(f"Error subsurface shape preview {i}: {sub_err}")
                    pygame.draw.rect(surf, VisConfig.RED, clipped_preview_rect, 1)
                except Exception as e:
                    print(f"Error rendering demo shape preview {i}: {e}")
                    pygame.draw.rect(surf, VisConfig.RED, clipped_preview_rect, 1)

            current_preview_y += preview_h + preview_padding

    def _render_too_small_message(self, text: str, area_rect: pygame.Rect):
        """Helper to render 'Too Small' messages."""
        try:
            # Use a font known to exist in game_area_renderer or init one here
            font = self.game_area_renderer.fonts.get("ui") or pygame.font.SysFont(
                None, 24
            )
            err_surf = font.render(text, True, VisConfig.GRAY)
            # Center the message within the specified area_rect relative to the screen
            target_rect = err_surf.get_rect(center=area_rect.center)
            self.screen.blit(err_surf, target_rect)
        except Exception as e:
            print(f"Error rendering 'too small' message: {e}")
