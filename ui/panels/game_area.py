# File: ui/panels/game_area.py
import pygame
import math
import traceback
import numpy as np  # Import numpy
from typing import List, Tuple, Optional, Dict, Any  # Added Optional, Dict, Any

from config import (
    VisConfig,
    EnvConfig,
    BLACK,
    BLUE,
    RED,
    GRAY,
    YELLOW,
    LIGHTG,
    WHITE,  # Added WHITE
    MCTS_MINI_GRID_BG_COLOR,
    MCTS_MINI_GRID_LINE_COLOR,
    MCTS_MINI_GRID_OCCUPIED_COLOR,
)
from environment.game_state import GameState
from environment.shape import Shape
from environment.triangle import Triangle


class GameAreaRenderer:
    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()
        # Cache a placeholder surface for inactive state
        self.placeholder_surface: pygame.Surface | None = None
        self.last_placeholder_size: Tuple[int, int] = (0, 0)
        self.last_placeholder_message: str = ""  # Cache the message too
        # Cache for best state rendering
        self.best_state_surface: pygame.Surface | None = None
        self.last_best_state_size: Tuple[int, int] = (0, 0)
        self.last_best_state_score: Optional[int] = None

    def _init_fonts(self):
        fonts = {}
        try:
            fonts["env_score"] = pygame.font.SysFont(None, 18)
            fonts["env_overlay"] = pygame.font.SysFont(None, 36)
            fonts["ui"] = pygame.font.SysFont(None, 24)
            fonts["placeholder"] = pygame.font.SysFont(None, 30)
            fonts["best_state_title"] = pygame.font.SysFont(
                None, 32
            )  # Font for best state title
            fonts["best_state_score"] = pygame.font.SysFont(
                None, 28
            )  # Font for best state score
        except Exception as e:
            print(f"Warning: SysFont error: {e}. Using default.")
            fonts["env_score"] = pygame.font.Font(None, 18)
            fonts["env_overlay"] = pygame.font.Font(None, 36)
            fonts["ui"] = pygame.font.Font(None, 24)
            fonts["placeholder"] = pygame.font.Font(None, 30)
            fonts["best_state_title"] = pygame.font.Font(None, 32)
            fonts["best_state_score"] = pygame.font.Font(None, 28)
        return fonts

    def render(
        self,
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        panel_width: int,
        panel_x_offset: int,
        is_running: bool = False,
        best_game_state_data: Optional[Dict[str, Any]] = None,  # Added best state data
    ):
        current_height = self.screen.get_height()
        ga_rect = pygame.Rect(panel_x_offset, 0, panel_width, current_height)

        if ga_rect.width <= 0 or ga_rect.height <= 0:
            return

        # If workers are running, display best state or specific placeholder
        if is_running:
            if best_game_state_data:
                # If we have data for the best state, render it
                self._render_best_game_state(ga_rect, best_game_state_data, env_config)
            else:
                # If no best state data yet, show "Running..." placeholder
                self._render_running_placeholder(
                    ga_rect, "Running Self-Play / Training..."
                )
            return  # Don't render individual envs when running

        # --- Original rendering logic if workers are NOT running ---
        # (This part remains unchanged)
        if num_envs <= 0:
            pygame.draw.rect(self.screen, (10, 10, 10), ga_rect)
            pygame.draw.rect(self.screen, (50, 50, 50), ga_rect, 1)
            return

        render_limit = self.vis_config.NUM_ENVS_TO_RENDER
        num_to_render = min(num_envs, render_limit) if render_limit > 0 else num_envs

        if num_to_render <= 0:
            pygame.draw.rect(self.screen, (10, 10, 10), ga_rect)
            pygame.draw.rect(self.screen, (50, 50, 50), ga_rect, 1)
            return

        cols_env, rows_env, cell_w, cell_h = self._calculate_grid_layout(
            ga_rect, num_to_render
        )

        min_cell_dim = 30
        if cell_w > min_cell_dim and cell_h > min_cell_dim:
            self._render_env_grid(
                envs,
                num_to_render,
                env_config,
                ga_rect,
                cols_env,
                rows_env,
                cell_w,
                cell_h,
            )
        else:
            self._render_too_small_message(ga_rect, cell_w, cell_h)

        if num_to_render < num_envs and len(envs) > 0:
            self._render_render_limit_text(ga_rect, num_to_render, num_envs)

    def _render_running_placeholder(self, ga_rect: pygame.Rect, message: str):
        """Renders a placeholder message, caching the surface."""
        current_size = ga_rect.size
        # Re-render placeholder only if size or message changes
        if (
            self.placeholder_surface is None
            or self.last_placeholder_size != current_size
            or self.last_placeholder_message != message
        ):
            self.placeholder_surface = pygame.Surface(current_size)
            self.placeholder_surface.fill((20, 20, 25))
            pygame.draw.rect(
                self.placeholder_surface,
                (60, 60, 70),
                self.placeholder_surface.get_rect(),
                1,
            )
            placeholder_font = self.fonts.get("placeholder")
            if placeholder_font:
                text_surf = placeholder_font.render(message, True, LIGHTG)
                text_rect = text_surf.get_rect(
                    center=self.placeholder_surface.get_rect().center
                )
                self.placeholder_surface.blit(text_surf, text_rect)
            self.last_placeholder_size = current_size
            self.last_placeholder_message = message  # Cache the message

        if self.placeholder_surface:
            self.screen.blit(self.placeholder_surface, ga_rect.topleft)

    def _render_best_game_state(
        self, ga_rect: pygame.Rect, state_data: Dict[str, Any], env_config: EnvConfig
    ):
        """Renders the best game state grid in the game area."""
        current_size = ga_rect.size
        current_score = state_data.get("score")

        # Check if cache needs update (size change or score change)
        if (
            self.best_state_surface is None
            or self.last_best_state_size != current_size
            or self.last_best_state_score != current_score
        ):

            self.best_state_surface = pygame.Surface(current_size)
            self.best_state_surface.fill((25, 25, 30))  # Slightly different background

            # Render the grid using the stored data
            grid_rect = pygame.Rect(
                0, 50, current_size[0], current_size[1] - 60
            )  # Leave space for title
            try:
                grid_subsurface = self.best_state_surface.subsurface(grid_rect)
                self._render_grid_from_data(grid_subsurface, state_data, env_config)
            except ValueError as e:
                print(f"Error creating subsurface for best state grid: {e}")
                pygame.draw.rect(self.best_state_surface, RED, grid_rect, 1)

            # Render Title and Score
            title_font = self.fonts.get("best_state_title")
            score_font = self.fonts.get("best_state_score")
            if title_font and score_font:
                title_surf = title_font.render("Best Game State", True, YELLOW)
                score_surf = score_font.render(f"Score: {current_score}", True, WHITE)

                title_rect = title_surf.get_rect(centerx=current_size[0] // 2, top=5)
                score_rect = score_surf.get_rect(
                    centerx=current_size[0] // 2, top=title_rect.bottom + 2
                )

                self.best_state_surface.blit(title_surf, title_rect)
                self.best_state_surface.blit(score_surf, score_rect)

            pygame.draw.rect(
                self.best_state_surface, YELLOW, self.best_state_surface.get_rect(), 1
            )  # Border

            self.last_best_state_size = current_size
            self.last_best_state_score = current_score

        if self.best_state_surface:
            self.screen.blit(self.best_state_surface, ga_rect.topleft)
        else:  # Fallback if surface creation failed
            self._render_running_placeholder(ga_rect, "Error rendering best state")

    def _render_grid_from_data(
        self, surf: pygame.Surface, state_data: Dict[str, Any], env_config: EnvConfig
    ):
        """Renders a grid based on stored occupancy/color data."""
        try:
            occupancy = state_data.get("occupancy")
            colors = state_data.get("colors")
            death = state_data.get("death")
            is_up = state_data.get("is_up")
            rows = state_data.get("rows", env_config.ROWS)
            cols = state_data.get("cols", env_config.COLS)

            if occupancy is None or colors is None or death is None or is_up is None:
                print("Error: Missing data for rendering best grid state.")
                pygame.draw.rect(surf, RED, surf.get_rect(), 2)
                return

            padding = (
                self.vis_config.ENV_GRID_PADDING * 2
            )  # More padding for the large view
            drawable_w, drawable_h = max(1, surf.get_width() - 2 * padding), max(
                1, surf.get_height() - 2 * padding
            )
            grid_rows, grid_cols_eff_width = rows, cols * 0.75 + 0.25

            if grid_rows <= 0 or grid_cols_eff_width <= 0:
                return
            scale_w, scale_h = drawable_w / grid_cols_eff_width, drawable_h / grid_rows
            final_scale = min(scale_w, scale_h)
            if final_scale <= 0:
                return

            final_grid_pixel_w, final_grid_pixel_h = (
                grid_cols_eff_width * final_scale,
                grid_rows * final_scale,
            )
            tri_cell_w, tri_cell_h = max(1, final_scale), max(1, final_scale)
            grid_ox, grid_oy = (
                padding + (drawable_w - final_grid_pixel_w) / 2,
                padding + (drawable_h - final_grid_pixel_h) / 2,
            )

            for r in range(rows):
                for c in range(cols):
                    if death[r, c]:
                        continue  # Skip death cells

                    temp_tri = Triangle(r, c, is_up=is_up[r, c])
                    try:
                        pts = temp_tri.get_points(
                            ox=grid_ox,
                            oy=grid_oy,
                            cw=int(tri_cell_w),
                            ch=int(tri_cell_h),
                        )
                        color = VisConfig.LIGHTG  # Default empty color
                        if occupancy[r, c]:
                            cell_color = colors[r, c]
                            # Handle potential None or non-tuple colors safely
                            if isinstance(cell_color, tuple) and len(cell_color) == 3:
                                color = cell_color
                            else:
                                color = (
                                    VisConfig.RED
                                )  # Fallback color if stored color is invalid
                        pygame.draw.polygon(surf, color, pts)
                        pygame.draw.polygon(surf, VisConfig.GRAY, pts, 1)  # Grid lines
                    except Exception as e:
                        # print(f"Minor error drawing triangle {r},{c}: {e}")
                        pass  # Ignore minor drawing errors for single triangles

        except Exception as e:
            print(f"Error rendering grid from data: {e}")
            traceback.print_exc()
            pygame.draw.rect(surf, RED, surf.get_rect(), 2)

    def _calculate_grid_layout(
        self, ga_rect: pygame.Rect, num_to_render: int
    ) -> Tuple[int, int, int, int]:
        if ga_rect.width <= 0 or ga_rect.height <= 0:
            return 0, 0, 0, 0
        aspect_ratio = ga_rect.width / max(1, ga_rect.height)
        cols_env = max(1, int(math.sqrt(num_to_render * aspect_ratio)))
        rows_env = max(1, math.ceil(num_to_render / cols_env))
        total_spacing_w = (cols_env + 1) * self.vis_config.ENV_SPACING
        total_spacing_h = (rows_env + 1) * self.vis_config.ENV_SPACING
        cell_w = max(1, (ga_rect.width - total_spacing_w) // cols_env)
        cell_h = max(1, (ga_rect.height - total_spacing_h) // rows_env)
        return cols_env, rows_env, cell_w, cell_h

    def _render_env_grid(
        self, envs, num_to_render, env_config, ga_rect, cols, rows, cell_w, cell_h
    ):
        env_idx = 0
        for r in range(rows):
            for c in range(cols):
                if env_idx >= num_to_render:
                    break
                env_x = ga_rect.x + self.vis_config.ENV_SPACING * (c + 1) + c * cell_w
                env_y = ga_rect.y + self.vis_config.ENV_SPACING * (r + 1) + r * cell_h
                env_rect = pygame.Rect(env_x, env_y, cell_w, cell_h)
                clipped_env_rect = env_rect.clip(self.screen.get_rect())

                if clipped_env_rect.width <= 0 or clipped_env_rect.height <= 0:
                    env_idx += 1
                    continue

                if env_idx < len(envs):
                    try:
                        sub_surf = self.screen.subsurface(clipped_env_rect)
                        self._render_single_env(sub_surf, envs[env_idx], env_config)
                    except ValueError as subsurface_error:
                        print(
                            f"Warning: Subsurface error env {env_idx} ({clipped_env_rect}): {subsurface_error}"
                        )
                        pygame.draw.rect(self.screen, (0, 0, 50), clipped_env_rect, 1)
                    except Exception as e_render_env:
                        print(f"Error rendering env {env_idx}: {e_render_env}")
                        traceback.print_exc()
                        pygame.draw.rect(self.screen, (50, 0, 50), clipped_env_rect, 1)
                else:
                    pygame.draw.rect(self.screen, (20, 20, 20), clipped_env_rect)
                    pygame.draw.rect(self.screen, (60, 60, 60), clipped_env_rect, 1)
                env_idx += 1
            if env_idx >= num_to_render:
                break

    def _render_single_env(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        cell_w, cell_h = surf.get_width(), surf.get_height()
        if cell_w <= 0 or cell_h <= 0:
            return

        bg_color = VisConfig.GRAY
        if env.is_line_clearing():
            bg_color = VisConfig.LINE_CLEAR_FLASH_COLOR
        elif env.is_game_over_flashing():
            bg_color = VisConfig.GAME_OVER_FLASH_COLOR
        elif env.is_blinking():
            bg_color = VisConfig.YELLOW
        elif env.is_over():
            bg_color = VisConfig.DARK_RED
        elif env.is_frozen():
            bg_color = (30, 30, 100)
        surf.fill(bg_color)

        shape_area_height_ratio = 0.20
        grid_area_height = math.floor(cell_h * (1.0 - shape_area_height_ratio))
        shape_area_height = cell_h - grid_area_height
        shape_area_y = grid_area_height

        grid_surf, shape_surf = None, None
        if grid_area_height > 0 and cell_w > 0:
            try:
                grid_surf = surf.subsurface(pygame.Rect(0, 0, cell_w, grid_area_height))
            except ValueError:
                pygame.draw.rect(
                    surf, VisConfig.RED, pygame.Rect(0, 0, cell_w, grid_area_height), 1
                )
        if shape_area_height > 0 and cell_w > 0:
            try:
                shape_rect = pygame.Rect(0, shape_area_y, cell_w, shape_area_height)
                shape_surf = surf.subsurface(shape_rect)
                shape_surf.fill((35, 35, 35))
            except ValueError:
                pygame.draw.rect(
                    surf,
                    VisConfig.RED,
                    pygame.Rect(0, shape_area_y, cell_w, shape_area_height),
                    1,
                )

        if grid_surf:
            self._render_single_env_grid(grid_surf, env, env_config)
        if shape_surf:
            self._render_shape_previews(shape_surf, env)

        try:
            score_text = f"GS: {env.game_score}"
            score_surf = self.fonts["env_score"].render(
                score_text, True, VisConfig.WHITE, (0, 0, 0, 180)
            )
            surf.blit(score_surf, (2, 2))
        except Exception as e:
            print(f"Error rendering score: {e}")

        if env.is_over():
            self._render_overlay_text(surf, "GAME OVER", VisConfig.RED)
        elif env.is_line_clearing() and env.last_line_clear_info:
            lines, tris, score = env.last_line_clear_info
            line_str = "Line" if lines == 1 else "Lines"
            clear_msg = f"{lines} {line_str} Cleared! ({tris} Tris)"
            self._render_overlay_text(surf, clear_msg, BLUE)

    def _render_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        try:
            overlay_font = self.fonts["env_overlay"]
            max_width = surf.get_width() * 0.9
            font_size = 36
            text_surf = overlay_font.render(text, True, VisConfig.WHITE)
            while text_surf.get_width() > max_width and font_size > 10:
                font_size -= 2
                overlay_font = pygame.font.SysFont(None, font_size)
                text_surf = overlay_font.render(text, True, VisConfig.WHITE)
            bg_color_rgba = (color[0] // 2, color[1] // 2, color[2] // 2, 220)
            text_surf_with_bg = overlay_font.render(
                text, True, VisConfig.WHITE, bg_color_rgba
            )
            text_rect = text_surf_with_bg.get_rect(center=surf.get_rect().center)
            surf.blit(text_surf_with_bg, text_rect)
        except Exception as e:
            print(f"Error rendering overlay text '{text}': {e}")

    def _render_single_env_grid(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        """Renders the hexagonal grid for a single environment."""
        try:
            padding = self.vis_config.ENV_GRID_PADDING
            drawable_w, drawable_h = max(1, surf.get_width() - 2 * padding), max(
                1, surf.get_height() - 2 * padding
            )
            grid_rows, grid_cols_eff_width = (
                env_config.ROWS,
                env_config.COLS * 0.75 + 0.25,
            )
            if grid_rows <= 0 or grid_cols_eff_width <= 0:
                return
            scale_w, scale_h = drawable_w / grid_cols_eff_width, drawable_h / grid_rows
            final_scale = min(scale_w, scale_h)
            if final_scale <= 0:
                return
            final_grid_pixel_w, final_grid_pixel_h = (
                grid_cols_eff_width * final_scale,
                grid_rows * final_scale,
            )
            tri_cell_w, tri_cell_h = max(1, final_scale), max(1, final_scale)
            grid_ox, grid_oy = (
                padding + (drawable_w - final_grid_pixel_w) / 2,
                padding + (drawable_h - final_grid_pixel_h) / 2,
            )
            is_highlighting = env.is_highlighting_cleared()
            cleared_coords = (
                set(env.get_cleared_triangle_coords()) if is_highlighting else set()
            )
            highlight_color = self.vis_config.LINE_CLEAR_HIGHLIGHT_COLOR

            if hasattr(env, "grid") and hasattr(env.grid, "triangles"):
                for r in range(env.grid.rows):
                    for c in range(env.grid.cols):
                        if not (
                            0 <= r < len(env.grid.triangles)
                            and 0 <= c < len(env.grid.triangles[r])
                        ):
                            continue
                        t = env.grid.triangles[r][c]
                        if not t.is_death and hasattr(t, "get_points"):
                            try:
                                pts = t.get_points(
                                    ox=grid_ox,
                                    oy=grid_oy,
                                    cw=int(tri_cell_w),
                                    ch=int(tri_cell_h),
                                )
                                color = VisConfig.LIGHTG
                                if is_highlighting and (r, c) in cleared_coords:
                                    color = highlight_color
                                elif t.is_occupied:
                                    color = t.color if t.color else VisConfig.RED
                                pygame.draw.polygon(surf, color, pts)
                                pygame.draw.polygon(surf, VisConfig.GRAY, pts, 1)
                            except Exception:
                                pass
            else:
                pygame.draw.rect(surf, VisConfig.RED, surf.get_rect(), 2)
                err_txt = self.fonts["ui"].render(
                    "Invalid Grid Data", True, VisConfig.RED
                )
                surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))
        except Exception as e:
            pygame.draw.rect(surf, VisConfig.RED, surf.get_rect(), 2)

    def render_mini_grid(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        """Renders a simplified grid onto a smaller surface (for MCTS nodes)."""
        try:
            padding = 1
            drawable_w, drawable_h = max(1, surf.get_width() - 2 * padding), max(
                1, surf.get_height() - 2 * padding
            )
            grid_rows, grid_cols_eff_width = (
                env_config.ROWS,
                env_config.COLS * 0.75 + 0.25,
            )
            if grid_rows <= 0 or grid_cols_eff_width <= 0:
                return
            scale_w, scale_h = drawable_w / grid_cols_eff_width, drawable_h / grid_rows
            final_scale = min(scale_w, scale_h)
            if final_scale <= 0:
                return
            final_grid_pixel_w, final_grid_pixel_h = (
                grid_cols_eff_width * final_scale,
                grid_rows * final_scale,
            )
            tri_cell_w, tri_cell_h = max(1, final_scale), max(1, final_scale)
            grid_ox, grid_oy = (
                padding + (drawable_w - final_grid_pixel_w) / 2,
                padding + (drawable_h - final_grid_pixel_h) / 2,
            )

            surf.fill(MCTS_MINI_GRID_BG_COLOR)

            grid = env.grid
            for r in range(grid.rows):
                for c in range(grid.cols):
                    if not (
                        0 <= r < len(grid.triangles) and 0 <= c < len(grid.triangles[r])
                    ):
                        continue
                    t = grid.triangles[r][c]
                    if not t.is_death and hasattr(t, "get_points"):
                        try:
                            pts = t.get_points(
                                ox=grid_ox,
                                oy=grid_oy,
                                cw=int(tri_cell_w),
                                ch=int(tri_cell_h),
                            )
                            color = MCTS_MINI_GRID_BG_COLOR
                            if t.is_occupied:
                                color = MCTS_MINI_GRID_OCCUPIED_COLOR
                            pygame.draw.polygon(surf, color, pts)
                            pygame.draw.polygon(surf, MCTS_MINI_GRID_LINE_COLOR, pts, 1)
                        except Exception:
                            pass
        except Exception as e:
            print(f"Error rendering mini-grid: {e}")
            pygame.draw.line(surf, RED, (0, 0), surf.get_size(), 1)

    def _render_shape_previews(self, surf: pygame.Surface, env: GameState):
        available_shapes = env.get_shapes()
        if not available_shapes:
            return
        surf_w, surf_h = surf.get_width(), surf.get_height()
        if surf_w <= 0 or surf_h <= 0:
            return
        num_shapes = len(available_shapes)
        padding = 4
        total_padding = (num_shapes + 1) * padding
        available_width = surf_w - total_padding
        if available_width <= 0:
            return
        width_per_shape = available_width / num_shapes
        height_limit = surf_h - 2 * padding
        preview_dim = max(5, min(width_per_shape, height_limit))
        start_x = (
            padding
            + (surf_w - (num_shapes * preview_dim + (num_shapes - 1) * padding)) / 2
        )
        start_y = padding + (surf_h - preview_dim) / 2
        current_x = start_x

        for shape in available_shapes:
            preview_rect = pygame.Rect(current_x, start_y, preview_dim, preview_dim)
            if preview_rect.right > surf_w - padding:
                break
            if shape is None:
                pygame.draw.rect(surf, (50, 50, 50), preview_rect, 1, border_radius=2)
                current_x += preview_dim + padding
                continue
            try:
                temp_shape_surf = pygame.Surface(
                    (preview_dim, preview_dim), pygame.SRCALPHA
                )
                temp_shape_surf.fill((0, 0, 0, 0))
                min_r, min_c, max_r, max_c = shape.bbox()
                shape_h, shape_w_eff = max(1, max_r - min_r + 1), max(
                    1, (max_c - min_c + 1) * 0.75 + 0.25
                )
                scale_h, scale_w = preview_dim / shape_h, preview_dim / shape_w_eff
                cell_size = max(1, min(scale_h, scale_w))
                self._render_single_shape(temp_shape_surf, shape, int(cell_size))
                surf.blit(temp_shape_surf, preview_rect.topleft)
                current_x += preview_dim + padding
            except Exception as e:
                pygame.draw.rect(surf, VisConfig.RED, preview_rect, 1)
                current_x += preview_dim + padding

    def _render_single_shape(self, surf: pygame.Surface, shape: Shape, cell_size: int):
        if not shape or not shape.triangles or cell_size <= 0:
            return
        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h, shape_w_eff = max(1, max_r - min_r + 1), max(
            1, (max_c - min_c + 1) * 0.75 + 0.25
        )
        if shape_w_eff <= 0 or shape_h <= 0:
            return
        total_w, total_h = shape_w_eff * cell_size, shape_h * cell_size
        offset_x = (surf.get_width() - total_w) / 2 - min_c * (cell_size * 0.75)
        offset_y = (surf.get_height() - total_h) / 2 - min_r * cell_size
        for dr, dc, up in shape.triangles:
            tri = Triangle(row=dr, col=dc, is_up=up)
            try:
                pts = tri.get_points(
                    ox=offset_x, oy=offset_y, cw=cell_size, ch=cell_size
                )
                pygame.draw.polygon(surf, shape.color, pts)
            except Exception:
                pass

    def _render_too_small_message(self, ga_rect: pygame.Rect, cell_w: int, cell_h: int):
        try:
            err_surf = self.fonts["ui"].render(
                f"Envs Too Small ({cell_w}x{cell_h})", True, VisConfig.GRAY
            )
            self.screen.blit(err_surf, err_surf.get_rect(center=ga_rect.center))
        except Exception as e:
            print(f"Error rendering 'too small' message: {e}")

    def _render_render_limit_text(
        self, ga_rect: pygame.Rect, num_rendered: int, num_total: int
    ):
        try:
            info_surf = self.fonts["ui"].render(
                f"Rendering {num_rendered}/{num_total} Envs",
                True,
                VisConfig.YELLOW,
                VisConfig.BLACK,
            )
            self.screen.blit(
                info_surf,
                info_surf.get_rect(bottomright=(ga_rect.right - 5, ga_rect.bottom - 5)),
            )
        except Exception as e:
            print(f"Error rendering limit text: {e}")
