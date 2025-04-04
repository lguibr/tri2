# File: ui/panels/game_area.py
# --- Game Area Rendering Logic ---
import pygame
import math
import traceback
from typing import List, Tuple
from config import VisConfig, EnvConfig
from environment.game_state import GameState
from environment.shape import Shape
from environment.triangle import Triangle


class GameAreaRenderer:
    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()

    def _init_fonts(self):
        fonts = {}
        try:
            fonts["env_score"] = pygame.font.SysFont(None, 18)
            fonts["env_overlay"] = pygame.font.SysFont(None, 36)
            fonts["ui"] = pygame.font.SysFont(None, 24)  # For 'too small' message
        except Exception as e:
            print(f"Warning: SysFont error: {e}. Using default.")
            fonts["env_score"] = pygame.font.Font(None, 18)
            fonts["env_overlay"] = pygame.font.Font(None, 36)
            fonts["ui"] = pygame.font.Font(None, 24)
        return fonts

    def render(self, envs: List[GameState], num_envs: int, env_config: EnvConfig):
        """Renders the grid of game environments."""
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(300, self.vis_config.LEFT_PANEL_WIDTH))
        ga_rect = pygame.Rect(lp_width, 0, current_width - lp_width, current_height)

        if num_envs <= 0 or ga_rect.width <= 0 or ga_rect.height <= 0:
            return

        render_limit = self.vis_config.NUM_ENVS_TO_RENDER
        num_to_render = num_envs if render_limit <= 0 else min(num_envs, render_limit)
        if num_to_render <= 0:
            return

        cols_env, rows_env, cell_w, cell_h = self._calculate_grid_layout(
            ga_rect, num_to_render
        )

        if cell_w > 10 and cell_h > 10:
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

        if num_to_render < num_envs:
            self._render_render_limit_text(ga_rect, num_to_render, num_envs)

    def _calculate_grid_layout(
        self, ga_rect: pygame.Rect, num_to_render: int
    ) -> Tuple[int, int, int, int]:
        """Calculates the layout parameters for the environment grid."""
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
        """Iterates through the grid positions and renders each environment."""
        env_idx = 0
        for r in range(rows):
            for c in range(cols):
                if env_idx >= num_to_render:
                    break
                env_x = ga_rect.x + self.vis_config.ENV_SPACING * (c + 1) + c * cell_w
                env_y = ga_rect.y + self.vis_config.ENV_SPACING * (r + 1) + r * cell_h
                env_rect = pygame.Rect(env_x, env_y, cell_w, cell_h)
                try:
                    sub_surf = self.screen.subsurface(env_rect)
                    tri_cell_w, tri_cell_h = self._calculate_triangle_size(
                        cell_w, cell_h, env_config
                    )
                    self._render_single_env(
                        sub_surf, envs[env_idx], int(tri_cell_w), int(tri_cell_h)
                    )
                    self._render_shape_previews(sub_surf, envs[env_idx], cell_w, cell_h)
                except ValueError as subsurface_error:
                    print(
                        f"Warning: Subsurface error env {env_idx}: {subsurface_error}"
                    )
                    pygame.draw.rect(self.screen, (0, 0, 50), env_rect, 1)
                except Exception as e_render_env:
                    print(f"Error rendering env {env_idx}: {e_render_env}")
                    pygame.draw.rect(self.screen, (50, 0, 50), env_rect, 1)
                env_idx += 1

    def _calculate_triangle_size(self, cell_w, cell_h, env_config):
        """Calculates the size of individual triangles within an env cell."""
        denominator_w = env_config.COLS * 0.75 + 0.25
        denominator_h = env_config.ROWS
        tri_cell_w = cell_w / denominator_w if denominator_w > 0 else 1
        tri_cell_h = cell_h / denominator_h if denominator_h > 0 else 1
        return tri_cell_w, tri_cell_h

    def _render_single_env(
        self, surf: pygame.Surface, env: GameState, cell_w: int, cell_h: int
    ):
        """Renders the grid, scores, and overlays for a single environment."""
        try:
            bg_color = (
                VisConfig.YELLOW
                if env.is_blinking()
                else (
                    (30, 30, 100)
                    if env.is_frozen() and not env.is_over()
                    else (20, 20, 20)
                )
            )
            surf.fill(bg_color)

            # Render Grid Triangles
            if (
                hasattr(env, "grid")
                and hasattr(env.grid, "triangles")
                and cell_w > 0
                and cell_h > 0
            ):
                for r in range(env.grid.rows):
                    for c in range(env.grid.cols):
                        t = env.grid.triangles[r][c]
                        if not hasattr(t, "get_points"):
                            continue
                        try:
                            pts = t.get_points(ox=0, oy=0, cw=cell_w, ch=cell_h)
                            color = VisConfig.GRAY
                            if t.is_death:
                                color = VisConfig.BLACK
                            elif t.is_occupied:
                                color = t.color if t.color else VisConfig.RED
                            pygame.draw.polygon(surf, color, pts)
                        except Exception as e_render:
                            print(f"Error rendering tri ({r},{c}): {e_render}")
            else:
                pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
                err_txt = self.fonts["env_overlay"].render(
                    "Invalid Grid" if cell_w > 0 else "Too Small", True, VisConfig.RED
                )
                surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))

            # Render Scores
            score_surf = self.fonts["env_score"].render(
                f"GS: {env.game_score} | R: {env.score:.1f}",
                True,
                VisConfig.WHITE,
                (0, 0, 0, 180),
            )
            surf.blit(score_surf, (4, 4))

            # Render Overlays
            if env.is_over():
                overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                overlay.fill((100, 0, 0, 180))
                surf.blit(overlay, (0, 0))
                over_text = self.fonts["env_overlay"].render(
                    "GAME OVER", True, VisConfig.WHITE
                )
                surf.blit(over_text, over_text.get_rect(center=surf.get_rect().center))
            elif env.is_frozen() and not env.is_blinking():
                freeze_text = self.fonts["env_overlay"].render(
                    "Frozen", True, VisConfig.WHITE
                )
                surf.blit(
                    freeze_text,
                    freeze_text.get_rect(
                        center=(surf.get_width() // 2, surf.get_height() - 15)
                    ),
                )

        except AttributeError as e:
            pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
            err_txt = self.fonts["env_overlay"].render(
                f"Attr Err: {e}", True, VisConfig.RED, VisConfig.BLACK
            )
            surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))
        except Exception as e:
            print(f"Unexpected Render Error in _render_single_env: {e}")
            traceback.print_exc()
            pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)

    def _render_shape_previews(
        self, surf: pygame.Surface, env: GameState, cell_w: int, cell_h: int
    ):
        """Renders small previews of available shapes in the top-right corner."""
        available_shapes = env.get_shapes()
        if not available_shapes:
            return

        preview_dim = max(10, min(cell_w // 6, cell_h // 6, 25))
        preview_spacing = 4
        total_preview_width = (
            len(available_shapes) * preview_dim
            + max(0, len(available_shapes) - 1) * preview_spacing
        )
        start_x = surf.get_width() - total_preview_width - preview_spacing
        start_y = preview_spacing

        for i, shape in enumerate(available_shapes):
            preview_x = start_x + i * (preview_dim + preview_spacing)
            if preview_x + preview_dim <= surf.get_width():
                temp_shape_surf = pygame.Surface(
                    (preview_dim, preview_dim), pygame.SRCALPHA
                )
                temp_shape_surf.fill((0, 0, 0, 0))
                preview_cell_size = max(2, preview_dim // 5)
                self._render_single_shape(temp_shape_surf, shape, preview_cell_size)
                surf.blit(temp_shape_surf, (preview_x, start_y))

    def _render_single_shape(self, surf: pygame.Surface, shape: Shape, cell_size: int):
        """Renders a single shape onto a given surface."""
        if not shape or not shape.triangles or cell_size <= 0:
            return
        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h_cells = max_r - min_r + 1
        shape_w_cells = max_c - min_c + 1
        if shape_w_cells <= 0 or shape_h_cells <= 0:
            return

        total_w_pixels = shape_w_cells * (cell_size * 0.75) + (cell_size * 0.25)
        total_h_pixels = shape_h_cells * cell_size
        if total_w_pixels <= 0 or total_h_pixels <= 0:
            return

        offset_x = (surf.get_width() - total_w_pixels) / 2 - min_c * (cell_size * 0.75)
        offset_y = (surf.get_height() - total_h_pixels) / 2 - min_r * cell_size

        for dr, dc, up in shape.triangles:
            tri = Triangle(row=dr, col=dc, is_up=up)  # Temp instance for points
            try:
                pts = tri.get_points(
                    ox=offset_x, oy=offset_y, cw=cell_size, ch=cell_size
                )
                pygame.draw.polygon(surf, shape.color, pts)
            except Exception as e:
                print(f"Warning: Error rendering shape preview tri ({dr},{dc}): {e}")

    def _render_too_small_message(self, ga_rect: pygame.Rect, cell_w: int, cell_h: int):
        """Displays a message when the environment cells are too small."""
        err_surf = self.fonts["ui"].render(
            f"Envs Too Small ({cell_w}x{cell_h})", True, VisConfig.GRAY
        )
        self.screen.blit(err_surf, err_surf.get_rect(center=ga_rect.center))

    def _render_render_limit_text(
        self, ga_rect: pygame.Rect, num_rendered: int, num_total: int
    ):
        """Displays text indicating not all environments are being rendered."""
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
