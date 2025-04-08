# File: ui/panels/game_area.py
import pygame
import math
import traceback
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

from config import (
    VisConfig,
    EnvConfig,
    TrainConfig,
    BLACK,
    BLUE,
    RED,
    GRAY,
    YELLOW,
    LIGHTG,
    WHITE,
    DARK_RED,
    LINE_CLEAR_FLASH_COLOR,
    GAME_OVER_FLASH_COLOR,
)
from environment.game_state import GameState
from environment.shape import Shape
from environment.triangle import Triangle

logger = logging.getLogger(__name__)


class GameAreaRenderer:
    """Renders the right panel: Shows live self-play worker environments."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()
        # Caches are less relevant now
        self.best_state_surface_cache: pygame.Surface | None = None
        self.last_best_state_size: Tuple[int, int] = (0, 0)
        self.last_best_state_score: Optional[int] = None
        self.last_best_state_step: Optional[int] = None
        self.placeholder_surface_cache: pygame.Surface | None = None
        self.last_placeholder_size: Tuple[int, int] = (0, 0)
        self.last_placeholder_message_key: str = ""

    def _init_fonts(self):
        """Initializes fonts used in the game area."""
        fonts = {}
        font_configs = {
            "env_score": 16,
            "env_overlay": 24,
            "env_info": 14,
            "ui": 24,
            "placeholder": 30,
            "placeholder_detail": 22,
            "best_state_title": 32,
            "best_state_score": 28,
            "best_state_step": 20,
        }
        for key, size in font_configs.items():
            try:
                fonts[key] = pygame.font.SysFont(None, size)
            except Exception:
                try:
                    fonts[key] = pygame.font.Font(None, size)
                except Exception as e:
                    logger.error(f"ERROR: Font '{key}' failed: {e}")
                    fonts[key] = None
        # Fallbacks
        if fonts.get("ui") is None:
            fonts["ui"] = pygame.font.Font(None, 24)
        if fonts.get("placeholder") is None:
            fonts["placeholder"] = pygame.font.Font(None, 30)
        if fonts.get("env_score") is None:
            fonts["env_score"] = pygame.font.Font(None, 16)
        if fonts.get("env_overlay") is None:
            fonts["env_overlay"] = pygame.font.Font(None, 24)
        if fonts.get("env_info") is None:
            fonts["env_info"] = pygame.font.Font(None, 14)
        return fonts

    def render(self, panel_width: int, panel_x_offset: int, **kwargs):
        """Renders the grid of live self-play environments."""
        current_height = self.screen.get_height()
        ga_rect = pygame.Rect(panel_x_offset, 0, panel_width, current_height)
        if ga_rect.width <= 0 or ga_rect.height <= 0:
            return

        pygame.draw.rect(self.screen, VisConfig.DARK_GRAY, ga_rect)

        worker_render_data: List[Optional[Dict[str, Any]]] = kwargs.get(
            "worker_render_data", []
        )
        num_envs_total: int = kwargs.get("num_envs", 0)
        env_config: Optional[EnvConfig] = kwargs.get("env_config")

        num_to_render = len(worker_render_data)

        if num_to_render <= 0 or not env_config:
            self._render_placeholder(ga_rect, "Waiting for Self-Play Workers...")
            return

        cols_env, rows_env, cell_w, cell_h_total = self._calculate_grid_layout(
            ga_rect, num_to_render
        )

        if cell_w > 10 and cell_h_total > 40:
            self._render_env_grid(
                worker_render_data,
                num_to_render,
                env_config,
                ga_rect,
                cols_env,
                rows_env,
                cell_w,
                cell_h_total,
            )
        else:
            self._render_too_small_message(ga_rect, cell_w, cell_h_total)

        if num_to_render < num_envs_total:
            self._render_render_limit_text(ga_rect, num_to_render, num_envs_total)

    def _render_placeholder(self, ga_rect: pygame.Rect, message: str):
        """Renders a simple placeholder message."""
        pygame.draw.rect(self.screen, (60, 60, 70), ga_rect, 1)
        font = self.fonts.get("placeholder")
        if font:
            text_surf = font.render(message, True, LIGHTG)
            self.screen.blit(text_surf, text_surf.get_rect(center=ga_rect.center))

    def _calculate_grid_layout(
        self, ga_rect: pygame.Rect, num_to_render: int
    ) -> Tuple[int, int, int, int]:
        """Calculates layout for multiple small environment grids."""
        if ga_rect.width <= 0 or ga_rect.height <= 0 or num_to_render <= 0:
            return 0, 0, 0, 0
        aspect = ga_rect.width / max(1, ga_rect.height)
        cols = max(1, int(math.sqrt(num_to_render * aspect)))
        rows = max(1, math.ceil(num_to_render / cols))
        while cols * rows < num_to_render:
            if (cols + 1) / rows < aspect:
                cols += 1
            else:
                rows += 1
        while cols * (rows - 1) >= num_to_render and rows > 1:
            rows -= 1
        while (cols - 1) * rows >= num_to_render and cols > 1:
            cols -= 1

        sp = self.vis_config.ENV_SPACING
        cw_total = max(1, (ga_rect.width - (cols + 1) * sp) // cols)
        ch_total = max(1, (ga_rect.height - (rows + 1) * sp) // rows)
        return cols, rows, cw_total, ch_total

    def _render_env_grid(
        self,
        worker_render_data,
        num_to_render,
        env_config,
        ga_rect,
        cols,
        rows,
        cell_w,
        cell_h_total,
    ):
        """Renders the grid of small environment previews."""
        env_idx = 0
        sp = self.vis_config.ENV_SPACING
        info_h = 12
        shapes_h = 22
        grid_cell_h = max(1, cell_h_total - shapes_h - info_h - 2)

        for r in range(rows):
            for c in range(cols):
                if env_idx >= num_to_render:
                    break
                env_x = ga_rect.x + sp * (c + 1) + c * cell_w
                env_y = ga_rect.y + sp * (r + 1) + r * cell_h_total
                env_rect_total = pygame.Rect(env_x, env_y, cell_w, cell_h_total)
                clip_rect_total = env_rect_total.clip(self.screen.get_rect())

                if clip_rect_total.width <= 0 or clip_rect_total.height <= 0:
                    env_idx += 1
                    continue

                render_data = (
                    worker_render_data[env_idx]
                    if env_idx < len(worker_render_data)
                    else None
                )

                grid_rect_local = pygame.Rect(0, 0, cell_w, grid_cell_h)
                shapes_rect_local = pygame.Rect(0, grid_cell_h + 1, cell_w, shapes_h)
                info_rect_local = pygame.Rect(
                    0, grid_cell_h + shapes_h + 2, cell_w, info_h
                )

                try:
                    cell_surf = self.screen.subsurface(clip_rect_total)
                    cell_surf.fill(VisConfig.DARK_GRAY)

                    if render_data and render_data.get("state"):
                        env_state: GameState = render_data["state"]
                        env_stats: Dict[str, Any] = render_data.get("stats", {})

                        if grid_rect_local.height > 0:
                            grid_surf = cell_surf.subsurface(grid_rect_local)
                            self._render_single_env_grid_part(
                                grid_surf, env_state, env_config
                            )

                        if shapes_rect_local.height > 0:
                            shapes_surf = cell_surf.subsurface(shapes_rect_local)
                            self._render_env_shapes(shapes_surf, env_state)

                        if info_rect_local.height > 0:
                            info_surf = cell_surf.subsurface(info_rect_local)
                            self._render_env_info(info_surf, env_idx, env_stats)

                    else:  # Placeholder
                        pygame.draw.rect(cell_surf, (20, 20, 20), cell_surf.get_rect())
                        pygame.draw.rect(
                            cell_surf, (60, 60, 60), cell_surf.get_rect(), 1
                        )
                        font = self.fonts.get("env_info")
                        if font:
                            status = (
                                render_data.get("stats", {}).get("status", "N/A")
                                if render_data
                                else "N/A"
                            )
                            text_surf = font.render(status, True, GRAY)
                            cell_surf.blit(
                                text_surf,
                                text_surf.get_rect(center=cell_surf.get_rect().center),
                            )

                    pygame.draw.rect(
                        cell_surf, VisConfig.LIGHTG, cell_surf.get_rect(), 1
                    )

                except ValueError as e:
                    logger.error(
                        f"Error creating subsurface for env cell {env_idx} ({clip_rect_total}): {e}"
                    )
                    pygame.draw.rect(self.screen, (50, 0, 50), clip_rect_total, 1)
                except Exception as e:
                    logger.error(f"Error rendering env cell {env_idx}: {e}")
                    pygame.draw.rect(self.screen, (50, 0, 50), clip_rect_total, 1)

                env_idx += 1
            if env_idx >= num_to_render:
                break

    def _render_single_env_grid_part(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        """Renders the grid portion of a single environment preview."""
        cw, ch = surf.get_width(), surf.get_height()
        bg = VisConfig.GRAY

        if env.is_line_clearing():
            bg = LINE_CLEAR_FLASH_COLOR
        elif env.is_game_over_flashing():
            bg = GAME_OVER_FLASH_COLOR
        elif env.is_blinking():
            bg = VisConfig.YELLOW
        elif env.is_over():
            bg = DARK_RED
        elif env.is_frozen():
            bg = (30, 30, 100)

        surf.fill(bg)

        try:
            pad = self.vis_config.ENV_GRID_PADDING
            dw, dh = max(1, cw - 2 * pad), max(1, ch - 2 * pad)
            gr, gcw = env_config.ROWS, env_config.COLS * 0.75 + 0.25
            scale = min(dw / gcw, dh / gr) if gr > 0 and gcw > 0 else 0
            if scale > 0:
                fpw, fph = gcw * scale, gr * scale
                tcw, tch = max(1, scale), max(1, scale)
                ox, oy = pad + (dw - fpw) / 2, pad + (dh - fph) / 2
                is_hl = env.is_highlighting_cleared()
                cleared = set(env.get_cleared_triangle_coords()) if is_hl else set()
                hl_color = self.vis_config.LINE_CLEAR_HIGHLIGHT_COLOR

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
                                        ox=ox, oy=oy, cw=int(tcw), ch=int(tch)
                                    )
                                    color = VisConfig.LIGHTG
                                    if is_hl and (r, c) in cleared:
                                        color = hl_color
                                    elif t.is_occupied:
                                        color = t.color if t.color else VisConfig.RED
                                    pygame.draw.polygon(surf, color, pts)
                                    pygame.draw.polygon(surf, VisConfig.GRAY, pts, 1)
                                except Exception:
                                    pass
                else:
                    pygame.draw.rect(surf, RED, surf.get_rect(), 2)
        except Exception:
            pygame.draw.rect(surf, RED, surf.get_rect(), 2)

        # Score Overlay (ensure it uses the correct font and value)
        try:
            score_font = self.fonts.get("env_score")
            if score_font:
                score_surf = score_font.render(
                    f"S: {env.game_score}", True, WHITE, (0, 0, 0, 180)
                )
                surf.blit(score_surf, (2, 2))
        except Exception as e:
            logger.error(f"Error rendering score overlay: {e}")

        # State Overlays
        if env.is_over():
            self._render_overlay_text(surf, "GAME OVER", RED)
        elif env.is_line_clearing() and env.last_line_clear_info:
            lines, _, _ = env.last_line_clear_info
            self._render_overlay_text(surf, f"{lines}L Clear!", BLUE)

    def _render_env_shapes(self, surf: pygame.Surface, env: GameState):
        """Renders the available shapes for a single environment below its grid."""
        shapes = env.get_shapes()
        sw, sh = surf.get_width(), surf.get_height()
        if not shapes or sw <= 0 or sh <= 0:
            return

        num_slots = env.env_config.NUM_SHAPE_SLOTS
        pad = 2
        total_pad_w = (num_slots + 1) * pad
        avail_w = sw - total_pad_w
        if avail_w <= 0:
            return

        w_per = avail_w / num_slots
        h_lim = sh - 2 * pad
        dim = max(5, int(min(w_per, h_lim)))

        start_x = pad + (sw - (num_slots * dim + (num_slots - 1) * pad)) / 2
        start_y = pad + (sh - dim) / 2
        curr_x = start_x

        for i in range(num_slots):
            shape = shapes[i] if i < len(shapes) else None
            rect = pygame.Rect(int(curr_x), int(start_y), dim, dim)
            if rect.right > sw - pad:
                break

            pygame.draw.rect(surf, (50, 50, 50), rect, border_radius=2)

            if shape:
                try:
                    shape_surf = pygame.Surface((dim, dim), pygame.SRCALPHA)
                    shape_surf.fill((0, 0, 0, 0))
                    # Pass the surface dimensions to _render_single_shape for correct scaling
                    self._render_single_shape(shape_surf, shape, dim, dim)
                    surf.blit(shape_surf, rect.topleft)
                except Exception as e:
                    logger.error(f"Error rendering shape in env preview: {e}")
                    pygame.draw.line(surf, RED, rect.topleft, rect.bottomright, 1)
            else:
                pygame.draw.line(surf, GRAY, rect.topleft, rect.bottomright, 1)
                pygame.draw.line(surf, GRAY, rect.topright, rect.bottomleft, 1)

            curr_x += dim + pad

    def _render_env_info(
        self, surf: pygame.Surface, worker_idx: int, stats: Dict[str, Any]
    ):
        """Renders worker ID and current game step."""
        font = self.fonts.get("env_info")
        if not font:
            return

        game_step = stats.get("game_steps", "?")
        # Use full words
        info_text = f"Worker: {worker_idx} Step: {game_step}"
        try:
            text_surf = font.render(info_text, True, LIGHTG)
            text_rect = text_surf.get_rect(centerx=surf.get_width() // 2, top=0)
            surf.blit(text_surf, text_rect)
        except Exception as e:
            logger.error(f"Error rendering env info: {e}")

    def _render_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders overlay text, scaling font if needed."""
        try:
            font = self.fonts.get("env_overlay")
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
            logger.error(f"Error rendering overlay '{text}': {e}")

    def _render_single_shape(
        self, surf: pygame.Surface, shape: Shape, target_w: int, target_h: int
    ):
        """Renders a single shape scaled to fit within target_w, target_h."""
        if not shape or not shape.triangles or target_w <= 0 or target_h <= 0:
            return

        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h_cells = max(1, max_r - min_r + 1)
        # Effective width calculation for triangles
        shape_w_cells_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)

        # Calculate scale to fit within target dimensions
        scale_h = target_h / shape_h_cells if shape_h_cells > 0 else target_h
        scale_w = target_w / shape_w_cells_eff if shape_w_cells_eff > 0 else target_w
        scale = max(1, min(scale_h, scale_w))  # Use the smaller scale to ensure fit

        # Calculate total pixel dimensions and offset for centering
        total_w = shape_w_cells_eff * scale
        total_h = shape_h_cells * scale
        ox = (target_w - total_w) / 2 - min_c * (scale * 0.75)
        oy = (target_h - total_h) / 2 - min_r * scale

        for dr, dc, is_up in shape.triangles:
            tri = Triangle(0, 0, is_up)
            try:
                # Calculate points relative to the shape's origin and apply offset
                rel_c = dc - min_c
                rel_r = dr - min_r
                tri_x = ox + rel_c * (scale * 0.75)
                tri_y = oy + rel_r * scale
                # Ensure points are integers for drawing
                pts = [
                    (int(p[0]), int(p[1]))
                    for p in tri.get_points(
                        ox=tri_x, oy=tri_y, cw=int(scale), ch=int(scale)
                    )
                ]
                pygame.draw.polygon(surf, shape.color, pts)
                pygame.draw.polygon(surf, BLACK, pts, 1)  # Border
            except Exception as e:
                logger.debug(f"Error drawing triangle in shape preview: {e}")

    def _render_too_small_message(self, ga_rect: pygame.Rect, cell_w: int, cell_h: int):
        """Renders a message if the env cells are too small."""
        font = self.fonts.get("ui")
        if font:
            surf = font.render(f"Envs Too Small ({cell_w}x{cell_h})", True, GRAY)
            self.screen.blit(surf, surf.get_rect(center=ga_rect.center))

    def _render_render_limit_text(
        self, ga_rect: pygame.Rect, num_rendered: int, num_total: int
    ):
        """Renders text indicating not all envs are shown."""
        font = self.fonts.get("ui")
        if font:
            surf = font.render(
                f"Rendering {num_rendered}/{num_total} Workers", True, YELLOW, BLACK
            )
            self.screen.blit(
                surf, surf.get_rect(bottomright=(ga_rect.right - 5, ga_rect.bottom - 5))
            )
