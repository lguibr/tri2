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
)
from environment.game_state import GameState
from environment.shape import Shape
from environment.triangle import Triangle

logger = logging.getLogger(__name__)


class GameAreaRenderer:
    """Renders the right panel: multi-env view (idle) or best state/placeholder (running)."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()
        # --- Surface Caching ---
        self.best_state_surface_cache: pygame.Surface | None = None
        self.last_best_state_size: Tuple[int, int] = (0, 0)
        self.last_best_state_score: Optional[int] = None
        self.last_best_state_step: Optional[int] = None
        self.placeholder_surface_cache: pygame.Surface | None = None
        self.last_placeholder_size: Tuple[int, int] = (0, 0)
        self.last_placeholder_message_key: str = ""  # Combined message and details

    def _init_fonts(self):
        """Initializes fonts used in the game area."""
        fonts = {}
        font_configs = {
            "env_score": 18,
            "env_overlay": 36,
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
        # Ensure essential fonts have fallbacks
        if fonts.get("ui") is None:
            fonts["ui"] = pygame.font.Font(None, 24)
        if fonts.get("placeholder") is None:
            fonts["placeholder"] = pygame.font.Font(None, 30)
        return fonts

    def render(self, panel_width: int, panel_x_offset: int, **kwargs):  # Use kwargs
        """Renders the game area panel based on running state."""
        current_height = self.screen.get_height()
        ga_rect = pygame.Rect(panel_x_offset, 0, panel_width, current_height)
        if ga_rect.width <= 0 or ga_rect.height <= 0:
            return

        is_running = kwargs.get("is_running", False)
        if is_running:
            self._render_running_state(
                ga_rect,
                kwargs.get("best_game_state_data"),
                kwargs.get("stats_summary"),
                kwargs.get("env_config"),
            )
        else:
            # Clear caches when idle to ensure they regenerate if needed later
            self.best_state_surface_cache = None
            self.placeholder_surface_cache = None
            self._render_idle_state(
                ga_rect,
                kwargs.get("envs", []),
                kwargs.get("num_envs", 0),
                kwargs.get("env_config"),
            )

    def _render_running_state(
        self,
        ga_rect: pygame.Rect,
        best_state_data: Optional[Dict[str, Any]],
        stats_summary: Optional[Dict[str, Any]],
        env_config: Optional[EnvConfig],
    ):
        """Renders the panel when the process is running."""
        if best_state_data and env_config:
            self._render_best_game_state(ga_rect, best_state_data, env_config)
        else:
            message = "Running AlphaZero..."
            details = []
            if stats_summary:
                game_num = stats_summary.get("current_self_play_game_number", 0)
                train_steps = stats_summary.get("training_steps_performed", 0)
                buffer_size = stats_summary.get("buffer_size", 0)
                min_buffer = TrainConfig().MIN_BUFFER_SIZE_TO_TRAIN
                details.append(
                    f"Playing Game: {game_num}"
                    if game_num > 0
                    else "Waiting for first game..."
                )
                details.append(f"Training Steps: {train_steps:,}".replace(",", "_"))
                details.append(
                    f"Buffer: {buffer_size:,}/{min_buffer:,}".replace(",", "_")
                )
            else:
                details.append("Waiting for stats...")
            self._render_running_placeholder(ga_rect, message, details)

    def _render_idle_state(
        self,
        ga_rect: pygame.Rect,
        envs: List[GameState],
        num_envs: int,
        env_config: Optional[EnvConfig],
    ):
        """Renders the panel when the process is idle."""
        render_limit = self.vis_config.NUM_ENVS_TO_RENDER
        if render_limit <= 0 or not env_config:
            self._render_placeholder(ga_rect, "Idle - Multi-Env View Disabled")
            return

        effective_num_envs = len(envs) if envs else num_envs
        if effective_num_envs <= 0:
            self._render_placeholder(ga_rect, "No Environments")
            return
        num_to_render = min(effective_num_envs, render_limit)
        if num_to_render <= 0:
            self._render_placeholder(ga_rect, "No Environments to Render")
            return

        cols_env, rows_env, cell_w, cell_h = self._calculate_grid_layout(
            ga_rect, num_to_render
        )
        if cell_w > 30 and cell_h > 30:
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
        if num_to_render < effective_num_envs:
            self._render_render_limit_text(ga_rect, num_to_render, effective_num_envs)

    def _render_placeholder(self, ga_rect: pygame.Rect, message: str):
        """Renders a simple placeholder message."""
        pygame.draw.rect(self.screen, (20, 20, 25), ga_rect)
        pygame.draw.rect(self.screen, (60, 60, 70), ga_rect, 1)
        font = self.fonts.get("placeholder")
        if font:
            text_surf = font.render(message, True, LIGHTG)
            self.screen.blit(text_surf, text_surf.get_rect(center=ga_rect.center))

    def _render_running_placeholder(
        self, ga_rect: pygame.Rect, message: str, details: List[str]
    ):
        """Renders a placeholder with more details, caching the surface."""
        current_size = ga_rect.size
        full_message_key = f"{message}::{'|'.join(details)}"  # Use combined key

        # Check cache validity
        if (
            self.placeholder_surface_cache is None
            or self.last_placeholder_size != current_size
            or self.last_placeholder_message_key != full_message_key
        ):

            logger.info(
                f"[GameArea] Recreating placeholder surface. Size: {current_size}, Key: {full_message_key}"
            )
            self.placeholder_surface_cache = pygame.Surface(current_size)
            self.placeholder_surface_cache.fill((20, 20, 25))
            pygame.draw.rect(
                self.placeholder_surface_cache,
                (60, 60, 70),
                self.placeholder_surface_cache.get_rect(),
                1,
            )

            font_p = self.fonts.get("placeholder")
            font_d = self.fonts.get("placeholder_detail")
            center_x = self.placeholder_surface_cache.get_rect().centerx
            h_needed = (font_p.get_linesize() + 5 if font_p else 0) + (
                len(details) * font_d.get_linesize() if font_d else 0
            )
            current_y = (
                self.placeholder_surface_cache.get_rect().centery - h_needed // 2
            )

            if font_p:
                surf = font_p.render(message, True, LIGHTG)
                rect = surf.get_rect(centerx=center_x, top=current_y)
                self.placeholder_surface_cache.blit(surf, rect)
                current_y = rect.bottom + 5
            if font_d:
                for line in details:
                    surf = font_d.render(line, True, WHITE)
                    rect = surf.get_rect(centerx=center_x, top=current_y)
                    self.placeholder_surface_cache.blit(surf, rect)
                    current_y += font_d.get_linesize()

            self.last_placeholder_size = current_size
            self.last_placeholder_message_key = full_message_key  # Update cache key

        # Blit cached surface
        if self.placeholder_surface_cache:
            self.screen.blit(self.placeholder_surface_cache, ga_rect.topleft)
        else:  # Fallback if cache creation failed
            self._render_placeholder(ga_rect, "Error rendering placeholder")

    def _render_best_game_state(
        self, ga_rect: pygame.Rect, state_data: Dict[str, Any], env_config: EnvConfig
    ):
        """Renders the best game state grid, caching the surface."""
        current_size = ga_rect.size
        score = state_data.get("score")
        step = state_data.get("step")

        # Check cache validity
        if (
            self.best_state_surface_cache is None
            or self.last_best_state_size != current_size
            or self.last_best_state_score != score
            or self.last_best_state_step != step
        ):

            logger.info(
                f"[GameArea] Recreating best state surface. Size: {current_size}, Score: {score}, Step: {step}"
            )
            self.best_state_surface_cache = pygame.Surface(current_size)
            self.best_state_surface_cache.fill(
                (25, 25, 30)
            )  # Slightly different background

            title_h = 60  # Height reserved for title/score/step
            grid_rect = pygame.Rect(
                0, title_h, current_size[0], current_size[1] - title_h
            )

            # Render the grid onto the cached surface
            try:
                grid_subsurface = self.best_state_surface_cache.subsurface(grid_rect)
                self._render_grid_from_data(grid_subsurface, state_data, env_config)
            except ValueError as e:
                logger.error(f"Subsurface error (best state): {e}")
                pygame.draw.rect(self.best_state_surface_cache, RED, grid_rect, 1)
            except Exception as e:
                logger.error(f"Error rendering best state grid: {e}", exc_info=True)
                pygame.draw.rect(self.best_state_surface_cache, RED, grid_rect, 2)

            # Render title and info onto the cached surface
            font_t = self.fonts.get("best_state_title")
            font_s = self.fonts.get("best_state_score")
            font_st = self.fonts.get("best_state_step")
            if font_t and font_s and font_st:
                surf_t = font_t.render("Best Game State Found", True, YELLOW)
                rect_t = surf_t.get_rect(centerx=current_size[0] // 2, top=5)
                surf_s = font_s.render(f"Score: {score}", True, WHITE)
                rect_s = surf_s.get_rect(
                    centerx=current_size[0] // 2, top=rect_t.bottom + 2
                )
                surf_st = font_st.render(
                    f"Step: {step:,}".replace(",", "_"), True, LIGHTG
                )
                rect_st = surf_st.get_rect(
                    centerx=current_size[0] // 2, top=rect_s.bottom + 1
                )
                self.best_state_surface_cache.blit(surf_t, rect_t)
                self.best_state_surface_cache.blit(surf_s, rect_s)
                self.best_state_surface_cache.blit(surf_st, rect_st)

            # Draw border on the cached surface
            pygame.draw.rect(
                self.best_state_surface_cache,
                YELLOW,
                self.best_state_surface_cache.get_rect(),
                1,
            )

            # Update cache keys
            self.last_best_state_size = current_size
            self.last_best_state_score = score
            self.last_best_state_step = step

        # Blit the cached surface
        if self.best_state_surface_cache:
            self.screen.blit(self.best_state_surface_cache, ga_rect.topleft)
        else:  # Fallback if cache creation failed
            self._render_placeholder(ga_rect, "Error rendering best state")

    def _render_grid_from_data(
        self, surf: pygame.Surface, state_data: Dict[str, Any], env_config: EnvConfig
    ):
        """Renders a grid based on stored occupancy/color data."""
        try:
            occ = state_data.get("occupancy")
            colors = state_data.get("colors")
            death = state_data.get("death")
            is_up = state_data.get("is_up")
            rows = state_data.get("rows", env_config.ROWS)
            cols = state_data.get("cols", env_config.COLS)
            if occ is None or colors is None or death is None or is_up is None:
                raise ValueError("Missing data for grid render")
            occ, death, is_up = (
                np.asarray(occ, bool),
                np.asarray(death, bool),
                np.asarray(is_up, bool),
            )
            pad = self.vis_config.ENV_GRID_PADDING * 2
            dw, dh = max(1, surf.get_width() - 2 * pad), max(
                1, surf.get_height() - 2 * pad
            )
            gr, gcw = rows, cols * 0.75 + 0.25
            scale = min(dw / gcw, dh / gr) if gr > 0 and gcw > 0 else 0
            if scale <= 0:
                return
            fpw, fph = gcw * scale, gr * scale
            tcw, tch = max(1, scale), max(1, scale)
            ox, oy = pad + (dw - fpw) / 2, pad + (dh - fph) / 2
            for r in range(rows):
                for c in range(cols):
                    if death[r, c]:
                        continue
                    tri = Triangle(r, c, is_up=is_up[r, c])
                    try:
                        pts = tri.get_points(ox=ox, oy=oy, cw=int(tcw), ch=int(tch))
                        color = VisConfig.LIGHTG
                        if occ[r, c]:
                            cell_color = colors[r][c]
                            color = (
                                tuple(cell_color)
                                if isinstance(cell_color, (list, tuple))
                                and len(cell_color) == 3
                                else VisConfig.RED
                            )
                        pygame.draw.polygon(surf, color, pts)
                        pygame.draw.polygon(surf, VisConfig.GRAY, pts, 1)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error rendering grid from data: {e}")
            pygame.draw.rect(surf, RED, surf.get_rect(), 2)

    def _calculate_grid_layout(
        self, ga_rect: pygame.Rect, num_to_render: int
    ) -> Tuple[int, int, int, int]:
        """Calculates layout for multiple small environment grids."""
        if ga_rect.width <= 0 or ga_rect.height <= 0:
            return 0, 0, 0, 0
        aspect = ga_rect.width / max(1, ga_rect.height)
        cols = max(1, int(math.sqrt(num_to_render * aspect)))
        rows = max(1, math.ceil(num_to_render / cols))
        sp = self.vis_config.ENV_SPACING
        cw = max(1, (ga_rect.width - (cols + 1) * sp) // cols)
        ch = max(1, (ga_rect.height - (rows + 1) * sp) // rows)
        return cols, rows, cw, ch

    def _render_env_grid(
        self, envs, num_to_render, env_config, ga_rect, cols, rows, cell_w, cell_h
    ):
        """Renders the grid of small environment previews."""
        env_idx = 0
        sp = self.vis_config.ENV_SPACING
        for r in range(rows):
            for c in range(cols):
                if env_idx >= num_to_render:
                    break
                env_x = ga_rect.x + sp * (c + 1) + c * cell_w
                env_y = ga_rect.y + sp * (r + 1) + r * cell_h
                env_rect = pygame.Rect(env_x, env_y, cell_w, cell_h)
                clip_rect = env_rect.clip(self.screen.get_rect())
                if clip_rect.width <= 0 or clip_rect.height <= 0:
                    env_idx += 1
                    continue
                if env_idx < len(envs) and envs[env_idx] is not None:
                    try:
                        self._render_single_env(
                            self.screen.subsurface(clip_rect), envs[env_idx], env_config
                        )
                    except Exception as e:
                        logger.error(f"Error rendering env {env_idx}: {e}")
                        pygame.draw.rect(self.screen, (50, 0, 50), clip_rect, 1)
                else:
                    pygame.draw.rect(self.screen, (20, 20, 20), clip_rect)
                    pygame.draw.rect(self.screen, (60, 60, 60), clip_rect, 1)
                env_idx += 1
            if env_idx >= num_to_render:
                break

    def _render_single_env(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        """Renders a single small environment preview."""
        cw, ch = surf.get_width(), surf.get_height()
        bg = VisConfig.GRAY
        if env.is_line_clearing():
            bg = VisConfig.LINE_CLEAR_FLASH_COLOR
        elif env.is_game_over_flashing():
            bg = VisConfig.GAME_OVER_FLASH_COLOR
        elif env.is_blinking():
            bg = VisConfig.YELLOW
        elif env.is_over():
            bg = VisConfig.DARK_RED
        elif env.is_frozen():
            bg = (30, 30, 100)
        surf.fill(bg)
        shape_h_ratio = 0.20
        grid_h = math.floor(ch * (1.0 - shape_h_ratio))
        shape_h = ch - grid_h
        shape_y = grid_h
        grid_surf, shape_surf = None, None
        if grid_h > 0 and cw > 0:
            try:
                grid_surf = surf.subsurface(pygame.Rect(0, 0, cw, grid_h))
            except ValueError:
                pass
        if shape_h > 0 and cw > 0:
            try:
                shape_surf = surf.subsurface(pygame.Rect(0, shape_y, cw, shape_h))
                shape_surf.fill((35, 35, 35))
            except ValueError:
                pass
        if grid_surf:
            self._render_single_env_grid(grid_surf, env, env_config)
        if shape_surf:
            self._render_shape_previews(shape_surf, env)
        try:
            score_surf = self.fonts["env_score"].render(
                f"GS: {env.game_score}", True, WHITE, (0, 0, 0, 180)
            )
            surf.blit(score_surf, (2, 2))
        except Exception:
            pass
        if env.is_over():
            self._render_overlay_text(surf, "GAME OVER", RED)
        elif env.is_line_clearing() and env.last_line_clear_info:
            lines, tris, _ = env.last_line_clear_info
            self._render_overlay_text(
                surf,
                f"{lines} {'Line' if lines==1 else 'Lines'} Cleared! ({tris} Tris)",
                BLUE,
            )

    def _render_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders overlay text like 'GAME OVER'."""
        try:
            font = self.fonts["env_overlay"]
            max_w = surf.get_width() * 0.9
            size = 36
            surf_txt = font.render(text, True, WHITE)
            while surf_txt.get_width() > max_w and size > 10:
                size -= 2
                font = pygame.font.SysFont(None, size)
                surf_txt = font.render(text, True, WHITE)
            bg_rgba = (color[0] // 2, color[1] // 2, color[2] // 2, 220)
            surf_bg = font.render(text, True, WHITE, bg_rgba)
            rect = surf_bg.get_rect(center=surf.get_rect().center)
            surf.blit(surf_bg, rect)
        except Exception as e:
            logger.error(f"Error rendering overlay '{text}': {e}")

    def _render_single_env_grid(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        """Renders the hexagonal grid for a single environment."""
        try:
            pad = self.vis_config.ENV_GRID_PADDING
            dw, dh = max(1, surf.get_width() - 2 * pad), max(
                1, surf.get_height() - 2 * pad
            )
            gr, gcw = env_config.ROWS, env_config.COLS * 0.75 + 0.25
            scale = min(dw / gcw, dh / gr) if gr > 0 and gcw > 0 else 0
            if scale <= 0:
                return
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

    def _render_shape_previews(self, surf: pygame.Surface, env: GameState):
        """Renders the small shape previews below the grid."""
        shapes = env.get_shapes()
        sw, sh = surf.get_width(), surf.get_height()
        if not shapes or sw <= 0 or sh <= 0:
            return
        num = len(shapes)
        pad = 4
        total_pad = (num + 1) * pad
        avail_w = sw - total_pad
        if avail_w <= 0:
            return
        w_per = avail_w / num
        h_lim = sh - 2 * pad
        dim = max(5, min(w_per, h_lim))
        start_x = pad + (sw - (num * dim + (num - 1) * pad)) / 2
        start_y = pad + (sh - dim) / 2
        curr_x = start_x
        for shape in shapes:
            rect = pygame.Rect(curr_x, start_y, dim, dim)
            if rect.right > sw - pad:
                break
            if shape is None:
                pygame.draw.rect(surf, (50, 50, 50), rect, 1, border_radius=2)
                curr_x += dim + pad
                continue
            try:
                temp_surf = pygame.Surface((dim, dim), pygame.SRCALPHA)
                temp_surf.fill((0, 0, 0, 0))
                min_r, min_c, max_r, max_c = shape.bbox()
                sh_h, sh_w = max(1, max_r - min_r + 1), max(
                    1, (max_c - min_c + 1) * 0.75 + 0.25
                )
                scale = (
                    max(1, min(dim / sh_h, dim / sh_w)) if sh_h > 0 and sh_w > 0 else 1
                )
                self._render_single_shape(temp_surf, shape, int(scale))
                surf.blit(temp_surf, rect.topleft)
                curr_x += dim + pad
            except Exception:
                pygame.draw.rect(surf, RED, rect, 1)
                curr_x += dim + pad

    def _render_single_shape(self, surf: pygame.Surface, shape: Shape, cell_size: int):
        """Renders a single shape scaled to fit."""
        if not shape or not shape.triangles or cell_size <= 0:
            return
        min_r, min_c, max_r, max_c = shape.bbox()
        sh_h, sh_w = max(1, max_r - min_r + 1), max(
            1, (max_c - min_c + 1) * 0.75 + 0.25
        )
        if sh_w <= 0 or sh_h <= 0:
            return
        total_w, total_h = sh_w * cell_size, sh_h * cell_size
        ox = (surf.get_width() - total_w) / 2 - min_c * (cell_size * 0.75)
        oy = (surf.get_height() - total_h) / 2 - min_r * cell_size
        for dr, dc, up in shape.triangles:
            tri = Triangle(row=dr, col=dc, is_up=up)
            try:
                pts = tri.get_points(ox=ox, oy=oy, cw=cell_size, ch=cell_size)
                pygame.draw.polygon(surf, shape.color, pts)
            except Exception:
                pass

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
                f"Rendering {num_rendered}/{num_total} Envs", True, YELLOW, BLACK
            )
            self.screen.blit(
                surf, surf.get_rect(bottomright=(ga_rect.right - 5, ga_rect.bottom - 5))
            )
