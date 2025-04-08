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
    DARK_RED,  # Added
    LINE_CLEAR_FLASH_COLOR,  # Added
    GAME_OVER_FLASH_COLOR,  # Added
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
        # Caches are less relevant now, but keep for potential future use (e.g., best state display on idle)
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
        if fonts.get("ui") is None:
            fonts["ui"] = pygame.font.Font(None, 24)
        if fonts.get("placeholder") is None:
            fonts["placeholder"] = pygame.font.Font(None, 30)
        if fonts.get("env_score") is None:
            fonts["env_score"] = pygame.font.Font(None, 18)
        if fonts.get("env_overlay") is None:
            fonts["env_overlay"] = pygame.font.Font(None, 36)
        return fonts

    def render(self, panel_width: int, panel_x_offset: int, **kwargs):
        """Renders the grid of live self-play environments."""
        current_height = self.screen.get_height()
        ga_rect = pygame.Rect(panel_x_offset, 0, panel_width, current_height)
        if ga_rect.width <= 0 or ga_rect.height <= 0:
            return

        # Always fill background, then render envs
        pygame.draw.rect(
            self.screen, VisConfig.DARK_GRAY, ga_rect
        )  # Darker background for env area

        envs: List[Optional[GameState]] = kwargs.get("envs", [])
        num_envs_total: int = kwargs.get("num_envs", 0)  # Total workers
        env_config: Optional[EnvConfig] = kwargs.get("env_config")

        num_to_render = len(envs)  # Number of states actually passed

        if num_to_render <= 0 or not env_config:
            self._render_placeholder(ga_rect, "Waiting for Self-Play Workers...")
            return

        cols_env, rows_env, cell_w, cell_h = self._calculate_grid_layout(
            ga_rect, num_to_render
        )

        if cell_w > 10 and cell_h > 10:  # Reduced threshold for smaller cells
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

        # Optionally show text if not all workers are being rendered
        if num_to_render < num_envs_total:
            self._render_render_limit_text(ga_rect, num_to_render, num_envs_total)

    def _render_placeholder(self, ga_rect: pygame.Rect, message: str):
        """Renders a simple placeholder message."""
        # Keep background consistent with render method
        # pygame.draw.rect(self.screen, (20, 20, 25), ga_rect)
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
        # Adjust cols/rows if calculation is inefficient
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

                env_state = envs[env_idx] if env_idx < len(envs) else None

                if env_state is not None:
                    try:
                        # Create a subsurface for each env
                        env_surf = self.screen.subsurface(clip_rect)
                        self._render_single_env(env_surf, env_state, env_config)
                    except ValueError as e:
                        logger.error(
                            f"Error creating subsurface for env {env_idx} ({clip_rect}): {e}"
                        )
                        pygame.draw.rect(
                            self.screen, (50, 0, 50), clip_rect, 1
                        )  # Error border
                    except Exception as e:
                        logger.error(f"Error rendering env {env_idx}: {e}")
                        pygame.draw.rect(
                            self.screen, (50, 0, 50), clip_rect, 1
                        )  # Error border
                else:
                    # Render placeholder for missing/stopped worker
                    pygame.draw.rect(
                        self.screen, (20, 20, 20), clip_rect
                    )  # Dark background
                    pygame.draw.rect(self.screen, (60, 60, 60), clip_rect, 1)  # Border
                    # Optional: Add text like "Stopped" or "Loading"
                    font = self.fonts.get("env_score")
                    if font:
                        text_surf = font.render("N/A", True, GRAY)
                        self.screen.blit(
                            text_surf, text_surf.get_rect(center=clip_rect.center)
                        )

                env_idx += 1
            if env_idx >= num_to_render:
                break

    def _render_single_env(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        """Renders a single small environment preview onto the provided surface."""
        cw, ch = surf.get_width(), surf.get_height()
        bg = VisConfig.GRAY  # Default background

        # Determine background based on state (using env methods)
        if env.is_line_clearing():
            bg = LINE_CLEAR_FLASH_COLOR
        elif env.is_game_over_flashing():
            bg = GAME_OVER_FLASH_COLOR
        elif env.is_blinking():
            bg = VisConfig.YELLOW  # Generic blink
        elif env.is_over():
            bg = DARK_RED
        elif env.is_frozen():
            bg = (30, 30, 100)  # Frozen state color

        surf.fill(bg)

        # --- Grid Rendering ---
        # Simplified grid rendering directly onto the surface
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
                                    color = VisConfig.LIGHTG  # Empty cell color
                                    if is_hl and (r, c) in cleared:
                                        color = hl_color
                                    elif t.is_occupied:
                                        color = t.color if t.color else VisConfig.RED
                                    pygame.draw.polygon(surf, color, pts)
                                    pygame.draw.polygon(
                                        surf, VisConfig.GRAY, pts, 1
                                    )  # Cell border
                                except Exception:
                                    pass  # Ignore minor drawing errors
                else:  # Grid data missing
                    pygame.draw.rect(surf, RED, surf.get_rect(), 2)
        except Exception:  # Major grid rendering error
            pygame.draw.rect(surf, RED, surf.get_rect(), 2)

        # --- Score Overlay ---
        try:
            score_font = self.fonts.get("env_score")
            if score_font:
                score_surf = score_font.render(
                    f"S: {env.game_score}",
                    True,
                    WHITE,
                    (0, 0, 0, 180),  # Semi-transparent black bg
                )
                surf.blit(score_surf, (2, 2))
        except Exception:
            pass  # Ignore score rendering errors

        # --- State Overlay Text (Game Over / Line Clear) ---
        if env.is_over():
            self._render_overlay_text(surf, "GAME OVER", RED)
        elif env.is_line_clearing() and env.last_line_clear_info:
            lines, tris, _ = env.last_line_clear_info
            self._render_overlay_text(
                surf,
                f"{lines} {'Line' if lines==1 else 'Lines'}!",  # Simplified message
                BLUE,
            )

        # --- Border ---
        pygame.draw.rect(
            surf, VisConfig.LIGHTG, surf.get_rect(), 1
        )  # Add border to each env cell

    def _render_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Renders overlay text like 'GAME OVER', scaling font if needed."""
        try:
            font = self.fonts.get("env_overlay")
            if not font:
                return

            max_w = surf.get_width() * 0.9
            original_size = font.get_height()  # Approximate original size
            current_size = original_size

            # Render and check width, reduce size if too wide
            surf_txt = font.render(text, True, WHITE)
            while surf_txt.get_width() > max_w and current_size > 8:
                current_size -= 2
                try:
                    font = pygame.font.SysFont(None, current_size)
                except:
                    font = pygame.font.Font(None, current_size)
                surf_txt = font.render(text, True, WHITE)

            # Render with background
            bg_rgba = (
                color[0] // 2,
                color[1] // 2,
                color[2] // 2,
                220,
            )  # Darker, semi-transparent bg
            surf_bg = font.render(text, True, WHITE, bg_rgba)
            rect = surf_bg.get_rect(center=surf.get_rect().center)
            surf.blit(surf_bg, rect)
        except Exception as e:
            logger.error(f"Error rendering overlay '{text}': {e}")

    def _render_shape_previews(self, surf: pygame.Surface, env: GameState):
        """Renders the small shape previews (Not typically shown in multi-env view)."""
        # This logic is usually for the demo mode, might not be needed here.
        # If needed, it would require significant adaptation for the small cell size.
        pass

    def _render_single_shape(self, surf: pygame.Surface, shape: Shape, cell_size: int):
        """Renders a single shape scaled to fit (Helper for previews)."""
        # This logic is usually for the demo mode.
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
                f"Rendering {num_rendered}/{num_total} Workers", True, YELLOW, BLACK
            )
            self.screen.blit(
                surf, surf.get_rect(bottomright=(ga_rect.right - 5, ga_rect.bottom - 5))
            )

    # --- Methods below might be deprecated or moved if only used by demo ---
    # _render_grid_from_data
    # _render_best_game_state
    # _render_running_placeholder
    # _render_idle_state (replaced by always rendering workers)
