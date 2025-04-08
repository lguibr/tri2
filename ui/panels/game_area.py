# File: ui/panels/game_area.py
import pygame
import math
import traceback
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import copy  # Keep for deepcopying best game state data if needed

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
    DARK_GREEN,
    GREEN,  # Added GREEN
    LINE_CLEAR_FLASH_COLOR,
    GAME_OVER_FLASH_COLOR,
)
from environment.game_state import GameState, StateType  # Added StateType
from environment.shape import Shape
from environment.triangle import Triangle

logger = logging.getLogger(__name__)


class GameAreaRenderer:
    """Renders the right panel: Shows live self-play worker environments and best game state."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()
        self.best_state_surface_cache: Optional[pygame.Surface] = None
        self.last_best_state_size: Tuple[int, int] = (0, 0)
        self.last_best_state_score: Optional[int] = None
        self.last_best_state_step: Optional[int] = None
        self.placeholder_surface_cache: Optional[pygame.Surface] = None
        self.last_placeholder_size: Tuple[int, int] = (0, 0)
        self.last_placeholder_message_key: str = ""

    def _init_fonts(self):
        """Initializes fonts used in the game area."""
        fonts = {}
        # Define font configurations {name: size}
        font_configs = {
            "env_score": 16,
            "env_overlay": 24,
            "env_info": 14,
            "ui": 24,
            "placeholder": 30,
            "placeholder_detail": 22,
            "best_state_title": 20,  # Slightly smaller title
            "best_state_score": 18,  # Smaller score/step
            "best_state_step": 16,
        }
        for key, size in font_configs.items():
            try:
                fonts[key] = pygame.font.SysFont(None, size)
            except Exception:
                try:
                    fonts[key] = pygame.font.Font(
                        None, size
                    )  # Fallback to default font
                except Exception as e:
                    logger.error(f"ERROR: Font '{key}' failed to load: {e}")
                    fonts[key] = None  # Mark as None if loading fails
        # Fallbacks for essential fonts
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
        if fonts.get("best_state_title") is None:
            fonts["best_state_title"] = pygame.font.Font(None, 20)
        if fonts.get("best_state_score") is None:
            fonts["best_state_score"] = pygame.font.Font(None, 18)
        if fonts.get("best_state_step") is None:
            fonts["best_state_step"] = pygame.font.Font(None, 16)
        return fonts

    def render(
        self,
        panel_width: int,
        panel_x_offset: int,
        worker_render_data: List[
            Optional[Dict[str, Any]]
        ],  # List of dicts {state_dict: StateType, stats: Dict}
        num_envs: int,
        env_config: Optional[EnvConfig],
        best_game_state_data: Optional[Dict[str, Any]],
    ):
        """Renders the grid of live self-play environments and best game found."""
        current_height = self.screen.get_height()
        ga_rect = pygame.Rect(panel_x_offset, 0, panel_width, current_height)

        if ga_rect.width <= 0 or ga_rect.height <= 0:
            return  # Nothing to render

        pygame.draw.rect(self.screen, VisConfig.DARK_GRAY, ga_rect)  # Background

        # --- Layout Calculation ---
        # Reserve space for the "Best Game" display at the top
        best_game_display_height = (
            int(current_height * 0.2) if best_game_state_data else 0
        )
        best_game_rect = pygame.Rect(
            ga_rect.x, ga_rect.y, ga_rect.width, best_game_display_height
        )
        env_grid_rect = pygame.Rect(
            ga_rect.x,
            ga_rect.y + best_game_display_height,
            ga_rect.width,
            ga_rect.height - best_game_display_height,
        )

        # --- Render Best Game State ---
        if best_game_display_height > 0 and best_game_rect.height > 20:
            self._render_best_game(best_game_rect, best_game_state_data, env_config)
        elif best_game_state_data and best_game_rect.height <= 20:
            logger.warning("Best game display area too small to render.")

        # --- Render Environment Grid ---
        num_to_render = self.vis_config.NUM_ENVS_TO_RENDER
        actual_render_data = worker_render_data[
            :num_to_render
        ]  # Use only the requested number

        if not actual_render_data or not env_config:
            self._render_placeholder(env_grid_rect, "Waiting for Self-Play Workers...")
            return

        cols_env, rows_env, cell_w, cell_h_total = self._calculate_grid_layout(
            env_grid_rect, len(actual_render_data)
        )

        if cell_w > 10 and cell_h_total > 40:  # Check if cells are reasonably sized
            self._render_env_grid(
                actual_render_data,
                env_config,
                env_grid_rect,
                cols_env,
                rows_env,
                cell_w,
                cell_h_total,
            )
        else:
            self._render_too_small_message(env_grid_rect, cell_w, cell_h_total)

        # Render text if not all workers are shown
        if len(actual_render_data) < num_envs:
            self._render_render_limit_text(ga_rect, len(actual_render_data), num_envs)

    def _render_placeholder(self, area_rect: pygame.Rect, message: str):
        """Renders a simple placeholder message."""
        # Draw border for the placeholder area
        pygame.draw.rect(self.screen, (60, 60, 70), area_rect, 1)
        font = self.fonts.get("placeholder")
        if font:
            text_surf = font.render(message, True, LIGHTG)
            self.screen.blit(text_surf, text_surf.get_rect(center=area_rect.center))

    def _calculate_grid_layout(
        self, available_rect: pygame.Rect, num_items: int
    ) -> Tuple[int, int, int, int]:
        """Calculates layout for multiple small environment grids within the available rect."""
        if available_rect.width <= 0 or available_rect.height <= 0 or num_items <= 0:
            return 0, 0, 0, 0

        # Try to fit items optimally within the aspect ratio
        aspect = available_rect.width / max(1, available_rect.height)
        cols = max(1, int(math.sqrt(num_items * aspect)))
        rows = max(1, math.ceil(num_items / cols))

        # Adjust rows/cols to minimize empty space while fitting num_items
        while cols * rows < num_items:
            if (cols + 1) / rows < aspect:  # Adding column is closer to aspect ratio
                cols += 1
            else:
                rows += 1
        while cols * (rows - 1) >= num_items and rows > 1:
            rows -= 1
        while (cols - 1) * rows >= num_items and cols > 1:
            cols -= 1

        sp = self.vis_config.ENV_SPACING
        cw_total = max(1, (available_rect.width - (cols + 1) * sp) // cols)
        ch_total = max(1, (available_rect.height - (rows + 1) * sp) // rows)

        return cols, rows, cw_total, ch_total

    def _render_env_grid(
        self,
        worker_render_data: List[
            Optional[Dict[str, Any]]
        ],  # List of dicts {state_dict: StateType, stats: Dict}
        env_config: EnvConfig,
        grid_area_rect: pygame.Rect,
        cols: int,
        rows: int,
        cell_w: int,
        cell_h_total: int,
    ):
        """Renders the grid of small environment previews."""
        env_idx = 0
        sp = self.vis_config.ENV_SPACING
        # Define heights for parts within a cell
        info_h = 12  # Height for worker ID, step
        shapes_h = 22  # Height for shape previews
        grid_cell_h = max(
            1, cell_h_total - shapes_h - info_h - 2
        )  # Remaining height for grid

        for r in range(rows):
            for c in range(cols):
                if env_idx >= len(worker_render_data):
                    break  # Stop if we've rendered all available data

                # Calculate position of the current cell
                env_x = grid_area_rect.x + sp * (c + 1) + c * cell_w
                env_y = grid_area_rect.y + sp * (r + 1) + r * cell_h_total
                env_rect_total = pygame.Rect(env_x, env_y, cell_w, cell_h_total)

                # Ensure the cell is within screen bounds before creating subsurface
                clip_rect_total = env_rect_total.clip(self.screen.get_rect())
                if clip_rect_total.width <= 0 or clip_rect_total.height <= 0:
                    env_idx += 1
                    continue

                render_data = worker_render_data[env_idx]

                # Define local rects relative to the cell's top-left (0,0)
                grid_rect_local = pygame.Rect(0, 0, cell_w, grid_cell_h)
                shapes_rect_local = pygame.Rect(0, grid_cell_h + 1, cell_w, shapes_h)
                info_rect_local = pygame.Rect(
                    0, grid_cell_h + shapes_h + 2, cell_w, info_h
                )

                try:
                    # Create a subsurface for the current cell
                    cell_surf = self.screen.subsurface(clip_rect_total)
                    cell_surf.fill(VisConfig.DARK_GRAY)  # Base background

                    if render_data and render_data.get("state_dict"):
                        env_state_dict: StateType = render_data["state_dict"]
                        env_stats: Dict[str, Any] = render_data.get("stats", {})

                        # Render Grid part using the state dictionary
                        if grid_rect_local.height > 0:
                            clipped_grid_rect_local = grid_rect_local.clip(
                                cell_surf.get_rect()
                            )
                            if (
                                clipped_grid_rect_local.width > 0
                                and clipped_grid_rect_local.height > 0
                            ):
                                grid_surf = cell_surf.subsurface(
                                    clipped_grid_rect_local
                                )
                                # Pass occupancy grid and orientation grid from state_dict
                                occupancy_grid = env_state_dict.get("grid")[
                                    0
                                ]  # Channel 0
                                orientation_grid = env_state_dict.get("grid")[
                                    1
                                ]  # Channel 1
                                self._render_single_env_grid_from_arrays(
                                    grid_surf,
                                    occupancy_grid,
                                    orientation_grid,
                                    env_config,
                                    env_stats,
                                )

                        # Render Shapes part using the state dictionary
                        if shapes_rect_local.height > 0:
                            clipped_shapes_rect_local = shapes_rect_local.clip(
                                cell_surf.get_rect()
                            )
                            if (
                                clipped_shapes_rect_local.width > 0
                                and clipped_shapes_rect_local.height > 0
                            ):
                                shapes_surf = cell_surf.subsurface(
                                    clipped_shapes_rect_local
                                )
                                # Pass shape features and availability from state_dict
                                shape_features = env_state_dict.get("shapes")
                                shape_availability = env_state_dict.get(
                                    "shape_availability"
                                )
                                self._render_env_shapes_from_arrays(
                                    shapes_surf,
                                    shape_features,
                                    shape_availability,
                                    env_config,
                                )

                        # Render Info part using stats
                        if info_rect_local.height > 0:
                            clipped_info_rect_local = info_rect_local.clip(
                                cell_surf.get_rect()
                            )
                            if (
                                clipped_info_rect_local.width > 0
                                and clipped_info_rect_local.height > 0
                            ):
                                info_surf = cell_surf.subsurface(
                                    clipped_info_rect_local
                                )
                                self._render_env_info(info_surf, env_idx, env_stats)

                    else:  # Render placeholder if no state_dict/data
                        pygame.draw.rect(
                            cell_surf, (20, 20, 20), cell_surf.get_rect()
                        )  # Darker placeholder bg
                        pygame.draw.rect(
                            cell_surf, (60, 60, 60), cell_surf.get_rect(), 1
                        )  # Border
                        font = self.fonts.get("env_info")
                        if font:
                            status = (
                                render_data.get("stats", {}).get("status", "N/A")
                                if render_data
                                else "N/A"
                            )
                            text_surf = font.render(status, True, GRAY)
                            # Center text within the cell
                            cell_surf.blit(
                                text_surf,
                                text_surf.get_rect(center=cell_surf.get_rect().center),
                            )

                    # Draw border around the entire cell
                    pygame.draw.rect(
                        cell_surf, VisConfig.LIGHTG, cell_surf.get_rect(), 1
                    )

                except ValueError as e:
                    logger.error(
                        f"Error creating subsurface for env cell {env_idx} ({clip_rect_total}): {e}"
                    )
                    pygame.draw.rect(
                        self.screen, (50, 0, 50), clip_rect_total, 1
                    )  # Indicate error with purple border
                except Exception as e:
                    logger.error(
                        f"Error rendering env cell {env_idx}: {e}", exc_info=True
                    )
                    pygame.draw.rect(
                        self.screen, (50, 0, 50), clip_rect_total, 1
                    )  # Indicate error

                env_idx += 1
            if env_idx >= len(worker_render_data):
                break

    def _render_single_env_grid_from_arrays(
        self,
        surf: pygame.Surface,
        occupancy_grid: np.ndarray,  # Shape [Rows, Cols]
        orientation_grid: np.ndarray,  # Shape [Rows, Cols]
        env_config: EnvConfig,
        stats: Dict[str, Any],  # Pass stats for score overlay
    ):
        """Renders the grid portion using occupancy and orientation arrays."""
        cw, ch = surf.get_width(), surf.get_height()
        # Determine background based on status (less info available than full GameState)
        bg = VisConfig.GRAY  # Default background
        # Simplified background logic based on stats if needed
        # if stats.get("status") == "Finished": bg = DARK_RED
        surf.fill(bg)

        try:
            pad = self.vis_config.ENV_GRID_PADDING
            dw, dh = max(1, cw - 2 * pad), max(1, ch - 2 * pad)
            gr, gcw_eff = env_config.ROWS, env_config.COLS * 0.75 + 0.25
            if gr <= 0 or gcw_eff <= 0:
                return

            scale = min(dw / gcw_eff, dh / gr)
            if scale <= 0:
                return

            tcw, tch = max(1, scale), max(1, scale)
            fpw, fph = gcw_eff * scale, gr * scale
            ox, oy = pad + (dw - fpw) / 2, pad + (dh - fph) / 2

            rows, cols = occupancy_grid.shape
            for r in range(rows):
                for c in range(cols):
                    is_occupied = occupancy_grid[r, c] > 0.5
                    orientation = orientation_grid[r, c]
                    is_up = orientation > 0.5  # Assuming 1.0 for up, -1.0 for down

                    # Basic check if cell is potentially playable (can refine if death mask available)
                    # This assumes death cells have occupancy 0 and orientation 0
                    is_playable = not (occupancy_grid[r, c] == 0 and orientation == 0)

                    if is_playable:
                        temp_tri = Triangle(r, c, is_up)
                        try:
                            pts = temp_tri.get_points(
                                ox=ox, oy=oy, cw=int(tcw), ch=int(tch)
                            )
                            color = VisConfig.LIGHTG  # Default empty color
                            if is_occupied:
                                # Use a fixed color as specific shape color isn't in state_dict easily
                                color = VisConfig.WHITE
                            pygame.draw.polygon(surf, color, pts)
                            pygame.draw.polygon(
                                surf, VisConfig.GRAY, pts, 1
                            )  # Grid lines
                        except Exception as tri_err:
                            # logger.info(f"Error drawing triangle ({r},{c}): {tri_err}")
                            pass  # Ignore errors for single triangles
        except Exception as grid_err:
            logger.error(f"Error rendering grid from arrays: {grid_err}")
            pygame.draw.rect(surf, RED, surf.get_rect(), 2)  # Indicate error

        # Score Overlay
        try:
            score_font = self.fonts.get("env_score")
            score = stats.get("game_score", "?")  # Get score from stats dict
            if score_font:
                score_surf = score_font.render(
                    f"S: {score}", True, WHITE, (0, 0, 0, 180)
                )
                surf.blit(score_surf, (2, 2))
        except Exception as e:
            logger.error(f"Error rendering score overlay: {e}")

        # State Overlays (Simplified - based on status if available)
        status = stats.get("status", "")
        if "Error" in status:
            self._render_overlay_text(surf, "ERROR", RED)
        elif "Finished" in status:
            self._render_overlay_text(surf, "DONE", BLUE)

    def _render_env_shapes_from_arrays(
        self,
        surf: pygame.Surface,
        shape_features: np.ndarray,  # Shape [NUM_SLOTS * FEATURES_PER_SHAPE]
        shape_availability: np.ndarray,  # Shape [NUM_SLOTS]
        env_config: EnvConfig,
    ):
        """Renders the available shapes using feature and availability arrays."""
        sw, sh = surf.get_width(), surf.get_height()
        if sw <= 0 or sh <= 0:
            return

        num_slots = env_config.NUM_SHAPE_SLOTS
        features_per_shape = env_config.SHAPE_FEATURES_PER_SHAPE
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

        # Reshape features if needed (assuming flattened)
        if (
            shape_features.ndim == 1
            and shape_features.shape[0] == num_slots * features_per_shape
        ):
            shape_features = shape_features.reshape((num_slots, features_per_shape))
        elif shape_features.shape != (num_slots, features_per_shape):
            logger.error(f"Unexpected shape_features shape: {shape_features.shape}")
            return  # Cannot render shapes

        for i in range(num_slots):
            is_available = (
                shape_availability[i] > 0.5 if i < len(shape_availability) else False
            )
            rect = pygame.Rect(int(curr_x), int(start_y), dim, dim)
            if rect.right > sw - pad:
                break

            pygame.draw.rect(
                surf, (50, 50, 50), rect, border_radius=2
            )  # Background box

            if is_available:
                try:
                    # Render a simple representation based on features (e.g., number of triangles)
                    # This is a placeholder - rendering exact shape from features is complex
                    num_tris_norm = shape_features[i, 0]  # Normalized num triangles
                    num_tris = int(num_tris_norm * 6.0)  # Denormalize approx
                    tris_font = self.fonts.get("env_info", pygame.font.Font(None, 14))
                    if tris_font:
                        tris_surf = tris_font.render(f"{num_tris}T", True, WHITE)
                        surf.blit(tris_surf, tris_surf.get_rect(center=rect.center))
                    else:  # Fallback circle
                        pygame.draw.circle(surf, WHITE, rect.center, dim // 3)

                except Exception as e:
                    logger.error(f"Error rendering shape preview from features: {e}")
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
        # Keep text concise
        info_text = f"W:{worker_idx} S:{game_step}"
        try:
            text_surf = font.render(info_text, True, LIGHTG)
            # Center the text horizontally within the info area
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

            max_w = surf.get_width() * 0.9  # Max width relative to surface
            original_size = font.get_height()  # Requires font init
            current_size = original_size  # Start with default size

            # Render text and check width
            surf_txt = font.render(text, True, WHITE)
            # Reduce font size if text is too wide
            while surf_txt.get_width() > max_w and current_size > 8:
                current_size -= 2
                try:
                    # Try system font first, then default
                    font = pygame.font.SysFont(None, current_size)
                except:
                    font = pygame.font.Font(None, current_size)
                surf_txt = font.render(text, True, WHITE)

            # Render with background color for better visibility
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

    def _render_single_shape(
        self,
        surf: pygame.Surface,
        shape: Shape,
        target_dim: int,  # Simplified: target square size
    ):
        """Renders a single shape scaled to fit within target_dim x target_dim."""
        if not shape or not shape.triangles or target_dim <= 0:
            return

        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h_cells = max(1, max_r - min_r + 1)
        shape_w_cells_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)

        scale_h = target_dim / shape_h_cells if shape_h_cells > 0 else target_dim
        scale_w = (
            target_dim / shape_w_cells_eff if shape_w_cells_eff > 0 else target_dim
        )
        scale = max(1, min(scale_h, scale_w))

        total_w_pixels = shape_w_cells_eff * scale
        total_h_pixels = shape_h_cells * scale
        ox = (target_dim - total_w_pixels) / 2 - min_c * (scale * 0.75)
        oy = (target_dim - total_h_pixels) / 2 - min_r * scale

        for dr, dc, is_up in shape.triangles:
            temp_tri = Triangle(0, 0, is_up)
            try:
                tri_x = ox + dc * (scale * 0.75)
                tri_y = oy + dr * scale
                pts = [
                    (int(p[0]), int(p[1]))
                    for p in temp_tri.get_points(
                        ox=tri_x, oy=tri_y, cw=int(scale), ch=int(scale)
                    )
                ]
                pygame.draw.polygon(surf, shape.color, pts)
                pygame.draw.polygon(surf, BLACK, pts, 1)
            except Exception as e:
                # logger.info(f"Error drawing triangle in shape preview: {e}")
                pass

    def _render_too_small_message(
        self, area_rect: pygame.Rect, cell_w: int, cell_h: int
    ):
        """Renders a message if the env cells are too small."""
        font = self.fonts.get("ui")
        if font:
            surf = font.render(f"Envs Too Small ({cell_w}x{cell_h})", True, GRAY)
            self.screen.blit(surf, surf.get_rect(center=area_rect.center))

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

    def _render_best_game(
        self,
        area_rect: pygame.Rect,
        best_game_data: Optional[Dict[str, Any]],
        env_config: Optional[EnvConfig],
    ):
        """Renders the best game state found so far."""
        pygame.draw.rect(self.screen, (20, 40, 20), area_rect)  # Dark green background
        pygame.draw.rect(self.screen, GREEN, area_rect, 1)  # Green border

        if not best_game_data or not env_config:
            self._render_placeholder(area_rect, "No Best Game Yet")
            return

        title_font = self.fonts.get("best_state_title")
        score_font = self.fonts.get("best_state_score")
        step_font = self.fonts.get("best_state_step")

        # --- Layout ---
        padding = 5
        text_area_width = 100  # Width for Score/Step text
        grid_area_width = area_rect.width - text_area_width - 3 * padding
        grid_area_height = area_rect.height - 2 * padding

        text_area_rect = pygame.Rect(
            area_rect.left + padding,
            area_rect.top + padding,
            text_area_width,
            grid_area_height,
        )
        grid_area_rect = pygame.Rect(
            text_area_rect.right + padding,
            area_rect.top + padding,
            grid_area_width,
            grid_area_height,
        )

        # --- Render Text ---
        if title_font and score_font and step_font:
            score = best_game_data.get("score", "N/A")
            step = best_game_data.get("step", "N/A")

            title_surf = title_font.render("Best Game", True, YELLOW)
            score_surf = score_font.render(f"Score: {score}", True, WHITE)
            step_surf = step_font.render(f"Step: {step}", True, LIGHTG)

            title_rect = title_surf.get_rect(
                midtop=(text_area_rect.centerx, text_area_rect.top + 2)
            )
            score_rect = score_surf.get_rect(
                midtop=(text_area_rect.centerx, title_rect.bottom + 4)
            )
            step_rect = step_surf.get_rect(
                midtop=(text_area_rect.centerx, score_rect.bottom + 2)
            )

            self.screen.blit(title_surf, title_rect)
            self.screen.blit(score_surf, score_rect)
            self.screen.blit(step_surf, step_rect)

        # --- Render Grid ---
        if grid_area_rect.width > 10 and grid_area_rect.height > 10:
            # Use the stored state dictionary to render
            game_state_dict = best_game_data.get("game_state_dict")
            if game_state_dict and isinstance(game_state_dict, dict):
                try:
                    if "grid" in game_state_dict:
                        grid_state_array = game_state_dict["grid"]
                        if (
                            isinstance(grid_state_array, np.ndarray)
                            and grid_state_array.ndim == 3
                            and grid_state_array.shape[0]
                            >= 2  # Need occupancy and orientation
                        ):
                            # Use the rendering function that takes arrays
                            self._render_grid_from_array(
                                grid_area_rect,
                                grid_state_array[0],  # Occupancy
                                grid_state_array[1],  # Orientation
                                env_config,
                            )
                        else:
                            logger.warning(
                                f"Best game state 'grid' data invalid shape: {grid_state_array.shape if isinstance(grid_state_array, np.ndarray) else type(grid_state_array)}"
                            )
                            self._render_placeholder(
                                grid_area_rect, "Grid Data Invalid"
                            )
                    else:
                        logger.warning("Best game state dict missing 'grid' key.")
                        self._render_placeholder(grid_area_rect, "Grid Data Missing")

                except Exception as e:
                    logger.error(f"Error rendering best game grid: {e}", exc_info=True)
                    self._render_placeholder(grid_area_rect, "Grid Render Error")
            else:
                logger.warning("Best game state data missing or not a dictionary.")
                self._render_placeholder(grid_area_rect, "State Missing")
        else:
            # Grid area too small
            pass

    def _render_grid_from_array(
        self,
        area_rect: pygame.Rect,
        occupancy_grid: np.ndarray,  # Shape [Rows, Cols]
        orientation_grid: np.ndarray,  # Shape [Rows, Cols]
        env_config: EnvConfig,
    ):
        """Simplified grid rendering directly from occupancy and orientation arrays."""
        try:
            clipped_area_rect = area_rect.clip(self.screen.get_rect())
            if clipped_area_rect.width <= 0 or clipped_area_rect.height <= 0:
                return
            grid_surf = self.screen.subsurface(clipped_area_rect)
            grid_surf.fill(BLACK)

            rows, cols = occupancy_grid.shape
            if rows == 0 or cols == 0:
                return

            pad = 1
            dw, dh = max(1, clipped_area_rect.width - 2 * pad), max(
                1, clipped_area_rect.height - 2 * pad
            )
            gcw_eff = env_config.COLS * 0.75 + 0.25
            if env_config.ROWS <= 0 or gcw_eff <= 0:
                return

            scale = min(dw / gcw_eff, dh / env_config.ROWS)
            if scale <= 0:
                return

            tcw, tch = max(1, scale), max(1, scale)
            fpw, fph = gcw_eff * scale, env_config.ROWS * scale
            ox, oy = pad + (dw - fpw) / 2, pad + (dh - fph) / 2

            for r in range(rows):
                for c in range(cols):
                    is_occupied = occupancy_grid[r, c] > 0.5
                    orientation = orientation_grid[r, c]
                    is_up = orientation > 0.5  # Assuming 1.0 for up, -1.0 for down
                    # Basic check if cell is potentially playable
                    is_playable = not (occupancy_grid[r, c] == 0 and orientation == 0)

                    if is_playable:
                        temp_tri = Triangle(r, c, is_up)
                        try:
                            pts = temp_tri.get_points(
                                ox=ox, oy=oy, cw=int(tcw), ch=int(tch)
                            )
                            color = VisConfig.LIGHTG  # Default empty
                            if is_occupied:
                                color = WHITE  # Fixed color for occupied in best view
                            pygame.draw.polygon(grid_surf, color, pts)
                            pygame.draw.polygon(grid_surf, GRAY, pts, 1)  # Grid lines
                        except Exception:
                            pass  # Ignore single triangle errors
        except ValueError as e:
            logger.error(
                f"Error creating subsurface for best grid ({clipped_area_rect}): {e}"
            )
            pygame.draw.rect(self.screen, RED, clipped_area_rect, 1)
        except Exception as e:
            logger.error(f"Error in _render_grid_from_array: {e}", exc_info=True)
            pygame.draw.rect(self.screen, RED, clipped_area_rect, 1)
