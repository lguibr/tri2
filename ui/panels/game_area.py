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
import numpy as np  # Import numpy


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
            fonts["ui"] = pygame.font.SysFont(None, 24)
        except Exception as e:
            print(f"Warning: SysFont error: {e}. Using default.")
            fonts["env_score"] = pygame.font.Font(None, 18)
            fonts["env_overlay"] = pygame.font.Font(None, 36)
            fonts["ui"] = pygame.font.Font(None, 24)
        return fonts

    def render(self, envs: List[GameState], num_envs: int, env_config: EnvConfig):
        # ... (render method remains mostly the same, calls _render_env_grid)
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

        # Check minimum size *before* trying to render grid or message
        min_cell_dim = 30  # Adjust as needed
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

        if num_to_render < num_envs:
            self._render_render_limit_text(ga_rect, num_to_render, num_envs)

    def _calculate_grid_layout(
        self, ga_rect: pygame.Rect, num_to_render: int
    ) -> Tuple[int, int, int, int]:
        # ... (this method remains the same)
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

                # Clip the rect to the screen to prevent subsurface errors
                clipped_env_rect = env_rect.clip(self.screen.get_rect())
                if clipped_env_rect.width <= 0 or clipped_env_rect.height <= 0:
                    env_idx += 1
                    continue  # Skip rendering if clipped rect is invalid

                try:
                    # Use the clipped rect for the subsurface
                    sub_surf = self.screen.subsurface(clipped_env_rect)
                    # Adjust rendering position if clipped (sub_surf topleft is relative)
                    # We'll handle offsets inside _render_single_env now
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

                env_idx += 1

    # --- REPLACED: _render_single_env ---
    def _render_single_env(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        """Renders the grid, shapes, scores, and overlays for a single environment cell."""
        cell_w = surf.get_width()
        cell_h = surf.get_height()
        if cell_w <= 0 or cell_h <= 0:
            return

        # --- MODIFIED: Background Color Logic ---
        # Determine background color based on env state, checking flash first
        bg_color = VisConfig.GRAY  # Default background
        if env.is_line_clearing():  # Check flash first
            bg_color = VisConfig.LINE_CLEAR_FLASH_COLOR
        elif env.is_blinking():
            bg_color = VisConfig.YELLOW
        elif env.is_frozen() and not env.is_over():
            bg_color = (30, 30, 100)  # Frozen blue
        elif env.is_over():
            bg_color = (40, 20, 20)  # Game over dark red
        # else: bg_color remains VisConfig.GRAY
        surf.fill(bg_color)
        # --- END MODIFIED ---

        # --- Define Areas ---
        shape_area_height_ratio = 0.20  # Use 20% for shapes
        grid_area_height = math.floor(cell_h * (1.0 - shape_area_height_ratio))
        shape_area_height = cell_h - grid_area_height
        shape_area_y = grid_area_height

        # Create subsurfaces for grid and shapes if dimensions are valid
        grid_surf = None
        shape_surf = None
        if grid_area_height > 0 and cell_w > 0:
            try:
                grid_rect = pygame.Rect(0, 0, cell_w, grid_area_height)
                grid_surf = surf.subsurface(grid_rect)
            except ValueError as e:
                print(f"Warning: Grid subsurface error ({grid_rect}): {e}")
                pygame.draw.rect(surf, VisConfig.RED, grid_rect, 1)

        if shape_area_height > 0 and cell_w > 0:
            try:
                shape_rect = pygame.Rect(0, shape_area_y, cell_w, shape_area_height)
                shape_surf = surf.subsurface(shape_rect)
                shape_surf.fill((35, 35, 35))  # Darker background for shapes
            except ValueError as e:
                print(f"Warning: Shape subsurface error ({shape_rect}): {e}")
                pygame.draw.rect(surf, VisConfig.RED, shape_rect, 1)

        # --- Render Grid ---
        if grid_surf:
            self._render_single_env_grid(grid_surf, env, env_config)

        # --- Render Shapes ---
        if shape_surf:
            self._render_shape_previews(shape_surf, env)

        # --- Render Scores (on top of everything in the cell) ---
        try:
            score_surf = self.fonts["env_score"].render(
                f"GS: {env.game_score} R: {env.score:.1f}",
                True,
                VisConfig.WHITE,
                (0, 0, 0, 180),  # Semi-transparent black background
            )
            # Position score at top-left of the main surface `surf`
            surf.blit(score_surf, (2, 2))
        except Exception as e:
            print(f"Error rendering score: {e}")

        # --- Render Overlays (Game Over/Frozen) ---
        if env.is_over():
            self._render_overlay_text(surf, "GAME OVER", VisConfig.RED)
        elif (
            env.is_frozen() and not env.is_blinking() and not env.is_line_clearing()
        ):  # Dont show frozen if flashing
            # Only show "Frozen" if not game over
            self._render_overlay_text(surf, "Frozen", VisConfig.BLUE)

    def _render_overlay_text(
        self, surf: pygame.Surface, text: str, color: Tuple[int, int, int]
    ):
        """Helper to render centered overlay text with a background."""
        try:
            overlay_font = self.fonts["env_overlay"]
            text_surf = overlay_font.render(
                text,
                True,
                VisConfig.WHITE,
                (color[0] // 2, color[1] // 2, color[2] // 2, 200),
            )
            text_rect = text_surf.get_rect(center=surf.get_rect().center)
            surf.blit(text_surf, text_rect)
        except Exception as e:
            print(f"Error rendering overlay text '{text}': {e}")

    # --- USE SIMPLIFIED SCALING ---
    def _render_single_env_grid(
        self, surf: pygame.Surface, env: GameState, env_config: EnvConfig
    ):
        """Renders the playable grid, scaled and centered within the surface."""
        try:
            padding = self.vis_config.ENV_GRID_PADDING
            drawable_w = max(
                1, surf.get_width() - 2 * padding
            )  # Ensure at least 1 pixel
            drawable_h = max(
                1, surf.get_height() - 2 * padding
            )  # Ensure at least 1 pixel

            # Calculate the required dimensions in "triangle units"
            grid_rows = env_config.ROWS
            grid_cols_effective_width = env_config.COLS * 0.75 + 0.25

            if grid_rows <= 0 or grid_cols_effective_width <= 0:
                return

            # Calculate scale factors based on available drawable space
            scale_w_based = drawable_w / grid_cols_effective_width
            scale_h_based = drawable_h / grid_rows

            # Choose the MINIMUM scale factor to ensure it fits in both dimensions
            final_scale = min(scale_w_based, scale_h_based)
            if final_scale <= 0:
                return  # Cannot render with zero or negative scale

            # Calculate final pixel dimensions of the grid using the chosen scale
            final_grid_pixel_w = grid_cols_effective_width * final_scale
            final_grid_pixel_h = grid_rows * final_scale

            # Calculate triangle cell dimensions based on the final scale
            # Assuming base unit height = 1, base unit width = 1 before scaling
            tri_cell_h = max(1, final_scale)
            tri_cell_w = max(
                1, final_scale
            )  # Simplification: Scale width same as height

            # Calculate centering offsets within the padded drawable area
            grid_ox = padding + (drawable_w - final_grid_pixel_w) / 2
            grid_oy = padding + (drawable_h - final_grid_pixel_h) / 2

            # Render Grid Triangles (only non-death)
            if hasattr(env, "grid") and hasattr(env.grid, "triangles"):
                for r in range(env.grid.rows):
                    for c in range(env.grid.cols):
                        # Check bounds carefully
                        if not (
                            0 <= r < len(env.grid.triangles)
                            and 0 <= c < len(env.grid.triangles[r])
                        ):
                            continue  # Skip if out of bounds
                        t = env.grid.triangles[r][c]
                        # --- Only render non-death cells ---
                        if not t.is_death:
                            if not hasattr(t, "get_points"):
                                continue
                            try:
                                pts = t.get_points(
                                    ox=grid_ox,
                                    oy=grid_oy,
                                    cw=int(tri_cell_w),
                                    ch=int(tri_cell_h),
                                )
                                color = VisConfig.LIGHTG  # Default empty color
                                if t.is_occupied:
                                    color = t.color if t.color else VisConfig.RED
                                pygame.draw.polygon(surf, color, pts)
                                # Optional: Draw border for empty cells
                                pygame.draw.polygon(
                                    surf, VisConfig.GRAY, pts, 1
                                )  # Add border for clarity
                            except Exception as e_render:
                                # Reduce log spam for minor render errors
                                # print(f"Error rendering tri ({r},{c}): {e_render}")
                                pass  # Silently skip triangle if points error
            else:
                # Draw error indication if grid is missing
                pygame.draw.rect(surf, VisConfig.RED, surf.get_rect(), 2)
                err_txt = self.fonts["ui"].render(
                    "Invalid Grid Data", True, VisConfig.RED
                )
                surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))

        except Exception as e:
            print(f"Unexpected Render Error in _render_single_env_grid: {e}")
            traceback.print_exc()
            pygame.draw.rect(surf, VisConfig.RED, surf.get_rect(), 2)

    # --- MODIFIED: Renders shapes in their allocated area ---
    def _render_shape_previews(self, surf: pygame.Surface, env: GameState):
        """Renders small previews of available shapes horizontally in the given surface."""
        available_shapes = env.get_shapes()
        if not available_shapes:
            return

        surf_w = surf.get_width()
        surf_h = surf.get_height()
        if surf_w <= 0 or surf_h <= 0:
            return

        num_shapes = len(available_shapes)
        padding = 4  # Padding around shapes
        total_padding_needed = (num_shapes + 1) * padding
        available_width_for_shapes = surf_w - total_padding_needed

        if available_width_for_shapes <= 0:
            return  # Not enough space

        # Calculate width per shape, limited by available height
        width_per_shape = available_width_for_shapes / num_shapes
        height_limit = surf_h - 2 * padding
        preview_dim = max(
            5, min(width_per_shape, height_limit)
        )  # Square-ish preview size

        start_x = (
            padding
            + (surf_w - (num_shapes * preview_dim + (num_shapes - 1) * padding)) / 2
        )
        start_y = padding + (surf_h - preview_dim) / 2

        current_x = start_x
        for shape in available_shapes:
            preview_rect = pygame.Rect(current_x, start_y, preview_dim, preview_dim)
            if (
                preview_rect.right > surf_w - padding
            ):  # Prevent overflow if calculation is slightly off
                break

            try:
                # Create a temporary surface for the shape preview
                temp_shape_surf = pygame.Surface(
                    (preview_dim, preview_dim), pygame.SRCALPHA
                )
                temp_shape_surf.fill((0, 0, 0, 0))  # Transparent background

                # Calculate cell size to fit the shape bbox within preview_dim
                min_r, min_c, max_r, max_c = shape.bbox()
                shape_h_cells = max(1, max_r - min_r + 1)
                shape_w_cells_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)

                scale_h = preview_dim / shape_h_cells
                scale_w = preview_dim / shape_w_cells_eff
                cell_size = max(
                    1, min(scale_h, scale_w)
                )  # Base cell size on limiting dimension

                # Render the shape centered on the temp surface
                self._render_single_shape(temp_shape_surf, shape, cell_size)

                # Blit the temp surface onto the main shape preview surface
                surf.blit(temp_shape_surf, preview_rect.topleft)
                current_x += preview_dim + padding

            except Exception as e:
                print(f"Error rendering shape preview: {e}")
                pygame.draw.rect(surf, VisConfig.RED, preview_rect, 1)  # Draw error box
                current_x += preview_dim + padding  # Still advance

    def _render_single_shape(self, surf: pygame.Surface, shape: Shape, cell_size: int):
        # ... (this method remains the same - renders a shape centered on the given surface)
        if not shape or not shape.triangles or cell_size <= 0:
            return
        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h_cells = max_r - min_r + 1
        shape_w_cells_eff = (max_c - min_c + 1) * 0.75 + 0.25
        if shape_w_cells_eff <= 0 or shape_h_cells <= 0:
            return

        # Calculate pixel dimensions required by the shape at this cell_size
        total_w_pixels = shape_w_cells_eff * cell_size
        total_h_pixels = shape_h_cells * cell_size

        # Calculate offset to center the shape on the surface
        # Offset relative to the shape's min_r, min_c
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
        # ... (this method remains the same)
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
        # ... (this method remains the same)
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
