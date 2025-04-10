# File: src/visualization/drawing/previews.py
import pygame
import logging
from typing import TYPE_CHECKING, Tuple, Dict

# Use relative imports within visualization package for core elements
from ..core import colors, coord_mapper

# Import the specific function needed directly using relative path
from .shapes import draw_shape

# Import Triangle and Shape from the new structs module
from src.structs import Triangle, Shape

if TYPE_CHECKING:
    from src.environment import GameState  # GameState remains in environment
    from src.config import EnvConfig, VisConfig

logger = logging.getLogger(__name__)

# === Preview Area Rendering ===


def render_previews(
    surface: pygame.Surface,
    game_state: "GameState",
    area_topleft: Tuple[int, int],  # Screen coord of preview area's top-left
    mode: str,
    env_config: "EnvConfig",
    vis_config: "VisConfig",
) -> Dict[int, pygame.Rect]:
    """Renders shape previews in their area. Returns dict {index: screen_rect}."""
    surface.fill(colors.PREVIEW_BG)
    preview_rects_screen: Dict[int, pygame.Rect] = {}
    num_slots = env_config.NUM_SHAPE_SLOTS
    pad = vis_config.PREVIEW_PADDING
    inner_pad = vis_config.PREVIEW_INNER_PADDING
    border = vis_config.PREVIEW_BORDER_WIDTH
    selected_border = vis_config.PREVIEW_SELECTED_BORDER_WIDTH

    if num_slots <= 0:
        return {}

    total_pad_h = (num_slots + 1) * pad
    available_h = surface.get_height() - total_pad_h
    slot_h = available_h / num_slots if num_slots > 0 else 0
    slot_w = surface.get_width() - 2 * pad

    current_y = pad

    for i in range(num_slots):
        slot_rect_local = pygame.Rect(pad, current_y, slot_w, slot_h)
        # Calculate screen coordinates for the returned dictionary
        slot_rect_screen = slot_rect_local.move(area_topleft)
        preview_rects_screen[i] = slot_rect_screen

        shape: Optional[Shape] = game_state.shapes[i]  # Use Shape from structs
        is_selected = mode == "play" and game_state.demo_selected_shape_idx == i

        # Draw border
        border_width = selected_border if is_selected else border
        border_color = (
            colors.PREVIEW_SELECTED_BORDER if is_selected else colors.PREVIEW_BORDER
        )
        pygame.draw.rect(surface, border_color, slot_rect_local, border_width)

        # Draw shape inside the slot if available
        if shape:
            # Calculate drawing area inside border and padding
            draw_area_w = slot_w - 2 * (border_width + inner_pad)
            draw_area_h = slot_h - 2 * (border_width + inner_pad)

            if draw_area_w > 0 and draw_area_h > 0:
                # Determine cell size based on shape bounds and available area
                min_r, min_c, max_r, max_c = shape.bbox()
                shape_rows = max_r - min_r + 1
                # Effective cols calculation needs care for triangles
                shape_cols_eff = (
                    (max_c - min_c + 1) * 0.75 + 0.25 if shape.triangles else 1
                )

                scale_w = (
                    draw_area_w / shape_cols_eff if shape_cols_eff > 0 else draw_area_w
                )
                scale_h = draw_area_h / shape_rows if shape_rows > 0 else draw_area_h
                cell_size = max(1.0, min(scale_w, scale_h))

                # Center the shape within the drawing area
                shape_render_w = shape_cols_eff * cell_size
                shape_render_h = shape_rows * cell_size
                draw_topleft_x = (
                    slot_rect_local.left
                    + border_width
                    + inner_pad
                    + (draw_area_w - shape_render_w) / 2
                )
                draw_topleft_y = (
                    slot_rect_local.top
                    + border_width
                    + inner_pad
                    + (draw_area_h - shape_render_h) / 2
                )

                # Use the directly imported draw_shape function
                draw_shape(
                    surface,
                    shape,
                    (draw_topleft_x, draw_topleft_y),
                    cell_size,
                    is_selected=is_selected,  # Pass selection status if needed by draw_shape
                    origin_offset=(-min_r, -min_c),  # Offset drawing by shape's min r/c
                )

        current_y += slot_h + pad

    return preview_rects_screen


# === Placement/Hover Preview Rendering ===


def draw_placement_preview(
    surface: pygame.Surface,
    shape: "Shape",  # Use Shape from structs
    r: int,
    c: int,
    is_valid: bool,  # Caller determines validity
    config: "EnvConfig",
) -> None:
    """Draws a semi-transparent shape snapped to the grid."""
    if not shape or not shape.triangles:
        return

    # Use coord_mapper to get rendering parameters for the target surface (grid)
    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    color = list(shape.color) + [150]  # Add alpha channel
    # Use a different color/alpha if invalid? For now, just use shape color + alpha
    # color = colors.PLACEMENT_VALID_COLOR if is_valid else colors.PLACEMENT_INVALID_COLOR

    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    temp_surface.fill((0, 0, 0, 0))  # Transparent background

    for dr, dc, is_up in shape.triangles:
        tri_r, tri_c = r + dr, c + dc
        # Create temporary triangle for geometry calculation
        temp_tri = Triangle(tri_r, tri_c, is_up)  # Use Triangle from structs
        pts = temp_tri.get_points(ox, oy, cw, ch)
        pygame.draw.polygon(temp_surface, color, pts)

    surface.blit(temp_surface, (0, 0))


def draw_floating_preview(
    surface: pygame.Surface,
    shape: "Shape",  # Use Shape from structs
    screen_pos: Tuple[int, int],  # Mouse position relative to the surface
    config: "EnvConfig",
) -> None:
    """Draws a semi-transparent shape floating at the screen position."""
    if not shape or not shape.triangles:
        return

    # Estimate cell size based on typical grid rendering (might not be perfect)
    # A fixed size might be better here? Or pass VisConfig?
    # Let's use a fixed moderate size for floating preview
    cell_size = 20.0
    color = list(shape.color) + [100]  # More transparent

    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    temp_surface.fill((0, 0, 0, 0))

    # Calculate relative center of the shape to draw around mouse pos
    min_r, min_c, max_r, max_c = shape.bbox()
    center_r = (min_r + max_r) / 2.0
    center_c = (min_c + max_c) / 2.0

    for dr, dc, is_up in shape.triangles:
        # Calculate position relative to mouse cursor, centered
        # Adjusting for triangle geometry (0.75 factor)
        pt_x = screen_pos[0] + (dc - center_c) * (cell_size * 0.75)
        pt_y = screen_pos[1] + (dr - center_r) * cell_size

        # Create temporary triangle for geometry calculation at this floating position
        # The r/c here are just for orientation, position is handled by pt_x, pt_y
        temp_tri = Triangle(0, 0, is_up)  # Use Triangle from structs
        # Get points relative to (0,0) with cell_size, then offset by pt_x, pt_y
        pts = [
            (px + pt_x, py + pt_y)
            for px, py in temp_tri.get_points(0, 0, cell_size, cell_size)
        ]
        pygame.draw.polygon(temp_surface, color, pts)

    surface.blit(temp_surface, (0, 0))
