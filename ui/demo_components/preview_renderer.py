# File: ui/demo_components/preview_renderer.py
import pygame
from typing import Tuple, Dict

from config import VisConfig, DemoConfig, RED, BLUE, GRAY
from environment.game_state import GameState
from ui.panels.game_area import GameAreaRenderer  # Import for base rendering


class DemoPreviewRenderer:
    """Renders the shape preview area in Demo mode."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        demo_config: DemoConfig,
        game_area_renderer: GameAreaRenderer,  # Use fonts/methods from GameAreaRenderer
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.demo_config = demo_config
        self.game_area_renderer = game_area_renderer  # Store reference
        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

    def render_shape_previews_area(
        self,
        demo_env: GameState,
        screen_width: int,
        clipped_game_rect: pygame.Rect,
        padding: int,
    ) -> Dict[int, pygame.Rect]:
        """Renders the shape preview area. Returns dict of preview rects."""
        self.shape_preview_rects.clear()  # Clear previous rects
        preview_area_w = min(150, screen_width - clipped_game_rect.right - padding // 2)
        if preview_area_w <= 20:
            return self.shape_preview_rects

        preview_area_rect = pygame.Rect(
            clipped_game_rect.right + padding // 2,
            clipped_game_rect.top,
            preview_area_w,
            clipped_game_rect.height,
        )
        clipped_preview_area_rect = preview_area_rect.clip(self.screen.get_rect())
        if (
            clipped_preview_area_rect.width <= 0
            or clipped_preview_area_rect.height <= 0
        ):
            return self.shape_preview_rects

        try:
            preview_area_surf = self.screen.subsurface(clipped_preview_area_rect)
            self.shape_preview_rects = self._render_demo_shape_previews(
                preview_area_surf, demo_env, preview_area_rect.topleft
            )
        except ValueError as e:
            print(f"Error subsurface demo shape preview area: {e}")
            pygame.draw.rect(self.screen, RED, clipped_preview_area_rect, 1)
        except Exception as e:
            print(f"Error rendering demo shape previews: {e}")
            traceback.print_exc()
        return self.shape_preview_rects

    def _render_demo_shape_previews(
        self, surf: pygame.Surface, env: GameState, area_topleft: Tuple[int, int]
    ) -> Dict[int, pygame.Rect]:
        """Renders the small previews of available shapes. Returns dict of screen rects."""
        calculated_rects: Dict[int, pygame.Rect] = {}
        surf.fill((25, 25, 25))
        all_slots = env.shapes
        selected_idx = env.demo_selected_shape_idx
        dragged_idx = env.demo_dragged_shape_idx

        num_slots = env.env_config.NUM_SHAPE_SLOTS
        surf_w, surf_h = surf.get_size()
        preview_padding = 5

        if num_slots <= 0:
            return calculated_rects

        preview_h = max(20, (surf_h - (num_slots + 1) * preview_padding) / num_slots)
        preview_w = max(20, surf_w - 2 * preview_padding)
        current_preview_y = preview_padding

        for i in range(num_slots):
            shape_in_slot = all_slots[i] if i < len(all_slots) else None
            preview_rect_local = pygame.Rect(
                preview_padding, current_preview_y, preview_w, preview_h
            )
            preview_rect_screen = preview_rect_local.move(area_topleft)
            calculated_rects[i] = preview_rect_screen

            clipped_preview_rect = preview_rect_local.clip(surf.get_rect())
            if clipped_preview_rect.width <= 0 or clipped_preview_rect.height <= 0:
                current_preview_y += preview_h + preview_padding
                continue

            bg_color = (40, 40, 40)
            border_color = GRAY
            border_width = 1
            if i == selected_idx and shape_in_slot is not None:
                border_color = self.demo_config.SELECTED_SHAPE_HIGHLIGHT_COLOR
                border_width = 3
            elif i == dragged_idx:
                border_color = (100, 100, 255)
                border_width = 2

            pygame.draw.rect(surf, bg_color, clipped_preview_rect, border_radius=3)
            pygame.draw.rect(
                surf, border_color, clipped_preview_rect, border_width, border_radius=3
            )

            if shape_in_slot is not None:
                self._render_single_shape_in_preview_box(
                    surf, shape_in_slot, preview_rect_local, clipped_preview_rect
                )

            current_preview_y += preview_h + preview_padding
        return calculated_rects

    def _render_single_shape_in_preview_box(
        self,
        surf: pygame.Surface,
        shape_obj,
        preview_rect: pygame.Rect,
        clipped_preview_rect: pygame.Rect,
    ):
        """Renders a single shape scaled to fit within its preview box."""
        try:
            inner_padding = 2
            shape_render_area_rect = pygame.Rect(
                inner_padding,
                inner_padding,
                clipped_preview_rect.width - 2 * inner_padding,
                clipped_preview_rect.height - 2 * inner_padding,
            )
            if shape_render_area_rect.width <= 0 or shape_render_area_rect.height <= 0:
                return

            temp_surf = pygame.Surface(shape_render_area_rect.size, pygame.SRCALPHA)
            temp_surf.fill((0, 0, 0, 0))

            min_r, min_c, max_r, max_c = shape_obj.bbox()
            shape_h_cells = max(1, max_r - min_r + 1)
            shape_w_cells_eff = max(1, (max_c - min_c + 1) * 0.75 + 0.25)
            scale_h = shape_render_area_rect.height / shape_h_cells
            scale_w = shape_render_area_rect.width / shape_w_cells_eff
            cell_size = max(1, min(scale_h, scale_w))

            # Use the GameAreaRenderer's shape rendering logic
            self.game_area_renderer._render_single_shape(
                temp_surf, shape_obj, int(cell_size)
            )

            surf.blit(
                temp_surf, shape_render_area_rect.move(preview_rect.topleft).topleft
            )

        except ValueError as sub_err:
            print(f"Error subsurface shape preview: {sub_err}")
            pygame.draw.rect(surf, RED, clipped_preview_rect, 1)
        except Exception as e:
            print(f"Error rendering demo shape preview: {e}")
            pygame.draw.rect(surf, RED, clipped_preview_rect, 1)

    def get_shape_preview_rects(self) -> Dict[int, pygame.Rect]:
        """Returns the dictionary of screen-relative shape preview rects."""
        return self.shape_preview_rects.copy()
