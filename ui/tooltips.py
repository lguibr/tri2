# File: ui/tooltips.py
# --- Tooltip Rendering Logic ---
import pygame
from typing import Tuple, Dict, Optional
from config import VisConfig


class TooltipRenderer:
    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.font_tooltip = self._init_font()
        self.hovered_stat_key: Optional[str] = None
        self.stat_rects: Dict[str, pygame.Rect] = {}  # Updated by UIRenderer
        self.tooltip_texts: Dict[str, str] = {}  # Updated by UIRenderer

    def _init_font(self):
        try:
            return pygame.font.SysFont(None, 18)
        except Exception as e:
            print(f"Warning: SysFont error: {e}. Using default.")
            return pygame.font.Font(None, 18)

    def check_hover(self, mouse_pos: Tuple[int, int]):
        """Checks if the mouse is hovering over any registered stat rect."""
        self.hovered_stat_key = None
        # Iterate in reverse order so tooltips for elements drawn last appear first
        for key, rect in reversed(self.stat_rects.items()):
            # Ensure rect is valid before checking collision
            if (
                rect
                and rect.width > 0
                and rect.height > 0
                and rect.collidepoint(mouse_pos)
            ):
                self.hovered_stat_key = key
                return  # Found one, stop checking

    def render_tooltip(self):
        """Renders the tooltip if a stat is being hovered over."""
        if not self.hovered_stat_key or self.hovered_stat_key not in self.tooltip_texts:
            return  # No active hover or no text for this key

        tooltip_text = self.tooltip_texts[self.hovered_stat_key]
        mouse_pos = pygame.mouse.get_pos()

        # --- Text Wrapping ---
        lines = []
        max_width = 300  # Max tooltip width
        words = tooltip_text.split(" ")
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_surf = self.font_tooltip.render(test_line, True, VisConfig.BLACK)
            if test_surf.get_width() <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)  # Add the last line

        # --- Rendering ---
        line_surfs = [
            self.font_tooltip.render(line, True, VisConfig.BLACK) for line in lines
        ]
        if not line_surfs:
            return  # No lines to render

        total_height = sum(s.get_height() for s in line_surfs)
        max_line_width = max(s.get_width() for s in line_surfs)

        padding = 5
        tooltip_rect = pygame.Rect(
            mouse_pos[0] + 15,  # Offset from cursor
            mouse_pos[1] + 10,
            max_line_width + padding * 2,
            total_height + padding * 2,
        )

        # Ensure tooltip stays on screen
        tooltip_rect.clamp_ip(self.screen.get_rect())

        # Draw background and border
        pygame.draw.rect(self.screen, VisConfig.YELLOW, tooltip_rect, border_radius=3)
        pygame.draw.rect(self.screen, VisConfig.BLACK, tooltip_rect, 1, border_radius=3)

        # Draw text lines
        current_y = tooltip_rect.y + padding
        for surf in line_surfs:
            self.screen.blit(surf, (tooltip_rect.x + padding, current_y))
            current_y += surf.get_height()

    def update_rects_and_texts(
        self, rects: Dict[str, pygame.Rect], texts: Dict[str, str]
    ):
        """Updates the dictionaries used for hover detection and text lookup."""
        self.stat_rects = rects
        self.tooltip_texts = texts
