# File: ui/overlays.py
import pygame
import time
from typing import Tuple, List
from config import VisConfig


class OverlayRenderer:
    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = self._init_fonts()

    def _init_fonts(self):
        fonts = {}
        try:
            fonts["overlay_title"] = pygame.font.SysFont(None, 36)
            fonts["overlay_text"] = pygame.font.SysFont(None, 24)
            # --- REMOVED: Toast font ---
            # fonts["toast"] = pygame.font.SysFont(None, 20)
        except Exception as e:
            print(f"Warning: SysFont error: {e}. Using default.")
            fonts["overlay_title"] = pygame.font.Font(None, 36)
            fonts["overlay_text"] = pygame.font.Font(None, 24)
            # --- REMOVED: Toast font ---
            # fonts["toast"] = pygame.font.Font(None, 20)
        return fonts

    def render_cleanup_confirmation(self):
        """Renders the confirmation dialog for cleanup."""
        current_width, current_height = self.screen.get_size()
        # Semi-transparent overlay
        overlay_surface = pygame.Surface(
            (current_width, current_height), pygame.SRCALPHA
        )
        overlay_surface.fill((0, 0, 0, 200))  # Black with alpha
        self.screen.blit(overlay_surface, (0, 0))

        center_x, center_y = current_width // 2, current_height // 2

        # Text lines
        prompt_l1 = self.fonts["overlay_title"].render(
            "DELETE CURRENT RUN DATA?", True, VisConfig.RED
        )
        prompt_l2 = self.fonts["overlay_text"].render(
            "(Agent Checkpoint & Buffer State)", True, VisConfig.WHITE
        )
        prompt_l3 = self.fonts["overlay_text"].render(
            "This action cannot be undone!", True, VisConfig.YELLOW
        )

        # Position and blit text
        self.screen.blit(
            prompt_l1, prompt_l1.get_rect(center=(center_x, center_y - 60))
        )
        self.screen.blit(
            prompt_l2, prompt_l2.get_rect(center=(center_x, center_y - 25))
        )
        self.screen.blit(prompt_l3, prompt_l3.get_rect(center=(center_x, center_y)))

        # Buttons
        confirm_yes_rect = pygame.Rect(center_x - 110, center_y + 30, 100, 40)
        confirm_no_rect = pygame.Rect(center_x + 10, center_y + 30, 100, 40)

        pygame.draw.rect(
            self.screen, (0, 150, 0), confirm_yes_rect, border_radius=5
        )  # Green YES
        pygame.draw.rect(
            self.screen, (150, 0, 0), confirm_no_rect, border_radius=5
        )  # Red NO

        yes_text = self.fonts["overlay_text"].render("YES", True, VisConfig.WHITE)
        no_text = self.fonts["overlay_text"].render("NO", True, VisConfig.WHITE)

        self.screen.blit(yes_text, yes_text.get_rect(center=confirm_yes_rect.center))
        self.screen.blit(no_text, no_text.get_rect(center=confirm_no_rect.center))

    def render_status_message(self, message: str, last_message_time: float) -> bool:
        """Renders a status message (e.g., after cleanup) temporarily."""
        # Check if message exists and hasn't expired
        if not message or (time.time() - last_message_time >= 5.0):
            return False  # Message expired or empty

        current_width, current_height = self.screen.get_size()
        lines = message.split("\n")
        max_width = 0
        msg_surfs = []

        # Render each line and find max width
        for line in lines:
            # Render with black background for better visibility
            msg_surf = self.fonts["overlay_text"].render(
                line, True, VisConfig.YELLOW, VisConfig.BLACK
            )
            msg_surfs.append(msg_surf)
            max_width = max(max_width, msg_surf.get_width())

        # Calculate background size
        total_height = (
            sum(s.get_height() for s in msg_surfs) + max(0, len(lines) - 1) * 2
        )  # Add spacing between lines
        padding = 5
        bg_rect = pygame.Rect(0, 0, max_width + padding * 2, total_height + padding * 2)

        # Position at bottom-center
        bg_rect.midbottom = (current_width // 2, current_height - 10)

        # Draw background and border
        pygame.draw.rect(self.screen, VisConfig.BLACK, bg_rect, border_radius=3)
        pygame.draw.rect(
            self.screen, VisConfig.YELLOW, bg_rect, 1, border_radius=3
        )  # Yellow border

        # Draw text lines centered within the background
        current_y = bg_rect.top + padding
        for msg_surf in msg_surfs:
            msg_rect = msg_surf.get_rect(midtop=(bg_rect.centerx, current_y))
            self.screen.blit(msg_surf, msg_rect)
            current_y += msg_surf.get_height() + 2  # Add spacing

        return True  # Indicate that a message was rendered