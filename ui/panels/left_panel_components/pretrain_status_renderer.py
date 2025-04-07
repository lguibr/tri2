# File: ui/panels/left_panel_components/pretrain_status_renderer.py
import pygame
from typing import Dict, Any, Tuple, Optional
from config import YELLOW, LIGHTG, GOOGLE_COLORS
from utils.helpers import format_eta  # Import from new location


class PretrainStatusRenderer:
    """Renders the status and progress of the pre-training phase."""

    def __init__(self, screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        self.screen = screen
        self.fonts = fonts
        self.status_font = fonts.get("pretrain_status", pygame.font.Font(None, 20))
        self.progress_font = fonts.get(
            "pretrain_progress_bar", pygame.font.Font(None, 14)
        )
        self.detail_font = fonts.get("pretrain_detail", pygame.font.Font(None, 16))

    def render(
        self, y_start: int, pretrain_info: Dict[str, Any], panel_width: int
    ) -> Tuple[int, Dict[str, pygame.Rect]]:
        """Renders the pre-training status block. Returns next_y and stat_rects."""
        stat_rects: Dict[str, pygame.Rect] = {}
        if not pretrain_info or not self.status_font or not self.detail_font:
            return y_start, stat_rects

        x_margin = 10
        current_y = y_start
        line_height_status = self.status_font.get_linesize()
        line_height_detail = self.detail_font.get_linesize()

        phase = pretrain_info.get("phase", "Unknown")
        status_text = f"Pre-Train: {phase}"
        status_color = YELLOW
        detail_text = ""

        overall_eta_str = format_eta(
            pretrain_info.get("overall_eta_seconds", pretrain_info.get("eta_seconds"))
        )
        if phase == "Random Play":
            games = pretrain_info.get("games_played", 0)
            target = pretrain_info.get("target_games", 0)
            pps = pretrain_info.get("plays_per_second", 0.0)
            num_envs = pretrain_info.get("num_envs", 0)
            status_text = f"Pre-Train: Random Play ({games:,}/{target:,})"
            detail_text = (
                f"{num_envs} Envs | {pps:.1f} Plays/s | ETA: {overall_eta_str}"
            )
            status_color = GOOGLE_COLORS[1]
        elif phase == "Sorting Games":
            status_text = "Pre-Train: Sorting Games..."
            status_color = (200, 150, 50)
        elif phase == "Replaying Top K":
            replayed = pretrain_info.get("games_replayed", 0)
            target = pretrain_info.get("target_games", 0)
            transitions = pretrain_info.get("transitions_collected", 0)
            status_text = f"Pre-Train: Replaying ({replayed:,}/{target:,})"
            detail_text = f"Collecting Transitions ({transitions:,})"
            status_color = (100, 180, 180)
        elif phase == "Updating Agent":
            epoch = pretrain_info.get("epoch", 0)
            total_epochs = pretrain_info.get("total_epochs", 0)
            status_text = f"Pre-Train: Updating (Epoch {epoch}/{total_epochs})"
            detail_text = f"Overall ETA: {overall_eta_str}"  # Show ETA here
            status_color = GOOGLE_COLORS[2]

        # Render Status Line
        status_surface = self.status_font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect(topleft=(x_margin, current_y))
        clip_width_status = max(0, panel_width - status_rect.left - x_margin)
        if status_rect.width > clip_width_status:
            self.screen.blit(
                status_surface,
                status_rect,
                area=pygame.Rect(0, 0, clip_width_status, status_rect.height),
            )
        else:
            self.screen.blit(status_surface, status_rect)
        stat_rects["Pre-training Status"] = status_rect
        current_y += line_height_status

        # Render Detail Line
        if detail_text:
            detail_surface = self.detail_font.render(detail_text, True, LIGHTG)
            detail_rect = detail_surface.get_rect(topleft=(x_margin + 2, current_y))
            clip_width_detail = max(0, panel_width - detail_rect.left - x_margin)
            if detail_rect.width > clip_width_detail:
                self.screen.blit(
                    detail_surface,
                    detail_rect,
                    area=pygame.Rect(0, 0, clip_width_detail, detail_rect.height),
                )
            else:
                self.screen.blit(detail_surface, detail_rect)
            current_y += line_height_detail
        current_y += 5  # Add final padding

        return current_y, stat_rects
