# File: src/visualization/drawing/hud.py
import pygame
from typing import TYPE_CHECKING, Dict, Optional, Any

# Use relative imports within visualization package
from ..core import colors

if TYPE_CHECKING:
    from ...environment import GameState  # Correct path


def render_hud(
    surface: pygame.Surface,
    game_state: "GameState",  # Can be a representative state or dummy
    mode: str,
    fonts: Dict[str, Optional[pygame.font.Font]],
    display_stats: Optional[Dict[str, Any]] = None,  # Should contain global stats now
) -> None:
    """
    Renders global information (like step count, worker status) at the bottom.
    Individual game scores are not shown here anymore.
    """
    screen_w, screen_h = surface.get_size()
    help_font = fonts.get("help")
    stats_font = fonts.get("help")  # Use same font for stats

    # --- Bottom HUD: Global Stats and Help Text ---
    bottom_y = screen_h - 10  # Bottom padding

    # Global Stats (bottom-left)
    stats_rect = None  # Initialize stats_rect
    if stats_font and display_stats:
        stats_items = []
        # Get global step from the trainer's progress bar if available
        train_progress = display_stats.get("train_progress")
        global_step = (
            train_progress.current_steps
            if train_progress
            else display_stats.get("global_step", "?")
        )

        episodes = display_stats.get("total_episodes", "?")
        sims = display_stats.get("total_simulations", "?")
        num_workers = display_stats.get("num_workers", "?")
        pending_tasks = display_stats.get("pending_tasks", "?")

        stats_items.append(f"Step: {global_step}")
        stats_items.append(f"Episodes: {episodes}")
        if isinstance(sims, (int, float)):
            sims_str = (
                f"{sims/1e6:.2f}M"
                if sims >= 1e6
                else (f"{sims/1e3:.1f}k" if sims >= 1000 else str(int(sims)))
            )
            stats_items.append(f"Sims: {sims_str}")
        stats_items.append(f"Workers: {pending_tasks}/{num_workers} busy")

        stats_text = " | ".join(stats_items)
        stats_surf = stats_font.render(stats_text, True, colors.CYAN)
        stats_rect = stats_surf.get_rect(bottomleft=(15, bottom_y))
        surface.blit(stats_surf, stats_rect)

    # Help Text (bottom-right)
    if help_font:
        help_text = "[ESC] Quit"
        # Add mode-specific help if needed, but keep it simple
        # if mode == "training_visual":
        #     help_text += " | Training Visual Mode"

        help_surf = help_font.render(help_text, True, colors.LIGHT_GRAY)
        help_rect = help_surf.get_rect(bottomright=(screen_w - 15, bottom_y))
        # Adjust position if overlapping with stats
        if stats_rect and stats_rect.right > help_rect.left - 10:
            help_rect.right = screen_w - 15  # Keep right aligned
            help_rect.bottom = (
                bottom_y - stats_rect.height - 5
            )  # Move up if stats are long

        surface.blit(help_surf, help_rect)
