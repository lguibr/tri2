# File: ui/panels/left_panel.py
import pygame
import os
import time
from typing import Dict, Any, Optional, Deque, Tuple

from config import (
    VisConfig,
    BufferConfig,
    StatsConfig,
    DQNConfig,
    DEVICE,
    TensorBoardConfig,
)
from config.general import TOTAL_TRAINING_STEPS
from ui.plotter import Plotter

# Tooltips specific to this panel
TOOLTIP_TEXTS = {
    "Status": "Current state: Paused, Buffering, Training, Confirm Cleanup, Cleaning, or Error.",
    "Global Steps": "Total environment steps taken / Total planned steps.",
    "Total Episodes": "Total completed episodes across all environments.",
    "Steps/Sec (Current)": f"Current avg Steps/Sec (~{StatsConfig.STATS_AVG_WINDOW} intervals). See plot for history.",
    "Buffer Fill": f"Current replay buffer fill % ({BufferConfig.REPLAY_BUFFER_SIZE / 1e6:.1f}M cap). See plot.",
    "PER Beta": f"Current PER IS exponent ({BufferConfig.PER_BETA_START:.1f}->1.0). See plot.",
    "Learning Rate": "Current learning rate. See plot for history/schedule.",
    "Train Button": "Click to Start/Pause training (or press 'P').",
    "Cleanup Button": "Click to DELETE agent ckpt & buffer for CURRENT run ONLY, then re-init.",
    "Device": f"Computation device detected ({DEVICE.type.upper()}).",
    "Network": f"CNN+MLP Fusion. Dueling={DQNConfig.USE_DUELING}, Noisy={DQNConfig.USE_NOISY_NETS}, C51={DQNConfig.USE_DISTRIBUTIONAL}",
    "TensorBoard Status": "Indicates TB logging status and log directory.",
    "Notification Area": "Displays the latest best achievements (RL Score, Game Score, Loss).",
    # --- MODIFIED Tooltips ---
    "Best RL Score Info": "Best RL Score achieved: Current Value (Previous Value) - Steps Ago",
    "Best Game Score Info": "Best Game Score achieved: Current Value (Previous Value) - Steps Ago",
    "Best Loss Info": "Best (Lowest) Loss achieved: Current Value (Previous Value) - Steps Ago",
    # --- END MODIFIED ---
}


class LeftPanelRenderer:
    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.stat_rects: Dict[str, pygame.Rect] = {}

    def _init_fonts(self):
        """Loads necessary fonts, attempting to load DejaVuSans for notifications."""
        fonts = {}
        default_font_size = 24
        status_font_size = 28
        logdir_font_size = 16
        plot_placeholder_size = 20
        notification_size = 19
        notification_label_size = 16

        notification_font_path = os.path.join("fonts", "DejaVuSans.ttf")
        try:
            fonts["notification"] = pygame.font.Font(
                notification_font_path, notification_size
            )
            print(f"Loaded notification font: {notification_font_path}")
        except pygame.error as e:
            print(
                f"Warning: Could not load '{notification_font_path}': {e}. Falling back to SysFont."
            )
            try:
                fonts["notification"] = pygame.font.SysFont(None, notification_size)
            except Exception:
                print("Warning: SysFont fallback failed. Using default font.")
                fonts["notification"] = pygame.font.Font(None, notification_size)

        try:
            fonts["ui"] = pygame.font.SysFont(None, default_font_size)
            fonts["status"] = pygame.font.SysFont(None, status_font_size)
            fonts["logdir"] = pygame.font.SysFont(None, logdir_font_size)
            fonts["plot_placeholder"] = pygame.font.SysFont(None, plot_placeholder_size)
            fonts["notification_label"] = pygame.font.SysFont(
                None, notification_label_size
            )
        except Exception as e:
            print(f"Warning: SysFont error loading other fonts: {e}. Using default.")
            fonts["ui"] = pygame.font.Font(None, default_font_size)
            fonts["status"] = pygame.font.Font(None, status_font_size)
            fonts["logdir"] = pygame.font.Font(None, logdir_font_size)
            fonts["plot_placeholder"] = pygame.font.Font(None, plot_placeholder_size)
            fonts["notification_label"] = fonts.get(
                "notification", pygame.font.Font(None, notification_label_size)
            )

        for key, size in [
            ("ui", default_font_size),
            ("status", status_font_size),
            ("logdir", logdir_font_size),
            ("plot_placeholder", plot_placeholder_size),
            ("notification", notification_size),
            ("notification_label", notification_label_size),
        ]:
            if key not in fonts:
                print(
                    f"Warning: Font '{key}' completely failed to load. Using pygame default."
                )
                fonts[key] = pygame.font.Font(None, size)
        return fonts

    # --- MODIFIED: Clarify steps ago ---
    def _format_steps_ago(self, current_step: int, best_step: int) -> str:
        """Formats the difference in steps into a readable string (k steps, M steps)."""
        if best_step <= 0 or current_step <= best_step:
            return "Now"
        diff = current_step - best_step
        if diff < 1000:
            return f"{diff} steps ago"  # Add "steps"
        elif diff < 1_000_000:
            return f"{diff / 1000:.1f}k steps ago"  # Add "steps"
        else:
            return f"{diff / 1_000_000:.1f}M steps ago"  # Add "steps"

    # --- END MODIFIED ---

    def render(
        self,
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
    ):
        """Renders the entire left panel."""
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(300, self.vis_config.LEFT_PANEL_WIDTH))
        lp_rect = pygame.Rect(0, 0, lp_width, current_height)

        status_color_map = {
            "Paused": (30, 30, 30),
            "Buffering": (30, 40, 30),
            "Training": (40, 30, 30),
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
        }
        bg_color = status_color_map.get(status, (30, 30, 30))
        pygame.draw.rect(self.screen, bg_color, lp_rect)

        self.stat_rects.clear()

        # 1. Buttons
        train_btn_rect = pygame.Rect(10, 10, 100, 40)
        cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 160, 40)
        self._draw_button(
            train_btn_rect,
            "Pause" if is_training and status == "Training" else "Train",
            (70, 70, 70),
        )
        self._draw_button(cleanup_btn_rect, "Cleanup This Run", (100, 40, 40))
        self.stat_rects["Train Button"] = train_btn_rect
        self.stat_rects["Cleanup Button"] = cleanup_btn_rect

        # 2. Status Text
        status_surf = self.fonts["status"].render(
            f"Status: {status}", True, VisConfig.YELLOW
        )
        status_rect = status_surf.get_rect(topleft=(10, train_btn_rect.bottom + 10))
        self.screen.blit(status_surf, status_rect)
        self.stat_rects["Status"] = status_rect
        current_y = status_rect.bottom + 5

        # 3. Notification Area
        notification_area_x = cleanup_btn_rect.right + 15
        notification_area_y = train_btn_rect.top
        notification_area_width = lp_width - notification_area_x - 10
        notification_line_height = self.fonts["notification"].get_linesize()
        notification_area_height = notification_line_height * 3 + 12

        if notification_area_width > 50:
            notification_area_rect = pygame.Rect(
                notification_area_x,
                notification_area_y,
                notification_area_width,
                notification_area_height,
            )
            self._render_notification_area(notification_area_rect, stats_summary)
            current_y = max(current_y, notification_area_rect.bottom + 10)
        else:
            current_y = max(current_y, train_btn_rect.bottom + 10)

        # 4. Info Text Block
        last_text_y = self._render_info_text_reduced(
            current_y, stats_summary, buffer_capacity, lp_width
        )

        # 5. TensorBoard Status
        last_text_y = self._render_tb_status(last_text_y + 10, tensorboard_log_dir)

        # 6. Plot Area
        self._render_plot_area(
            last_text_y + 15, lp_width, current_height, plot_data, status
        )

    def _draw_button(self, rect: pygame.Rect, text: str, color: Tuple[int, int, int]):
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        lbl_surf = self.fonts["ui"].render(text, True, VisConfig.WHITE)
        self.screen.blit(lbl_surf, lbl_surf.get_rect(center=rect.center))

    def _render_notification_area(self, area_rect: pygame.Rect, stats: Dict[str, Any]):
        """Renders the latest best score/loss messages with details."""
        pygame.draw.rect(self.screen, (45, 45, 45), area_rect, border_radius=3)
        pygame.draw.rect(self.screen, VisConfig.LIGHTG, area_rect, 1, border_radius=3)
        self.stat_rects["Notification Area"] = area_rect

        padding = 5
        line_height = self.fonts["notification"].get_linesize()
        label_font = self.fonts["notification_label"]
        value_font = self.fonts["notification"]
        label_color = VisConfig.LIGHTG
        value_color = VisConfig.WHITE
        prev_color = VisConfig.GRAY
        time_color = (180, 180, 100)

        current_step = stats.get("global_step", 0)

        def render_line(
            y_pos, label, current_val, prev_val, best_step, val_format, tooltip_key
        ):
            if not label_font or not value_font:
                return

            # Label
            label_surf = label_font.render(label, True, label_color)
            label_rect = label_surf.get_rect(topleft=(area_rect.left + padding, y_pos))
            self.screen.blit(label_surf, label_rect)
            current_x = label_rect.right + 4

            # Current Value
            current_val_str = (
                val_format.format(current_val)
                if current_val > -float("inf") and current_val < float("inf")
                else "N/A"
            )
            val_surf = value_font.render(current_val_str, True, value_color)
            val_rect = val_surf.get_rect(topleft=(current_x, y_pos))
            self.screen.blit(val_surf, val_rect)
            current_x = val_rect.right + 4

            # Previous Value
            prev_val_str = (
                f"({val_format.format(prev_val)})"
                if prev_val > -float("inf") and prev_val < float("inf")
                else "(N/A)"
            )
            prev_surf = label_font.render(prev_val_str, True, prev_color)
            prev_rect = prev_surf.get_rect(topleft=(current_x, y_pos + 1))
            self.screen.blit(prev_surf, prev_rect)
            current_x = prev_rect.right + 6

            # Steps Ago (using updated formatting)
            steps_ago_str = self._format_steps_ago(current_step, best_step)
            time_surf = label_font.render(steps_ago_str, True, time_color)
            time_rect = time_surf.get_rect(topleft=(current_x, y_pos + 1))
            # Clip rendering to the notification area width
            available_width = area_rect.right - time_rect.left - padding
            if time_rect.width > available_width:
                time_rect.width = available_width  # Truncate if too long
                # Optionally add "..." if truncated? For now, just clip.
            self.screen.blit(time_surf, time_rect, area=time_rect)  # Use area clipping

            combined_line_rect = (
                label_rect.union(val_rect).union(prev_rect).union(time_rect)
            )
            self.stat_rects[tooltip_key] = combined_line_rect.clip(area_rect)

        y = area_rect.top + padding
        render_line(
            y,
            "RL Score:",
            stats.get("best_score", -float("inf")),
            stats.get("previous_best_score", -float("inf")),
            stats.get("best_score_step", 0),
            "{:.2f}",
            "Best RL Score Info",
        )
        y += line_height
        render_line(
            y,
            "Game Score:",
            stats.get("best_game_score", -float("inf")),
            stats.get("previous_best_game_score", -float("inf")),
            stats.get("best_game_score_step", 0),
            "{:.0f}",
            "Best Game Score Info",
        )
        y += line_height
        render_line(
            y,
            "Loss:",
            stats.get("best_loss", float("inf")),
            stats.get("previous_best_loss", float("inf")),
            stats.get("best_loss_step", 0),
            "{:.4f}",
            "Best Loss Info",
        )

    def _render_info_text_reduced(
        self, y_start: int, stats: Dict[str, Any], capacity: int, panel_width: int
    ) -> int:
        """Renders the main block of statistics text."""
        line_height = self.fonts["ui"].get_linesize()
        buffer_size = stats.get("buffer_size", 0)
        buffer_perc = (buffer_size / max(1, capacity) * 100) if capacity > 0 else 0.0
        global_step = stats.get("global_step", 0)

        info_lines = [
            (
                "Global Steps",
                f"{global_step/1e6:.2f}M / {TOTAL_TRAINING_STEPS/1e6:.1f}M",
            ),
            ("Total Episodes", f"{stats.get('total_episodes', 0)}"),
            ("Steps/Sec (Current)", f"{stats.get('steps_per_second', 0.0):.1f}"),
            ("Buffer Fill", f"{buffer_perc:.1f}% ({buffer_size/1e6:.2f}M)"),
            (
                "PER Beta",
                f"{stats.get('beta', 0.0):.3f}" if BufferConfig.USE_PER else "N/A",
            ),
            ("Learning Rate", f"{stats.get('current_lr', 0.0):.1e}"),
            ("Device", f"{DEVICE.type.upper()}"),
            (
                "Network",
                f"Duel={DQNConfig.USE_DUELING}, Noisy={DQNConfig.USE_NOISY_NETS}, C51={DQNConfig.USE_DISTRIBUTIONAL}",
            ),
        ]

        last_y = y_start
        current_stat_rects = {}
        for idx, (key, value_str) in enumerate(info_lines):
            try:
                key_surf = self.fonts["ui"].render(f"{key}:", True, VisConfig.LIGHTG)
                key_rect = key_surf.get_rect(topleft=(10, y_start + idx * line_height))
                value_surf = self.fonts["ui"].render(
                    f"{value_str}", True, VisConfig.WHITE
                )
                value_rect = value_surf.get_rect(
                    topleft=(key_rect.right + 5, key_rect.top)
                )
                combined_rect = key_rect.union(value_rect)
                combined_rect.width = min(combined_rect.width, panel_width - 15)
                self.screen.blit(key_surf, key_rect)
                self.screen.blit(value_surf, value_rect)
                current_stat_rects[key] = combined_rect
                last_y = combined_rect.bottom
            except Exception as e:
                print(f"Error rendering stat line '{key}': {e}")
                last_y += line_height

        self.stat_rects.update(current_stat_rects)
        return last_y

    def _render_tb_status(self, y_start: int, log_dir: Optional[str]) -> int:
        """Renders the TensorBoard status line and log directory."""
        tb_active = TensorBoardConfig.LOG_HISTOGRAMS or TensorBoardConfig.LOG_IMAGES
        tb_color = VisConfig.GOOGLE_COLORS[0] if tb_active else VisConfig.GRAY
        tb_text = f"TensorBoard: {'Logging Active' if tb_active else 'Logging Minimal'}"

        tb_surf = self.fonts["ui"].render(tb_text, True, tb_color)
        tb_rect = tb_surf.get_rect(topleft=(10, y_start))
        self.screen.blit(tb_surf, tb_rect)

        last_y = tb_rect.bottom
        combined_tb_rect = tb_rect

        if log_dir:
            try:
                rel_log_dir = os.path.relpath(log_dir)
                panel_char_width = self.vis_config.LEFT_PANEL_WIDTH // 8
                if len(rel_log_dir) > panel_char_width:
                    parts = log_dir.replace("\\", "/").split("/")
                    if len(parts) >= 2:
                        rel_log_dir = os.path.join("...", *parts[-2:])
                    else:
                        rel_log_dir = os.path.basename(log_dir)
            except ValueError:
                rel_log_dir = os.path.basename(log_dir)

            dir_surf = self.fonts["logdir"].render(
                f"Log Dir: {rel_log_dir}", True, VisConfig.LIGHTG
            )
            dir_rect = dir_surf.get_rect(topleft=(10, tb_rect.bottom + 2))
            self.screen.blit(dir_surf, dir_rect)
            combined_tb_rect = tb_rect.union(dir_rect)
            last_y = combined_tb_rect.bottom

        self.stat_rects["TensorBoard Status"] = combined_tb_rect
        return last_y

    def _render_plot_area(
        self,
        y_start: int,
        panel_width: int,
        screen_height: int,
        plot_data: Dict[str, Deque],
        status: str,
    ):
        """Renders the Matplotlib plot area."""
        plot_area_height = screen_height - y_start - 10
        plot_area_width = panel_width - 20

        if plot_area_width <= 50 or plot_area_height <= 50:
            return

        plot_surface = self.plotter.get_cached_or_updated_plot(
            plot_data, plot_area_width, plot_area_height
        )
        plot_area_rect = pygame.Rect(10, y_start, plot_area_width, plot_area_height)

        if plot_surface:
            self.screen.blit(plot_surface, plot_area_rect.topleft)
        else:
            pygame.draw.rect(self.screen, (40, 40, 40), plot_area_rect, 1)
            placeholder_text = "Waiting for data..."
            if status == "Buffering":
                placeholder_text = "Buffering... Waiting for data..."
            elif status == "Error":
                placeholder_text = "Plotting disabled due to error."

            placeholder_surf = self.fonts["plot_placeholder"].render(
                placeholder_text, True, VisConfig.GRAY
            )
            placeholder_rect = placeholder_surf.get_rect(center=plot_area_rect.center)
            clipped_placeholder_rect = placeholder_rect.clip(plot_area_rect)
            blit_pos = (clipped_placeholder_rect.left, clipped_placeholder_rect.top)
            blit_area = clipped_placeholder_rect.move(
                -placeholder_rect.left, -placeholder_rect.top
            )
            self.screen.blit(placeholder_surf, blit_pos, area=blit_area)

    def get_stat_rects(self) -> Dict[str, pygame.Rect]:
        """Returns the dictionary of rects for tooltip handling."""
        return self.stat_rects.copy()

    def get_tooltip_texts(self) -> Dict[str, str]:
        """Returns the dictionary of tooltip texts."""
        return TOOLTIP_TEXTS
