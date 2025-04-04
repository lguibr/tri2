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

# Tooltips specific to this panel (Keep definition here for locality)
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
    "Best RL Score Info": "Best RL Score achieved: Current Value (Previous Value) - Steps Ago",
    "Best Game Score Info": "Best Game Score achieved: Current Value (Previous Value) - Steps Ago",
    "Best Loss Info": "Best (Lowest) Loss achieved: Current Value (Previous Value) - Steps Ago",
    "Play Demo Button": "Click to enter interactive play mode. Gameplay fills the replay buffer.",
}


class LeftPanelRenderer:
    def __init__(self, screen: pygame.Surface, vis_config: VisConfig, plotter: Plotter):
        self.screen = screen
        self.vis_config = vis_config
        self.plotter = plotter
        self.fonts = self._init_fonts()
        self.stat_rects: Dict[str, pygame.Rect] = {}  # Rects for tooltip hit detection

    def _init_fonts(self):
        """Loads necessary fonts, attempting to load DejaVuSans for notifications."""
        fonts = {}
        default_font_size = 24
        status_font_size = 28
        logdir_font_size = 16
        plot_placeholder_size = 20
        notification_size = 19
        notification_label_size = 16

        # Try loading DejaVuSans for better unicode/symbol support if needed later
        # For now, SysFont is generally sufficient. Keep fallback logic.
        notification_font_path = os.path.join("fonts", "DejaVuSans.ttf")
        try:
            # Attempt loading specific font first
            fonts["notification"] = pygame.font.Font(
                notification_font_path, notification_size
            )
            # print(f"Loaded notification font: {notification_font_path}") # Optional Log
        except (pygame.error, FileNotFoundError) as e:
            # print(f"Warning: Could not load '{notification_font_path}': {e}. Falling back...") # Optional Log
            try:
                fonts["notification"] = pygame.font.SysFont(None, notification_size)
            except Exception:
                # print("Warning: SysFont fallback failed. Using default font.") # Optional Log
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
            # Fallback for notification_label if SysFont failed
            fonts["notification_label"] = fonts.get(
                "notification", pygame.font.Font(None, notification_label_size)
            )

        # Final check for missing fonts
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

    def render(
        self,
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        tensorboard_log_dir: Optional[str],
        plot_data: Dict[str, Deque],
        app_state: str,
    ):
        """Renders the entire left panel by calling sub-render methods."""
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(300, self.vis_config.LEFT_PANEL_WIDTH))
        lp_rect = pygame.Rect(0, 0, lp_width, current_height)

        # Background based on status
        status_color_map = {
            "Paused": (30, 30, 30),
            "Buffering": (30, 40, 30),
            "Training": (40, 30, 30),
            "Confirm Cleanup": (50, 20, 20),
            "Cleaning": (60, 30, 30),
            "Error": (60, 0, 0),
            "Playing Demo": (30, 30, 40),
            "Initializing": (40, 40, 40),
        }
        bg_color = status_color_map.get(status, (30, 30, 30))
        pygame.draw.rect(self.screen, bg_color, lp_rect)

        self.stat_rects.clear()  # Reset rects for tooltips each frame

        # --- Render sections sequentially ---
        current_y = 10  # Starting Y position

        # 1. Buttons and Status (Layout depends on app_state)
        notification_area_rect, current_y = self._render_buttons_and_status(
            current_y, lp_width, app_state, is_training, status
        )

        # 2. Notification Area (if applicable)
        if notification_area_rect:
            self._render_notification_area(notification_area_rect, stats_summary)
            # Adjust current_y to be below the notifications if they were rendered
            current_y = max(current_y, notification_area_rect.bottom + 10)

        # 3. Info Text Block
        current_y = self._render_info_text(
            current_y, stats_summary, buffer_capacity, lp_width
        )

        # 4. TensorBoard Status
        current_y = self._render_tb_status(
            current_y + 10, tensorboard_log_dir, lp_width
        )

        # 5. Plot Area
        self._render_plot_area(
            current_y + 15, lp_width, current_height, plot_data, status
        )

    def _render_buttons_and_status(
        self,
        y_start: int,
        panel_width: int,
        app_state: str,
        is_training: bool,
        status: str,
    ) -> Tuple[Optional[pygame.Rect], int]:
        """Renders the top buttons (if in MainMenu) and the status line.
        Returns the Rect for the notification area (or None) and the next Y coordinate.
        """
        notification_rect = None
        next_y = y_start

        if app_state == "MainMenu":
            # Define button rects
            train_btn_rect = pygame.Rect(10, y_start, 100, 40)
            cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, y_start, 160, 40)
            demo_btn_rect = pygame.Rect(cleanup_btn_rect.right + 10, y_start, 120, 40)

            # Draw buttons
            self._draw_button(
                train_btn_rect,
                "Pause" if is_training and status == "Training" else "Train",
                (70, 70, 70),
            )
            self._draw_button(cleanup_btn_rect, "Cleanup This Run", (100, 40, 40))
            self._draw_button(demo_btn_rect, "Play Demo", (40, 100, 40))

            # Register button rects for tooltips
            self.stat_rects["Train Button"] = train_btn_rect
            self.stat_rects["Cleanup Button"] = cleanup_btn_rect
            self.stat_rects["Play Demo Button"] = demo_btn_rect

            # Calculate notification area position (to the right of buttons)
            notification_x = demo_btn_rect.right + 15
            notification_w = panel_width - notification_x - 10
            if notification_w > 50:
                line_h = self.fonts["notification"].get_linesize()
                notification_h = line_h * 3 + 12  # Height for 3 lines of notifications
                notification_rect = pygame.Rect(
                    notification_x, y_start, notification_w, notification_h
                )

            next_y = train_btn_rect.bottom + 10  # Next element starts below buttons

        # Render Status Text (Always)
        status_text = f"Status: {status}"
        if app_state == "Playing":
            status_text = "Status: Playing Demo"
        elif app_state != "MainMenu":
            status_text = f"Status: {app_state}"  # e.g., Initializing

        status_surf = self.fonts["status"].render(status_text, True, VisConfig.YELLOW)
        # Position status text below buttons if they exist, otherwise at the top
        status_rect_top = next_y if app_state == "MainMenu" else y_start
        status_rect = status_surf.get_rect(topleft=(10, status_rect_top))
        self.screen.blit(status_surf, status_rect)
        if app_state == "MainMenu":  # Only add tooltip for status in main menu
            self.stat_rects["Status"] = status_rect

        # Determine the starting Y for the *next* block (below status and potentially buttons)
        final_next_y = status_rect.bottom + 5

        return notification_rect, final_next_y

    def _draw_button(self, rect: pygame.Rect, text: str, color: Tuple[int, int, int]):
        """Helper to draw a single button."""
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        lbl_surf = self.fonts["ui"].render(text, True, VisConfig.WHITE)
        self.screen.blit(lbl_surf, lbl_surf.get_rect(center=rect.center))

    def _format_steps_ago(self, current_step: int, best_step: int) -> str:
        """Formats the difference in steps into a readable string (k steps, M steps)."""
        if best_step <= 0 or current_step <= best_step:
            return "Now"
        diff = current_step - best_step
        if diff < 1000:
            return f"{diff} steps ago"
        elif diff < 1_000_000:
            return f"{diff / 1000:.1f}k steps ago"
        else:
            return f"{diff / 1_000_000:.1f}M steps ago"

    def _render_notification_area(self, area_rect: pygame.Rect, stats: Dict[str, Any]):
        """Renders the latest best score/loss messages with details."""
        pygame.draw.rect(self.screen, (45, 45, 45), area_rect, border_radius=3)
        pygame.draw.rect(self.screen, VisConfig.LIGHTG, area_rect, 1, border_radius=3)
        self.stat_rects["Notification Area"] = area_rect  # Tooltip for the whole area

        padding = 5
        line_height = self.fonts["notification"].get_linesize()
        label_font = self.fonts["notification_label"]
        value_font = self.fonts["notification"]
        label_color = VisConfig.LIGHTG
        value_color = VisConfig.WHITE
        prev_color = VisConfig.GRAY
        time_color = (180, 180, 100)  # Yellowish for time

        current_step = stats.get("global_step", 0)

        # --- Helper function to render a single notification line ---
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

            # Current Value (handle inf/-inf)
            current_val_str = "N/A"
            if isinstance(current_val, (int, float)) and abs(current_val) != float(
                "inf"
            ):
                current_val_str = val_format.format(current_val)
            val_surf = value_font.render(current_val_str, True, value_color)
            val_rect = val_surf.get_rect(topleft=(current_x, y_pos))
            self.screen.blit(val_surf, val_rect)
            current_x = val_rect.right + 4

            # Previous Value (handle inf/-inf)
            prev_val_str = "(N/A)"
            if isinstance(prev_val, (int, float)) and abs(prev_val) != float("inf"):
                prev_val_str = f"({val_format.format(prev_val)})"
            prev_surf = label_font.render(prev_val_str, True, prev_color)
            prev_rect = prev_surf.get_rect(
                topleft=(current_x, y_pos + 1)
            )  # Slightly offset
            self.screen.blit(prev_surf, prev_rect)
            current_x = prev_rect.right + 6

            # Steps Ago
            steps_ago_str = self._format_steps_ago(current_step, best_step)
            time_surf = label_font.render(steps_ago_str, True, time_color)
            time_rect = time_surf.get_rect(
                topleft=(current_x, y_pos + 1)
            )  # Slightly offset

            # Clip rendering to the notification area width
            available_width = area_rect.right - time_rect.left - padding
            clip_rect = pygame.Rect(0, 0, max(0, available_width), time_rect.height)
            if time_rect.width > available_width > 0:
                self.screen.blit(time_surf, time_rect, area=clip_rect)
            elif available_width > 0:  # Only blit if space available
                self.screen.blit(time_surf, time_rect)

            # Create combined rect for tooltip hover detection (clipped to area)
            combined_rect = label_rect.union(val_rect).union(prev_rect).union(time_rect)
            self.stat_rects[tooltip_key] = combined_rect.clip(area_rect)

        # --- End render_line helper ---

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

    def _render_info_text(
        self, y_start: int, stats: Dict[str, Any], capacity: int, panel_width: int
    ) -> int:
        """Renders the main block of statistics text."""
        line_height = self.fonts["ui"].get_linesize()
        buffer_size = stats.get("buffer_size", 0)
        buffer_perc = (buffer_size / max(1, capacity) * 100) if capacity > 0 else 0.0
        global_step = stats.get("global_step", 0)

        # Define lines to render
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
        x_pos_key = 10
        x_pos_val_offset = 5  # Space between key and value

        for idx, (key, value_str) in enumerate(info_lines):
            current_y = y_start + idx * line_height
            try:
                # Render Key
                key_surf = self.fonts["ui"].render(f"{key}:", True, VisConfig.LIGHTG)
                key_rect = key_surf.get_rect(topleft=(x_pos_key, current_y))
                self.screen.blit(key_surf, key_rect)

                # Render Value
                value_surf = self.fonts["ui"].render(
                    f"{value_str}", True, VisConfig.WHITE
                )
                value_rect = value_surf.get_rect(
                    topleft=(key_rect.right + x_pos_val_offset, current_y)
                )

                # Clip value rendering if it exceeds panel width
                clip_width = max(
                    0, panel_width - value_rect.left - 10
                )  # 10px right margin
                if value_rect.width > clip_width:
                    self.screen.blit(
                        value_surf,
                        value_rect,
                        area=pygame.Rect(0, 0, clip_width, value_rect.height),
                    )
                else:
                    self.screen.blit(value_surf, value_rect)

                # Register combined rect for tooltip
                combined_rect = key_rect.union(value_rect)
                combined_rect.width = min(
                    combined_rect.width, panel_width - x_pos_key - 10
                )  # Clip tooltip rect too
                self.stat_rects[key] = combined_rect
                last_y = combined_rect.bottom

            except Exception as e:
                print(f"Error rendering stat line '{key}': {e}")
                last_y = current_y + line_height  # Advance Y even on error

        return last_y  # Return Y position below the last rendered item

    def _render_tb_status(
        self, y_start: int, log_dir: Optional[str], panel_width: int
    ) -> int:
        """Renders the TensorBoard status line and log directory."""
        tb_active = (
            TensorBoardConfig.LOG_HISTOGRAMS
            or TensorBoardConfig.LOG_IMAGES
            or TensorBoardConfig.LOG_SHAPE_PLACEMENT_Q_VALUES
        )
        tb_color = VisConfig.GOOGLE_COLORS[0] if tb_active else VisConfig.GRAY
        tb_text = f"TensorBoard: {'Logging Active' if tb_active else 'Logging Minimal'}"

        tb_surf = self.fonts["ui"].render(tb_text, True, tb_color)
        tb_rect = tb_surf.get_rect(topleft=(10, y_start))
        self.screen.blit(tb_surf, tb_rect)
        self.stat_rects["TensorBoard Status"] = tb_rect  # Initial rect for tooltip

        last_y = tb_rect.bottom

        if log_dir:
            try:
                # Attempt to shorten log directory path for display
                panel_char_width = max(
                    10, panel_width // self.fonts["logdir"].size("A")[0]
                )
                try:
                    rel_log_dir = os.path.relpath(log_dir)
                except ValueError:  # Handle different drives on Windows
                    rel_log_dir = log_dir

                if len(rel_log_dir) > panel_char_width:
                    parts = log_dir.replace("\\", "/").split("/")
                    if len(parts) >= 2:
                        rel_log_dir = os.path.join("...", *parts[-2:])
                        if len(rel_log_dir) > panel_char_width:  # If still too long
                            rel_log_dir = (
                                "..."
                                + os.path.basename(log_dir)[-(panel_char_width - 3) :]
                            )
                    else:  # Just the run ID folder
                        rel_log_dir = (
                            "..." + os.path.basename(log_dir)[-(panel_char_width - 3) :]
                        )

            except Exception:  # Fallback on any path manipulation error
                rel_log_dir = os.path.basename(log_dir)

            dir_surf = self.fonts["logdir"].render(
                f"Log Dir: {rel_log_dir}", True, VisConfig.LIGHTG
            )
            dir_rect = dir_surf.get_rect(topleft=(10, tb_rect.bottom + 2))

            # Clip rendering if needed
            clip_width = max(0, panel_width - dir_rect.left - 10)
            if dir_rect.width > clip_width:
                self.screen.blit(
                    dir_surf,
                    dir_rect,
                    area=pygame.Rect(0, 0, clip_width, dir_rect.height),
                )
            else:
                self.screen.blit(dir_surf, dir_rect)

            # Update tooltip rect to include the log dir line
            combined_tb_rect = tb_rect.union(dir_rect)
            combined_tb_rect.width = min(
                combined_tb_rect.width, panel_width - 10 - combined_tb_rect.left
            )
            self.stat_rects["TensorBoard Status"] = combined_tb_rect
            last_y = dir_rect.bottom

        return last_y

    def _render_plot_area(
        self,
        y_start: int,
        panel_width: int,
        screen_height: int,
        plot_data: Dict[str, Deque],
        status: str,
    ):
        """Renders the Matplotlib plot area using the Plotter."""
        plot_area_height = screen_height - y_start - 10  # Bottom margin
        plot_area_width = panel_width - 20  # Left/right margins

        if plot_area_width <= 50 or plot_area_height <= 50:
            # Area too small to render plot, maybe show a placeholder
            return

        # Get the plot surface (cached or newly generated)
        plot_surface = self.plotter.get_cached_or_updated_plot(
            plot_data, plot_area_width, plot_area_height
        )
        plot_area_rect = pygame.Rect(10, y_start, plot_area_width, plot_area_height)

        if plot_surface:
            self.screen.blit(plot_surface, plot_area_rect.topleft)
        else:
            # Render placeholder if plot couldn't be generated
            pygame.draw.rect(self.screen, (40, 40, 40), plot_area_rect, 1)
            placeholder_text = "Waiting for data..."
            if status == "Buffering":
                placeholder_text = "Buffering... Waiting for plot data..."
            elif status == "Error":
                placeholder_text = "Plotting disabled due to error."
            elif not plot_data or not any(plot_data.values()):
                placeholder_text = "No plot data yet..."

            placeholder_surf = self.fonts["plot_placeholder"].render(
                placeholder_text, True, VisConfig.GRAY
            )
            placeholder_rect = placeholder_surf.get_rect(center=plot_area_rect.center)

            # Clip placeholder text rendering to fit within the plot area
            blit_pos = (
                max(plot_area_rect.left, placeholder_rect.left),
                max(plot_area_rect.top, placeholder_rect.top),
            )
            clip_area_rect = plot_area_rect.clip(placeholder_rect)
            blit_area = clip_area_rect.move(
                -placeholder_rect.left, -placeholder_rect.top
            )

            if blit_area.width > 0 and blit_area.height > 0:
                self.screen.blit(placeholder_surf, blit_pos, area=blit_area)

    def get_stat_rects(self) -> Dict[str, pygame.Rect]:
        """Returns the dictionary of rects for tooltip handling."""
        return self.stat_rects.copy()

    def get_tooltip_texts(self) -> Dict[str, str]:
        """Returns the dictionary of tooltip texts."""
        return TOOLTIP_TEXTS
