# File: ui/renderer.py
import pygame
import math
import time
import os  # Import os for path manipulations
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

from config import (
    VisConfig,
    EnvConfig,
    ModelConfig,
    DQNConfig,
    DEVICE,
    BufferConfig,
    StatsConfig,
    TrainConfig,
    # WandbConfig removed
)
from environment.game_state import GameState
from environment.shape import Shape
from environment.triangle import Triangle


# Updated Tooltips (Removed WandB, Added TensorBoard)
TOOLTIP_TEXTS = {
    "Status": "Current state: Paused, Buffering (collecting initial data), Training, Confirm Cleanup, or Error.",
    "Global Steps": "Total environment steps across all parallel environments.",
    "Total Episodes": "Total completed episodes across all environments.",
    "Steps/Sec": f"Average global steps processed per second (rolling average over ~{StatsConfig.STATS_AVG_WINDOW} logs).",
    "Avg RL Score": f"Average RL reward sum per episode (last {StatsConfig.STATS_AVG_WINDOW} eps).",
    "Best RL Score": "Highest RL reward sum in a single episode this run.",
    "Avg Game Score": f"Average game score per episode (last {StatsConfig.STATS_AVG_WINDOW} eps).",
    "Best Game Score": "Highest game score in a single episode this run.",
    "Avg Length": f"Average steps per episode (last {StatsConfig.STATS_AVG_WINDOW} eps).",
    "Avg Lines Clr": f"Average lines cleared per episode (last {StatsConfig.STATS_AVG_WINDOW} eps).",
    "Avg Loss": f"Average DQN loss (last {StatsConfig.STATS_AVG_WINDOW} training steps).",
    "Avg Max Q": f"Average max predicted Q-value (last {StatsConfig.STATS_AVG_WINDOW} training batches).",
    "PER Beta": f"PER Importance Sampling exponent (anneals {BufferConfig.PER_BETA_START:.1f} -> 1.0). {BufferConfig.PER_BETA_FRAMES/1e6:.1f}M steps.",
    "Buffer": f"Replay buffer fill status ({BufferConfig.REPLAY_BUFFER_SIZE / 1e6:.1f}M capacity).",
    "Train Button": "Click to Start/Pause the training process (or press 'P').",
    "Cleanup Button": "Click to delete saved agent & buffer for the CURRENT run, then restart components.",
    "Device": f"Computation device ({DEVICE.type.upper()}).",
    "Network": f"Agent Network Architecture (CNN+MLP Fusion). Noisy={DQNConfig.USE_NOISY_NETS}, Dueling={DQNConfig.USE_DUELING}",
    "TensorBoard Status": "Status of TensorBoard logging. Log directory path is shown.",
}


class UIRenderer:
    """Handles rendering the Pygame UI, including stats and game environments."""

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config

        # Fonts
        try:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_env_score = pygame.font.SysFont(None, 18)
            self.font_env_overlay = pygame.font.SysFont(None, 36)
            self.font_tooltip = pygame.font.SysFont(None, 18)
            self.font_status = pygame.font.SysFont(None, 28)
            self.font_logdir = pygame.font.SysFont(
                None, 16
            )  # Smaller font for log dir path
        except Exception as e:
            print(f"Warning: Error initializing SysFont: {e}. Using default font.")
            self.font_ui = pygame.font.Font(None, 24)
            self.font_env_score = pygame.font.Font(None, 18)
            self.font_env_overlay = pygame.font.Font(None, 36)
            self.font_tooltip = pygame.font.Font(None, 18)
            self.font_status = pygame.font.Font(None, 28)
            self.font_logdir = pygame.font.Font(None, 16)

        self.stat_rects: Dict[str, pygame.Rect] = {}
        self.hovered_stat_key: Optional[str] = None
        # self.wandb_link_rect removed

    def _render_left_panel(
        self,
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        tensorboard_log_dir: Optional[str],  # Path to TensorBoard log dir
    ):
        """Renders the left information panel."""
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(250, self.vis_config.LEFT_PANEL_WIDTH))
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

        # Buttons
        train_btn_rect = pygame.Rect(10, 10, 100, 40)
        pygame.draw.rect(self.screen, (70, 70, 70), train_btn_rect, border_radius=5)
        btn_text = "Pause" if is_training else "Train"
        lbl_surf = self.font_ui.render(btn_text, True, VisConfig.WHITE)
        self.screen.blit(lbl_surf, lbl_surf.get_rect(center=train_btn_rect.center))

        cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 160, 40)  # Wider
        pygame.draw.rect(self.screen, (100, 40, 40), cleanup_btn_rect, border_radius=5)
        cleanup_lbl_surf = self.font_ui.render(
            "Cleanup This Run", True, VisConfig.WHITE
        )  # Updated text
        self.screen.blit(
            cleanup_lbl_surf, cleanup_lbl_surf.get_rect(center=cleanup_btn_rect.center)
        )

        # Status Text
        status_surf = self.font_status.render(
            f"Status: {status}", True, VisConfig.YELLOW
        )
        status_rect = status_surf.get_rect(topleft=(10, train_btn_rect.bottom + 10))
        self.screen.blit(status_surf, status_rect)

        # Info Text & Tooltips
        self.stat_rects.clear()
        self.stat_rects["Train Button"] = train_btn_rect
        self.stat_rects["Cleanup Button"] = cleanup_btn_rect
        self.stat_rects["Status"] = status_rect
        # self.wandb_link_rect removed

        buffer_size = stats_summary.get("buffer_size", 0)
        buffer_perc = (
            (buffer_size / buffer_capacity * 100) if buffer_capacity > 0 else 0.0
        )

        info_lines_data = [
            (
                "Global Steps",
                f"{stats_summary.get('global_step', 0)/1e6:.2f}M / {TrainConfig.TOTAL_TRAINING_STEPS/1e6:.1f}M",
            ),
            ("Total Episodes", f"{stats_summary.get('total_episodes', 0)}"),
            ("Steps/Sec", f"{stats_summary.get('steps_per_second', 0.0):.1f}"),
            (
                "Avg RL Score",
                f"({stats_summary.get('num_ep_scores', 0)}): {stats_summary.get('avg_score_100', 0.0):.2f}",
            ),
            ("Best RL Score", f"{stats_summary.get('best_score', 0.0):.2f}"),
            (
                "Avg Game Score",
                f"({stats_summary.get('num_game_scores', 0)}): {stats_summary.get('avg_game_score_100', 0.0):.1f}",
            ),
            ("Best Game Score", f"{stats_summary.get('best_game_score', 0.0):.1f}"),
            (
                "Avg Length",
                f"({stats_summary.get('num_ep_lengths', 0)}): {stats_summary.get('avg_length_100', 0.0):.1f}",
            ),
            (
                "Avg Lines Clr",
                f"({stats_summary.get('num_lines_cleared', 0)}): {stats_summary.get('avg_lines_cleared_100', 0.0):.2f}",
            ),
            (
                "Avg Loss",
                f"({stats_summary.get('num_losses', 0)}): {stats_summary.get('avg_loss_100', 0.0):.4f}",
            ),
            (
                "Avg Max Q",
                f"({stats_summary.get('num_avg_max_qs', 0)}): {stats_summary.get('avg_max_q_100', 0.0):.3f}",
            ),
            (
                "PER Beta",
                (
                    f"{stats_summary.get('beta', 0.0):.3f}"
                    if BufferConfig.USE_PER
                    else "N/A"
                ),
            ),
            (
                "Buffer",
                f"{buffer_size/1e6:.2f}M / {buffer_capacity/1e6:.1f}M ({buffer_perc:.1f}%)",
            ),
            ("Device", f"{DEVICE.type.upper()}"),
            ("Network", f"CNN+MLP Fusion"),
        ]

        text_y_start = status_rect.bottom + 10
        line_height = self.font_ui.get_linesize()

        for idx, (key, value_str) in enumerate(info_lines_data):
            line_text = f"{key}: {value_str}"
            line_surf = self.font_ui.render(line_text, True, VisConfig.WHITE)
            line_rect = line_surf.get_rect(
                topleft=(10, text_y_start + idx * line_height)
            )
            line_rect.width = min(line_rect.width, lp_width - 20)  # Limit rect width
            self.screen.blit(line_surf, line_rect)
            self.stat_rects[key] = line_rect

        # --- TensorBoard Status ---
        tb_y_start = text_y_start + len(info_lines_data) * line_height + 10
        tb_status_text = "TensorBoard: Logging Active"
        tb_status_color = VisConfig.GOOGLE_COLORS[0]  # Green

        tb_surf = self.font_ui.render(tb_status_text, True, tb_status_color)
        tb_rect = tb_surf.get_rect(topleft=(10, tb_y_start))
        self.screen.blit(tb_surf, tb_rect)
        self.stat_rects["TensorBoard Status"] = tb_rect  # Add tooltip hover

        # Display Log Directory Path (make it shorter)
        if tensorboard_log_dir:
            try:
                # Show relative path if possible, or just the last couple of dirs
                rel_log_dir = os.path.relpath(tensorboard_log_dir)
                if len(rel_log_dir) > 50:  # Heuristic for too long path
                    parts = tensorboard_log_dir.split(os.sep)[-3:]  # Show last 3 parts
                    rel_log_dir = os.path.join("...", *parts)
            except ValueError:  # Happens if on different drives (Windows)
                rel_log_dir = tensorboard_log_dir  # Show full path

            dir_surf = self.font_logdir.render(
                f"Log Dir: {rel_log_dir}", True, VisConfig.LIGHTG
            )
            dir_rect = dir_surf.get_rect(topleft=(10, tb_rect.bottom + 2))
            self.screen.blit(dir_surf, dir_rect)
            # Make the directory text area also trigger the tooltip
            combined_tb_rect = tb_rect.union(dir_rect)
            self.stat_rects["TensorBoard Status"] = combined_tb_rect

    def _render_shape_preview(self, surf: pygame.Surface, shape: Shape, cell_size: int):
        if not shape or not shape.triangles:
            return
        min_r, min_c, max_r, max_c = shape.bbox()
        shape_h_cells = max_r - min_r + 1
        shape_w_cells = max_c - min_c + 1
        total_w_pixels = shape_w_cells * (cell_size * 0.75) + (cell_size * 0.25)
        total_h_pixels = shape_h_cells * cell_size
        offset_x = (surf.get_width() - total_w_pixels) / 2 - min_c * (cell_size * 0.75)
        offset_y = (surf.get_height() - total_h_pixels) / 2 - min_r * cell_size
        for dr, dc, up in shape.triangles:
            tri = Triangle(row=dr, col=dc, is_up=up)
            pts = tri.get_points(ox=offset_x, oy=offset_y, cw=cell_size, ch=cell_size)
            pygame.draw.polygon(surf, shape.color, pts)

    def _render_env(
        self, surf: pygame.Surface, env: GameState, cell_w: int, cell_h: int
    ):
        try:
            bg_color = (
                VisConfig.YELLOW
                if env.is_blinking()
                else (
                    (30, 30, 100)
                    if env.is_frozen() and not env.is_over()
                    else (20, 20, 20)
                )
            )
            surf.fill(bg_color)

            # Render Grid Triangles
            if hasattr(env, "grid") and hasattr(env.grid, "triangles"):
                for r in range(env.grid.rows):
                    for c in range(env.grid.cols):
                        t = env.grid.triangles[r][c]
                        if not hasattr(t, "get_points"):
                            continue
                        try:
                            pts = t.get_points(ox=0, oy=0, cw=cell_w, ch=cell_h)
                            color = VisConfig.GRAY
                            if t.is_death:
                                color = VisConfig.BLACK
                            elif t.is_occupied:
                                color = t.color if t.color else VisConfig.RED
                            pygame.draw.polygon(surf, color, pts)
                        except Exception as e_render:
                            print(f"Error rendering tri ({r},{c}): {e_render}")
            else:
                pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
                err_txt = self.font_env_overlay.render(
                    "Invalid Grid", True, VisConfig.RED
                )
                surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))

            # Render Scores
            rl_score_val = env.score
            game_score_val = env.game_score
            score_surf = self.font_env_score.render(
                f"GS: {game_score_val} | R: {rl_score_val:.1f}",
                True,
                VisConfig.WHITE,
                (0, 0, 0, 180),
            )
            surf.blit(score_surf, (4, 4))

            # Render Overlays
            if env.is_over():
                overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                overlay.fill((100, 0, 0, 180))
                surf.blit(overlay, (0, 0))
                over_text = self.font_env_overlay.render(
                    "GAME OVER", True, VisConfig.WHITE
                )
                surf.blit(over_text, over_text.get_rect(center=surf.get_rect().center))
            elif env.is_frozen() and not env.is_blinking():
                freeze_text = self.font_env_overlay.render(
                    "Frozen", True, VisConfig.WHITE
                )
                surf.blit(
                    freeze_text,
                    freeze_text.get_rect(
                        center=(surf.get_width() // 2, surf.get_height() - 15)
                    ),
                )

        except AttributeError as e:
            pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
            err_txt = self.font_env_overlay.render(
                f"Attr Err: {e}", True, VisConfig.RED, VisConfig.BLACK
            )
            surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))
        except Exception as e:
            print(f"Unexpected Render Error in _render_env: {e}")
            pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
            traceback.print_exc()

    def _render_game_area(
        self, envs: List[GameState], num_envs: int, env_config: EnvConfig
    ):
        current_width, current_height = self.screen.get_size()
        lp_width = min(current_width, max(250, self.vis_config.LEFT_PANEL_WIDTH))
        ga_rect = pygame.Rect(lp_width, 0, current_width - lp_width, current_height)

        if num_envs <= 0 or ga_rect.width <= 0 or ga_rect.height <= 0:
            return

        render_limit = self.vis_config.NUM_ENVS_TO_RENDER
        num_envs_to_render = (
            num_envs if render_limit <= 0 else min(num_envs, render_limit)
        )
        if num_envs_to_render <= 0:
            return

        aspect_ratio = ga_rect.width / ga_rect.height
        cols_env = max(1, int(math.sqrt(num_envs_to_render * aspect_ratio)))
        rows_env = math.ceil(num_envs_to_render / cols_env)

        total_spacing_w = (cols_env + 1) * self.vis_config.ENV_SPACING
        total_spacing_h = (rows_env + 1) * self.vis_config.ENV_SPACING
        cell_w = (ga_rect.width - total_spacing_w) // cols_env if cols_env > 0 else 0
        cell_h = (ga_rect.height - total_spacing_h) // rows_env if rows_env > 0 else 0

        if cell_w > 10 and cell_h > 10:
            env_idx = 0
            for r in range(rows_env):
                for c in range(cols_env):
                    if env_idx >= num_envs_to_render:
                        break
                    env_x = (
                        ga_rect.x + self.vis_config.ENV_SPACING * (c + 1) + c * cell_w
                    )
                    env_y = (
                        ga_rect.y + self.vis_config.ENV_SPACING * (r + 1) + r * cell_h
                    )
                    env_rect = pygame.Rect(env_x, env_y, cell_w, cell_h)
                    try:
                        sub_surf = self.screen.subsurface(env_rect)
                        tri_cell_w = cell_w / (env_config.COLS * 0.75 + 0.25)
                        tri_cell_h = cell_h / env_config.ROWS
                        self._render_env(
                            sub_surf, envs[env_idx], int(tri_cell_w), int(tri_cell_h)
                        )
                        # Shape Previews
                        available_shapes = envs[env_idx].get_shapes()
                        if available_shapes:
                            preview_dim = max(10, min(cell_w // 6, cell_h // 6, 25))
                            preview_spacing = 4
                            total_preview_width = (
                                len(available_shapes) * preview_dim
                                + max(0, len(available_shapes) - 1) * preview_spacing
                            )
                            start_x = (
                                sub_surf.get_width()
                                - total_preview_width
                                - preview_spacing
                            )
                            start_y = preview_spacing
                            for i, shape in enumerate(available_shapes):
                                preview_x = start_x + i * (
                                    preview_dim + preview_spacing
                                )
                                temp_shape_surf = pygame.Surface(
                                    (preview_dim, preview_dim), pygame.SRCALPHA
                                )
                                temp_shape_surf.fill((0, 0, 0, 0))
                                preview_cell_size = max(2, preview_dim // 4)
                                self._render_shape_preview(
                                    temp_shape_surf, shape, preview_cell_size
                                )
                                sub_surf.blit(temp_shape_surf, (preview_x, start_y))
                    except ValueError:
                        pygame.draw.rect(
                            self.screen, (0, 0, 50), env_rect, 1
                        )  # Subsurface error
                    except Exception as e_render_env:
                        print(f"Error rendering env {env_idx}: {e_render_env}")
                        pygame.draw.rect(self.screen, (50, 0, 50), env_rect, 1)
                    env_idx += 1
        else:
            err_surf = self.font_ui.render(
                f"Envs Too Small ({cell_w}x{cell_h})", True, VisConfig.GRAY
            )
            self.screen.blit(err_surf, err_surf.get_rect(center=ga_rect.center))

        # Display text if not all envs are rendered
        if num_envs_to_render < num_envs:
            info_surf = self.font_ui.render(
                f"Rendering {num_envs_to_render}/{num_envs} Envs",
                True,
                VisConfig.YELLOW,
                VisConfig.BLACK,
            )
            self.screen.blit(
                info_surf,
                info_surf.get_rect(bottomright=(ga_rect.right - 5, ga_rect.bottom - 5)),
            )

    def _render_cleanup_confirmation(self):
        current_width, current_height = self.screen.get_size()
        overlay = pygame.Surface((current_width, current_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        center_x, center_y = current_width // 2, current_height // 2
        prompt_l1 = self.font_env_overlay.render(
            "DELETE CURRENT RUN DATA?", True, VisConfig.RED
        )
        self.screen.blit(
            prompt_l1, prompt_l1.get_rect(center=(center_x, center_y - 60))
        )
        prompt_l2 = self.font_ui.render(
            "(Agent Checkpoint & Buffer State)", True, VisConfig.WHITE
        )  # Simplified
        self.screen.blit(
            prompt_l2, prompt_l2.get_rect(center=(center_x, center_y - 25))
        )
        prompt_l3 = self.font_ui.render(
            "This action cannot be undone!", True, VisConfig.YELLOW
        )
        self.screen.blit(prompt_l3, prompt_l3.get_rect(center=(center_x, center_y)))
        confirm_yes_rect = pygame.Rect(center_x - 110, center_y + 30, 100, 40)
        confirm_no_rect = pygame.Rect(center_x + 10, center_y + 30, 100, 40)
        pygame.draw.rect(self.screen, (0, 150, 0), confirm_yes_rect, border_radius=5)
        pygame.draw.rect(self.screen, (150, 0, 0), confirm_no_rect, border_radius=5)
        yes_text = self.font_ui.render("YES", True, VisConfig.WHITE)
        no_text = self.font_ui.render("NO", True, VisConfig.WHITE)
        self.screen.blit(yes_text, yes_text.get_rect(center=confirm_yes_rect.center))
        self.screen.blit(no_text, no_text.get_rect(center=confirm_no_rect.center))

    def _render_status_message(self, message: str, last_message_time: float):
        if message and (time.time() - last_message_time < 5.0):
            current_width, current_height = self.screen.get_size()
            lines = message.split("\n")
            max_width = 0
            msg_surfs = []
            for line in lines:
                msg_surf = self.font_ui.render(
                    line, True, VisConfig.YELLOW, VisConfig.BLACK
                )
                msg_surfs.append(msg_surf)
                max_width = max(max_width, msg_surf.get_width())

            total_height = (
                sum(s.get_height() for s in msg_surfs) + max(0, len(lines) - 1) * 2
            )
            bg_rect = pygame.Rect(0, 0, max_width + 10, total_height + 10)
            bg_rect.midbottom = (current_width // 2, current_height - 10)
            pygame.draw.rect(self.screen, VisConfig.BLACK, bg_rect, border_radius=3)
            current_y = bg_rect.top + 5
            for msg_surf in msg_surfs:
                msg_rect = msg_surf.get_rect(midtop=(bg_rect.centerx, current_y))
                self.screen.blit(msg_surf, msg_rect)
                current_y += msg_surf.get_height() + 2
            return True
        return False

    def _render_tooltip(self):
        if self.hovered_stat_key and self.hovered_stat_key in TOOLTIP_TEXTS:
            tooltip_text = TOOLTIP_TEXTS[self.hovered_stat_key]
            mouse_pos = pygame.mouse.get_pos()
            lines = []
            max_width = 300
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
            lines.append(current_line)

            line_surfs = [
                self.font_tooltip.render(line, True, VisConfig.BLACK) for line in lines
            ]
            total_height = sum(s.get_height() for s in line_surfs)
            max_line_width = max(s.get_width() for s in line_surfs)

            padding = 5
            tooltip_rect = pygame.Rect(
                mouse_pos[0] + 15,
                mouse_pos[1] + 10,
                max_line_width + padding * 2,
                total_height + padding * 2,
            )
            tooltip_rect.clamp_ip(self.screen.get_rect())

            pygame.draw.rect(
                self.screen, VisConfig.YELLOW, tooltip_rect, border_radius=3
            )
            pygame.draw.rect(
                self.screen, VisConfig.BLACK, tooltip_rect, 1, border_radius=3
            )

            current_y = tooltip_rect.y + padding
            for surf in line_surfs:
                self.screen.blit(surf, (tooltip_rect.x + padding, current_y))
                current_y += surf.get_height()

    def check_hover(self, mouse_pos: Tuple[int, int]):
        self.hovered_stat_key = None
        # Iterate in reverse order so tooltips for elements drawn last appear first
        for key, rect in reversed(self.stat_rects.items()):
            if rect.collidepoint(mouse_pos):
                self.hovered_stat_key = key
                return  # Found one, stop checking

    def render_all(
        self,
        is_training: bool,
        status: str,
        stats_summary: Dict[str, Any],
        buffer_capacity: int,
        envs: List[GameState],
        num_envs: int,
        env_config: EnvConfig,
        cleanup_confirmation_active: bool,
        cleanup_message: str,
        last_cleanup_message_time: float,
        tensorboard_log_dir: Optional[str],  # Use TB log dir
    ):
        """Renders all UI components."""
        try:
            self.screen.fill(VisConfig.BLACK)
            self._render_left_panel(
                is_training, status, stats_summary, buffer_capacity, tensorboard_log_dir
            )
            self._render_game_area(envs, num_envs, env_config)

            if cleanup_confirmation_active:
                self._render_cleanup_confirmation()

            message_active = self._render_status_message(
                cleanup_message, last_cleanup_message_time
            )

            if not cleanup_confirmation_active and not message_active:
                self._render_tooltip()

            pygame.display.flip()
        except pygame.error as e:
            print(f"Pygame rendering error: {e}")
        except Exception as e:
            print(f"Unexpected critical rendering error: {e}")
            import traceback

            traceback.print_exc()
