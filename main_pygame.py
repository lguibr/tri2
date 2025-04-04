# File: main_pygame.py
import sys
import math
import pygame
import numpy as np
import os
import time
from typing import List, Tuple, Optional

# Import configurations - Ensure RewardConfig is imported if defined separately
from config import (
    VisConfig,
    EnvConfig,
    DQNConfig,
    TrainConfig,
    BufferConfig,
    ExplorationConfig,
    ModelConfig,
    StatsConfig,
    RewardConfig,  # Import RewardConfig
    DEVICE,
    RANDOM_SEED,
    BUFFER_SAVE_PATH,
)

# Import core components
try:
    from environment.game_state import GameState
    from environment.shape import Shape
    from environment.triangle import Triangle
    from environment.grid import (
        Grid,
    )  # Import Grid if needed directly (e.g., for rendering helpers)
except ImportError as e:
    print(f"Error importing environment components: {e}")
    print("Please ensure environment files and __init__.py exist.")
    sys.exit(1)

from agent.dqn_agent import DQNAgent
from agent.replay_buffer.base_buffer import ReplayBufferBase
from agent.replay_buffer.buffer_utils import create_replay_buffer
from training.trainer import Trainer
from stats.stats_recorder import (
    StatsRecorderBase,
    SimpleStatsRecorder,
)

# Import SQLiteLogger if used
try:
    from stats.sqlite_logger import SQLiteLogger
except ImportError:
    SQLiteLogger = None  # Define as None if not available
    print("SQLiteLogger not found, logging to DB disabled.")

# Import Plotter
from visualization.plotter import FourStatsPlotter
from utils.helpers import set_random_seeds, ensure_numpy

# --- Rendering Helpers ---


def render_shape_preview(surf: pygame.Surface, shape: Shape, cell_size: int):
    """Renders a single shape centered in the surface."""
    if not shape or not shape.triangles:  # Check if shape exists and has triangles
        return

    min_r, min_c, max_r, max_c = shape.bbox()
    shape_h_cells = max_r - min_r + 1
    shape_w_cells = max_c - min_c + 1

    total_w_pixels = shape_w_cells * (cell_size * 0.75) + (
        cell_size * 0.25
    )  # Approx width
    total_h_pixels = shape_h_cells * cell_size

    offset_x = (surf.get_width() - total_w_pixels) / 2 - min_c * (cell_size * 0.75)
    offset_y = (surf.get_height() - total_h_pixels) / 2 - min_r * cell_size

    for dr, dc, up in shape.triangles:
        abs_r, abs_c = dr, dc
        tri = Triangle(row=abs_r, col=abs_c, is_up=up)
        pts = tri.get_points(ox=offset_x, oy=offset_y, cw=cell_size, ch=cell_size)

        # Draw the triangle polygon fill
        pygame.draw.polygon(surf, shape.color, pts)
        # REMOVED: pygame.draw.polygon(surf, VisConfig.WHITE, pts, 1) # White outline


def render_env(
    surf: pygame.Surface,
    env: GameState,
    cell_w: int,
    cell_h: int,
    font: pygame.font.Font,
):
    """Renders a single environment state onto a Pygame surface."""
    try:
        bg_color = (20, 20, 20)
        if env.is_blinking():
            bg_color = (100, 100, 0)
        elif env.is_frozen() and not env.is_over():
            bg_color = (30, 30, 100)
        surf.fill(bg_color)

        if (
            hasattr(env, "grid")
            and hasattr(env.grid, "triangles")
            and isinstance(env.grid.triangles, list)
            and len(env.grid.triangles) == env.grid.rows
            and all(
                isinstance(row, list) and len(row) == env.grid.cols
                for row in env.grid.triangles
            )
        ):

            for r in range(env.grid.rows):
                for c in range(env.grid.cols):
                    t = env.grid.triangles[r][c]
                    if not hasattr(t, "get_points"):
                        continue

                    try:
                        pts = t.get_points(ox=0, oy=0, cw=cell_w, ch=cell_h)
                        if t.is_death:
                            color = VisConfig.BLACK
                        elif t.is_occupied:
                            color = t.color if t.color else VisConfig.RED
                        else:
                            color = VisConfig.GRAY
                        # Draw triangle fill
                        pygame.draw.polygon(surf, color, pts)
                        # REMOVED: pygame.draw.polygon(surf, VisConfig.LIGHTG, pts, 1) # Light gray outline
                    except Exception as e_render:
                        print(f"Error rendering triangle at ({r},{c}): {e_render}")
                        pygame.draw.rect(
                            surf,
                            (255, 100, 0),
                            (t.col * (cell_w * 0.75), t.row * cell_h, cell_w, cell_h),
                            1,
                        )
        else:
            pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
            err_txt = font.render("Invalid Grid", True, VisConfig.RED)
            surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))

        if env.is_over():
            overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            overlay.fill((100, 0, 0, 180))
            surf.blit(overlay, (0, 0))
            over_text = font.render("GAME OVER", True, VisConfig.WHITE)
            surf.blit(over_text, over_text.get_rect(center=surf.get_rect().center))
        elif env.is_frozen():
            freeze_text = font.render("Frozen", True, VisConfig.WHITE)
            surf.blit(
                freeze_text,
                freeze_text.get_rect(
                    center=(surf.get_width() // 2, surf.get_height() - 15)
                ),
            )
    except AttributeError as e:
        pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
        err_txt = font.render(f"Attr Error: {e}", True, VisConfig.RED, VisConfig.BLACK)
        surf.blit(err_txt, err_txt.get_rect(center=surf.get_rect().center))
    except Exception as e:
        print(f"Unexpected Render Error in render_env: {e}")
        pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
        import traceback

        traceback.print_exc()


# --- Main Application Class ---


class MainApp:
    def __init__(self):
        print("Initializing Pygame Application...")
        set_random_seeds(RANDOM_SEED)
        pygame.init()
        pygame.font.init()

        self.vis_config = VisConfig
        self.env_config = EnvConfig
        self.reward_config = RewardConfig
        self.dqn_config = DQNConfig
        self.train_config = TrainConfig
        self.buffer_config = BufferConfig
        self.exploration_config = ExplorationConfig
        self.model_config = ModelConfig
        self.stats_config = StatsConfig
        self.num_envs = self.env_config.NUM_ENVS

        self.screen = pygame.display.set_mode(
            (self.vis_config.SCREEN_WIDTH, self.vis_config.SCREEN_HEIGHT),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption("TriCrack DQN - Enhanced Training")
        self.clock = pygame.time.Clock()

        try:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_env_score = pygame.font.SysFont(None, 18)
            self.font_env_overlay = pygame.font.SysFont(None, 36)
        except Exception as e:
            print(f"Warning: Error initializing SysFont: {e}. Using default font.")
            self.font_ui = pygame.font.Font(None, 24)
            self.font_env_score = pygame.font.Font(None, 18)
            self.font_env_overlay = pygame.font.Font(None, 36)

        self.is_training = False
        self.cleanup_confirmation_active = False
        self.last_cleanup_message_time = 0
        self.cleanup_message = ""

        print("Initializing RL Components...")
        self._initialize_rl_components()
        self.plotter = FourStatsPlotter(smooth_window=10)
        print("Initialization Complete. Ready to start.")

    def _initialize_envs(self) -> List[GameState]:
        """Creates and validates the list of parallel game environments."""
        print(f"Initializing {self.num_envs} game environments...")
        try:
            envs = [GameState() for _ in range(self.num_envs)]
            s_test = envs[0].reset()
            s_np = ensure_numpy(s_test)
            if s_np.shape[0] != self.env_config.STATE_DIM:
                raise ValueError(
                    f"FATAL: Environment state dimension mismatch! "
                    f"GameState.get_state() returned length {s_np.shape[0]}, "
                    f"but EnvConfig.STATE_DIM is {self.env_config.STATE_DIM}. "
                    f"Check GameState implementation and config.py."
                )
            _ = envs[0].valid_actions()
            try:
                _, _ = envs[0].step(0)
            except Exception as step_e:
                print(f"Warning: Initial env.step(0) check failed: {step_e}")
            print(
                f"Successfully initialized and validated {self.num_envs} environments."
            )
            return envs
        except Exception as e:
            print(f"FATAL ERROR during environment initialization: {e}")
            import traceback

            traceback.print_exc()
            pygame.quit()
            sys.exit(1)

    def _initialize_stats_recorder(self) -> StatsRecorderBase:
        """Creates the appropriate statistics recorder based on config."""
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            try:
                self.stats_recorder.close()
            except Exception as e:
                print(f"Warning: Error closing previous stats recorder: {e}")

        if self.stats_config.USE_SQLITE_LOGGING and SQLiteLogger is not None:
            print(f"Using SQLite Logger (DB: {self.stats_config.SQLITE_DB_PATH})")
            db_dir = os.path.dirname(self.stats_config.SQLITE_DB_PATH)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            try:
                return SQLiteLogger(
                    db_path=self.stats_config.SQLITE_DB_PATH,
                    console_log_interval=self.stats_config.LOG_INTERVAL_STEPS,
                    avg_window=100,
                    log_transitions=self.stats_config.LOG_TRANSITIONS_TO_DB,
                )
            except Exception as e:
                print(
                    f"Error initializing SQLiteLogger: {e}. Falling back to SimpleStatsRecorder."
                )
                return SimpleStatsRecorder(
                    console_log_interval=self.stats_config.LOG_INTERVAL_STEPS,
                    avg_window=100,
                )
        else:
            if self.stats_config.USE_SQLITE_LOGGING and SQLiteLogger is None:
                print(
                    "SQLite logging enabled in config, but SQLiteLogger failed to import."
                )
            print("Using Simple In-Memory Stats Recorder.")
            return SimpleStatsRecorder(
                console_log_interval=self.stats_config.LOG_INTERVAL_STEPS,
                avg_window=100,
            )

    def _initialize_rl_components(self):
        """Initializes or re-initializes all RL-related components."""
        print("Initializing/Re-initializing RL components...")
        self.envs: List[GameState] = self._initialize_envs()
        self.agent: DQNAgent = DQNAgent(
            config=self.model_config,
            dqn_config=self.dqn_config,
            env_config=self.env_config,
        )
        self.buffer: ReplayBufferBase = create_replay_buffer(
            config=self.buffer_config, dqn_config=self.dqn_config
        )
        self.stats_recorder: StatsRecorderBase = self._initialize_stats_recorder()
        self.trainer: Trainer = Trainer(
            envs=self.envs,
            agent=self.agent,
            buffer=self.buffer,
            stats_recorder=self.stats_recorder,
            env_config=self.env_config,
            dqn_config=self.dqn_config,
            train_config=self.train_config,
            buffer_config=self.buffer_config,
            exploration_config=self.exploration_config,
            model_config=self.model_config,
        )
        print("RL components initialization finished.")

    def _cleanup_data(self):
        """Stops training, deletes checkpoints, buffer, logs, and re-initializes."""
        print("\n--- CLEANUP DATA INITIATED ---")
        self.is_training = False
        self.cleanup_confirmation_active = False

        if hasattr(self, "trainer") and self.trainer:
            print("Running trainer cleanup (flushes buffer, saves final state)...")
            try:
                self.trainer.cleanup()
            except Exception as e:
                print(f"Error during trainer cleanup: {e}")
        else:
            print("Trainer object not found, skipping trainer cleanup.")

        ckpt_path = self.model_config.SAVE_PATH
        try:
            if os.path.isfile(ckpt_path):
                os.remove(ckpt_path)
                print(f"Deleted agent checkpoint: {ckpt_path}")
                self.cleanup_message = "Agent checkpoint deleted."
            else:
                print(f"Agent checkpoint not found (already deleted?): {ckpt_path}")
                self.cleanup_message = "Agent ckpt not found."
        except OSError as e:
            print(f"Error deleting agent checkpoint {ckpt_path}: {e}")
            self.cleanup_message = f"Error deleting agent ckpt: {e}"

        buffer_path = BUFFER_SAVE_PATH
        try:
            if os.path.isfile(buffer_path):
                os.remove(buffer_path)
                print(f"Deleted buffer state: {buffer_path}")
                self.cleanup_message += "\nBuffer state deleted."
            else:
                print(f"Buffer state not found: {buffer_path}")
                self.cleanup_message += "\nBuffer state not found."
        except OSError as e:
            print(f"Error deleting buffer state {buffer_path}: {e}")
            self.cleanup_message += f"\nError deleting buffer: {e}"

        db_path = self.stats_config.SQLITE_DB_PATH
        try:
            if isinstance(self.stats_recorder, SQLiteLogger):
                self.stats_recorder.close()
            if os.path.isfile(db_path):
                os.remove(db_path)
                print(f"Deleted log database: {db_path}")
                self.cleanup_message += "\nLog DB deleted."
            else:
                if self.stats_config.USE_SQLITE_LOGGING:
                    print(f"Log database not found: {db_path}")
                    self.cleanup_message += "\nLog DB not found."
                else:
                    self.cleanup_message += "\n(SQLite logging disabled)"
        except OSError as e:
            print(f"Error deleting log database {db_path}: {e}")
            self.cleanup_message += f"\nError deleting log DB: {e}"
        except Exception as e_close:
            print(f"Error closing stats recorder before deleting DB: {e_close}")
            self.cleanup_message += f"\nError closing logger: {e_close}"

        print("Re-initializing RL components after cleanup...")
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            try:
                self.stats_recorder.close()
            except Exception:
                pass
        self._initialize_rl_components()
        self.plotter = FourStatsPlotter(smooth_window=10)
        print("--- CLEANUP DATA COMPLETE ---")
        self.last_cleanup_message_time = time.time()

    def _handle_input(self) -> bool:
        """Processes Pygame events. Returns False if app should quit."""
        global_mouse_pos = pygame.mouse.get_pos()
        sw, sh = self.screen.get_size()
        train_btn_rect = pygame.Rect(10, 10, 100, 40)
        cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 120, 40)
        confirm_yes_rect = pygame.Rect(sw // 2 - 110, sh // 2 + 30, 100, 40)
        confirm_no_rect = pygame.Rect(sw // 2 + 10, sh // 2 + 30, 100, 40)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                try:
                    self.screen = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE
                    )
                    print(f"Window resized to: {event.w} x {event.h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.cleanup_confirmation_active:
                        self.cleanup_confirmation_active = False
                        self.cleanup_message = "Cleanup cancelled."
                        self.last_cleanup_message_time = time.time()
                    else:
                        return False
                elif event.key == pygame.K_p and not self.cleanup_confirmation_active:
                    self.is_training = not self.is_training
                    action = "STARTED" if self.is_training else "PAUSED"
                    print(f"Training {action} (P key pressed)")
                    if not self.is_training and hasattr(
                        self.trainer, "_save_checkpoint"
                    ):
                        print("Saving checkpoint on pause...")
                        self.trainer._save_checkpoint(is_final=False)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.cleanup_confirmation_active:
                    if confirm_yes_rect.collidepoint(global_mouse_pos):
                        self._cleanup_data()
                    elif confirm_no_rect.collidepoint(global_mouse_pos):
                        self.cleanup_confirmation_active = False
                        self.cleanup_message = "Cleanup cancelled."
                        self.last_cleanup_message_time = time.time()
                else:
                    if train_btn_rect.collidepoint(global_mouse_pos):
                        self.is_training = not self.is_training
                        action = "STARTED" if self.is_training else "PAUSED"
                        print(f"Training {action} (Train/Stop button clicked)")
                        if not self.is_training and hasattr(
                            self.trainer, "_save_checkpoint"
                        ):
                            print("Saving checkpoint on pause...")
                            self.trainer._save_checkpoint(is_final=False)
                    elif cleanup_btn_rect.collidepoint(global_mouse_pos):
                        self.is_training = False
                        self.cleanup_confirmation_active = True
                        print("Cleanup requested - confirmation needed.")
        return True

    def _update(self):
        """Performs one step of the training process if active."""
        if not self.is_training or self.cleanup_confirmation_active:
            return
        try:
            step_start_time = time.time()
            self.trainer.step()
            step_duration = time.time() - step_start_time
            self.stats_recorder.log_summary(
                self.trainer.global_step
            )  # Delegate logging frequency check
            if self.vis_config.VISUAL_STEP_DELAY > 0:
                remaining_delay = self.vis_config.VISUAL_STEP_DELAY - step_duration
                if remaining_delay > 0:
                    time.sleep(remaining_delay)
        except Exception as e:
            print(
                f"\n--- ERROR DURING TRAINING UPDATE (Step: {getattr(self.trainer, 'global_step', 'N/A')}) ---"
            )
            import traceback

            traceback.print_exc()
            print(f"--- Pausing training due to error. Check logs. ---")
            self.is_training = False

    def _render(self):
        """Draws the entire application state to the screen."""
        try:
            self.screen.fill(VisConfig.BLACK)
            current_width, current_height = self.screen.get_size()

            # --- 1. Left Panel ---
            lp_width = min(current_width, max(200, VisConfig.LEFT_PANEL_WIDTH))
            lp_rect = pygame.Rect(0, 0, lp_width, current_height)
            pygame.draw.rect(self.screen, (30, 30, 30), lp_rect)

            # Buttons
            train_btn_rect = pygame.Rect(10, 10, 100, 40)
            pygame.draw.rect(self.screen, (70, 70, 70), train_btn_rect, border_radius=5)
            btn_text = "Stop" if self.is_training else "Train"
            lbl_surf = self.font_ui.render(btn_text, True, VisConfig.WHITE)
            self.screen.blit(lbl_surf, lbl_surf.get_rect(center=train_btn_rect.center))
            cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 120, 40)
            pygame.draw.rect(
                self.screen, (100, 40, 40), cleanup_btn_rect, border_radius=5
            )
            cleanup_lbl_surf = self.font_ui.render(
                "Cleanup Data", True, VisConfig.WHITE
            )
            self.screen.blit(
                cleanup_lbl_surf,
                cleanup_lbl_surf.get_rect(center=cleanup_btn_rect.center),
            )

            # Info Text
            stats_summary = self.stats_recorder.get_summary(self.trainer.global_step)
            buffer_size = stats_summary.get("buffer_size", 0)
            buffer_cap = getattr(self.buffer, "capacity", 0)
            buffer_perc = (buffer_size / buffer_cap * 100) if buffer_cap > 0 else 0.0
            info_lines = [
                f"Global Steps: {self.trainer.global_step}",
                f"Total Episodes: {stats_summary.get('total_episodes', 0)}",
                f"Steps/Sec: {stats_summary.get('steps_per_second', 0.0):.1f}",
                f"Avg Score ({stats_summary.get('num_ep_scores', 0)}): {stats_summary.get('avg_score_100', 0.0):.2f}",
                f"Best Score: {stats_summary.get('best_score', 0.0):.2f}",
                f"Avg Length ({stats_summary.get('num_ep_lengths', 0)}): {stats_summary.get('avg_length_100', 0.0):.1f}",
                f"Avg Loss ({stats_summary.get('num_losses', 0)}): {stats_summary.get('avg_loss_100', 0.0):.4f}",
                f"Avg Max Q ({stats_summary.get('num_avg_max_qs', 0)}): {stats_summary.get('avg_max_q_100', 0.0):.3f}",
                f"Epsilon: {stats_summary.get('epsilon', 0.0):.3f}",
                (
                    f"PER Beta: {stats_summary.get('beta', 0.0):.3f}"
                    if self.buffer_config.USE_PER
                    else "PER Beta: N/A"
                ),
                f"Buffer: {buffer_size}/{buffer_cap} ({buffer_perc:.1f}%)",
                f"Training: {'ACTIVE' if self.is_training else 'PAUSED'}",
                f"Device: {DEVICE.type.upper()}",
                f"Model: {self.model_config.MODEL_TYPE}",
            ]
            text_y_start = train_btn_rect.bottom + 15
            line_height = self.font_ui.get_linesize()
            for idx, line in enumerate(info_lines):
                line_surf = self.font_ui.render(line, True, VisConfig.WHITE)
                self.screen.blit(line_surf, (10, text_y_start + idx * line_height))

            # Plot Area
            chart_y_start = text_y_start + len(info_lines) * line_height + 10
            chart_height = max(100, current_height - chart_y_start - 10)
            chart_rect = pygame.Rect(10, chart_y_start, lp_width - 20, chart_height)

            if chart_rect.width > 0 and chart_rect.height > 0:
                if chart_rect.width > 20 and chart_rect.height > 50:
                    try:
                        chart_subsurface = self.screen.subsurface(chart_rect)
                        self.plotter.update_data(
                            self.trainer.global_step, stats_summary
                        )
                        self.plotter.render(chart_subsurface)
                    except ValueError as e:
                        print(
                            f"Error creating/using plot subsurface (Rect: {chart_rect}): {e}"
                        )
                        pygame.draw.rect(self.screen, (50, 0, 0), chart_rect, 1)
                    except Exception as plot_e:
                        print(f"Error during plotting: {plot_e}")
                        pygame.draw.rect(self.screen, (50, 0, 50), chart_rect, 1)
                else:
                    pygame.draw.rect(self.screen, (50, 50, 0), chart_rect)
                    err_surf = self.font_ui.render(
                        "Plot Area Too Small", True, VisConfig.GRAY
                    )
                    self.screen.blit(
                        err_surf, err_surf.get_rect(center=chart_rect.center)
                    )
            else:
                pygame.draw.rect(self.screen, (50, 0, 0), chart_rect)

            # --- 2. Game Area ---
            ga_rect = pygame.Rect(lp_width, 0, current_width - lp_width, current_height)
            if self.num_envs > 0 and ga_rect.width > 0 and ga_rect.height > 0:
                aspect_ratio = (
                    ga_rect.width / ga_rect.height if ga_rect.height > 0 else 1
                )
                cols_env = max(1, int(math.sqrt(self.num_envs * aspect_ratio)))
                rows_env = math.ceil(self.num_envs / cols_env)
                total_spacing_w = (cols_env + 1) * self.vis_config.ENV_SPACING
                total_spacing_h = (rows_env + 1) * self.vis_config.ENV_SPACING
                cell_w = (
                    (ga_rect.width - total_spacing_w) // cols_env if cols_env > 0 else 0
                )
                cell_h = (
                    (ga_rect.height - total_spacing_h) // rows_env
                    if rows_env > 0
                    else 0
                )

                if cell_w > 10 and cell_h > 10:
                    env_idx = 0
                    for r in range(rows_env):
                        for c in range(cols_env):
                            if env_idx >= self.num_envs:
                                break
                            env_x = (
                                ga_rect.x
                                + self.vis_config.ENV_SPACING * (c + 1)
                                + c * cell_w
                            )
                            env_y = (
                                ga_rect.y
                                + self.vis_config.ENV_SPACING * (r + 1)
                                + r * cell_h
                            )
                            env_rect = pygame.Rect(env_x, env_y, cell_w, cell_h)
                            try:
                                sub_surf = self.screen.subsurface(env_rect)
                                tri_cell_w = cell_w / self.env_config.COLS * 1.333
                                tri_cell_h = cell_h / self.env_config.ROWS
                                render_env(
                                    sub_surf,
                                    self.envs[env_idx],
                                    int(tri_cell_w),
                                    int(tri_cell_h),
                                    self.font_env_overlay,
                                )
                                score_val = self.trainer.current_episode_scores[env_idx]
                                score_surf = self.font_env_score.render(
                                    f"S: {score_val:.1f}",
                                    True,
                                    VisConfig.WHITE,
                                    (0, 0, 0, 180),
                                )
                                sub_surf.blit(score_surf, (4, 4))

                                # Shape Previews
                                available_shapes = self.envs[env_idx].get_shapes()
                                if available_shapes:
                                    preview_dim = max(
                                        10, min(cell_w // 6, cell_h // 6, 25)
                                    )
                                    preview_spacing = 4
                                    total_preview_width = (
                                        len(available_shapes) * preview_dim
                                        + max(0, len(available_shapes) - 1)
                                        * preview_spacing
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
                                        render_shape_preview(
                                            temp_shape_surf, shape, preview_cell_size
                                        )
                                        sub_surf.blit(
                                            temp_shape_surf, (preview_x, start_y)
                                        )
                                        # REMOVED: Border around shape preview
                                        # pygame.draw.rect( sub_surf, VisConfig.LIGHTG, (preview_x-1, start_y-1, preview_dim+2, preview_dim+2), 1 )
                            except ValueError as e:
                                print(f"Error creating env subsurface {env_idx}: {e}")
                                pygame.draw.rect(self.screen, (0, 0, 50), env_rect, 1)
                            except Exception as e_render_env:
                                print(f"Error rendering env {env_idx}: {e_render_env}")
                                pygame.draw.rect(self.screen, (50, 0, 50), env_rect, 1)
                            env_idx += 1
                else:
                    err_surf = self.font_ui.render(
                        "Envs Too Small to Render", True, VisConfig.GRAY
                    )
                    self.screen.blit(err_surf, err_surf.get_rect(center=ga_rect.center))

            # --- 3. Cleanup Confirmation Overlay ---
            if self.cleanup_confirmation_active:
                overlay = pygame.Surface(
                    (current_width, current_height), pygame.SRCALPHA
                )
                overlay.fill((0, 0, 0, 200))
                self.screen.blit(overlay, (0, 0))
                center_x, center_y = current_width // 2, current_height // 2
                prompt_l1 = self.font_env_overlay.render(
                    "DELETE ALL SAVED DATA?", True, VisConfig.RED
                )
                self.screen.blit(
                    prompt_l1, prompt_l1.get_rect(center=(center_x, center_y - 60))
                )
                prompt_l2 = self.font_ui.render(
                    "(Agent Checkpoint, Buffer State, Log Database)",
                    True,
                    VisConfig.WHITE,
                )
                self.screen.blit(
                    prompt_l2, prompt_l2.get_rect(center=(center_x, center_y - 25))
                )
                prompt_l3 = self.font_ui.render(
                    "This action cannot be undone!", True, VisConfig.YELLOW
                )
                self.screen.blit(
                    prompt_l3, prompt_l3.get_rect(center=(center_x, center_y))
                )
                confirm_yes_rect = pygame.Rect(center_x - 110, center_y + 30, 100, 40)
                confirm_no_rect = pygame.Rect(center_x + 10, center_y + 30, 100, 40)
                pygame.draw.rect(
                    self.screen, (0, 150, 0), confirm_yes_rect, border_radius=5
                )
                pygame.draw.rect(
                    self.screen, (150, 0, 0), confirm_no_rect, border_radius=5
                )
                yes_text = self.font_ui.render("YES", True, VisConfig.WHITE)
                no_text = self.font_ui.render("NO", True, VisConfig.WHITE)
                self.screen.blit(
                    yes_text, yes_text.get_rect(center=confirm_yes_rect.center)
                )
                self.screen.blit(
                    no_text, no_text.get_rect(center=confirm_no_rect.center)
                )

            # --- 4. Cleanup Status Message ---
            if self.cleanup_message and (
                time.time() - self.last_cleanup_message_time < 5.0
            ):
                lines = self.cleanup_message.split("\n")
                max_width = 0
                msg_surfs = []
                for line in lines:
                    msg_surf = self.font_ui.render(
                        line, True, VisConfig.YELLOW, VisConfig.BLACK
                    )
                    msg_surfs.append(msg_surf)
                    max_width = max(max_width, msg_surf.get_width())
                total_height = (
                    sum(s.get_height() for s in msg_surfs) + (len(lines) - 1) * 2
                )
                bg_rect = pygame.Rect(0, 0, max_width + 10, total_height + 10)
                bg_rect.midbottom = (current_width // 2, current_height - 10)
                pygame.draw.rect(self.screen, VisConfig.BLACK, bg_rect, border_radius=3)
                current_y = bg_rect.top + 5
                for msg_surf in msg_surfs:
                    msg_rect = msg_surf.get_rect(midtop=(bg_rect.centerx, current_y))
                    self.screen.blit(msg_surf, msg_rect)
                    current_y += msg_surf.get_height() + 2
            elif self.cleanup_message and (
                time.time() - self.last_cleanup_message_time >= 5.0
            ):
                self.cleanup_message = ""

            pygame.display.flip()
        except pygame.error as e:
            print(f"Pygame rendering error: {e}")
        except Exception as e:
            print(f"Unexpected critical rendering error: {e}")
            import traceback

            traceback.print_exc()
            self.is_training = False

    def run(self):
        """Main application loop."""
        print("Starting main application loop...")
        running = True
        while running:
            running = self._handle_input()
            if not running:
                break
            self._update()
            self._render()
            self.clock.tick(self.vis_config.FPS if self.vis_config.FPS > 0 else 0)

        print("Exiting application...")
        if hasattr(self, "trainer") and self.trainer:
            print("Performing final trainer cleanup...")
            self.trainer.cleanup()
        pygame.quit()
        print("Application exited.")


if __name__ == "__main__":
    print("--- Pre-Run Checks ---")
    try:
        print("Checking GameState and Configuration Compatibility...")
        gs_test = GameState()
        gs_test.reset()
        s_test = gs_test.get_state()
        state_len = len(s_test)
        config_dim = EnvConfig.STATE_DIM
        if state_len != config_dim:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!! FATAL ERROR: State Dimension Mismatch             !!!")
            print(f"!!! GameState.get_state() length: {state_len}")
            print(f"!!! EnvConfig.STATE_DIM:          {config_dim}")
            print(f"!!! Please check environment/game_state.py and config.py")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            sys.exit(1)
        else:
            print(f"GameState state dimension check PASSED (Length: {state_len}).")
        _ = gs_test.valid_actions()
        print("GameState basic instantiation and method checks PASSED.")
        del gs_test
    except NameError:
        print("FATAL ERROR: GameState class not found. Check environment imports.")
        sys.exit(1)
    except ImportError as e:
        print(f"FATAL ERROR: Failed to import environment components: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"FATAL ERROR during pre-run checks: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR during GameState pre-check: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("--- Pre-Run Checks Complete ---")

    app = MainApp()
    app.run()
