import sys
import math
import pygame
import numpy as np
import os
import time
from typing import List, Tuple, Optional 

# Import configurations
from config import (
    VisConfig,
    EnvConfig,
    DQNConfig,
    TrainConfig,
    BufferConfig,
    ExplorationConfig,
    ModelConfig,
    StatsConfig,
    DEVICE,
    RANDOM_SEED,
    BUFFER_SAVE_PATH,
)

# Import core components
try:
    from environment.game_state import GameState
    from environment.shape import Shape
    from environment.triangle import (
        Triangle,
    )  
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
from stats.sqlite_logger import SQLiteLogger
from visualization.plotter import FourStatsPlotter
from utils.helpers import set_random_seeds, ensure_numpy


def render_shape_preview(surf: pygame.Surface, shape: Shape, cell_size: int):
    """Renders a single shape centered in the surface."""
    if not shape:
        return

    min_r, min_c, max_r, max_c = shape.bbox()
    shape_h_cells = max_r - min_r + 1
    shape_w_cells = max_c - min_c + 1

    # Estimate pixel size needed
    total_w_pixels = shape_w_cells * (cell_size * 0.75) + (cell_size * 0.25)
    total_h_pixels = shape_h_cells * cell_size

    # Calculate offset to center the shape roughly within the given surface
    offset_x = (surf.get_width() - total_w_pixels) / 2 - min_c * (cell_size * 0.75)
    offset_y = (surf.get_height() - total_h_pixels) / 2 - min_r * cell_size

    for dr, dc, up in shape.triangles:
        # Mock triangle for get_points
        tri = type("obj", (object,), {"row": dr, "col": dc, "is_up": up})()
        pts = Triangle.get_points(tri, offset_x, offset_y, cell_size, cell_size)
        pygame.draw.polygon(surf, shape.color, pts)
        pygame.draw.polygon(surf, VisConfig.WHITE, pts, 1)  # Keep outline


def render_env(
    surf: pygame.Surface,
    env: GameState,
    cell_w: int,
    cell_h: int,
    font: pygame.font.Font,
):
    """Renders a single environment state onto a surface."""
    try:
        # Base background
        bg_color = (20, 20, 20)
        if env.is_blinking():
            bg_color = VisConfig.YELLOW
        elif env.is_frozen():
            bg_color = VisConfig.BLUE
        surf.fill(bg_color)

        # Grid rendering
        if (
            hasattr(env, "grid")
            and hasattr(env.grid, "triangles")
            and isinstance(env.grid.triangles, list)
        ):
            for row in env.grid.triangles:
                if isinstance(row, list):
                    for t in row:
                        if all(
                            hasattr(t, attr)
                            for attr in [
                                "get_points",
                                "is_death",
                                "is_occupied",
                                "color",
                            ]
                        ):
                            try:
                                pts = t.get_points(0, 0, cell_w, cell_h)
                                if t.is_death:
                                    color = VisConfig.BLACK
                                elif t.is_occupied:
                                    color = t.color if t.color else (200, 50, 50)
                                else:
                                    color = VisConfig.GRAY
                                pygame.draw.polygon(surf, color, pts)
                                pygame.draw.polygon(surf, VisConfig.WHITE, pts, 1)
                            except Exception as e_render:
                                print(f"Error rendering triangle: {e_render}")
                                pygame.draw.rect(
                                    surf, (255, 100, 0), surf.get_rect(), 1
                                )
                                break
                        else:
                            pygame.draw.rect(surf, (255, 100, 0), surf.get_rect(), 1)
                            break
                    else:
                        continue
                    break
                else:
                    pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
                    break
        else:
            pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)

        # Overlay Game Over / Freeze Text
        if env.is_over():
            surf.fill((100, 0, 0, 180), special_flags=pygame.BLEND_RGBA_MULT)
            over_text = font.render("GAME OVER", True, VisConfig.WHITE)
            surf.blit(over_text, over_text.get_rect(center=surf.get_rect().center))
        elif env.is_frozen():
            freeze_text = font.render("Frozen", True, VisConfig.WHITE)
            surf.blit(
                freeze_text,
                freeze_text.get_rect(
                    center=(surf.get_width() // 2, surf.get_height() - 20)
                ),
            )

    except AttributeError as e:
        pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)
    except Exception as e:
        print(f"Unexpected Render Error: {e}")
        pygame.draw.rect(surf, (255, 0, 0), surf.get_rect(), 2)


class MainApp:
    def __init__(self):
        print("Initializing Pygame Application...")
        set_random_seeds(RANDOM_SEED)
        pygame.init()
        pygame.font.init()
        self.train_config = TrainConfig
        self.dqn_config = DQNConfig
        self.buffer_config = BufferConfig
        self.model_config = ModelConfig
        self.env_config = EnvConfig
        self.exploration_config = ExplorationConfig
        self.stats_config = StatsConfig
        self.num_envs = EnvConfig.NUM_ENVS
        self.screen = pygame.display.set_mode(
            (VisConfig.SCREEN_WIDTH, VisConfig.SCREEN_HEIGHT), pygame.RESIZABLE
        )
        pygame.display.set_caption("TriCrack DQN - Enhanced")
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont(None, 24)
            self.font_large = pygame.font.SysFont(None, 48)
        except Exception as e:
            print(f"Error initializing SysFont: {e}. Using default.")
            self.font = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 48)

        self.cleanup_confirmation_active = False
        self.last_cleanup_message_time = 0
        self.cleanup_message = ""

        print("Initializing RL Components...")
        self._initialize_rl_components()

        self.plotter: FourStatsPlotter = FourStatsPlotter()
        self.is_training: bool = False
        print("Initialization Complete.")

    def _initialize_rl_components(self):
        print("Initializing/Re-initializing RL components...")
        self.envs: List[GameState] = self._initialize_envs()
        self.agent: DQNAgent = DQNAgent(
            self.model_config, self.dqn_config, self.env_config
        )
        self.buffer: ReplayBufferBase = create_replay_buffer(
            self.buffer_config, self.dqn_config
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

    def _initialize_envs(self) -> List[GameState]:
        try:
            envs = [GameState() for _ in range(EnvConfig.NUM_ENVS)]
            s = envs[0].get_state()
            s_np = ensure_numpy(s)
            if s_np.shape[0] != EnvConfig.STATE_DIM:
                raise ValueError(
                    f"Env state dim {s_np.shape[0]} != Config STATE_DIM {EnvConfig.STATE_DIM}. Check GameState.get_state() and EnvConfig.STATE_DIM."
                )
            _ = envs[0].valid_actions()
            envs[0].reset()
            print(f"Initialized {EnvConfig.NUM_ENVS} environments.")
            return envs
        except Exception as e:
            print(f"FATAL: Failed to initialize environments: {e}")
            pygame.quit()
            sys.exit(1)

    def _initialize_stats_recorder(self) -> StatsRecorderBase:
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            self.stats_recorder.close()
        if StatsConfig.USE_SQLITE_LOGGING:
            print(f"Using SQLite Logger (DB: {StatsConfig.SQLITE_DB_PATH})")
            db_dir = os.path.dirname(StatsConfig.SQLITE_DB_PATH)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            return SQLiteLogger(
                db_path=StatsConfig.SQLITE_DB_PATH,
                console_log_interval=StatsConfig.LOG_INTERVAL_STEPS,
                avg_window=100,
                log_transitions=StatsConfig.LOG_TRANSITIONS_TO_DB,
            )
        else:
            print("Using Simple In-Memory Stats Recorder")
            return SimpleStatsRecorder(
                console_log_interval=StatsConfig.LOG_INTERVAL_STEPS, avg_window=100
            )

    def _cleanup_data(self):
        print("\n--- CLEANUP INITIATED ---")
        self.is_training = False
        self.cleanup_confirmation_active = False
        if hasattr(self, "trainer") and self.trainer:
            print("Running trainer cleanup...")
            self.trainer.cleanup()
        else:
            print("Trainer not found, skipping.")

        ckpt_path = ModelConfig.SAVE_PATH
        try:
            if os.path.isfile(ckpt_path):
                os.remove(ckpt_path)
                print(f"Deleted checkpoint: {ckpt_path}")
                self.cleanup_message = "Ckpt deleted."
            else:
                print(f"Checkpoint not found: {ckpt_path}")
                self.cleanup_message = "Ckpt not found."
        except OSError as e:
            print(f"Error deleting checkpoint {ckpt_path}: {e}")
            self.cleanup_message = f"Error deleting ckpt: {e}"

        buffer_path = BUFFER_SAVE_PATH
        try:
            if os.path.isfile(buffer_path):
                os.remove(buffer_path)
                print(f"Deleted buffer state: {buffer_path}")
                self.cleanup_message += "\nBuffer deleted."
            else:
                print(f"Buffer state not found: {buffer_path}")
                self.cleanup_message += "\nBuffer not found."
        except OSError as e:
            print(f"Error deleting buffer state {buffer_path}: {e}")
            self.cleanup_message += f"\nError deleting buffer: {e}"

        db_path = StatsConfig.SQLITE_DB_PATH
        try:
            if isinstance(self.stats_recorder, SQLiteLogger):
                self.stats_recorder.close()
            if os.path.isfile(db_path):
                os.remove(db_path)
                print(f"Deleted log database: {db_path}")
                self.cleanup_message += "\nLog DB deleted."
            else:
                print(f"Log database not found: {db_path}")
                self.cleanup_message += "\nLog DB not found."
        except OSError as e:
            print(f"Error deleting log database {db_path}: {e}")
            self.cleanup_message += f"\nError deleting log DB: {e}"
        except Exception as e_close:
            print(f"Error closing stats recorder: {e_close}")
            self.cleanup_message += f"\nError closing logger: {e_close}"

        print("Re-initializing RL components...")
        if hasattr(self, "stats_recorder") and self.stats_recorder:
            self.stats_recorder.close()  
        self._initialize_rl_components()
        self.plotter = FourStatsPlotter()

        print("--- CLEANUP COMPLETE ---")
        self.last_cleanup_message_time = time.time()

    def _handle_input(self) -> bool:
        global_mouse_pos = pygame.mouse.get_pos()
        train_btn_rect = pygame.Rect(10, 10, 100, 40)
        cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 120, 40)
        sw, sh = self.screen.get_size()
        confirm_yes_rect = pygame.Rect(sw // 2 - 110, sh // 2 + 25, 100, 40)
        confirm_no_rect = pygame.Rect(sw // 2 + 10, sh // 2 + 25, 100, 40)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                try:
                    self.screen = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE
                    )
                except pygame.error as e:
                    print(f"Error resizing window: {e}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.cleanup_confirmation_active:
                        self.cleanup_confirmation_active = False
                    else:
                        return False
                elif event.key == pygame.K_p and not self.cleanup_confirmation_active:
                    self.is_training = not self.is_training
                    print(
                        f"Training {'STARTED' if self.is_training else 'PAUSED'} (P key)"
                    )
                    if not self.is_training:
                        self.trainer._save_checkpoint()
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
                        print(
                            f"Training {'STARTED' if self.is_training else 'PAUSED'} (Button)"
                        )
                        if not self.is_training:
                            self.trainer._save_checkpoint()
                    elif cleanup_btn_rect.collidepoint(global_mouse_pos):
                        self.is_training = False
                        self.cleanup_confirmation_active = True
                        print("Cleanup requested - confirm.")
        return True

    def _update(self):
        if not self.is_training or self.cleanup_confirmation_active:
            return
        try:
            self.trainer.step()
            self.stats_recorder.log_summary(self.trainer.global_step)
        except Exception as e:
            print(
                f"\n--- ERROR DURING TRAINING UPDATE (Step: {self.trainer.global_step}) ---"
            )
            import traceback

            traceback.print_exc()
            print(f"--- Pausing training due to error ---")
            self.is_training = False

    def _render(self):
        """Draws the current state to the screen."""
        try:
            self.screen.fill(VisConfig.BLACK)
            current_width, current_height = self.screen.get_size()

            # --- Left Panel ---
            lp_rect = pygame.Rect(0, 0, VisConfig.LEFT_PANEL_WIDTH, current_height)
            pygame.draw.rect(self.screen, (30, 30, 30), lp_rect)

            # Buttons (Train, Cleanup)
            train_btn_rect = pygame.Rect(10, 10, 100, 40)
            pygame.draw.rect(self.screen, (70, 70, 70), train_btn_rect)
            btn_text = "Stop" if self.is_training else "Train"
            lbl_surf = self.font.render(btn_text, True, VisConfig.WHITE)
            self.screen.blit(lbl_surf, lbl_surf.get_rect(center=train_btn_rect.center))
            cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 120, 40)
            pygame.draw.rect(self.screen, (100, 40, 40), cleanup_btn_rect)
            cleanup_lbl_surf = self.font.render("Cleanup Data", True, VisConfig.WHITE)
            self.screen.blit(
                cleanup_lbl_surf,
                cleanup_lbl_surf.get_rect(center=cleanup_btn_rect.center),
            )

            # Info Text
            stats_summary = self.stats_recorder.get_summary()
            buffer_size = stats_summary.get("buffer_size", len(self.buffer))
            buffer_cap = self.buffer.capacity
            sps = stats_summary.get("steps_per_second", 0.0)

            lines = [
                f"Steps: {self.trainer.global_step}",
                f"Episodes: {stats_summary.get('total_episodes', 0)}",
                f"Speed: {sps:.1f} steps/s",
                f"Epsilon: {stats_summary.get('epsilon', 0.0):.3f}",
                f"Beta: {stats_summary.get('beta', 0.0):.3f}",
                f"Best Score: {stats_summary.get('best_score', -float('inf')):.1f}",
                f"Avg Score ({stats_summary.get('num_ep_scores', 0)}): {stats_summary.get('avg_score_100', 0.0):.1f}",
                f"Avg Ep Len ({stats_summary.get('num_ep_lengths',0)}): {stats_summary.get('avg_length_100', 0.0):.1f}",
                f"Buffer: {buffer_size}/{buffer_cap}",
                f"Training: {'ON' if self.is_training else 'OFF'}",
            ]
            text_y_start = 60
            for idx, line in enumerate(lines):
                line_surf = self.font.render(line, True, VisConfig.WHITE)
                self.screen.blit(line_surf, (10, text_y_start + idx * 25))

            # Plot Area
            chart_y_start = text_y_start + len(lines) * 25 + 10
            chart_height = max(
                100, current_height - chart_y_start - 10
            ) 
            chart_rect = pygame.Rect(
                10, chart_y_start, VisConfig.LEFT_PANEL_WIDTH - 20, chart_height
            )

            if chart_rect.width > 5 and chart_rect.height > 5:
                try:
                    chart_subsurface = self.screen.subsurface(chart_rect)
                    self.plotter.update_data(self.trainer.global_step, stats_summary)
                    self.plotter.render(chart_subsurface)
                except ValueError:
                    pygame.draw.rect(self.screen, (50, 0, 0), chart_rect)
            else:
                pygame.draw.rect(self.screen, (50, 0, 0), chart_rect)

            ga_rect = pygame.Rect(
                VisConfig.LEFT_PANEL_WIDTH,
                0,
                current_width - VisConfig.LEFT_PANEL_WIDTH,
                current_height,
            )
            if self.num_envs > 0 and ga_rect.width > 0 and ga_rect.height > 0:
                aspect_ratio = (
                    ga_rect.width / ga_rect.height if ga_rect.height > 0 else 1
                )
                cols_env = max(1, int(math.sqrt(self.num_envs * aspect_ratio)))
                rows_env = math.ceil(self.num_envs / cols_env)
                cell_w = ga_rect.width // cols_env
                cell_h = ga_rect.height // rows_env

                env_idx = 0
                for r in range(rows_env):
                    for c in range(cols_env):
                        if env_idx >= self.num_envs:
                            break
                        env_x = ga_rect.x + c * cell_w + VisConfig.ENV_SPACING // 2
                        env_y = ga_rect.y + r * cell_h + VisConfig.ENV_SPACING // 2
                        env_w = cell_w - VisConfig.ENV_SPACING
                        env_h = cell_h - VisConfig.ENV_SPACING

                        if env_w > 5 and env_h > 5:
                            env_rect = pygame.Rect(env_x, env_y, env_w, env_h)
                            try:
                                # Get subsurface for this env
                                sub_surf = self.screen.subsurface(env_rect)

                                # Render the environment grid and state overlays
                                tri_cell_w = (
                                    env_w // EnvConfig.COLS
                                    if EnvConfig.COLS > 0
                                    else env_w
                                )
                                tri_cell_h = (
                                    env_h // EnvConfig.ROWS
                                    if EnvConfig.ROWS > 0
                                    else env_h
                                )
                                render_env(
                                    sub_surf,
                                    self.envs[env_idx],
                                    tri_cell_w,
                                    tri_cell_h,
                                    self.font_large,
                                )

                                # Display current episode score
                                score_surf = self.font.render(
                                    f"{self.trainer.current_episode_scores[env_idx]:.1f}",
                                    True,
                                    VisConfig.WHITE,
                                    (0, 0, 0, 150),  # Semi-transparent background
                                )
                                sub_surf.blit(score_surf, (5, 5))

                                # <<< NEW >>> Render Shape Previews for this environment
                                available_shapes = self.envs[env_idx].get_shapes()
                                num_shapes = len(available_shapes)
                                if num_shapes > 0:
                                    # Calculate size and position for previews (e.g., top-right corner)
                                    preview_dim = min(
                                        env_w // 6, env_h // 6, 30
                                    )  # Small previews
                                    preview_spacing = 4
                                    total_preview_width = (
                                        num_shapes * preview_dim
                                        + max(0, num_shapes - 1) * preview_spacing
                                    )
                                    start_x = (
                                        sub_surf.get_width()
                                        - total_preview_width
                                        - preview_spacing
                                    )  # Align right
                                    start_y = preview_spacing  # Align top

                                    for i, shape in enumerate(available_shapes):
                                        preview_x = start_x + i * (
                                            preview_dim + preview_spacing
                                        )
                                        # Create a temporary surface for the shape preview
                                        temp_shape_surf = pygame.Surface(
                                            (preview_dim, preview_dim), pygame.SRCALPHA
                                        )
                                        # Render the shape onto the temp surface (use smaller cell size)
                                        render_shape_preview(
                                            temp_shape_surf,
                                            shape,
                                            max(1, preview_dim // 4),
                                        )  # Adjust cell size based on preview dim
                                        # Blit the temp surface onto the environment's subsurface
                                        sub_surf.blit(
                                            temp_shape_surf, (preview_x, start_y)
                                        )
                                        # Optional: Add border around preview
                                        pygame.draw.rect(
                                            sub_surf,
                                            VisConfig.LIGHTG,
                                            (
                                                preview_x,
                                                start_y,
                                                preview_dim,
                                                preview_dim,
                                            ),
                                            1,
                                        )

                            except ValueError as e:  # Handle subsurface errors
                                pygame.draw.rect(
                                    self.screen, (0, 0, 50), env_rect, 1
                                )  # Draw placeholder border
                        else:
                            # Draw placeholder if cell size is too small
                            env_rect = pygame.Rect(
                                env_x, env_y, max(1, env_w), max(1, env_h)
                            )
                            pygame.draw.rect(self.screen, (0, 50, 0), env_rect, 1)
                        env_idx += 1

            if self.cleanup_confirmation_active:
                overlay = pygame.Surface(
                    (current_width, current_height), pygame.SRCALPHA
                )
                overlay.fill((0, 0, 0, 180))
                self.screen.blit(overlay, (0, 0))
                prompt_l1 = self.font_large.render(
                    "DELETE ALL DATA?", True, VisConfig.RED
                )
                prompt_l2 = self.font.render(
                    "(Checkpoints, Logs, Buffer)", True, VisConfig.WHITE
                )
                prompt_l3 = self.font.render(
                    "This cannot be undone!", True, VisConfig.YELLOW
                )
                self.screen.blit(
                    prompt_l1,
                    prompt_l1.get_rect(
                        center=(current_width // 2, current_height // 2 - 60)
                    ),
                )
                self.screen.blit(
                    prompt_l2,
                    prompt_l2.get_rect(
                        center=(current_width // 2, current_height // 2 - 25)
                    ),
                )
                self.screen.blit(
                    prompt_l3,
                    prompt_l3.get_rect(
                        center=(current_width // 2, current_height // 2)
                    ),
                )
                confirm_yes_rect = pygame.Rect(
                    current_width // 2 - 110, current_height // 2 + 25, 100, 40
                )
                confirm_no_rect = pygame.Rect(
                    current_width // 2 + 10, current_height // 2 + 25, 100, 40
                )
                pygame.draw.rect(
                    self.screen, (0, 150, 0), confirm_yes_rect, border_radius=5
                )
                pygame.draw.rect(
                    self.screen, (150, 0, 0), confirm_no_rect, border_radius=5
                )
                yes_text = self.font.render("YES", True, VisConfig.WHITE)
                no_text = self.font.render("NO", True, VisConfig.WHITE)
                self.screen.blit(
                    yes_text, yes_text.get_rect(center=confirm_yes_rect.center)
                )
                self.screen.blit(
                    no_text, no_text.get_rect(center=confirm_no_rect.center)
                )

            if self.cleanup_message and (
                time.time() - self.last_cleanup_message_time < 5.0
            ):
                lines = self.cleanup_message.split("\n")
                y_offset = 10
                for i, line in enumerate(lines):
                    msg_surf = self.font.render(
                        line, True, VisConfig.YELLOW, VisConfig.BLACK
                    )
                    msg_rect = msg_surf.get_rect(
                        midbottom=(
                            current_width // 2,
                            current_height - y_offset - i * 20,
                        )
                    )
                    self.screen.blit(msg_surf, msg_rect)
            elif self.cleanup_message and (
                time.time() - self.last_cleanup_message_time >= 5.0
            ):
                self.cleanup_message = ""

            pygame.display.flip()

        except pygame.error as e:
            print(f"Pygame rendering error: {e}")
        except Exception as e:
            print(f"Unexpected rendering error: {e}")
            import traceback

            traceback.print_exc()

    def run(self):
        running = True
        while running:
            running = self._handle_input()
            if not running:
                break
            self._update()
            self._render()
            self.clock.tick(VisConfig.FPS)
        print("Exiting application...")
        if hasattr(self, "trainer") and self.trainer:
            self.trainer.cleanup()  # Ensure cleanup
        pygame.quit()


if __name__ == "__main__":
    try:
        print("Performing pre-run checks...")
        gs_test = GameState()
        gs_test.reset()
        s_test = gs_test.get_state()
        a_test = gs_test.valid_actions()
        state_len = len(s_test)
        config_dim = EnvConfig.STATE_DIM
        if state_len != config_dim:
            print(
                f"\n!!! ERROR: GameState.get_state() length ({state_len}) != EnvConfig.STATE_DIM ({config_dim}) !!!"
            )
            print(
                "Check state representation in environment/game_state.py and dimension in config.py."
            )
            sys.exit(1)
        print(f"GameState basic checks passed (State Dim: {state_len}).")
        del gs_test
    except NameError:
        print("ERROR: GameState class not found.")
        sys.exit(1)
    except ImportError as e:
        print(f"ERROR: Failed to import environment components: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: Error during GameState pre-check: {e}")
        sys.exit(1)

    app = MainApp()
    app.run()
