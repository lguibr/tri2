# File: ui/mcts_visualizer/node_renderer.py
import pygame
import math
from typing import Tuple, Optional, TYPE_CHECKING

from config import (
    VisConfig,
    WHITE,
    BLACK,
    RED,
    BLUE,
    YELLOW,
    GRAY,
    LIGHTG,
    CYAN,
    ORANGE,
)
from config.constants import (
    MCTS_NODE_WIN_COLOR,
    MCTS_NODE_LOSS_COLOR,
    MCTS_NODE_NEUTRAL_COLOR,
    MCTS_NODE_BORDER_COLOR,
    MCTS_NODE_SELECTED_BORDER_COLOR,
    MCTS_NODE_TEXT_COLOR,
    MCTS_NODE_PRIOR_COLOR,
    MCTS_NODE_SCORE_COLOR,
    MCTS_MINI_GRID_BG_COLOR,
    MCTS_MINI_GRID_LINE_COLOR,
    MCTS_MINI_GRID_OCCUPIED_COLOR,
    MCTS_EDGE_COLOR,
)
from mcts.node import MCTSNode

if TYPE_CHECKING:
    from ui.panels.game_area import GameAreaRenderer  # For type hinting


class MCTSNodeRenderer:
    """Renders a single MCTS node for visualization, including a mini-grid."""

    BASE_NODE_RADIUS = 25  # Increased base radius
    BASE_NODE_BORDER_WIDTH = 1
    BASE_FONT_SIZE = 10  # Smaller base font for more info
    MIN_NODE_RADIUS = 8
    MAX_NODE_RADIUS = 80
    MIN_FONT_SIZE = 6
    MAX_FONT_SIZE = 14
    GRID_RENDER_THRESHOLD_RADIUS = 15  # Only render grid if node radius is above this

    def __init__(self, screen: pygame.Surface, vis_config: VisConfig):
        self.screen = screen
        self.vis_config = vis_config
        self.font: Optional[pygame.font.Font] = None
        self.game_area_renderer: Optional["GameAreaRenderer"] = (
            None  # To render mini-grid
        )

    def set_game_area_renderer(self, renderer: "GameAreaRenderer"):
        """Sets the reference to the GameAreaRenderer."""
        self.game_area_renderer = renderer

    def _get_scaled_font(self, zoom: float) -> pygame.font.Font:
        """Gets a font scaled based on the zoom level."""
        scaled_size = int(self.BASE_FONT_SIZE * math.sqrt(zoom))
        clamped_size = max(self.MIN_FONT_SIZE, min(self.MAX_FONT_SIZE, scaled_size))
        try:
            # Consider caching fonts if performance becomes an issue
            return pygame.font.SysFont(None, clamped_size)
        except Exception:
            return pygame.font.Font(None, clamped_size)

    def _render_mini_grid(self, node: MCTSNode, surface: pygame.Surface):
        """Renders the game state grid onto the provided surface."""
        if not self.game_area_renderer:
            pygame.draw.line(surface, RED, (0, 0), surface.get_size(), 1)
            return

        # Use a simplified version of GameAreaRenderer's grid rendering
        try:
            padding = 1  # Minimal padding inside the node
            drawable_w = max(1, surface.get_width() - 2 * padding)
            drawable_h = max(1, surface.get_height() - 2 * padding)
            env_config = node.game_state.env_config  # Get config from node's state
            grid_rows, grid_cols_eff_width = (
                env_config.ROWS,
                env_config.COLS * 0.75 + 0.25,
            )
            if grid_rows <= 0 or grid_cols_eff_width <= 0:
                return

            scale_w = drawable_w / grid_cols_eff_width
            scale_h = drawable_h / grid_rows
            final_scale = min(scale_w, scale_h)
            if final_scale <= 0:
                return

            tri_cell_w, tri_cell_h = max(1, final_scale), max(1, final_scale)
            final_grid_pixel_w = grid_cols_eff_width * final_scale
            final_grid_pixel_h = grid_rows * final_scale
            grid_ox = padding + (drawable_w - final_grid_pixel_w) / 2
            grid_oy = padding + (drawable_h - final_grid_pixel_h) / 2

            surface.fill(MCTS_MINI_GRID_BG_COLOR)  # Background for the grid area

            grid = node.game_state.grid
            for r in range(grid.rows):
                for c in range(grid.cols):
                    if not (
                        0 <= r < len(grid.triangles) and 0 <= c < len(grid.triangles[r])
                    ):
                        continue
                    t = grid.triangles[r][c]
                    if not t.is_death and hasattr(t, "get_points"):
                        try:
                            pts = t.get_points(
                                ox=grid_ox,
                                oy=grid_oy,
                                cw=int(tri_cell_w),
                                ch=int(tri_cell_h),
                            )
                            color = MCTS_MINI_GRID_BG_COLOR  # Default empty color
                            if t.is_occupied:
                                # Use a simple bright color for occupied cells in mini-grid
                                color = MCTS_MINI_GRID_OCCUPIED_COLOR
                            pygame.draw.polygon(surface, color, pts)
                            # Draw subtle grid lines
                            pygame.draw.polygon(
                                surface, MCTS_MINI_GRID_LINE_COLOR, pts, 1
                            )
                        except Exception:
                            pass  # Ignore drawing errors for single triangles
        except Exception as e:
            print(f"Error rendering mini-grid: {e}")
            pygame.draw.line(surface, RED, (0, 0), surface.get_size(), 1)

    def render(
        self,
        node: MCTSNode,
        pos: Tuple[int, int],
        zoom: float,
        is_selected: bool = False,
    ):
        """Draws the node circle, mini-grid, and info, scaled by zoom."""
        self.font = self._get_scaled_font(zoom)
        if not self.font:
            return

        scaled_radius = int(self.BASE_NODE_RADIUS * zoom)
        node_radius = max(
            self.MIN_NODE_RADIUS, min(self.MAX_NODE_RADIUS, scaled_radius)
        )
        border_width = max(1, int(self.BASE_NODE_BORDER_WIDTH * zoom))

        value = node.mean_action_value
        if value > 0.1:
            color = MCTS_NODE_WIN_COLOR
        elif value < -0.1:
            color = MCTS_NODE_LOSS_COLOR
        else:
            color = MCTS_NODE_NEUTRAL_COLOR

        # Create surface for the node content (grid + border)
        node_diameter = node_radius * 2
        node_surface = pygame.Surface((node_diameter, node_diameter), pygame.SRCALPHA)
        node_surface.fill((0, 0, 0, 0))  # Transparent background

        # Render mini-grid if node is large enough
        if node_radius >= self.GRID_RENDER_THRESHOLD_RADIUS:
            grid_surface = pygame.Surface(
                (node_diameter, node_diameter), pygame.SRCALPHA
            )
            self._render_mini_grid(node, grid_surface)
            # Clip the grid to a circle
            pygame.draw.circle(
                grid_surface,
                (255, 255, 255, 0),
                (node_radius, node_radius),
                node_radius,
            )  # Transparent circle mask
            grid_surface.set_colorkey(
                (255, 255, 255, 0)
            )  # Make transparent area the colorkey
            node_surface.blit(grid_surface, (0, 0))
        else:
            # Draw solid color if too small for grid
            pygame.draw.circle(
                node_surface, color, (node_radius, node_radius), node_radius
            )

        # Draw border
        border_color = (
            MCTS_NODE_SELECTED_BORDER_COLOR if is_selected else MCTS_NODE_BORDER_COLOR
        )
        pygame.draw.circle(
            node_surface,
            border_color,
            (node_radius, node_radius),
            node_radius,
            border_width,
        )

        # Blit the node surface onto the main screen
        node_rect = node_surface.get_rect(center=pos)
        self.screen.blit(node_surface, node_rect)

        # Render text info below the node if radius is sufficient
        if node_radius > 10:
            visits_str = f"N:{node.visit_count}"
            value_str = f"Q:{value:.2f}"
            prior_str = f"P:{node.prior:.2f}"
            score_str = f"S:{node.game_state.game_score}"

            text_y_offset = node_rect.bottom + 2  # Start text below node rect
            line_height = self.font.get_linesize()

            visits_surf = self.font.render(visits_str, True, MCTS_NODE_TEXT_COLOR)
            value_surf = self.font.render(value_str, True, MCTS_NODE_TEXT_COLOR)
            prior_surf = self.font.render(prior_str, True, MCTS_NODE_PRIOR_COLOR)
            score_surf = self.font.render(score_str, True, MCTS_NODE_SCORE_COLOR)

            # Center text horizontally below the node
            self.screen.blit(
                visits_surf, visits_surf.get_rect(midtop=(pos[0], text_y_offset))
            )
            self.screen.blit(
                value_surf,
                value_surf.get_rect(midtop=(pos[0], text_y_offset + line_height)),
            )
            self.screen.blit(
                prior_surf,
                prior_surf.get_rect(midtop=(pos[0], text_y_offset + 2 * line_height)),
            )
            self.screen.blit(
                score_surf,
                score_surf.get_rect(midtop=(pos[0], text_y_offset + 3 * line_height)),
            )

    def draw_edge(
        self,
        parent_pos: Tuple[int, int],
        child_pos: Tuple[int, int],
        line_width: int = 1,
        color: Tuple[int, int, int] = MCTS_EDGE_COLOR,
    ):
        """Draws a line connecting parent and child nodes with variable width/color."""
        clamped_width = max(1, min(line_width, 5))
        pygame.draw.aaline(
            self.screen, color, parent_pos, child_pos
        )  # Use anti-aliased line
