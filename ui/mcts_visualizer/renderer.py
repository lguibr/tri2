# File: ui/mcts_visualizer/renderer.py
import pygame
import math
from typing import Optional, Dict, Tuple, TYPE_CHECKING

from config import VisConfig, BLACK, WHITE, GRAY, YELLOW
from config.constants import (
    MCTS_INFO_TEXT_COLOR,
    MCTS_EDGE_COLOR,
    MCTS_EDGE_HIGHLIGHT_COLOR,
)
from mcts.node import MCTSNode
from .node_renderer import MCTSNodeRenderer
from .tree_layout import TreeLayout

if TYPE_CHECKING:
    from ui.panels.game_area import GameAreaRenderer


class MCTSVisualizer:
    """Renders the MCTS tree visualization with pan and zoom."""

    MIN_ZOOM = 0.1
    MAX_ZOOM = 5.0
    EDGE_HIGHLIGHT_THRESHOLD = 0.7  # Fraction of max visits to highlight edge

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: VisConfig,
        fonts: Dict[str, pygame.font.Font],
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.fonts = fonts
        self.node_renderer = MCTSNodeRenderer(screen, vis_config)
        self.info_font = fonts.get("ui", pygame.font.Font(None, 24))

        self.camera_offset_x = 0
        self.camera_offset_y = 0
        self.zoom_level = 1.0

        self.layout: Optional[TreeLayout] = None
        self.positions: Dict[MCTSNode, Tuple[int, int]] = {}

    def set_game_area_renderer(self, renderer: "GameAreaRenderer"):
        """Provides the GameAreaRenderer to the NodeRenderer for mini-grid drawing."""
        self.node_renderer.set_game_area_renderer(renderer)

    def reset_camera(self):
        """Resets camera pan and zoom."""
        self.camera_offset_x = 0
        self.camera_offset_y = 0
        self.zoom_level = 1.0
        print("MCTS Camera Reset")

    def pan_camera(self, delta_x: int, delta_y: int):
        """Pans the camera view."""
        self.camera_offset_x += delta_x
        self.camera_offset_y += delta_y

    def zoom_camera(self, factor: float, mouse_pos: Tuple[int, int]):
        """Zooms the camera view towards/away from the mouse position."""
        old_zoom = self.zoom_level
        self.zoom_level *= factor
        self.zoom_level = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.zoom_level))
        zoom_change = self.zoom_level / old_zoom

        world_mouse_x = (mouse_pos[0] - self.camera_offset_x) / old_zoom
        world_mouse_y = (mouse_pos[1] - self.camera_offset_y) / old_zoom

        new_offset_x = mouse_pos[0] - world_mouse_x * self.zoom_level
        new_offset_y = mouse_pos[1] - world_mouse_y * self.zoom_level

        self.camera_offset_x = new_offset_x
        self.camera_offset_y = new_offset_y

    def _world_to_screen(self, world_x: int, world_y: int) -> Tuple[int, int]:
        """Converts world coordinates (from layout) to screen coordinates."""
        screen_x = int(world_x * self.zoom_level + self.camera_offset_x)
        screen_y = int(world_y * self.zoom_level + self.camera_offset_y)
        return screen_x, screen_y

    def render(self, root_node: Optional[MCTSNode]):
        """Draws the MCTS tree and related info, applying camera transforms."""
        self.screen.fill(BLACK)

        if root_node is None:
            self._render_message("No MCTS data available.")
            return
        if not root_node.children and root_node.is_terminal:
            self._render_message("Root node is terminal.")
            pos = self._world_to_screen(self.screen.get_width() // 2, 100)
            self.node_renderer.render(root_node, pos, self.zoom_level, is_selected=True)
            self._render_info(root_node)
            return
        if not root_node.children and not root_node.is_expanded:
            self._render_message("MCTS Root not expanded (0 simulations?).")
            return

        if self.layout is None or self.layout.root != root_node:
            self.layout = TreeLayout(
                root_node, self.screen.get_width(), self.screen.get_height()
            )
            self.positions = self.layout.calculate_layout()

        max_child_visits = 0
        best_child_node: Optional[MCTSNode] = None
        if root_node.children:
            try:
                # Find best child based on visits for highlighting
                best_child_node = max(
                    root_node.children.values(), key=lambda n: n.visit_count
                )
                max_child_visits = best_child_node.visit_count
            except ValueError:  # Handle empty children dict case
                max_child_visits = 0
                best_child_node = None

        # Render edges first
        edges_to_render = []
        for node, world_pos in self.positions.items():
            if node.parent and node.parent in self.positions:
                parent_world_pos = self.positions[node.parent]
                parent_screen_pos = self._world_to_screen(*parent_world_pos)
                child_screen_pos = self._world_to_screen(*world_pos)

                line_width = 1
                edge_color = MCTS_EDGE_COLOR
                is_best_edge = False

                # Highlight edge from root to best child (based on visits)
                if (
                    node.parent == root_node
                    and node == best_child_node
                    and max_child_visits > 0
                ):
                    line_width = 3
                    edge_color = MCTS_EDGE_HIGHLIGHT_COLOR
                    is_best_edge = True

                edges_to_render.append(
                    (
                        (parent_screen_pos, child_screen_pos, line_width, edge_color),
                        is_best_edge,
                    )
                )

        # Sort edges to draw non-highlighted ones first
        edges_to_render.sort(
            key=lambda x: x[1]
        )  # False (non-best) comes before True (best)

        for edge_params, _ in edges_to_render:
            self.node_renderer.draw_edge(*edge_params)

        # Render nodes on top
        for node, world_pos in self.positions.items():
            screen_pos = self._world_to_screen(*world_pos)
            # Basic visibility check (culling) - expand bounds slightly
            render_radius = int(MCTSNodeRenderer.MAX_NODE_RADIUS * self.zoom_level)
            if (
                -render_radius < screen_pos[0] < self.screen.get_width() + render_radius
                and -render_radius
                < screen_pos[1]
                < self.screen.get_height() + render_radius
            ):
                self.node_renderer.render(
                    node, screen_pos, self.zoom_level, is_selected=(node == root_node)
                )

        self._render_info(root_node)

    def _render_message(self, message: str):
        """Displays a message centered on the screen."""
        if not self.info_font:
            return
        text_surf = self.info_font.render(message, True, WHITE)
        text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
        self.screen.blit(text_surf, text_rect)

    def _render_info(self, root_node: MCTSNode):
        """Displays information about the MCTS search and controls."""
        if not self.info_font:
            return
        sims = root_node.visit_count
        info_text = f"MCTS | Sims: {sims} | Zoom: {self.zoom_level:.2f}x | Drag=Pan | Scroll=Zoom | ESC=Exit"
        text_surf = self.info_font.render(info_text, True, MCTS_INFO_TEXT_COLOR)
        self.screen.blit(text_surf, (10, 10))

        if root_node.children:
            try:
                best_action_visits_node = max(
                    root_node.children.values(), key=lambda n: n.visit_count
                )
                best_action_visits = best_action_visits_node.action_taken

                best_action_q_node = max(
                    root_node.children.values(), key=lambda n: n.mean_action_value
                )
                best_action_q = best_action_q_node.action_taken

                best_action_text = f"Best Action (Visits): {best_action_visits} | Best Action (Q-Value): {best_action_q}"
                action_surf = self.info_font.render(best_action_text, True, YELLOW)
                self.screen.blit(action_surf, (10, 10 + self.info_font.get_linesize()))
            except ValueError:
                pass
