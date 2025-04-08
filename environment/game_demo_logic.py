# File: environment/game_demo_logic.py
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .game_state import GameState
    from .shape import Shape


class GameDemoLogic:
    """Handles logic specific to the interactive Demo/Debug mode."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def select_shape_for_drag(self, shape_index: int):
        """Selects a shape for dragging if available."""
        if (
            0 <= shape_index < len(self.gs.shapes)
            and self.gs.shapes[shape_index] is not None
        ):
            # If clicking the already selected shape, deselect it
            if self.gs.demo_selected_shape_idx == shape_index:
                self.gs.demo_selected_shape_idx = -1  # Indicate no selection
                self.gs.demo_dragged_shape_idx = None
                self.gs.demo_snapped_position = None
            else:
                self.gs.demo_selected_shape_idx = shape_index
                self.gs.demo_dragged_shape_idx = shape_index
                self.gs.demo_snapped_position = None  # Reset snap on new selection

    def deselect_dragged_shape(self):
        """Deselects any currently dragged shape."""
        self.gs.demo_dragged_shape_idx = None
        self.gs.demo_snapped_position = None
        # Keep demo_selected_shape_idx as is, maybe user wants to re-drag later

    def update_snapped_position(self, grid_pos: Optional[Tuple[int, int]]):
        """Updates the snapped position based on grid hover."""
        if self.gs.demo_dragged_shape_idx is None:
            self.gs.demo_snapped_position = None
            return

        shape_to_drag = self.gs.shapes[self.gs.demo_dragged_shape_idx]
        if shape_to_drag is None:
            self.gs.demo_snapped_position = None
            return

        if grid_pos is not None and self.gs.grid.can_place(
            shape_to_drag, grid_pos[0], grid_pos[1]
        ):
            self.gs.demo_snapped_position = grid_pos
        else:
            self.gs.demo_snapped_position = None

    def place_dragged_shape(self) -> bool:
        """Attempts to place the currently dragged shape at the snapped position."""
        if (
            self.gs.demo_dragged_shape_idx is not None
            and self.gs.demo_snapped_position is not None
        ):
            shape_idx = self.gs.demo_dragged_shape_idx
            r, c = self.gs.demo_snapped_position

            # Encode the action based on the demo state
            action_index = self.gs.logic.encode_action(shape_idx, r, c)

            # Use the core game logic step function
            _, done = self.gs.logic.step(action_index)

            # Reset demo drag state after placement attempt
            self.gs.demo_dragged_shape_idx = None
            self.gs.demo_snapped_position = None
            self.gs.demo_selected_shape_idx = -1  # Deselect after placement

            return not done  # Return True if placement was successful (game not over)
        return False

    def get_dragged_shape_info(
        self,
    ) -> Tuple[Optional["Shape"], Optional[Tuple[int, int]]]:
        """Returns the currently dragged shape and its snapped position."""
        if self.gs.demo_dragged_shape_idx is None:
            return None, None
        shape = self.gs.shapes[self.gs.demo_dragged_shape_idx]
        return shape, self.gs.demo_snapped_position

    def toggle_triangle_debug(self, row: int, col: int):
        """Toggles the occupied state of a triangle in debug mode and checks for line clears."""
        if not self.gs.grid.valid(row, col):
            return

        tri = self.gs.grid.triangles[row][col]
        if tri.is_death:
            return  # Cannot toggle death cells

        # Toggle state
        tri.is_occupied = not tri.is_occupied
        self.gs.grid._occupied_np[row, col] = tri.is_occupied  # Update numpy array
        tri.color = self.gs.vis_config.WHITE if tri.is_occupied else None

        # Manually trigger line clear check for the toggled triangle
        toggled_triangle_set = {tri}
        lines_cleared, tris_cleared, cleared_coords = self.gs.grid.clear_lines(
            newly_occupied_triangles=toggled_triangle_set  # Pass the toggled tri
        )

        if lines_cleared > 0:
            # Update score and visual timers if lines were cleared
            self.gs.triangles_cleared_this_episode += tris_cleared
            score_increase = (lines_cleared**2) * 10 + tris_cleared
            self.gs.game_score += score_increase
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.6
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                tris_cleared,
                score_increase,
            )
            print(f"[Debug] Cleared {lines_cleared} lines ({tris_cleared} tris).")
