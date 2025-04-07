# File: environment/game_demo_logic.py
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .game_state import GameState
    from .shape import Shape


class GameDemoLogic:
    """Handles logic specific to the interactive demo and debug modes."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def update_demo_selection_after_placement(self, placed_slot_index: int):
        """Selects the next available shape slot after placement in demo mode."""
        num_slots = self.gs.env_config.NUM_SHAPE_SLOTS
        if num_slots <= 0:
            return
        available_indices = [i for i, s in enumerate(self.gs.shapes) if s is not None]
        if not available_indices:
            self.gs.demo_selected_shape_idx = 0
        else:
            self.gs.demo_selected_shape_idx = available_indices[0]

    def select_shape_for_drag(self, shape_index: int):
        """Selects a shape to be dragged by the mouse."""
        if self.gs.game_over or self.gs.freeze_time > 0:
            return
        if (
            0 <= shape_index < len(self.gs.shapes)
            and self.gs.shapes[shape_index] is not None
        ):
            self.gs.demo_dragged_shape_idx = shape_index
            self.gs.demo_selected_shape_idx = shape_index
            self.gs.demo_snapped_position = None
            print(f"[Demo] Dragging shape index: {shape_index}")
        else:
            self.gs.demo_dragged_shape_idx = None
            print(f"[Demo] Invalid shape index {shape_index} or shape is None.")

    def deselect_dragged_shape(self):
        """Deselects the currently dragged shape."""
        if self.gs.demo_dragged_shape_idx is not None:
            print(f"[Demo] Deselected shape index: {self.gs.demo_dragged_shape_idx}")
            self.gs.demo_dragged_shape_idx = None
            self.gs.demo_snapped_position = None

    def update_snapped_position(self, grid_pos: Optional[Tuple[int, int]]):
        """Updates the snapped position if the dragged shape can be placed there."""
        if self.gs.demo_dragged_shape_idx is None:
            self.gs.demo_snapped_position = None
            return

        shape_to_check = self.gs.shapes[self.gs.demo_dragged_shape_idx]
        if shape_to_check is None:
            self.gs.demo_snapped_position = None
            return

        if grid_pos is not None:
            target_row, target_col = grid_pos
            if self.gs.grid.can_place(shape_to_check, target_row, target_col):
                if self.gs.demo_snapped_position != grid_pos:
                    self.gs.demo_snapped_position = grid_pos
            else:
                self.gs.demo_snapped_position = None
        else:
            self.gs.demo_snapped_position = None

    def place_dragged_shape(self) -> bool:
        """Attempts to place the currently dragged and snapped shape."""
        if self.gs.game_over or self.gs.freeze_time > 0:
            return False
        if (
            self.gs.demo_dragged_shape_idx is None
            or self.gs.demo_snapped_position is None
        ):
            print("[Demo] Cannot place: No shape dragged or not snapped.")
            return False

        shape_slot_index = self.gs.demo_dragged_shape_idx
        target_row, target_col = self.gs.demo_snapped_position
        shape_to_place = self.gs.shapes[shape_slot_index]

        if shape_to_place is not None and self.gs.grid.can_place(
            shape_to_place, target_row, target_col
        ):
            print(
                f"[Demo] Placing shape {shape_slot_index} at {target_row},{target_col}"
            )
            locations_per_shape = self.gs.grid.rows * self.gs.grid.cols
            action_index = shape_slot_index * locations_per_shape + (
                target_row * self.gs.grid.cols + target_col
            )
            _, _ = self.gs.logic.step(action_index)  # Use logic helper

            self.gs.demo_dragged_shape_idx = None
            self.gs.demo_snapped_position = None
            return True
        else:
            print(f"[Demo] Invalid placement attempt at {target_row},{target_col}")
            return False

    def get_dragged_shape_info(
        self,
    ) -> Tuple[Optional["Shape"], Optional[Tuple[int, int]]]:
        """Returns the currently dragged shape object and its snapped position."""
        if self.gs.demo_dragged_shape_idx is None:
            return None, None
        shape = self.gs.shapes[self.gs.demo_dragged_shape_idx]
        return shape, self.gs.demo_snapped_position

    def toggle_triangle_debug(self, row: int, col: int):
        """Toggles the state of a triangle for debugging and checks for lines."""
        if not self.gs.grid.valid(row, col):
            print(f"[Debug] Invalid coords: ({row}, {col})")
            return

        triangle = self.gs.grid.triangles[row][col]
        if triangle.is_death:
            print(f"[Debug] Cannot toggle death cell: ({row}, {col})")
            return

        triangle.is_occupied = not triangle.is_occupied
        if triangle.is_occupied:
            triangle.color = self.gs.vis_config.YELLOW
        else:
            triangle.color = None
        print(
            f"[Debug] Toggled ({row}, {col}) to {'Occupied' if triangle.is_occupied else 'Empty'}"
        )

        lines_cleared, triangles_cleared, cleared_coords = self.gs.grid.clear_lines()
        line_clear_reward = self.gs.logic._calculate_line_clear_reward(
            triangles_cleared
        )  # Use logic helper

        if triangles_cleared > 0:
            print(
                f"[Debug] Cleared {triangles_cleared} triangles in {lines_cleared} lines."
            )
            self.gs.game_score += triangles_cleared * 2
            self.gs.triangles_cleared_this_episode += triangles_cleared
            self.gs.blink_time = 0.5
            self.gs.freeze_time = 0.5
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.5
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                triangles_cleared,
                line_clear_reward,
            )
        else:
            if self.gs.line_clear_highlight_time <= 0:
                self.gs.cleared_triangles_coords = []
            self.gs.last_line_clear_info = None

        self.gs.game_over = False
        self.gs.game_over_flash_time = 0.0
