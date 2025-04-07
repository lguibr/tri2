from typing import TYPE_CHECKING, List, Tuple, Optional
import time

if TYPE_CHECKING:
    from .game_state import GameState
    from .shape import Shape


class GameLogic:
    """Handles the core game mechanics like stepping, placement, and line clearing."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        # Note: GameState.valid_actions() now checks is_frozen() first.
        # This method assumes the game is not frozen when called.
        if self.gs.game_over:  # Still check game_over here
            return []

        valid_action_indices: List[int] = []
        locations_per_shape = self.gs.grid.rows * self.gs.grid.cols
        for shape_slot_index, current_shape in enumerate(self.gs.shapes):
            if not current_shape:
                continue
            for target_row in range(self.gs.grid.rows):
                for target_col in range(self.gs.grid.cols):
                    if self.gs.grid.can_place(current_shape, target_row, target_col):
                        action_index = shape_slot_index * locations_per_shape + (
                            target_row * self.gs.grid.cols + target_col
                        )
                        valid_action_indices.append(action_index)
        return valid_action_indices

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        locations_per_shape = self.gs.grid.rows * self.gs.grid.cols
        shape_slot_index = action_index // locations_per_shape
        position_index = action_index % locations_per_shape
        target_row = position_index // self.gs.grid.cols
        target_col = position_index % self.gs.grid.cols
        return (shape_slot_index, target_row, target_col)

    def _check_fundamental_game_over(self) -> bool:
        """Checks if any available shape can be placed anywhere."""
        for current_shape in self.gs.shapes:
            if not current_shape:
                continue
            for target_row in range(self.gs.grid.rows):
                for target_col in range(self.gs.grid.cols):
                    if self.gs.grid.can_place(current_shape, target_row, target_col):
                        return False
        return True

    # Removed _calculate_placement_reward
    # Removed _calculate_line_clear_reward
    # Removed _calculate_state_penalty

    def _handle_invalid_placement(self):
        """Handles the state change for an invalid placement attempt."""
        self.gs._last_action_valid = False
        # No reward returned

    def _handle_game_over_state_change(self):
        """Handles the state change when the game ends."""
        if self.gs.game_over:
            return
        self.gs.game_over = True
        if self.gs.freeze_time <= 0:  # Only set freeze if not already frozen
            self.gs.freeze_time = 1.0
        self.gs.game_over_flash_time = 0.6
        # No reward returned

    def _handle_valid_placement(
        self,
        shape_to_place: "Shape",
        shape_slot_index: int,
        target_row: int,
        target_col: int,
    ):
        """Handles the state change for a valid placement."""
        self.gs._last_action_valid = True

        self.gs.grid.place(shape_to_place, target_row, target_col)
        self.gs.shapes[shape_slot_index] = None
        self.gs.game_score += len(shape_to_place.triangles)
        self.gs.pieces_placed_this_episode += 1

        lines_cleared, triangles_cleared, cleared_coords = self.gs.grid.clear_lines()
        # Removed line clear reward calculation
        self.gs.triangles_cleared_this_episode += triangles_cleared

        if triangles_cleared > 0:
            self.gs.game_score += triangles_cleared * 2
            self.gs.blink_time = 0.5
            self.gs.freeze_time = 0.5  # Set freeze time for animation
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.5
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                triangles_cleared,
                0.0,
            )  # Reward is 0

        # Removed state penalty calculation
        # Removed new hole penalty calculation

        if all(s is None for s in self.gs.shapes):
            from .shape import Shape  # Local import to avoid cycle

            self.gs.shapes = [
                Shape() for _ in range(self.gs.env_config.NUM_SHAPE_SLOTS)
            ]

        if self._check_fundamental_game_over():
            self._handle_game_over_state_change()

        self.gs.demo_logic.update_demo_selection_after_placement(shape_slot_index)
        # No reward returned

    def step(self, action_index: int) -> Tuple[Optional[dict], bool]:
        """
        Performs one game step based on the action index.
        Updates the internal game state and returns (None, is_game_over).
        The state representation should be fetched separately via get_state().
        """
        # Update timers at the very beginning of the step
        self.gs._update_timers()

        # Check game over state *after* timer update
        if self.gs.game_over:
            return (None, True)

        # Check if frozen *after* timer update
        if self.gs.is_frozen():
            # print(f"[GameLogic] Step called while frozen ({self.gs.freeze_time:.3f}s left). Skipping action.") # DEBUG
            return (
                None,
                False,
            )  # Return False for done, as game is just paused

        # --- If not frozen and not game over, proceed with action ---
        shape_slot_index, target_row, target_col = self.decode_action(action_index)

        shape_to_place = (
            self.gs.shapes[shape_slot_index]
            if 0 <= shape_slot_index < len(self.gs.shapes)
            else None
        )
        # Check if the specific action is valid (placement possible)
        is_valid_placement = shape_to_place is not None and self.gs.grid.can_place(
            shape_to_place, target_row, target_col
        )

        # Removed potential calculation

        if is_valid_placement:
            self._handle_valid_placement(
                shape_to_place, shape_slot_index, target_row, target_col
            )
        else:
            # An invalid action was chosen (e.g., by the agent or debug click)
            # print(f"[GameLogic] Invalid placement attempt: Action {action_index} -> Slot {shape_slot_index}, Pos ({target_row},{target_col})") # DEBUG
            self._handle_invalid_placement()
            # Check if *any* move is possible after this invalid attempt
            if self._check_fundamental_game_over():
                self._handle_game_over_state_change()

        return (None, self.gs.game_over)
