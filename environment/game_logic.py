# File: environment/game_logic.py
import random
from typing import TYPE_CHECKING, List, Tuple, Optional, Set

from .shape import Shape
from .triangle import Triangle

if TYPE_CHECKING:
    from .game_state import GameState


class GameLogic:
    """Handles core game mechanics like placing shapes, clearing lines, and game over conditions."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state

    def step(self, action_index: int) -> Tuple[Optional[float], bool]:
        """
        Performs one game step based on the action index.
        Returns (reward, is_game_over). Reward is currently always None.
        """
        shape_slot_index, target_row, target_col = self.decode_action(action_index)

        # Validate shape index and shape existence
        if not (0 <= shape_slot_index < len(self.gs.shapes)):
            self.gs._last_action_valid = False
            self.gs.game_over = True
            self.gs.game_over_flash_time = 0.5
            return None, True
        shape_to_place = self.gs.shapes[shape_slot_index]

        if shape_to_place is None or not self.gs.grid.can_place(
            shape_to_place, target_row, target_col
        ):
            self.gs._last_action_valid = False
            self.gs.game_over = True
            self.gs.game_over_flash_time = 0.5  # Visual effect timer
            return None, True

        # Place the shape and get the set of newly occupied triangles
        newly_occupied = self.gs.grid.place(shape_to_place, target_row, target_col)
        self.gs.pieces_placed_this_episode += 1
        self.gs.shapes[shape_slot_index] = None  # Remove shape from slot

        # Clear lines using the optimized method
        lines_cleared, tris_cleared, cleared_coords = self.gs.grid.clear_lines(
            newly_occupied_triangles=newly_occupied
        )

        if lines_cleared > 0:
            self.gs.triangles_cleared_this_episode += tris_cleared
            # Update score (simple scoring for now)
            score_increase = (lines_cleared**2) * 10 + tris_cleared
            self.gs.game_score += score_increase
            # Set visual effect timers
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.6
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                tris_cleared,
                score_increase,
            )

        # Refill shape slots if needed (now checks if all are empty)
        self._refill_shape_slots()

        # Check game over condition (if any slot cannot place its shape)
        # This check needs to happen *after* potential refill
        if self._check_game_over():
            self.gs.game_over = True
            self.gs.game_over_flash_time = 0.5  # Visual effect timer
            return None, True

        self.gs._last_action_valid = True
        return None, False  # Reward is None, game not over

    def _refill_shape_slots(self):
        """Refills shape slots with new random shapes ONLY if all slots are empty."""
        # Check if all slots are currently empty (None)
        if all(s is None for s in self.gs.shapes):
            num_slots = self.gs.env_config.NUM_SHAPE_SLOTS
            self.gs.shapes = [Shape() for _ in range(num_slots)]

    def _check_game_over(self) -> bool:
        """Checks if the game is over because no available shape can be placed."""
        # If all slots are empty, the game is definitely not over yet (wait for refill)
        if all(s is None for s in self.gs.shapes):
            return False

        for shape in self.gs.shapes:
            if shape is not None:
                # Check if this shape has at least one valid placement anywhere
                for r in range(self.gs.grid.rows):
                    for c in range(self.gs.grid.cols):
                        if self.gs.grid.can_place(shape, r, c):
                            return False  # Found a valid move, game not over
        # If we get here, it means there was at least one shape, but none could be placed
        return True

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        valid_action_indices = []
        for shape_slot_index, shape in enumerate(self.gs.shapes):
            if shape is None:
                continue
            for r in range(self.gs.grid.rows):
                for c in range(self.gs.grid.cols):
                    if self.gs.grid.can_place(shape, r, c):
                        action_index = self.encode_action(shape_slot_index, r, c)
                        valid_action_indices.append(action_index)
        return valid_action_indices

    def encode_action(self, shape_slot_index: int, row: int, col: int) -> int:
        """Encodes (shape_slot, row, col) into a single action index."""
        return (
            shape_slot_index * (self.gs.grid.rows * self.gs.grid.cols)
            + row * self.gs.grid.cols
            + col
        )

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        grid_size = self.gs.grid.rows * self.gs.grid.cols
        shape_slot_index = action_index // grid_size
        remainder = action_index % grid_size
        row = remainder // self.gs.grid.cols
        col = remainder % self.gs.grid.cols
        return shape_slot_index, row, col
