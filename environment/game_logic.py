# File: environment/game_logic.py
import numpy as np
import time
import copy
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from utils.types import ActionType, StateType

from .grid import Grid
from .shape import Shape

if TYPE_CHECKING:
    from .game_state import GameState

ActionType = int


class GameLogic:
    """Handles core game logic like piece placement, line clearing, and game over checks."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state
        self.env_config = game_state.env_config

    def _check_game_over(self) -> bool:
        """Checks if any placement is possible for any available shape."""
        if self.gs.game_over:  # Already over
            return True
        if not any(
            s is not None for s in self.gs.shapes
        ):  # No shapes left is not game over
            return False
        if not self.valid_actions():  # Check if any valid actions exist
            self.gs.game_over = True
            self.gs.game_over_flash_time = 1.0  # Trigger flash
            return True
        return False

    def _check_and_handle_line_clears(self) -> Tuple[int, int, float]:
        """Checks for and clears lines, applies score/effects. Returns lines, tris, score_delta."""
        lines_cleared, tris_cleared, cleared_coords = self.gs.grid.clear_lines()
        score_delta = 0.0

        if lines_cleared > 0:
            # --- Scoring ---
            # Simple scoring: base points per triangle + bonus per line
            base_score_per_tri = 0.1
            bonus_per_line = 1.0
            score_delta = (tris_cleared * base_score_per_tri) + (
                lines_cleared * bonus_per_line
            )
            # Apply multiplier for multi-line clears
            if lines_cleared > 1:
                score_delta *= 1.5 ** (lines_cleared - 1)

            self.gs.game_score += int(score_delta * 10)  # Example: scale score
            self.gs.triangles_cleared_this_episode += tris_cleared

            # --- Visual Timers ---
            self.gs.line_clear_flash_time = 0.4  # Duration of background flash
            self.gs.line_clear_highlight_time = 0.6  # Duration of yellow highlight
            # Only freeze if lines were cleared
            self.gs.freeze_time = 0.1  # Short freeze after line clear
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (lines_cleared, tris_cleared, score_delta)

        return lines_cleared, tris_cleared, score_delta

    def encode_action(self, shape_slot_index: int, row: int, col: int) -> ActionType:
        """Encodes a placement action into a single integer index."""
        shape_offset = shape_slot_index * (self.env_config.ROWS * self.env_config.COLS)
        grid_offset = row * self.env_config.COLS + col
        return shape_offset + grid_offset

    def decode_action(self, action_index: ActionType) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot_index, row, col)."""
        grid_size = self.env_config.ROWS * self.env_config.COLS
        shape_slot_index = action_index // grid_size
        grid_offset = action_index % grid_size
        row = grid_offset // self.env_config.COLS
        col = grid_offset % self.env_config.COLS
        return shape_slot_index, row, col

    def valid_actions(self) -> List[ActionType]:
        """Returns a list of valid action indices for the current state."""
        valid_action_indices = []
        for shape_slot_index, shape in enumerate(self.gs.shapes):
            if shape is None:
                continue  # Skip empty slots
            for r in range(self.env_config.ROWS):
                for c in range(self.env_config.COLS):
                    if self.gs.grid.can_place(shape, r, c):
                        action_index = self.encode_action(shape_slot_index, r, c)
                        valid_action_indices.append(action_index)
        return valid_action_indices

    def step(self, action_index: ActionType) -> Tuple[StateType, bool]:
        """
        Performs one game step based on the action index.
        Returns (new_state, is_game_over).
        """
        if self.gs.is_frozen() or self.gs.is_over():
            # If frozen or game over, no action is taken, return current state
            # Update timers even if frozen to allow unfreezing
            self.gs._update_timers()
            return self.gs.get_state(), self.gs.is_over()

        shape_slot_index, target_row, target_col = self.decode_action(action_index)

        # --- Validate Action ---
        shape_to_place = (
            self.gs.shapes[shape_slot_index]
            if shape_slot_index < len(self.gs.shapes)
            else None
        )
        can_place = shape_to_place is not None and self.gs.grid.can_place(
            shape_to_place, target_row, target_col
        )

        self.gs._last_action_valid = can_place

        if not can_place:
            # Invalid action - perhaps apply penalty or just do nothing
            # For AlphaZero, MCTS should only explore valid actions,
            # but this handles unexpected calls.
            self.gs.blink_time = 0.2  # Short blink for invalid move attempt
            # Update timers and check for game over (in case this was the only possibility)
            self.gs._update_timers()
            self._check_game_over()
            return self.gs.get_state(), self.gs.is_over()

        # --- Execute Valid Action ---
        # 1. Place the shape
        self.gs.grid.place(shape_to_place, target_row, target_col)
        self.gs.pieces_placed_this_episode += 1

        # 2. Remove shape from available shapes and refill if necessary
        self.gs.shapes[shape_slot_index] = None
        if all(s is None for s in self.gs.shapes):
            self.gs.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]

        # 3. Check for and handle line clears
        self._check_and_handle_line_clears()

        # 4. Check for game over condition
        self._check_game_over()

        # 5. Update timers (handles freeze, blink, flashes)
        self.gs._update_timers()

        # 6. Return new state and game over status
        return self.gs.get_state(), self.gs.is_over()

    def get_last_action_validity(self) -> bool:
        """Returns whether the last attempted action was valid."""
        return self.gs._last_action_valid
