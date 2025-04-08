# File: environment/game_logic.py
import time
import numpy as np
from typing import TYPE_CHECKING, Tuple, List, Optional

if TYPE_CHECKING:
    from .game_state import GameState
    from .shape import Shape

# Constants for scoring (can be moved to config)
BASE_SCORE_PER_TRIANGLE = 0.01
LINE_CLEAR_BONUS_PER_LINE = 0.1
PERFECT_CLEAR_BONUS = 10.0


class GameLogic:
    """Handles the core game mechanics like placing shapes, clearing lines, scoring."""

    def __init__(self, game_state: "GameState"):
        self.gs = game_state
        self.env_config = game_state.env_config

    def step(self, action_index: int) -> Tuple[Optional[float], bool]:
        """
        Performs one game step based on the action index.
        Handles shape placement, line clearing, scoring, and game over checks.
        Visual effect timers are SET here but DO NOT cause delays.
        Returns (reward, is_game_over). Reward is currently always None.
        """
        # --- 1. Decode Action ---
        shape_slot_index, target_row, target_col = self.decode_action(action_index)
        shape_to_place = self.gs.shapes[shape_slot_index]

        # --- 2. Validate Action ---
        if shape_to_place is None or not self.gs.grid.can_place(
            shape_to_place, target_row, target_col
        ):
            # Invalid move: Game Over
            self.gs.game_over = True
            self.gs._last_action_valid = False
            # Set game over flash timer (for visuals only)
            self.gs.game_over_flash_time = 0.5
            return None, True  # No reward, game is over

        # --- 3. Place Shape ---
        self.gs.grid.place(shape_to_place, target_row, target_col)
        self.gs.shapes[shape_slot_index] = None  # Remove placed shape
        self.gs.pieces_placed_this_episode += 1
        self.gs._last_action_valid = True

        # --- 4. Clear Lines & Score ---
        lines_cleared, triangles_in_lines, cleared_coords = self.gs.grid.clear_lines()
        score_increase = 0.0
        if lines_cleared > 0:
            score_increase += triangles_in_lines * BASE_SCORE_PER_TRIANGLE
            score_increase += lines_cleared * LINE_CLEAR_BONUS_PER_LINE
            # Check for perfect clear (optional bonus)
            # if self._is_perfect_clear():
            #     score_increase += PERFECT_CLEAR_BONUS

            self.gs.game_score += score_increase
            self.gs.triangles_cleared_this_episode += triangles_in_lines
            # Set visual effect timers (these won't block execution)
            self.gs.line_clear_flash_time = 0.3
            self.gs.line_clear_highlight_time = 0.5
            self.gs.cleared_triangles_coords = cleared_coords
            self.gs.last_line_clear_info = (
                lines_cleared,
                triangles_in_lines,
                score_increase,
            )

        # --- 5. Refill Shapes ---
        self._refill_shapes()

        # --- 6. Check Game Over ---
        if self._check_game_over_condition():
            self.gs.game_over = True
            # Set game over flash timer (for visuals only)
            self.gs.game_over_flash_time = 0.5
            return None, True  # No reward, game is over

        # --- 7. Return State ---
        # Reward is None as it's handled by outcome in AlphaZero
        return None, False  # No reward, game not over

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        actions_per_shape = self.env_config.ROWS * self.env_config.COLS
        shape_slot_index = action_index // actions_per_shape
        placement_index = action_index % actions_per_shape
        target_row = placement_index // self.env_config.COLS
        target_col = placement_index % self.env_config.COLS
        return shape_slot_index, target_row, target_col

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        valid_action_indices = []
        actions_per_shape = self.env_config.ROWS * self.env_config.COLS
        for shape_slot_index, shape in enumerate(self.gs.shapes):
            if shape is None:
                continue  # Skip empty slots
            for r in range(self.env_config.ROWS):
                for c in range(self.env_config.COLS):
                    if self.gs.grid.can_place(shape, r, c):
                        action_index = (
                            shape_slot_index * actions_per_shape
                            + r * self.env_config.COLS
                            + c
                        )
                        valid_action_indices.append(action_index)
        return valid_action_indices

    def _refill_shapes(self):
        """Refills empty shape slots with new random shapes."""
        from .shape import Shape  # Local import to avoid potential cycles

        for i in range(len(self.gs.shapes)):
            if self.gs.shapes[i] is None:
                self.gs.shapes[i] = Shape()

    def _check_game_over_condition(self) -> bool:
        """Checks if the game is over (no valid moves for any available shape)."""
        # Check if any shape can be placed anywhere
        for shape_slot_index, shape in enumerate(self.gs.shapes):
            if shape is None:
                continue
            for r in range(self.env_config.ROWS):
                for c in range(self.env_config.COLS):
                    if self.gs.grid.can_place(shape, r, c):
                        return False  # Found a valid move
        return True  # No valid moves found for any shape

    def _is_perfect_clear(self) -> bool:
        """Checks if the grid is completely empty."""
        for r in range(self.gs.grid.rows):
            for c in range(self.gs.grid.cols):
                tri = self.gs.grid.triangles[r][c]
                if not tri.is_death and tri.is_occupied:
                    return False
        return True
