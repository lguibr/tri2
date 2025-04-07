# File: environment/game_logic.py
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

    def _calculate_placement_reward(self, placed_shape: "Shape") -> float:
        return self.gs.rewards.REWARD_PLACE_PER_TRI * len(placed_shape.triangles)

    def _calculate_line_clear_reward(self, triangles_cleared: int) -> float:
        return triangles_cleared * self.gs.rewards.REWARD_PER_CLEARED_TRIANGLE

    def _calculate_state_penalty(self) -> float:
        penalty = 0.0
        max_height = self.gs.grid.get_max_height()
        bumpiness = self.gs.grid.get_bumpiness()
        num_holes = self.gs.grid.count_holes()
        penalty += max_height * self.gs.rewards.PENALTY_MAX_HEIGHT_FACTOR
        penalty += bumpiness * self.gs.rewards.PENALTY_BUMPINESS_FACTOR
        penalty += num_holes * self.gs.rewards.PENALTY_HOLE_PER_HOLE
        return penalty

    def _handle_invalid_placement(self) -> float:
        self.gs._last_action_valid = False
        return self.gs.rewards.PENALTY_INVALID_MOVE

    def _handle_game_over_state_change(self) -> float:
        if self.gs.game_over:
            return 0.0
        self.gs.game_over = True
        if self.gs.freeze_time <= 0:  # Only set freeze if not already frozen
            self.gs.freeze_time = 1.0
        self.gs.game_over_flash_time = 0.6
        return self.gs.rewards.PENALTY_GAME_OVER

    def _handle_valid_placement(
        self,
        shape_to_place: "Shape",
        shape_slot_index: int,
        target_row: int,
        target_col: int,
    ) -> float:
        self.gs._last_action_valid = True
        step_reward = 0.0

        step_reward += self._calculate_placement_reward(shape_to_place)
        holes_before = self.gs.grid.count_holes()

        self.gs.grid.place(shape_to_place, target_row, target_col)
        self.gs.shapes[shape_slot_index] = None
        self.gs.game_score += len(shape_to_place.triangles)
        self.gs.pieces_placed_this_episode += 1

        lines_cleared, triangles_cleared, cleared_coords = self.gs.grid.clear_lines()
        line_clear_reward = self._calculate_line_clear_reward(triangles_cleared)
        step_reward += line_clear_reward
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
                line_clear_reward,
            )

        holes_after = self.gs.grid.count_holes()
        new_holes_created = max(0, holes_after - holes_before)

        step_reward += self._calculate_state_penalty()
        step_reward += new_holes_created * self.gs.rewards.PENALTY_NEW_HOLE

        if all(s is None for s in self.gs.shapes):
            from .shape import Shape  # Local import to avoid cycle

            self.gs.shapes = [
                Shape() for _ in range(self.gs.env_config.NUM_SHAPE_SLOTS)
            ]

        if self._check_fundamental_game_over():
            step_reward += self._handle_game_over_state_change()

        self.gs.demo_logic.update_demo_selection_after_placement(shape_slot_index)
        return step_reward

    def step(self, action_index: int) -> Tuple[float, bool]:
        """Performs one game step based on the action index."""
        # Update timers at the very beginning of the step
        self.gs._update_timers()

        # Check game over state *after* timer update
        if self.gs.game_over:
            return (0.0, True)

        # Check if frozen *after* timer update
        if self.gs.is_frozen():
            # If frozen, still calculate potential change but return 0 extrinsic reward
            # print(f"[GameLogic] Step called while frozen ({self.gs.freeze_time:.3f}s left). Skipping action.") # DEBUG
            current_potential = self.gs.features.calculate_potential()
            pbrs_reward = (
                (self.gs.ppo_config.GAMMA * current_potential - self.gs._last_potential)
                if self.gs.rewards.ENABLE_PBRS
                else 0.0
            )
            self.gs._last_potential = current_potential
            total_reward = (
                self.gs.rewards.REWARD_ALIVE_STEP + pbrs_reward
            )  # Still give alive reward? Maybe not if frozen? Let's keep it for now.
            self.gs.score += total_reward
            return (
                total_reward,
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

        potential_before_action = self.gs.features.calculate_potential()

        if is_valid_placement:
            extrinsic_reward = self._handle_valid_placement(
                shape_to_place, shape_slot_index, target_row, target_col
            )
        else:
            # An invalid action was chosen (e.g., by the agent or debug click)
            # print(f"[GameLogic] Invalid placement attempt: Action {action_index} -> Slot {shape_slot_index}, Pos ({target_row},{target_col})") # DEBUG
            extrinsic_reward = self._handle_invalid_placement()
            # Check if *any* move is possible after this invalid attempt
            if self._check_fundamental_game_over():
                extrinsic_reward += self._handle_game_over_state_change()

        # Add alive reward only if the game didn't end *during* this step
        if not self.gs.game_over:
            extrinsic_reward += self.gs.rewards.REWARD_ALIVE_STEP

        potential_after_action = self.gs.features.calculate_potential()
        pbrs_reward = 0.0
        if self.gs.rewards.ENABLE_PBRS:
            pbrs_reward = (
                self.gs.ppo_config.GAMMA * potential_after_action
                - potential_before_action
            )

        total_reward = extrinsic_reward + pbrs_reward
        self.gs._last_potential = potential_after_action

        self.gs.score += total_reward
        return (total_reward, self.gs.game_over)
