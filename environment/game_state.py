# File: environment/game_state.py
import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Union, Deque
import copy

# --- MOVED IMPORT TO TOP LEVEL ---
from config import EnvConfig, RewardConfig, PPOConfig

# --- END MOVED IMPORT ---

from .grid import Grid
from .shape import Shape

StateType = Dict[str, np.ndarray]


class GameState:
    """Represents the state of a single game instance."""

    def __init__(self):
        # Config instances are now available due to top-level import
        self.env_config = EnvConfig()
        self.rewards = RewardConfig()
        self.ppo_config = PPOConfig()  # Needed for gamma in PBRS

        self.grid = Grid(self.env_config)  # Pass env_config to Grid
        self.shapes: List[Optional[Shape]] = []
        self.score: float = 0.0  # Cumulative RL score for the episode
        self.game_score: int = 0  # In-game score metric
        self.lines_cleared_this_episode: int = 0
        self.pieces_placed_this_episode: int = 0

        # Timers for visual effects
        self.blink_time: float = 0.0
        self.last_time: float = time.time()
        self.freeze_time: float = 0.0
        self.line_clear_flash_time: float = 0.0
        self.line_clear_highlight_time: float = 0.0
        self.game_over_flash_time: float = 0.0
        self.cleared_triangles_coords: List[Tuple[int, int]] = []

        self.game_over: bool = False
        self._last_action_valid: bool = True

        # Demo mode state
        self.demo_selected_shape_idx: int = 0
        self.demo_target_row: int = self.env_config.ROWS // 2
        self.demo_target_col: int = self.env_config.COLS // 2

        # State for PBRS
        self._last_potential: float = 0.0

        self.reset()

    def reset(self) -> StateType:
        """Resets the game to its initial state."""
        self.grid = Grid(self.env_config)  # Re-create grid object
        self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]
        self.score = 0.0
        self.game_score = 0
        self.lines_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0

        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.line_clear_flash_time = 0.0
        self.line_clear_highlight_time = 0.0
        self.game_over_flash_time = 0.0
        self.cleared_triangles_coords = []

        self.game_over = False
        self._last_action_valid = True
        self.last_time = time.time()

        self.demo_selected_shape_idx = 0
        self.demo_target_row = self.env_config.ROWS // 2
        self.demo_target_col = self.env_config.COLS // 2

        self._last_potential = self._calculate_potential()

        return self.get_state()

    def _calculate_potential(self) -> float:
        """Calculates the potential function based on current grid state for PBRS."""
        if not self.rewards.ENABLE_PBRS:
            return 0.0

        potential = 0.0
        max_height = self.grid.get_max_height()
        num_holes = self.grid.count_holes()
        bumpiness = self.grid.get_bumpiness()

        potential += self.rewards.PBRS_HEIGHT_COEF * max_height
        potential += self.rewards.PBRS_HOLE_COEF * num_holes
        potential += self.rewards.PBRS_BUMPINESS_COEF * bumpiness

        return potential

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        if self.game_over or self.freeze_time > 0:
            return []
        valid_action_indices: List[int] = []
        locations_per_shape = self.grid.rows * self.grid.cols
        for shape_slot_index, current_shape in enumerate(self.shapes):
            if not current_shape:
                continue
            for target_row in range(self.grid.rows):
                for target_col in range(self.grid.cols):
                    if self.grid.can_place(current_shape, target_row, target_col):
                        action_index = shape_slot_index * locations_per_shape + (
                            target_row * self.grid.cols + target_col
                        )
                        valid_action_indices.append(action_index)
        return valid_action_indices

    def _check_fundamental_game_over(self) -> bool:
        """Checks if any available shape can be placed anywhere."""
        for current_shape in self.shapes:
            if not current_shape:
                continue
            for target_row in range(self.grid.rows):
                for target_col in range(self.grid.cols):
                    if self.grid.can_place(current_shape, target_row, target_col):
                        return False  # Found a valid placement
        return True  # No shape can be placed anywhere

    def is_over(self) -> bool:
        return self.game_over

    def is_frozen(self) -> bool:
        return self.freeze_time > 0

    def is_line_clearing(self) -> bool:
        return self.line_clear_flash_time > 0

    def is_highlighting_cleared(self) -> bool:
        return self.line_clear_highlight_time > 0

    def is_game_over_flashing(self) -> bool:
        return self.game_over_flash_time > 0

    def is_blinking(self) -> bool:
        return self.blink_time > 0

    def get_cleared_triangle_coords(self) -> List[Tuple[int, int]]:
        return self.cleared_triangles_coords

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        locations_per_shape = self.grid.rows * self.grid.cols
        shape_slot_index = action_index // locations_per_shape
        position_index = action_index % locations_per_shape
        target_row = position_index // self.grid.cols
        target_col = position_index % self.grid.cols
        return (shape_slot_index, target_row, target_col)

    def _update_timers(self):
        """Updates timers for visual effects."""
        now = time.time()
        delta_time = now - self.last_time
        self.last_time = now
        self.freeze_time = max(0, self.freeze_time - delta_time)
        self.blink_time = max(0, self.blink_time - delta_time)
        self.line_clear_flash_time = max(0, self.line_clear_flash_time - delta_time)
        self.line_clear_highlight_time = max(
            0, self.line_clear_highlight_time - delta_time
        )
        self.game_over_flash_time = max(0, self.game_over_flash_time - delta_time)
        if self.line_clear_highlight_time <= 0 and self.cleared_triangles_coords:
            self.cleared_triangles_coords = []  # Clear coords after highlight fades

    def _calculate_placement_reward(self, placed_shape: Shape) -> float:
        return self.rewards.REWARD_PLACE_PER_TRI * len(placed_shape.triangles)

    def _calculate_line_clear_reward(self, lines_cleared: int) -> float:
        if lines_cleared == 1:
            return self.rewards.REWARD_CLEAR_1
        if lines_cleared == 2:
            return self.rewards.REWARD_CLEAR_2
        if lines_cleared >= 3:
            return self.rewards.REWARD_CLEAR_3PLUS
        return 0.0

    def _calculate_state_penalty(self) -> float:
        """Calculates penalties based on grid state (height, holes, bumpiness)."""
        penalty = 0.0
        max_height = self.grid.get_max_height()
        bumpiness = self.grid.get_bumpiness()
        num_holes = self.grid.count_holes()
        penalty += max_height * self.rewards.PENALTY_MAX_HEIGHT_FACTOR
        penalty += bumpiness * self.rewards.PENALTY_BUMPINESS_FACTOR
        penalty += num_holes * self.rewards.PENALTY_HOLE_PER_HOLE
        return penalty

    def _handle_invalid_placement(self) -> float:
        """Handles an invalid placement attempt."""
        self._last_action_valid = False
        return self.rewards.PENALTY_INVALID_MOVE

    def _handle_game_over_state_change(self) -> float:
        """Sets game over state and returns penalty."""
        if self.game_over:
            return 0.0  # Already over
        self.game_over = True
        if self.freeze_time <= 0:
            self.freeze_time = 1.0
        self.game_over_flash_time = 0.6
        return self.rewards.PENALTY_GAME_OVER

    def _handle_valid_placement(
        self,
        shape_to_place: Shape,
        shape_slot_index: int,
        target_row: int,
        target_col: int,
    ) -> float:
        """Handles a valid placement, updates grid, score, and returns reward components."""
        self._last_action_valid = True
        step_reward = 0.0

        step_reward += self._calculate_placement_reward(shape_to_place)
        holes_before = self.grid.count_holes()

        self.grid.place(shape_to_place, target_row, target_col)
        self.shapes[shape_slot_index] = None  # Remove placed shape
        self.game_score += len(shape_to_place.triangles)
        self.pieces_placed_this_episode += 1

        lines_cleared, triangles_cleared, cleared_coords = self.grid.clear_filled_rows()
        self.lines_cleared_this_episode += lines_cleared
        step_reward += self._calculate_line_clear_reward(lines_cleared)

        if triangles_cleared > 0:
            self.game_score += triangles_cleared * 2  # Bonus for cleared triangles
            self.blink_time = 0.5
            self.freeze_time = 0.5
            self.line_clear_flash_time = 0.3
            self.line_clear_highlight_time = 0.5
            self.cleared_triangles_coords = cleared_coords

        holes_after = self.grid.count_holes()
        new_holes_created = max(0, holes_after - holes_before)

        step_reward += self._calculate_state_penalty()
        step_reward += new_holes_created * self.rewards.PENALTY_NEW_HOLE

        # Refill shapes if all slots are empty
        if all(s is None for s in self.shapes):
            self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]

        # Check for game over *after* placement and refill
        if self._check_fundamental_game_over():
            step_reward += self._handle_game_over_state_change()

        self._update_demo_selection_after_placement(shape_slot_index)
        return step_reward

    def step(self, action_index: int) -> Tuple[float, bool]:
        """Performs one game step based on the action index."""
        self._update_timers()

        if self.game_over:
            return (0.0, True)

        # If frozen (e.g., during line clear animation), only apply alive reward and PBRS
        if self.is_frozen():
            current_potential = self._calculate_potential()
            pbrs_reward = (
                self.ppo_config.GAMMA * current_potential - self._last_potential
            )
            self._last_potential = current_potential
            total_reward = self.rewards.REWARD_ALIVE_STEP + (
                pbrs_reward if self.rewards.ENABLE_PBRS else 0.0
            )
            self.score += total_reward
            return (total_reward, False)

        shape_slot_index, target_row, target_col = self.decode_action(action_index)

        shape_to_place = (
            self.shapes[shape_slot_index]
            if 0 <= shape_slot_index < len(self.shapes)
            else None
        )
        is_valid_placement = shape_to_place is not None and self.grid.can_place(
            shape_to_place, target_row, target_col
        )

        potential_before_action = self._calculate_potential()

        if is_valid_placement:
            extrinsic_reward = self._handle_valid_placement(
                shape_to_place, shape_slot_index, target_row, target_col
            )
        else:
            extrinsic_reward = self._handle_invalid_placement()
            # Check for game over immediately after invalid move if it leads to no possible moves
            if self._check_fundamental_game_over():
                extrinsic_reward += self._handle_game_over_state_change()

        # Add alive reward if not game over
        if not self.game_over:
            extrinsic_reward += self.rewards.REWARD_ALIVE_STEP

        # Calculate Potential-Based Reward Shaping (PBRS)
        potential_after_action = self._calculate_potential()
        pbrs_reward = 0.0
        if self.rewards.ENABLE_PBRS:
            pbrs_reward = (
                self.ppo_config.GAMMA * potential_after_action - potential_before_action
            )

        total_reward = extrinsic_reward + pbrs_reward
        self._last_potential = potential_after_action  # Update potential for next step

        self.score += total_reward
        return (total_reward, self.game_over)

    def _calculate_potential_placement_outcomes(self) -> Dict[str, float]:
        """Calculates potential outcomes (lines, holes, height, bumpiness) for valid moves."""
        valid_actions = self.valid_actions()
        if not valid_actions:
            return {
                "max_lines": 0.0,
                "min_holes": 0.0,
                "min_height": float(self.grid.get_max_height()),
                "min_bump": float(self.grid.get_bumpiness()),
            }

        max_lines_cleared = 0
        min_new_holes = float("inf")
        min_resulting_height = float("inf")
        min_resulting_bumpiness = float("inf")
        initial_holes = self.grid.count_holes()

        for action_index in valid_actions:
            shape_slot_index, target_row, target_col = self.decode_action(action_index)
            shape_to_place = self.shapes[shape_slot_index]
            if shape_to_place is None:
                continue

            temp_grid = copy.deepcopy(self.grid)
            temp_grid.place(shape_to_place, target_row, target_col)
            lines_cleared, _, _ = temp_grid.clear_filled_rows()
            holes_after = temp_grid.count_holes()
            height_after = temp_grid.get_max_height()
            bumpiness_after = temp_grid.get_bumpiness()
            new_holes_created = max(0, holes_after - initial_holes)

            max_lines_cleared = max(max_lines_cleared, lines_cleared)
            min_new_holes = min(min_new_holes, new_holes_created)
            min_resulting_height = min(min_resulting_height, height_after)
            min_resulting_bumpiness = min(min_resulting_bumpiness, bumpiness_after)

        # Handle cases where no valid moves were found despite valid_actions list
        if min_new_holes == float("inf"):
            min_new_holes = 0.0
        if min_resulting_height == float("inf"):
            min_resulting_height = float(self.grid.get_max_height())
        if min_resulting_bumpiness == float("inf"):
            min_resulting_bumpiness = float(self.grid.get_bumpiness())

        return {
            "max_lines": float(max_lines_cleared),
            "min_holes": float(min_new_holes),
            "min_height": float(min_resulting_height),
            "min_bump": float(min_resulting_bumpiness),
        }

    def get_state(self) -> StateType:
        """Returns the current game state as a dictionary of numpy arrays."""
        grid_state = self.grid.get_feature_matrix()  # (C, H, W)

        # Shape Features
        shape_features_per = self.env_config.SHAPE_FEATURES_PER_SHAPE
        num_shapes_expected = self.env_config.NUM_SHAPE_SLOTS
        shape_feature_matrix = np.zeros(
            (num_shapes_expected, shape_features_per), dtype=np.float32
        )
        max_tris_norm = 6.0  # Normalize features based on expected max values
        max_h_norm = float(self.grid.rows)
        max_w_norm = float(self.grid.cols)
        for i in range(num_shapes_expected):
            s = self.shapes[i] if i < len(self.shapes) else None
            if s:
                tri_list = s.triangles
                n_tris = len(tri_list)
                ups = sum(1 for (_, _, is_up) in tri_list if is_up)
                downs = n_tris - ups
                min_r, min_c, max_r, max_c = s.bbox()
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                shape_feature_matrix[i, 0] = np.clip(
                    float(n_tris) / max_tris_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 1] = np.clip(
                    float(ups) / max_tris_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 2] = np.clip(
                    float(downs) / max_tris_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 3] = np.clip(
                    float(height) / max_h_norm, 0.0, 1.0
                )
                shape_feature_matrix[i, 4] = np.clip(
                    float(width) / max_w_norm, 0.0, 1.0
                )

        # Shape Availability
        shape_availability_dim = self.env_config.SHAPE_AVAILABILITY_DIM
        shape_availability_vector = np.zeros(shape_availability_dim, dtype=np.float32)
        for i in range(min(num_shapes_expected, shape_availability_dim)):
            if i < len(self.shapes) and self.shapes[i] is not None:
                shape_availability_vector[i] = 1.0

        # Explicit Features
        explicit_features_dim = self.env_config.EXPLICIT_FEATURES_DIM
        explicit_features_vector = np.zeros(explicit_features_dim, dtype=np.float32)
        num_holes = self.grid.count_holes()
        col_heights = self.grid.get_column_heights()
        avg_height = np.mean(col_heights) if col_heights else 0
        max_height = max(col_heights) if col_heights else 0
        bumpiness = self.grid.get_bumpiness()
        max_possible_holes = self.env_config.ROWS * self.env_config.COLS
        max_possible_bumpiness = self.env_config.ROWS * (self.env_config.COLS - 1)
        explicit_features_vector[0] = np.clip(
            num_holes / max(1, max_possible_holes), 0.0, 1.0
        )
        explicit_features_vector[1] = np.clip(
            avg_height / self.env_config.ROWS, 0.0, 1.0
        )
        explicit_features_vector[2] = np.clip(
            max_height / self.env_config.ROWS, 0.0, 1.0
        )
        explicit_features_vector[3] = np.clip(
            bumpiness / max(1, max_possible_bumpiness), 0.0, 1.0
        )
        explicit_features_vector[4] = np.clip(
            self.lines_cleared_this_episode / 100.0, 0.0, 1.0
        )  # Normalize episode stats
        explicit_features_vector[5] = np.clip(
            self.pieces_placed_this_episode / 500.0, 0.0, 1.0
        )

        # Optional: Potential Placement Outcomes
        if self.env_config.CALCULATE_POTENTIAL_OUTCOMES_IN_STATE:
            potential_outcomes = self._calculate_potential_placement_outcomes()
            max_possible_lines = self.env_config.ROWS
            max_possible_new_holes = max_possible_holes
            explicit_features_vector[6] = np.clip(
                potential_outcomes["max_lines"] / max(1, max_possible_lines), 0.0, 1.0
            )
            explicit_features_vector[7] = np.clip(
                potential_outcomes["min_holes"] / max(1, max_possible_new_holes),
                0.0,
                1.0,
            )
            explicit_features_vector[8] = np.clip(
                potential_outcomes["min_height"] / self.env_config.ROWS, 0.0, 1.0
            )
            explicit_features_vector[9] = np.clip(
                potential_outcomes["min_bump"] / max(1, max_possible_bumpiness),
                0.0,
                1.0,
            )
        else:
            explicit_features_vector[6:10] = 0.0  # Zero out if not calculated

        state_dict: StateType = {
            "grid": grid_state.astype(np.float32),
            "shapes": shape_feature_matrix.reshape(-1).astype(
                np.float32
            ),  # Flatten shape features
            "shape_availability": shape_availability_vector.astype(np.float32),
            "explicit_features": explicit_features_vector.astype(np.float32),
        }
        return state_dict

    def get_shapes(self) -> List[Optional[Shape]]:
        return self.shapes

    # --- Demo Mode Methods ---
    def _update_demo_selection_after_placement(self, placed_slot_index: int):
        """Selects the next available shape slot after placement in demo mode."""
        num_slots = self.env_config.NUM_SHAPE_SLOTS
        if num_slots <= 0:
            return
        next_idx = (placed_slot_index + 1) % num_slots
        for _ in range(num_slots):
            if 0 <= next_idx < len(self.shapes) and self.shapes[next_idx] is not None:
                self.demo_selected_shape_idx = next_idx
                return
            next_idx = (next_idx + 1) % num_slots
        # If all became None (e.g., after refill), find the first available one
        if all(s is None for s in self.shapes):
            first_available = next(
                (i for i, s in enumerate(self.shapes) if s is not None), 0
            )
            self.demo_selected_shape_idx = first_available

    def cycle_shape(self, direction: int):
        """Cycles the selected shape in demo mode among available shapes."""
        if self.game_over or self.freeze_time > 0:
            return
        num_slots = self.env_config.NUM_SHAPE_SLOTS
        if num_slots <= 0:
            return
        available_indices = [
            i for i, s in enumerate(self.shapes) if s is not None and 0 <= i < num_slots
        ]
        if not available_indices:
            return
        try:
            current_list_idx = available_indices.index(self.demo_selected_shape_idx)
        except ValueError:
            current_list_idx = 0  # Default if current selection is somehow invalid
        if self.demo_selected_shape_idx not in available_indices:
            self.demo_selected_shape_idx = available_indices[0]
        new_list_idx = (current_list_idx + direction) % len(available_indices)
        self.demo_selected_shape_idx = available_indices[new_list_idx]

    def move_target(self, delta_row: int, delta_col: int):
        """Moves the placement target cursor in demo mode."""
        if self.game_over or self.freeze_time > 0:
            return
        self.demo_target_row = np.clip(
            self.demo_target_row + delta_row, 0, self.grid.rows - 1
        )
        self.demo_target_col = np.clip(
            self.demo_target_col + delta_col, 0, self.grid.cols - 1
        )

    def get_action_for_current_selection(self) -> Optional[int]:
        """Gets the action index corresponding to the current demo selection, if valid."""
        if self.game_over or self.freeze_time > 0:
            return None
        shape_slot_index = self.demo_selected_shape_idx
        current_shape = (
            self.shapes[shape_slot_index]
            if 0 <= shape_slot_index < len(self.shapes)
            else None
        )
        if current_shape is None:
            return None
        target_row, target_col = self.demo_target_row, self.demo_target_col
        if self.grid.can_place(current_shape, target_row, target_col):
            locations_per_shape = self.grid.rows * self.grid.cols
            action_index = shape_slot_index * locations_per_shape + (
                target_row * self.grid.cols + target_col
            )
            return action_index
        else:
            return None  # Invalid placement

    def get_current_selection_info(self) -> Tuple[Optional[Shape], int, int]:
        """Returns the currently selected shape object and target coordinates for demo rendering."""
        shape_slot_index = self.demo_selected_shape_idx
        current_shape = (
            self.shapes[shape_slot_index]
            if 0 <= shape_slot_index < len(self.shapes)
            else None
        )
        return current_shape, self.demo_target_row, self.demo_target_col
