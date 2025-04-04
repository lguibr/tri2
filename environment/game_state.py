# File: environment/game_state.py
import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Union  # Added Dict, Union
from collections import deque
from typing import Deque

from .grid import Grid
from .shape import Shape
from config import EnvConfig, RewardConfig

# --- MODIFIED: Define StateType for clarity ---
StateType = Dict[str, np.ndarray]  # e.g., {"grid": ndarray, "shapes": ndarray}
# --- END MODIFIED ---


class GameState:
    def __init__(self):
        self.env_config = EnvConfig()  # Store config instance
        self.grid = Grid(self.env_config)  # Pass config to Grid
        self.shapes: List[Optional[Shape]] = [
            Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)
        ]
        self.score = 0.0  # Cumulative RL reward
        self.game_score = 0  # Game-specific score
        self.lines_cleared_this_episode = 0
        self.blink_time = 0.0
        self.last_time = time.time()
        self.freeze_time = 0.0
        self.game_over = False
        self._last_action_valid = True
        self.rewards = RewardConfig
        # --- NEW: Attributes for interactive play ---
        self.demo_selected_shape_idx: int = 0
        self.demo_target_row: int = self.env_config.ROWS // 2
        self.demo_target_col: int = self.env_config.COLS // 2
        # --- END NEW ---

    def reset(self) -> StateType:  # Return new StateType
        self.grid = Grid(self.env_config)
        self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]
        self.score = 0.0
        self.game_score = 0
        self.lines_cleared_this_episode = 0
        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.game_over = False
        self._last_action_valid = True
        self.last_time = time.time()
        # --- NEW: Reset demo state ---
        self.demo_selected_shape_idx = 0
        self.demo_target_row = self.env_config.ROWS // 2
        self.demo_target_col = self.env_config.COLS // 2
        # --- END NEW ---
        return self.get_state()

    def valid_actions(self) -> List[int]:
        if self.game_over or self.freeze_time > 0:
            return []
        acts = []
        locations_per_shape = self.grid.rows * self.grid.cols
        for i, sh in enumerate(self.shapes):
            if not sh:
                continue
            # --- OPTIMIZATION: Check only potentially valid root cells ---
            # Instead of checking all R*C cells, we could optimize,
            # but for moderate grids, checking all is simpler and likely fast enough.
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    if self.grid.can_place(sh, r, c):
                        action_index = i * locations_per_shape + (
                            r * self.grid.cols + c
                        )
                        acts.append(action_index)
        return acts

    def is_over(self) -> bool:
        return self.game_over

    def is_frozen(self) -> bool:
        return self.freeze_time > 0

    def decode_act(self, a: int) -> Tuple[int, int, int]:
        locations_per_shape = self.grid.rows * self.grid.cols
        s_idx = a // locations_per_shape
        pos_idx = a % locations_per_shape
        rr = pos_idx // self.grid.cols
        cc = pos_idx % self.grid.cols
        return (s_idx, rr, cc)

    def _update_timers(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        self.freeze_time = max(0, self.freeze_time - dt)
        self.blink_time = max(0, self.blink_time - dt)

    def _handle_invalid_placement(self) -> float:
        self._last_action_valid = False
        reward = self.rewards.PENALTY_INVALID_MOVE
        # Check if game is over due to invalid move AND no valid moves left
        # Note: This check might be redundant if step() already checks valid_actions()
        # before calling this. Let's assume step() handles the game over check.
        # if not self.valid_actions():
        #     self.game_over = True
        #     self.freeze_time = 1.0
        #     reward += self.rewards.PENALTY_GAME_OVER
        return reward

    def _handle_valid_placement(
        self, shp: Shape, s_idx: int, rr: int, cc: int
    ) -> float:
        self._last_action_valid = True
        reward = 0.0  # Start with zero reward for placement

        # Place the shape
        self.grid.place(shp, rr, cc)
        self.shapes[s_idx] = None  # Remove shape from available slots
        self.game_score += len(shp.triangles)  # Update game score

        # Clear lines and get rewards/score
        lines_cleared, triangles_cleared = self.grid.clear_filled_rows()
        self.lines_cleared_this_episode += lines_cleared

        # Apply line clear rewards (more significant now)
        if lines_cleared == 1:
            reward += self.rewards.REWARD_CLEAR_1
        elif lines_cleared == 2:
            reward += self.rewards.REWARD_CLEAR_2
        elif lines_cleared >= 3:
            reward += self.rewards.REWARD_CLEAR_3PLUS

        # Bonus game score for cleared triangles
        if triangles_cleared > 0:
            self.game_score += triangles_cleared * 2
            self.blink_time = 0.5
            self.freeze_time = 0.5

        # Penalty for creating holes (can be tuned)
        num_holes = self.grid.count_holes()
        reward += num_holes * self.rewards.PENALTY_HOLE_PER_HOLE

        # Refill shapes if all slots are empty
        if all(x is None for x in self.shapes):
            self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]

        # Check if game is over after placement (no more valid moves)
        # This check is crucial here
        if not self.valid_actions():
            self.game_over = True
            self.freeze_time = 1.0
            reward += self.rewards.PENALTY_GAME_OVER  # Apply game over penalty

        return reward

    def step(self, a: int) -> Tuple[float, bool]:
        """Performs one game step based on the chosen action."""
        self._update_timers()

        if self.game_over:
            return (0.0, True)
        if self.freeze_time > 0:
            return (0.0, False)

        # Check if the chosen action 'a' is actually valid *now*
        # This prevents issues if the agent selects an action that became invalid
        # between the time valid_actions() was called and step() is executed.
        # We rely on the agent masking correctly. If an invalid action is passed,
        # we penalize it.

        s_idx, rr, cc = self.decode_act(a)
        shp = self.shapes[s_idx] if 0 <= s_idx < len(self.shapes) else None

        # Check placement validity
        is_valid_placement = shp is not None and self.grid.can_place(shp, rr, cc)

        if is_valid_placement:
            current_rl_reward = self._handle_valid_placement(shp, s_idx, rr, cc)
        else:
            current_rl_reward = self._handle_invalid_placement()
            # If the *only* reason the game ends is because the agent chose an invalid
            # move when valid moves *did* exist, we might not set game_over here,
            # but the penalty should discourage it. If no valid moves exist at all,
            # game_over should have been set earlier or will be set in _handle_valid_placement
            # after the next valid move (if any). Let's ensure game over is checked robustly.
            if not self.valid_actions():  # Double check if NO valid actions remain
                if not self.game_over:  # Avoid applying penalty twice
                    self.game_over = True
                    self.freeze_time = 1.0
                    current_rl_reward += self.rewards.PENALTY_GAME_OVER

        # No REWARD_ALIVE_STEP anymore
        self.score += current_rl_reward
        return (current_rl_reward, self.game_over)

    # --- MODIFIED: get_state returns a dictionary ---
    def get_state(self) -> StateType:
        """
        Generates the state representation as a dictionary containing
        'grid' (numpy array [C, H, W]) and 'shapes' (numpy array [N_SLOTS, FEAT_PER_SHAPE]).
        """
        # 1. Grid Features
        grid_state = self.grid.get_feature_matrix()  # Shape: [C, H, W] C=2 now

        # 2. Shape Features
        shape_features_per = self.env_config.SHAPE_FEATURES_PER_SHAPE
        num_shapes_expected = self.env_config.NUM_SHAPE_SLOTS
        shape_features_total = num_shapes_expected * shape_features_per
        shape_rep = np.zeros(
            (num_shapes_expected, shape_features_per), dtype=np.float32
        )

        # Normalization constants (adjust if needed)
        max_tris_norm = 6.0
        max_h_norm = float(self.grid.rows)
        max_w_norm = float(self.grid.cols)

        for i in range(num_shapes_expected):
            s = self.shapes[i] if i < len(self.shapes) else None
            if s:
                tri = s.triangles
                n = len(tri)
                ups = sum(1 for (_, _, u) in tri if u)
                dns = n - ups
                mnr, mnc, mxr, mxc = s.bbox()
                height = mxr - mnr + 1
                width = mxc - mnc + 1

                # Normalize features (clip to [0, 1])
                shape_rep[i, 0] = np.clip(float(n) / max_tris_norm, 0.0, 1.0)
                shape_rep[i, 1] = np.clip(float(ups) / max_tris_norm, 0.0, 1.0)
                shape_rep[i, 2] = np.clip(float(dns) / max_tris_norm, 0.0, 1.0)
                shape_rep[i, 3] = np.clip(float(height) / max_h_norm, 0.0, 1.0)
                shape_rep[i, 4] = np.clip(float(width) / max_w_norm, 0.0, 1.0)
            # else: Features remain 0 if shape slot is empty

        # --- Return dictionary ---
        state_dict = {
            "grid": grid_state.astype(np.float32),
            "shapes": shape_rep.astype(
                np.float32
            ),  # Shape: [NUM_SLOTS, FEAT_PER_SHAPE]
        }
        return state_dict

    # --- END MODIFIED ---

    def is_blinking(self) -> bool:
        return self.blink_time > 0

    def get_shapes(self) -> List[Shape]:
        return [s for s in self.shapes if s is not None]

    # --- NEW: Methods for Interactive Control ---
    def cycle_shape(self, direction: int):
        """Cycles the selected shape index (direction +1 or -1)."""
        if self.game_over or self.freeze_time > 0:
            return
        num_slots = self.env_config.NUM_SHAPE_SLOTS
        if num_slots <= 0:
            return

        current_idx = self.demo_selected_shape_idx
        for _ in range(num_slots):  # Try all slots
            current_idx = (current_idx + direction + num_slots) % num_slots
            if self.shapes[current_idx] is not None:
                self.demo_selected_shape_idx = current_idx
                return  # Found a non-empty slot

    def move_target(self, dr: int, dc: int):
        """Moves the target placement coordinate."""
        if self.game_over or self.freeze_time > 0:
            return
        self.demo_target_row = np.clip(self.demo_target_row + dr, 0, self.grid.rows - 1)
        self.demo_target_col = np.clip(self.demo_target_col + dc, 0, self.grid.cols - 1)

    def get_action_for_current_selection(self) -> Optional[int]:
        """Converts the current demo selection (shape, row, col) into an action index."""
        if self.game_over or self.freeze_time > 0:
            return None
        s_idx = self.demo_selected_shape_idx
        shp = self.shapes[s_idx] if 0 <= s_idx < len(self.shapes) else None
        if shp is None:
            return None  # No shape in selected slot

        rr, cc = self.demo_target_row, self.demo_target_col

        # Check if the placement at the target is valid *now*
        if self.grid.can_place(shp, rr, cc):
            locations_per_shape = self.grid.rows * self.grid.cols
            action_index = s_idx * locations_per_shape + (rr * self.grid.cols + cc)
            return action_index
        else:
            return None  # Invalid placement at target

    def get_current_selection_info(self) -> Tuple[Optional[Shape], int, int]:
        """Returns the currently selected shape, target row, and target col."""
        s_idx = self.demo_selected_shape_idx
        shp = self.shapes[s_idx] if 0 <= s_idx < len(self.shapes) else None
        return shp, self.demo_target_row, self.demo_target_col

    # --- END NEW ---
