# File: environment/game_state.py
# File: environment/game_state.py
import time
import numpy as np
from typing import List, Optional, Tuple
from .grid import Grid
from .shape import Shape

# <<< Import RewardConfig >>>
from config import EnvConfig, RewardConfig


class GameState:
    def __init__(self):
        self.grid = Grid()
        self.shapes: List[Optional[Shape]] = [Shape() for _ in range(3)]
        self.score = 0.0
        self.blink_time = 0.0
        self.last_time = time.time()
        self.freeze_time = 0.0
        self.game_over = False
        self._last_action_valid = True
        # <<< Use RewardConfig constants >>>
        self.rewards = RewardConfig

    def reset(self) -> np.ndarray:
        self.grid = Grid()
        self.shapes = [Shape() for _ in range(3)]
        self.score = 0.0
        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.game_over = False
        self._last_action_valid = True
        self.last_time = time.time()
        return self.get_state()

    def valid_actions(self) -> List[int]:
        if self.game_over or self.freeze_time > 0:
            return []
        acts = []
        for i, sh in enumerate(self.shapes):
            if not sh:  # Skip empty shape slots
                continue
            # Iterate through all possible reference point placements (r, c)
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    # Check if the shape *can* be placed with its top-left-most point at (r, c)
                    # Note: The reference point logic might differ. Assuming (r,c) is the top-left anchor.
                    # The current Shape implementation uses (0,0) as the first triangle's relative pos.
                    # We need to test placement relative to the shape's *own* origin.
                    # The action encoding assumes placing shape `i`'s reference (0,0) at grid (rr,cc).
                    if self.grid.can_place(sh, r, c):
                        # Encode action: shape_index * total_locations + location_index
                        action_index = i * (self.grid.rows * self.grid.cols) + (
                            r * self.grid.cols + c
                        )
                        acts.append(action_index)
        return acts

    def is_over(self) -> bool:
        return self.game_over

    def is_frozen(self) -> bool:
        return self.freeze_time > 0

    def decode_act(self, a: int) -> Tuple[int, int, int]:
        """Decodes action index into (shape_idx, row, col)."""
        locations_per_shape = self.grid.rows * self.grid.cols
        s_idx = a // locations_per_shape
        pos_idx = a % locations_per_shape
        rr = pos_idx // self.grid.cols
        cc = pos_idx % self.grid.cols
        return (s_idx, rr, cc)

    def step(self, a: int) -> Tuple[float, bool]:
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        self.freeze_time = max(0, self.freeze_time - dt)
        self.blink_time = max(0, self.blink_time - dt)

        if self.game_over:
            return (0.0, True)  # No reward, already done
        if self.freeze_time > 0:
            return (0.0, False)  # No action/reward during freeze

        s_idx, rr, cc = self.decode_act(a)
        current_step_reward = 0.0

        # --- Validate Action ---
        if not (0 <= s_idx < len(self.shapes)):
            shp = None  # Invalid shape index
        else:
            shp = self.shapes[s_idx]

        is_valid_placement = shp is not None and self.grid.can_place(shp, rr, cc)

        if not is_valid_placement:
            self._last_action_valid = False
            current_step_reward += self.rewards.PENALTY_INVALID_MOVE

            # Check if game should end due to no possible moves *at all*
            if not self.valid_actions():
                self.game_over = True
                self.freeze_time = 1.0  # Freeze on game over
                current_step_reward += self.rewards.PENALTY_GAME_OVER
                return (current_step_reward, True)
            else:
                # Invalid move, but game continues
                return (current_step_reward, False)

        # --- Action is Valid ---
        self._last_action_valid = True
        assert shp is not None  # Should be true if is_valid_placement is true

        # 1. Base reward for placing
        current_step_reward += self.rewards.REWARD_PLACE_PER_TRI * len(shp.triangles)

        # 2. Place the shape
        self.grid.place(shp, rr, cc)
        self.shapes[s_idx] = None  # Consume the shape

        # 3. Check for line clears
        lines_cleared = self.grid.clear_filled_rows()

        # 4. Line clear reward bonus
        line_clear_bonus = 0.0
        if lines_cleared == 1:
            line_clear_bonus = self.rewards.REWARD_CLEAR_1
        elif lines_cleared == 2:
            line_clear_bonus = self.rewards.REWARD_CLEAR_2
        elif lines_cleared >= 3:
            line_clear_bonus = self.rewards.REWARD_CLEAR_3PLUS
        current_step_reward += line_clear_bonus

        # Visual feedback for clearing
        if lines_cleared > 0:
            self.blink_time = 0.5
            self.freeze_time = 0.5

        # 5. Hole Penalty (Applied *after* placement and clearing)
        num_holes = self.grid.count_holes()
        hole_penalty = num_holes * self.rewards.PENALTY_HOLE_PER_HOLE
        current_step_reward += hole_penalty

        # 6. Replenish shapes if all used
        if all(x is None for x in self.shapes):
            self.shapes = [Shape() for _ in range(3)]

        # 7. Check Game Over (No valid moves for *any* remaining/new shape)
        # This check must happen *after* potentially replenishing shapes
        if not self.valid_actions():
            self.game_over = True
            self.freeze_time = 1.0
            current_step_reward += self.rewards.PENALTY_GAME_OVER

        # Update internal score tracker (sum of rewards)
        self.score += current_step_reward

        return (current_step_reward, self.game_over)

    def get_state(self) -> np.ndarray:
        """Returns the current state as a flattened numpy array."""
        # Board state (Occupied, Is_Up, Is_Death) per cell
        # Uses the grid's own feature matrix method now
        board_state = self.grid.get_feature_matrix().flatten()  # [C*H*W]

        # Available shapes state (5 features per shape: num_tris, up_tris, dn_tris, height, width)
        shape_features_per = EnvConfig.SHAPE_FEATURES_PER_SHAPE
        num_shapes_expected = EnvConfig.NUM_SHAPE_SLOTS
        shape_features_total = num_shapes_expected * shape_features_per
        shape_rep = np.zeros(shape_features_total, dtype=np.float32)
        idx = 0
        max_tris_norm = 5.0  # Normalize by max expected tris in a shape
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
                # Normalize features
                shape_rep[idx] = float(n) / max_tris_norm
                shape_rep[idx + 1] = (
                    float(ups) / max_tris_norm
                )  # Normalize by max tris too
                shape_rep[idx + 2] = (
                    float(dns) / max_tris_norm
                )  # Normalize by max tris too
                shape_rep[idx + 3] = float(height) / max_h_norm
                shape_rep[idx + 4] = float(width) / max_w_norm
            # Else: keep features as 0 for empty slots
            idx += shape_features_per

        # Concatenate board and shape features
        state_array = np.concatenate((board_state, shape_rep))

        # --- Dimension Check ---
        if len(state_array) != EnvConfig.STATE_DIM:
            raise ValueError(
                f"CRITICAL ERROR [GameState.get_state]: Generated state length {len(state_array)} != EnvConfig.STATE_DIM {EnvConfig.STATE_DIM}. Mismatch between calculation and config OR Grid/Shape feature extraction error."
            )
        # --- End Dimension Check ---

        return state_array

    def is_blinking(self) -> bool:
        return self.blink_time > 0

    def get_shapes(self) -> List[Shape]:
        """Returns the list of currently available shapes (non-None)."""
        # Ensure returned list only contains actual Shape objects
        return [s for s in self.shapes if s is not None]
