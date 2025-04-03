import time
import numpy as np
from typing import List, Optional, Tuple
from .grid import Grid
from .shape import Shape
from config import EnvConfig


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
        if (
            self.game_over or self.freeze_time > 0
        ):  # <<< MODIFIED >>> No actions if frozen
            return []
        acts = []
        for i, sh in enumerate(self.shapes):
            if not sh:
                continue
            # Cache can_place results? Maybe not worth it.
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    if self.grid.can_place(sh, r, c):
                        acts.append(
                            i * (self.grid.rows * self.grid.cols)
                            + (r * self.grid.cols + c)
                        )
        return acts

    def is_over(self) -> bool:
        return self.game_over

    def is_frozen(self) -> bool:
        return self.freeze_time > 0

    def decode_act(self, a: int) -> Tuple[int, int, int]:
        s_idx = a // (self.grid.rows * self.grid.cols)
        pos = a % (self.grid.rows * self.grid.cols)
        rr = pos // self.grid.cols
        cc = pos % self.grid.cols
        return (s_idx, rr, cc)

    def step(self, a: int) -> Tuple[float, bool]:
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # Update timers first
        self.freeze_time = max(0, self.freeze_time - dt)
        self.blink_time = max(0, self.blink_time - dt)

        if self.game_over:
            return (0.0, True)
        if self.freeze_time > 0:
            return (0.0, self.game_over)  # Return current game_over state if frozen

        s_idx, rr, cc = self.decode_act(a)

        # Check action validity
        if not (0 <= s_idx < len(self.shapes)):
            shp = None
        else:
            shp = self.shapes[s_idx]

        if not shp or not self.grid.can_place(shp, rr, cc):
            # Invalid action selected by agent (shouldn't happen if agent uses valid_actions)
            # OR game state progressed such that placement is no longer valid.
            self._last_action_valid = False
            # Penalize invalid move heavily? Or just give 0 reward? Let's give 0.
            # Check if *any* valid moves remain. If not, game over.
            if not self.valid_actions():
                self.game_over = True
                self.freeze_time = 1.0  # Freeze on game over
                return (0.0, True)  # Return game over
            else:
                return (0.0, False)  # Invalid move, but game continues

        # --- Action is valid, proceed ---
        self._last_action_valid = True

        # --- Reward Calculation ---
        # Base reward for placing shape
        base_reward = 0.01 * len(shp.triangles)  # Small reward for placing

        # Place the shape
        self.grid.place(shp, rr, cc)
        placed_shape_size = len(shp.triangles)  # Store for potential reward shaping
        self.shapes[s_idx] = None  # Remove used shape

        # Check for line clears
        lines_cleared = self.grid.clear_filled_rows()

        # Line clear reward (make it significant)
        line_clear_bonus = 0.0
        if lines_cleared == 1:
            line_clear_bonus = 1.0
        elif lines_cleared == 2:
            line_clear_bonus = 3.0  # Encourage multi-line clears
        elif lines_cleared >= 3:
            line_clear_bonus = 6.0

        if lines_cleared > 0:
            self.blink_time = 0.5  # Visual feedback
            self.freeze_time = 0.5

        # --- Potential Reward Shaping (More advanced) ---
        # Example: Reward based on grid density or potential future lines?
        # grid_density = self.grid.density() # Need to implement Grid.density()
        # potential_lines = self.grid.count_potential_lines() # Need to implement this heuristic
        # shaping_reward = -0.01 * grid_density + 0.05 * potential_lines # Example
        # reward = base_reward + line_clear_bonus + shaping_reward
        # --- End Reward Shaping Example ---

        reward = base_reward + line_clear_bonus  # Current simple reward
        self.score += reward

        # Replenish shapes if needed
        if all(x is None for x in self.shapes):
            self.shapes = [Shape() for _ in range(3)]

        # Check game over condition (no valid moves for *any* remaining shape)
        if not self.valid_actions():
            self.game_over = True
            self.freeze_time = 1.0  # Freeze on game over
            # <<< Optional: Add large negative reward for game over? >>>
            # reward -= 5.0 # Example penalty

        return (reward, self.game_over)

    def get_state(self) -> np.ndarray:
        """Returns the current state as a flattened numpy array."""
        # Board state (Occupied, Is_Up, Is_Death) per cell
        board_features = self.grid.rows * self.grid.cols * 3
        board = np.zeros(board_features, dtype=np.float32)
        idx = 0
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                t = self.grid.triangles[r][c]
                board[idx] = 1.0 if t.is_occupied else 0.0
                board[idx + 1] = 1.0 if t.is_up else 0.0
                board[idx + 2] = 1.0 if t.is_death else 0.0
                idx += 3

        # Available shapes state (5 features per shape: num_tris, up_tris, dn_tris, height, width)
        shape_features_per = 5
        num_shapes_expected = 3
        shape_features = num_shapes_expected * shape_features_per
        shape_rep = np.zeros(shape_features, dtype=np.float32)
        idx = 0
        for i in range(num_shapes_expected):
            s = self.shapes[i] if i < len(self.shapes) else None
            if s:
                tri = s.triangles
                n = len(tri)
                ups = sum(1 for (_, _, u) in tri if u)
                dns = n - ups
                mnr, mnc, mxr, mxc = s.bbox()
                shape_rep[idx] = float(n) / 5.0
                shape_rep[idx + 1] = float(ups) / 5.0
                shape_rep[idx + 2] = float(dns) / 5.0
                shape_rep[idx + 3] = float(mxr - mnr + 1) / self.grid.rows
                shape_rep[idx + 4] = float(mxc - mnc + 1) / self.grid.cols
            idx += shape_features_per

        # Concatenate board and shape features
        state_array = np.concatenate((board, shape_rep))

        expected_dim = (self.grid.rows * self.grid.cols * 3) + (
            num_shapes_expected * shape_features_per
        )
        if len(state_array) != expected_dim:
            print(
                f"CRITICAL WARNING [GameState.get_state]: Generated state length {len(state_array)} != calculated expected dim {expected_dim}. Check logic."
            )
        if len(state_array) != EnvConfig.STATE_DIM:
            print(
                f"CRITICAL WARNING [GameState.get_state]: Generated state length {len(state_array)} != EnvConfig.STATE_DIM {EnvConfig.STATE_DIM}. Check config value."
            )
            raise ValueError(
                "State dimension mismatch between GameState.get_state() and EnvConfig.STATE_DIM"
            )

        return state_array

    def is_blinking(self) -> bool:
        return self.blink_time > 0

    def get_shapes(self) -> List[Shape]:
        """Returns the list of currently available shapes (non-None)."""
        return [s for s in self.shapes if s is not None]
