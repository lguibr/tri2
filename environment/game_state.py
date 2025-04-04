# File: environment/game_state.py
import time
import numpy as np
from typing import List, Optional, Tuple
from collections import deque  # Import deque
from typing import Deque  # Import Deque

from .grid import Grid
from .shape import Shape
from config import EnvConfig, RewardConfig


class GameState:
    def __init__(self):
        self.grid = Grid()
        self.shapes: List[Optional[Shape]] = [
            Shape() for _ in range(EnvConfig.NUM_SHAPE_SLOTS)
        ]
        self.score = 0.0  # Cumulative RL reward
        self.game_score = 0  # Game-specific score
        self.lines_cleared_this_episode = 0  # <<< NEW >>> Track lines cleared
        self.blink_time = 0.0
        self.last_time = time.time()
        self.freeze_time = 0.0
        self.game_over = False
        self._last_action_valid = True
        self.rewards = RewardConfig

    def reset(self) -> np.ndarray:
        self.grid = Grid()
        self.shapes = [Shape() for _ in range(EnvConfig.NUM_SHAPE_SLOTS)]
        self.score = 0.0
        self.game_score = 0
        self.lines_cleared_this_episode = 0  # <<< NEW >>> Reset lines cleared
        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.game_over = False
        self._last_action_valid = True
        self.last_time = time.time()
        return self.get_state()

    # valid_actions, is_over, is_frozen, decode_act remain the same
    def valid_actions(self) -> List[int]:
        if self.game_over or self.freeze_time > 0:
            return []
        acts = []
        locations_per_shape = self.grid.rows * self.grid.cols
        for i, sh in enumerate(self.shapes):
            if not sh:
                continue
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
        if not self.valid_actions():
            self.game_over = True
            self.freeze_time = 1.0
            reward += self.rewards.PENALTY_GAME_OVER
        return reward

    def _handle_valid_placement(
        self, shp: Shape, s_idx: int, rr: int, cc: int
    ) -> float:
        self._last_action_valid = True
        reward = 0.0
        reward += self.rewards.REWARD_PLACE_PER_TRI * len(shp.triangles)
        self.game_score += len(shp.triangles)
        self.grid.place(shp, rr, cc)
        self.shapes[s_idx] = None

        lines_cleared, triangles_cleared = self.grid.clear_filled_rows()
        self.lines_cleared_this_episode += lines_cleared  # <<< NEW >>> Accumulate lines

        if lines_cleared == 1:
            reward += self.rewards.REWARD_CLEAR_1
        elif lines_cleared == 2:
            reward += self.rewards.REWARD_CLEAR_2
        elif lines_cleared >= 3:
            reward += self.rewards.REWARD_CLEAR_3PLUS
        if triangles_cleared > 0:
            self.game_score += triangles_cleared * 2
            self.blink_time = 0.5
            self.freeze_time = 0.5

        num_holes = self.grid.count_holes()
        reward += num_holes * self.rewards.PENALTY_HOLE_PER_HOLE

        if all(x is None for x in self.shapes):
            self.shapes = [Shape() for _ in range(EnvConfig.NUM_SHAPE_SLOTS)]

        if not self.valid_actions():
            self.game_over = True
            self.freeze_time = 1.0
            reward += self.rewards.PENALTY_GAME_OVER
        return reward

    def step(self, a: int) -> Tuple[float, bool]:
        self._update_timers()
        if self.game_over:
            return (0.0, True)
        if self.freeze_time > 0:
            return (0.0, False)

        s_idx, rr, cc = self.decode_act(a)
        shp = self.shapes[s_idx] if 0 <= s_idx < len(self.shapes) else None
        is_valid_placement = shp is not None and self.grid.can_place(shp, rr, cc)

        current_rl_reward = (
            self._handle_valid_placement(shp, s_idx, rr, cc)
            if is_valid_placement
            else self._handle_invalid_placement()
        )

        if not self.game_over:
            current_rl_reward += self.rewards.REWARD_ALIVE_STEP

        self.score += current_rl_reward
        return (current_rl_reward, self.game_over)

    # get_state remains the same
    def get_state(self) -> np.ndarray:
        board_state = self.grid.get_feature_matrix().flatten()
        shape_features_per = EnvConfig.SHAPE_FEATURES_PER_SHAPE
        num_shapes_expected = EnvConfig.NUM_SHAPE_SLOTS
        shape_features_total = num_shapes_expected * shape_features_per
        shape_rep = np.zeros(shape_features_total, dtype=np.float32)
        idx = 0
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
                shape_rep[idx] = np.clip(float(n) / max_tris_norm, 0.0, 1.0)
                shape_rep[idx + 1] = np.clip(float(ups) / max_tris_norm, 0.0, 1.0)
                shape_rep[idx + 2] = np.clip(float(dns) / max_tris_norm, 0.0, 1.0)
                shape_rep[idx + 3] = np.clip(float(height) / max_h_norm, 0.0, 1.0)
                shape_rep[idx + 4] = np.clip(float(width) / max_w_norm, 0.0, 1.0)
            idx += shape_features_per
        state_array = np.concatenate((board_state, shape_rep))
        if len(state_array) != EnvConfig.STATE_DIM:
            raise ValueError(
                f"State length mismatch: {len(state_array)} vs {EnvConfig.STATE_DIM}"
            )
        return state_array

    def is_blinking(self) -> bool:
        return self.blink_time > 0

    def get_shapes(self) -> List[Shape]:
        return [s for s in self.shapes if s is not None]
