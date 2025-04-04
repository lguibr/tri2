# File: environment/game_state.py
import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from collections import deque
from typing import Deque

from .grid import Grid
from .shape import Shape
from config import EnvConfig, RewardConfig

StateType = Dict[str, np.ndarray]


class GameState:
    def __init__(self):
        self.env_config = EnvConfig()
        self.grid = Grid(self.env_config)
        self.shapes: List[Optional[Shape]] = [
            Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)
        ]
        self.score = 0.0
        self.game_score = 0
        self.lines_cleared_this_episode = 0
        self.blink_time = 0.0
        self.last_time = time.time()
        self.freeze_time = 0.0
        self.line_clear_flash_time = 0.0
        self.line_clear_highlight_time: float = 0.0
        self.cleared_triangles_coords: List[Tuple[int, int]] = []
        self.game_over = False
        self._last_action_valid = True
        self.rewards = RewardConfig()  # Instantiate RewardConfig
        self.demo_selected_shape_idx: int = 0
        self.demo_target_row: int = self.env_config.ROWS // 2
        self.demo_target_col: int = self.env_config.COLS // 2

    def reset(self) -> StateType:
        self.grid = Grid(self.env_config)
        self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]
        self.score = 0.0
        self.game_score = 0
        self.lines_cleared_this_episode = 0
        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.line_clear_flash_time = 0.0
        self.line_clear_highlight_time = 0.0
        self.cleared_triangles_coords = []
        self.game_over = False
        self._last_action_valid = True
        self.last_time = time.time()
        self.demo_selected_shape_idx = 0
        self.demo_target_row = self.env_config.ROWS // 2
        self.demo_target_col = self.env_config.COLS // 2
        return self.get_state()

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

    def is_line_clearing(self) -> bool:
        return self.line_clear_flash_time > 0

    def is_highlighting_cleared(self) -> bool:
        return self.line_clear_highlight_time > 0

    def get_cleared_triangle_coords(self) -> List[Tuple[int, int]]:
        return self.cleared_triangles_coords

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
        self.line_clear_flash_time = max(0, self.line_clear_flash_time - dt)
        self.line_clear_highlight_time = max(0, self.line_clear_highlight_time - dt)
        if self.line_clear_highlight_time <= 0 and self.cleared_triangles_coords:
            self.cleared_triangles_coords = []

    def _handle_invalid_placement(self) -> float:
        self._last_action_valid = False
        reward = self.rewards.PENALTY_INVALID_MOVE
        return reward

    def _handle_valid_placement(
        self, shp: Shape, s_idx: int, rr: int, cc: int
    ) -> float:
        self._last_action_valid = True
        # --- MODIFIED: Start with survival reward ---
        reward = self.rewards.REWARD_ALIVE_STEP
        # --- END MODIFIED ---

        self.grid.place(shp, rr, cc)
        self.shapes[s_idx] = None
        self.game_score += len(shp.triangles)

        num_slots = self.env_config.NUM_SHAPE_SLOTS
        found_next = False
        if num_slots > 0:
            next_idx = (s_idx + 1) % num_slots
            for _ in range(num_slots):
                if (
                    0 <= next_idx < len(self.shapes)
                    and self.shapes[next_idx] is not None
                ):
                    self.demo_selected_shape_idx = next_idx
                    found_next = True
                    break
                next_idx = (next_idx + 1) % num_slots
            if not found_next:
                self.demo_selected_shape_idx = 0

        lines_cleared, triangles_cleared, cleared_coords = self.grid.clear_filled_rows()
        self.lines_cleared_this_episode += lines_cleared

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
            self.line_clear_flash_time = 0.3
            self.line_clear_highlight_time = 0.5
            self.cleared_triangles_coords = cleared_coords

        num_holes = self.grid.count_holes()
        reward += num_holes * self.rewards.PENALTY_HOLE_PER_HOLE

        if all(x is None for x in self.shapes):
            self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]
            if not found_next:
                first_available = next(
                    (i for i, s in enumerate(self.shapes) if s is not None), 0
                )
                self.demo_selected_shape_idx = first_available

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

        if is_valid_placement:
            current_rl_reward = self._handle_valid_placement(shp, s_idx, rr, cc)
        else:
            current_rl_reward = self._handle_invalid_placement()
            if not self.valid_actions():
                if not self.game_over:
                    self.game_over = True
                    self.freeze_time = 1.0
                    current_rl_reward += self.rewards.PENALTY_GAME_OVER

        self.score += current_rl_reward
        return (current_rl_reward, self.game_over)

    def get_state(self) -> StateType:
        grid_state = self.grid.get_feature_matrix()
        shape_features_per = self.env_config.SHAPE_FEATURES_PER_SHAPE
        num_shapes_expected = self.env_config.NUM_SHAPE_SLOTS
        shape_rep = np.zeros(
            (num_shapes_expected, shape_features_per), dtype=np.float32
        )

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

                shape_rep[i, 0] = np.clip(float(n) / max_tris_norm, 0.0, 1.0)
                shape_rep[i, 1] = np.clip(float(ups) / max_tris_norm, 0.0, 1.0)
                shape_rep[i, 2] = np.clip(float(dns) / max_tris_norm, 0.0, 1.0)
                shape_rep[i, 3] = np.clip(float(height) / max_h_norm, 0.0, 1.0)
                shape_rep[i, 4] = np.clip(float(width) / max_w_norm, 0.0, 1.0)

        state_dict = {
            "grid": grid_state.astype(np.float32),
            "shapes": shape_rep.astype(np.float32),
        }
        return state_dict

    def is_blinking(self) -> bool:
        return self.blink_time > 0

    def get_shapes(self) -> List[Shape]:
        return [s for s in self.shapes if s is not None]

    def cycle_shape(self, direction: int):
        if self.game_over or self.freeze_time > 0:
            return
        num_slots = self.env_config.NUM_SHAPE_SLOTS
        if num_slots <= 0:
            return

        current_idx = self.demo_selected_shape_idx
        for _ in range(num_slots):
            current_idx = (current_idx + direction + num_slots) % num_slots
            if (
                0 <= current_idx < len(self.shapes)
                and self.shapes[current_idx] is not None
            ):
                self.demo_selected_shape_idx = current_idx
                return
        # Keep current index if all slots are empty

    def move_target(self, dr: int, dc: int):
        if self.game_over or self.freeze_time > 0:
            return
        self.demo_target_row = np.clip(self.demo_target_row + dr, 0, self.grid.rows - 1)
        self.demo_target_col = np.clip(self.demo_target_col + dc, 0, self.grid.cols - 1)

    def get_action_for_current_selection(self) -> Optional[int]:
        if self.game_over or self.freeze_time > 0:
            return None
        s_idx = self.demo_selected_shape_idx
        shp = self.shapes[s_idx] if 0 <= s_idx < len(self.shapes) else None
        if shp is None:
            return None

        rr, cc = self.demo_target_row, self.demo_target_col

        if self.grid.can_place(shp, rr, cc):
            locations_per_shape = self.grid.rows * self.grid.cols
            action_index = s_idx * locations_per_shape + (rr * self.grid.cols + cc)
            return action_index
        else:
            return None

    def get_current_selection_info(self) -> Tuple[Optional[Shape], int, int]:
        s_idx = self.demo_selected_shape_idx
        shp = self.shapes[s_idx] if 0 <= s_idx < len(self.shapes) else None
        return shp, self.demo_target_row, self.demo_target_col
