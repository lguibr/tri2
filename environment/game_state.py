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
        self.rewards = RewardConfig()  # Load reward config
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
        self.game_over_flash_time: float = 0.0
        self.cleared_triangles_coords: List[Tuple[int, int]] = []
        self.game_over = False
        self._last_action_valid = True
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
        self.game_over_flash_time = 0.0
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

    def _check_fundamental_game_over(self) -> bool:
        for sh in self.shapes:
            if not sh:
                continue
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    if self.grid.can_place(sh, r, c):
                        return False
        return True

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
        self.game_over_flash_time = max(0, self.game_over_flash_time - dt)
        if self.line_clear_highlight_time <= 0 and self.cleared_triangles_coords:
            self.cleared_triangles_coords = []

    def _handle_invalid_placement(self) -> float:
        self._last_action_valid = False
        reward = self.rewards.PENALTY_INVALID_MOVE
        return reward

    def _handle_game_over_state_change(self) -> float:
        if self.game_over:
            return 0.0
        self.game_over = True
        if self.freeze_time <= 0:
            self.freeze_time = 1.0
        self.game_over_flash_time = 0.6
        return self.rewards.PENALTY_GAME_OVER

    def _handle_valid_placement(
        self, shp: Shape, s_idx: int, rr: int, cc: int
    ) -> float:
        self._last_action_valid = True
        reward = self.rewards.REWARD_PLACE_PER_TRI * len(shp.triangles)

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

        # Add penalties after placement and potential clears
        max_height = self.grid.get_max_height()
        reward += max_height * self.rewards.PENALTY_MAX_HEIGHT_FACTOR

        bumpiness = self.grid.get_bumpiness()
        reward += bumpiness * self.rewards.PENALTY_BUMPINESS_FACTOR

        # Hole penalty is applied implicitly via height/bumpiness to some extent,
        # but can be kept for explicit discouragement.
        # num_holes = self.grid.count_holes()
        # reward += num_holes * self.rewards.PENALTY_HOLE_PER_HOLE

        if all(x is None for x in self.shapes):
            self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]
            if not found_next:
                first_available = next(
                    (i for i, s in enumerate(self.shapes) if s is not None), 0
                )
                self.demo_selected_shape_idx = first_available

        if self._check_fundamental_game_over():
            reward += self._handle_game_over_state_change()

        return reward

    def step(self, a: int) -> Tuple[float, bool]:
        self._update_timers()

        if self.game_over:
            return (0.0, True)

        s_idx, rr, cc = self.decode_act(a)
        shp = self.shapes[s_idx] if 0 <= s_idx < len(self.shapes) else None

        is_valid_placement = shp is not None and self.grid.can_place(shp, rr, cc)

        if self.is_frozen():
            current_rl_reward = self._handle_invalid_placement()
        elif is_valid_placement:
            current_rl_reward = self._handle_valid_placement(shp, s_idx, rr, cc)
        else:
            current_rl_reward = self._handle_invalid_placement()
            if self._check_fundamental_game_over():
                current_rl_reward += self._handle_game_over_state_change()

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

    # --- Demo Mode Methods ---
    def cycle_shape(self, direction: int):
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
            current_list_idx = 0

        new_list_idx = (current_list_idx + direction) % len(available_indices)
        self.demo_selected_shape_idx = available_indices[new_list_idx]

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
