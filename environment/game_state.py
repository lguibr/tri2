# File: environment/game_state.py
import time
import numpy as np
from typing import List, Optional, Tuple, Dict

from config import EnvConfig, RewardConfig, PPOConfig, VisConfig
from .grid import Grid
from .shape import Shape
from .game_logic import GameLogic
from .game_state_features import GameStateFeatures
from .game_demo_logic import GameDemoLogic

StateType = Dict[str, np.ndarray]


class GameState:
    """
    Represents the state of a single game instance.
    Delegates logic to helper classes: GameLogic, GameStateFeatures, GameDemoLogic.
    Timer updates are now primarily handled within GameLogic.step().
    """

    def __init__(self):
        self.env_config = EnvConfig()
        self.rewards = RewardConfig()
        self.ppo_config = PPOConfig()
        self.vis_config = VisConfig()

        self.grid = Grid(self.env_config)
        self.shapes: List[Optional[Shape]] = []
        self.score: float = 0.0
        self.game_score: int = 0
        self.triangles_cleared_this_episode: int = 0
        self.pieces_placed_this_episode: int = 0

        # Timers
        self.blink_time: float = 0.0
        self._last_timer_update_time: float = (
            time.monotonic()
        )  # Use monotonic clock for intervals
        self.freeze_time: float = 0.0
        self.line_clear_flash_time: float = 0.0
        self.line_clear_highlight_time: float = 0.0
        self.game_over_flash_time: float = 0.0
        self.cleared_triangles_coords: List[Tuple[int, int]] = []
        self.last_line_clear_info: Optional[Tuple[int, int, float]] = None

        self.game_over: bool = False
        self._last_action_valid: bool = True
        self._last_potential: float = 0.0

        # Demo state
        self.demo_selected_shape_idx: int = 0
        self.demo_dragged_shape_idx: Optional[int] = None
        self.demo_snapped_position: Optional[Tuple[int, int]] = None

        # Helper classes
        self.logic = GameLogic(self)
        self.features = GameStateFeatures(self)
        self.demo_logic = GameDemoLogic(self)

        self.reset()

    def reset(self) -> StateType:
        """Resets the game to its initial state."""
        self.grid = Grid(self.env_config)
        self.shapes = [Shape() for _ in range(self.env_config.NUM_SHAPE_SLOTS)]
        self.score = 0.0
        self.game_score = 0
        self.triangles_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0

        self.blink_time = 0.0
        self.freeze_time = 0.0
        self.line_clear_flash_time = 0.0
        self.line_clear_highlight_time = 0.0
        self.game_over_flash_time = 0.0
        self.cleared_triangles_coords = []
        self.last_line_clear_info = None

        self.game_over = False
        self._last_action_valid = True
        self._last_timer_update_time = time.monotonic()  # Reset timer time

        self.demo_selected_shape_idx = 0
        self.demo_dragged_shape_idx = None
        self.demo_snapped_position = None

        self._last_potential = self.features.calculate_potential()

        return self.get_state()

    def step(self, action_index: int) -> Tuple[float, bool]:
        """
        Performs one game step based on the action index.
        Timer updates are handled within GameLogic.step().
        """
        return self.logic.step(action_index)

    def get_state(self) -> StateType:
        """Returns the current game state as a dictionary of numpy arrays."""
        return self.features.get_state()

    def valid_actions(self) -> List[int]:
        """Returns a list of valid action indices for the current state."""
        # Removed is_frozen check here. Logic.step handles frozen state.
        # The collector also checks is_frozen separately.
        return self.logic.valid_actions()

    def decode_action(self, action_index: int) -> Tuple[int, int, int]:
        """Decodes an action index into (shape_slot, row, col)."""
        return self.logic.decode_action(action_index)

    def is_over(self) -> bool:
        return self.game_over

    def is_frozen(self) -> bool:
        # Check freeze_time before allowing actions or progression
        is_currently_frozen = self.freeze_time > 0
        return is_currently_frozen

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

    def get_shapes(self) -> List[Optional[Shape]]:
        return self.shapes

    def _update_timers(self):
        """Updates timers for visual effects based on elapsed time."""
        now = time.monotonic()  # Use monotonic clock
        delta_time = now - self._last_timer_update_time
        self._last_timer_update_time = now

        # Ensure delta_time is non-negative
        delta_time = max(0.0, delta_time)

        self.freeze_time = max(0, self.freeze_time - delta_time)
        self.blink_time = max(0, self.blink_time - delta_time)
        self.line_clear_flash_time = max(0, self.line_clear_flash_time - delta_time)
        self.line_clear_highlight_time = max(
            0, self.line_clear_highlight_time - delta_time
        )
        self.game_over_flash_time = max(0, self.game_over_flash_time - delta_time)

        # Clear highlight coords only when highlight time runs out
        if self.line_clear_highlight_time <= 0 and self.cleared_triangles_coords:
            self.cleared_triangles_coords = []
        # Clear line clear info only when flash time runs out
        if self.line_clear_flash_time <= 0 and self.last_line_clear_info is not None:
            self.last_line_clear_info = None

    # --- Demo Mode Methods (Delegated) ---
    def select_shape_for_drag(self, shape_index: int):
        self.demo_logic.select_shape_for_drag(shape_index)

    def deselect_dragged_shape(self):
        self.demo_logic.deselect_dragged_shape()

    def update_snapped_position(self, grid_pos: Optional[Tuple[int, int]]):
        self.demo_logic.update_snapped_position(grid_pos)

    def place_dragged_shape(self) -> bool:
        # Update timers before placing shape in demo mode as well
        self._update_timers()
        return self.demo_logic.place_dragged_shape()

    def get_dragged_shape_info(
        self,
    ) -> Tuple[Optional[Shape], Optional[Tuple[int, int]]]:
        return self.demo_logic.get_dragged_shape_info()

    def toggle_triangle_debug(self, row: int, col: int):
        # Update timers before toggling in debug mode
        self._update_timers()
        self.demo_logic.toggle_triangle_debug(row, col)
