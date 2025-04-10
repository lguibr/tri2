# File: src/environment/core/game_state.py
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import logging
import random

# Use relative imports within the environment package
from ...config import EnvConfig
from ...utils.types import ActionType

# Import necessary submodules directly using relative paths
from ..grid.grid_data import GridData
from ..grid import logic as GridLogic
from ..shapes import logic as ShapeLogic
from .action_codec import encode_action, decode_action
from ..logic.actions import get_valid_actions
from ..logic.step import execute_placement

# Import Shape from the new structs module
from src.structs import Shape

logger = logging.getLogger(__name__)


class GameState:
    """Represents the mutable state of the game. Does not handle NN feature extraction."""

    def __init__(
        self, config: Optional[EnvConfig] = None, initial_seed: Optional[int] = None
    ):
        self.env_config = config if config else EnvConfig()
        self._rng = (
            random.Random(initial_seed) if initial_seed is not None else random.Random()
        )

        self.grid_data: GridData = None  # type: ignore Will be initialized
        self.shapes: List[Optional[Shape]] = []  # Uses Shape from structs
        self.game_score: float = 0.0
        self.game_over: bool = False
        self.triangles_cleared_this_episode: int = 0
        self.pieces_placed_this_episode: int = 0
        self.current_step: int = 0  # Track moves within the episode

        # State for interactive/debug modes (kept separate from core game logic)
        self.demo_selected_shape_idx: int = -1
        self.demo_snapped_position: Optional[Tuple[int, int]] = None
        self.debug_highlight_pos: Optional[Tuple[int, int]] = None
        # State for visualization during training (attached by worker)
        self.display_stats: Dict[str, Any] = {}

        self.reset()

    def reset(self):
        """Resets the game to the initial state."""
        self.grid_data = GridData(self.env_config)
        self.shapes = [None] * self.env_config.NUM_SHAPE_SLOTS
        self.game_score = 0.0
        self.triangles_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0
        self.game_over = False
        self.current_step = 0
        self.demo_selected_shape_idx = -1
        self.demo_snapped_position = None
        self.debug_highlight_pos = None
        self.display_stats = {}

        ShapeLogic.refill_shape_slots(self, self._rng)  # Uses ShapeLogic relatively

        if not self.valid_actions():
            logger.warning(
                "Game is over immediately after reset (no valid initial moves)."
            )
            self.game_over = True

    def step(self, action_index: ActionType) -> Tuple[float, float, bool]:
        """Performs one game step. Returns (value_estimate_placeholder, reward, done)."""
        if self.is_over():
            logger.warning("Attempted to step in a game that is already over.")
            return 0.0, 0.0, True

        shape_idx, r, c = decode_action(action_index, self.env_config)
        reward = execute_placement(self, shape_idx, r, c, self._rng)
        self.current_step += 1

        # Check for game over *after* placement and shape refill
        if not self.game_over and not self.valid_actions():
            self.game_over = True
            # Log only once when the state transitions to game over
            logger.info(f"Game over detected after step {self.current_step}.")

        # Placeholder value (0.0) is returned, actual value comes from NN/MCTS later
        return 0.0, reward, self.game_over

    def valid_actions(self) -> List[ActionType]:
        """Returns a list of valid encoded action indices."""
        return get_valid_actions(self)

    def is_over(self) -> bool:
        """Checks if the game is over."""
        return self.game_over

    def get_outcome(self) -> float:
        """Returns the terminal outcome value (e.g., final score). Used by MCTS."""
        if not self.is_over():
            logger.warning("get_outcome() called on a non-terminal state.")
        return self.game_score

    def copy(self) -> "GameState":
        """Creates a deep copy for simulations (e.g., MCTS)."""
        new_state = GameState.__new__(GameState)
        new_state.env_config = self.env_config
        # Correctly copy the RNG state
        new_state._rng = random.Random()  # Create a new instance
        new_state._rng.setstate(self._rng.getstate())  # Set its state
        new_state.grid_data = self.grid_data.deepcopy()
        new_state.shapes = [
            s.copy() if s else None for s in self.shapes
        ]  # Uses Shape from structs
        new_state.game_score = self.game_score
        new_state.game_over = self.game_over
        new_state.triangles_cleared_this_episode = self.triangles_cleared_this_episode
        new_state.pieces_placed_this_episode = self.pieces_placed_this_episode
        new_state.current_step = self.current_step
        new_state.demo_selected_shape_idx = self.demo_selected_shape_idx
        new_state.demo_snapped_position = self.demo_snapped_position
        new_state.debug_highlight_pos = self.debug_highlight_pos
        new_state.display_stats = self.display_stats.copy()
        return new_state

    def __str__(self) -> str:
        shape_strs = [str(s) if s else "None" for s in self.shapes]
        return f"GameState(Step:{self.current_step}, Score:{self.game_score:.1f}, Over:{self.is_over()}, Shapes:[{', '.join(shape_strs)}])"
