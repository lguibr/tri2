# File: src/environment/logic/actions.py
import logging
from typing import List, TYPE_CHECKING

# Use relative imports within environment package
from ..core.action_codec import encode_action
from ..grid import logic as GridLogic  # Use grid logic for can_place

if TYPE_CHECKING:
    from ..core.game_state import GameState
    from ...utils.types import ActionType

logger = logging.getLogger(__name__)


def get_valid_actions(state: "GameState") -> List["ActionType"]:
    """
    Calculates and returns a list of all valid encoded action indices
    for the current game state.
    """
    valid_actions: List["ActionType"] = []
    # Iterate through each shape slot
    for shape_idx, shape in enumerate(state.shapes):
        if shape is None:
            continue  # Skip empty slots

        # Iterate through all possible anchor points (r, c) on the grid
        for r in range(state.env_config.ROWS):
            for c in range(state.env_config.COLS):
                # Check if the shape can be placed at this anchor point
                if GridLogic.can_place(state.grid_data, shape, r, c):
                    # If valid, encode the action and add it to the list
                    action_index = encode_action(shape_idx, r, c, state.env_config)
                    valid_actions.append(action_index)

    # logger.info(f"Found {len(valid_actions)} valid actions for step {state.current_step}.")
    return valid_actions
