# File: src/environment/logic/step.py
import logging
import random
from typing import TYPE_CHECKING, Tuple, List, Set

# Use relative imports within environment package
from ..grid import logic as GridLogic
from ..shapes import logic as ShapeLogic

# Import Triangle and Shape from the new structs module
from src.structs import Triangle, Shape

if TYPE_CHECKING:
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def calculate_reward(
    num_placed_triangles: int, lines_cleared: int, triangles_cleared: int
) -> float:
    """
    Calculates the reward based on placed shape and cleared lines/triangles.

    Args:
        num_placed_triangles: Number of triangles in the shape that was placed.
        lines_cleared: Number of lines cleared by the placement.
        triangles_cleared: Total number of triangles cleared in those lines.

    Returns:
        The calculated reward value.
    """
    # Score: 1 point per placed triangle + 2 points per cleared triangle
    placement_score = num_placed_triangles * 1.0
    clear_score = triangles_cleared * 2.0

    # Optional: Add bonus for clearing multiple lines at once?
    line_bonus = 0.0
    if lines_cleared == 1:
        line_bonus = 0  # Base score already covers triangles
    elif lines_cleared == 2:
        line_bonus = 10  # Example bonus
    elif lines_cleared >= 3:
        line_bonus = 30  # Example bonus

    total_reward = placement_score + clear_score + line_bonus
    # logger.info(f"Reward Calc: Placed={num_placed_triangles} ({placement_score:.1f}), Cleared={triangles_cleared} ({clear_score:.1f}), Lines={lines_cleared} (Bonus={line_bonus:.1f}) -> Total={total_reward:.1f}")
    return total_reward


def execute_placement(
    game_state: "GameState", shape_idx: int, r: int, c: int, rng: random.Random
) -> float:
    """
    Places the selected shape, updates grid, clears lines, refills shapes,
    and calculates reward. Modifies game_state in place.

    Returns:
        The reward obtained from this placement.
    """
    shape_to_place: Shape | None = game_state.shapes[shape_idx]
    if not shape_to_place:
        logger.error(f"Attempted to place an empty shape slot: {shape_idx}")
        return 0.0  # No reward for invalid action

    if not GridLogic.can_place(game_state.grid_data, shape_to_place, r, c):
        logger.error(
            f"Attempted invalid placement: Shape {shape_idx} at ({r},{c}). Should have been caught by valid_actions."
        )
        # Consider adding a penalty here? For now, return 0 reward.
        game_state.game_over = True  # Invalid move leads to game over
        return -10.0  # Penalty for invalid move attempt

    # --- Place the shape ---
    newly_occupied_triangles: Set[Triangle] = set()
    num_placed_triangles = 0
    for dr, dc, is_up in shape_to_place.triangles:
        tri_r, tri_c = r + dr, c + dc
        if game_state.grid_data.valid(tri_r, tri_c):
            tri = game_state.grid_data.triangles[tri_r][tri_c]
            if not tri.is_occupied and not tri.is_death:
                tri.is_occupied = True
                tri.color = shape_to_place.color
                game_state.grid_data._occupied_np[tri_r, tri_c] = True
                newly_occupied_triangles.add(tri)
                num_placed_triangles += 1
            else:
                # This case should ideally not happen if can_place was checked
                logger.warning(
                    f"Overlap detected during placement at ({tri_r},{tri_c}) despite can_place check."
                )
        else:
            logger.warning(
                f"Triangle ({tri_r},{tri_c}) out of bounds during placement."
            )

    game_state.pieces_placed_this_episode += 1

    # --- Check for line clears ---
    lines_cleared, triangles_cleared, _ = GridLogic.check_and_clear_lines(
        game_state.grid_data, newly_occupied_triangles
    )
    game_state.triangles_cleared_this_episode += triangles_cleared

    # --- Calculate Reward ---
    # Pass the actual number of placed triangles to the reward function
    reward = calculate_reward(num_placed_triangles, lines_cleared, triangles_cleared)
    game_state.game_score += reward

    # --- Refill shape slot ---
    game_state.shapes[shape_idx] = None  # Clear the used slot
    ShapeLogic.refill_shape_slots(game_state, rng)

    # --- Check for game over (if refill fails or no valid moves remain) ---
    # Game over is checked *after* placement/refill in GameState.step()

    return reward
