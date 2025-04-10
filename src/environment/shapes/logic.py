# File: src/environment/shapes/logic.py
import random
import logging
from typing import TYPE_CHECKING, List, Tuple
from collections import deque

# Import Shape and SHAPE_COLORS from the new structs module
from src.structs import Shape, SHAPE_COLORS

if TYPE_CHECKING:
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def generate_random_shape(rng: random.Random) -> Shape:
    """
    Generates a random shape with 1 to 5 triangles, ensuring correct
    internal orientation alternation.
    """
    num_triangles = rng.randint(1, 5)
    color = rng.choice(SHAPE_COLORS)
    triangles: List[Tuple[int, int, bool]] = []

    # Start with a single triangle at relative (0, 0)
    start_r, start_c = 0, 0
    # Determine orientation based on relative parity (0+0)%2 == 0 -> is_up=True
    # This establishes the reference parity for the shape.
    start_is_up = True
    triangles.append((start_r, start_c, start_is_up))
    occupied_relative = {(start_r, start_c)}
    # Store (r, c, is_up) in the frontier
    frontier = deque([(start_r, start_c, start_is_up)])

    while len(triangles) < num_triangles and frontier:
        curr_r, curr_c, curr_is_up = frontier.popleft()

        # Define potential relative neighbor coordinates
        potential_neighbors_rel = [
            (curr_r, curr_c - 1),  # Left
            (curr_r, curr_c + 1),  # Right
            (curr_r + 1 if curr_is_up else curr_r - 1, curr_c),  # Vertical
        ]
        rng.shuffle(potential_neighbors_rel)

        for next_r, next_c in potential_neighbors_rel:
            if (next_r, next_c) not in occupied_relative:
                # Determine the required orientation for this relative position
                # based on the reference parity established by the start node.
                required_next_is_up = (next_r + next_c) % 2 == 0

                # Check compatibility with current triangle based on connection type
                is_compatible = False
                if (
                    next_r == curr_r
                ):  # Horizontal connection requires opposite orientation
                    is_compatible = required_next_is_up != curr_is_up
                elif next_c == curr_c:  # Vertical connection requires same orientation
                    is_compatible = required_next_is_up == curr_is_up

                if is_compatible:
                    # Add the new triangle if compatible
                    triangles.append((next_r, next_c, required_next_is_up))
                    occupied_relative.add((next_r, next_c))
                    frontier.append((next_r, next_c, required_next_is_up))
                    if len(triangles) == num_triangles:
                        break  # Reached desired size

        if len(triangles) == num_triangles:
            break

    # Normalize coordinates so the top-leftmost point is near (0,0)
    if not triangles:  # Should not happen, but safeguard
        return Shape([], color)

    min_r = min(t[0] for t in triangles)
    min_c = min(t[1] for t in triangles)
    normalized_triangles = [(r - min_r, c - min_c, is_up) for r, c, is_up in triangles]

    # Final check: Recalculate is_up based on normalized coords relative to the
    # new origin's parity to ensure consistency before returning.
    # The parity of the new origin (min_r, min_c) relative to the start (0,0) matters.
    # The parity of (r_norm, c_norm) should match the parity of (r, c).
    final_triangles = []
    for r_norm, c_norm, _ in normalized_triangles:
        # Original r = r_norm + min_r, Original c = c_norm + min_c
        # Original parity check: (r_norm + min_r + c_norm + min_c) % 2 == 0
        # This determines the correct is_up for the normalized coordinates.
        final_is_up = (r_norm + c_norm + min_r + min_c) % 2 == 0
        final_triangles.append((r_norm, c_norm, final_is_up))

    return Shape(final_triangles, color)


def refill_shape_slots(state: "GameState", rng: random.Random) -> None:
    """
    Refills shape slots *only* if all slots are currently empty.
    """
    # Check if all slots are empty (None)
    if all(s is None for s in state.shapes):
        logger.info("All shape slots are empty. Generating new shapes.")
        for i in range(state.env_config.NUM_SHAPE_SLOTS):
            state.shapes[i] = generate_random_shape(rng)
    # else: # If some slots still have shapes, do nothing
    #     logger.info("Some shape slots still contain shapes. Not refilling.")
