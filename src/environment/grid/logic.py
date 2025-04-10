# File: src/environment/grid/logic.py
import logging
from typing import List, Tuple, Set, Dict, Optional, Deque
from collections import deque  # Import deque for BFS

# Use relative imports within environment package
from ...config import EnvConfig

# Import Triangle and Shape from the new structs module
from src.structs import Triangle, Shape

# Import GridData for type hinting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .grid_data import GridData

logger = logging.getLogger(__name__)


def link_neighbors(grid_data: "GridData") -> None:
    """Sets neighbor references for each triangle in the GridData."""
    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            tri = grid_data.triangles[r][c]
            if grid_data.valid(r, c - 1):
                tri.neighbor_left = grid_data.triangles[r][c - 1]
            if grid_data.valid(r, c + 1):
                tri.neighbor_right = grid_data.triangles[r][c + 1]
            nr, nc = (r + 1, c) if tri.is_up else (r - 1, c)
            if grid_data.valid(nr, nc):
                tri.neighbor_vert = grid_data.triangles[nr][nc]


def _get_line_neighbors(tri: Triangle, direction: str) -> List[Triangle]:
    """Helper to get relevant neighbors for line tracing in a specific direction."""
    neighbors = []
    if direction == "horizontal":
        if tri.neighbor_left:
            neighbors.append(tri.neighbor_left)
        if tri.neighbor_right:
            neighbors.append(tri.neighbor_right)
    elif direction == "diag1":  # Diagonal: Top-left to Bottom-right-ish
        if tri.is_up:
            # Connects to down-pointing neighbors via left side or vertical tip
            if tri.neighbor_left and not tri.neighbor_left.is_up:
                neighbors.append(tri.neighbor_left)
            if tri.neighbor_vert and not tri.neighbor_vert.is_up:
                neighbors.append(tri.neighbor_vert)
        else:  # Down-pointing triangle
            # Connects to up-pointing neighbors via right side or vertical tip
            if tri.neighbor_right and tri.neighbor_right.is_up:
                neighbors.append(tri.neighbor_right)
            if tri.neighbor_vert and tri.neighbor_vert.is_up:
                neighbors.append(tri.neighbor_vert)
    elif direction == "diag2":  # Diagonal: Top-right to Bottom-left-ish
        if tri.is_up:
            # Connects to down-pointing neighbors via right side or vertical tip
            if tri.neighbor_right and not tri.neighbor_right.is_up:
                neighbors.append(tri.neighbor_right)
            if tri.neighbor_vert and not tri.neighbor_vert.is_up:
                neighbors.append(tri.neighbor_vert)
        else:  # Down-pointing triangle
            # Connects to up-pointing neighbors via left side or vertical tip
            if tri.neighbor_left and tri.neighbor_left.is_up:
                neighbors.append(tri.neighbor_left)
            if tri.neighbor_vert and tri.neighbor_vert.is_up:
                neighbors.append(tri.neighbor_vert)
    # Filter out death cells from potential line neighbors
    return [n for n in neighbors if not n.is_death]


def initialize_lines_and_index(grid_data: "GridData") -> None:
    """
    Identifies all sets of playable triangles forming potential lines
    by tracing connections along horizontal and diagonal axes using BFS.
    Populates grid_data.potential_lines and grid_data._triangle_to_lines_map.
    """
    grid_data.potential_lines = set()
    grid_data._triangle_to_lines_map = {}
    visited_in_direction: Dict[str, Set[Triangle]] = {
        "horizontal": set(),
        "diag1": set(),
        "diag2": set(),
    }
    min_line_length = grid_data.config.MIN_LINE_LENGTH

    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            start_node = grid_data.triangles[r][c]
            if start_node.is_death:
                continue

            for direction in ["horizontal", "diag1", "diag2"]:
                if start_node not in visited_in_direction[direction]:
                    current_line: Set[Triangle] = set()
                    queue: Deque[Triangle] = deque([start_node])
                    visited_this_bfs: Set[Triangle] = {start_node}

                    while queue:
                        tri = queue.popleft()
                        # Only add non-death triangles to the line itself
                        if not tri.is_death:
                            current_line.add(tri)
                        # Mark as visited for this direction to avoid redundant BFS starts
                        visited_in_direction[direction].add(tri)

                        neighbors = _get_line_neighbors(tri, direction)
                        for neighbor in neighbors:
                            # Explore neighbor if not visited in *this specific BFS run*
                            if neighbor not in visited_this_bfs:
                                visited_this_bfs.add(neighbor)
                                queue.append(neighbor)

                    # Store the line if it meets the minimum length requirement
                    if len(current_line) >= min_line_length:
                        line_frozenset = frozenset(current_line)
                        grid_data.potential_lines.add(line_frozenset)
                        # Update the index map for each triangle in the valid line
                        for tri_in_line in current_line:
                            if tri_in_line not in grid_data._triangle_to_lines_map:
                                grid_data._triangle_to_lines_map[tri_in_line] = set()
                            grid_data._triangle_to_lines_map[tri_in_line].add(
                                line_frozenset
                            )


def can_place(grid_data: "GridData", shape: Shape, r: int, c: int) -> bool:
    """Checks if a shape can be placed at the target location."""
    for dr, dc, is_up_shape in shape.triangles:
        nr, nc = r + dr, c + dc
        if not grid_data.valid(nr, nc):
            return False
        # Use pre-computed numpy arrays for faster checks
        if (
            grid_data._death_np[nr, nc]
            or grid_data._occupied_np[nr, nc]
            or (grid_data.triangles[nr][nc].is_up != is_up_shape)
        ):
            return False
    return True


def check_and_clear_lines(
    grid_data: "GridData", newly_occupied_triangles: Set[Triangle]
) -> Tuple[int, int, List[Tuple[int, int]]]:
    """
    Checks for completed lines based on newly occupied triangles.
    Clears all triangles belonging to *any* completed line.
    Counts lines cleared and *total* triangles involved in those lines (including duplicates).
    Returns lines cleared count, total triangles cleared count (for scoring),
    and coordinates of unique cleared triangles.
    """
    lines_to_check: Set[frozenset[Triangle]] = set()
    if newly_occupied_triangles:
        for tri in newly_occupied_triangles:
            if tri in grid_data._triangle_to_lines_map:
                lines_to_check.update(grid_data._triangle_to_lines_map[tri])
    else:
        # If no specific triangles provided, check all potential lines (less efficient)
        lines_to_check = grid_data.potential_lines

    completed_lines: List[frozenset[Triangle]] = []
    for line_set in lines_to_check:
        if not line_set:
            continue
        # Check occupancy using the numpy array for speed
        if all(grid_data._occupied_np[tri.row, tri.col] for tri in line_set):
            completed_lines.append(line_set)

    if not completed_lines:
        return 0, 0, []

    lines_cleared_count = len(completed_lines)
    all_tris_in_cleared_lines: List[Triangle] = []
    cleared_tris_unique: Set[Triangle] = set()

    # Collect all triangles from all completed lines
    for line_set in completed_lines:
        all_tris_in_cleared_lines.extend(list(line_set))
        cleared_tris_unique.update(line_set)

    # Count total triangles for scoring (includes duplicates)
    tris_cleared_count_for_score = len(all_tris_in_cleared_lines)

    # Clear the unique set of triangles
    coords: List[Tuple[int, int]] = []
    for tri in cleared_tris_unique:
        # Check using numpy array first (should be occupied based on above check)
        if (
            not grid_data._death_np[tri.row, tri.col]
            and grid_data._occupied_np[tri.row, tri.col]
        ):
            # Update Triangle object state
            tri.is_occupied = False
            tri.color = None
            # Update internal numpy array state
            grid_data._occupied_np[tri.row, tri.col] = False
            coords.append((tri.row, tri.col))

    if lines_cleared_count > 0:
        logger.info(
            f"Cleared {lines_cleared_count} lines, involving {tris_cleared_count_for_score} total triangles (cleared {len(cleared_tris_unique)} unique)."
        )

    # Return lines count, total triangles count (for score), and unique coords
    return lines_cleared_count, tris_cleared_count_for_score, coords
