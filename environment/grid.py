# File: environment/grid.py
import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Deque
from collections import deque  # Import deque for BFS

from config import EnvConfig
from .triangle import Triangle
from .shape import Shape


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig):
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.triangles: List[List[Triangle]] = self._create(env_config)
        self._link_neighbors()
        # Store potential lines as frozensets for easy hashing/set operations
        self.potential_lines: Set[frozenset[Triangle]] = self._identify_playable_lines()

    def _create(self, env_config: EnvConfig) -> List[List[Triangle]]:
        """
        Initializes the grid with playable and death cells.
        The playable area defined by cols_per_row is further reduced by
        making the first and last cell of that range into death cells.
        """
        # This list now defines the width *before* trimming the ends.
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]
        if len(cols_per_row) != self.rows:
            raise ValueError("cols_per_row length mismatch")
        # Check if the *intended* width exceeds COLUMNS, not the trimmed width
        if max(cols_per_row) > self.cols:
            raise ValueError("cols_per_row exceeds EnvConfig.COLS")

        grid: List[List[Triangle]] = []
        for r in range(self.rows):
            row_tris: List[Triangle] = []
            # Get the intended number of playable cells for this row before trimming
            intended_playable_width = cols_per_row[r]

            # Calculate padding based on the *intended* width
            total_padding = self.cols - intended_playable_width
            pad_l = total_padding // 2
            # Calculate the index marking the start of the right padding
            pad_r = self.cols - (total_padding - pad_l)

            # Define the actual playable column range by trimming one cell from each end
            # The first playable column index
            playable_start_col = pad_l + 1
            # The first non-playable column index after the playable segment
            playable_end_col = pad_r - 1

            for c in range(self.cols):
                # Determine if the cell is playable or a death cell
                # A cell is playable if its index c is within [playable_start_col, playable_end_col)
                # and if the intended width was greater than 2 (otherwise trimming makes it 0 or less)
                is_playable = (
                    intended_playable_width > 2
                    and playable_start_col <= c < playable_end_col
                )

                is_death = not is_playable
                is_up = (r + c) % 2 == 0
                row_tris.append(Triangle(r, c, is_up=is_up, is_death=is_death))
            grid.append(row_tris)
        return grid

    # --- Rest of the Grid class remains the same as the previous fix ---
    # (Includes _link_neighbors, _get_line_neighbors, _identify_playable_lines,
    #  valid, can_place, place, clear_lines, get_column_heights, get_max_height,
    #  get_bumpiness, count_holes, get_feature_matrix, get_color_data, get_death_data)
    # --- Make sure to copy the corrected methods from the previous response here ---

    def _link_neighbors(self) -> None:
        """Sets neighbor references for each triangle."""
        for r in range(self.rows):
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if self.valid(r, c - 1):
                    tri.neighbor_left = self.triangles[r][c - 1]
                if self.valid(r, c + 1):
                    tri.neighbor_right = self.triangles[r][c + 1]
                nr, nc = (r + 1, c) if tri.is_up else (r - 1, c)
                if self.valid(nr, nc):
                    tri.neighbor_vert = self.triangles[nr][nc]

    def _get_line_neighbors(self, tri: Triangle, direction: str) -> List[Triangle]:
        """Helper to get relevant neighbors for line tracing in a specific direction."""
        neighbors = []
        if direction == "horizontal":
            if tri.neighbor_left:
                neighbors.append(tri.neighbor_left)
            if tri.neighbor_right:
                neighbors.append(tri.neighbor_right)
        elif direction == "diag1":  # Top-left to Bottom-right style ('/')
            if tri.is_up:
                if tri.neighbor_left and not tri.neighbor_left.is_up:
                    neighbors.append(tri.neighbor_left)
                if tri.neighbor_vert and not tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
            else:  # Down triangle
                if tri.neighbor_right and tri.neighbor_right.is_up:
                    neighbors.append(tri.neighbor_right)
                if tri.neighbor_vert and tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
        elif direction == "diag2":  # Top-right to Bottom-left style ('\')
            if tri.is_up:
                if tri.neighbor_right and not tri.neighbor_right.is_up:
                    neighbors.append(tri.neighbor_right)
                if tri.neighbor_vert and not tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
            else:  # Down triangle
                if tri.neighbor_left and tri.neighbor_left.is_up:
                    neighbors.append(tri.neighbor_left)
                if tri.neighbor_vert and tri.neighbor_vert.is_up:
                    neighbors.append(tri.neighbor_vert)
        # Filter out neighbors that are death cells, as they cannot be part of a line
        return [n for n in neighbors if not n.is_death]

    def _identify_playable_lines(self) -> Set[frozenset[Triangle]]:
        """
        Identifies all sets of playable triangles forming potential lines
        by tracing connections along horizontal and diagonal axes.
        Uses BFS to find maximal contiguous lines.
        """
        potential_lines: Set[frozenset[Triangle]] = set()
        visited_in_direction: Dict[str, Set[Triangle]] = {
            "horizontal": set(),
            "diag1": set(),
            "diag2": set(),
        }
        min_line_length = 3  # Define a minimum length for a set to be considered a line

        for r in range(self.rows):
            for c in range(self.cols):
                start_node = self.triangles[r][c]

                if start_node.is_death:
                    continue

                for direction in ["horizontal", "diag1", "diag2"]:
                    if start_node not in visited_in_direction[direction]:
                        # Start BFS from this node for this direction
                        current_line: Set[Triangle] = set()
                        queue: Deque[Triangle] = deque([start_node])
                        visited_this_bfs: Set[Triangle] = {start_node}

                        while queue:
                            tri = queue.popleft()
                            # Only add non-death triangles to the line being built
                            if not tri.is_death:
                                current_line.add(tri)
                            # Still mark visited globally even if it's a death cell explored from
                            # (though BFS starts only on non-death cells)
                            visited_in_direction[direction].add(tri)

                            # Get neighbors (already filtered for death cells by _get_line_neighbors)
                            neighbors = self._get_line_neighbors(tri, direction)
                            for neighbor in neighbors:
                                # Check neighbor is not already visited in *this specific BFS run*
                                if neighbor not in visited_this_bfs:
                                    visited_this_bfs.add(neighbor)
                                    queue.append(neighbor)

                        # Add the found line if it meets the minimum length
                        # Ensure the line actually contains non-death triangles
                        if len(current_line) >= min_line_length:
                            potential_lines.add(frozenset(current_line))

        return potential_lines

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(self, shape: Shape, r: int, c: int) -> bool:
        """Checks if a shape can be placed at the target location."""
        for dr, dc, is_up_shape in shape.triangles:
            nr, nc = r + dr, c + dc
            if not self.valid(nr, nc):
                return False
            tri = self.triangles[nr][nc]
            # Check death, occupied, AND orientation match
            if tri.is_death or tri.is_occupied or (tri.is_up != is_up_shape):
                return False
        return True

    def place(self, shape: Shape, r: int, c: int) -> None:
        """Places a shape onto the grid (assumes can_place was checked)."""
        for dr, dc, _ in shape.triangles:
            nr, nc = r + dr, c + dc
            if self.valid(nr, nc):
                tri = self.triangles[nr][nc]
                if not tri.is_death and not tri.is_occupied:
                    tri.is_occupied = True
                    tri.color = shape.color

    def clear_lines(self) -> Tuple[int, int, List[Tuple[int, int]]]:
        """Checks and clears completed lines. Returns lines, tris cleared, coords."""
        cleared_tris_total: Set[Triangle] = set()
        lines_cleared_count = 0

        for line_set in self.potential_lines:
            # line_set should already contain only non-death triangles due to _identify_playable_lines logic
            if not line_set:  # Skip empty sets if they somehow occur
                continue

            # Check if ALL triangles in this potential line are occupied
            if all(tri.is_occupied for tri in line_set):
                cleared_tris_total.update(line_set)
                lines_cleared_count += 1

        tris_cleared_count = 0
        coords: List[Tuple[int, int]] = []
        if not cleared_tris_total:
            return 0, 0, []

        for tri in cleared_tris_total:
            # Double check is_occupied and not is_death before clearing
            if not tri.is_death and tri.is_occupied:
                tris_cleared_count += 1
                tri.is_occupied = False
                tri.color = None
                coords.append((tri.row, tri.col))

        return lines_cleared_count, tris_cleared_count, coords

    def get_column_heights(self) -> List[int]:
        """Calculates the height of occupied cells in each column."""
        heights = [0] * self.cols
        for c in range(self.cols):
            col_max_r = -1
            for r in range(self.rows):
                if (
                    not self.triangles[r][c].is_death
                    and self.triangles[r][c].is_occupied
                ):
                    col_max_r = max(col_max_r, r)
            heights[c] = col_max_r + 1
        return heights

    def get_max_height(self) -> int:
        """Returns the maximum height across all columns."""
        heights = self.get_column_heights()
        return max(heights) if heights else 0

    def get_bumpiness(self) -> int:
        """Calculates the total absolute difference between adjacent column heights."""
        heights = self.get_column_heights()
        bumpiness = 0
        # Iterate through adjacent columns to calculate bumpiness
        for i in range(len(heights) - 1):
            # Consider bumpiness between all adjacent columns based on their heights
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def count_holes(self) -> int:
        """Counts empty, non-death cells below the highest occupied cell in the same column."""
        holes = 0
        heights = self.get_column_heights()
        for c in range(self.cols):
            # Only check for holes up to the calculated height of the column
            # If height is 0, range(0) is empty, so no holes checked.
            for r in range(heights[c]):  # Iterate from row 0 up to height-1
                tri = self.triangles[r][c]
                # A hole is a non-death cell that is NOT occupied but is BELOW the column height
                if not tri.is_death and not tri.is_occupied:
                    holes += 1
        return holes

    def get_feature_matrix(self) -> np.ndarray:
        """Returns the grid state as a 2-channel numpy array (Occupancy, Orientation)."""
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if not tri.is_death:
                    grid_state[0, r, c] = 1.0 if tri.is_occupied else 0.0
                    grid_state[1, r, c] = 1.0 if tri.is_up else -1.0  # Or 0.0
        return grid_state

    def get_color_data(self) -> List[List[Optional[Tuple[int, int, int]]]]:
        """Returns a 2D list of colors for occupied cells."""
        color_data = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if tri.is_occupied and not tri.is_death:
                    color_data[r][c] = tri.color
        return color_data

    def get_death_data(self) -> np.ndarray:
        """Returns a boolean numpy array indicating death cells."""
        death_data = np.zeros((self.rows, self.cols), dtype=bool)
        for r in range(self.rows):
            for c in range(self.cols):
                death_data[r, c] = self.triangles[r][c].is_death
        return death_data
