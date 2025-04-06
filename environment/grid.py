import numpy as np
from typing import List, Tuple, Set, Dict

from config import EnvConfig


from .triangle import Triangle
from .shape import Shape


class Grid:
    """Represents the game board composed of Triangles."""

    def __init__(self, env_config: EnvConfig): 
        self.rows = env_config.ROWS
        self.cols = env_config.COLS
        self.triangles: List[List[Triangle]] = []
        self._create(env_config)  # Pass config to _create
        self._link_neighbors()  # Link neighbors after creation
        self._identify_playable_lines()  # Identify all potential lines

    def _create(self, env_config: EnvConfig) -> None:
        """Initializes the grid with playable and death cells."""
        # Example pattern for a hexagon-like board within the grid bounds
        cols_per_row = [9, 11, 13, 15, 15, 13, 11, 9]  # Specific to 8 rows

        if len(cols_per_row) != self.rows:
            raise ValueError(
                f"Grid._create error: Length of cols_per_row ({len(cols_per_row)}) must match EnvConfig.ROWS ({self.rows})"
            )
        if max(cols_per_row) > self.cols:
            raise ValueError(
                f"Grid._create error: Max playable columns ({max(cols_per_row)}) exceeds EnvConfig.COLS ({self.cols})"
            )

        self.triangles = []
        for r in range(self.rows):
            row_triangles: List[Triangle] = []
            base_playable_cols = cols_per_row[r]

            # Calculate padding for death cells
            initial_death_cols_left = (
                (self.cols - base_playable_cols) // 2
                if base_playable_cols < self.cols
                else 0
            )
            initial_first_death_col_right = initial_death_cols_left + base_playable_cols

            # Adjustment for Specific Hex Grid Pattern (makes it slightly narrower)
            adjusted_death_cols_left = initial_death_cols_left + 1
            adjusted_first_death_col_right = initial_first_death_col_right - 1

            for c in range(self.cols):
                is_death_cell = (
                    (c < adjusted_death_cols_left)
                    or (
                        c >= adjusted_first_death_col_right
                        and adjusted_first_death_col_right > adjusted_death_cols_left
                    )
                    or (base_playable_cols <= 2)  # Treat very narrow rows as death
                )
                is_up_cell = (r + c) % 2 == 0  
                triangle = Triangle(r, c, is_up=is_up_cell, is_death=is_death_cell)
                row_triangles.append(triangle)
            self.triangles.append(row_triangles)

    def _link_neighbors(self) -> None:
        """Iterates through the grid and sets neighbor references for each triangle."""
        for r in range(self.rows):
            for c in range(self.cols):
                current_tri = self.triangles[r][c]

                # Neighbor Left (X)
                if self.valid(r, c - 1):
                    current_tri.neighbor_left = self.triangles[r][c - 1]

                # Neighbor Right (Y)
                if self.valid(r, c + 1):
                    current_tri.neighbor_right = self.triangles[r][c + 1]

                # Neighbor Vertical (Z)
                if current_tri.is_up:
                    # Up triangle's vertical neighbor is below
                    if self.valid(r + 1, c):
                        current_tri.neighbor_vert = self.triangles[r + 1][c]
                else:
                    # Down triangle's vertical neighbor is above
                    if self.valid(r - 1, c):
                        current_tri.neighbor_vert = self.triangles[r - 1][c]

    def _identify_playable_lines(self):
        """
        Identifies all sets of playable triangles that form a complete line
        along horizontal (1-thick) and diagonal (2-thick) axes.
        Stores these sets for efficient checking later.
        """
        self.potential_lines: List[Set[Triangle]] = []

        # 1. Horizontal Lines (1-thick)
        for r in range(self.rows):
            line_triangles: List[Triangle] = []
            is_playable_row = False
            for c in range(self.cols):
                tri = self.triangles[r][c]
                if not tri.is_death:
                    line_triangles.append(tri)
                    is_playable_row = True
                else:
                    # If we hit a death cell after finding playable cells, store the line segment
                    if line_triangles:
                        self.potential_lines.append(set(line_triangles))
                        line_triangles = []
                    # Reset if the row started with death cells
                    if not is_playable_row:
                        line_triangles = []

            # Add the last segment if the row ended with playable cells
            if line_triangles:
                self.potential_lines.append(set(line_triangles))

        # Helper function to generate single diagonal lines
        def get_single_diagonal_lines(
            k_func, r_range, c_func
        ) -> Dict[int, Set[Triangle]]:
            single_lines: Dict[int, Set[Triangle]] = {}
            min_k = float("inf")
            max_k = float("-inf")
            # Determine k range by checking all valid cells
            for r_check in range(self.rows):
                for c_check in range(self.cols):
                    if (
                        self.valid(r_check, c_check)
                        and not self.triangles[r_check][c_check].is_death
                    ):
                        k_val = k_func(r_check, c_check)
                        min_k = min(min_k, k_val)
                        max_k = max(max_k, k_val)

            if min_k > max_k:  # No playable cells found
                return {}

            for k in range(min_k, max_k + 1):
                line_triangles: List[Triangle] = []
                is_playable_line = False
                for r in r_range:
                    c = c_func(k, r)
                    if self.valid(r, c):
                        tri = self.triangles[r][c]
                        if not tri.is_death:
                            line_triangles.append(tri)
                            is_playable_line = True
                        else:
                            if line_triangles:  # End of a playable segment
                                if k not in single_lines:
                                    single_lines[k] = set()
                                single_lines[k].update(line_triangles)
                            if not is_playable_line:  # Reset if started with death
                                line_triangles = []
                            else:  # Break segment
                                line_triangles = []
                    elif (
                        line_triangles
                    ):  # End of playable segment due to invalid coords
                        if k not in single_lines:
                            single_lines[k] = set()
                        single_lines[k].update(line_triangles)
                        line_triangles = []

                if line_triangles:  # Add last segment
                    if k not in single_lines:
                        single_lines[k] = set()
                    single_lines[k].update(line_triangles)
            return single_lines

        # 2. Diagonal Lines TL-BR (k = c - r) - Combine adjacent k and k+1
        single_diag_tlbr = get_single_diagonal_lines(
            k_func=lambda r, c: c - r,
            r_range=range(self.rows),
            c_func=lambda k, r: k + r,
        )
        # Determine the actual range of k present in the dictionary keys
        k_values_tlbr = sorted(single_diag_tlbr.keys())
        if k_values_tlbr:
            for i in range(len(k_values_tlbr) - 1):
                k1 = k_values_tlbr[i]
                k2 = k_values_tlbr[i + 1]
                # Check if keys are adjacent (k+1)
                if k2 == k1 + 1:
                    # Combine the sets for the 2-thick diagonal line
                    combined_line = single_diag_tlbr[k1].union(single_diag_tlbr[k2])
                    if combined_line:  # Ensure not empty
                        self.potential_lines.append(combined_line)

        # 3. Diagonal Lines TR-BL (k = r + c) - Combine adjacent k and k+1
        single_diag_trbl = get_single_diagonal_lines(
            k_func=lambda r, c: r + c,
            r_range=range(self.rows),
            c_func=lambda k, r: k - r,
        )
        # Determine the actual range of k present in the dictionary keys
        k_values_trbl = sorted(single_diag_trbl.keys())
        if k_values_trbl:
            for i in range(len(k_values_trbl) - 1):
                k1 = k_values_trbl[i]
                k2 = k_values_trbl[i + 1]
                # Check if keys are adjacent (k+1)
                if k2 == k1 + 1:
                    # Combine the sets for the 2-thick diagonal line
                    combined_line = single_diag_trbl[k1].union(single_diag_trbl[k2])
                    if combined_line:  # Ensure not empty
                        self.potential_lines.append(combined_line)

        # Filter out potential lines that might be empty due to edge cases
        self.potential_lines = [line for line in self.potential_lines if line]
        # print(f"[Grid] Identified {len(self.potential_lines)} potential playable lines (H: 1-thick, D: 2-thick).")

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def can_place(
        self, shape_to_place: Shape, target_row: int, target_col: int
    ) -> bool:
        """Checks if a shape can be placed at the target location."""
        for dr, dc, is_up_shape_tri in shape_to_place.triangles:
            nr, nc = target_row + dr, target_col + dc
            if not self.valid(nr, nc):
                return False  # Out of bounds
            grid_triangle = self.triangles[nr][nc]
            # Cannot place on death cells, occupied cells, or cells with mismatching orientation
            if (
                grid_triangle.is_death
                or grid_triangle.is_occupied
                or (grid_triangle.is_up != is_up_shape_tri)
            ):
                return False
        return True  # All shape triangles can be placed

    def place(self, shape_to_place: Shape, target_row: int, target_col: int) -> None:
        """Places a shape onto the grid (assumes can_place was checked)."""
        for dr, dc, _ in shape_to_place.triangles:
            nr, nc = target_row + dr, target_col + dc
            if self.valid(nr, nc):
                grid_triangle = self.triangles[nr][nc]
                # Only occupy non-death, non-occupied cells
                if not grid_triangle.is_death and not grid_triangle.is_occupied:
                    grid_triangle.is_occupied = True
                    grid_triangle.color = shape_to_place.color

    def clear_lines(self) -> Tuple[int, int, List[Tuple[int, int]]]:
        """
        Checks for completed lines based on pre-identified potential lines.
        Clears all triangles belonging to any completed lines simultaneously.
        Returns the number of lines cleared, total triangles cleared, and their coordinates.
        """
        cleared_triangles_in_this_step: Set[Triangle] = set()
        lines_cleared_count = 0

        # Iterate through all potential lines identified during initialization
        for line_set in self.potential_lines:
            is_complete = True
            if not line_set:
                is_complete = False
            else:
                for triangle in line_set:
                    if not triangle.is_occupied:
                        is_complete = False
                        break

            if is_complete:
                cleared_triangles_in_this_step.update(line_set)
                lines_cleared_count += 1

        triangles_cleared_count = 0
        cleared_triangles_coords: List[Tuple[int, int]] = []

        if not cleared_triangles_in_this_step:
            return 0, 0, []  # Return 3 values even if none cleared

        for triangle in cleared_triangles_in_this_step:
            if not triangle.is_death and triangle.is_occupied:
                triangles_cleared_count += 1
                triangle.is_occupied = False
                triangle.color = None
                cleared_triangles_coords.append((triangle.row, triangle.col))

        return lines_cleared_count, triangles_cleared_count, cleared_triangles_coords


    def get_column_heights(self) -> List[int]:
        """Calculates the height of occupied cells in each column."""
        heights = [0] * self.cols
        for c in range(self.cols):
            for r in range(self.rows - 1, -1, -1):  # Iterate from top down
                triangle = self.triangles[r][c]
                if triangle.is_occupied and not triangle.is_death:
                    heights[c] = r + 1  # Height is row index + 1
                    break  # Found highest occupied cell in this column
        return heights

    def get_max_height(self) -> int:
        """Returns the maximum height across all columns."""
        heights = self.get_column_heights()
        return max(heights) if heights else 0

    def get_bumpiness(self) -> int:
        """Calculates the total absolute difference between adjacent column heights."""
        heights = self.get_column_heights()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness

    def count_holes(self) -> int:
        """Counts the number of empty, non-death cells below an occupied cell in the same column."""
        holes = 0
        for c in range(self.cols):
            occupied_above_found = False
            for r in range(self.rows - 1, -1, -1):  # Iterate from top down
                triangle = self.triangles[r][c]
                if triangle.is_death:
                    occupied_above_found = (
                        False  # Reset if we hit a death cell column top
                    )
                    continue  # Skip death cells

                if triangle.is_occupied:
                    occupied_above_found = True
                elif not triangle.is_occupied and occupied_above_found:
                    # Found an empty cell below an occupied one in this column
                    holes += 1
        return holes

    def get_feature_matrix(self) -> np.ndarray:
        """Returns the grid state as a 2-channel numpy array (Occupancy, Orientation)."""
        # Channel 0: Occupancy (1.0 if occupied and not death, 0.0 otherwise)
        # Channel 1: Orientation (1.0 if pointing up and not death, 0.0 otherwise)
        grid_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                triangle = self.triangles[r][c]
                if not triangle.is_death:
                    grid_state[0, r, c] = 1.0 if triangle.is_occupied else 0.0
                    grid_state[1, r, c] = 1.0 if triangle.is_up else 0.0
        return grid_state
