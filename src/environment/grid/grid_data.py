# File: src/environment/grid/grid_data.py
import numpy as np
from typing import List, Tuple, Set, Dict, Optional
import logging

# Use relative imports within environment package
from ...config import EnvConfig
from . import logic as GridLogic  # Import logic module

# Import Triangle from the new structs module
from src.structs import Triangle

logger = logging.getLogger(__name__)


class GridData:
    """Holds the grid state (triangles, occupancy, death zones)."""

    def __init__(self, config: EnvConfig):
        self.rows = config.ROWS
        self.cols = config.COLS
        self.config = config
        self.triangles: List[List[Triangle]] = self._create(
            config
        )  # Uses Triangle from structs
        # Call link_neighbors from the logic module
        GridLogic.link_neighbors(self)

        self._occupied_np = np.array(
            [[t.is_occupied for t in r] for r in self.triangles], dtype=bool
        )
        self._death_np = np.array(
            [[t.is_death for t in r] for r in self.triangles], dtype=bool
        )

        self.potential_lines: Set[frozenset[Triangle]] = (
            set()
        )  # Uses Triangle from structs
        self._triangle_to_lines_map: Dict[Triangle, Set[frozenset[Triangle]]] = (
            {}
        )  # Uses Triangle from structs
        # Call initialize_lines_and_index from the logic module
        GridLogic.initialize_lines_and_index(self)
        logger.info(
            f"GridData initialized ({self.rows}x{self.cols}). Found {len(self.potential_lines)} potential lines."
        )

    def _create(
        self, config: EnvConfig
    ) -> List[List[Triangle]]:  # Uses Triangle from structs
        """Initializes the grid, marking death cells based on COLS_PER_ROW."""
        cols_per_row = config.COLS_PER_ROW
        if len(cols_per_row) != self.rows:
            raise ValueError(
                f"COLS_PER_ROW length mismatch: {len(cols_per_row)} vs {self.rows}"
            )
        if max(cols_per_row, default=0) > self.cols:
            raise ValueError(
                f"Max COLS_PER_ROW exceeds COLS: {max(cols_per_row, default=0)} vs {self.cols}"
            )

        grid = []
        for r in range(self.rows):
            row_tris = []
            intended_width = cols_per_row[r]
            padding = self.cols - intended_width
            pad_left = padding // 2
            start_col = pad_left
            end_col = self.cols - (padding - pad_left)  # Exclusive end

            # Border columns are death cells
            actual_start = start_col + 1
            actual_end = end_col - 1  # Exclusive end

            for c in range(self.cols):
                is_playable = intended_width >= 3 and actual_start <= c < actual_end
                is_death = not is_playable
                is_up = (r + c) % 2 == 0
                row_tris.append(
                    Triangle(r, c, is_up, is_death=is_death)
                )  # Uses Triangle from structs
            grid.append(row_tris)
        return grid

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_occupied_state(self) -> np.ndarray:
        """Returns a copy of the occupancy numpy array."""
        return self._occupied_np.copy()

    def get_death_state(self) -> np.ndarray:
        """Returns a copy of the death zone numpy array."""
        return self._death_np.copy()

    def deepcopy(self) -> "GridData":
        """Creates a deep copy of the grid data."""
        new_grid = GridData.__new__(GridData)
        new_grid.rows = self.rows
        new_grid.cols = self.cols
        new_grid.config = self.config
        new_grid.triangles = [
            [tri.copy() for tri in row] for row in self.triangles
        ]  # Uses Triangle from structs
        # Relink neighbors in the new copy using logic module
        GridLogic.link_neighbors(new_grid)
        new_grid._occupied_np = self._occupied_np.copy()
        new_grid._death_np = self._death_np.copy()
        # Re-initialize line structures using logic module
        new_grid.potential_lines = set()
        new_grid._triangle_to_lines_map = {}
        GridLogic.initialize_lines_and_index(new_grid)
        return new_grid

    def __str__(self) -> str:
        # Simple representation for debugging
        return f"GridData({self.rows}x{self.cols})"
