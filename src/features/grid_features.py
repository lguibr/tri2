# File: src/features/grid_features.py
# (Content moved from src/environment/grid/grid_features.py)
# Note: Ensure Numba is installed (`pip install numba`)
import numpy as np
import numba
from numba import njit, prange

# --- Numba Optimized Grid Feature Calculations ---
# These functions operate on numpy arrays for speed.


@njit(cache=True)
def get_column_heights(
    occupied: np.ndarray, death: np.ndarray, rows: int, cols: int
) -> np.ndarray:
    """Calculates the height of each column (highest occupied non-death cell)."""
    heights = np.zeros(cols, dtype=np.int32)
    for c in prange(cols):
        max_r = -1
        for r in range(rows):
            # Check if the cell is occupied AND not a death cell
            if occupied[r, c] and not death[r, c]:
                max_r = r
        # Height is row index + 1 (or 0 if column is empty/all death)
        heights[c] = max_r + 1
    return heights


@njit(cache=True)
def count_holes(
    occupied: np.ndarray, death: np.ndarray, heights: np.ndarray, rows: int, cols: int
) -> int:
    """Counts the number of empty, non-death cells below the column height."""
    holes = 0
    for c in prange(cols):
        col_height = heights[c]
        # Iterate from row 0 up to (but not including) the column height
        for r in range(col_height):
            # A hole is an empty cell that is not a death cell, below the highest block
            if not occupied[r, c] and not death[r, c]:
                holes += 1
    return holes


@njit(cache=True)
def get_bumpiness(heights: np.ndarray) -> float:
    """Calculates the total absolute difference between adjacent column heights."""
    bumpiness = 0.0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness


# Add other potential Numba-optimized grid features here if needed
# e.g., completed lines (might be complex with triangles), wells, etc.
