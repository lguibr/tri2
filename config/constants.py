"""
Defines constants shared across different modules, primarily visual elements,
to avoid circular imports and keep configuration clean.
"""

# Colors (RGB tuples 0-255)
WHITE: tuple[int, int, int] = (255, 255, 255)
BLACK: tuple[int, int, int] = (0, 0, 0)
LIGHTG: tuple[int, int, int] = (140, 140, 140)
GRAY: tuple[int, int, int] = (50, 50, 50)
DARK_GRAY: tuple[int, int, int] = (30, 30, 30)
RED: tuple[int, int, int] = (255, 50, 50)
DARK_RED: tuple[int, int, int] = (80, 10, 10)
BLUE: tuple[int, int, int] = (50, 50, 255)
YELLOW: tuple[int, int, int] = (255, 255, 100)
GREEN: tuple[int, int, int] = (50, 200, 50)
DARK_GREEN: tuple[int, int, int] = (20, 80, 20)  # Added Dark Green
ORANGE: tuple[int, int, int] = (255, 165, 0)
PURPLE: tuple[int, int, int] = (128, 0, 128)
CYAN: tuple[int, int, int] = (0, 255, 255)

GOOGLE_COLORS: list[tuple[int, int, int]] = [
    (15, 157, 88),  # Green
    (244, 180, 0),  # Yellow/Orange
    (66, 133, 244),  # Blue
    (219, 68, 55),  # Red
]
LINE_CLEAR_FLASH_COLOR: tuple[int, int, int] = (180, 180, 220)
LINE_CLEAR_HIGHLIGHT_COLOR: tuple[int, int, int, int] = (255, 255, 0, 180)  # RGBA
GAME_OVER_FLASH_COLOR: tuple[int, int, int] = (255, 0, 0)

# MCTS Visualization Colors
MCTS_NODE_WIN_COLOR: tuple[int, int, int] = DARK_GREEN  # Use darker green for node fill
MCTS_NODE_LOSS_COLOR: tuple[int, int, int] = DARK_RED
MCTS_NODE_NEUTRAL_COLOR: tuple[int, int, int] = DARK_GRAY  # Use darker gray
MCTS_NODE_BORDER_COLOR: tuple[int, int, int] = GRAY  # Lighter border
MCTS_NODE_SELECTED_BORDER_COLOR: tuple[int, int, int] = YELLOW
MCTS_EDGE_COLOR: tuple[int, int, int] = GRAY  # Lighter edge color
MCTS_EDGE_HIGHLIGHT_COLOR: tuple[int, int, int] = WHITE
MCTS_INFO_TEXT_COLOR: tuple[int, int, int] = WHITE
MCTS_NODE_TEXT_COLOR: tuple[int, int, int] = WHITE
MCTS_NODE_PRIOR_COLOR: tuple[int, int, int] = CYAN
MCTS_NODE_SCORE_COLOR: tuple[int, int, int] = ORANGE
MCTS_MINI_GRID_BG_COLOR: tuple[int, int, int] = (40, 40, 40)  # Background for mini-grid
MCTS_MINI_GRID_LINE_COLOR: tuple[int, int, int] = (70, 70, 70)  # Lines for mini-grid
MCTS_MINI_GRID_OCCUPIED_COLOR: tuple[int, int, int] = (
    200,
    200,
    200,
)  # Occupied cells in mini-grid
