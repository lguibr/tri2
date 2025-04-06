# File: config/constants.py
# NEW FILE
"""
Defines constants shared across different modules, primarily visual elements,
to avoid circular imports and keep configuration clean.
"""

# Colors (RGB tuples 0-255)
WHITE: tuple[int, int, int] = (255, 255, 255)
BLACK: tuple[int, int, int] = (0, 0, 0)
LIGHTG: tuple[int, int, int] = (140, 140, 140)
GRAY: tuple[int, int, int] = (50, 50, 50)
RED: tuple[int, int, int] = (255, 50, 50)
DARK_RED: tuple[int, int, int] = (80, 10, 10)
BLUE: tuple[int, int, int] = (50, 50, 255)
YELLOW: tuple[int, int, int] = (255, 255, 100)
GOOGLE_COLORS: list[tuple[int, int, int]] = [
    (15, 157, 88),  # Green
    (244, 180, 0),  # Yellow/Orange
    (66, 133, 244),  # Blue
    (219, 68, 55),  # Red
]
LINE_CLEAR_FLASH_COLOR: tuple[int, int, int] = (180, 180, 220)
LINE_CLEAR_HIGHLIGHT_COLOR: tuple[int, int, int, int] = (255, 255, 0, 180)  # RGBA
GAME_OVER_FLASH_COLOR: tuple[int, int, int] = (255, 0, 0)

# Add other simple, shared constants here if needed.
