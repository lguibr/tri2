# File: src/visualization/core/README.md
# Visualization Core Submodule (`src.visualization.core`)

## Purpose and Architecture

This submodule contains the central classes and foundational elements for the visualization system. It orchestrates rendering, manages layout and coordinate systems, and defines core visual properties like colors and fonts.

-   **Render Orchestration:**
    -   `Visualizer`: The main class for rendering in interactive modes ("play", "debug"). It maintains the Pygame screen, calculates layout using `layout.py`, manages cached preview area rectangles, and calls appropriate drawing functions from `src.visualization.drawing` based on the `GameState` and interaction mode. It handles visual feedback like hover previews and selection highlights.
    -   `GameRenderer`: **Adapted renderer** specifically for the non-interactive training visualization mode. It now uses `layout.py` to divide the screen into a worker game grid area and a statistics area. It renders multiple worker `GameState` objects in the top grid and displays statistics plots (using `src.stats.Plotter`) and progress bars in the bottom area. It takes a dictionary mapping worker IDs to `GameState` objects (`Dict[int, GameState]`) and a dictionary of global statistics.
-   **Layout Management:**
    -   `layout.py`: Contains the `calculate_layout` function, which determines the size and position of the main UI areas (worker grid, stats area, plots) based on the screen dimensions and `VisConfig`.
-   **Coordinate System:**
    -   `coord_mapper.py`: Provides essential mapping functions:
        -   `_calculate_render_params`: Internal helper to get scaling and offset for grid rendering.
        -   `get_grid_coords_from_screen`: Converts mouse/screen coordinates into logical grid (row, column) coordinates, handling the triangular grid geometry.
        -   `get_preview_index_from_screen`: Converts mouse/screen coordinates into the index of the shape preview slot being pointed at.
-   **Visual Properties:**
    -   `colors.py`: Defines a centralized palette of named color constants (RGB tuples) used throughout the visualization drawing functions.
    -   `fonts.py`: Contains the `load_fonts` function to load and manage Pygame font objects based on sizes defined in `VisConfig`.

## Exposed Interfaces

-   **Classes:**
    -   `Visualizer`: Renderer for interactive modes.
        -   `__init__(...)`
        -   `render(game_state: GameState, mode: str)`
        -   `ensure_layout() -> Dict[str, pygame.Rect]`
        -   `screen`: Public attribute (Pygame Surface).
        -   `preview_rects`: Public attribute (cached preview area rects).
    -   `GameRenderer`: Renderer for combined multi-game/stats training visualization.
        -   `__init__(...)`
        -   `render(worker_states: Dict[int, GameState], global_stats: Optional[Dict[str, Any]])`
        -   `screen`: Public attribute (Pygame Surface).
-   **Functions:**
    -   `calculate_layout(screen_width: int, screen_height: int, vis_config: VisConfig, bottom_margin: int) -> Dict[str, pygame.Rect]`
    -   `load_fonts() -> Dict[str, Optional[pygame.font.Font]]`
    -   `get_grid_coords_from_screen(screen_pos: Tuple[int, int], grid_area_rect: pygame.Rect, config: EnvConfig) -> Optional[Tuple[int, int]]`
    -   `get_preview_index_from_screen(screen_pos: Tuple[int, int], preview_rects: Dict[int, pygame.Rect]) -> Optional[int]`
-   **Modules:**
    -   `colors`: Provides color constants (e.g., `colors.RED`).

## Dependencies

-   **`src.config`**:
    -   `VisConfig`, `EnvConfig`: Used for layout, fonts, coordinate mapping.
-   **`src.environment`**:
    -   `GameState`, `GridData`: Needed for rendering and coordinate mapping.
-   **`src.stats`**:
    -   `Plotter`: Used by `GameRenderer`.
-   **`src.utils`**:
    -   `types`: `StatsCollectorData`.
-   **`src.visualization.drawing`**:
    -   Drawing functions (`grid`, `previews`, `hud`, `highlight`) are called by `Visualizer` and `GameRenderer`.
-   **`src.visualization.ui`**:
    -   `ProgressBar`: Used by `GameRenderer`.
-   **`pygame`**:
    -   Used for surfaces, rectangles, fonts, display management.
-   **Standard Libraries:** `typing`, `logging`, `math`.

---

**Note:** Please keep this README updated when changing the core rendering logic, layout calculations, coordinate mapping, or the interfaces of `Visualizer` or `GameRenderer`. Accurate documentation is crucial for maintainability.