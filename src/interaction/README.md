# File: src/interaction/README.md
# Interaction Module (`src.interaction`)

## Purpose and Architecture

This module handles user input (keyboard and mouse) for interactive modes of the application, such as "play" and "debug". It bridges the gap between raw Pygame events and actions within the game simulation (`GameState`).

-   **Event Processing:** `event_processor.py` handles common Pygame events like quitting (QUIT, ESC) and window resizing. It acts as a generator, yielding other events for mode-specific processing.
-   **Input Handler:** The `InputHandler` class is the main entry point. It receives Pygame events (via the `event_processor`), determines the current interaction mode ("play" or "debug"), and delegates event handling and hover updates to specific handler functions.
-   **Mode-Specific Handlers:** `play_mode_handler.py` and `debug_mode_handler.py` contain the logic specific to each mode:
    -   `play`: Handles selecting shapes from the preview area and placing them on the grid via clicks. Updates hover previews.
    -   `debug`: Handles toggling the state of individual triangles on the grid via clicks. Updates hover highlights.
-   **Decoupling:** It separates input handling logic from the core game simulation (`environment`) and rendering (`visualization`), although it needs references to both to function.

## Exposed Interfaces

-   **Classes:**
    -   `InputHandler`:
        -   `__init__(game_state: GameState, visualizer: Visualizer, mode: str, env_config: EnvConfig)`
        -   `handle_input() -> bool`: Processes events for one frame, returns `False` if quitting.
-   **Functions:**
    -   `process_pygame_events(visualizer: Visualizer) -> Generator[pygame.event.Event, Any, bool]`: Processes common events, yields others.
    -   `handle_play_click(...)`: Handles clicks in play mode.
    -   `update_play_hover(...)`: Updates hover state in play mode.
    -   `handle_debug_click(...)`: Handles clicks in debug mode.
    -   `update_debug_hover(...)`: Updates hover state in debug mode.

## Dependencies

-   **`src.environment`**:
    -   `GameState`: Modifies the game state based on user actions (placing shapes, toggling debug cells).
    -   `EnvConfig`: Used for coordinate mapping and action encoding.
    -   `GridLogic`, `ActionCodec`: Used by mode-specific handlers.
-   **`src.visualization`**:
    -   `Visualizer`: Used to get layout information (`grid_rect`, `preview_rects`) and for coordinate mapping (`get_grid_coords_from_screen`, `get_preview_index_from_screen`). Also updated directly during resize events.
    -   `VisConfig`: Accessed via `Visualizer`.
-   **`pygame`**:
    -   Relies heavily on Pygame for event handling (`pygame.event`, `pygame.mouse`) and constants (`MOUSEBUTTONDOWN`, `KEYDOWN`, etc.).
-   **Standard Libraries:** `typing`, `logging`.

---

**Note:** Please keep this README updated when adding new interaction modes, changing input handling logic, or modifying the interfaces between interaction, environment, and visualization. Accurate documentation is crucial for maintainability.
Use code with caution.
