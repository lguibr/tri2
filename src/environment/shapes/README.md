# File: src/environment/shapes/README.md
# Environment Shapes Submodule (`src.environment.shapes`)

## Purpose and Architecture

This submodule defines the logic for managing placeable shapes within the game environment.

-   **Shape Representation:** The `Shape` class (defined in `src.structs`) stores the geometry of a shape as a list of relative triangle coordinates (`(dr, dc, is_up)`) and its color.
-   **Shape Logic:** The `logic.py` module (exposed as `ShapeLogic`) contains functions related to shapes:
    -   `generate_random_shape`: Creates a new `Shape` instance with a random configuration of triangles and a random color (using `SHAPE_COLORS` from `src.structs`). The complexity and variety of shapes are determined here.
    -   `refill_shape_slots`: Manages the available shapes in the `GameState`. When a shape is placed, this function is called to potentially replace the used slot with a new random shape, ensuring the player/agent always has a selection (up to `NUM_SHAPE_SLOTS`).

## Exposed Interfaces

-   **Modules/Namespaces:**
    -   `logic` (often imported as `ShapeLogic`):
        -   `generate_random_shape(rng: random.Random) -> Shape`
        -   `refill_shape_slots(game_state: GameState, rng: random.Random)`

## Dependencies

-   **`src.environment.core`**:
    -   `GameState`: Used by `ShapeLogic.refill_shape_slots` to access and modify the list of available shapes.
-   **`src.config`**:
    -   `EnvConfig`: Accessed via `GameState` (e.g., for `NUM_SHAPE_SLOTS`).
-   **`src.structs`**:
    -   Uses `Shape`, `SHAPE_COLORS`.
-   **Standard Libraries:** `typing`, `random`, `logging`.

---

**Note:** Please keep this README updated when changing the shape generation algorithm or the logic for managing shape slots in the game state. Accurate documentation is crucial for maintainability.