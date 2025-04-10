# File: src/environment/logic/README.md
# Environment Logic Submodule (`src.environment.logic`)

## Purpose and Architecture

This submodule contains higher-level game logic that orchestrates interactions between the `GameState`, `GridData`, and `Shape` objects, primarily focusing on action validation and execution.

-   **Action Validation:** `get_valid_actions` determines the complete set of legal moves (encoded as integers) available in the current `GameState`. It iterates through available shapes and possible grid positions, using `GridLogic.can_place` for validation.
-   **Action Execution:** `execute_placement` handles the consequences of taking a valid placement action. It updates the `GridData` (placing the shape), checks for and clears lines (using `GridLogic.check_and_clear_lines`), refills the shape slot (using `ShapeLogic.refill_shape_slots`), calculates the reward for the step, and updates the game score and statistics.
-   **Reward Calculation:** `calculate_reward` defines the reward function based on game events like placing a piece, clearing lines, and potentially game over penalties (though the latter is often handled by the value target in RL).
-   **Decoupling:** It separates the complex logic of validating and executing actions from the core `GameState` class, keeping `GameState.step` relatively clean.

## Exposed Interfaces

-   **Functions:**
    -   `get_valid_actions(game_state: GameState) -> List[ActionType]`
    -   `execute_placement(game_state: GameState, shape_idx: int, r: int, c: int, rng: random.Random) -> float`
    -   `calculate_reward(placed_shape: Shape, lines_cleared: int, triangles_cleared: int, is_game_over: bool) -> float` (Primarily used internally by `execute_placement`)

## Dependencies

-   **`src.environment.core`**:
    -   `GameState`: The primary object these functions operate on.
    -   `action_codec`: Used by `get_valid_actions` to encode valid moves.
-   **`src.environment.grid`**:
    -   `GridData`: Accessed via `GameState`.
    -   `GridLogic`: Specifically `can_place` and `check_and_clear_lines`.
-   **`src.environment.shapes`**:
    -   `Shape`: Accessed via `GameState`.
    -   `ShapeLogic`: Specifically `refill_shape_slots`.
-   **`src.config`**:
    -   `EnvConfig`: Accessed via `GameState`.
-   **`src.utils.types`**:
    -   `ActionType`: Used in function signatures.
-   **Standard Libraries:** `typing`, `logging`, `random`.

---

**Note:** Please keep this README updated when changing the rules for valid actions, the process of executing an action, or the reward calculation logic. Accurate documentation is crucial for maintainability.