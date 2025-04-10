# File: src/rl/self_play/README.md
# RL Self-Play Submodule (`src.rl.self_play`)

## Purpose and Architecture

This submodule focuses specifically on generating game episodes through self-play, driven by the current neural network and MCTS. It is designed to run in parallel using Ray actors managed by the `TrainingOrchestrator`.

-   **`worker.py`:** Defines the `SelfPlayWorker` class, decorated with `@ray.remote`.
    -   Each `SelfPlayWorker` actor runs independently, typically on a separate CPU core.
    -   It initializes its own `GameState` environment and `NeuralNetwork` instance (usually on the CPU).
    -   It receives configuration objects (`EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`) during initialization.
    -   It has a `set_weights` method allowing the `TrainingOrchestrator` to periodically update its local neural network with the latest trained weights from the central model.
    -   Its main method, `run_episode`, simulates a complete game episode:
        -   Uses its local `NeuralNetwork` evaluator and `MCTSConfig` to run MCTS (`src.mcts.run_mcts_simulations`).
        -   Selects actions based on MCTS results (`src.mcts.strategy.policy.select_action_based_on_visits`).
        -   Generates policy targets (`src.mcts.strategy.policy.get_policy_target`).
        -   Stores `(GameState, policy_target, placeholder_value)` tuples.
        -   Steps its local game environment (`GameState.step`).
        -   Backfills the value target after the episode ends.
        -   Returns the collected `Experience` list, final score, episode length, and final `GameState` to the orchestrator.
    -   Optionally, one worker (typically worker 0) can be designated to collect intermediate `GameState` objects with attached statistics, which the orchestrator can then forward to the visualization queue.

## Exposed Interfaces

-   **Classes:**
    -   `SelfPlayWorker`: Ray actor class.
        -   `__init__(...)`
        -   `run_episode() -> SelfPlayResult`: Runs one episode and returns results.
        -   `set_weights(weights: Dict)`: Updates the actor's local network weights.
-   **Types:**
    -   `SelfPlayResult = Tuple[List[Experience], float, int, GameState]` (where `Experience` contains `GameState`).

## Dependencies

-   **`src.config`**:
    -   `EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`.
-   **`src.nn`**:
    -   `NeuralNetwork`: Instantiated locally within the actor.
-   **`src.mcts`**:
    -   Core MCTS functions and types.
-   **`src.environment`**:
    -   `GameState`, `EnvConfig`: Used to instantiate and step through the game simulation locally.
-   **`src.utils`**:
    -   `types`: `Experience`, `ActionType`, `PolicyTargetMapping`.
    -   `helpers`: `get_device`, `set_random_seeds`.
-   **`numpy`**:
    -   Used by MCTS strategies.
-   **`ray`**:
    -   The `@ray.remote` decorator makes this a Ray actor.
-   **`torch`**:
    -   Used by the local `NeuralNetwork`.
-   **Standard Libraries:** `typing`, `logging`, `random`.

---

**Note:** Please keep this README updated when changing the self-play episode generation logic within the actor, the data collected (`Experience`), or the interaction with MCTS or the environment. Accurate documentation is crucial for maintainability.
