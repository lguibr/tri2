# File: src/rl/README.md
# Reinforcement Learning Module (`src.rl`)

## Purpose and Architecture

This module contains the core components and orchestration logic for training the AlphaTriangle agent using reinforcement learning, specifically inspired by the AlphaZero methodology. It combines **parallel self-play data generation using Ray** with centralized neural network training.

-   **Core Components (`src.rl.core`):**
    -   `TrainingOrchestrator`: The main class that manages the entire training process. It initializes and manages Ray actors (`SelfPlayWorker`) for parallel self-play, coordinates data collection into the `ExperienceBuffer`, and triggers training steps via the `Trainer`. It handles loading/saving checkpoints and buffers, logging to MLflow, and the main asynchronous training loop.
    -   `Trainer`: Responsible for performing the neural network update steps on the main process/device. It takes batches of experience from the buffer, uses `src.features` to extract features, calculates losses, and updates the network weights.
    -   `ExperienceBuffer`: A replay buffer that stores experiences (`(GameState, policy_target, value_target)`) generated during self-play.
-   **Self-Play Components (`src.rl.self_play`):**
    -   `worker`: Defines the `SelfPlayWorker` Ray actor. Each actor runs game episodes independently using MCTS and its local copy of the neural network (weights updated periodically by the orchestrator). It collects `GameState` objects, MCTS policy targets, and the final game outcome to generate experiences.
-   **Types (`src.rl.types`):**
    -   Defines Pydantic models like `SelfPlayResult` for structured data transfer between Ray actors and the orchestrator.

## Exposed Interfaces

-   **Core:**
    -   `TrainingOrchestrator`:
        -   `__init__(...)`
        -   `run_training_loop()`
        -   `request_stop()`
    -   `Trainer`:
        -   `__init__(nn_interface: NeuralNetwork, train_config: TrainConfig, env_config: EnvConfig)`
        -   `train_step(batch: ExperienceBatch) -> Optional[Dict[str, float]]`
        -   `load_optimizer_state(state_dict: dict)`
        -   `get_current_lr() -> float`
    -   `ExperienceBuffer`:
        -   `__init__(config: TrainConfig)`
        -   `add(experience: Experience)`
        -   `add_batch(experiences: List[Experience])`
        -   `sample(batch_size: int) -> Optional[ExperienceBatch]`
        -   `is_ready() -> bool`
        -   `__len__() -> int`
-   **Self-Play:**
    -   `SelfPlayWorker`: Ray actor class (primarily used internally by `TrainingOrchestrator`).
        -   `run_episode() -> SelfPlayResult`
        -   `set_weights(weights: Dict)`
-   **Types:**
    -   `SelfPlayResult`: Pydantic model for self-play results.

## Dependencies

-   **`src.config`**:
    -   `TrainConfig`, `EnvConfig`, `PersistenceConfig`, `MCTSConfig`, `ModelConfig`: Used extensively by all components.
-   **`src.nn`**:
    -   `NeuralNetwork`: Used by the `Trainer` and instantiated within `SelfPlayWorker`. **May include Transformer layers.**
-   **`src.features`**:
    -   `extract_state_features`: Used by `Trainer` and `NeuralNetwork`.
-   **`src.mcts`**:
    -   Core MCTS components used by `SelfPlayWorker`.
-   **`src.environment`**:
    -   `GameState`, `EnvConfig`: Used by `SelfPlayWorker` and stored in the buffer.
-   **`src.data`**:
    -   `DataManager`: Used by `TrainingOrchestrator`.
-   **`src.stats`**:
    -   `StatsCollectorActor`: Used by `TrainingOrchestrator` to collect metrics.
-   **`src.utils`**:
    -   `types`: `Experience`, `ExperienceBatch`, etc.
    -   `helpers`.
-   **`src.structs`**:
    -   Implicitly used via `GameState`.
-   **`torch`**:
    -   Used heavily by `Trainer` and `NeuralNetwork`.
-   **`ray`**:
    -   Used by `TrainingOrchestrator` and `SelfPlayWorker` for parallelization.
-   **`mlflow`**:
    -   Used by `TrainingOrchestrator` for logging.
-   **Standard Libraries:** `typing`, `logging`, `os`, `time`, `queue`, `threading`, `random`, `collections.deque`.

---

**Note:** Please keep this README updated when changing the overall training flow, the responsibilities of the orchestrator, trainer, or buffer, or the self-play generation process (especially regarding Ray usage). Accurate documentation is crucial for maintainability.