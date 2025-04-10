# File: src/rl/core/README.md
# RL Core Submodule (`src.rl.core`)

## Purpose and Architecture

This submodule contains the central classes that manage and execute the reinforcement learning training loop, coordinating parallel self-play workers and centralized training updates.

-   **`orchestrator.py`:** Defines the main `TrainingOrchestrator` class.
    -   It initializes all necessary components (NN, Buffer, Trainer, DataManager, Configs, StatsCollector).
    -   It creates and manages a pool of `SelfPlayWorker` Ray actors (`src.rl.self_play.worker`).
    -   It orchestrates the main asynchronous training loop:
        -   Launching self-play episode tasks on remote actors.
        -   Collecting completed episode results (`Experience` data) using `ray.wait()`.
        -   Adding collected data to the `ExperienceBuffer`.
        -   Performing training steps using the `Trainer` when the buffer is ready.
        -   Periodically updating the network weights on the remote `SelfPlayWorker` actors.
        -   Handling checkpoint saving/loading via `DataManager`.
        -   Logging metrics and configurations via `StatsCollector` and MLflow (using helpers).
    -   It manages high-level state like `global_step`, `episodes_played`, `stop_requested`, etc.
    -   It handles graceful shutdown of Ray actors and signaling the visualizer.
-   **`orchestrator_helpers.py`:** Contains helper functions used by the `TrainingOrchestrator` for tasks like:
    -   Loading initial state (checkpoint, buffer, stats).
    -   Saving checkpoints and buffers via `DataManager`.
    -   Logging configurations and metrics to MLflow.
    -   Processing results returned by `SelfPlayWorker` actors.
    -   Running a training step using the `Trainer`.
    -   Logging results from self-play and training steps.
    -   Updating the `visual_state_queue`.
-   **`Trainer`:** This class encapsulates the logic for updating the neural network's weights on the main process/device.
    -   It holds the main `NeuralNetwork` interface, optimizer, and scheduler.
    -   Its `train_step` method takes a batch of experiences, calls `src.features.extract_state_features`, performs forward/backward passes, calculates losses, and updates weights.
-   **`ExperienceBuffer`:** This class implements a replay buffer storing `Experience` tuples (`(GameState, policy_target, value_target)`).

## Exposed Interfaces

-   **Classes:**
    -   `TrainingOrchestrator`: (from `orchestrator.py`)
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

## Dependencies

-   **`src.config`**:
    -   `TrainConfig`, `EnvConfig`, `PersistenceConfig`, `MCTSConfig`, `ModelConfig`.
-   **`src.nn`**:
    -   `NeuralNetwork`.
-   **`src.features`**:
    -   `extract_state_features`: Used by `Trainer`.
-   **`src.mcts`**:
    -   `MCTSConfig`, `Node`: Used by Orchestrator (passed to self-play workers).
-   **`src.environment`**:
    -   `GameState`, `EnvConfig`: Used by Orchestrator (passed to self-play workers) and stored in Buffer/Experience.
-   **`src.data`**:
    -   `DataManager`.
-   **`src.stats`**:
    -   `StatsCollector`: Used by Orchestrator.
-   **`src.utils`**:
    -   `types`: `Experience`, `ExperienceBatch`, etc.
    -   `helpers`.
-   **`src.rl.self_play`**:
    -   `SelfPlayWorker`, `SelfPlayResult`: Used by Orchestrator.
-   **`torch`**:
    -   Used heavily by `Trainer`.
-   **`ray`**:
    -   Used by `TrainingOrchestrator` to manage actors and tasks.
-   **`mlflow`**:
    -   Used by `orchestrator_helpers`.
-   **Standard Libraries:** `typing`, `logging`, `os`, `time`, `queue`, `threading`, `random`, `collections.deque`, `json`.

---

**Note:** Please keep this README updated when changing the responsibilities or interfaces of the Orchestrator components, Trainer, or Buffer, or how they interact with each other and other modules, especially regarding the use of Ray actors. Accurate documentation is crucial for maintainability.