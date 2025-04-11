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
        -   Adding collected data to the `ExperienceBuffer` (assigning initial max priority if using PER).
        -   Performing training steps using the `Trainer` when the buffer is ready.
        -   **If using PER, updates priorities in the `ExperienceBuffer` using TD errors returned by the `Trainer`.**
        -   Periodically updating the network weights on the remote `SelfPlayWorker` actors.
        -   Handling checkpoint saving/loading via `DataManager`.
        -   Logging metrics and configurations via `StatsCollector` and MLflow (using helpers).
    -   It manages high-level state like `global_step`, `episodes_played`, `stop_requested`, etc.
    -   It handles graceful shutdown of Ray actors and signaling the visualizer.
-   **`orchestrator_helpers.py`:** Contains helper functions used by the `TrainingOrchestrator` for tasks like logging, processing self-play results, and running the training step (including PER priority updates).
-   **`Trainer`:** This class encapsulates the logic for updating the neural network's weights on the main process/device.
    -   It holds the main `NeuralNetwork` interface, optimizer, and scheduler.
    -   Its `train_step` method takes a batch of experiences (potentially with PER indices and weights), calls `src.features.extract_state_features`, performs forward/backward passes, calculates losses (**applying importance sampling weights if using PER**), updates weights, and **returns calculated TD errors for PER priority updates**.
-   **`ExperienceBuffer`:** This class implements a replay buffer storing `Experience` tuples (`(GameState, policy_target, value_target)`). **It now supports Prioritized Experience Replay (PER) via a SumTree, including prioritized sampling and priority updates, based on configuration.**
-   **`visual_state_actor.py`:** Defines a simple Ray actor (`VisualStateActor`) to hold the latest `GameState` from each worker, facilitating updates for the visual training mode without direct orchestrator-to-worker communication for visualization state.

## Exposed Interfaces

-   **Classes:**
    -   `TrainingOrchestrator`: (from `orchestrator.py`)
        -   `__init__(...)`
        -   `run_training_loop()`
        -   `request_stop()`
    -   `Trainer`:
        -   `__init__(nn_interface: NeuralNetwork, train_config: TrainConfig, env_config: EnvConfig)`
        -   `train_step(per_sample: PERBatchSample) -> Optional[Tuple[Dict[str, float], np.ndarray]]`: Takes PER sample, returns loss info and TD errors.
        -   `load_optimizer_state(state_dict: dict)`
        -   `get_current_lr() -> float`
    -   `ExperienceBuffer`:
        -   `__init__(config: TrainConfig)`
        -   `add(experience: Experience)`
        -   `add_batch(experiences: List[Experience])`
        -   `sample(batch_size: int, current_train_step: Optional[int] = None) -> Optional[PERBatchSample]`: Samples batch, requires step for PER beta.
        -   `update_priorities(tree_indices: np.ndarray, td_errors: np.ndarray)`: Updates priorities for PER.
        -   `is_ready() -> bool`
        -   `__len__() -> int`
    -   `VisualStateActor`: Ray actor for visualization state.

## Dependencies

-   **`src.config`**:
    -   `TrainConfig`, `EnvConfig`, `PersistenceConfig`, `MCTSConfig`, `ModelConfig`. **`TrainConfig` includes PER parameters.**
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
    -   `StatsCollectorActor`: Used by Orchestrator.
-   **`src.utils`**:
    -   `types`: `Experience`, `ExperienceBatch`, **`PERBatchSample`**, etc.
    -   `helpers`.
-   **`src.rl.self_play`**:
    -   `SelfPlayWorker`, `SelfPlayResult`: Used by Orchestrator.
-   **`torch`**:
    -   Used heavily by `Trainer`.
-   **`ray`**:
    -   Used by `TrainingOrchestrator` and `VisualStateActor`.
-   **`mlflow`**:
    -   Used by `orchestrator_helpers`.
-   **Standard Libraries:** `typing`, `logging`, `os`, `time`, `queue`, `threading`, `random`, `collections.deque`, `json`, `numpy`.

---

**Note:** Please keep this README updated when changing the responsibilities or interfaces of the Orchestrator components, Trainer, or Buffer, or how they interact with each other and other modules, especially regarding the use of Ray actors and PER. Accurate documentation is crucial for maintainability.
