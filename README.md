# File: README.md
# AlphaTriangle Project

## Overview

AlphaTriangle is a project implementing an artificial intelligence agent based on AlphaZero principles to learn and play a custom puzzle game involving placing triangular shapes onto a grid. The agent learns through self-play reinforcement learning, guided by Monte Carlo Tree Search (MCTS) and a deep neural network (PyTorch).

The project includes:
*   A playable version of the triangle puzzle game using Pygame.
*   An implementation of the MCTS algorithm tailored for the game.
*   A deep neural network (policy and value heads) implemented in PyTorch, featuring convolutional layers and **optional Transformer Encoder layers**.
*   A reinforcement learning pipeline coordinating **parallel self-play (using Ray)**, data storage, and network training.
*   Visualization tools for interactive play, debugging, and monitoring training progress (**with near real-time plot updates**).
*   Experiment tracking using MLflow.

## Core Technologies

*   **Python 3.10+**
*   **Pygame:** For game visualization and interactive modes.
*   **PyTorch:** For the deep learning model (CNNs, **optional Transformers**) and training, with CUDA support.
*   **NumPy:** For numerical operations, especially state representation.
*   **Ray:** For parallelizing self-play data generation and statistics collection across multiple CPU cores/processes.
*   **Numba:** (Optional, used in `features.grid_features`) For performance optimization of specific grid calculations.
*   **Cloudpickle:** For serializing the experience replay buffer and training checkpoints.
*   **MLflow:** For logging parameters, metrics, and artifacts (checkpoints, buffers) during training runs.

## Project Structure

```markdown
.
├── .alphatriangle_data/ # Root directory for ALL persistent data
│   ├── mlruns/          # MLflow tracking data (metrics, params, artifacts link here)
│   └── runs/            # Stores temporary/local artifacts per run
│       └── <run_name>/  # Data specific to a training run
│           ├── checkpoints/ # Saved model weights, optimizer state, etc. (.pkl)
│           ├── buffers/     # Saved experience replay buffers (.pkl)
│           ├── logs/        # Log files (if not solely using MLflow/stdout)
│           └── configs.json # Copy of run configuration
├── src/ # Source code for the project
│   ├── config/ # Configuration files (game rules, NN architecture, training params)
│   ├── data/ # Data management (saving/loading checkpoints, buffers)
│   ├── environment/ # Game environment logic (state, actions, rules - NO NN features)
│   ├── features/ # Feature extraction logic (GameState -> NN input)
│   ├── interaction/ # User input handling for interactive modes
│   ├── mcts/ # Monte Carlo Tree Search implementation
│   ├── nn/ # Neural Network definition and interface
│   ├── rl/ # Reinforcement Learning pipeline (orchestrator, trainer, self-play actors)
│   ├── stats/ # Statistics collection (Ray actor) and plotting
│   ├── structs/ # Core data structures (Triangle, Shape) to avoid circular imports
│   ├── utils/ # Common utilities (types, helpers, geometry)
│   └── visualization/ # Pygame-based visualization components
├── requirements.txt # Python package dependencies
├── run_interactive.py # Script to run the game in interactive (play/debug) modes
├── run_training_headless.py # Script to run training without visualization (logs to MLflow)
├── run_training_visual.py # Script to run training with live visualization (logs to MLflow)
└── README.md # This file
```

## Key Modules (`src`)

*   **`config`:** Centralized configuration classes. `PersistenceConfig` defines the unified `.alphatriangle_data` structure.
*   **`structs`:** Defines core, low-level data structures (`Triangle`, `Shape`) and constants (`SHAPE_COLORS`).
*   **`environment`:** Defines the game rules, `GameState`, action encoding/decoding, and grid/shape *logic*.
*   **`features`:** Contains logic to convert `GameState` objects into numerical features (`StateType`).
*   **`nn`:** Contains the PyTorch `nn.Module` definition (`AlphaTriangleNet` - CNNs + **optional Transformers**) and a wrapper class (`NeuralNetwork`).
*   **`mcts`:** Implements the Monte Carlo Tree Search algorithm (`Node`, `run_mcts_simulations`).
*   **`rl`:** Orchestrates the reinforcement learning loop (`TrainingOrchestrator`), manages network updates (`Trainer`), handles **parallel self-play data generation using Ray actors (`SelfPlayWorker`)**, and stores experiences (`ExperienceBuffer`). The `TrainingOrchestrator` logs parameters, metrics, and artifacts to MLflow.
*   **`stats`:** Contains the `StatsCollectorActor` (Ray actor) for **asynchronous statistics collection** and the `Plotter` class for rendering plots using Matplotlib.
*   **`visualization`:** Uses Pygame to render the game state, previews, HUD, plots, etc. `GameRenderer` fetches data from `StatsCollectorActor` for **near real-time plot updates**.
*   **`interaction`:** Handles keyboard/mouse input for interactive modes.
*   **`data`:** Manages saving and loading of training artifacts like NN checkpoints and replay buffers (`DataManager`) within the `.alphatriangle_data/runs/<run_name>/` structure.
*   **`utils`:** Provides common helper functions, shared type definitions, and geometry helpers.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd alphatriangle # Or your project directory name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have the correct PyTorch version installed for your system (CPU/CUDA). See [pytorch.org](https://pytorch.org/). Ray may have specific system requirements.*
4.  **(Optional) Add data directory to `.gitignore`:**
    Create or edit the `.gitignore` file in your project root and add the line:
    ```
    .alphatriangle_data/
    ```

## Running the Code

Execute the following scripts from the **project root directory**:

*   **Interactive Play Mode:**
    ```bash
    python run_interactive.py --mode play
    ```
*   **Interactive Debug Mode:**
    ```bash
    python run_interactive.py --mode debug
    ```
*   **Headless Training:**
    ```bash
    python run_training_headless.py
    ```
    (Runs the RL training loop without visualization, using Ray for parallel self-play. Logs parameters, metrics, and artifacts to MLflow. Uses CUDA if available for training).
*   **Training with Visualization:**
    ```bash
    python run_training_visual.py
    ```
    (Runs the RL training loop and displays the game state from workers and **near real-time plots**. Uses Ray for parallel self-play and stats collection. Uses CUDA if available for training. Logs parameters, metrics, and artifacts to MLflow).
*   **Monitoring Training (MLflow UI):**
    While training (headless or visual), or after runs have completed, open a separate terminal in the project root and run:
    ```bash
    mlflow ui --backend-store-uri file:./.alphatriangle_data/mlruns
    ```
    Then navigate to `http://localhost:5000` (or the specified port) in your browser. This UI allows you to compare runs, view logged parameters/metrics, and access saved artifacts (like checkpoints).

## Configuration

All major parameters are defined in the classes within the `src/config/` directory. Modify these files to experiment with different settings (e.g., enable/configure Transformer layers in `model_config.py`). The `src/config/validation.py` script performs basic checks on startup.

## Data Storage

All persistent data, including MLflow tracking data and run-specific artifacts, is stored within the `.alphatriangle_data/` directory in the project root.

*   **`.alphatriangle_data/mlruns/`**: Contains MLflow tracking data (managed by MLflow).
    *   `<experiment_id>/<run_id>/`: Contains subdirectories for `artifacts/`, `metrics/`, `params/`, and `meta.yaml`. MLflow logs artifacts by copying them from the temporary run directory (`.alphatriangle_data/runs/<run_name>/...`) into its own artifact store, typically within this structure.
*   **`.alphatriangle_data/runs/`**: Contains temporary and local artifacts generated during training runs.
    *   `<run_name>/`: Directory specific to a training run (identified by `RUN_NAME`).
        *   `checkpoints/`: Stores saved checkpoints (`.pkl` files containing model weights, optimizer state, training progress, and stats collector state).
        *   `buffers/`: Stores saved experience replay buffers (`.pkl` files).
        *   `logs/`: Optional directory for non-MLflow log files.
        *   `configs.json`: A JSON copy of the configuration used for the run.

The `DataManager` saves files to the appropriate subdirectory within `.alphatriangle_data/runs/<run_name>/`, and the `TrainingOrchestrator` (via helpers) logs these files as artifacts to MLflow, which then manages them within the `.alphatriangle_data/mlruns/` structure. Checkpoints and buffers are saved robustly using `cloudpickle`.

## Maintainability

This project includes README files within each major `src` submodule. **Please keep these READMEs updated** when making changes to the code's structure, interfaces, or core logic. Accurate documentation significantly aids understanding and future development. Consider regenerating documentation snippets if major refactoring occurs.