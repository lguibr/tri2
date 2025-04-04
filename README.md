# TriCrack DQN - Pygame & TensorBoard RL Agent 🎮🧠📊

This project implements a Deep Q-Network (DQN) agent trained to play "TriCrack", a custom Tetris-like game built with Pygame where players place triangular polyominoes (shapes) onto a triangular grid. The agent uses several advanced DQN techniques, including Noisy Nets for exploration, Dueling Architecture, Double DQN, Prioritized Experience Replay (PER), and N-Step Returns.

The project features a real-time Pygame visualization displaying multiple parallel game environments, performance statistics, and agent status. Comprehensive logging is integrated using TensorBoard, allowing detailed tracking of metrics, histograms of key values (Q-values, losses, rewards), environment state images, hyperparameters, and the model graph.

## Key Features ✨

*   **Advanced DQN Implementation:**
    *   **Noisy Nets:** Uses noisy linear layers for efficient exploration, eliminating the need for epsilon-greedy scheduling.
    *   **Dueling DQN:** Separates value and advantage streams for better policy evaluation.
    *   **Double DQN:** Decouples action selection and evaluation to reduce Q-value overestimation.
    *   **Prioritized Experience Replay (PER):** Samples transitions based on TD-error magnitude for more efficient learning.
    *   **N-Step Returns:** Improves sample efficiency by bootstrapping over multiple steps.
*   **Custom Environment:** Includes the "TriCrack" game logic (`environment/`), featuring a unique triangular grid, shape generation, placement rules, and scoring.
*   **Pygame Visualization:** Provides an interactive UI (`ui/`) showing:
    *   Live rendering of multiple parallel environments (configurable number).
    *   Real-time statistics panel (scores, loss, SPS, buffer status, etc.).
    *   Agent status (Training, Paused, Buffering, Error).
    *   Interactive buttons (Start/Pause, Cleanup).
    *   Informative tooltips for UI elements.
*   **TensorBoard Logging:** Comprehensive logging via `stats/tensorboard_logger.py`:
    *   Scalar metrics (rewards, loss, Q-values, episode length, SPS, etc.).
    *   Histograms (Q-value distributions, TD-errors, actions, rewards per step).
    *   Image logging (sample environment states).
    *   Hyperparameter logging (saves run configuration).
    *   Model graph visualization.
*   **Configurable Network Architecture:** Uses a fusion network (`agent/networks/agent_network.py`):
    *   CNN branch processes the grid state.
    *   MLP branch processes features of available shapes.
    *   Features are fused and passed through further MLP layers before the Dueling heads.
*   **Vectorized Environments:** Runs multiple environments in parallel (`EnvConfig.NUM_ENVS`) for stable and faster training.
*   **Highly Configurable:** Centralized configuration file (`config.py`) allows easy modification of nearly all aspects: environment dimensions, reward shaping, network architecture, DQN hyperparameters, training loop settings, buffer settings, logging, and visualization.
*   **Checkpointing & Resuming:** Saves agent model state and replay buffer state periodically and allows resuming training from saved checkpoints/buffers.

## Screenshots / Demo 📸

*(It's highly recommended to add screenshots or a GIF here!)*

*   **Placeholder:** *[Screenshot of the Pygame UI showing multiple environments, the stats panel, and buttons]*
    *   *Caption:* The main Pygame interface showing live training progress with multiple environments rendered simultaneously and key statistics displayed on the left panel.
*   **Placeholder:** *[Screenshot of the TensorBoard dashboard showing scalar plots like average score and loss]*
    *   *Caption:* TensorBoard scalar plots tracking episode rewards, loss, and other metrics over training steps.
*   **Placeholder:** *[Screenshot of the TensorBoard dashboard showing histograms like Q-value distribution]*
    *   *Caption:* TensorBoard histograms providing insights into the distribution of Q-values, TD-errors, and actions during training.
*   **Placeholder:** *[Screenshot of the TensorBoard dashboard showing logged environment images]*
    *   *Caption:* TensorBoard image tab displaying sample environment states logged periodically during training.

## Requirements 📜

*   Python (>= 3.9 recommended)
*   Pygame (>= 2.1.0)
*   NumPy (>= 1.20.0)
*   PyTorch (>= 1.10.0)
*   TensorBoard
*   Cloudpickle
*   Torchvision (potentially needed by TensorBoard for image logging, included for safety)

You can install the dependencies using the provided `requirements.txt` file.

## Setup & Installation ⚙️

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
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

## Configuration 🛠️

The core configuration file is `config.py`. This file centralizes *all* hyperparameters and settings for the environment, agent, training, buffer, visualization, and logging.

Key configuration classes and parameters include:

*   **General:** `DEVICE` (auto-detects CUDA/MPS/CPU), `RANDOM_SEED`, paths.
*   **`EnvConfig`:** `NUM_ENVS`, `ROWS`, `COLS`, state/action dimensions.
*   **`RewardConfig`:** Coefficients for reward shaping (placing shapes, clearing lines, penalties).
*   **`DQNConfig`:** `GAMMA`, `TARGET_UPDATE_FREQ`, `LEARNING_RATE`, algorithm variants (`USE_DOUBLE_DQN`, `USE_DUELING`, `USE_NOISY_NETS`).
*   **`TrainConfig`:** `BATCH_SIZE`, `LEARN_START_STEP`, `TOTAL_TRAINING_STEPS`, `LEARN_FREQ`, checkpointing (`CHECKPOINT_SAVE_FREQ`, `LOAD_CHECKPOINT_PATH`).
*   **`BufferConfig`:** `REPLAY_BUFFER_SIZE`, N-Step settings (`USE_N_STEP`, `N_STEP`), PER settings (`USE_PER`, `PER_ALPHA`, `PER_BETA_START`, `PER_BETA_FRAMES`).
*   **`ModelConfig.Network`:** CNN parameters (`CONV_CHANNELS`, kernels, etc.), MLP parameters (`SHAPE_MLP_HIDDEN_DIM`, `COMBINED_FC_DIMS`), activations, batch norm, dropout.
*   **`VisConfig`:** Pygame window size, colors, number of environments to render (`NUM_ENVS_TO_RENDER`).
*   **`TensorBoardConfig`:** Logging flags (`LOG_HISTOGRAMS`, `LOG_IMAGES`), frequencies.

**Important:** Modify `config.py` directly to experiment with different settings. The project automatically creates run-specific directories for logs and checkpoints based on the `RUN_ID` generated at startup.

**Loading Checkpoints/Buffers:**
To resume training or load a pre-trained model/buffer, set the `LOAD_CHECKPOINT_PATH` and/or `LOAD_BUFFER_PATH` variables in `TrainConfig` within `config.py` to the respective file paths. Ensure the loaded state is compatible with the current configuration (especially model architecture and buffer settings like PER/N-Step).

## Running the Code ▶️

To start the training process with the Pygame visualization:

```bash
python main_pygame.py
```

The application will initialize the environments, agent, buffer, and logger, then open the Pygame window.

**UI Controls:**

*   **Train/Pause Button:** Click to toggle the training process.
*   **'P' Key:** Keyboard shortcut to toggle training.
*   **Cleanup This Run Button:** Click to *delete the saved agent checkpoint and buffer file specifically for the current run* and re-initialize the agent/buffer/trainer. **Use with caution!** This is useful for restarting a run from scratch without changing the `RUN_ID` or TensorBoard logs. A confirmation prompt will appear.
*   **'ESC' Key:** Cancels the cleanup confirmation prompt or exits the application if the prompt is not active.
*   **Window Resizing:** The Pygame window is resizable.

The console will print status updates, configuration details, and periodic summary statistics (if `StatsConfig.CONSOLE_LOG_FREQ` > 0).

## TensorBoard Integration 📈

All detailed logs are saved to the `logs/tensorboard/<RUN_ID>` directory. To view them:

1.  Make sure you have TensorBoard installed (`pip install tensorboard`).
2.  Run TensorBoard, pointing it to the base log directory:
    ```bash
    tensorboard --logdir logs
    ```
    *(Make sure you run this command from the project's root directory or provide the absolute path to the `logs` folder).*
3.  Open your web browser and navigate to the URL provided by TensorBoard (usually `http://localhost:6006`).

In TensorBoard, you can explore:

*   **Scalars:** Track metrics like average rewards, loss, Q-values, episode length, steps per second, PER beta, etc., over time.
*   **Histograms:** Visualize the distribution of Q-values, TD-errors, actions taken, and rewards received per step. This helps diagnose training stability and agent behavior.
*   **Images:** View sample snapshots of environment states logged periodically (if `TensorBoardConfig.LOG_IMAGES` is enabled).
*   **HParams:** See the hyperparameters used for the run and compare final metrics across different runs.
*   **Graphs:** Visualize the computational graph of the neural network model.

## Code Structure 📁

```
.
├── agent/                # DQN Agent logic
│   ├── networks/         # Neural network modules (CNN+MLP fusion, Noisy Layer)
│   │   ├── agent_network.py
│   │   ├── noisy_layer.py
│   │   └── __init__.py
│   ├── replay_buffer/    # Replay buffer implementations
│   │   ├── base_buffer.py
│   │   ├── buffer_utils.py # Factory function
│   │   ├── nstep_buffer.py # N-Step wrapper
│   │   ├── prioritized_buffer.py
│   │   ├── sum_tree.py     # Helper for PER
│   │   ├── uniform_buffer.py
│   │   └── __init__.py
│   ├── dqn_agent.py      # Main agent class
│   ├── model_factory.py  # Creates the network instance
│   └── __init__.py
├── checkpoints/          # Directory for saving model and buffer states (run-specific subdirs)
├── config.py             # Central configuration file
├── environment/          # TriCrack game environment logic
│   ├── game_state.py     # Main environment class, step logic, state representation
│   ├── grid.py           # Triangular grid logic
│   ├── shape.py          # Shape generation and properties
│   ├── triangle.py       # Single triangle cell representation
│   └── __init__.py
├── logs/                 # Directory for TensorBoard logs (run-specific subdirs)
├── stats/                # Statistics recording and logging
│   ├── simple_stats_recorder.py # Basic in-memory recorder
│   ├── stats_recorder.py # Base abstract class
│   ├── tensorboard_logger.py # Logs to TensorBoard
│   └── __init__.py
├── training/             # Training orchestration
│   ├── trainer.py        # Coordinates agent-environment interaction, learning updates
│   └── __init__.py
├── ui/                   # Pygame User Interface rendering
│   ├── renderer.py       # Handles drawing the UI, envs, stats, tooltips
│   └── __init__.py
├── utils/                # Utility functions and types
│   ├── helpers.py        # Device selection, seeding, saving/loading objects
│   ├── types.py          # Type definitions (Transition, StateType, etc.)
│   └── __init__.py
├── main_pygame.py        # Main application entry point, Pygame loop
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Core Concepts & Implementation Details 🧐

*   **Environment (TriCrack):** The game involves placing randomly generated triangular shapes (1-5 triangles) onto a grid. Rows are cleared if all non-"death" cells in them are occupied. The goal is typically to maximize score (shaped reward) or lines cleared. "Death" cells form a border and cannot be used.
*   **State Representation:** The state is a flat NumPy array concatenating:
    1.  **Grid Features:** A flattened representation of the grid state `[3, H, W]`, where channels represent `Occupied`, `Is_Up`, `Is_Death`.
    2.  **Shape Features:** A flattened representation of features for each available shape slot (`EnvConfig.NUM_SHAPE_SLOTS`). Features per shape include normalized counts of triangles, up-pointing triangles, down-pointing triangles, height, and width (`EnvConfig.SHAPE_FEATURES_PER_SHAPE`).
*   **Action Space:** The action space is discrete and potentially large: `Action = Shape_Slot_Index * (Grid_Row * Grid_Col) + Grid_Position_Index`. The agent must select which available shape to place and where its root triangle should go. Invalid actions (e.g., placing outside bounds, overlapping occupied/death cells) are masked out during action selection.
*   **Reward Shaping:** The RL reward (`GameState.score`) is shaped to guide learning, defined in `RewardConfig`. It includes small rewards for placing triangles, larger rewards for clearing lines, penalties for invalid moves and creating holes, a large penalty for game over, and a small reward for surviving each step. The `GameState.game_score` tracks a separate, simpler game-native score.
*   **DQN Variants:**
    *   **Double DQN:** Uses the online network to select the best action for the next state and the target network to evaluate that action, reducing overestimation bias.
    *   **Dueling DQN:** The network head splits into a Value stream (state value V(s)) and an Advantage stream (action advantages A(s,a)). These are combined (`Q = V + (A - mean(A))`) to get final Q-values, often leading to better performance by learning state values more effectively.
    *   **Noisy Nets:** Replaces standard linear layers in the network heads with `NoisyLinear` layers. These layers add learnable parametric noise to weights and biases, inducing exploration based on the agent's uncertainty. This often performs better than traditional epsilon-greedy exploration.
    *   **PER:** Stores experiences in a SumTree based on their TD-error. Transitions with higher errors (more "surprising") are sampled more frequently. Importance sampling weights are used to correct the bias introduced by non-uniform sampling.
    *   **N-Step Returns:** Calculates returns over N steps instead of just one, allowing rewards to propagate faster and often stabilizing learning. The `NStepBufferWrapper` handles this calculation.
*   **Network Architecture:** The `AgentNetwork` uses a multi-branch approach suitable for the heterogeneous state representation:
    1.  A CNN processes the spatial grid features.
    2.  An MLP processes the shape features.
    3.  The flattened outputs are concatenated and fed into a fusion MLP.
    4.  The final layer(s) implement the (Dueling) Q-value heads, using NoisyLinear if enabled.
*   **Visualization & Logging:** `UIRenderer` manages the Pygame display. `TensorBoardStatsRecorder` leverages `SimpleStatsRecorder` for in-memory averaging (used by the UI) and writes detailed data, histograms, images, etc., to TensorBoard files using `torch.utils.tensorboard.SummaryWriter`.

## Potential Improvements / Future Work 🚀

*   **Hyperparameter Optimization:** Systematically tune learning rates, network sizes, buffer capacity, PER/N-Step parameters, reward coefficients, etc.
*   **Architecture Exploration:** Experiment with different CNN architectures (e.g., ResNet blocks), attention mechanisms, or alternative ways to fuse grid and shape information.
*   **Advanced RL Algorithms:** Implement more modern algorithms like Rainbow DQN, PPO, SAC (requires adaptation for discrete action space), or MuZero.
*   **Game Mechanics:** Add features like gravity after line clears, different shape generation logic, or varying difficulty levels.
*   **Performance Optimization:** Profile code (e.g., environment stepping, network forward pass) and optimize bottlenecks. Investigate alternative parallelization strategies if needed.
*   **Testing Framework:** Add unit and integration tests for environment logic, agent components, and buffer operations.
*   **Curriculum Learning:** Start with simpler configurations (smaller grid, fewer shapes) and gradually increase complexity.

## License 📄

*(Specify your license here, e.g., MIT License)*

This project is licensed under the MIT License. See the LICENSE file for details.

---

*Happy Training!*