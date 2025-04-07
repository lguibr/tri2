**AlphaZero Core Concepts Explained**

AlphaZero learns by combining a powerful Neural Network (NN) with Monte Carlo Tree Search (MCTS) through self-play.

1.  **Neural Network (NN):**
    *   **Input:** The current game state (`GameState.get_state()`).
    *   **Output:** Two heads:
        *   **Policy Head (π):** Predicts a probability distribution over all possible *next moves* from the current state. This acts as a prior, guiding the MCTS search towards promising moves.
        *   **Value Head (v):** Predicts the expected *outcome* of the game from the current state (e.g., a value between -1 for loss, 0 for draw, +1 for win).
    *   **Architecture:** Often uses convolutional layers (like ResNet blocks) to process the board state, followed by fully connected layers leading to the policy and value heads. Your `ModelConfig` provides a good starting point.

2.  **Monte Carlo Tree Search (MCTS):**
    *   **Purpose:** For a given game state, MCTS explores the possible future game trajectories to determine the *best* move to make *right now*. It builds a search tree where nodes are game states and edges are actions.
    *   **Process (for each move decision):**
        *   **Selection:** Start at the root node (current state). Traverse down the tree by repeatedly selecting the child node with the highest score according to a formula (like UCB1: `ValueEstimate + ExplorationBonus * sqrt(log(ParentVisits) / ChildVisits)`). The NN's policy output (π) influences the ExplorationBonus, biasing the search towards moves the NN thinks are good. The NN's value output (v) can be used as the initial ValueEstimate for new nodes.
        *   **Expansion:** When a leaf node (a state not yet fully explored or added to the tree) is reached, expand it by adding one or more children representing possible next states after taking valid actions. Get the policy (π) and value (v) for this leaf state from the NN. Use π to initialize child node priors and v as the initial value estimate for this node.
        *   **Simulation (Rollout):** *Crucially, the original AlphaZero often doesn't perform full random rollouts.* Instead, the **NN's value prediction (v)** for the expanded leaf node is directly used as the estimated outcome of the game from that point. This makes the search much more efficient than random simulations.
        *   **Backpropagation:** Update the statistics (visit count `N`, total action value `W` or `Q`) of all nodes along the path from the expanded leaf node back up to the root, using the value (v) obtained during expansion/simulation. The value estimate for a node becomes `Q = W / N`.
    *   **Output:** After running many simulations (e.g., 100, 800, 1600), MCTS provides an *improved* policy distribution for the root state (current state). This distribution is usually based on the visit counts (`N`) of the children nodes of the root (often normalized: `probs = N^(1/temperature)`). This improved policy is the target the NN learns to predict.

**Self-Play:**
    *   The agent plays games against itself.
    *   For each move in a game:
        *   Run MCTS from the current state using the *current* NN.
        *   Select the actual move to play based on the MCTS result (e.g., sample proportionally to visit counts, especially early in the game to encourage exploration; later, deterministically pick the most visited move).
        *   Update the game state.
    *   At the end of the game, record the final outcome (Win=+1, Loss=-1, Draw=0).
    *   Store the collected data for each step: `(state, mcts_policy_target, final_outcome)`.

**Training:**
    *   Periodically (or continuously), sample batches of `(state, mcts_policy_target, final_outcome)` data collected from self-play games.
    *   Train the NN:
        *   **Policy Loss:** Minimize the difference (e.g., cross-entropy) between the NN's policy output (π) for the `state` and the `mcts_policy_target`.
        *   **Value Loss:** Minimize the difference (e.g., mean squared error) between the NN's value output (v) for the `state` and the actual `final_outcome` (z).
        *   Combine these losses (often with regularization) and update the NN weights using an optimizer (e.g., Adam).

**Analogy:** The NN is like an improving intuition about the game. MCTS is like focused thinking/planning using that intuition to find the best immediate move. Self-play generates the experience (games) needed for the intuition (NN) to learn from the thinking (MCTS) and the results (game outcomes).

**Does MCTS need huge amounts of random data?** Not exactly. MCTS *itself* is the search process. The *self-play* phase generates the data *using* MCTS guided by the NN. The quality of this data improves as the NN gets better. You need many self-play games, but the moves within those games are intelligently selected by MCTS, not purely random (except maybe in the simulation phase if you choose to implement it that way, but using the NN value is standard).

**Step-by-Step Implementation Plan**

Here's a high-level, detailed plan to refactor towards AlphaZero:

**Phase 1: Implement AlphaZero Components**

3.  **Define Neural Network (`AlphaZeroNet`):**
    *   Create a new file (e.g., `agent/alphazero_net.py`).
    *   Define a class `AlphaZeroNet(torch.nn.Module)`.
    *   Use `ModelConfig` to define the architecture (e.g., CNN backbone similar to existing, potentially ResNet blocks).
    *   Implement the `forward` method:
        *   Input: State dictionary from `GameState.get_state()`. Process grid, shapes, features appropriately.
        *   Output: `policy_logits` (raw scores before softmax, shape `[batch_size, action_dim]`) and `value` (scalar estimate, shape `[batch_size, 1]`).
    *   Implement `get_state_dict` and `load_state_dict` (standard PyTorch).
4.  **Implement MCTS:**
    *   Create new files (e.g., `mcts/node.py`, `mcts/search.py`).
    *   **`MCTSNode` Class:** Represents a node in the search tree. Stores:
        *   `state`: The game state this node represents (can be lightweight if needed).
        *   `parent`: Reference to the parent node.
        *   `children`: Dictionary mapping action -> child `MCTSNode`.
        *   `visit_count (N)`: How many times this node was visited during backpropagation.
        *   `total_action_value (W)` or `mean_action_value (Q)`: Sum or average of values backpropagated through this node.
        *   `prior_probability (P)`: Policy prior from the NN for the action leading to this node (stored in the child).
        *   `action_taken`: The action that led from the parent to this node.
        *   `is_expanded`: Boolean flag.
    *   **`MCTS` Class:** Orchestrates the search.
        *   `__init__(self, nn_agent: AlphaZeroNet, env_config: EnvConfig, mcts_config)`: Takes the NN and configs.
        *   `run_simulations(self, root_state: GameState, num_simulations: int)`: Main MCTS loop.
            *   Creates the `root_node`.
            *   Repeatedly calls `_select`, `_expand`, `_simulate` (using NN value), `_backpropagate`.
        *   `_select(self, node: MCTSNode)`: Traverses the tree using UCB1 or PUCT formula, returning the leaf node to expand.
        *   `_expand(self, node: MCTSNode)`: If node isn't terminal, get valid actions, get policy/value from NN for the node's state, create child nodes, initialize their priors (P).
        *   `_simulate(self, node: MCTSNode)`: **Crucially, just return the value (v) predicted by the NN during the `_expand` step for this node.** No random rollout needed typically.
        *   `_backpropagate(self, node: MCTSNode, value: float)`: Update `N` and `W` (or `Q`) for the node and its ancestors up to the root.
        *   `get_policy_target(self, root_node: MCTSNode, temperature: float)`: After simulations, calculate the improved policy target based on child visit counts (`N^(1/temperature)`), normalized. Returns a probability distribution over actions.

**Phase 2: Implement Workers and Integration**

5.  **Implement Self-Play Worker:**
    *   Create a class (e.g., `workers/self_play_worker.py`).
    *   Takes the NN, `EnvConfig`, MCTS instance (or creates one), a shared data buffer (e.g., `queue.Queue` or custom buffer), stop/pause events.
    *   `run()` method:
        *   Loops indefinitely (until `stop_event`).
        *   Plays a full game:
            *   `game = GameState()`, `game.reset()`.
            *   `game_data = []` (to store `(state, policy_target, player)` tuples for this game).
            *   While `not game.is_over()`:
                *   `policy_target = mcts.run_simulations(game.get_state(), num_simulations)`
                *   `current_state_features = game.get_state()` # Get state *before* the move
                *   `game_data.append((current_state_features, policy_target, game.current_player))` # Store state and MCTS target
                *   `action = choose_action(policy_target, temperature)` # Choose actual move (probabilistic early, deterministic later)
                *   `_, done = game.step(action)`
            *   `final_outcome = determine_outcome(game)` # Get win/loss/draw (+1/-1/0)
            *   Assign the `final_outcome` to all stored tuples in `game_data`.
            *   Put `game_data` into the shared buffer.
            *   Log episode stats via `StatsAggregator`.
6.  **Implement Training Worker:**
    *   Create a class (e.g., `workers/training_worker.py`).
    *   Takes the NN, optimizer, shared data buffer, `StatsAggregator`, stop event.
    *   `run()` method:
        *   Loops indefinitely (until `stop_event`).
        *   Samples a batch of `(state, policy_target, outcome)` from the buffer.
        *   Performs NN forward pass: `policy_logits, value = nn(batch_states)`.
        *   Calculates policy loss (e.g., `CrossEntropyLoss(policy_logits, batch_policy_targets)`).
        *   Calculates value loss (e.g., `MSELoss(value, batch_outcomes)`).
        *   Calculates total loss (+ regularization).
        *   Performs `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
        *   Logs losses and other training metrics (LR, etc.) via `StatsAggregator.record_step()`.
7.  **Integrate Components:**
    *   **`AppInitializer`:** Instantiate `AlphaZeroNet`, `MCTS`, `SelfPlayWorker`, `TrainingWorker`, shared buffer, optimizer. Pass references correctly.
    *   **`AppWorkerManager`:** Modify `start_worker_threads` and `stop_worker_threads` to manage the new `SelfPlayWorker` and `TrainingWorker` threads.
    *   **`CheckpointManager`:** Update `save_checkpoint` to include `nn.state_dict()`, `optimizer.state_dict()`, and the updated `stats_aggregator.state_dict()`. Update `load_checkpoint` accordingly.
    *   **`StatsAggregator` / Loggers:** Ensure they track and log `policy_loss`, `value_loss`, and potentially MCTS statistics passed via `record_step` or `record_episode`.
    *   **`main_pygame.py`:** Ensure the main loop correctly starts/stops workers, fetches stats from the aggregator for rendering, and handles shutdown gracefully.
    *   **UI (`LeftPanelRenderer`, `Plotter`):** Update to display new stats (NN losses) and remove obsolete PPO stats.

**Phase 3: Refinement and Tuning**

8.  **Debugging:** Thoroughly test interactions between MCTS, NN, self-play, and training.
9.  **Tuning:** Adjust hyperparameters:
    *   MCTS: `num_simulations`, UCB1/PUCT exploration constant (`c_puct`).
    *   Self-Play: Temperature parameter for action selection.
    *   NN: Architecture (`ModelConfig`), learning rate, optimizer parameters, regularization strength.
    *   Training: Batch size, buffer size, training frequency vs. self-play generation speed.
10. **Profiling:** Use `analyze_profile.py` to identify bottlenecks (MCTS or NN inference are common).