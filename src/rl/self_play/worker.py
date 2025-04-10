# File: src/rl/self_play/worker.py
import logging
import random
import numpy as np
import ray
import torch
import time
from typing import List, Tuple, Optional, Generator, Any, Dict

# --- Package Imports ---
from src.environment import GameState, EnvConfig
from src.mcts import (
    Node,
    MCTSConfig,
    run_mcts_simulations,
    select_action_based_on_visits,
    get_policy_target,
)
from src.nn import NeuralNetwork
from src.config import ModelConfig, TrainConfig
from src.utils import get_device, set_random_seeds

# Experience type now expects GameState
from src.utils.types import Experience, ActionType, PolicyTargetMapping
# Import SelfPlayResult Pydantic model from local rl types
from ..types import SelfPlayResult # Updated import

# Import handle type for VisualStateActor
from ..core.visual_state_actor import VisualStateActor

logger = logging.getLogger(__name__)

@ray.remote
class SelfPlayWorker:
    """
    A Ray actor responsible for running self-play episodes using MCTS and a NN.
    Pushes its state periodically to a central VisualStateActor.
    """

    def __init__(
        self,
        actor_id: int,
        env_config: EnvConfig,
        mcts_config: MCTSConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
        initial_weights: Optional[Dict] = None,
        seed: Optional[int] = None,
        worker_device_str: str = "cpu", # Accept device string from config
        visual_state_actor_handle: Optional[
            ray.actor.ActorHandle
        ] = None,  # Handle for vis actor
    ):
        self.actor_id = actor_id
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.model_config = model_config
        self.train_config = train_config
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)
        self.worker_device_str = worker_device_str # Store configured device string
        self.visual_state_actor = visual_state_actor_handle  # Store handle

        # --- Configure Logging within the Actor ---
        worker_log_level = logging.INFO # Reduce default log level for workers
        log_format = (
            f"%(asctime)s [%(levelname)s] [W{self.actor_id}] %(name)s: %(message)s"
        )
        # Use basicConfig with force=True to ensure it works in Ray actors
        logging.basicConfig(level=worker_log_level, format=log_format, force=True)
        # Get logger instance *after* basicConfig
        global logger
        logger = logging.getLogger(__name__)
        # Set levels for other loggers if needed (e.g., reduce MCTS verbosity)
        logging.getLogger("src.mcts.core.search").setLevel(logging.INFO)
        logging.getLogger("src.mcts.strategy.selection").setLevel(logging.INFO)
        logging.getLogger("src.mcts.strategy.expansion").setLevel(logging.INFO)
        logging.getLogger("src.mcts.strategy.backpropagation").setLevel(logging.INFO)
        # -----------------------------------------

        set_random_seeds(self.seed)

        # Use the configured device string for this worker
        self.device = get_device(self.worker_device_str)
        self.nn_evaluator = NeuralNetwork(
            model_config=self.model_config,
            env_config=self.env_config,
            train_config=self.train_config,
            device=self.device, # Pass the determined device
        )
        if initial_weights:
            self.set_weights(initial_weights)
        else:
            self.nn_evaluator.model.eval()

        logger.info(
            f"Worker initialized on device {self.device}. Seed: {self.seed}. LogLevel: {logging.getLevelName(logger.getEffectiveLevel())}"
        )
        logger.debug("Worker init complete.")

    def set_weights(self, weights: Dict):
        """Updates the neural network weights."""
        try:
            # Ensure weights are loaded to the worker's specific device
            self.nn_evaluator.set_weights(weights)
            logger.debug(f"Weights updated.")
        except Exception as e:
            logger.error(f"Failed to set weights: {e}", exc_info=True)

    def _push_visual_state(self, game_state: GameState):
        """Asynchronously pushes the current game state to the visual actor."""
        if self.visual_state_actor:
            logger.debug(
                f"Pushing state (step {game_state.current_step}) to visual actor."
            )
            try:
                # Pass the game state directly, VisualStateActor handles it
                self.visual_state_actor.update_state.remote(self.actor_id, game_state)
            except Exception as e:
                logger.error(f"Failed to push visual state: {e}")

    def run_episode(self) -> SelfPlayResult:
        """
        Runs a single episode of self-play using MCTS and the internal neural network.
        Pushes state updates to the visual actor. Returns a SelfPlayResult Pydantic model.
        """
        self.nn_evaluator.model.eval()
        episode_seed = self.seed + random.randint(0, 1000)
        game = GameState(self.env_config, initial_seed=episode_seed)
        self._push_visual_state(game)  # Push initial state
        raw_experiences: List[Tuple[GameState, PolicyTargetMapping, float]] = []
        logger.debug(f"Starting episode with seed {episode_seed}")

        while not game.is_over():
            # Push state update before potentially long MCTS
            self._push_visual_state(game)

            root_node = Node(state=game)
            logger.debug(f"Running MCTS for step {game.current_step}...")
            mcts_max_depth = run_mcts_simulations(
                root_node, self.mcts_config, self.nn_evaluator
            )
            logger.debug(
                f"MCTS finished for step {game.current_step}. Max Depth: {mcts_max_depth}"
            )

            temp = (
                self.mcts_config.temperature_initial
                if game.current_step < self.mcts_config.temperature_anneal_steps
                else self.mcts_config.temperature_final
            )
            policy_target = get_policy_target(root_node, temperature=1.0)
            action = select_action_based_on_visits(root_node, temperature=temp)

            if action is None:
                logger.error(
                    f"MCTS failed to select action at step {game.current_step}. State: {game}. Aborting."
                )
                game.game_over = True
                break

            display_stats: Dict[str, Any] = {
                "game_step": game.current_step + 1,
                "mcts_simulations": self.mcts_config.num_simulations,
                "mcts_root_visits": root_node.visit_count,
                "mcts_temperature": temp,
                "mcts_root_value": root_node.value_estimate,
                "mcts_selected_action": action,
                "mcts_tree_depth": mcts_max_depth,
            }

            state_to_store = game.copy()
            state_to_store.display_stats = display_stats.copy()
            raw_experiences.append((state_to_store, policy_target, 0.0))

            _, _, done = game.step(action)

            # Attach stats to the main game object for the *next* state push
            game.display_stats = display_stats

            if done:
                break

        final_outcome = game.get_outcome()
        logger.debug(
            f"Episode finished. Outcome: {final_outcome}, Steps: {game.current_step}"
        )

        processed_experiences: List[Experience] = [
            (gs, policy, final_outcome) for gs, policy, _ in raw_experiences
        ]

        # Prepare final state for return and push final update
        if not hasattr(game, "display_stats") or not game.display_stats:
            game.display_stats = {}
        game.display_stats["game_step"] = game.current_step
        game.display_stats["final_score"] = final_outcome
        game.display_stats.pop("mcts_selected_action", None)
        last_mcts_depth = (
            raw_experiences[-1][0].display_stats.get("mcts_tree_depth", "?")
            if raw_experiences
            else "?"
        )
        game.display_stats["mcts_tree_depth"] = last_mcts_depth
        self._push_visual_state(game)  # Push final state

        # Return the Pydantic model instance
        return SelfPlayResult(
            episode_experiences=processed_experiences,
            final_score=final_outcome,
            episode_steps=game.current_step,
            final_game_state=game,
        )