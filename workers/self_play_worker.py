# File: workers/self_play_worker.py
import threading
import time
import queue
import traceback
import numpy as np
import torch
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional

from environment.game_state import GameState, StateType
from mcts import MCTS
from config import EnvConfig, MCTSConfig
from utils.types import ActionType

if TYPE_CHECKING:
    from agent.alphazero_net import AlphaZeroNet
    from stats.aggregator import StatsAggregator

# Raw data stored during the game: (state_features, mcts_policy_target, player_perspective)
ExperienceTuple = Tuple[StateType, Dict[ActionType, float], int]
# Data put into the queue: (state_features, mcts_policy_target, final_game_outcome)
ProcessedExperience = Tuple[StateType, Dict[ActionType, float], float]


class SelfPlayWorker(threading.Thread):
    """
    Worker thread that plays games against itself using MCTS and the current agent
    to generate training data.
    """

    def __init__(
        self,
        worker_id: int,
        agent: "AlphaZeroNet",
        mcts: MCTS,
        experience_queue: queue.Queue,
        stats_aggregator: "StatsAggregator",
        stop_event: threading.Event,
        env_config: EnvConfig,
        mcts_config: MCTSConfig,
        device: torch.device,
        games_per_iteration: int = 1,
        max_game_steps: Optional[int] = None,
    ):
        super().__init__(daemon=True, name=f"SelfPlayWorker-{worker_id}")
        self.worker_id = worker_id
        self.agent = agent
        self.mcts = mcts
        self.experience_queue = experience_queue
        self.stats_aggregator = stats_aggregator
        self.stop_event = stop_event
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.device = device
        self.games_per_iteration = games_per_iteration
        self.max_game_steps = max_game_steps if max_game_steps else float("inf")

        print(f"[SelfPlayWorker-{self.worker_id}] Initialized.")

    def get_init_args(self) -> Dict[str, Any]:
        """Returns arguments needed to re-initialize the thread."""
        return {
            "worker_id": self.worker_id,
            "agent": self.agent,
            "mcts": self.mcts,
            "experience_queue": self.experience_queue,
            "stats_aggregator": self.stats_aggregator,
            "stop_event": self.stop_event,
            "env_config": self.env_config,
            "mcts_config": self.mcts_config,
            "device": self.device,
            "games_per_iteration": self.games_per_iteration,
            "max_game_steps": (
                self.max_game_steps if self.max_game_steps != float("inf") else None
            ),
        }

    def _get_temperature(self, game_step: int) -> float:
        """Calculates the MCTS temperature based on the game step."""
        if game_step < self.mcts_config.TEMPERATURE_ANNEAL_STEPS:
            progress = game_step / max(
                1, self.mcts_config.TEMPERATURE_ANNEAL_STEPS
            )  # Avoid division by zero
            temp = (
                self.mcts_config.TEMPERATURE_INITIAL * (1 - progress)
                + self.mcts_config.TEMPERATURE_FINAL * progress
            )
            return temp
        else:
            return self.mcts_config.TEMPERATURE_FINAL

    def run(self):
        """Main loop for the self-play worker."""
        print(f"[SelfPlayWorker-{self.worker_id}] Starting run loop.")
        game_count = 0
        while not self.stop_event.is_set():
            try:
                start_time = time.time()
                game_data: List[ExperienceTuple] = []
                game = GameState()  # The GameState object for the current game
                current_state = game.reset()
                game_steps = 0

                while not game.is_over() and game_steps < self.max_game_steps:
                    if self.stop_event.is_set():
                        break

                    # Ensure agent is in eval mode for MCTS predictions
                    self.agent.eval()
                    with torch.no_grad():
                        root_node = self.mcts.run_simulations(
                            root_state=game,
                            num_simulations=self.mcts_config.NUM_SIMULATIONS,
                        )

                    temperature = self._get_temperature(game_steps)
                    policy_target = self.mcts.get_policy_target(root_node, temperature)

                    # Store state *before* action is taken
                    # Player perspective is always 1 for single-player
                    game_data.append((current_state, policy_target, 1))

                    action = self.mcts.choose_action(root_node, temperature)
                    _, done = game.step(action)
                    current_state = game.get_state()  # Get new state features
                    game_steps += 1

                if self.stop_event.is_set():
                    break

                final_outcome = game.get_outcome()  # Get final outcome (-1, 0, or 1)
                processed_data: List[ProcessedExperience] = []
                for state_features, policy, player in game_data:
                    # The outcome is the final game result from the perspective of the player at that state
                    # In single player, player is always 1, so outcome_for_player = final_outcome
                    outcome_for_player = final_outcome * player
                    processed_data.append((state_features, policy, outcome_for_player))

                if processed_data:
                    try:
                        # Put the list of tuples for the entire game into the queue
                        self.experience_queue.put(processed_data, timeout=1.0)
                    except queue.Full:
                        print(
                            f"[SelfPlayWorker-{self.worker_id}] Warning: Experience queue full. Discarding game data."
                        )
                        # Optionally sleep briefly to allow training worker to catch up
                        time.sleep(0.1)
                    except Exception as q_err:
                        print(
                            f"[SelfPlayWorker-{self.worker_id}] Error putting data in queue: {q_err}"
                        )

                game_duration = time.time() - start_time
                # Pass the current global step from the aggregator for accurate best score tracking
                current_global_step = self.stats_aggregator.storage.current_global_step
                # Pass the completed game object to record_episode
                self.stats_aggregator.record_episode(
                    episode_score=final_outcome,  # Use the final outcome
                    episode_length=game_steps,
                    episode_num=self.stats_aggregator.storage.total_episodes + 1,
                    global_step=current_global_step,
                    game_score=game.game_score,
                    triangles_cleared=game.triangles_cleared_this_episode,
                    game_state_for_best=game,  # Pass the GameState object
                )
                game_count += 1

            except Exception as e:
                print(
                    f"[SelfPlayWorker-{self.worker_id}] CRITICAL ERROR in run loop: {e}"
                )
                traceback.print_exc()
                time.sleep(5)  # Avoid rapid error loops

        print(f"[SelfPlayWorker-{self.worker_id}] Run loop finished.")
