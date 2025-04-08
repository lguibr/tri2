# File: workers/self_play_worker.py
import threading
import time
import queue
import traceback
import torch
import copy
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
import logging
import numpy as np

from environment.game_state import GameState, StateType
from mcts import MCTS
from config import EnvConfig, MCTSConfig
from utils.types import ActionType

if TYPE_CHECKING:
    from agent.alphazero_net import AlphaZeroNet
    from stats.aggregator import StatsAggregator

ExperienceTuple = Tuple[StateType, Dict[ActionType, float], int]
ProcessedExperienceBatch = List[Tuple[StateType, Dict[ActionType, float], float]]

logger = logging.getLogger(__name__)


class SelfPlayWorker(threading.Thread):
    """Plays games using MCTS to generate training data."""

    INTERMEDIATE_STATS_INTERVAL_SEC = 5.0

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
        self.log_prefix = f"[SelfPlayWorker-{self.worker_id}]"
        self.last_intermediate_stats_time = 0.0

        # --- State for UI Rendering ---
        self._current_game_state_lock = threading.Lock()
        self._current_game_state: Optional[GameState] = None
        self._last_stats: Dict[str, Any] = {
            "status": "Initialized",
            "game_steps": 0,
        }  # Include game_steps
        # --- End State for UI Rendering ---

        logger.info(f"{self.log_prefix} Initialized.")

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

    # --- Methods for UI Rendering ---
    def get_current_render_data(self) -> Dict[str, Any]:
        """Returns a dictionary containing state copy and last stats (thread-safe)."""
        with self._current_game_state_lock:
            state_copy = None
            if self._current_game_state:
                try:
                    state_copy = copy.deepcopy(self._current_game_state)
                except Exception as e:
                    logger.error(f"{self.log_prefix} Error deepcopying game state: {e}")
            return {"state": state_copy, "stats": self._last_stats.copy()}

    def _update_render_state(self, game_state: GameState, stats: Dict[str, Any]):
        """Updates the state exposed for rendering (thread-safe)."""
        with self._current_game_state_lock:
            self._current_game_state = game_state
            self._last_stats = stats.copy()  # Store a copy of the stats dict
            # Update timers within the game state for visual effects
            if hasattr(game_state, "_update_timers"):
                game_state._update_timers()

    # --- End Methods for UI Rendering ---

    def _get_temperature(self, game_step: int) -> float:
        """Calculates the MCTS temperature based on the game step."""
        if game_step < self.mcts_config.TEMPERATURE_ANNEAL_STEPS:
            progress = game_step / max(1, self.mcts_config.TEMPERATURE_ANNEAL_STEPS)
            return (
                self.mcts_config.TEMPERATURE_INITIAL * (1 - progress)
                + self.mcts_config.TEMPERATURE_FINAL * progress
            )
        return self.mcts_config.TEMPERATURE_FINAL

    def _play_one_game(self) -> Optional[ProcessedExperienceBatch]:
        """Plays a single game and returns the processed experience."""
        current_game_num = self.stats_aggregator.storage.total_episodes + 1
        logger.info(f"{self.log_prefix} Starting game {current_game_num}")
        start_time = time.monotonic()
        game_data: List[ExperienceTuple] = []
        game = GameState()
        current_state_features = game.reset()
        game_steps = 0
        self.last_intermediate_stats_time = time.monotonic()

        # Initial state update for UI
        self._update_render_state(game, {"status": "Starting", "game_steps": 0})

        recording_step = {
            "current_self_play_game_number": current_game_num,
            "current_self_play_game_steps": 0,
            "buffer_size": self.experience_queue.qsize(),
        }
        self.stats_aggregator.record_step(recording_step)
        logger.info(
            f"{self.log_prefix} Game {current_game_num} started. Buffer size: {recording_step['buffer_size']}"
        )

        while not game.is_over() and game_steps < self.max_game_steps:
            if self.stop_event.is_set():
                logger.info(
                    f"{self.log_prefix} Stop event set during game {current_game_num}. Aborting."
                )
                self._update_render_state(
                    game, {"status": "Stopped", "game_steps": game_steps}
                )
                return None

            current_time = time.monotonic()
            if (
                current_time - self.last_intermediate_stats_time
                > self.INTERMEDIATE_STATS_INTERVAL_SEC
            ):
                recording_step = {
                    "current_self_play_game_number": current_game_num,
                    "current_self_play_game_steps": game_steps,
                    "buffer_size": self.experience_queue.qsize(),
                }
                self.stats_aggregator.record_step(recording_step)
                self.last_intermediate_stats_time = current_time

            mcts_start_time = time.monotonic()
            self.agent.eval()
            with torch.no_grad():
                root_node, mcts_stats = self.mcts.run_simulations(
                    root_state=game, num_simulations=self.mcts_config.NUM_SIMULATIONS
                )
            mcts_duration = time.monotonic() - mcts_start_time
            mcts_stats["mcts_total_duration"] = mcts_duration
            mcts_stats["game_steps"] = game_steps  # Add game steps to stats dict
            logger.debug(
                f"{self.log_prefix} Game {current_game_num} Step {game_steps}: MCTS took {mcts_duration:.4f}s"
            )

            # Update render state *before* choosing action/stepping
            self._update_render_state(game, mcts_stats)

            temperature = self._get_temperature(game_steps)
            policy_target = self.mcts.get_policy_target(root_node, temperature)
            game_data.append((current_state_features, policy_target, 1))

            action = self.mcts.choose_action(root_node, temperature)
            step_start_time = time.monotonic()
            _, done = game.step(action)
            step_duration = time.monotonic() - step_start_time
            logger.debug(
                f"{self.log_prefix} Game {current_game_num} Step {game_steps}: Game step took {step_duration:.4f}s"
            )

            current_state_features = game.get_state()
            game_steps += 1

            # Record MCTS stats for this step
            step_stats_for_aggregator = {
                "mcts_sim_time": mcts_stats.get("mcts_total_duration", 0.0),
                "mcts_nn_time": mcts_stats.get("total_nn_prediction_time", 0.0),
                "mcts_nodes_explored": mcts_stats.get("nodes_created", 0),
                "mcts_avg_depth": mcts_stats.get("avg_leaf_depth", 0.0),
                "buffer_size": self.experience_queue.qsize(),
            }
            self.stats_aggregator.record_step(step_stats_for_aggregator)

        # Final state update for UI
        self._update_render_state(
            game, {"status": "Finished", "game_steps": game_steps}
        )

        if self.stop_event.is_set():
            logger.info(
                f"{self.log_prefix} Stop event set after game {current_game_num} finished. Not processing."
            )
            return None

        final_outcome = game.get_outcome()
        processed_data: ProcessedExperienceBatch = [
            (state, policy, final_outcome * player)
            for state, policy, player in game_data
        ]

        game_duration = time.monotonic() - start_time
        logger.info(
            f"{self.log_prefix} Game {current_game_num} finished in {game_duration:.2f}s "
            f"({game_steps} steps). Outcome: {final_outcome}, Score: {game.game_score}. "
            f"Queueing {len(processed_data)} experiences."
        )

        current_global_step = self.stats_aggregator.storage.current_global_step
        self.stats_aggregator.record_episode(
            episode_outcome=final_outcome,
            episode_length=game_steps,
            episode_num=current_game_num,
            global_step=current_global_step,
            game_score=game.game_score,
            triangles_cleared=game.triangles_cleared_this_episode,
            game_state_for_best=game,
        )
        return processed_data

    def run(self):
        """Main loop for the self-play worker."""
        logger.info(f"{self.log_prefix} Starting run loop.")
        while not self.stop_event.is_set():
            try:
                processed_data = self._play_one_game()
                if processed_data is None:
                    break
                if processed_data:
                    try:
                        q_put_start = time.monotonic()
                        self.experience_queue.put_nowait(processed_data)
                        q_put_duration = time.monotonic() - q_put_start
                        logger.debug(
                            f"{self.log_prefix} Added game data to queue (qsize: {self.experience_queue.qsize()}) "
                            f"in {q_put_duration:.4f}s."
                        )
                    except queue.Full:
                        logger.warning(
                            f"{self.log_prefix} Experience queue full. Discarding game data."
                        )
                        time.sleep(0.1)
                    except Exception as q_err:
                        logger.error(
                            f"{self.log_prefix} Error putting data in queue: {q_err}"
                        )
            except Exception as e:
                logger.critical(
                    f"{self.log_prefix} CRITICAL ERROR in run loop: {e}", exc_info=True
                )
                error_state = GameState()
                error_state.game_over = True
                self._update_render_state(
                    error_state, {"status": "Error", "game_steps": 0}
                )
                time.sleep(1.0)

        logger.info(f"{self.log_prefix} Run loop finished.")
        with self._current_game_state_lock:
            self._current_game_state = None
            self._last_stats = {"status": "Stopped", "game_steps": 0}
