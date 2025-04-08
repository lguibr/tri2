import threading
import time
import queue
import traceback
import torch
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
import logging

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

    # Add a constant for how often to report intermediate progress
    INTERMEDIATE_STATS_INTERVAL_SEC = 5.0  # Report every 5 seconds

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
        self.last_intermediate_stats_time = 0.0  # Track time for intermediate stats
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
        # Get game number *before* incrementing total_episodes in aggregator
        # Add 1 because total_episodes is 0-based count of *completed* games
        current_game_num = self.stats_aggregator.storage.total_episodes + 1
        logger.info(f"{self.log_prefix} Starting game {current_game_num}")
        start_time = time.monotonic()
        game_data: List[ExperienceTuple] = []
        game = GameState()
        current_state_features = game.reset()
        game_steps = 0
        self.last_intermediate_stats_time = (
            time.monotonic()
        )  # Reset timer for this game
        recording_step = {
            "current_self_play_game_number": current_game_num,
            "current_self_play_game_steps": 0,
            "buffer_size": self.experience_queue.qsize(),  # Include buffer size
        }
        # Report starting the game immediately
        self.stats_aggregator.record_step(recording_step)
        logger.info(
            f"{self.log_prefix} Game {current_game_num} started. Buffer size: {recording_step['buffer_size']} and recording_step {recording_step}."
        )

        while not game.is_over() and game_steps < self.max_game_steps:
            if self.stop_event.is_set():
                logger.info(
                    f"{self.log_prefix} Stop event set during game {current_game_num}. Aborting."
                )
                return None  # Stop early

            # --- Report Intermediate Progress ---
            current_time = time.monotonic()
            if (
                current_time - self.last_intermediate_stats_time
                > self.INTERMEDIATE_STATS_INTERVAL_SEC
            ):
                recording_step = {
                    "current_self_play_game_number": current_game_num,
                    "current_self_play_game_steps": game_steps,
                    "buffer_size": self.experience_queue.qsize(),  # Include buffer size
                }
                logger.info(
                    f"{self.log_prefix} Game {current_game_num} started. Buffer size: {recording_step['buffer_size']} and recording_step {recording_step}."
                )
                self.stats_aggregator.record_step(recording_step)
                self.last_intermediate_stats_time = current_time
            # --- End Intermediate Progress ---

            mcts_start_time = time.monotonic()
            self.agent.eval()  # Ensure agent is in eval mode
            with torch.no_grad():
                root_node = self.mcts.run_simulations(
                    root_state=game, num_simulations=self.mcts_config.NUM_SIMULATIONS
                )
            mcts_duration = time.monotonic() - mcts_start_time
            logger.debug(
                f"{self.log_prefix} Game {current_game_num} Step {game_steps}: MCTS took {mcts_duration:.4f}s"
            )

            temperature = self._get_temperature(game_steps)
            policy_target = self.mcts.get_policy_target(root_node, temperature)
            game_data.append(
                (current_state_features, policy_target, 1)
            )  # Player 1 perspective

            action = self.mcts.choose_action(root_node, temperature)
            step_start_time = time.monotonic()
            _, done = game.step(action)
            step_duration = time.monotonic() - step_start_time
            logger.debug(
                f"{self.log_prefix} Game {current_game_num} Step {game_steps}: Game step took {step_duration:.4f}s"
            )

            current_state_features = game.get_state()
            game_steps += 1

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

        # Record episode stats *after* processing data
        current_global_step = self.stats_aggregator.storage.current_global_step
        self.stats_aggregator.record_episode(
            episode_outcome=final_outcome,
            episode_length=game_steps,
            episode_num=current_game_num,  # Pass the game number used during play
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
                    # This happens if stop_event was set during the game
                    break
                if processed_data:
                    try:
                        q_put_start = time.monotonic()
                        # Use put_nowait or a short timeout to avoid blocking if queue is full
                        # self.experience_queue.put(processed_data, timeout=1.0)
                        self.experience_queue.put_nowait(processed_data)
                        q_put_duration = time.monotonic() - q_put_start
                        logger.debug(
                            f"{self.log_prefix} Added game data to queue (qsize: {self.experience_queue.qsize()}) "
                            f"in {q_put_duration:.4f}s."
                        )
                    except queue.Full:
                        logger.warning(
                            f"{self.log_prefix} Experience queue full. Discarding game data. Consider increasing buffer or reducing self-play workers."
                        )
                        # Optional: Sleep briefly to avoid busy-waiting if queue stays full
                        time.sleep(0.1)
                    except Exception as q_err:
                        logger.error(
                            f"{self.log_prefix} Error putting data in queue: {q_err}"
                        )
            except Exception as e:
                logger.critical(
                    f"{self.log_prefix} CRITICAL ERROR in run loop: {e}", exc_info=True
                )
                # Optional: Add a small delay to prevent rapid error loops
                time.sleep(1.0)

        logger.info(f"{self.log_prefix} Run loop finished.")
