import threading
import time
import queue
import traceback
import torch
import copy
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
import logging
import numpy as np
import multiprocessing as mp
import ray
import asyncio

from environment.game_state import GameState, StateType
from environment.shape import Shape
from mcts import MCTS
from config import EnvConfig, MCTSConfig, TrainConfig
from utils.types import ActionType

if TYPE_CHECKING:
    from ray.util.queue import Queue as RayQueue

    AgentPredictorHandle = ray.actor.ActorHandle
    StatsAggregatorHandle = ray.actor.ActorHandle

ExperienceTuple = Tuple[StateType, Dict[ActionType, float], int]
ProcessedExperienceBatch = List[Tuple[StateType, Dict[ActionType, float], float]]

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class SelfPlayWorker:
    """Plays games using MCTS to generate training data (Ray Actor)."""

    INTERMEDIATE_STATS_INTERVAL_SEC = 5.0

    def __init__(
        self,
        worker_id: int,
        agent_predictor: "AgentPredictorHandle",
        mcts_config: MCTSConfig,
        env_config: EnvConfig,
        experience_queue: "RayQueue",
        stats_aggregator: "StatsAggregatorHandle",
        max_game_steps: Optional[int] = None,
    ):
        self.worker_id = worker_id
        self.agent_predictor = agent_predictor
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.mcts = MCTS(
            agent_predictor=self.agent_predictor,
            config=self.mcts_config,
            env_config=self.env_config,
        )
        self.experience_queue = experience_queue
        self.stats_aggregator = stats_aggregator
        self.max_game_steps = max_game_steps if max_game_steps else float("inf")
        self.log_prefix = f"[SelfPlayWorker-{self.worker_id}]"
        self.last_intermediate_stats_time = 0.0

        self._current_render_state_dict: Optional[StateType] = None
        self._last_stats: Dict[str, Any] = {
            "status": "Initialized",
            "game_steps": 0,
            "game_score": 0,
            "mcts_sim_time": 0.0,
            "mcts_nn_time": 0.0,
            "mcts_nodes_explored": 0,
            "mcts_avg_depth": 0.0,
            "available_shapes_data": [],
        }
        self._stop_requested = False
        logger.info(f"{self.log_prefix} Initialized as Ray Actor.")

    async def get_current_render_data(self) -> Optional[Dict[str, Any]]:
        """Returns a serializable dictionary for rendering (async)."""
        if self._current_render_state_dict:
            return {
                "state_dict": self._current_render_state_dict,
                "stats": self._last_stats.copy(),
            }
        else:
            return {"state_dict": None, "stats": self._last_stats.copy()}

    def _update_render_state(
        self, game_state: Optional[GameState], stats: Dict[str, Any]
    ):
        """Updates the state dictionary and stats exposed for rendering."""
        if game_state:
            try:
                self._current_render_state_dict = game_state.get_state()
                available_shapes_data = []
                for shape_obj in game_state.shapes:
                    if shape_obj:
                        available_shapes_data.append(
                            {"triangles": shape_obj.triangles, "color": shape_obj.color}
                        )
                    else:
                        available_shapes_data.append(None)
            except Exception as e:
                logger.error(f"{self.log_prefix} Error getting game state dict: {e}")
                self._current_render_state_dict = None
                available_shapes_data = []
        else:
            self._current_render_state_dict = None
            available_shapes_data = []

        current_game_score = (
            game_state.game_score
            if game_state
            else self._last_stats.get("game_score", 0)
        )

        ui_stats = {
            "status": stats.get("status", self._last_stats.get("status", "Unknown")),
            "game_steps": stats.get(
                "game_steps", self._last_stats.get("game_steps", 0)
            ),
            "game_score": current_game_score,
            "mcts_sim_time": stats.get(
                "mcts_total_duration", self._last_stats.get("mcts_sim_time", 0.0)
            ),
            "mcts_nn_time": stats.get(
                "total_nn_prediction_time", self._last_stats.get("mcts_nn_time", 0.0)
            ),
            "mcts_nodes_explored": stats.get(
                "nodes_created", self._last_stats.get("mcts_nodes_explored", 0)
            ),
            "mcts_avg_depth": stats.get(
                "avg_leaf_depth", self._last_stats.get("mcts_avg_depth", 0.0)
            ),
            "available_shapes_data": available_shapes_data,
        }
        self._last_stats.update(ui_stats)

    def _get_temperature(self, game_step: int) -> float:
        """Calculates the MCTS temperature based on the game step."""
        if game_step < self.mcts_config.TEMPERATURE_ANNEAL_STEPS:
            progress = game_step / max(1, self.mcts_config.TEMPERATURE_ANNEAL_STEPS)
            return (
                self.mcts_config.TEMPERATURE_INITIAL * (1 - progress)
                + self.mcts_config.TEMPERATURE_FINAL * progress
            )
        return self.mcts_config.TEMPERATURE_FINAL

    async def _play_one_game(self) -> Optional[ProcessedExperienceBatch]:
        """Plays a single game and returns the processed experience (async)."""
        current_game_num_ref = self.stats_aggregator.get_total_episodes.remote()
        current_game_num = await current_game_num_ref + 1
        logger.info(f"{self.log_prefix} Starting game {current_game_num}")
        start_time = time.monotonic()
        game_data: List[ExperienceTuple] = []
        game = GameState()
        current_state_features = game.reset()
        game_steps = 0
        self.last_intermediate_stats_time = time.monotonic()
        self._update_render_state(game, {"status": "Starting", "game_steps": 0})

        # Initial Stats Update (Async)
        # Call qsize() directly, assume it returns int (based on error)
        buffer_size = self.experience_queue.qsize()
        recording_step = {
            "current_self_play_game_number": current_game_num,
            "current_self_play_game_steps": 0,
            "buffer_size": buffer_size,
        }
        self.stats_aggregator.record_step.remote(recording_step)
        logger.info(
            f"{self.log_prefix} Game {current_game_num} started. Buffer size: {buffer_size}"
        )

        while not game.is_over() and game_steps < self.max_game_steps:
            if self._stop_requested:
                logger.info(
                    f"{self.log_prefix} Stop requested during game {current_game_num}. Aborting."
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
                self._update_render_state(
                    game, {"status": "Running", "game_steps": game_steps}
                )
                # Call qsize() directly, assume it returns int
                buffer_size = self.experience_queue.qsize()
                recording_step = {
                    "current_self_play_game_number": current_game_num,
                    "current_self_play_game_steps": game_steps,
                    "buffer_size": buffer_size,
                }
                self.stats_aggregator.record_step.remote(recording_step)
                self.last_intermediate_stats_time = current_time

            mcts_start_time = time.monotonic()
            try:
                root_node, mcts_stats = self.mcts.run_simulations(
                    root_state=game, num_simulations=self.mcts_config.NUM_SIMULATIONS
                )
            except Exception as mcts_err:
                if self._stop_requested:
                    logger.info(
                        f"{self.log_prefix} MCTS interrupted (stop requested) for game {current_game_num}, step {game_steps}. Aborting game."
                    )
                    self._update_render_state(
                        game, {"status": "Stopped", "game_steps": game_steps}
                    )
                    return None
                else:
                    logger.error(
                        f"{self.log_prefix} MCTS failed for game {current_game_num}, step {game_steps}: {mcts_err}",
                        exc_info=True,
                    )
                    game.game_over = True
                    break

            mcts_duration = time.monotonic() - mcts_start_time
            mcts_stats["game_steps"] = game_steps
            mcts_stats["status"] = "Running (MCTS)"
            logger.debug(
                f"{self.log_prefix} Game {current_game_num} Step {game_steps}: MCTS took {mcts_duration:.4f}s"
            )
            self._update_render_state(game, mcts_stats)

            temperature = self._get_temperature(game_steps)
            policy_target = self.mcts.get_policy_target(root_node, temperature)
            game_data.append((copy.deepcopy(current_state_features), policy_target, 1))

            action = self.mcts.choose_action(root_node, temperature)
            if action == -1:
                logger.error(
                    f"{self.log_prefix} MCTS failed to choose an action. Aborting game {current_game_num}."
                )
                game.game_over = True
                break

            step_start_time = time.monotonic()
            _, done = game.step(action)
            step_duration = time.monotonic() - step_start_time
            logger.debug(
                f"{self.log_prefix} Game {current_game_num} Step {game_steps}: Game step took {step_duration:.4f}s"
            )
            current_state_features = game.get_state()
            game_steps += 1

            # Record MCTS Stats (Async Fire-and-forget)
            # Call qsize() directly, assume it returns int
            buffer_size = self.experience_queue.qsize()
            step_stats_for_aggregator = {
                "mcts_sim_time": mcts_stats.get("mcts_total_duration", 0.0),
                "mcts_nn_time": mcts_stats.get("total_nn_prediction_time", 0.0),
                "mcts_nodes_explored": mcts_stats.get("nodes_created", 0),
                "mcts_avg_depth": mcts_stats.get("avg_leaf_depth", 0.0),
                "buffer_size": buffer_size,
            }
            self.stats_aggregator.record_step.remote(step_stats_for_aggregator)

        status = (
            "Finished (Max Steps)"
            if game_steps >= self.max_game_steps and not game.is_over()
            else "Finished"
        )
        self._update_render_state(game, {"status": status, "game_steps": game_steps})

        if self._stop_requested:
            logger.info(
                f"{self.log_prefix} Stop requested after game {current_game_num} finished. Not processing."
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

        final_state_for_best = game.get_state()
        self.stats_aggregator.record_episode.remote(
            episode_outcome=final_outcome,
            episode_length=game_steps,
            episode_num=current_game_num,
            game_score=game.game_score,
            triangles_cleared=game.triangles_cleared_this_episode,
            game_state_for_best=final_state_for_best,
        )
        return processed_data

    async def run_loop(self):
        """Main loop for the self-play worker actor (async)."""
        logger.info(f"{self.log_prefix} Starting run loop.")
        while not self._stop_requested:
            try:
                processed_data = await self._play_one_game()
                if processed_data is None:
                    if self._stop_requested:
                        break
                    else:
                        logger.warning(
                            f"{self.log_prefix} Game play returned None without stop signal. Continuing."
                        )
                        await asyncio.sleep(0.5)
                        continue

                if processed_data:
                    try:
                        q_put_start = time.monotonic()
                        # Use put_async for putting data
                        await self.experience_queue.put_async(
                            processed_data, timeout=1.0
                        )
                        q_put_duration = time.monotonic() - q_put_start
                        # Call qsize() directly, assume it returns int
                        qsize = self.experience_queue.qsize()
                        logger.debug(
                            f"{self.log_prefix} Added game data to queue (qsize: {qsize}) in {q_put_duration:.4f}s."
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"{self.log_prefix} Experience queue put timed out. Discarding game data."
                        )
                    except Exception as q_err:
                        logger.error(
                            f"{self.log_prefix} Error putting data in queue: {q_err}"
                        )
                        if self._stop_requested:
                            break
                        await asyncio.sleep(0.5)

            except Exception as e:
                logger.critical(
                    f"{self.log_prefix} CRITICAL ERROR in run loop: {e}", exc_info=True
                )
                self._update_render_state(None, {"status": "Error"})
                if self._stop_requested:
                    break
                await asyncio.sleep(5.0)

        logger.info(f"{self.log_prefix} Run loop finished.")
        self._update_render_state(None, {"status": "Stopped"})

    def stop(self):
        """Signals the actor to stop gracefully."""
        logger.info(f"{self.log_prefix} Stop requested.")
        self._stop_requested = True

    def health_check(self):
        """Ray health check method."""
        return "OK"
