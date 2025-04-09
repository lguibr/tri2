import time
import queue
import logging
import logging.handlers
import multiprocessing as mp
import traceback
import sys
from typing import Optional, Dict, Any
import ray
from ray.util.queue import Queue as RayQueue
import asyncio  # Added asyncio

try:
    from config import VisConfig, EnvConfig, TrainConfig, MCTSConfig, set_device
    from utils.helpers import get_device as get_torch_device, set_random_seeds
    from utils.init_checks import run_pre_checks
    from app_state import AppState
    from app_init import AppInitializer
    from app_logic import AppLogic
    from app_workers import AppWorkerManager
    from app_setup import initialize_directories
    from environment.game_state import GameState
except ImportError as e:
    print(f"[Logic Process Import Error] {e}", file=sys.stderr)
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    sys.exit(1)

RENDER_DATA_SENTINEL = "RENDER_DATA"
COMMAND_SENTINEL = "COMMAND"
STOP_SENTINEL = "STOP"
ERROR_SENTINEL = "ERROR"
PAYLOAD_KEY = "payload"


def run_logic_process(
    stop_event: mp.Event,
    command_queue: mp.Queue,
    render_data_queue: mp.Queue,
    checkpoint_to_load: Optional[str],
    log_queue: Optional[mp.Queue] = None,
):
    ray_initialized = False
    try:
        if not ray.is_initialized():
            ray.init(logging_level=logging.WARNING, ignore_reinit_error=True)
            ray_initialized = True
            print("[Logic Process] Ray initialized.")
        else:
            print("[Logic Process] Ray already initialized.")
            ray_initialized = True
    except Exception as ray_init_err:
        print(
            f"[Logic Process] FATAL: Ray initialization failed: {ray_init_err}",
            file=sys.stderr,
        )
        stop_event.set()
        try:
            render_data_queue.put({ERROR_SENTINEL: f"Ray Init Failed: {ray_init_err}"})
        except Exception:
            pass
        return

    if log_queue:
        qh = logging.handlers.QueueHandler(log_queue)
        root = logging.getLogger()
        if not any(isinstance(h, logging.handlers.QueueHandler) for h in root.handlers):
            root.addHandler(qh)
            root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Logic Process starting...")
    logic_start_time = time.time()

    logic_app_state = None
    try:
        vis_config = VisConfig()
        env_config = EnvConfig()
        train_config_instance = TrainConfig()
        mcts_config = MCTSConfig()
        worker_stop_event = mp.Event()

        # Use Ray Queue Actor
        experience_ray_queue = RayQueue(
            maxsize=train_config_instance.BUFFER_CAPACITY * 2
        )
        logger.info(
            f"[Logic Process] Ray Experience Queue created (maxsize={train_config_instance.BUFFER_CAPACITY * 2})."
        )

        logic_app_state = type(
            "LogicAppState",
            (object,),
            {
                "vis_config": vis_config,
                "env_config": env_config,
                "train_config_instance": train_config_instance,
                "mcts_config": mcts_config,
                "app_state": AppState.INITIALIZING,
                "status": "Initializing...",
                "stop_event": stop_event,
                "worker_stop_event": worker_stop_event,
                "experience_queue": experience_ray_queue,  # Use Ray Queue handle
                "device": get_torch_device(),
                "checkpoint_to_load": checkpoint_to_load,
                "initializer": None,
                "logic": None,
                "worker_manager": None,
                "agent_predictor": None,
                "stats_aggregator": None,  # Actor handles
                "ui_utils": None,
                "cleanup_confirmation_active": False,
                "cleanup_message": "",
                "last_cleanup_message_time": 0.0,
                "total_gpu_memory_bytes": None,
                "current_global_step": 0,
                "set_state": lambda self, new_state: setattr(
                    self, "app_state", new_state
                ),
                "set_status": lambda self, new_status: setattr(
                    self, "status", new_status
                ),
                "set_cleanup_confirmation": lambda self, active: setattr(
                    self, "cleanup_confirmation_active", active
                ),
                "set_cleanup_message": lambda self, msg, msg_time: (
                    setattr(self, "cleanup_message", msg),
                    setattr(self, "last_cleanup_message_time", msg_time),
                ),
                "get_render_data": None,
            },
        )()
        set_device(logic_app_state.device)

        initializer = AppInitializer(logic_app_state)
        logic = AppLogic(logic_app_state)
        worker_manager = AppWorkerManager(logic_app_state)
        logic_app_state.initializer = initializer
        logic_app_state.logic = logic
        logic_app_state.worker_manager = worker_manager

        logger.info("Initializing directories...")
        initialize_directories()
        set_random_seeds(
            mcts_config.RANDOM_SEED if hasattr(mcts_config, "RANDOM_SEED") else 42
        )
        logger.info("Running pre-checks...")
        run_pre_checks()
        logger.info("Initializing RL components and Ray actors...")
        initializer.initialize_logic_components()  # Initializes actors

        logic_app_state.set_state(AppState.MAIN_MENU)
        logic_app_state.set_status("Ready")
        logic.check_initial_completion_status()
        logger.info("--- Logic Initialization Complete ---")

        # Define async get_render_data
        async def _get_render_data_async(app_obj) -> Dict[str, Any]:
            worker_render_task = None
            if (
                app_obj.worker_manager.is_any_worker_running()
                and app_obj.app_state == AppState.MAIN_MENU
            ):
                num_to_render = app_obj.vis_config.NUM_ENVS_TO_RENDER
                if num_to_render > 0:
                    worker_render_task = (
                        app_obj.worker_manager.get_worker_render_data_async(
                            num_to_render
                        )
                    )

            # Fetch stats data from StatsAggregatorActor
            plot_data_ref, summary_ref, best_game_ref = None, None, None
            if app_obj.stats_aggregator:  # Check if handle exists
                plot_data_ref = app_obj.stats_aggregator.get_plot_data.remote()
                summary_ref = app_obj.stats_aggregator.get_summary.remote(
                    app_obj.current_global_step
                )
                best_game_ref = (
                    app_obj.stats_aggregator.get_best_game_state_data.remote()
                )

            # Gather results concurrently
            results = await asyncio.gather(
                (
                    worker_render_task
                    if worker_render_task
                    else asyncio.sleep(0, result=[])
                ),  # Handle no task case
                plot_data_ref if plot_data_ref else asyncio.sleep(0, result={}),
                summary_ref if summary_ref else asyncio.sleep(0, result={}),
                best_game_ref if best_game_ref else asyncio.sleep(0, result=None),
                return_exceptions=True,  # Handle potential errors from remote calls
            )

            # Process results, handling potential errors
            worker_render_data_result = (
                results[0]
                if not isinstance(results[0], Exception)
                else ([None] * app_obj.vis_config.NUM_ENVS_TO_RENDER)
            )
            plot_data = results[1] if not isinstance(results[1], Exception) else {}
            stats_summary = results[2] if not isinstance(results[2], Exception) else {}
            best_game_state_data = (
                results[3] if not isinstance(results[3], Exception) else None
            )

            # Log errors if any occurred during gather
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    task_name = ["worker_render", "plot_data", "summary", "best_game"][
                        i
                    ]
                    logger.error(f"Error fetching {task_name} from actor: {res}")

            data = {
                "app_state": app_obj.app_state.value,
                "status": app_obj.status,
                "cleanup_confirmation_active": app_obj.cleanup_confirmation_active,
                "cleanup_message": app_obj.cleanup_message,
                "last_cleanup_message_time": app_obj.last_cleanup_message_time,
                "update_progress_details": {},
                "demo_env_state": (
                    app_obj.demo_env.get_state() if app_obj.demo_env else None
                ),
                "demo_env_is_over": (
                    app_obj.demo_env.is_over() if app_obj.demo_env else False
                ),
                "demo_env_score": (
                    app_obj.demo_env.game_score if app_obj.demo_env else 0
                ),
                "demo_env_dragged_shape_idx": (
                    app_obj.demo_env.demo_dragged_shape_idx
                    if app_obj.demo_env
                    else None
                ),
                "demo_env_snapped_pos": (
                    app_obj.demo_env.demo_snapped_position if app_obj.demo_env else None
                ),
                "demo_env_selected_shape_idx": (
                    app_obj.demo_env.demo_selected_shape_idx if app_obj.demo_env else -1
                ),
                "env_config_rows": app_obj.env_config.ROWS,
                "env_config_cols": app_obj.env_config.COLS,
                "env_config_num_shape_slots": app_obj.env_config.NUM_SHAPE_SLOTS,
                "num_envs": app_obj.train_config_instance.NUM_SELF_PLAY_WORKERS,
                "plot_data": plot_data,
                "stats_summary": stats_summary,
                "best_game_state_data": best_game_state_data,
                "agent_param_count": app_obj.initializer.agent_param_count,
                "worker_counts": app_obj.worker_manager.get_active_worker_counts(),
                "is_process_running": app_obj.worker_manager.is_any_worker_running(),
                "worker_render_data": worker_render_data_result,
            }
            return data

        logic_app_state.get_render_data = _get_render_data_async.__get__(
            logic_app_state
        )

    except Exception as init_err:
        logger.critical(f"Logic Initialization failed: {init_err}", exc_info=True)
        stop_event.set()
        try:
            render_data_queue.put({ERROR_SENTINEL: f"Logic Init Failed: {init_err}"})
        except Exception:
            pass
        if ray_initialized:
            ray.shutdown()
        return

    # --- Main Logic Loop (Async) ---
    last_render_send_time = 0
    render_send_interval = 1.0 / 30.0

    async def main_loop():
        nonlocal last_render_send_time
        while not stop_event.is_set():
            loop_start = time.monotonic()

            # Process Commands (Synchronous)
            try:
                command_data = command_queue.get_nowait()
                if isinstance(command_data, dict) and COMMAND_SENTINEL in command_data:
                    command = command_data[COMMAND_SENTINEL]
                    logger.info(f"Received command from UI: {command}")
                    if command == STOP_SENTINEL:
                        stop_event.set()
                        break
                    logic_method_name = command_data.get(COMMAND_SENTINEL)
                    logic_method = getattr(
                        logic_app_state.logic, logic_method_name, None
                    )
                    if callable(logic_method):
                        payload = command_data.get(PAYLOAD_KEY)
                        if payload is not None:
                            logic_method(payload)
                        else:
                            logic_method()
                    else:
                        logger.warning(f"Unknown command: {logic_method_name}")
                elif command_data is not None:
                    logger.warning(f"Invalid data on command queue: {command_data}")
            except queue.Empty:
                pass
            except (EOFError, BrokenPipeError):
                logger.warning("Command queue connection lost.")
                stop_event.set()
                break
            except Exception as cmd_err:
                logger.error(f"Error processing command: {cmd_err}", exc_info=True)

            # Update Logic State (Synchronous)
            logic_app_state.logic.update_status_and_check_completion()

            # Send Render Data (Async)
            current_time = time.monotonic()
            if current_time - last_render_send_time > render_send_interval:
                try:
                    render_data = await logic_app_state.get_render_data()
                    render_data_queue.put(
                        {RENDER_DATA_SENTINEL: render_data}, block=False
                    )
                    last_render_send_time = current_time
                except queue.Full:
                    logger.debug("Render data queue full.")
                    last_render_send_time = current_time
                except (EOFError, BrokenPipeError):
                    logger.warning("Render data queue connection lost.")
                    stop_event.set()
                    break
                except Exception as send_err:
                    logger.error(
                        f"Error sending render data: {send_err}", exc_info=True
                    )

            # Loop Timing
            loop_duration = time.monotonic() - loop_start
            sleep_time = max(0, 0.005 - loop_duration)
            await asyncio.sleep(sleep_time)

    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.warning("Logic process received KeyboardInterrupt.")
        stop_event.set()
    except Exception as loop_err:
        logger.critical(f"Critical error in logic main loop: {loop_err}", exc_info=True)
        stop_event.set()

    # --- Shutdown Logic ---
    logger.info("Logic Process shutting down...")
    try:
        if logic_app_state:
            if logic_app_state.worker_manager:
                logic_app_state.worker_manager.stop_all_workers()  # Stops Ray actors
            if logic_app_state.logic:
                logic_app_state.logic.save_final_checkpoint()  # CheckpointManager interacts with actors
            if logic_app_state.initializer:
                logic_app_state.initializer.close_stats_recorder()
            # Terminate other actors if needed (e.g., AgentPredictor, StatsAggregatorActor)
            if logic_app_state.agent_predictor:
                try:
                    ray.kill(logic_app_state.agent_predictor)
                except Exception as e:
                    logger.error(f"Error killing AgentPredictor: {e}")
            if logic_app_state.stats_aggregator and isinstance(
                logic_app_state.stats_aggregator, ray.actor.ActorHandle
            ):
                try:
                    ray.kill(logic_app_state.stats_aggregator)
                except Exception as e:
                    logger.error(f"Error killing StatsAggregatorActor: {e}")
        else:
            logger.warning("logic_app_state not initialized during shutdown sequence.")
    except Exception as shutdown_err:
        logger.error(
            f"Error during logic process shutdown: {shutdown_err}", exc_info=True
        )
    finally:
        try:
            render_data_queue.put(STOP_SENTINEL)
        except Exception as q_err_final:
            logger.warning(f"Could not send final STOP sentinel to UI: {q_err_final}")
        if ray_initialized:
            logger.info("Shutting down Ray...")
            try:
                ray.shutdown()
            except Exception as ray_down_err:
                logger.error(f"Error during Ray shutdown: {ray_down_err}")
        logger.info(
            f"Logic Process finished. Runtime: {time.time() - logic_start_time:.2f}s"
        )
