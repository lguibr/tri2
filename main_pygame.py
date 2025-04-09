# File: main_pygame.py
import sys
import time
import threading
import logging
import logging.handlers
import argparse
import os
import traceback
import multiprocessing as mp
from typing import Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from config import BASE_CHECKPOINT_DIR, set_run_id, get_run_id, get_run_log_dir
    from training.checkpoint_manager import find_latest_run_and_checkpoint
    from logger import TeeLogger
    from ui_process import run_ui_process
    from logic_process import run_logic_process
except ImportError as e:
    print(f"Error importing core modules/functions: {e}\n{traceback.format_exc()}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s|%(processName)s] %(message)s"
)
logger = logging.getLogger(__name__)

tee_logger_instance: Optional[TeeLogger] = None
log_listener_thread: Optional[threading.Thread] = None


# --- Logging Setup Functions (remain the same) ---
def setup_logging_queue_listener(log_queue: mp.Queue):
    global log_listener_thread

    def listener_process():
        listener_logger = logging.getLogger("LogListener")
        listener_logger.info("Log listener started.")
        while True:
            try:
                record = log_queue.get()
                if record is None:
                    break
                logger_handler = logging.getLogger(record.name)
                logger_handler.handle(record)
            except (EOFError, OSError):
                listener_logger.warning("Log queue closed or broken pipe.")
                break
            except Exception as e:
                print(f"Log listener error: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        listener_logger.info("Log listener stopped.")

    log_listener_thread = threading.Thread(
        target=listener_process, daemon=True, name="LogListener"
    )
    log_listener_thread.start()
    return log_listener_thread


def setup_logging_and_run_id(args: argparse.Namespace):
    global tee_logger_instance
    run_id_source = "New"
    if args.load_checkpoint:
        try:
            run_id_from_path = os.path.basename(os.path.dirname(args.load_checkpoint))
            if run_id_from_path.startswith("run_"):
                set_run_id(run_id_from_path)
                run_id_source = f"Explicit Checkpoint ({get_run_id()})"
            else:
                get_run_id()
                run_id_source = (
                    f"New (Explicit Ckpt Path Invalid: {args.load_checkpoint})"
                )
        except Exception as e:
            logger.warning(
                f"Could not determine run_id from checkpoint path '{args.load_checkpoint}': {e}. Generating new."
            )
            get_run_id()
            run_id_source = f"New (Error parsing ckpt path)"
    else:
        latest_run_id, _ = find_latest_run_and_checkpoint(BASE_CHECKPOINT_DIR)
        if latest_run_id:
            set_run_id(latest_run_id)
            run_id_source = f"Resumed Latest ({get_run_id()})"
        else:
            get_run_id()
            run_id_source = f"New (No previous runs found)"
    current_run_id = get_run_id()
    print(f"Run ID: {current_run_id} (Source: {run_id_source})")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        log_file_dir = get_run_log_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, "console_output.log")
        tee_logger_instance = TeeLogger(log_file_path, sys.stdout)
        sys.stdout = tee_logger_instance
        sys.stderr = tee_logger_instance
        print(f"Main process console output will be mirrored to: {log_file_path}")
    except Exception as e:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.error(f"Error setting up TeeLogger: {e}", exc_info=True)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logger.info(f"Main process logging level set to: {args.log_level.upper()}")
    logger.info(f"Using Run ID: {current_run_id}")
    if args.load_checkpoint:
        logger.info(f"Attempting to load checkpoint: {args.load_checkpoint}")
    return original_stdout, original_stderr


def cleanup_logging(
    original_stdout, original_stderr, log_queue: Optional[mp.Queue], exit_code
):
    print("[Main Finally] Restoring stdout/stderr and closing logger...")
    if log_queue:
        try:
            log_queue.put(None)
            log_queue.close()
            log_queue.join_thread()
        except Exception as qe:
            print(f"Error closing log queue: {qe}", file=original_stderr)
    if log_listener_thread:
        try:
            log_listener_thread.join(timeout=2.0)
            if log_listener_thread.is_alive():
                print(
                    "Warning: Log listener thread did not join cleanly.",
                    file=original_stderr,
                )
        except Exception as le:
            print(f"Error joining log listener thread: {le}", file=original_stderr)
    if tee_logger_instance:
        try:
            if isinstance(sys.stdout, TeeLogger):
                sys.stdout.flush()
            if isinstance(sys.stderr, TeeLogger):
                sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            tee_logger_instance.close()
            print("[Main Finally] TeeLogger closed and streams restored.")
        except Exception as log_close_err:
            original_stdout.write(f"ERROR closing TeeLogger: {log_close_err}\n")
            traceback.print_exc(file=original_stderr)
    print(f"[Main Finally] Exiting with code {exit_code}.")
    sys.exit(exit_code)


# =========================================================================
# Main Execution Block
# =========================================================================
if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Could not set start method to 'spawn', using default.")

    parser = argparse.ArgumentParser(description="AlphaZero Trainer - Multiprocess")
    parser.add_argument(
        "--load-checkpoint", type=str, default=None, help="Path to checkpoint."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    args = parser.parse_args()

    original_stdout, original_stderr = setup_logging_and_run_id(args)

    ui_to_logic_queue = mp.Queue()
    logic_to_ui_queue = mp.Queue(maxsize=10)
    stop_event = mp.Event()
    log_queue = mp.Queue()
    log_listener = setup_logging_queue_listener(log_queue)

    # Set daemon=True for simpler exit handling, rely on stop_event and timeouts
    ui_process = mp.Process(
        target=run_ui_process,
        args=(stop_event, ui_to_logic_queue, logic_to_ui_queue, log_queue),
        name="UIProcess",
        daemon=True,
    )
    logic_process = mp.Process(
        target=run_logic_process,
        args=(
            stop_event,
            ui_to_logic_queue,
            logic_to_ui_queue,
            args.load_checkpoint,
            log_queue,
        ),
        name="LogicProcess",
        daemon=True,
    )

    exit_code = 0
    try:
        logger.info("Starting UI process...")
        ui_process.start()
        logger.info("Starting Logic process...")
        logic_process.start()

        # --- Wait for Processes ---
        while True:  # Loop indefinitely until stop_event or error
            if stop_event.is_set():
                logger.info("Stop event detected by main process. Exiting wait loop.")
                break
            if not logic_process.is_alive():
                logger.warning("Logic process terminated unexpectedly. Signaling stop.")
                stop_event.set()
                exit_code = 1  # Indicate error
                break
            if not ui_process.is_alive():
                logger.warning("UI process terminated unexpectedly. Signaling stop.")
                stop_event.set()
                exit_code = 1  # Indicate error
                break
            try:
                # Sleep briefly to prevent busy-waiting
                time.sleep(0.2)
            except KeyboardInterrupt:
                logger.warning(
                    "Main process received KeyboardInterrupt. Signaling stop..."
                )
                stop_event.set()
                exit_code = 130
                break  # Exit the waiting loop

    except Exception as main_err:
        logger.critical(
            f"Error in main process coordination: {main_err}", exc_info=True
        )
        stop_event.set()
        exit_code = 1

    finally:
        logger.info("Main process initiating cleanup...")
        if not stop_event.is_set():
            stop_event.set()  # Ensure stop is signaled

        time.sleep(0.5)  # Allow processes to potentially react

        # --- Join Processes with Timeouts ---
        join_timeout_logic = 10.0  # More time for logic to save
        join_timeout_ui = 3.0

        logger.info(
            f"Waiting for Logic process to join (timeout: {join_timeout_logic}s)..."
        )
        if logic_process.is_alive():
            logic_process.join(timeout=join_timeout_logic)
        if logic_process.is_alive():
            logger.warning("Logic process did not join cleanly. Terminating.")
            try:
                logic_process.terminate()
                logic_process.join(1.0)
            except Exception as term_err:
                logger.error(f"Error terminating Logic process: {term_err}")

        logger.info(f"Waiting for UI process to join (timeout: {join_timeout_ui}s)...")
        if ui_process.is_alive():
            ui_process.join(timeout=join_timeout_ui)
        if ui_process.is_alive():
            logger.warning("UI process did not join cleanly. Terminating.")
            try:
                ui_process.terminate()
                ui_process.join(1.0)
            except Exception as term_err:
                logger.error(f"Error terminating UI process: {term_err}")

        logger.info("Processes joined or terminated.")
        cleanup_logging(original_stdout, original_stderr, log_queue, exit_code)
