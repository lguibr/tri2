import sqlite3
import time
import threading
import os
from queue import Queue, Empty, Full
import atexit
from typing import Dict, Any, Optional, List
import numpy as np
from .stats_recorder import StatsRecorderBase, SimpleStatsRecorder
from config import StatsConfig, BufferConfig


class SQLiteLogger(SimpleStatsRecorder):
    """Logs stats to SQLite DB in a worker thread. Inherits console logging."""

    def __init__(
        self,
        db_path: str,
        console_log_interval: int = 1000,
        avg_window: int = 100,
        log_transitions: bool = False,
    ):
        # super().__init__(console_log_interval, avg_window)
        self.db_path = db_path
        # self.log_transitions = log_transitions # Currently unused in provided schema
        self._queue: Queue = Queue(maxsize=20000)  # Increased maxsize
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self._db_error_count = 0
        self._max_db_errors = 10
        self._lock = threading.Lock()

        super().__init__(console_log_interval, avg_window)

        print(f"[SQLiteLogger] Initializing database at: {db_path}")
        self._initialize_db()
        self._start_thread()
        atexit.register(self.close)  # Ensure cleanup on exit

    def _initialize_db(self):
        # No lock needed here, called only during init
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            # Connect with higher timeout, allow access from worker thread
            self._conn = sqlite3.connect(
                self.db_path, timeout=20.0, check_same_thread=False
            )
            self._conn.execute(
                "PRAGMA journal_mode=WAL;"
            )  # Write-Ahead Logging for concurrency
            self._conn.execute(
                "PRAGMA synchronous = NORMAL;"
            )  # Slightly faster than FULL
            self._cursor = self._conn.cursor()

            # Training Log Table (Summarized Step Data)
            self._cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS training_log (
                    log_id INTEGER PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    global_step INTEGER UNIQUE, -- Ensure steps aren't duplicated
                    avg_loss_100 REAL,
                    avg_grad_100 REAL,
                    epsilon REAL,
                    beta REAL,
                    avg_step_reward_1k REAL,
                    buffer_size INTEGER,
                    steps_per_second REAL
                )
            """
            )
            # Episode Log Table
            self._cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_log (
                    log_id INTEGER PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    episode_num INTEGER UNIQUE, -- Prevent duplicate episodes
                    score REAL,
                    length INTEGER,
                    global_step INTEGER,
                    best_score_so_far REAL -- <<< NEW >>> Track best score progression
                )
            """
            )
            # Index important columns for faster queries if needed later
            self._cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_step ON training_log (global_step);"
            )
            self._cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_episode_num ON episode_log (episode_num);"
            )

            self._conn.commit()
            print("[SQLiteLogger] Database initialized successfully.")
        except sqlite3.Error as e:
            print(f"[SQLiteLogger] FATAL Error initializing database: {e}")
            if self._conn:
                self._conn.close()  # Close connection if error occurred after open
            self._conn, self._cursor = None, None
            self._db_error_count = self._max_db_errors  # Prevent worker start

    def _start_thread(self):
        # No lock needed for check, but start the thread
        if self._conn is None or self._db_error_count >= self._max_db_errors:
            print("[SQLiteLogger] Cannot start worker thread: DB unavailable.")
            return
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker, name="SQLiteLoggerThread", daemon=True
            )
            self._thread.start()
            print("[SQLiteLogger] Worker thread started.")

    def _worker(self):
        """Worker thread function to process queue and write to DB."""
        batch = []
        last_commit_time = time.time()
        commit_interval = 2.0  # Commit less frequently for better performance
        items_per_commit = 200

        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)  # Longer timeout
                batch.append(item)
                self._queue.task_done()
            except Empty:
                # Commit if interval passed or stop event set (and batch exists)
                if batch and (
                    time.time() - last_commit_time > commit_interval
                    or self._stop_event.is_set()
                ):
                    pass  # Proceed to commit logic below
                else:
                    continue  # No data and not time to commit yet

            # Commit batch if large enough, interval passed, or stopping
            if batch and (
                len(batch) >= items_per_commit
                or time.time() - last_commit_time > commit_interval
                or self._stop_event.is_set()
            ):

                with self._lock:
                    if self._conn is None or self._cursor is None:
                        continue  # Check connection again inside lock

                    try:
                        # Use executemany for potential performance improvement if structure allows
                        # For mixed types, loop through batch is fine
                        for log_type, data in batch:
                            timestamp = time.time()  # Log time of commit batch start
                            if log_type == "episode":
                                self._cursor.execute(
                                    "INSERT OR IGNORE INTO episode_log (timestamp, episode_num, score, length, global_step, best_score_so_far) VALUES (?, ?, ?, ?, ?, ?)",
                                    (
                                        timestamp,
                                        data["episode_num"],
                                        data["score"],
                                        data["length"],
                                        data["global_step"],
                                        data["best_score"],
                                    ),
                                )
                            elif log_type == "step_summary":
                                # <<< MODIFIED >>> Use summary stats names, add SPS
                                self._cursor.execute(
                                    "INSERT OR IGNORE INTO training_log (timestamp, global_step, avg_loss_100, avg_grad_100, epsilon, beta, avg_step_reward_1k, buffer_size, steps_per_second) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        timestamp,
                                        data["global_step"],
                                        data["avg_loss_100"],
                                        data["avg_grad_100"],
                                        data["epsilon"],
                                        data["beta"],
                                        data["avg_step_reward_1k"],
                                        data["buffer_size"],
                                        data["steps_per_second"],
                                    ),
                                )
                            # elif log_type == "transition" and self.log_transitions: ...

                        self._conn.commit()
                        batch = []  # Clear batch
                        last_commit_time = time.time()

                    except sqlite3.Error as e:
                        self._db_error_count += 1
                        print(
                            f"[SQLiteLogger Worker] Error writing batch (Count: {self._db_error_count}/{self._max_db_errors}): {e}"
                        )
                        # Consider rollback? For simplicity, just discard batch on error.
                        batch = []
                        if self._db_error_count >= self._max_db_errors:
                            print(
                                "[SQLiteLogger Worker] MAX DB ERRORS REACHED. STOPPING LOGGING."
                            )
                            self._stop_event.set()
                    except Exception as e:  # Catch other unexpected errors
                        print(
                            f"[SQLiteLogger Worker] Unexpected error processing batch: {e}"
                        )
                        import traceback

                        traceback.print_exc()
                        batch = []  # Discard batch

        # Final commit check after loop ends
        if batch:
            with self._lock:
                if self._conn and self._cursor:
                    try:
                        # Duplicate commit logic for final items
                        for log_type, data in batch:
                            timestamp = time.time()
                            if log_type == "episode":
                                self._cursor.execute(
                                    "INSERT OR IGNORE INTO episode_log (timestamp, episode_num, score, length, global_step, best_score_so_far) VALUES (?, ?, ?, ?, ?, ?)",
                                    (
                                        timestamp,
                                        data["episode_num"],
                                        data["score"],
                                        data["length"],
                                        data["global_step"],
                                        data["best_score"],
                                    ),
                                )
                            elif log_type == "step_summary":
                                self._cursor.execute(
                                    "INSERT OR IGNORE INTO training_log (timestamp, global_step, avg_loss_100, avg_grad_100, epsilon, beta, avg_step_reward_1k, buffer_size, steps_per_second) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        timestamp,
                                        data["global_step"],
                                        data["avg_loss_100"],
                                        data["avg_grad_100"],
                                        data["epsilon"],
                                        data["beta"],
                                        data["avg_step_reward_1k"],
                                        data["buffer_size"],
                                        data["steps_per_second"],
                                    ),
                                )
                        self._conn.commit()
                        print(
                            f"[SQLiteLogger Worker] Committed final batch of {len(batch)} items."
                        )
                    except sqlite3.Error as e:
                        print(f"[SQLiteLogger Worker] Error during final commit: {e}")
                    except Exception as e:
                        print(
                            f"[SQLiteLogger Worker] Unexpected error during final commit: {e}"
                        )

        print("[SQLiteLogger] Worker thread finished.")

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
    ):
        # Call super to update in-memory stats (including best_score)
        super().record_episode(episode_score, episode_length, episode_num, global_step)
        # Queue data for DB
        if not self._stop_event.is_set():
            try:
                db_data = {
                    "episode_num": episode_num,
                    "score": episode_score,
                    "length": episode_length,
                    "global_step": global_step,
                    "best_score": self.best_score,  # Get best score tracked by SimpleStatsRecorder
                }
                self._queue.put_nowait(("episode", db_data))
            except Full:
                print("[SQLiteLogger] Warning: Log queue full. Discarding episode log.")
            except Exception as e:
                print(f"[SQLiteLogger] Error queueing episode log: {e}")

    def record_step(self, step_data: Dict[str, Any]):
        # Record in memory immediately for console summary
        super().record_step(step_data)
        # DB logging of step summary is now handled within log_summary

    def log_summary(self, global_step: int):
        # Call superclass to calculate stats and print to console if interval passed
        should_log_console = (
            global_step > 0
            and global_step - self.last_log_step >= self.console_log_interval
        )

        if should_log_console:
            # Let superclass calculate summary and print
            super().log_summary(global_step)

            # Now queue the *same summary* that was just printed (or calculated) for DB logging
            if not self._stop_event.is_set():
                summary_for_db = (
                    self.get_summary()
                )  # Get the freshly calculated summary
                try:
                    self._queue.put_nowait(("step_summary", summary_for_db))
                except Full:
                    print(
                        "[SQLiteLogger] Warning: Log queue full. Discarding step summary log."
                    )
                except Exception as e:
                    print(f"[SQLiteLogger] Error queueing step summary log: {e}")

    def close(self):
        # Prevent double closing
        if hasattr(self, "_closed") and self._closed:
            return
        self._closed = True  # Mark as closed

        print("[SQLiteLogger] Closing logger...")
        self._stop_event.set()

        # Wait for queue processing (with timeout)
        # self._queue.join() # Can hang if worker dies unexpectedly

        if self._thread and self._thread.is_alive():
            print("[SQLiteLogger] Joining worker thread...")
            self._thread.join(timeout=10.0)  # Increased timeout
            if self._thread.is_alive():
                print("[SQLiteLogger] Warning: Worker thread join timed out.")

        # Close DB connection safely
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                    print("[SQLiteLogger] Database connection closed.")
                except sqlite3.Error as e:
                    print(f"[SQLiteLogger] Error closing database connection: {e}")
            self._conn = None
            self._cursor = None

        # Call superclass close if necessary (currently no-op)
        super().close()
        print("[SQLiteLogger] Close complete.")
