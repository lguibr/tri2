# File: stats/sqlite_logger.py
import sqlite3
import threading
import queue
import time
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# Import the base class
from .stats_recorder import StatsRecorderBase, SimpleStatsRecorder
# Import config if needed for table definitions etc. (not strictly needed for fix)
# from config import StatsConfig


# --- Constants for Database ---
TABLE_EPISODES = "episodes"
TABLE_STEPS = "steps"
TABLE_TRANSITIONS = "transitions" # Only used if LOG_TRANSITIONS_TO_DB is True

# Define schema using lists of (column_name, sql_type) tuples
SCHEMA_EPISODES = [
    ("episode_num", "INTEGER PRIMARY KEY"),
    ("global_step", "INTEGER"),
    ("score", "REAL"),
    ("length", "INTEGER"),
    ("timestamp", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
]

SCHEMA_STEPS = [
    ("global_step", "INTEGER PRIMARY KEY"),
    ("timestamp", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ("sps", "REAL"), # Steps per second
    ("avg_loss", "REAL"),
    ("avg_grad", "REAL"),
    ("avg_max_q", "REAL"),
    ("avg_score_100", "REAL"),
    ("avg_length_100", "REAL"),
    ("avg_step_reward_1k", "REAL"),
    ("epsilon", "REAL"),
    ("beta", "REAL"),
    ("buffer_size", "INTEGER"),
    ("total_episodes", "INTEGER"), # Store total episode count at this step
]

SCHEMA_TRANSITIONS = [
    ("log_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("global_step", "INTEGER"),
    ("env_id", "INTEGER"), # Optional: Track which env generated transition
    # Store state/next_state as BLOB (pickled numpy array) or JSON string? BLOB is simpler.
    ("state", "BLOB"),
    ("action", "INTEGER"),
    ("reward", "REAL"),
    ("next_state", "BLOB"),
    ("done", "INTEGER"), # Store boolean as 0 or 1
    ("timestamp", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
]


class SQLiteLogger(SimpleStatsRecorder):
    """
    Logs training statistics to an SQLite database using a background thread.
    Inherits from SimpleStatsRecorder to reuse its in-memory averaging logic.
    """

    def __init__(
        self,
        db_path: str,
        console_log_interval: int = 1000,
        avg_window: int = 100,
        log_transitions: bool = False, # Flag to enable detailed transition logging
        commit_interval: float = 5.0, # Seconds between DB commits
    ):
        # Initialize the parent class (SimpleStatsRecorder) for in-memory tracking
        super().__init__(console_log_interval, avg_window)

        self.db_path = db_path
        self.log_transitions = log_transitions
        self.commit_interval = commit_interval
        self._conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self._log_queue: queue.Queue = queue.Queue(maxsize=10000) # Buffer DB operations
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._last_commit_time = time.time()

        print(f"[SQLiteLogger] Initializing database at: {self.db_path}")
        self._initialize_db()
        self._start_worker()
        print("[SQLiteLogger] Database initialized successfully.")


    def _initialize_db(self):
        """Connects to the DB and creates tables if they don't exist."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            # Connect to the database (will create if doesn't exist)
            # Use WAL mode for better concurrency (optional but good practice)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0) # Allow access from worker thread
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._cursor = self._conn.cursor()

            # --- Create Tables ---
            self._create_table_if_not_exists(TABLE_EPISODES, SCHEMA_EPISODES)
            self._create_table_if_not_exists(TABLE_STEPS, SCHEMA_STEPS)
            if self.log_transitions:
                print("[SQLiteLogger] Transition logging ENABLED.")
                self._create_table_if_not_exists(TABLE_TRANSITIONS, SCHEMA_TRANSITIONS)

            self._conn.commit() # Commit table creation
        except sqlite3.Error as e:
            print(f"[SQLiteLogger] CRITICAL DB Error during initialization: {e}")
            # Should we raise an exception here or allow fallback?
            # For now, log error; operations will likely fail later.
            self._conn = None # Prevent further operations if init failed
            self._cursor = None
            raise e # Re-raise to indicate critical failure

    def _create_table_if_not_exists(self, table_name: str, schema: List[Tuple[str, str]]):
        """Creates a table with the given schema if it doesn't exist."""
        if not self._cursor: return
        try:
            columns_sql = ", ".join([f'"{name}" {type}' for name, type in schema])
            create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql})'
            self._cursor.execute(create_sql)
            # Optional: Add indices for faster querying later
            if table_name == TABLE_EPISODES and "global_step" in [s[0] for s in schema]:
                 self._cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_step ON "{table_name}" (global_step)')
            if table_name == TABLE_STEPS and "global_step" in [s[0] for s in schema]:
                 # global_step is PK, index is automatic
                 pass
            print(f"[SQLiteLogger] Table '{table_name}' checked/created.")
        except sqlite3.Error as e:
            print(f"[SQLiteLogger] DB Error creating table {table_name}: {e}")
            raise e # Propagate error

    def _db_worker(self):
        """Background thread function to process the log queue."""
        print("[SQLiteLogger] Worker thread started.")
        while not self._stop_event.is_set() or not self._log_queue.empty():
            items_processed = 0
            try:
                # Process items in batches for efficiency
                batch = []
                while not self._log_queue.empty() and items_processed < 200: # Process up to 200 items at once
                    try:
                         item = self._log_queue.get_nowait()
                         batch.append(item)
                         items_processed += 1
                    except queue.Empty:
                         break # Should not happen due to outer check, but be safe

                if batch and self._conn and self._cursor:
                    try:
                        for command, params in batch:
                             self._cursor.execute(command, params)
                        # Commit periodically, not after every item/batch
                        now = time.time()
                        if now - self._last_commit_time > self.commit_interval:
                            self._conn.commit()
                            self._last_commit_time = now
                            # print(f"[SQLiteLogger Worker] Committed {items_processed} items.") # Debug
                    except sqlite3.Error as e:
                        print(f"[SQLiteLogger Worker] DB Error executing batch: {e}")
                        # Consider adding failed items back to queue or logging them
                    except Exception as e: # Catch other potential errors
                         print(f"[SQLiteLogger Worker] Non-DB Error processing batch: {e}")

                # If queue is empty, wait a bit before checking again
                if items_processed == 0:
                    time.sleep(0.1) # Wait 100ms if queue was empty

            except Exception as e:
                print(f"[SQLiteLogger Worker] Unexpected error in worker loop: {e}")
                time.sleep(1) # Avoid busy-looping on unexpected errors

        # --- Final Commit on Exit ---
        if self._conn:
            try:
                print("[SQLiteLogger Worker] Performing final commit...")
                self._conn.commit()
                print("[SQLiteLogger Worker] Final commit successful.")
            except sqlite3.Error as e:
                print(f"[SQLiteLogger Worker] DB Error during final commit: {e}")
        print("[SQLiteLogger] Worker thread stopped.")


    def _start_worker(self):
        """Starts the background database worker thread."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._db_worker, daemon=True)
            self._worker_thread.start()

    def _stop_worker(self):
        """Signals the worker thread to stop and waits for it."""
        if self._worker_thread and self._worker_thread.is_alive():
            print("[SQLiteLogger] Stopping worker thread...")
            self._stop_event.set()
            # Wait for the thread to finish (with a timeout)
            self._worker_thread.join(timeout=self.commit_interval + 2.0)
            if self._worker_thread.is_alive():
                print("[SQLiteLogger] Warning: Worker thread did not stop gracefully.")
            else:
                print("[SQLiteLogger] Worker thread joined.")
        self._worker_thread = None

    def _queue_log(self, command: str, params: Tuple = ()):
        """Adds a SQL command and parameters to the queue."""
        try:
            self._log_queue.put_nowait((command, params))
        except queue.Full:
            print("[SQLiteLogger] Warning: Log queue is full. Discarding oldest log entry.")
            # Discard oldest item to make space
            try:
                self._log_queue.get_nowait()
                self._log_queue.put_nowait((command, params)) # Retry putting
            except queue.Empty:
                pass # Should not happen if queue was full
            except queue.Full: # Still full after dropping? Log error.
                 print("[SQLiteLogger] CRITICAL: Log queue full even after dropping. Logging may be lost.")


    # --- Overridden Methods ---

    def record_episode(
        self,
        episode_score: float,
        episode_length: int,
        episode_num: int,
        global_step: Optional[int] = None,
    ):
        """Records episode stats in memory and queues DB insert."""
        # Call parent first to update in-memory deques and best score logic
        super().record_episode(episode_score, episode_length, episode_num, global_step)

        # Queue database insert command
        cols = [s[0] for s in SCHEMA_EPISODES if s[0] != 'timestamp'] # Exclude default timestamp
        placeholders = ", ".join(["?"] * len(cols))
        sql = f'INSERT INTO "{TABLE_EPISODES}" ({", ".join(f"\"{c}\"" for c in cols)}) VALUES ({placeholders})'
        # Ensure global_step is provided, default to 0 if not
        gs = global_step if global_step is not None else 0
        params = (episode_num, gs, episode_score, episode_length)
        self._queue_log(sql, params)


    def log_summary(self, global_step: int):
        """Logs summary to console (via parent) and queues DB insert for step data."""
        # Only log to console and DB based on interval
        if (global_step == 0 or
            global_step < self.last_log_step + self.console_log_interval):
            return # Not time to log yet

        # --- Console Logging (handled by parent) ---
        # The parent's log_summary calculates SPS and prints.
        # We call it first to trigger the printout.
        super().log_summary(global_step) # This updates self.last_log_step

        # --- Database Logging ---
        # Get the summary dictionary calculated by the parent
        # <<< FIX: Pass global_step to get_summary >>>
        summary = super().get_summary(global_step)

        # Queue the step summary insert/update
        # Use INSERT OR REPLACE to handle potential duplicate steps if logging happens too fast
        # Or handle potential errors in worker? Let's use INSERT OR REPLACE for simplicity.
        cols = [s[0] for s in SCHEMA_STEPS if s[0] != 'timestamp']
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT OR REPLACE INTO \"{TABLE_STEPS}\" ({', '.join(f'\"{c}\"' for c in cols)}) VALUES ({placeholders})"

        # Prepare parameters in the correct order according to SCHEMA_STEPS
        params = (
            global_step,
            summary.get("steps_per_second", 0.0),
            summary.get("avg_loss_100", 0.0),
            summary.get("avg_grad_100", 0.0),
            summary.get("avg_max_q_100", 0.0),
            summary.get("avg_score_100", 0.0),
            summary.get("avg_length_100", 0.0),
            summary.get("avg_step_reward_1k", 0.0),
            summary.get("epsilon", 0.0),
            summary.get("beta", 0.0),
            summary.get("buffer_size", 0),
            summary.get("total_episodes", 0),
        )
        self._queue_log(sql, params)

    # --- Transition Logging (Optional) ---
    # This method isn't part of the base class, called explicitly if needed.
    def record_transition(
        self, global_step: int, env_id: int,
        state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Queues a detailed transition log if enabled."""
        if not self.log_transitions:
            return
        if not self._conn: # Don't queue if DB connection failed
             return

        import pickle # Use pickle for numpy arrays

        try:
            # Serialize numpy arrays to BLOBs using pickle
            state_blob = sqlite3.Binary(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
            next_state_blob = sqlite3.Binary(pickle.dumps(next_state, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as e:
            print(f"Error pickling state/next_state for DB logging: {e}")
            return # Skip logging this transition if serialization fails

        cols = [s[0] for s in SCHEMA_TRANSITIONS if s[0] not in ['log_id', 'timestamp']]
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO \"{TABLE_TRANSITIONS}\" ({', '.join([f'\"{c}\"' for c in cols])}) VALUES ({placeholders})"
        params = (
            global_step, env_id,
            state_blob, action, reward, next_state_blob, int(done) # Convert bool to int
        )
        self._queue_log(sql, params)


    # --- Overridden get_summary (pass argument) ---
    def get_summary(self, current_global_step: int) -> Dict[str, Any]:
        """
        Calls the parent's get_summary method, passing the required argument.
        This ensures SPS is calculated correctly.
        """
        # <<< FIX: Pass the argument to the parent's implementation >>>
        return super().get_summary(current_global_step)


    def close(self):
        """Signals the worker to stop, closes the DB connection."""
        print("[SQLiteLogger] Closing database connection...")
        # Signal worker to stop and wait for it to finish processing queue
        self._stop_worker()

        # Close the database connection
        if self._conn:
            try:
                self._conn.close()
                print("[SQLiteLogger] Database connection closed.")
            except sqlite3.Error as e:
                print(f"[SQLiteLogger] DB Error closing connection: {e}")
        self._conn = None
        self._cursor = None
        # Call parent close (though SimpleStatsRecorder.close does nothing)
        super().close()