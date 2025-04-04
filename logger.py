# File: logger.py
import os
import sys
from typing import TextIO


class TeeLogger:
    """Redirects stdout/stderr to both the console and a log file."""

    def __init__(self, filepath: str, original_stream: TextIO):
        self.terminal = original_stream
        self.log_file: Optional[TextIO] = None
        try:
            log_dir = os.path.dirname(filepath)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            # Use buffering=1 for line buffering
            self.log_file = open(filepath, "w", encoding="utf-8", buffering=1)
            print(f"[TeeLogger] Logging console output to: {filepath}")
        except Exception as e:
            self.terminal.write(
                f"FATAL ERROR: Could not open log file {filepath}: {e}\n"
            )
            # Continue without file logging if opening fails

    def write(self, message: str):
        self.terminal.write(message)
        if self.log_file:
            try:
                self.log_file.write(message)
            except Exception:
                # Silently ignore errors writing to log file to avoid loops
                pass

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass  # Silently ignore errors flushing log file

    def close(self):
        if self.log_file:
            try:
                self.log_file.close()
                self.log_file = None
            except Exception as e:
                self.terminal.write(f"Warning: Error closing log file: {e}\n")

    def __del__(self):
        # Ensure file is closed if logger object is garbage collected
        self.close()
