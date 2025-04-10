# File: src/config/persistence_config.py
from typing import Optional
import os
from pydantic import BaseModel, Field, computed_field

class PersistenceConfig(BaseModel):
    """Configuration for saving/loading artifacts (Pydantic model)."""

    ROOT_DATA_DIR: str = Field(".alphatriangle_data")
    RUNS_DIR_NAME: str = Field("runs")
    MLFLOW_DIR_NAME: str = Field("mlruns")

    CHECKPOINT_SAVE_DIR_NAME: str = Field("checkpoints")
    BUFFER_SAVE_DIR_NAME: str = Field("buffers")
    GAME_STATE_SAVE_DIR_NAME: str = Field("game_states")
    LOG_DIR_NAME: str = Field("logs")

    LATEST_CHECKPOINT_FILENAME: str = Field("latest.pkl")
    BEST_CHECKPOINT_FILENAME: str = Field("best.pkl")
    BUFFER_FILENAME: str = Field("buffer.pkl")
    CONFIG_FILENAME: str = Field("configs.json")

    # RUN_NAME is typically set dynamically by TrainConfig, but provide a default
    RUN_NAME: str = Field("default_run")

    SAVE_GAME_STATES: bool = Field(False)
    GAME_STATE_SAVE_FREQ_EPISODES: int = Field(5, ge=1)

    SAVE_BUFFER: bool = Field(True)
    BUFFER_SAVE_FREQ_STEPS: int = Field(10, ge=1)

    @computed_field # type: ignore[misc]
    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        """Constructs the file URI for MLflow tracking."""
        # Ensure the path is absolute for MLflow
        abs_path = os.path.abspath(os.path.join(self.ROOT_DATA_DIR, self.MLFLOW_DIR_NAME))
        # Replace backslashes with forward slashes for file URI compatibility on Windows
        uri_path = abs_path.replace(os.sep, '/')
        # Ensure the URI starts with file:///
        if not uri_path.startswith('/'):
            uri_path = '/' + uri_path
        return f"file://{uri_path}"

    def get_run_base_dir(self, run_name: Optional[str] = None) -> str:
        """Gets the base directory for a specific run."""
        name = run_name if run_name else self.RUN_NAME
        return os.path.join(self.ROOT_DATA_DIR, self.RUNS_DIR_NAME, name)