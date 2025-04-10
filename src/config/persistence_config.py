from typing import Optional
import os
from pathlib import Path  # Import pathlib
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

    @computed_field  # type: ignore[misc]
    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        """Constructs the file URI for MLflow tracking using pathlib."""
        # Get the absolute path using pathlib
        abs_path = Path(self.ROOT_DATA_DIR).joinpath(self.MLFLOW_DIR_NAME).resolve()
        # Convert Path object to a file URI (handles drive letters etc.)
        return abs_path.as_uri()

    def get_run_base_dir(self, run_name: Optional[str] = None) -> str:
        """Gets the base directory for a specific run."""
        name = run_name if run_name else self.RUN_NAME
        # Use pathlib here too for consistency, return as string
        return str(Path(self.ROOT_DATA_DIR).joinpath(self.RUNS_DIR_NAME, name))

    def get_mlflow_abs_path(self) -> str:
        """Gets the absolute OS path to the MLflow directory as a string."""
        # Use pathlib to resolve the absolute path and return as string
        abs_path = Path(self.ROOT_DATA_DIR).joinpath(self.MLFLOW_DIR_NAME).resolve()
        return str(abs_path)
