# File: src/config/env_config.py
from typing import List, Tuple
from pydantic import BaseModel, Field, computed_field, field_validator

class EnvConfig(BaseModel):
    """Configuration for the game environment (Pydantic model)."""
    ROWS: int = Field(8, gt=0)
    COLS: int = Field(15, gt=0)
    NUM_SHAPE_SLOTS: int = Field(3, gt=0)
    MIN_LINE_LENGTH: int = Field(3, gt=0)

    # Example hourglass shape definition for the grid
    COLS_PER_ROW: List[int] = Field(default=[9, 11, 13, 15, 15, 13, 11, 9])

    @field_validator('COLS_PER_ROW')
    @classmethod
    def check_cols_per_row(cls, v: List[int], info) -> List[int]:
        rows = info.data.get('ROWS')
        cols = info.data.get('COLS')
        if rows is None or cols is None:
            # This can happen during initial validation before all fields are set
            # Pydantic usually handles this, but we add a check.
            # Alternatively, make ROWS/COLS positional args in __init__ if needed earlier.
            return v # Skip validation if ROWS/COLS not available yet

        if len(v) != rows:
            raise ValueError(f"COLS_PER_ROW length ({len(v)}) must equal ROWS ({rows})")
        if max(v, default=0) > cols:
            raise ValueError(f"Max COLS_PER_ROW ({max(v, default=0)}) cannot exceed COLS ({cols})")
        return v

    @computed_field # type: ignore[misc]
    @property
    def ACTION_DIM(self) -> int:
        """Total number of possible actions (shape_slot * row * col)."""
        return self.NUM_SHAPE_SLOTS * self.ROWS * self.COLS