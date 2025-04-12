# File: src/config/train_config.py
import time
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class TrainConfig(BaseModel):
    """Configuration for the training process (Pydantic model)."""

    RUN_NAME: str = Field(
        default_factory=lambda: f"train_run_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    LOAD_CHECKPOINT_PATH: Optional[str] = Field(None)
    LOAD_BUFFER_PATH: Optional[str] = Field(None)
    AUTO_RESUME_LATEST: bool = Field(True)
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(
        "auto"
    )  # Default to auto (prioritizes CUDA)
    RANDOM_SEED: int = Field(42)

    # --- Training Loop ---
    # CHANGE: Increased for serious run
    MAX_TRAINING_STEPS: Optional[int] = Field(default=500_000, ge=1)

    # --- Workers & Batching ---
    # CHANGE: Increased for serious run
    NUM_SELF_PLAY_WORKERS: int = Field(12, ge=1)
    WORKER_DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(
        "cpu"
    )  # Default workers to CPU
    # CHANGE: Increased for serious run
    BATCH_SIZE: int = Field(256, ge=1)
    # CHANGE: Increased for serious run
    BUFFER_CAPACITY: int = Field(100_000, ge=1)
    # CHANGE: Increased for serious run
    MIN_BUFFER_SIZE_TO_TRAIN: int = Field(20_000, ge=1)
    WORKER_UPDATE_FREQ_STEPS: int = Field(10, ge=1)  # Keep relatively frequent updates

    # --- Optimizer ---
    OPTIMIZER_TYPE: Literal["Adam", "AdamW", "SGD"] = Field("AdamW")
    LEARNING_RATE: float = Field(1e-4, gt=0)  # Keep default, adjust based on results
    WEIGHT_DECAY: float = Field(1e-5, ge=0)
    GRADIENT_CLIP_VALUE: Optional[float] = Field(default=1.0)  # Allow None or positive

    # --- LR Scheduler ---
    LR_SCHEDULER_TYPE: Optional[Literal["StepLR", "CosineAnnealingLR"]] = Field(
        default="CosineAnnealingLR"
    )
    LR_SCHEDULER_T_MAX: Optional[int] = Field(
        default=None
    )  # Will be set based on MAX_TRAINING_STEPS
    LR_SCHEDULER_ETA_MIN: float = Field(1e-6, ge=0)

    # --- Loss Weights ---
    POLICY_LOSS_WEIGHT: float = Field(1.0, ge=0)
    VALUE_LOSS_WEIGHT: float = Field(1.0, ge=0)
    ENTROPY_BONUS_WEIGHT: float = Field(0.01, ge=0)

    # --- Checkpointing ---
    # CHANGE: Increased for serious run
    CHECKPOINT_SAVE_FREQ_STEPS: int = Field(1000, ge=1)

    # --- Prioritized Experience Replay (PER) ---
    USE_PER: bool = Field(True)  # Enable/disable PER
    PER_ALPHA: float = Field(
        0.6, ge=0
    )  # Priority exponent (0=uniform, 1=full priority)
    PER_BETA_INITIAL: float = Field(
        0.4, ge=0, le=1.0
    )  # Initial importance sampling exponent
    PER_BETA_FINAL: float = Field(
        1.0, ge=0, le=1.0
    )  # Final importance sampling exponent
    PER_BETA_ANNEAL_STEPS: Optional[int] = Field(
        None
    )  # Steps to anneal beta (None=MAX_TRAINING_STEPS)
    PER_EPSILON: float = Field(
        1e-5, gt=0
    )  # Small value added to priorities to ensure non-zero probability

    @model_validator(mode="after")
    def check_buffer_sizes(self) -> "TrainConfig":
        if self.MIN_BUFFER_SIZE_TO_TRAIN > self.BUFFER_CAPACITY:
            raise ValueError(
                "MIN_BUFFER_SIZE_TO_TRAIN cannot be greater than BUFFER_CAPACITY."
            )
        if self.BATCH_SIZE > self.BUFFER_CAPACITY:
            raise ValueError("BATCH_SIZE cannot be greater than BUFFER_CAPACITY.")
        if self.BATCH_SIZE > self.MIN_BUFFER_SIZE_TO_TRAIN:
            print(
                f"Warning: BATCH_SIZE ({self.BATCH_SIZE}) is larger than MIN_BUFFER_SIZE_TO_TRAIN ({self.MIN_BUFFER_SIZE_TO_TRAIN}). This might lead to slow startup."
            )
        return self

    @model_validator(mode="after")
    def set_scheduler_t_max(self) -> "TrainConfig":
        if (
            self.LR_SCHEDULER_TYPE == "CosineAnnealingLR"
            and self.LR_SCHEDULER_T_MAX is None
        ):
            if self.MAX_TRAINING_STEPS is not None:
                self.LR_SCHEDULER_T_MAX = self.MAX_TRAINING_STEPS
            else:
                # CHANGE: Match increased default MAX_TRAINING_STEPS
                self.LR_SCHEDULER_T_MAX = 500_000
                print(
                    f"Warning: MAX_TRAINING_STEPS is None, setting LR_SCHEDULER_T_MAX to default {self.LR_SCHEDULER_T_MAX}"
                )
        if self.LR_SCHEDULER_T_MAX is not None and self.LR_SCHEDULER_T_MAX <= 0:
            raise ValueError("LR_SCHEDULER_T_MAX must be positive if set.")
        return self

    @model_validator(mode="after")
    def set_per_beta_anneal_steps(self) -> "TrainConfig":
        if self.USE_PER and self.PER_BETA_ANNEAL_STEPS is None:
            if self.MAX_TRAINING_STEPS is not None:
                self.PER_BETA_ANNEAL_STEPS = self.MAX_TRAINING_STEPS
            else:
                # CHANGE: Match increased default MAX_TRAINING_STEPS
                self.PER_BETA_ANNEAL_STEPS = 500_000
                print(
                    f"Warning: MAX_TRAINING_STEPS is None, setting PER_BETA_ANNEAL_STEPS to default {self.PER_BETA_ANNEAL_STEPS}"
                )
        if self.PER_BETA_ANNEAL_STEPS is not None and self.PER_BETA_ANNEAL_STEPS <= 0:
            raise ValueError("PER_BETA_ANNEAL_STEPS must be positive if set.")
        return self

    @field_validator("GRADIENT_CLIP_VALUE")
    @classmethod
    def check_gradient_clip(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("GRADIENT_CLIP_VALUE must be positive if set.")
        return v

    @field_validator("PER_BETA_FINAL")
    @classmethod
    def check_per_beta_final(cls, v: float, info) -> float:
        initial_beta = info.data.get("PER_BETA_INITIAL")
        if initial_beta is not None and v < initial_beta:
            raise ValueError("PER_BETA_FINAL cannot be less than PER_BETA_INITIAL")
        return v
