import time
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

class TrainConfig(BaseModel):
    """Configuration for the training process (Pydantic model)."""

    RUN_NAME: str = Field(default_factory=lambda: f"train_run_{time.strftime('%Y%m%d_%H%M%S')}")
    LOAD_CHECKPOINT_PATH: Optional[str] = Field(None)
    LOAD_BUFFER_PATH: Optional[str] = Field(None)
    AUTO_RESUME_LATEST: bool = Field(True)
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field("auto") # Default to auto (prioritizes CUDA)
    RANDOM_SEED: int = Field(42)

    # --- Training Loop ---
    # Reduced steps for faster runs
    MAX_TRAINING_STEPS: Optional[int] = Field(default=50_000, ge=1)

    # --- Workers & Batching ---
    NUM_SELF_PLAY_WORKERS: int = Field(12, ge=1) # Keep workers, MCTS sims reduced instead
    WORKER_DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field("cpu") # Default workers to CPU
    BATCH_SIZE: int = Field(64, ge=1) # Reduced batch size slightly
    BUFFER_CAPACITY: int = Field(20_000, ge=1) # Reduced buffer capacity significantly
    MIN_BUFFER_SIZE_TO_TRAIN: int = Field(5_000, ge=1) # Reduced min buffer size significantly
    WORKER_UPDATE_FREQ_STEPS: int = Field(10, ge=1) # Keep update freq relative to steps

    # --- Optimizer ---
    OPTIMIZER_TYPE: Literal["Adam", "AdamW", "SGD"] = Field("AdamW")
    LEARNING_RATE: float = Field(1e-4, gt=0) # Keep LR, scheduler will handle decay
    WEIGHT_DECAY: float = Field(1e-5, ge=0)
    GRADIENT_CLIP_VALUE: Optional[float] = Field(default=1.0) # Allow None or positive

    # --- LR Scheduler ---
    LR_SCHEDULER_TYPE: Optional[Literal["StepLR", "CosineAnnealingLR"]] = Field(default="CosineAnnealingLR")
    LR_SCHEDULER_T_MAX: Optional[int] = Field(default=None) # Will be set based on MAX_TRAINING_STEPS
    LR_SCHEDULER_ETA_MIN: float = Field(1e-6, ge=0)

    # --- Loss Weights ---
    POLICY_LOSS_WEIGHT: float = Field(1.0, ge=0)
    VALUE_LOSS_WEIGHT: float = Field(1.0, ge=0)
    ENTROPY_BONUS_WEIGHT: float = Field(0.01, ge=0)

    # --- Checkpointing ---
    CHECKPOINT_SAVE_FREQ_STEPS: int = Field(500, ge=1) # Reduced save frequency relative to total steps

    @model_validator(mode='after')
    def check_buffer_sizes(self) -> 'TrainConfig':
        if self.MIN_BUFFER_SIZE_TO_TRAIN > self.BUFFER_CAPACITY:
            raise ValueError("MIN_BUFFER_SIZE_TO_TRAIN cannot be greater than BUFFER_CAPACITY.")
        if self.BATCH_SIZE > self.BUFFER_CAPACITY:
            raise ValueError("BATCH_SIZE cannot be greater than BUFFER_CAPACITY.")
        if self.BATCH_SIZE > self.MIN_BUFFER_SIZE_TO_TRAIN:
             print(f"Warning: BATCH_SIZE ({self.BATCH_SIZE}) is larger than MIN_BUFFER_SIZE_TO_TRAIN ({self.MIN_BUFFER_SIZE_TO_TRAIN}). This might lead to slow startup.")
        return self

    @model_validator(mode='after')
    def set_scheduler_t_max(self) -> 'TrainConfig':
        if self.LR_SCHEDULER_TYPE == "CosineAnnealingLR" and self.LR_SCHEDULER_T_MAX is None:
            if self.MAX_TRAINING_STEPS is not None:
                self.LR_SCHEDULER_T_MAX = self.MAX_TRAINING_STEPS # Adjust T_max based on new MAX_TRAINING_STEPS
            else:
                # Fallback if MAX_TRAINING_STEPS is somehow None despite default
                self.LR_SCHEDULER_T_MAX = 50_000 # Match reduced default
                print(f"Warning: MAX_TRAINING_STEPS is None, setting LR_SCHEDULER_T_MAX to default {self.LR_SCHEDULER_T_MAX}")
        # Ensure T_max is positive if set
        if self.LR_SCHEDULER_T_MAX is not None and self.LR_SCHEDULER_T_MAX <= 0:
            raise ValueError("LR_SCHEDULER_T_MAX must be positive if set.")
        return self

    @field_validator('GRADIENT_CLIP_VALUE')
    @classmethod
    def check_gradient_clip(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("GRADIENT_CLIP_VALUE must be positive if set.")
        return v