# File: src/mcts/core/config.py
from pydantic import BaseModel, Field, field_validator


class MCTSConfig(BaseModel):
    """Configuration for Monte Carlo Tree Search (Pydantic model)."""

    # Increase simulation count significantly
    num_simulations: int = Field(1600, ge=1)
    puct_coefficient: float = Field(1.0, gt=0)
    temperature_initial: float = Field(1.0, ge=0)
    temperature_final: float = Field(0.01, ge=0)
    temperature_anneal_steps: int = Field(10_000, ge=0) 
    dirichlet_alpha: float = Field(0.3, gt=0)
    dirichlet_epsilon: float = Field(0.25, ge=0, le=1.0)
    max_search_depth: int = Field(100, ge=1) 

    @field_validator("temperature_final")
    @classmethod
    def check_temp_final_le_initial(cls, v: float, info) -> float:
        initial_temp = info.data.get("temperature_initial")
        if initial_temp is not None and v > initial_temp:
            raise ValueError(
                "temperature_final cannot be greater than temperature_initial"
            )
        return v
