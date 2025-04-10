# File: src/features/extractor.py
import numpy as np
from typing import TYPE_CHECKING

# Import necessary components from environment and utils
from src.environment import GameState  # Import GameState directly
from src.utils.types import StateType  # Use core StateType (StateDict)
from src.config import EnvConfig, ModelConfig  # Need configs for dimensions

# Import grid_features from the same features package
from . import grid_features

# Import Triangle from the new structs module
from src.structs import Triangle

if TYPE_CHECKING:
    pass


class GameStateFeatures:
    """Extracts features from GameState for NN input."""

    def __init__(self, game_state: "GameState", model_config: ModelConfig):
        self.gs = game_state
        self.env_config = game_state.env_config  # Get env_config from GameState
        self.model_config = model_config  # Need model_config for feature dimensions

    def _get_grid_state(self) -> np.ndarray:
        """Returns grid occupancy and death state as channels."""
        grid_occupied = self.gs.grid_data.get_occupied_state()
        grid_death = self.gs.grid_data.get_death_state()
        # Ensure the number of channels matches ModelConfig
        expected_channels = self.model_config.GRID_INPUT_CHANNELS
        grid_state = np.stack([grid_occupied, grid_death], axis=0).astype(np.float32)
        if grid_state.shape[0] != expected_channels:
            # This basic extractor only provides 2 channels.
            # If ModelConfig expects more, this needs adjustment or ModelConfig change.
            # For now, we'll raise an error if mismatched.
            raise ValueError(
                f"Mismatch between extracted grid channels ({grid_state.shape[0]}) and ModelConfig.GRID_INPUT_CHANNELS ({expected_channels})"
            )
        return grid_state

    def _get_shape_features(self) -> np.ndarray:
        """Extracts features for each shape slot."""
        num_slots = self.env_config.NUM_SHAPE_SLOTS
        # Calculate features per shape based on ModelConfig's expectation
        # This assumes OTHER_NN_INPUT_FEATURES_DIM is correctly set in ModelConfig
        # based on the concatenation logic in get_combined_other_features.
        # We need a way to know how many features *this function* produces per shape.
        # Let's define it locally for clarity, ensuring it matches the calculation
        # used for OTHER_NN_INPUT_FEATURES_DIM in ModelConfig.
        FEATURES_PER_SHAPE_HERE = 7  # Number of features calculated below
        shape_feature_matrix = np.zeros(
            (num_slots, FEATURES_PER_SHAPE_HERE), dtype=np.float32
        )

        for i, shape in enumerate(
            self.gs.shapes
        ):  # Uses Shape from structs (via GameState)
            if shape and shape.triangles:
                n_tris = len(shape.triangles)
                ups = sum(1 for _, _, is_up in shape.triangles if is_up)
                downs = n_tris - ups
                min_r, min_c, max_r, max_c = shape.bbox()
                height = max_r - min_r + 1
                width_eff = (max_c - min_c + 1) * 0.75 + 0.25 if n_tris > 0 else 0

                # Populate features
                shape_feature_matrix[i, 0] = np.clip(n_tris / 5.0, 0, 1)
                shape_feature_matrix[i, 1] = ups / n_tris if n_tris > 0 else 0
                shape_feature_matrix[i, 2] = downs / n_tris if n_tris > 0 else 0
                shape_feature_matrix[i, 3] = np.clip(
                    height / self.env_config.ROWS, 0, 1
                )
                shape_feature_matrix[i, 4] = np.clip(
                    width_eff / self.env_config.COLS, 0, 1
                )
                shape_feature_matrix[i, 5] = np.clip(
                    ((min_r + max_r) / 2.0) / self.env_config.ROWS, 0, 1
                )
                shape_feature_matrix[i, 6] = np.clip(
                    ((min_c + max_c) / 2.0) / self.env_config.COLS, 0, 1
                )

        return shape_feature_matrix.flatten()

    def _get_shape_availability(self) -> np.ndarray:
        """Returns a binary vector indicating which shape slots are filled."""
        return np.array([1.0 if s else 0.0 for s in self.gs.shapes], dtype=np.float32)

    def _get_explicit_features(self) -> np.ndarray:
        """Extracts scalar features like score, heights, holes, etc."""
        # Define the number of explicit features calculated here.
        # This must match the calculation for OTHER_NN_INPUT_FEATURES_DIM in ModelConfig.
        EXPLICIT_FEATURES_DIM_HERE = 6
        features = np.zeros(EXPLICIT_FEATURES_DIM_HERE, dtype=np.float32)
        occupied = self.gs.grid_data.get_occupied_state()  # Use method from GridData
        death = self.gs.grid_data.get_death_state()  # Use method from GridData
        rows, cols = self.env_config.ROWS, self.env_config.COLS

        heights = grid_features.get_column_heights(occupied, death, rows, cols)
        holes = grid_features.count_holes(occupied, death, heights, rows, cols)
        bump = grid_features.get_bumpiness(heights)
        total_playable_cells = np.sum(~death)

        # Populate features
        features[0] = np.clip(self.gs.game_score / 100.0, -5.0, 5.0)
        features[1] = np.mean(heights) / rows if rows > 0 else 0
        features[2] = np.max(heights) / rows if rows > 0 else 0
        features[3] = holes / total_playable_cells if total_playable_cells > 0 else 0
        features[4] = (bump / (cols - 1)) / rows if cols > 1 and rows > 0 else 0
        features[5] = np.clip(self.gs.pieces_placed_this_episode / 100.0, 0, 1)

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def get_combined_other_features(self) -> np.ndarray:
        """Combines all non-grid features into a single flat vector."""
        shape_feats = self._get_shape_features()
        avail_feats = self._get_shape_availability()
        explicit_feats = self._get_explicit_features()
        combined = np.concatenate([shape_feats, avail_feats, explicit_feats])

        # Validate shape against ModelConfig expectation
        expected_dim = self.model_config.OTHER_NN_INPUT_FEATURES_DIM
        if combined.shape[0] != expected_dim:
            raise ValueError(
                f"Combined other_features dimension mismatch! Extracted {combined.shape[0]}, but ModelConfig expects {expected_dim}"
            )

        return combined.astype(np.float32)


def extract_state_features(
    game_state: "GameState", model_config: ModelConfig
) -> StateType:
    """
    Extracts and returns the state dictionary {grid, other_features} for NN input.
    Requires ModelConfig to ensure dimensions match the network's expectations.
    """
    extractor = GameStateFeatures(game_state, model_config)
    state_dict: StateType = {
        "grid": extractor._get_grid_state(),
        "other_features": extractor.get_combined_other_features(),
    }
    return state_dict
