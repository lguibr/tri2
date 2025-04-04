# File: utils/init_checks.py
# --- Pre-Run Sanity Checks ---
import sys
import traceback
import numpy as np  # Import numpy

# --- MODIFIED: Import EnvConfig ---
from config import EnvConfig

# --- END MODIFIED ---

# Import core components
try:
    from environment.game_state import GameState
except ImportError as e:
    print(f"Error importing environment: {e}")
    sys.exit(1)


def run_pre_checks() -> bool:
    """Performs basic checks on GameState and configuration compatibility."""
    print("--- Pre-Run Checks ---")
    try:
        print("Checking GameState and Configuration Compatibility...")
        # --- MODIFIED: Instantiate EnvConfig ---
        env_config_instance = EnvConfig()
        # --- END MODIFIED ---

        gs_test = GameState()
        gs_test.reset()
        # --- MODIFIED: Check dictionary state structure ---
        s_test_dict = gs_test.get_state()

        if not isinstance(s_test_dict, dict):
            raise TypeError(
                f"GameState.get_state() should return a dict, but got {type(s_test_dict)}"
            )
        print("GameState state type check PASSED (returned dict).")

        # Check 'grid' component
        if "grid" not in s_test_dict:
            raise KeyError("State dictionary missing 'grid' key.")
        grid_state = s_test_dict["grid"]
        expected_grid_shape = env_config_instance.GRID_STATE_SHAPE
        if not isinstance(grid_state, np.ndarray):
            raise TypeError(
                f"State 'grid' component should be numpy array, but got {type(grid_state)}"
            )
        if grid_state.shape != expected_grid_shape:
            raise ValueError(
                f"State 'grid' shape mismatch! GameState:{grid_state.shape}, EnvConfig:{expected_grid_shape}"
            )
        print(f"GameState 'grid' state shape check PASSED (Shape: {grid_state.shape}).")

        # Check 'shapes' component
        if "shapes" not in s_test_dict:
            raise KeyError("State dictionary missing 'shapes' key.")
        shape_state = s_test_dict["shapes"]
        expected_shape_shape = (
            env_config_instance.NUM_SHAPE_SLOTS,
            env_config_instance.SHAPE_FEATURES_PER_SHAPE,
        )
        if not isinstance(shape_state, np.ndarray):
            raise TypeError(
                f"State 'shapes' component should be numpy array, but got {type(shape_state)}"
            )
        if shape_state.shape != expected_shape_shape:
            raise ValueError(
                f"State 'shapes' shape mismatch! GameState:{shape_state.shape}, EnvConfig:{expected_shape_shape}"
            )
        print(
            f"GameState 'shapes' state shape check PASSED (Shape: {shape_state.shape})."
        )
        # --- END MODIFIED ---

        _ = gs_test.valid_actions()
        print("GameState valid_actions check PASSED.")

        if not hasattr(gs_test, "game_score"):
            raise AttributeError("GameState missing 'game_score' attribute!")
        print("GameState 'game_score' attribute check PASSED.")

        if not hasattr(gs_test, "lines_cleared_this_episode"):
            raise AttributeError(
                "GameState missing 'lines_cleared_this_episode' attribute!"
            )
        print("GameState 'lines_cleared_this_episode' attribute check PASSED.")

        del gs_test
        print("--- Pre-Run Checks Complete ---")
        return True
    except (NameError, ImportError) as e:
        print(f"FATAL ERROR: Import/Name error: {e}")
    except (
        ValueError,
        AttributeError,
        TypeError,
        KeyError,
    ) as e:  # Added TypeError, KeyError
        print(f"FATAL ERROR during pre-run checks: {e}")
    except Exception as e:
        print(f"FATAL ERROR during GameState pre-check: {e}")
        traceback.print_exc()

    # If any check fails and raises an exception that's caught here, exit.
    sys.exit(1)  # Exit if checks fail
