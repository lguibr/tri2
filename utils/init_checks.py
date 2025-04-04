# File: utils/init_checks.py
# --- Pre-Run Sanity Checks ---
import sys
import traceback
from config import EnvConfig

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
        gs_test = GameState()
        gs_test.reset()
        s_test = gs_test.get_state()
        if len(s_test) != EnvConfig.STATE_DIM:
            raise ValueError(
                f"State Dim Mismatch! GameState:{len(s_test)}, EnvConfig:{EnvConfig.STATE_DIM}"
            )
        print(f"GameState state dimension check PASSED (Length: {len(s_test)}).")

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
    except (ValueError, AttributeError) as e:
        print(f"FATAL ERROR during pre-run checks: {e}")
    except Exception as e:
        print(f"FATAL ERROR during GameState pre-check: {e}")
        traceback.print_exc()

    # If any check fails and raises an exception that's caught here, exit.
    sys.exit(1)  # Exit if checks fail
