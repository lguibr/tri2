# File: run_interactive.py
# File: run_interactive.py
# Change: Updated imports for app, environment, visualization.
import sys
import os
import argparse
import logging
import traceback

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Updated imports using the new structure
from src import app, config, utils

# Environment and Visualization might be needed for type hints or specific setup later
# but the Application class handles their internal use.
from src import environment
from src import visualization
from src.mcts import MCTSConfig  # Import Pydantic MCTSConfig


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaTriangle Interactive Modes")
    parser.add_argument(
        "--mode",
        type=str,
        default="play",
        choices=["play", "debug"],
        help="Interaction mode ('play' or 'debug')",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.info(f"Running in {args.mode.capitalize()} mode...")
    utils.set_random_seeds(args.seed)

    # Instantiate MCTSConfig needed for validation function
    mcts_config = MCTSConfig() # Instantiate Pydantic model
    # Pass MCTSConfig instance to validation
    config.print_config_info_and_validate(mcts_config)

    try:
        # Pass mode to the Application constructor
        # Application now uses the refactored environment and visualization internally
        app_instance = app.Application(mode=args.mode)
        app_instance.run()
    except ImportError as e:
        logger.error(f"ImportError: {e}")
        logger.error("Please ensure:")
        logger.error("1. You are running from the project root directory.")
        logger.error("2. All required files exist in 'src'.")
        logger.error(
            "3. Dependencies are installed (`pip install -r requirements.txt`)."
        )
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

    logger.info("Exiting.")