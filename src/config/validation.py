# File: src/config/validation.py
import logging
from pydantic import ValidationError

# Import config classes directly
from .env_config import EnvConfig
from .model_config import ModelConfig
from .train_config import TrainConfig
from .vis_config import VisConfig
from .persistence_config import PersistenceConfig
from .app_config import APP_NAME

# Import MCTSConfig from its location
from .mcts_config import MCTSConfig # Updated import

logger = logging.getLogger(__name__)

def print_config_info_and_validate(mcts_config_instance: MCTSConfig): # Accept instance
    """Prints configuration summary and performs validation using Pydantic."""
    print("-" * 40)
    print("Configuration Validation & Summary")
    print("-" * 40)
    all_valid = True
    configs_validated = {}

    config_classes = {
        "Environment": EnvConfig,
        "Model": ModelConfig,
        "Training": TrainConfig,
        "Visualization": VisConfig,
        "Persistence": PersistenceConfig,
        "MCTS": MCTSConfig, # Add MCTSConfig here
    }

    # Validate and store instances
    for name, ConfigClass in config_classes.items():
        try:
            # If an instance is passed (like mcts_config_instance), use it, otherwise instantiate
            if name == "MCTS" and mcts_config_instance is not None:
                instance = mcts_config_instance
                # Re-validate if needed, though Pydantic validates on init
                # MCTSConfig.model_validate(instance.model_dump()) # Optional re-validation
                print(f"[{name}] - Instance provided OK")
            else:
                instance = ConfigClass() # Instantiate with defaults or loaded values
                print(f"[{name}] - Validated OK")
            configs_validated[name] = instance
        except ValidationError as e:
            logger.error(f"Validation failed for {name} Config:")
            logger.error(e)
            all_valid = False
            configs_validated[name] = None # Mark as invalid
        except Exception as e:
            logger.error(f"Unexpected error instantiating/validating {name} Config: {e}")
            all_valid = False
            configs_validated[name] = None

    print("-" * 40)
    print("Configuration Values:")
    print("-" * 40)

    # Print validated config values
    for name, instance in configs_validated.items():
        print(f"--- {name} Config ---")
        if instance:
            # Use model_dump for clean output
            dump_data = instance.model_dump()
            for field_name, value in dump_data.items():
                # Exclude potentially long lists/dicts from summary?
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {field_name}: [List with {len(value)} items]")
                elif isinstance(value, dict) and len(value) > 5:
                    print(f"  {field_name}: {{Dict with {len(value)} keys}}")
                else:
                    print(f"  {field_name}: {value}")
        else:
            print("  <Validation Failed>")
        print("-" * 20)


    print("-" * 40)
    if not all_valid:
        logger.critical("Configuration validation failed. Please check errors above.")
        raise ValueError("Invalid configuration settings.")
    else:
        logger.info("All configurations validated successfully.")
    print("-" * 40)