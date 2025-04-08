class MCTSConfig:
    """Configuration parameters for the Monte Carlo Tree Search."""

    # Exploration constant (c_puct in PUCT formula)
    # Balances exploitation (Q value) and exploration (prior P and visit counts N)
    # Higher values encourage exploring less-visited actions with high priors.
    PUCT_C: float = 1.5

    # Number of MCTS simulations to run for each move decision.
    # More simulations generally lead to stronger play but take more time.
    NUM_SIMULATIONS: int = 100

    # Temperature parameter for action selection during self-play.
    # Controls the randomness of move selection based on visit counts.
    # Higher temperature -> more exploration (sample proportionally to N^(1/temp))
    # Lower temperature -> more exploitation (closer to choosing the most visited action)
    # Often starts high (e.g., 1.0) and anneals to a small value (e.g., 0.1 or 0) during the game.
    TEMPERATURE_INITIAL: float = 1.0
    TEMPERATURE_FINAL: float = 0.01
    TEMPERATURE_ANNEAL_STEPS: int = (
        30  # Number of game steps over which to anneal temperature
    )

    # Dirichlet noise parameters for exploration at the root node during self-play.
    # Adds noise to the prior probabilities from the network to encourage exploration,
    # especially early in training.
    # Alpha determines the shape of the distribution, Epsilon the weight of the noise.
    DIRICHLET_ALPHA: float = 0.3
    DIRICHLET_EPSILON: float = 0.25

    # Maximum depth for the MCTS search tree (optional, can prevent excessive depth)
    MAX_SEARCH_DEPTH: int = 100
