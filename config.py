
DEFAULT_CONFIGURATION = {
    # Model parameters
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,

    # Rollout parameters
    "players_per_simulation": 8,
    "action_limit": 30, # Number of actions a player can do in a turn before they get skipped
    "number_of_battles_per_player_turn": 3,
    "number_of_battles_per_simulation": 20,
    "max_num_threads": 24,

    # Random action parameters
    "epsilon": 0.95,
    "epsilon_decay": 0.999,
    "epsilon_min": 0.01,

    # Training parameters
}