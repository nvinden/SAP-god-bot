
DEFAULT_CONFIGURATION = {
    # Model parameters
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,

    # Rollout parameters
    "players_per_simulation": 32,
    "action_limit": 100, # Number of actions a player can do in a turn before they get skipped
    
}