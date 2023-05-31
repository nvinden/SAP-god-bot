import torch

DEFAULT_CONFIGURATION = {
    # Model parameters
    "d_model": 512,
    "nhead": 16,
    "num_layers": 16,

    # Rollout parameters
    "players_per_simulation": 28,
    "action_limit": 30, # Number of actions a player can do in a turn before they get skipped
    "number_of_battles_per_player_turn": 31,
    "number_of_battles_per_simulation": 20,
    "max_num_threads": 12,

    # Random action parameters
    "epsilon": 0.95,
    "epsilon_decay": 0.80,
    "epsilon_min": 0.10,

    # Training parameters
    "epochs": 20,
    "batch_size": 64,
    "number_of_updates_per_optimization_step": 800,
    "gamma": 0.9999,
    "learning_rate": 0.001,
}

rollout_device = "cuda"
training_device = "cuda"