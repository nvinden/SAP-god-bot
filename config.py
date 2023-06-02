VERBOSE = False
N_ACTIONS = 47

DEFAULT_CONFIGURATION = {
    # Model parameters
    "d_model": 512,
    "nhead": 16,
    "num_layers": 16,

    # Rollout parameters
    "players_per_simulation": 20,
    "action_limit": 20, # Number of actions a player can do in a turn before they get skipped
    "number_of_battles_per_player_turn": 31,
    "number_of_battles_per_simulation": 20,
    "max_num_threads": 16,

    # Random action parameters
    "epsilon": 0.95,
    "epsilon_decay": 0.975,
    "epsilon_min": 0.075,

    # Training parameters
    "epochs": 100,
    "batch_size": 64,
    "number_of_updates_per_optimization_step": 6000,
    "gamma": 0.999,
    "learning_rate": 0.001,

    # Action Illegalization parameters
    "illegalize_rolling": (0, 10),
    "illegalize_freeze_unfreeze": (0, 20),
    "illegalize_combine": (0, 0),

    # Reward parameters
    "allow_negative_rewards": True,
    "allow_stat_increase_as_reward": True
}

rollout_device = "cpu"
training_device = "cuda"