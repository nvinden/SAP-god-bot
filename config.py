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
    "number_of_battles_per_player_turn": 5,
    "number_of_battles_per_simulation": 20,
    "max_num_threads": 24,

    # Evaluation parameters
    "number_of_evaluation_turns": 15,

    # Random action parameters
    "epsilon": 0.95,
    "epsilon_decay": 0.975,
    "epsilon_min": 0.075,

    # Training parameters
    "epochs": 100,
    "batch_size": 64,
    "num_updates_per_sample": 8.0,
    "gamma": 0.999,
    "learning_rate": 0.003,
    "win_percentage_threshold_past_teams": 0.55,

    # Action Illegalization parameters
    "illegalize_rolling": (0, 10),
    "illegalize_freeze_unfreeze": (0, 20),
    "illegalize_combine": (0, 0),

    # Reward parameters
    "allow_negative_rewards": True,
    "allow_stat_increase_as_reward": True
}

rollout_device = "cuda"
training_device = "cuda"