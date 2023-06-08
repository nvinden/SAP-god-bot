VERBOSE = False
N_ACTIONS = 383
USE_WANDB = True

rollout_device = "cuda"
training_device = "cuda"

DEFAULT_CONFIGURATION = {
    # Model parameters
    "d_model": 512,
    "nhead": 10, #32,
    "num_layers": 8, #16,

    # Rollout parameters
    "players_per_simulation": 30, #20,
    "action_limit": 15, # Number of actions a player can do in a turn before they get skipped
    "number_of_battles_per_player_turn": 5,
    "number_of_battles_per_simulation": 20, #20,
    "max_num_threads": 6, #12,

    # Evaluation parameters
    "number_of_evaluation_turns": 15,

    # Random action parameters
    "epsilon": 0.95,
    "epsilon_decay": 0.985,
    "epsilon_min": 0.075,

    # Training parameters
    "epochs": 300,
    "batch_size": 32,
    "num_updates_per_sample": 8.0,
    "gamma": 0.95,
    "learning_rate": 0.0001, # 0.001
    "win_percentage_threshold_past_teams": 0.55,
    "freeze_transformer_epochs": 5, 

    # Action Illegalization parameters
    "illegalize_rolling": (0, 0),
    "illegalize_freeze_unfreeze": (0, 25),
    "illegalize_combine": (0, 0),

    # Reward parameters
    "allow_negative_rewards": True,
    "allow_stat_increase_as_reward": False,
    "allow_combine_reward": False,
}

PRETRAIN_DEFAULT_CONFIGURATION = {
    # Model parameters
    #"d_model": 512,
    "nhead": 10, #32,
    "num_layers": 8, #16,

    # Training parameters
    "epochs": 100_000,
    "batch_size": 32,
    "num_updates_per_sample": 8.0,
    "learning_rate": 0.0001,

    # Players
    "number_of_players": 32
}