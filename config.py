VERBOSE = False
N_ACTIONS = 383
USE_WANDB = True

rollout_device = "cuda"
training_device = "cuda"

DEFAULT_CONFIGURATION = {
    # Model parameters
    "d_model": 316,
    "nhead": 8,
    "num_layers": 5,

    # Rollout parameters
    "players_per_simulation": 30,
    "action_limit": 15, # Number of actions a player can do in a turn before they get skipped
    "number_of_battles_per_player_turn": 5,
    "number_of_battles_per_simulation": 17,
    "max_num_threads": 6, #12,

    # Evaluation parameters
    "number_of_evaluation_turns": 15,

    # Random action parameters
    "epsilon": 0.95,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.075,

    # Training parameters
    "epochs": 300,
    "batch_size": 128,
    "num_updates_per_sample": 8.0,
    "gamma": 0.985,
    "learning_rate": 0.00007,
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
    "allow_penalty_for_unused_gold": True,
    "allow_multi_freeze_unfreeze_penalty": True
}

PRETRAIN_DEFAULT_CONFIGURATION = {
    # Model parameters
    "d_model": 316,
    "nhead": 8,
    "num_layers": 5,

    # Training parameters
    "epochs": 100_000,
    "batch_size": 32,
    "num_updates_per_sample": 8.0,
    "learning_rate": 0.0001,

    # Players
    "number_of_players": 32
}

REORDER_DEFAULT_CONFIGURATION = {
    # Model parameters
    "d_model": 216,
    "nhead": 3,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 216,

    # Rollout parameters
    "action_limit": 15, # Number of actions a player can do in a turn before they get skipped
    "number_of_rollout_turns": 15,
    "max_num_threads": 24,
    "number_of_players": 50,

    # Training parameters
    "random_to_rollout_ratio": 2.0,
    "epochs": 100,
    "batch_size": 32,
    "num_updates_per_sample": 16,
    "learning_rate": 0.0001,
}