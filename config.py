VERBOSE = False
N_ACTIONS = 383
USE_WANDB = False

rollout_device = "cuda"
training_device = "cuda"

DEFAULT_CONFIGURATION = {
    # Model parameters
    "d_model": 316,
    "nhead": 3,#3,
    "num_layers": 2, #2,

    # Rollout parameters
    "players_per_simulation": 30,
    "action_limit": 15, # Number of actions a player can do in a turn before they get skipped
    "number_of_battles_per_player_turn": 5,
    "number_of_battles_per_simulation": 20,
    "max_num_threads": 10, #12,

    # Evaluation parameters
    "number_of_evaluation_turns": 20,

    # Random action parameters
    "epsilon": 0.20,
    "epsilon_decay": 1.00,
    "epsilon_min": 0.15,

    # Training parameters
    "epochs": 300,
    "batch_size": 64,
    "num_updates_per_sample": 8.0,
    "gamma": 0.995,
    "learning_rate": 0.0000045, #Original tested value 0.00007
    "learning_rate_decay": 1.00,
    "win_percentage_threshold_past_teams": 0.55,
    "freeze_transformer_epochs": 5, 
    "target_net_update_epochs": 10,
    "first_epoch_no_learning": True,

    # Action Illegalization parameters
    "illegalize_rolling": (0, 0),
    "illegalize_freeze_unfreeze": (0, 500),
    "illegalize_combine": (0, 0),

    # Reward parameters
    "allow_negative_rewards": True,
    "allow_stat_increase_as_reward": False,
    "allow_combine_reward": False,
    "allow_penalty_for_unused_gold": True,
    "allow_multi_freeze_unfreeze_penalty": True,

    # Pretrain Transformer parameters
    "pretrain_transformer": False,
    "full_part_combo_split": [0.0, 0.80, 0.20],
    "partial_number_of_actions": 45,
    "pretrain_only_buy_turn_allowable_items": True,
    "train_on_increasing_partial_actions": False,
    "eval_mask_type": "part"
}

PRETRAIN_DEFAULT_CONFIGURATION = {
    # Model parameters
    "d_model": 316,
    "nhead": 3,
    "num_layers": 2,

    # Training parameters
    "epochs": 100_000,
    "batch_size": 32,
    "num_updates_per_sample": 8.0,
    "learning_rate": 0.0001,

    # Players
    "number_of_players": 32,

    "pretrain_transformer": True
}