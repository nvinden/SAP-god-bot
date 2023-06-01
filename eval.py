import sys
import os
from copy import deepcopy
import time
import re
import random

import torch

import numpy as np

current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from sapai.pets import Pet
from sapai.shop import Shop
from sapai.teams import Team
from sapai.battle import Battle
from sapai.player import Player, GoldException, WrongObjectException, FullTeamException

from model import SAPAI, N_ACTIONS
from model_actions import call_action_from_q_index, num_agent_actions, agent_actions_list, action_beginning_index, action_index_to_action_type, create_available_action_mask
from config import DEFAULT_CONFIGURATION, rollout_device, training_device

# This function runs "number of rollouts" number of games, each preforming the highest Q value action
# for each player. The function returns the average number of wins for the player, when fighting against.
# 1-1, 2-2, 3-3, 4-4, 5-5 ... pigs
def evaluate_model(net : SAPAI, config : dict = None, number_of_rollouts : int = 16, number_of_turns = 10, max_number_of_actions = 15) -> tuple[list, dict]:
    net = net.to(rollout_device)
    net.set_device("rollout")
    net.eval()

    if config is None:
        config = DEFAULT_CONFIGURATION

    players = [Player() for _ in range(number_of_rollouts)]
    results = []
    actions_used = {action : 0 for action in agent_actions_list}

    # For each turn
    for turn_number in range(number_of_turns):
        active_players = [player for player in players]
        for action_number in range(max_number_of_actions):
            if len(active_players) == 0:
                break

            state_encodings = net.state_to_encoding(active_players)
            q_value, _ = net(state_encodings)

            action_mask = [create_available_action_mask(player) for player in active_players]

            max_q_action_mask = np.stack(deepcopy(action_mask))
            max_q_action_mask = torch.tensor(max_q_action_mask, dtype = torch.float32, device = rollout_device).requires_grad_(False)

            # Adding mask to the max q action vector
            where_zeros = max_q_action_mask < 0.5
            q_value[where_zeros] = -9999999

            actions_taken = torch.argmax(q_value, dim = 1).cpu().numpy()

            action_type_taken_idx = [action_index_to_action_type(action) for action in actions_taken]
            for action_type in action_type_taken_idx: actions_used[action_type] += 1

            # For each action
            for player, action in zip(active_players, actions_taken):
                call_action_from_q_index(player, action.item())

            # Remove players that have pressed end turn
            ended_turns = actions_taken != (q_value.shape[-1] - 1)
            ended_turns = ended_turns.tolist()
            active_players = np.array(active_players)
            active_players = active_players[ended_turns]
            active_players = active_players.tolist()

        # Battle the players
        win_list = np.array([battle_increasing_pigs(player, max_stats=5) for player in players])
        avg_wins = np.mean(win_list)

        # Next Turn
        for player in players:
            player.start_turn()

        results.append(avg_wins)

    return results, actions_used

def battle_increasing_pigs(player : Player, max_stats : 50) -> int:
    n_wins = -1

    #player.buy_pet(0)

    empty_team = Team()
    battle = Battle(player.team, empty_team)
    result = battle.battle()

    if result == 0: # Player won
        n_wins += 1


    for i in range(1, max_stats + 1):
        pig_team = _create_pig_team(i)
        
        battle = Battle(player.team, pig_team)
        result = battle.battle()

        if result == 0: # Player won
            n_wins = 1
    
    return n_wins
            
def _create_pig_team(stats : int):
    assert stats > 0 and stats <= 50

    team = Team()
    pig = Pet('pig')
    pig.set_attack(stats)
    pig.set_health(stats)

    for _ in range(5):
        team.append(deepcopy(pig))

    return team

def action_idx_to_string(action_idx : int) -> str:
    # This code is gross, but it just defines the ranges the actions take
    ranges = [sum([num_agent_actions[agent_actions_list[j]] for j in range(i + 1)]) for i in range(len(agent_actions_list))]

    assert action_idx >= 0 and action_idx < ranges[-1]

    if action_idx == 0: # Roll
        return "Rolled"
    elif action_idx >= ranges[0] and action_idx < ranges[1]: # Buy pet
        player_index = action_idx - ranges[0]
        return f"Bought pet at index {player_index}"
    elif action_idx >= ranges[1] and action_idx < ranges[2]: # Sell pet
        player_index = action_idx - ranges[1]
        return f"Sold pet at index {player_index}"
    elif action_idx >= ranges[2] and action_idx < ranges[3]: # Buy food
        player_index = (action_idx - ranges[2]) % 5
        food_index = (action_idx - ranges[2]) // 5
        return f"Bought food at index {food_index} and put it on pet at index {player_index}"
    elif action_idx >= ranges[3] and action_idx < ranges[4]: # Combine
        player_index = action_idx - ranges[3]
        return f"Combined pets at index {player_index}"
    elif action_idx >= ranges[4] and action_idx < ranges[5]: # Freeze
        player_index = action_idx - ranges[4]
        return f"Froze pet at index {player_index}"
    elif action_idx >= ranges[5] and action_idx < ranges[6]: # Unfreeze
        player_index = action_idx - ranges[5]
        return f"Unfroze pet at index {player_index}"
    elif action_idx >= ranges[6] and action_idx < ranges[7]: # End turn
        return "Ended turn"
    
    return "something_went_wrong"

def visualize_rollout(net: SAPAI):
    net = net.to(rollout_device)
    net.set_device("rollout")
    net.eval()

    player = Player()

    while True:
        state_encoding = net.state_to_encoding([player])
        q_value, _ = net(state_encoding)

        action_number = torch.argmax(q_value, dim = 1).cpu().numpy()[0]
        return_signal = call_action_from_q_index(player, action_number)

        action_str = action_idx_to_string(action_number)
        print("Action:", action_str)
        print("Return signal:", return_signal)
        print("Player:", player)

        if action_number == (q_value.shape[-1] - 1):
            break

def test_legal_move_masking():
    player = Player()

    while(True):
        # Legal moves mask
        mask = create_available_action_mask(player)

        # Testing moves out and recording results
        all_move_results = [call_action_from_q_index(deepcopy(player), i) for i in range(len(mask))]

        for i, (mask_legal, run_legal) in enumerate(zip(mask, all_move_results)):
            mask_legal = int(mask_legal)

            result_num = 1 if run_legal in ["success", "end_turn"] else 0
            if result_num != mask_legal:
                print("FAILURE: ", mask_legal, run_legal, action_idx_to_string(i))
                call_action_from_q_index(deepcopy(player), i)
                _ = create_available_action_mask(player)

        all_double_legal_moves = [i for i, (mask_legal, run_legal) in enumerate(zip(mask, all_move_results)) if mask_legal == 1 and run_legal in ["success", "end_turn"]]
        random_double_legal_move = random.choice(all_double_legal_moves)
        
        call_action_from_q_index(player, random_double_legal_move)

        print("AFTER: ", player)
        print("Move: ", action_idx_to_string(random_double_legal_move))
        print()
        print()

        if random_double_legal_move == N_ACTIONS - 1: # Turn ended
            player.start_turn()

        
    
