import sys
import os
from copy import deepcopy
import time
import re

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
from model_actions import call_action_from_q_index
from config import DEFAULT_CONFIGURATION, rollout_device, training_device

# This function runs "number of rollouts" number of games, each preforming the highest Q value action
# for each player. The function returns the average number of wins for the player, when fighting against.
# 1-1, 2-2, 3-3, 4-4, 5-5 ... pigs
def evaluate_model(net : SAPAI, config : dict = None, number_of_rollouts : int = 16, number_of_turns = 10, max_number_of_actions = 15):
    net = net.to(rollout_device)
    net.set_device("rollout")
    net.eval()

    if config is None:
        config = DEFAULT_CONFIGURATION

    players = [Player() for _ in range(number_of_rollouts)]
    results = []

    # For each turn
    for turn_number in range(number_of_turns):
        active_players = [player for player in players]
        for action_number in range(max_number_of_actions):
            if len(active_players) == 0:
                break

            state_encodings = net.state_to_encoding(active_players)
            q_value, _ = net(state_encodings)
            actions_taken = torch.argmax(q_value, dim = 1).cpu().numpy()

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

        results.append(avg_wins)

    return results

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