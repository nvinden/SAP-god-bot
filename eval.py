import sys
import os
from copy import deepcopy
import time
import re
import random

import torch

import numpy as np
from collections import defaultdict

current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from sapai.pets import Pet
from sapai.shop import Shop
from sapai.teams import Team
from sapai.battle import Battle
from sapai.player import Player, GoldException, WrongObjectException, FullTeamException

from model import SAPAI
from model_actions import call_action_from_q_index, num_agent_actions, agent_actions_list, action_beginning_index, action_index_to_action_type, create_available_action_mask, action2index,idx2pet, idx2food
from config import DEFAULT_CONFIGURATION, rollout_device, training_device, N_ACTIONS, USE_WANDB
from past_teams import PastTeamsOrganizer
import wandb

# This function runs "number of rollouts" number of games, each preforming the highest Q value action
# for each player. The function returns the average number of wins for the player, when fighting against.
# 1-1, 2-2, 3-3, 4-4, 5-5 ... pigs
def evaluate_model(net : SAPAI, pt_organizer : PastTeamsOrganizer, epoch : int, config : dict = None, number_of_rollouts : int = 25, max_number_of_actions = 20, number_of_pigs = 10) -> tuple[list, dict]:
    net = net.to(rollout_device)
    net.set_device("rollout")
    net.eval()

    team_file_string = ""

    if config is None:
        config = DEFAULT_CONFIGURATION

    if epoch % 3 == 0:
        players = [Player() for _ in range(number_of_rollouts)]
    else:
        players = [Player() for _ in range(number_of_pigs)]
    results = []

    # Data from the evaluation
    actions_used = {action : 0 for action in agent_actions_list}
    pets_bought = {pet : 0 for pet in idx2pet.values()}
    food_bought = {food : 0 for food in idx2food.values()}

    past_player_win_percetages = []

    # For each turn
    for turn_number in range(config["number_of_evaluation_turns"]):
        active_players = [player for player in players]
        active_player_indexes = [i for i in range(len(players))]
        for action_number in range(max_number_of_actions):
            if len(active_players) == 0:
                break

            actions_taken = np.array([get_best_legal_move(player, net, config = config, epoch = epoch) for player in active_players])

            action_type_taken_idx = [action_index_to_action_type(action) for action in actions_taken]
            for action_type in action_type_taken_idx: actions_used[action_type] += 1

            # For each action
            for player, action in zip(active_players, actions_taken):
                try:
                    if action != N_ACTIONS - 1:
                        call_action_from_q_index(player, action.item())
                except:
                    player.gold += 1
                    player.roll()

                # if bought pet
                if action >= action_beginning_index[action2index["buy_pet"]] and action < action_beginning_index[action2index["buy_pet"]] + num_agent_actions["buy_pet"]:
                    pets_bought[idx2pet[action - action_beginning_index[action2index["buy_pet"]]]] += 1

                # if bought food
                if action >= action_beginning_index[action2index["buy_food"]] and action < action_beginning_index[action2index["buy_food"]] + num_agent_actions["buy_food"]:
                    food_bought[idx2food[action - action_beginning_index[action2index["buy_food"]]]] += 1

            # Remove players that have pressed end turn
            not_ended_turns = actions_taken != (N_ACTIONS - 1)
            not_ended_turns = not_ended_turns.tolist()
            active_players = np.array(active_players)
            active_players = active_players[not_ended_turns]
            active_players = active_players.tolist()

        for player in players:
            player.end_turn()

        # Battle past players
        if epoch % 3 == 0:
            past_teams_win_percentage = [pt_organizer.battle_past_teams(player.team, turn_number, number_of_rollouts) for player in players]
            winner_list = [win_percentage > 0.5 for win_percentage in past_teams_win_percentage]
            past_teams_win_percentage = np.mean(past_teams_win_percentage)
            past_player_win_percetages.append(past_teams_win_percentage)
        else:
            winner_list = [True for _ in range(number_of_pigs)]

        # Battle the pigs
        win_list = np.array([battle_increasing_pigs(player, max_stats=25) for player in np.array(players)[:number_of_pigs]])
        avg_wins = np.mean(win_list)

        # Add teams to the out_file
        team_file_string += "Turn: " + str(turn_number) + "\n"
        for player in players[:10]:
            team_file_string += str(player.team) + "\n"
        team_file_string += "\n"

        # Next Turn
        for player, winner in zip(players, winner_list):
            pt_organizer.add_on_deck(deepcopy(player.team), turn_number)
            player.start_turn(winner = winner)

        results.append(avg_wins)

    # Save the file as "teams.txt"
    with open("teams.txt", "w") as f:
        f.write(team_file_string)
    
    if USE_WANDB:
        wandb.save("teams.txt")

    if epoch % 3 == 0:
        return results, actions_used, past_player_win_percetages, pets_bought, food_bought
    return results, actions_used, None, pets_bought, food_bought

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
        
        try:
            battle = Battle(player.team, pig_team)
            result = battle.battle()
        except Exception as e:
            print(e)
            print(player.team)
            print(pig_team)
            result = 1

        if result == 0: # Player won
            n_wins += 1
        elif result == 1: # Player lost
            break
    
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
        return f"Bought {idx2pet[player_index]}"
    elif action_idx >= ranges[1] and action_idx < ranges[2]: # Sell pet
        player_index = action_idx - ranges[1]
        return f"Sold {idx2pet[player_index]}"
    elif action_idx >= ranges[2] and action_idx < ranges[3]: # Buy food
        player_index = action_idx - ranges[2]
        return f"Bought {idx2food[player_index]}"
    elif action_idx >= ranges[3] and action_idx < ranges[4]: # Combine
        player_index = action_idx - ranges[3]
        return f"Combined {idx2pet[player_index]}"
    elif action_idx >= ranges[4] and action_idx < ranges[5]: # Freeze
        player_index = action_idx - ranges[4]
        if player_index < len(idx2pet): return f"Froze {idx2pet[player_index]}"
        else: return f"Froze {idx2food[player_index - len(idx2pet)]}"
    elif action_idx >= ranges[5] and action_idx < ranges[6]: # Unfreeze
        player_index = action_idx - ranges[5]
        if player_index < len(idx2pet): return f"Unfroze {idx2pet[player_index]}"
        else: return f"Unfroze {idx2food[player_index - len(idx2pet)]}"
    elif action_idx >= ranges[6] and action_idx < ranges[7]: # End turn
        return "Ended turn"
    
    return "something_went_wrong"

def visualize_rollout(net: SAPAI, config : dict):
    net = net.to(rollout_device)
    net.set_device("rollout")
    net.eval()

    player = Player()

    epoch = 100


    while True:
        action_number, q_values = get_best_legal_move(player, net, config = config, epoch = epoch, return_q_values = True)
        return_signal = call_action_from_q_index(player, action_number)

        top_values = torch.topk(q_values, 5)
        top_values = [f"{top_values[0][i].item():.2f} {action_idx_to_string(top_values[1][i].item())}" for i in range(5)]

        action_str = action_idx_to_string(action_number)
        print("Player:", player)
        print("Action:", action_str)
        print("Top Moves:", top_values)
        print("Top actions:")
        print()

        if action_number == (N_ACTIONS - 1):
            player.end_turn()
            player.start_turn(winner = True)
            epoch += 1

def test_legal_move_masking():
    player = Player()

    player.gold += 17
    player.buy_pet(0)
    player.team.append(deepcopy(player.team.slots[0]))
    player.team.append(deepcopy(player.team.slots[0]))
    player.roll()
    player.buy_pet(0)
    player.buy_pet(0)

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

def get_best_legal_move(player : Player, net : SAPAI, config : dict, epoch = 20, epoch_masking = True, mask = None, return_q_values = False) -> int:
    state_encodings = net.state_to_encoding(player)
    q_value = net(state_encodings)
    q_value = q_value.squeeze(0)

    if mask is None:
        action_mask = create_available_action_mask(player)

        # if epoch makes turns illegal
        if epoch_masking:
            epoch_illegal_mask = create_epoch_illegal_mask(epoch, config)
            action_mask = action_mask * epoch_illegal_mask

        max_q_action_mask = np.stack(deepcopy(action_mask))
        max_q_action_mask = torch.tensor(max_q_action_mask, dtype = torch.int32, device = rollout_device).requires_grad_(False)
    else:
        max_q_action_mask = torch.tensor(mask.astype(np.int32), dtype = torch.int32, device = rollout_device).requires_grad_(False)

    # Adding mask to the max q action vector
    where_zeros = max_q_action_mask < 0.5
    q_value[where_zeros] = -9999999

    action_taken = int(torch.argmax(q_value).cpu().numpy())

    if return_q_values:
        return action_taken, q_value
    return action_taken

def create_epoch_illegal_mask(epoch, config):
    action_mask = np.ones(N_ACTIONS)
    if epoch >= config["illegalize_rolling"][0] and epoch <= config["illegalize_rolling"][1]:
        action_mask[action_beginning_index[action2index["roll"]]:action_beginning_index[action2index["roll"] + 1]] = 0
    if epoch >= config["illegalize_freeze_unfreeze"][0] and epoch <= config["illegalize_freeze_unfreeze"][1]:
        action_mask[action_beginning_index[action2index["freeze"]]:action_beginning_index[action2index["freeze"] + 1]] = 0
        action_mask[action_beginning_index[action2index["unfreeze"]]:action_beginning_index[action2index["unfreeze"] + 1]] = 0
    if epoch >= config["illegalize_combine"][0] and epoch <= config["illegalize_combine"][1]:
        action_mask[action_beginning_index[action2index["combine"]]:action_beginning_index[action2index["combine"] + 1]] = 0

    return action_mask

        
    
