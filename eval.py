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
from model_actions import call_action_from_q_index, num_agent_actions, agent_actions_list, action_beginning_index, action_index_to_action_type, create_available_action_mask, action2index,idx2pet, idx2food, create_available_item_mask, pet2idx, idx2pet
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

    team_file_string = f"EPOCH: {epoch}\n"
    food_placed_string = f"EPOCH: {epoch}\n"

    if config is None:
        config = DEFAULT_CONFIGURATION

    if epoch % 3 == 0:
        players = [Player() for _ in range(number_of_rollouts)]
        if config["pretrain_transformer"]:
            pretrain_masks = create_available_item_mask(players, config, [config["eval_mask_type"]] * number_of_rollouts)
        else:
            pretrain_masks = None
    else:
        players = [Player() for _ in range(number_of_pigs)]
        if config["pretrain_transformer"]:
            pretrain_masks = create_available_item_mask(players, config, [config["eval_mask_type"]] * number_of_pigs)
        else:
            pretrain_masks = None
    results = []

    # Data from the evaluation
    actions_used = {action : [0] * config["number_of_evaluation_turns"] for action in agent_actions_list}
    pets_bought = {pet : [0] * config["number_of_evaluation_turns"] for pet in idx2pet.values()}
    food_bought = {food : [0] * config["number_of_evaluation_turns"] for food in idx2food.values()}
    pets_sold = {pet : [0] * config["number_of_evaluation_turns"] for pet in idx2pet.values()}

    past_player_win_percetages = []

    # For each turn
    for turn_number in range(config["number_of_evaluation_turns"]):
        active_players = [player for player in players]
        active_pretrain_masks = None if pretrain_masks is None else np.array([pretrain_mask for pretrain_mask in pretrain_masks])
        for action_number in range(max_number_of_actions):
            if len(active_players) == 0:
                break

            _, actions_taken = net.get_masked_q_values(active_players, epoch = epoch, return_best_move = True, pretrain_action_mask = active_pretrain_masks)

            action_type_taken_idx = [action_index_to_action_type(action) for action in actions_taken["all"]]
            for action_type in action_type_taken_idx: actions_used[action_type][turn_number] += 1

            # For each action
            for player, action, food_best, sell_best in zip(active_players, actions_taken["all"], actions_taken["food"], actions_taken["sell"]):
                return_food_val = None
                return_sell_val = None

                copied_player = deepcopy(player)

                try:
                    if action != N_ACTIONS - 1:
                        _, return_food_val, return_sell_val = call_action_from_q_index(player, action.item(), food_best_move = food_best, sell_best_move = sell_best, epsilon = -0.10, config = config)
                except:
                    player.gold += 1
                    player.roll()
                    continue

                # If food was bought
                if return_food_val is not None:
                    pet_who_ate = idx2pet[return_food_val]
                    food_placed_string += str(copied_player.team)
                    food_placed_string += f"{pet_who_ate} ate food {idx2food[action.item() - action_beginning_index[action2index['buy_food']]]}\n\n"

                # if bought pet
                if action >= action_beginning_index[action2index["buy_pet"]] and action < action_beginning_index[action2index["buy_pet"]] + num_agent_actions["buy_pet"]:
                    pets_bought[idx2pet[action - action_beginning_index[action2index["buy_pet"]]]][turn_number] += 1

                # if bought food
                if action >= action_beginning_index[action2index["buy_food"]] and action < action_beginning_index[action2index["buy_food"]] + num_agent_actions["buy_food"]:
                    food_bought[idx2food[action - action_beginning_index[action2index["buy_food"]]]][turn_number] += 1

                # if pet sold
                if action >= action_beginning_index[action2index["sell"]] and action < action_beginning_index[action2index["sell"]] + num_agent_actions["sell"]:
                    pets_sold[idx2pet[action - action_beginning_index[action2index["sell"]]]][turn_number] += 1

                if return_sell_val is not None:
                    pet_name = idx2pet[return_sell_val % len(idx2pet)]
                    if return_sell_val < 5:
                        pets_sold[pet_name][turn_number] += 1
                        actions_used["sell"][turn_number] += 1
                    else:
                        actions_used["combine"][turn_number] += 1

            # Remove players that have pressed end turn
            not_ended_turns = actions_taken["all"] != (N_ACTIONS - 1)
            not_ended_turns = not_ended_turns.tolist()
            active_players = np.array(active_players)
            active_players = active_players[not_ended_turns]
            active_players = active_players.tolist()

            if active_pretrain_masks is not None:
                active_pretrain_masks = np.array(active_pretrain_masks)[not_ended_turns]

        for player in players:
            try:
                player.end_turn()
            except:
                continue

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

    # Make teams and food_place directory if they do not aleready exist
    if not os.path.isdir("additional_info"):
        os.mkdir("additional_info")

    # Save the file as "teams.txt" appending to the end of the file
    with open(f"additional_info/teams_{epoch}.txt", "w") as f:
        f.write(team_file_string)
    
    # Save the file as "food_placed.txt"
    with open(f"additional_info/food_placed_{epoch}.txt", "w") as f:
        f.write(food_placed_string)
    
    if USE_WANDB:
        wandb.save(f"additional_info/teams_{epoch}.txt")
        wandb.save(f"additional_info/food_placed_{epoch}.txt")

    if epoch % 3 == 0:
        return results, actions_used, past_player_win_percetages, pets_bought, food_bought, pets_sold
    return results, actions_used, None, pets_bought, food_bought, pets_sold

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

    pretrain_mask = None
    if config["pretrain_transformer"]:
        pretrain_mask = create_available_item_mask(player, config, ["full"])

    while True:
        q_values, action_number = net.get_masked_q_values(player, return_best_move = True, pretrain_action_mask = pretrain_mask)
        action_number_main = int(action_number["all"])

        ret_val, food_return_index, sell_return_index = call_action_from_q_index(player, action_number_main, food_best_move=int(action_number["food"]), sell_best_move=int(action_number["sell"]), epsilon = -0.1, config = config)

        top_values = torch.topk(q_values['all'], 5)
        top_q = top_values[0][0].detach().cpu().numpy()
        top_action_idx = top_values[1][0].detach().cpu().numpy()
        top_values = [f"{float(top_q[i]):.2f} {action_idx_to_string(top_action_idx[i])}" for i in range(5)]

        action_str = action_idx_to_string(action_number_main)
        print("Ret_val", ret_val)
        print("Player:", player)
        print("Action:", action_str)
        print("Top Moves:", top_values)
        if food_return_index is not None:
            print("Fed to Pet: ", idx2pet[food_return_index])
        if sell_return_index is not None:
            print("Bought Sold Pet: ", idx2pet[sell_return_index])
        print()

        if action_number_main == (N_ACTIONS - 1):
            player.end_turn()
            player.start_turn(winner = True)
            epoch += 1

def test_legal_move_masking(config):
    player = Player()
    net = SAPAI()

    while(True):
        # Legal moves mask
        if config["pretrain_transformer"]:
            pretrain_mask = create_available_item_mask(player, config, ["full"])
        else:
            pretrain_mask = None

        mask = net.get_masked_q_values(player, return_masks = True, pretrain_action_mask = pretrain_mask)[0]

        # Testing moves out and recording results
        all_move_results = [call_action_from_q_index(deepcopy(player), i, config=config) for i in range(len(mask))]

        for i, (mask_legal, run_legal) in enumerate(zip(mask, all_move_results)):
            mask_legal = int(mask_legal)

            result_num = 1 if run_legal in ["success", "end_turn"] else 0
            if mask_legal == 1 and result_num == 0:
                print("FAILURE: ", mask_legal, run_legal, action_idx_to_string(i))
                call_action_from_q_index(deepcopy(player), i, config=config)
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