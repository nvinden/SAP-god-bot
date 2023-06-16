import sys
import os
from copy import deepcopy
import time
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict
import itertools
import datetime
import pickle
from Levenshtein import ratio

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed


current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from model_actions import *
from model import SAPAI, SAPAIReorder
from config import REORDER_DEFAULT_CONFIGURATION, rollout_device, training_device, VERBOSE, N_ACTIONS, USE_WANDB
from eval import evaluate_model, visualize_rollout, test_legal_move_masking, get_best_legal_move, create_epoch_illegal_mask
from past_teams import PastTeamsOrganizer

from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player
from sapai.battle import Battle
from sapai.pets import Pet
from sapai.foods import Food
import math

import wandb

def get_random_pet_percentages():
    # Generating percent chances pet shows up in shop or in team
    none_pet_share = 30.0
    random_pet_percentages = {"pet-none" : none_pet_share / (none_pet_share + len(pet2idx))}
    for pet in pet2idx.keys():
        random_pet_percentages[pet] = 1.0 / (none_pet_share + len(pet2idx))

    return random_pet_percentages

def get_random_food_percentages():
    # Generating percent chances pet shows up in shop or in team
    none_pet_share = 10.0
    random_pet_percentages = {"food-none" : none_pet_share / (none_pet_share + len(food2idx))}
    for pet in food2idx.keys():
        random_pet_percentages[pet] = 1.0 / (none_pet_share + len(food2idx))

    return random_pet_percentages

random_pet_percentages = get_random_pet_percentages()
random_food_percentages = get_random_food_percentages()

# Rollout generation:
def generate_teams_from_rollout(number_to_generate, net, config):
    net = net.to(rollout_device)
    net.set_device("rollout")
    net.eval()

    players = [Player() for _ in range(number_to_generate)]
    team_list = []

    # For each turn
    for turn_number in range(config["number_of_rollout_turns"]):
        active_players = [player for player in players]

        for action_number in range(config["action_limit"]):
            if len(active_players) == 0:
                break

            actions_taken = np.array([get_best_legal_move(player, net, config = config, epoch = 25) for player in active_players])

            # For each action
            for player, action in zip(active_players, actions_taken):
                try:
                    if action != N_ACTIONS - 1:
                        call_action_from_q_index(player, action.item(), food_net = net, epsilon = -0.10)
                except:
                    player.gold += 1
                    player.roll()
                    continue

            # Remove players that have pressed end turn
            not_ended_turns = actions_taken != (N_ACTIONS - 1)
            not_ended_turns = not_ended_turns.tolist()
            active_players = np.array(active_players)
            active_players = active_players[not_ended_turns]
            active_players = active_players.tolist()

        for player in players:
            player.end_turn()

        # Adding Teams to team list
        current_teams = [deepcopy(player.team) for player in players]
        team_list.extend(current_teams)

        # Next Turn
        for player in players:
            player.start_turn()

    return team_list

# Random team generation:
def generate_teams_from_random_process(number_to_generate):
    return [random_team() for _ in range(number_to_generate)]

def random_team():
    player = Player()

    # 1) Random pets in the team
    n_team_pets = random.randint(0, 5)
    pet_percentage_list = [random_pet_percentages[pet] for pet in random_pet_percentages.keys()]
    pets_list = [pet.replace("pet-", "") for pet in random_pet_percentages.keys()]
    selected_team_pets = random.choices(pets_list, pet_percentage_list, k = n_team_pets)

    team = Team(selected_team_pets)

    player = Player(team = team)

    # 2) Randomly eat foods
    food_percentage_list = [random_food_percentages[food] for food in random_food_percentages.keys()]
    foods_list = [food for food in random_food_percentages.keys()]
    selected_foods = random.choices(foods_list, food_percentage_list, k = len(player.team.filled))
    for pet_no, food in zip(player.team.filled, selected_foods):
        if food != "food-none" and food != "food-sleeping-pill":
            try:
                player.team.slots[pet_no].pet.eat(Food(food))
            except:
                pass
    for pet_no in player.team.filled:
        for _ in range(random.randint(0, 5)):
            player.team.slots[pet_no].pet.gain_experience()

    # 5) Randomly give pets more stats
    for pet_no in player.team.filled:
        if random.uniform(0, 1) < 0.3:
            player.team.slots[pet_no].pet.set_health(random.randint(player.team.slots[pet_no].pet.health, 50))
            player.team.slots[pet_no].pet.set_attack(random.randint(player.team.slots[pet_no].pet.attack, 50))

    # 6) Randomly choose shop level
    shop_level = random.randint(0, 15)
    for _ in range(shop_level):
        player.end_turn()
    
    player.gold += 1
    player.roll()

    # Randomly choose wins and lives
    player.wins = random.randint(0, 10)
    player.lives = random.randint(0, 5)

    # Check if any pet has negative health or attack
    for pet_no in player.team.filled:
        if player.team.slots[pet_no].pet.health < 0 or player.team.slots[pet_no].pet.attack < 0:
            raise Exception("Negative health or attack")

    return player.team

def battle_increasing_pigs(team : Team, battle_frame : 50, pig_multiplier = None) -> int:
    n_wins = -1

    if isinstance(battle_frame, int):
        battle_frame = [0, battle_frame]
    elif isinstance(battle_frame, list):
        battle_frame = battle_frame

    #player.buy_pet(0)

    if len(team.filled) > 0 and battle_frame[0] == 0:
        n_wins  += 1

    for i in range(max(battle_frame[0], 1), battle_frame[1] + 1):
        pig_team = _create_pig_team(i, pig_multiplier)
        
        try:
            battle = Battle(team, pig_team)
            result = battle.battle()
        except Exception as e:
            #print(e)
            #print(team)
            #print(pig_team)
            result = 1

        if result == 0: # Player won
            n_wins += 1
        elif result == 1: # Player lost
            break

    n_wins += battle_frame[0]
    
    return n_wins
            
def _create_pig_team(stats : int, pig_multiplier = None):
    assert stats > 0 and stats <= 50

    team = Team()
    pig = Pet('pig')
    pig.set_attack(stats)
    pig.set_health(stats)

    for _ in range(5):
        team.append(deepcopy(pig))

    if pig_multiplier is not None:
        assert len(pig_multiplier) == 5
        for i in range(5):
            team.slots[i].pet.set_attack(max(min(int(team.slots[i].pet.attack * pig_multiplier[i]), 50), 1))
            team.slots[i].pet.set_health(max(min(int(team.slots[i].pet.health * pig_multiplier[i]), 50), 1))

    return team

# Generating training sets:
def generate_training_set(teams) -> list:
    training_set = []

    for team_number, team in enumerate(teams):
        if len(team.filled) in [4, 5]:
            heur_order_indexes = auto_order_team(team, return_order_indexes=True, return_team=False)
            order_permutations = generate_order_permutations(team)
            random.shuffle(order_permutations)
            order_permutations = sorted(order_permutations, key=lambda x: ratio(x, heur_order_indexes), reverse=True)
            order_permutations = order_permutations[:24]
        else:
            order_permutations = generate_order_permutations(team)

        order_score_data = []

        worst_wins = 10000
        best_wins = -10000

        n_wins = 0

        for order in order_permutations:
            battle_team = deepcopy(team)
            battle_team = Team([battle_team[x] for x in order], seed_state=battle_team.seed_state)

            if worst_wins > 1000 and best_wins < 0:
                battle_frame = [1, 12]
            elif worst_wins > 1000:
                battle_frame = [max(worst_wins - 2, 1), 12]
            elif best_wins < 0:
                battle_frame = [1, min(best_wins + 2, 12)]
            else:
                battle_frame = [max(worst_wins - 2, 1), min(best_wins + 2, 12)]

            n_wins = battle_increasing_pigs(battle_team, battle_frame, pig_multiplier = [1.5, 1.0, 1.0, 0.5, 0.5])

            if n_wins > best_wins:
                best_wins = n_wins
            if n_wins < worst_wins:
                worst_wins = n_wins

            order_score_data.append((order, n_wins))

        sorted_order_score_data = sorted(order_score_data, key=lambda x: x[1], reverse=True)
        best_orders = []
        for order, n_wins in sorted_order_score_data:
            if n_wins == sorted_order_score_data[0][1]:
                best_orders.append(order)
            else: break

        training_set.append((team, best_orders))
        
        #print(order_score_data)

    return training_set

def generate_training_set_parallel(teams, config) -> list:
    training_set = []

    num_threads = min(os.cpu_count(), config["max_num_threads"])

    # Splitting teams into chunks
    random.shuffle(teams)
    total_team_chunks = [teams[i * len(teams) // num_threads: (i + 1) * len(teams) // num_threads] for i in range(num_threads)]

    lazy_results = []
    for i, curr_chunk in enumerate(total_team_chunks):
        lazy_results.append(dask.delayed(generate_training_set)(deepcopy(curr_chunk)))

    dask_out = dask.compute(lazy_results, scheduler='processes')

    for chunk in dask_out[0]:
        training_set.extend(chunk)
        
    return training_set
            
def generate_order_permutations(team):
    filled_team = [i for i in range(5) if team.slots[i].pet.name != "pet-none"]
    empty_team = [i for i in range(5) if team.slots[i].pet.name == "pet-none"]
    permutation_list = list(itertools.permutations(filled_team))
    permutation_list = [permutation + tuple(empty_team) for permutation in permutation_list]
    return permutation_list

# Training
def update_reorder_model(reorder_model : SAPAIReorder, training_set : list, config : dict):
    reorder_model = deepcopy(reorder_model)
    reorder_model.train()

    optimizer = torch.optim.Adam(reorder_model.parameters(), lr=config["learning_rate"])

    temp_training_set = []
    for team, orders in training_set:
        for order in orders:
            temp_training_set.append((team, order))
    training_set = temp_training_set

    for epoch_num in range(1, config["num_updates_per_sample"] + 1):
        losses = []

        random.shuffle(training_set)
        batches = [training_set[i * config["batch_size"]: (i + 1) * config["batch_size"]] for i in range(len(training_set) // config["batch_size"])]

        for batch in batches:
            optimizer.zero_grad()

            teams = [x[0] for x in batch]
            ground_truth = [x[1] for x in batch]

            # Generating the data
            encoder_encodings = reorder_model.generate_input_encodings(teams)
            decoder_encodings = reorder_model.generate_output_encodings(ground_truth)
            prediction = reorder_model(encoder_encodings, decoder_encodings)

            # Calculating the loss
            loss = reorder_model.loss(prediction, ground_truth)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print("Epoch: {}, Loss: {}".format(epoch_num, np.mean(losses)))
    
    return reorder_model

def evaluate_model(eval_set : list, reorder_model : SAPAIReorder, config : dict):
    ideal_orders_found = 0
    total_loss = 0

    with torch.no_grad():
        for team, orders in eval_set:
            encoder_encodings = reorder_model.generate_input_encodings([team])
            pred_order, pred_logits = reorder_model.inference(encoder_encodings)
            pred_order = tuple(pred_order[0].tolist())

            # Found
            if pred_order not in orders:
                lowest_loss = 1000000.0
                for order in orders:
                    loss = reorder_model.loss(pred_logits, [order]).item()
                    if loss < lowest_loss:
                        lowest_loss = loss
                loss = lowest_loss
            else:
                ideal_orders_found += 1

                best_order = orders.index(pred_order)
                loss = reorder_model.loss(pred_logits, [orders[best_order]]).item()

            total_loss += loss

        print("Loss: {}".format(total_loss / len(eval_set)))
        print("Ideal orders found: {} / {}".format(ideal_orders_found, len(eval_set)))

def train():
    config = REORDER_DEFAULT_CONFIGURATION
    SAVE_DATA = True
    n_players = config["number_of_players"]
    
    policy_model = SAPAI(phase = "training")
    reorder_model = SAPAIReorder(config = config)

    # Loading policy model
    model_path = "model-test_new_food_best.pt"
    if model_path is not None and os.path.exists(model_path):
        loaded_data = torch.load(model_path)
        policy_model.load_state_dict(loaded_data["policy_model"])


    for epoch in range(1, config["epochs"] + 1):
        # Loading the data
        start_time = time.time()
        random_teams = generate_teams_from_random_process(int(n_players * config["number_of_rollout_turns"] * config["random_to_rollout_ratio"]))
        end_time = time.time()
        print("Generate random teams: ", end_time - start_time)

        start_time = time.time()
        rollout_teams = generate_teams_from_rollout(n_players, policy_model, config)
        end_time = time.time()
        print("Rollout teams: ", end_time - start_time)

        # Time
        start_time = time.time()
        training_set = generate_training_set_parallel(rollout_teams + random_teams, config)
        #training_set = generate_training_set(rollout_teams + random_teams)
        end_time = time.time()
        print("Generate training set: ", end_time - start_time)
        print(len(training_set))

        # Training
        start_time = time.time()
        reorder_model = update_reorder_model(reorder_model, training_set, config)

        # Evaluation
        eval_random_teams = generate_teams_from_random_process(75)
        eval_rollout_teams = generate_teams_from_rollout(5, policy_model, config)
        eval_set = generate_training_set_parallel(eval_rollout_teams + eval_random_teams, config = config)

        evaluate_model(eval_set, reorder_model, config)

        if epoch % 3 == 0:
            # Saving the model
            torch.save({
                "reorder_model": reorder_model.state_dict(),
                "config": config
            }, f"reorder-model_{epoch}.pt")

        if SAVE_DATA:
            if not os.path.isdir("reorder_data"):
                os.mkdir("reorder_data")

            path = os.path.join("reorder_data", f"reorder-model-training-data_epoch{epoch}.pkl")

            with open(path, "wb") as f:
                pickle.dump(training_set, f)

        print()


if __name__ == "__main__":
    train()