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
from itertools import chain
import datetime
import pickle

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed

import pandas as pd


current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from model_actions import *
from model import SAPAI
from config import DEFAULT_CONFIGURATION, rollout_device, training_device, VERBOSE, N_ACTIONS, USE_WANDB
from eval import evaluate_model, visualize_rollout, test_legal_move_masking, get_best_legal_move, create_epoch_illegal_mask
from past_teams import PastTeamsOrganizer

from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player
from sapai.battle import Battle
import math

import wandb

# This process can be ran on multiple threads, it will run many simulations, and return the results
def run_simulation(net : SAPAI, config : dict, epsilon : float, epoch : int, pt_organizer : PastTeamsOrganizer) -> list:
    net = net.to(rollout_device)
    net.set_device("rollout")
    net.eval()

    player_list = np.array([Player() for _ in range(config["players_per_simulation"])])
    local_experience_replay = []
        
    # This stores the actions that the agent can take, it is the first instance in the action list

    #############
    # SHOP TURN #
    #############

    for turn_number in range(config["number_of_battles_per_simulation"]):
        # Run a turn: Completes when all players have ended their turn or
        # when the action limit has reached the action limit

        player_turn_action_replays = defaultdict(list)
        active_players = np.arange(config["players_per_simulation"]) #[i for i in range(config["players_per_simulation"])]

        for action_turn_number in range(config['action_limit']):
            if len(active_players) == 0: break
            active_player_list = player_list[active_players]
            current_state_encodings = net.state_to_encoding(active_player_list)
            i_to_player_number = {i: active_players[i] for i in range(len(active_players))}
            player_number_to_i = {active_players[i]: i for i in range(len(active_players))}

            # 1) Choosing actions:
            # Choosing whether the agent should take a random action or not
            decision_vector = np.random.uniform(0, 1, size=len(active_player_list))
            random_actions = decision_vector < epsilon

            move_actions = np.zeros(shape = len(active_player_list), dtype = np.uint32)
            number_random = np.count_nonzero(random_actions)
            number_q_actions = len(active_player_list) - number_random

            action_mask = np.array([create_available_action_mask(player) for player in active_player_list])
            action_epoch_illegal_moves = create_epoch_illegal_mask(epoch, config)
            action_mask = np.array([mask * action_epoch_illegal_moves for i, mask in enumerate(action_mask)])
            action_mask = action_mask.astype(np.uint32)
            action_mask_indexes = [np.where(mask > 0.5)[0] for mask in action_mask]

            if number_random > 0: # Random Actions:
                # This will first choose a random action type (buy, sell, skip, etc), then
                # randomly choose a sub action of that action, this is to prevent the agent
                # from overprioritizin actions with many sub actions

                # Using Mask
                random_mask_idx = [mask for mask, is_random in zip(action_mask_indexes, random_actions) if is_random]
                
                random_actions_indices = np.array([np.random.choice(mask) for mask in random_mask_idx])

                move_actions[random_actions] = random_actions_indices

            # Max Q Action:
            if number_q_actions > 0:
                Q_actions = np.logical_not(random_actions)
                Q_players = np.array(active_player_list)[Q_actions]
                Q_mask = np.array(action_mask)[Q_actions]
                max_q_actions_indices = [get_best_legal_move(player, net, config = config, epoch = epoch, mask = mask) for player, mask in zip(Q_players, Q_mask)]
                move_actions[Q_actions] = np.array(max_q_actions_indices)

            ended_turns_mask = move_actions == N_ACTIONS - 1

            # 2) Performing actions and getting mid-turn rewards:
            for player_number, player, action in zip(active_players, active_player_list, move_actions):
                if config["allow_stat_increase_as_reward"] and reward is not None:
                    before_action_total_stats = sum([slot.health + slot.attack for slot in player.team.slots if not slot.empty])

                result_string, _, selected_pet = call_action_from_q_index(player, int(action), food_net = net, epsilon = epsilon)

                if selected_pet is not None:
                    selected_pet = pet2idx[selected_pet]

                reward = result_string_to_rewards[result_string]

                next_state_encoding = net.state_to_encoding(player)

                if config["allow_stat_increase_as_reward"] and reward is not None:
                    after_action_total_stats = sum([slot.health + slot.attack for slot in player.team.slots if not slot.empty])

                    increase_stats_reward = max((after_action_total_stats - before_action_total_stats), 0.0)

                    if increase_stats_reward > 1:
                        increase_stats_reward = np.log10(increase_stats_reward) / 5.0
                        reward += increase_stats_reward

                player_turn_action_replays[player_number].append({
                    "state_encoding": current_state_encodings[player_number_to_i[player_number]].cpu().numpy(),
                    "action": action.item(),
                    "reward": reward,
                    "next_state": deepcopy(player),
                    "next_state_encoding": torch.squeeze(next_state_encoding).cpu().numpy(),
                    "food_action": selected_pet,
                    "food_reward": None,
                })

            # 3) Removing players who have ended their turn
            players_who_ended_turn = np.where(ended_turns_mask == True)[0]
            active_players = list(active_players)
            for ended_player_index in players_who_ended_turn: active_players.remove(i_to_player_number[ended_player_index])
            active_players = np.array(active_players)

        # Forcing an end to the turn for all players who havent yet ended their turn
        for player_number in active_players:
            player = player_list[player_number]
            player_temp = deepcopy(player)
            result_string = call_action_from_q_index(player_temp, N_ACTIONS - 1)
            encoding = np.squeeze(net.state_to_encoding(player).cpu().numpy())
            encoding_next_state = np.squeeze(net.state_to_encoding(player).cpu().numpy())

            player_turn_action_replays[player_number].append({
                "state_encoding": deepcopy(encoding),
                "action": N_ACTIONS - 1,
                "reward": None,
                "next_state_encoding": deepcopy(encoding_next_state),
                "next_state": deepcopy(player),
                "food_action": None,
                "food_reward": None,
            })

        ##########
        # BATTLE #
        ##########

        player_end_turn_rewards, player_total_match_results = preform_battles(player_list = player_list, config = config, turn_number = turn_number, pt_organizer = pt_organizer)

        ############
        # UPDATING #
        ############

        # Updating players lives and wins
        for i, (player, results) in enumerate(zip(player_list, player_total_match_results)):
            if results['wins'] > results['losses']: # WIN
                player.wins += 1
            elif results['losses'] > results['wins']: #LOSS
                player.lives -= 1

            player_end_turn_rewards[i] /= config["number_of_battles_per_player_turn"]
            
            if player.lives <= 0: # Adding round loss
                player_end_turn_rewards[i] += result_string_to_rewards["game_loss"]
            if player.wins >= 10: # Adding round win
                player_end_turn_rewards[i] += result_string_to_rewards["game_win"]


            if VERBOSE:
                print(f"Player {i:02d} has {player.wins:02d} wins and {player.lives} lives remaining")
                print(f"\tFightR: {player_end_turn_rewards[i]}")
                print(f"\t{results['wins']:02d} wins, {results['losses']:02d} losses, {results['draws']:02d} draws")
                team_string = str(player.team).replace('\n', ' ').strip()
                team_string = re.sub(r'\s{2,}', ' ', team_string)
                print(f"\tPlayer team: {team_string}")
        if VERBOSE: print()

        # Update the player_turn_action_replays
        for player_number in player_turn_action_replays:
            freeze_count = 0
            unfreeze_count = 0
            for action_replay in player_turn_action_replays[player_number]:
                if action_replay["reward"] is None:
                    action_replay["reward"] = player_end_turn_rewards[player_number]

                    if config["allow_penalty_for_unused_gold"]:
                        action_replay["reward"] -= 0.1 * action_replay["next_state"].gold

                    # Adding the food reward (same as battle reward) for each food action that is not None
                    for action_replay_food in player_turn_action_replays[player_number]:
                        if action_replay_food["food_action"] is not None:
                            action_replay_food["food_reward"] = player_end_turn_rewards[player_number]

                if config['allow_multi_freeze_unfreeze_penalty']:
                    if action_replay["action"] >= action_beginning_index[action2index["freeze"]] and action_replay["action"] < action_beginning_index[action2index["freeze"]] + num_agent_actions["freeze"]:
                        if freeze_count >= 3:
                            action_replay["reward"] -= 0.1 * freeze_count
                        freeze_count += 1
                    if action_replay["action"] >= action_beginning_index[action2index["unfreeze"]] and action_replay["action"] < action_beginning_index[action2index["unfreeze"]] + num_agent_actions["unfreeze"]:
                        if unfreeze_count >= 3:
                            action_replay["reward"] -= 0.1 * unfreeze_count
                        unfreeze_count += 1

                if config['allow_combine_reward'] and action_replay["action"] >= action_beginning_index[action2index["combine"]] and action_replay["action"] < action_beginning_index[action2index["combine"]] + num_agent_actions["combine"]:
                    action_replay["reward"] += 1.0

                if not config["allow_negative_rewards"]:
                    action_replay["reward"] = max(0, action_replay["reward"])
            
            local_experience_replay.extend(deepcopy(player_turn_action_replays[player_number]))

        # Starting the next turn for each player
        for player_number, player in enumerate(player_list):
            won_last_round = player_total_match_results[player_number]["wins"] > player_total_match_results[player_number]["losses"]
            player.start_turn(winner = won_last_round)

    return local_experience_replay, deepcopy(pt_organizer)

def preform_battles(player_list, config, turn_number, pt_organizer : PastTeamsOrganizer): 
    # Randomly pair players together, and battle them, recording the results
    player_end_turn_rewards = [0 for _ in range(config["players_per_simulation"])]
    player_total_match_results = [defaultdict(int) for _ in range(config["players_per_simulation"])]
    
    is_teams_on_deck = pt_organizer.is_active_teams()

    if is_teams_on_deck:
        # Auto ordering team before battle
        for index, player in enumerate(player_list):
            auto_order_team(player)

            for opponent_team in pt_organizer.sample(config["number_of_battles_per_player_turn"], turn_number):
                winner = pt_organizer.battle(deepcopy(player.team), opponent_team)

                # Record the results
                if winner == 0: # Player won
                    player_end_turn_rewards[index] += result_string_to_rewards["round_win"]
                    player_total_match_results[index]["wins"] += 1
                elif winner == 1: # Opponent won
                    player_end_turn_rewards[index] += result_string_to_rewards["round_loss"]
                    player_total_match_results[index]["losses"] += 1
                else: # Draw
                    player_total_match_results[index]["draws"] += 1

                #print(opponent_team)

    else:
        # BATTLING EMPTY TEAMS
        for i, player in enumerate(player_list):
            if len(player.team.filled) > 0: # Player won
                player_end_turn_rewards[i] += result_string_to_rewards["round_win"] * config["number_of_battles_per_player_turn"]
            else: # Opponent won or draw
                player_end_turn_rewards[i] += result_string_to_rewards["round_loss"] * config["number_of_battles_per_player_turn"]

    # Adding teams to the PastTeamsOrganizer
    for player in player_list:
        pt_organizer.add_on_deck(deepcopy(player.team), turn_number = turn_number)

    return player_end_turn_rewards, player_total_match_results

def run_simulation_multiprocess(net : SAPAI, config : dict, epsilon : float, epoch : int, pt_organizer : PastTeamsOrganizer) -> tuple[list, PastTeamsOrganizer]:
    num_threads = min(os.cpu_count(), config["max_num_threads"])

    # Creating sub pt_organizers that only have 10 teams each, and are randomly sampled from the main pt_organizer
    sub_pt_organizers = [pt_organizer.create_random_samples_clone(max_past_teams = 10) for _ in range(num_threads)]
    
    lazy_results = []
    for i in range(num_threads):
        lazy_results.append(dask.delayed(run_simulation)(deepcopy(net), deepcopy(config), epsilon, epoch, sub_pt_organizers[i]))

    dask_out = dask.compute(lazy_results, scheduler='processes')
    
    pt_organizer_out = deepcopy(pt_organizer)

    # Adding up all the organizers together
    for organizer in dask_out[0]:
        for i in range(len(organizer[1].active_past_teams)):
            pt_organizer_out.active_past_teams[i].extend(organizer[1].active_past_teams[i])
        for i in range(len(organizer[1].on_deck_past_teams)):
            pt_organizer_out.on_deck_past_teams[i].extend(organizer[1].on_deck_past_teams[i])

    # Concatenating list of lists into a single list
    results_out = []
    for result in dask_out[0]:
        results_out.extend(result[0])
    random.shuffle(results_out)
        
    return results_out, deepcopy(pt_organizer_out)

def update_prediction_weights(policy_net : SAPAI, target_net : SAPAI, experience_replay : list, config : dict, epoch : int):
    policy_net = policy_net.to(training_device)
    policy_net.set_device("training")
    policy_net.train()

    target_net = target_net.to(training_device)
    target_net.set_device("training")
    target_net.train()

    random.shuffle(experience_replay)

    # putting policy net into training mode and target net into eval mode
    policy_net = deepcopy(policy_net)
    policy_net.train()
    target_net.train()

    # Freeze the target net
    for param in target_net.parameters():
        param.requires_grad = False

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=config["learning_rate"])

    losses = []
    food_losses = []

    num_updates = int(math.ceil(int(len(experience_replay) / config["batch_size"] * config["num_updates_per_sample"]) / float(config["batch_size"])) * config["batch_size"])

    number_food_updates_in_last_300 = 0

    #Update the prediction weights
    for update_number in range(1, num_updates + 1):
        optimizer.zero_grad()
        
        # Randomly samples from experience replay
        batches = random.sample(experience_replay, config["batch_size"])
        batches = {k: [dic[k] for dic in batches] for k in batches[0]}
        batches['state_encoding'] = torch.tensor(np.stack(batches['state_encoding'])).to(training_device)
        batches['next_state_encoding'] = torch.tensor(np.stack(batches['next_state_encoding'])).to(training_device)
        batches['action'] = np.array(batches["action"], dtype = np.int32)
        batches['reward'] = torch.tensor(batches['reward']).to(training_device).to(torch.float)

        predictions = policy_net(batches['state_encoding'])
        predictions = predictions[torch.arange(predictions.size(0)), batches['action']]

        target = target_net(batches['next_state_encoding'])

        # Creating legal action mask the best target next value
        action_mask = np.array([create_available_action_mask(player) for player in batches['next_state']])
        epoch_illegal_mask = create_epoch_illegal_mask(epoch, config)
        action_mask = action_mask * epoch_illegal_mask
        where_zeros = action_mask < 0.5
        target[where_zeros] = -9999999

        target_values, _ = torch.max(target, dim = 1)
        target_values = batches['reward'] + config['gamma'] * target_values
        loss = criterion(predictions, target_values)
        losses.append(loss.item())

        # Making food predictions and targets
        food_mask = np.array([True if food_action is not None else False for food_action in batches["food_action"]])
        if food_mask.sum() > 1:
            food_encodings = batches["state_encoding"][food_mask]
            food_actions = np.array(batches["food_action"])[food_mask].astype(np.int32)
            food_rewards = torch.tensor(np.array(batches["food_reward"]).astype(np.float32)[food_mask], dtype = torch.float).to(training_device)
            actions = batches["action"][food_mask]
            food_target = target[food_mask]

            food_name_list = [idx2food[idx - action_beginning_index[action2index["buy_food"]]] for idx in actions]
            food_encodings = policy_net.add_food_index_to_encoding(food_encodings, food_name_list)
            _, food_predictions = policy_net(food_encodings, return_food_actions = True)
            food_predictions = food_predictions[torch.arange(food_predictions.size(0)), food_actions]

            food_target_values, _ = torch.max(food_target, dim = 1)
            #food_target_values = food_rewards + config['gamma'] * food_target_values
            food_target_values = config['gamma'] * food_target_values # Error correction with always 0 food
            
            
            # For the normal prediction
            curr_loss = criterion(food_predictions, food_target_values)
            food_losses.append(curr_loss.item())
            loss += curr_loss

            number_food_updates_in_last_300 += 1

        loss.backward()
        optimizer.step()

        del loss

        if update_number % 300 == 0:
            average_loss_past = np.mean(losses[-300:])
            average_food_loss_past = np.mean(food_losses[-number_food_updates_in_last_300:])
            total_average_loss = average_loss_past + average_food_loss_past
            print(f"Step {update_number:04d}: Total Loss {total_average_loss}, Normal Loss: {average_loss_past}, Average Food Loss: {average_food_loss_past}")

            number_food_updates_in_last_300 = 0

    return policy_net, np.mean(losses)

# Running with a loop
def run_with_loop(net, config, epsilon, epoch : int, pt_organizer : PastTeamsOrganizer):
    experience_replay = []
    start_time = time.time()
    #for i in range(config["max_num_threads"]):
    for i in range(1):
        experience_replay_out, pt_organizer_out = run_simulation(deepcopy(net), config, epsilon, epoch, pt_organizer)
        experience_replay.extend(experience_replay_out)
        print(f"Run: {i}")
    elapsed_time = time.time() - start_time
    print(f"Elapsed time (loop): {elapsed_time} seconds")
    print(f"Experience replay length: {len(experience_replay)}")
    print("Number of threads: ", config["max_num_threads"])
    print("Number of players per thread: ", config["players_per_simulation"])
    print(rollout_device)
    return experience_replay, pt_organizer_out, elapsed_time

def run_with_processes(net, config, epsilon, epoch : int, pt_organizer : PastTeamsOrganizer):
    start_time = time.time()
    experience_replay, pt_organizer = run_simulation_multiprocess(deepcopy(net), config, epsilon, epoch, pt_organizer)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time (process): {elapsed_time} seconds")
    print(f"Experience replay length: {len(experience_replay)}")
    print("Number of threads: ", config["max_num_threads"])
    print("Number of players per thread: ", config["players_per_simulation"])
    print(rollout_device)
    return experience_replay, pt_organizer, elapsed_time

def train():
    config = DEFAULT_CONFIGURATION

    # Get the current date and time
    if USE_WANDB:
        wandb_id = wandb.util.generate_id()
        run_name = "run_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        #wandb.init(resume = "must", id = "ysxidhkr")
        wandb.init(project = "sap-god-bot", name = "Pretrain Food, Freeze Penalty, Food Fix", id = wandb_id)
        wandb.config.update(config, allow_val_change = True)
        print("WandB ID: ", wandb_id)

    policy_net = SAPAI(config = config)
    target_net = deepcopy(policy_net)

    greedy_pt_organizer = PastTeamsOrganizer(config, max_past_teams=75)
    epsilon_pt_organizer = PastTeamsOrganizer(config, max_past_teams=75)

    starting_epoch = 1

    # Random action hyperparameters
    epsilon = config["epsilon"]
    epsilon_min = config["epsilon_min"]
    epsilon_decay = config["epsilon_decay"]

    # Loading runs
    train_mode = "none" # "pretrained", "scratch", "continue", "none"
    train_path = "model_test_156.pt"

    if train_mode == "continue":
        loaded_values = torch.load(train_path)
        if "policy_model" in loaded_values:
            policy_net.load_state_dict(loaded_values["policy_model"])
        if "target_model" in loaded_values:
            target_net.load_state_dict(loaded_values["target_model"])

        starting_epoch = loaded_values["epoch"]
        epsilon = loaded_values["epsilon"]
        epsilon_pt_organizer = loaded_values["epsilon_pt_organizer"]
        greedy_pt_organizer = loaded_values["greedy_pt_organizer"]

        epsilon_pt_organizer.set_new_maxlen(75)
        greedy_pt_organizer.set_new_maxlen(75)
    elif train_mode == "pretrained":
        checkpoint = torch.load(train_path)
        policy_net.load_state_dict(checkpoint["model"])
        target_net.load_state_dict(checkpoint["model"])
    elif train_mode == "continue_just_model":
        loaded_model = torch.load(train_path)
        policy_net.load_state_dict(loaded_model)
        target_net.load_state_dict(loaded_model)

    # Delete "food_placed.txt" and "teams.txt" if they exist
    if os.path.exists("food_placed.txt"):
        os.remove("food_placed.txt")
    if os.path.exists("teams.txt"):
        os.remove("teams.txt")

    if USE_WANDB:
        wandb.watch(policy_net)

    #eval_results, actions_used = evaluate_model(policy_net, config, epoch = 0)
    #visualize_rollout(policy_net, config)
    #eval_results, actions_used, past_team_win_percentages, pets_bought, food_bought = evaluate_model(policy_net, greedy_pt_organizer, config = config, epoch = 2)

    for epoch in range(starting_epoch, config["epochs"] + 1):
        print("Epoch: ", epoch)
        print("Epsilon: ", epsilon)
    
        # Running with multithreading
        #experience_replay, epsilon_pt_organizer, er_elapsed_time = run_with_loop(policy_net, config, epsilon, epoch, epsilon_pt_organizer)
        experience_replay, epsilon_pt_organizer, er_elapsed_time = run_with_processes(policy_net, config, epsilon, epoch, epsilon_pt_organizer)

        start_time = time.time()
        policy_net, avg_loss = update_prediction_weights(policy_net, target_net, experience_replay, config, epoch = epoch)
        update_elapsed_time = time.time() - start_time
        print(f"Average loss: {avg_loss}")
        print(f"Elapsed time: {update_elapsed_time} seconds")

        start_time = time.time()
        eval_results, actions_used, past_team_win_percentages, pets_bought, food_bought = evaluate_model(policy_net, greedy_pt_organizer, config = config, epoch = epoch)
        print(eval_results)
        print({k : sum(v) for k, v in actions_used.items()})
        evaluate_elapsed_time = time.time() - start_time
        print(f"Elapsed time: {evaluate_elapsed_time} seconds")

        # If the model beats the past opponents more on average change up the previous models
        if past_team_win_percentages is not None:
            avg_win_percentage = np.mean(past_team_win_percentages)
            print("Average win percentage: ", avg_win_percentage)
            if avg_win_percentage > config["win_percentage_threshold_past_teams"]:
                print(f"Updating past teams with win percentage: {avg_win_percentage}")
                greedy_pt_organizer.update_teams()
                epsilon_pt_organizer.update_teams()

            # Saving model
            save_dict = {
                "policy_model": policy_net.state_dict(),
                "target_model": target_net.state_dict(),
                "epoch": epoch,
                "epsilon": epsilon,
                "epsilon_pt_organizer": epsilon_pt_organizer,
                "greedy_pt_organizer": greedy_pt_organizer
            }
            torch.save(save_dict, f"model_test_{epoch}.pt")
            print("Model saved")

            if USE_WANDB:
                wandb.save(f"model_test_{epoch}.pt")

        if epoch % 5 == 0:
            target_net = deepcopy(policy_net)
            print("TARGET NET MODEL WEIGHTS UPDATED ON EPOCH: ", epoch)
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        print()

        if USE_WANDB:
            win_percentage = np.mean(past_team_win_percentages) if past_team_win_percentages is not None else None
            if win_percentage is not None:
                beat_past_opponent = win_percentage > config["win_percentage_threshold_past_teams"]
            else:
                beat_past_opponent = None
                
            wandb_logging_dict = {
                "epoch": epoch,
                "epsilon": epsilon,
                "avg_loss": avg_loss,
                "avg_win_percentage": win_percentage,
                "beat_past_opponent": beat_past_opponent,
                "avg_win_percentage_per_round": past_team_win_percentages,
                "sum_pigs_defeated": np.sum(eval_results),
                "max_pigs_defeated": np.max(eval_results),
                "experience_replay_length": len(experience_replay),
                "experience_replay_time_taken": er_elapsed_time,
                "evaluate_time_taken": evaluate_elapsed_time,
                "update_time_taken": update_elapsed_time,
            }

            pig_defeated_dict = {"pigs_defeated_round_" + str(i + 1): eval_results[i] for i in range(len(eval_results))}
            wandb_logging_dict["pigs_defeated_by_round"] = pig_defeated_dict

            # Adding pets and food bought dictionaries
            wandb_logging_dict["actions_used"] = {k : sum(v) for k, v in actions_used.items()}
            wandb_logging_dict["pets_bought"] = {k : sum(v) for k, v in pets_bought.items()}
            wandb_logging_dict["food_bought"] = {k : sum(v) for k, v in food_bought.items()}

            wandb.log(wandb_logging_dict)

            # Tables
            wandb.log({"pets_bought_table": wandb.Table(dataframe = pd.DataFrame(pets_bought))})
            wandb.log({"food_bought_table": wandb.Table(dataframe = pd.DataFrame(food_bought))})
            wandb.log({"actions_used_table": wandb.Table(dataframe = pd.DataFrame(actions_used))})

if __name__ == "__main__":
    print(f"Cuda available: {torch.cuda.is_available()}", flush = True)
    print(f"Rollout device used: {rollout_device}", flush = True)
    print(f"Training device used: {training_device}\n", flush = True)

    train()