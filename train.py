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
import pickle

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed


current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from model_actions import *
from model import SAPAI, N_ACTIONS
from config import DEFAULT_CONFIGURATION, rollout_device, training_device
from eval import evaluate_model

from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player
from sapai.battle import Battle

VERBOSE = False

# This process can be ran on multiple threads, it will run many simulations, and return the results
def run_simulation(net : SAPAI, config : dict, epsilon : float) -> list:
    net = net.to(rollout_device)
    net.set_device("rollout")
    net.eval()

    player_list = [Player() for _ in range(config["players_per_simulation"])]
    local_experience_replay = []
    # This stores the actions that the agent can take, it is the first instance in the action list
    action_beginning_index_temp = np.array([num_agent_actions[action_name] for action_name in agent_actions_list])
    action_beginning_index = []
    for i in range(0, len(action_beginning_index_temp)):
        action_beginning_index.append(action_beginning_index_temp[:i].sum())
    del action_beginning_index_temp

    #############
    # SHOP TURN #
    #############

    for turn_number in range(config["number_of_battles_per_simulation"]):
        # Run a turn: Completes when all players have ended their turn or
        # when the action limit has reached the action limit

        player_turn_action_replays = defaultdict(list)
        active_players = [i for i in range(config["players_per_simulation"])]

        for action_turn_number in range(config['action_limit']):
            active_player_list = [player_list[i] for i in active_players]
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

            if number_random > 0: # Random Actions:
                # This will first choose a random action type (buy, sell, skip, etc), then
                # randomly choose a sub action of that action, this is to prevent the agent
                # from overprioritizin actions with many sub actions
                action_types = np.random.randint(0, len(agent_actions_list), size = number_random)
                random_actions_indices = [np.random.randint(0, num_agent_actions[agent_actions_list[action_idx]]) for action_idx in action_types]
                random_actions_indices = [val + action_beginning_index[action_idx] for val, action_idx in zip(random_actions_indices, action_types)]
                move_actions[random_actions] = random_actions_indices

            # Max Q Action:
            if number_q_actions > 0:
                max_q_action_vector, _ = net(current_state_encodings[np.logical_not(random_actions), :])
                max_q_actions_indices = torch.argmax(max_q_action_vector, dim = 1)
                move_actions[np.logical_not(random_actions)] = max_q_actions_indices.cpu()

            ended_turns_mask = move_actions == N_ACTIONS - 1

            # 2) Performing actions and getting mid-turn rewards:
            for player_number, player, action in zip(active_players, active_player_list, move_actions):
                #current_state = deepcopy(player)
                result_string = call_action_from_q_index(player, int(action))
                reward = result_string_to_rewards[result_string] * config["players_per_simulation"]
                next_state_encoding = net.state_to_encoding(player)
                #next_state = deepcopy(player)

                player_turn_action_replays[player_number].append({
                    #"state": current_state,
                    "state_encoding": current_state_encodings[player_number_to_i[player_number]],
                    "action": action,
                    "reward": reward,
                    #"next_state": next_state,
                    "next_state_encoding": torch.squeeze(next_state_encoding),
                })

            # 3) Removing players who have ended their turn
            players_who_ended_turn = np.where(ended_turns_mask == True)[0]
            for ended_player_index in players_who_ended_turn:
                active_players.remove(i_to_player_number[ended_player_index])

        ##########
        # BATTLE #
        ##########
        
        # Randomly pair players together, and battle them, recording the results
        player_total_rewards = [0 for _ in range(config["players_per_simulation"])]
        player_total_match_results = [defaultdict(int) for _ in range(config["players_per_simulation"])]
        players_with_remaining_battles = {i for i in range(config["players_per_simulation"])}
        total_battles_remaining = [config["number_of_battles_per_player_turn"] for _ in range(len(player_list))]
        sum_total_battles_remaining = sum(total_battles_remaining)
        
        # Auto ordering team before battle
        for player in player_list:
            auto_order_team(player)

        index = 0
        while sum_total_battles_remaining > 0:
            if index not in players_with_remaining_battles:
                index = (index + 1) % len(player_list)
                continue

            # Randomly pair players together
            possible_opponents = list(players_with_remaining_battles.difference({index}))
            np.random.shuffle(possible_opponents)
            opponents = sorted(possible_opponents, key = lambda x: total_battles_remaining[x])[:config["number_of_battles_per_player_turn"]]

            for opponent in opponents:
                # Battle the players
                player_team = player_list[index].team
                opponent_team = player_list[opponent].team
                battle = Battle(player_team, opponent_team)
                winner = battle.battle()

                # Record the results
                if winner == 0: # Player won
                    player_total_rewards[index] += result_string_to_rewards["round_win"]
                    player_total_rewards[opponent] += result_string_to_rewards["round_loss"]
                    player_total_match_results[index]["wins"] += 1
                    player_total_match_results[opponent]["losses"] += 1
                elif winner == 1: # Opponent won
                    player_total_rewards[index] += result_string_to_rewards["round_loss"]
                    player_total_rewards[opponent] += result_string_to_rewards["round_win"]
                    player_total_match_results[index]["losses"] += 1
                    player_total_match_results[opponent]["wins"] += 1
                else: # Draw
                    player_total_match_results[index]["draws"] += 1
                    player_total_match_results[opponent]["draws"] += 1

                # Update the remaining battles
                total_battles_remaining[index] -= 1
                total_battles_remaining[opponent] -= 1
                if total_battles_remaining[index] == 0:
                    players_with_remaining_battles.remove(index)
                if total_battles_remaining[opponent] == 0:
                    players_with_remaining_battles.remove(opponent)

                sum_total_battles_remaining -= 2
            
            index = (index + 1) % len(player_list)

        
        # Updating players lives and wins
        for i, (player, results) in enumerate(zip(player_list, player_total_match_results)):
            if results['wins'] > results['losses']: # WIN
                player.wins += 1
            elif results['losses'] > results['wins']: #LOSS
                player.lives -= 1
            
            if player.lives <= 0: # Adding round loss
                player_total_rewards[i] += result_string_to_rewards["game_loss"]
            if player.wins >= 10: # Adding round win
                player_total_rewards[i] += result_string_to_rewards["game_win"]

            if VERBOSE:
                print(f"Player {i:02d} has {player.wins:02d} wins and {player.lives} lives remaining")
                print(f"\t{player_total_rewards[i] / config['number_of_battles_per_player_turn']} total rewards")
                print(f"\t{results['wins']:02d} wins, {results['losses']:02d} losses, {results['draws']:02d} draws")
                team_string = str(player.team).replace('\n', ' ').strip()
                team_string = re.sub(r'\s{2,}', ' ', team_string)
                print(f"\tPlayer team: {team_string}")
        if VERBOSE: print()

        # Update the player_turn_action_replays
        for player_number in player_turn_action_replays:
            for action_replay in player_turn_action_replays[player_number]:
                action_replay["reward"] += player_total_rewards[player_number]
                action_replay["reward"] /= config["number_of_battles_per_player_turn"]
            
            local_experience_replay.extend(deepcopy(player_turn_action_replays[player_number]))

    return local_experience_replay

def run_simulation_multiprocess(net : SAPAI, config : dict, epsilon : float) -> list:
    num_threads = min(os.cpu_count(), config["max_num_threads"])

    lazy_results = []
    for _ in range(num_threads):
        lazy_results.append(dask.delayed(run_simulation)(deepcopy(net), deepcopy(config), epsilon))
        time.sleep(0.1)

    results = dask.compute(lazy_results, scheduler='processes')

    # Concatenating list of lists into a single list
    results = list(chain.from_iterable(results[0]))
    random.shuffle(results)
        
    return results

def update_prediction_weights(policy_net : SAPAI, target_net : SAPAI, experience_replay : list, config : dict):
    policy_net = policy_net.to(training_device)
    policy_net.set_device("training")
    policy_net.train()

    target_net = target_net.to(training_device)
    target_net.set_device("training")
    target_net.eval()

    random.shuffle(experience_replay)

    # putting policy net into training mode and target net into eval mode
    policy_net = deepcopy(policy_net)
    policy_net.train()
    target_net.eval()

    # Freeze the target net
    for param in target_net.parameters():
        param.requires_grad = False

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=config["learning_rate"])

    losses = []

    # Update the prediction weights
    for update_number in range(1, config['number_of_updates_per_optimization_step'] + 1):
        optimizer.zero_grad()
        
        # Randomly samples from experience replay
        batches = random.sample(experience_replay, config["batch_size"])
        batches = {k: [dic[k] for dic in batches] for k in batches[0]}
        batches['state_encoding'] = torch.stack(batches['state_encoding']).to(training_device)
        batches['next_state_encoding'] = torch.stack(batches['next_state_encoding']).to(training_device)
        batches['action'] = np.array(batches["action"], dtype = np.int32)
        batches['reward'] = torch.tensor(batches['reward']).to(training_device)

        predictions, _ = policy_net(batches['state_encoding'])
        predictions = predictions[torch.arange(predictions.size(0)), batches['action']]

        target, _ = policy_net(batches['next_state_encoding'])
        target, _ = torch.max(target, dim = 1)
        target = batches['reward'] + config['gamma'] * target

        loss = criterion(predictions, target)
        loss.backward()

        losses.append(loss.item())
        optimizer.step()

        del loss

        if update_number % 50 == 0:
            average_loss_past_50 = np.mean(losses[-50:])
            print(f"Step {update_number:04d}: Loss {average_loss_past_50}")

    return policy_net

# Running with a loop
def run_with_loop(net, config, epsilon):
    experience_replay = []
    start_time = time.time()
    for _ in range(config["max_num_threads"]):
        experience_replay.extend(run_simulation(deepcopy(net), config, epsilon))
    elapsed_time = time.time() - start_time
    print(f"Elapsed time (loop): {elapsed_time} seconds")
    print(f"Experience replay length: {len(experience_replay)}")
    print("Number of threads: ", config["max_num_threads"])
    print("Number of players per thread: ", config["players_per_simulation"])
    print(rollout_device)
    return experience_replay

def run_with_processes(net, config, epsilon):
    start_time = time.time()
    experience_replay = run_simulation_multiprocess(deepcopy(net), config, epsilon)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time (process): {elapsed_time} seconds")
    print(f"Experience replay length: {len(experience_replay)}")
    print("Number of threads: ", config["max_num_threads"])
    print("Number of players per thread: ", config["players_per_simulation"])
    print(rollout_device)
    return experience_replay

def train():
    shop = Shop()
    team = Team()
    player = Player(shop, team)

    config = DEFAULT_CONFIGURATION

    policy_net = SAPAI(config = config)
    target_net = deepcopy(policy_net)

    # Random action hyperparameters
    epsilon = config["epsilon"]
    epsilon_min = config["epsilon_min"]
    epsilon_decay = config["epsilon_decay"]

    for epoch in range(config["epochs"]):
    
        # Running with multithreading
        experience_replay = run_with_loop(policy_net, config, epsilon)
        #experience_replay = run_with_processes(policy_net, config, epsilon)

        # Saving experience replay
        #pickle.dump(experience_replay, open("experience_replay_test_32_players.pkl", "wb"))
        #pickle.dump(experience_replay[:5000], open("experience_replay_test_32_players_small.pkl", "wb"))


        # Loading experience replay
        #experience_replay_large = pickle.load(open("experience_replay_test.pkl", "rb"))
        #experience_replay = pickle.load(open("experience_replay_test_32_players.pkl", "rb"))

        policy_net = update_prediction_weights(policy_net, target_net, experience_replay, config)

        start_time = time.time()
        eval_results = evaluate_model(policy_net, config)
        print(eval_results)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time (loop): {elapsed_time} seconds")

        if epoch % 2 == 1:
            target_net = deepcopy(policy_net)
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Saving model
        torch.save(policy_net.state_dict(), f"model_test_{epoch}.pt")


if __name__ == "__main__":
    print(f"Cuda available: {torch.cuda.is_available()}", flush = True)
    print(f"Rollout device used: {rollout_device}", flush = True)
    print(f"Training device used: {training_device}", flush = True)
    train()