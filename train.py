import sys
import os
from copy import deepcopy
import time

import torch
import multiprocessing as mp
import numpy as np
from collections import defaultdict
from itertools import combinations, chain
import concurrent.futures as cf

import dask
from dask.distributed import Client
from dask import delayed


current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from model_actions import *
from model import SAPAI, N_ACTIONS
from config import DEFAULT_CONFIGURATION

from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player
from sapai.battle import Battle

def run_simulation2(net : SAPAI, config : dict, epsilon : float) -> list:
    encodings = [Player() for _ in range(config["players_per_simulation"])]
    encodings = net.state_to_encoding(encodings)
    actions, _ = net(encodings)
    return [np.argmax(actions.detach().numpy())]

# This process can be ran on multiple threads, it will run many simulations, and return the results
def run_simulation(net : SAPAI, config : dict, epsilon : float) -> list:

    player_list = [Player() for _ in range(config["players_per_simulation"])]
    local_experience_replay = []

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
                random_actions_indices = np.random.randint(0, N_ACTIONS, size = number_random)
                move_actions[random_actions] = random_actions_indices

            # Max Q Action:
            if number_q_actions > 0:
                max_q_action_vector, _ = net(current_state_encodings[np.logical_not(random_actions), :])
                max_q_actions_indices = torch.argmax(max_q_action_vector, dim = 1)
                move_actions[np.logical_not(random_actions)] = max_q_actions_indices

            ended_turns_mask = move_actions == N_ACTIONS - 1

            # 2) Performing actions and getting mid-turn rewards:
            for player_number, player, action in zip(active_players, active_player_list, move_actions):
                current_state = deepcopy(player)
                result_string = call_action_from_q_index(player, int(action))
                reward = result_string_to_rewards[result_string]
                next_state_encoding = net.state_to_encoding(player)
                next_state = deepcopy(player)

                player_turn_action_replays[player_number].append({
                    "state": current_state,
                    "state_encoding": i_to_player_number[player_number_to_i[player_number]],
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "next_state_encoding": next_state_encoding,
                })

            # 3) Removing players who have ended their turn
            players_who_ended_turn = np.where(ended_turns_mask == True)[0]
            for ended_player_index in players_who_ended_turn:
                active_players.remove(i_to_player_number[ended_player_index])

        ##########
        # BATTLE #
        ##########
        
        # Randomly pair players together, and battle them, recording the results
        player_total_wins = [0 for _ in range(config["players_per_simulation"])]
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
                battle = Battle(player_list[index].team, player_list[opponent].team)
                winner = battle.battle()

                # Record the results
                if winner == 0: # Player 1 won
                    player_total_wins[index] += result_string_to_rewards["round_win"]
                    player_total_wins[opponent] += result_string_to_rewards["round_loss"]
                elif winner == 1: # Player 2 won
                    player_total_wins[index] += result_string_to_rewards["round_win"]
                    player_total_wins[opponent] += result_string_to_rewards["round_loss"]

                # Checking if player has died
                if player_list[index].lives <= 0:
                    player_total_wins[index] += result_string_to_rewards["game_loss"]
                if player_list[opponent].lives <= 0:
                    player_total_wins[opponent] += result_string_to_rewards["game_loss"]

                # Checking if player has won
                if player_list[index].wins >= 10:
                    player_total_wins[index] += result_string_to_rewards["game_win"]
                if player_list[opponent].wins >= 10:
                    player_total_wins[opponent] += result_string_to_rewards["game_win"]


                # Update the remaining battles
                total_battles_remaining[index] -= 1
                total_battles_remaining[opponent] -= 1
                if total_battles_remaining[index] == 0:
                    players_with_remaining_battles.remove(index)
                if total_battles_remaining[opponent] == 0:
                    players_with_remaining_battles.remove(opponent)

                sum_total_battles_remaining -= 2
            
            index = (index + 1) % len(player_list)

        # Update the player_turn_action_replays
        for player_number in player_turn_action_replays:
            for action_replay in player_turn_action_replays[player_number]:
                action_replay["reward"] += player_total_wins[player_number]
            
            local_experience_replay.extend(deepcopy(player_turn_action_replays[player_number]))

    return local_experience_replay

def run_simulation_on_all_threads(net : SAPAI, config : dict, epsilon : float) -> list:
    num_threads = min(os.cpu_count(), config["max_num_threads"])

    args = [net, config, epsilon]

    # Create a ThreadPoolExecutor with the maximum available threads
    with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for _ in range(num_threads):
            future = executor.submit(run_simulation, *args)
            futures.append(future)

        master_experience_replay = []

        # Wait for the threads to complete and retrieve the results
        for future in cf.as_completed(futures):
            result = future.result()
            master_experience_replay.extend(result)


    # Do something with the master list
    return master_experience_replay

def run_simulation_on_all_threads_dask(net : SAPAI, config : dict, epsilon : float) -> list:
    num_threads = min(os.cpu_count(), config["max_num_threads"])

    lazy_results = []
    for _ in range(num_threads):
        lazy_results.append(dask.delayed(run_simulation)(deepcopy(net), deepcopy(config), epsilon))

    results = dask.compute(lazy_results, scheduler='processes')

    # Concatenating list of lists into a single list
    results = list(chain.from_iterable(results[0]))
    
    return results

def train():
    shop = Shop()
    team = Team()
    player = Player(shop, team)

    config = DEFAULT_CONFIGURATION

    net = SAPAI(config = config)

    # Random action hyperparameters
    epsilon = config["epsilon"]
    epsilon_decay = config["epsilon_decay"]
    epsilon_min = config["epsilon_min"]

    # Running with multithreading
    '''
    start_time = time.time()
    experience_replay = run_simulation_on_all_threads_dask(deepcopy(net), config, epsilon)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time (threaded): {elapsed_time} seconds")
    print(len(experience_replay))
    '''

    # Running with a loop
    experience_replay = []
    start_time = time.time()
    for i in range(4):
        experience_replay.extend(run_simulation(deepcopy(net), config, epsilon))
    print(f"Elapsed time (loop): {elapsed_time * 6} seconds")
    print(len(experience_replay))


if __name__ == "__main__":
    train()