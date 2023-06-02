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
from model import SAPAI
from config import DEFAULT_CONFIGURATION, rollout_device, training_device, VERBOSE, N_ACTIONS
from eval import evaluate_model, visualize_rollout, test_legal_move_masking, get_best_legal_move, create_epoch_illegal_mask

from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player
from sapai.battle import Battle

# This process can be ran on multiple threads, it will run many simulations, and return the results
def run_simulation(net : SAPAI, config : dict, epsilon : float, epoch : int) -> list:
    net = net.to(rollout_device)
    net.set_device("rollout")
    net.eval()

    player_list = [Player() for _ in range(config["players_per_simulation"])]
    local_experience_replay = []
    # This stores the actions that the agent can take, it is the first instance in the action list

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

            action_mask = np.array([create_available_action_mask(player) for player in active_player_list])
            action_epoch_illegal_moves = create_epoch_illegal_mask(epoch, config)
            action_mask = np.array([mask * action_epoch_illegal_moves for i, mask in enumerate(action_mask)])
            action_mask = action_mask.astype(np.uint32)
            action_mask_indexes = [np.where(mask == 1)[0] for mask in action_mask]

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
                #current_state = deepcopy(player)
                before_action_player = deepcopy(player)
                after_action_player = deepcopy(player)
                result_string = call_action_from_q_index(after_action_player, int(action))

                reward = result_string_to_rewards[result_string]

                if config["allow_stat_increase_as_reward"] and reward is not None:
                    before_action_total_stats = sum([slot.health + slot.attack for slot in before_action_player.team.slots if not slot.empty])
                    after_action_total_stats = sum([slot.health + slot.attack for slot in after_action_player.team.slots if not slot.empty])

                    increase_stats_reward = max((after_action_total_stats - before_action_total_stats), 0.0)

                    if increase_stats_reward > 1:
                        increase_stats_reward = np.log10(increase_stats_reward) / 2
                        reward += increase_stats_reward
                
                next_state_encoding = net.state_to_encoding(after_action_player)
                #next_state = deepcopy(player)

                player_turn_action_replays[player_number].append({
                    #"state": current_state,
                    "state_encoding": current_state_encodings[player_number_to_i[player_number]].cpu().numpy(),
                    "action": action.item(),
                    "reward": reward,
                    #"next_state": next_state,
                    "next_state_encoding": torch.squeeze(next_state_encoding).cpu().numpy(),
                })

                call_action_from_q_index(player, int(action))

            # 3) Removing players who have ended their turn
            players_who_ended_turn = np.where(ended_turns_mask == True)[0]
            for ended_player_index in players_who_ended_turn:
                active_players.remove(i_to_player_number[ended_player_index])

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
            })

        ##########
        # BATTLE #
        ##########

        player_end_turn_rewards, player_total_match_results = preform_battles(player_list, player_turn_action_replays, net, config, epoch = epoch)

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
            for action_replay in player_turn_action_replays[player_number]:
                if action_replay["reward"] is None:
                    action_replay["reward"] = player_end_turn_rewards[player_number]

                # If epoch is smaller than a certain amount, reward buying pets and selling pets
                #if epoch < 10:
                    #if action_replay["action"] >= action_beginning_index[action2index["buy_pet"]] and action_replay["action"] < action_beginning_index[action2index["buy_pet"]] + num_agent_actions["buy_pet"]:
                        #action_replay["reward"] += 0.03
                #    elif action_replay["action"] >= action_beginning_index[action2index["sell"]] and action_replay["action"] < action_beginning_index[action2index["sell"]] + num_agent_actions["sell"]:
                #        action_replay["reward"] += 0.1

                if action_replay["action"] >= action_beginning_index[action2index["combine"]] and action_replay["action"] < action_beginning_index[action2index["combine"]] + num_agent_actions["combine"]:
                        action_replay["reward"] += 1.0

                if not config["allow_negative_rewards"]:
                    action_replay["reward"] = max(0, action_replay["reward"])
            
            local_experience_replay.extend(deepcopy(player_turn_action_replays[player_number]))

        # Starting the next turn for each player
        for player in player_list:
            player.start_turn()

    return local_experience_replay

def preform_battles(player_list, player_turn_action_replays, net, config, epoch):
    # Randomly pair players together, and battle them, recording the results
    player_end_turn_rewards = [0 for _ in range(config["players_per_simulation"])]
    player_total_match_results = [defaultdict(int) for _ in range(config["players_per_simulation"])]
    players_with_remaining_battles = {i for i in range(config["players_per_simulation"])}
    total_battles_remaining = [config["number_of_battles_per_player_turn"] for _ in range(len(player_list))]
    sum_total_battles_remaining = sum(total_battles_remaining)
    
    # Auto ordering team before battle
    for player in player_list:
        auto_order_team(player)

    # Preforming battles
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
            pl_temp = deepcopy(player_team)
            pl_temp.move_forward()
            op_temp = deepcopy(opponent_team)
            op_temp.move_forward()

            try:
                battle = Battle(pl_temp, op_temp)
                winner = battle.battle()
            except Exception as e:
                #print(e, flush = True)
                #print(pl_temp, flush = True)
                #print(op_temp, flush = True)
                winner = 0

            # Record the results
            if winner == 0: # Player won
                player_end_turn_rewards[index] += result_string_to_rewards["round_win"]
                player_end_turn_rewards[opponent] += result_string_to_rewards["round_loss"]
                player_total_match_results[index]["wins"] += 1
                player_total_match_results[opponent]["losses"] += 1
            elif winner == 1: # Opponent won
                player_end_turn_rewards[index] += result_string_to_rewards["round_loss"]
                player_end_turn_rewards[opponent] += result_string_to_rewards["round_win"]
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

    # BATTLING EMPTY TEAMS IF EPOCH ALLOWS IT
    for i, player in enumerate(player_list):
        if len(player.team.filled) > 0: # Player won
            player_end_turn_rewards[i] += result_string_to_rewards["round_win"] / 2.0 * config["number_of_battles_per_player_turn"]
        else: # Opponent won or draw
            player_end_turn_rewards[i] += result_string_to_rewards["round_loss"] / 2.5 * config["number_of_battles_per_player_turn"]

    # BATTLING A TEAM WITH ONLY ONE PET IF EPOCH ALLOWS IT

    return player_end_turn_rewards, player_total_match_results

def run_simulation_multiprocess(net : SAPAI, config : dict, epsilon : float, epoch : int) -> list:
    num_threads = min(os.cpu_count(), config["max_num_threads"])

    lazy_results = []
    for _ in range(num_threads):
        lazy_results.append(dask.delayed(run_simulation)(deepcopy(net), deepcopy(config), epsilon, epoch))
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
        batches['state_encoding'] = torch.tensor(np.stack(batches['state_encoding'])).to(training_device)
        batches['next_state_encoding'] = torch.tensor(np.stack(batches['next_state_encoding'])).to(training_device)
        batches['action'] = np.array(batches["action"], dtype = np.int32)
        batches['reward'] = torch.tensor(batches['reward']).to(training_device).to(torch.float)

        predictions, _ = policy_net(batches['state_encoding'])
        predictions = predictions[torch.arange(predictions.size(0)), batches['action']]

        target, _ = target_net(batches['next_state_encoding'])
        target, _ = torch.max(target, dim = 1)
        target = batches['reward'] + config['gamma'] * target

        loss = criterion(predictions, target)
        loss.backward()

        losses.append(loss.item())
        optimizer.step()

        del loss

        if update_number % 300 == 0:
            average_loss_past = np.mean(losses[-300:])
            print(f"Step {update_number:04d}: Loss {average_loss_past}")

    return policy_net

# Running with a loop
def run_with_loop(net, config, epsilon, epoch : int):
    experience_replay = []
    start_time = time.time()
    #for i in range(config["max_num_threads"]):
    for i in range(1):
        experience_replay.extend(run_simulation(deepcopy(net), config, epsilon, epoch))
        print(f"Run: {i}")
    elapsed_time = time.time() - start_time
    print(f"Elapsed time (loop): {elapsed_time} seconds")
    print(f"Experience replay length: {len(experience_replay)}")
    print("Number of threads: ", config["max_num_threads"])
    print("Number of players per thread: ", config["players_per_simulation"])
    print(rollout_device)
    return experience_replay

def run_with_processes(net, config, epsilon, epoch : int):
    start_time = time.time()
    experience_replay = run_simulation_multiprocess(deepcopy(net), config, epsilon, epoch)
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

    # Loading weights
    #policy_net.load_state_dict(torch.load("model_test_1.pt"))
    #eval_results, actions_used = evaluate_model(policy_net, config, epoch = 0)
    #visualize_rollout(policy_net, config)

    # Random action hyperparameters
    epsilon = config["epsilon"]
    epsilon_min = config["epsilon_min"]
    epsilon_decay = config["epsilon_decay"]

    for epoch in range(1, config["epochs"] + 1):
        print("Epoch: ", epoch)
        print("Epsilon: ", epsilon)
    
        # Running with multithreading
        #experience_replay = run_with_loop(policy_net, config, epsilon, epoch)
        experience_replay = run_with_processes(policy_net, config, epsilon, epoch)

        # Saving experience replay
        #pickle.dump(experience_replay, open("experience_replay_test_32_players.pkl", "wb"))
        #pickle.dump(experience_replay[:5000], open("experience_replay_test_32_players_small.pkl", "wb"))


        # Loading experience replay
        #experience_replay_large = pickle.load(open("experience_replay_test.pkl", "rb"))
        #experience_replay = pickle.load(open("experience_replay_test_32_players.pkl", "rb"))

        policy_net = update_prediction_weights(policy_net, target_net, experience_replay, config)

        start_time = time.time()
        eval_results, actions_used = evaluate_model(policy_net, config, epoch = epoch)
        print(eval_results)
        print(actions_used)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time (loop): {elapsed_time} seconds")

        if epoch % 5 == 0:
            target_net = deepcopy(policy_net)
            print("TARGET NET MODEL WEIGHTS UPDATED ON EPOCH: ", epoch)
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Saving model
        torch.save(policy_net.state_dict(), f"model_test_{epoch}.pt")
        print("Model saved")

        print()

if __name__ == "__main__":
    print(f"Cuda available: {torch.cuda.is_available()}", flush = True)
    print(f"Rollout device used: {rollout_device}", flush = True)
    print(f"Training device used: {training_device}\n", flush = True)

    #test_legal_move_masking()

    train()