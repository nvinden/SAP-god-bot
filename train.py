import sys
import os
from copy import deepcopy
import time

import torch
import multiprocessing as mp
import numpy as np

current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from model_actions import *
from model import SAPAI
from config import DEFAULT_CONFIGURATION

from sapai.pets import Pet
from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player

# This process can be ran on multiple threads, it will run many simulations, and return the results
def run_simulation(net : SAPAI, config : dict) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    player_list = [Player() for _ in range(config["players_per_simulation"])]
    player_rewards = [0 for _ in range(config["players_per_simulation"])]
    player_rollout_action_replays = [[] for _ in range(config["players_per_simulation"])]


    # Run a turn: Completes when all players have ended their turn or
    # when the action limit has reached the action limit
    player_turn_action_replays = [[] for _ in range(config["players_per_simulation"])]
    for action_number in range(config['action_limit']):
        current_state_encodings = net.state_to_encoding(player_list)
        actions, v = net(current_state_encodings)

        # Choosing whether the agent should take a random action or not
        decision_vector = np.random.uniform(0, 1, size=32)

        pass


    return local_experience_replay

def train():
    shop = Shop()
    team = Team()
    player = Player(shop, team)

    config = DEFAULT_CONFIGURATION

    net = SAPAI(config = config)

    # Run the simulation
    run_simulation(deepcopy(net), config)

if __name__ == "__main__":
    train()