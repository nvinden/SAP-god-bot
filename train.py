import sys
import os
from copy import deepcopy
import time

current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from model_actions import *
from model import SAPAI
from config import DEFAULT_CONFIGURATION

from sapai.pets import Pet
from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player

import multiprocessing as mp

def train():
    shop = Shop()
    team = Team()
    player = Player(shop, team)

    net = SAPAI(config = DEFAULT_CONFIGURATION)

    print(buy_pet(player, 0))
    print(buy_pet(player, 0))
    print(buy_food(player, 0, 1))
    print(roll(player))

    # Forward pass
    encoding = net.state_to_encoding(player)
    out = net(encoding)

    start_time = time.time()
    for _ in range(100):
        encoding = net.state_to_encoding(player)
    end_time = time.time()

    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")


    '''
    for i in range(42):
    player = Player()
    player.gold = 100

    #print(player.shop, player.team, "gold:", player.gold, "\n")

    code = call_action_from_q_index(player, i)
    print("code:", code)

    print("done")
    '''

    # Function testing
    '''
    buy_pet(player, 0)
    bought_pet = player.team.slots[0]
    player.team.slots[1] = deepcopy(bought_pet)
    player.team.slots[2] = deepcopy(bought_pet)
    
    print(player.shop, player.team, "gold:", player.gold, "\n")
    buy_food(player, 0, 0)
    combine(player, 0)

    freeze(player, 0)
    freeze(player, 0)
    unfreeze(player, 0)

    end_turn(player)

    print(player.shop, player.team, "gold:", player.gold, "\n")
    '''

if __name__ == "__main__":
    train()