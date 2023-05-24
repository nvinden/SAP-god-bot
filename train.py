import sys
import os

current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from model_actions import *

from sapai.pets import Pet
from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player

import multiprocessing as mp

def train():
    shop = Shop()
    team = Team()

    player = Player()
    player.gold = 100

    print(player.shop, player.team, "gold:", player.gold, "\n")

    # Function testing
    sell(player, 0)

    print(player.shop, player.team, "gold:", player.gold, "\n")

if __name__ == "__main__":
    train()