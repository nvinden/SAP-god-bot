# This file contains the functions that convert the limited actions that the
# nvinden agents can preform, and does those actions in the SAP api

import sys
import os

current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from sapai.pets import Pet
from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player, GoldException, WrongObjectException, FullTeamException

# This dictonary contains all the actions that the agent can take, and
# the total number of variations of the action
agent_actions = {
    "roll": 1,
    "buy_pet": 6,
    "sell": 5,
    "buy_food": 10,
    "combine": 5,
    "freeze": 7,
    "unfreeze": 7,
    "next_turn": 1
}

failure_codes = {"not_enough_gold" : 1, "not_enough_space": 1, "sold_empty_slot": 1}

# Rolling the shop
def roll(shop : Shop) -> None:
    shop.roll()

# Buys a pet from the shop, adds it to the players band
# TODO instead translate the pet idx not from shop indexes, but to the
# indexes of the PETS in the shop
def buy_pet(player : Player, pet_idx : int) -> str:
    try:
        player.buy_pet(pet_idx)
        return "success"
    except GoldException as e:
        return "not_enough_gold"
    except FullTeamException as e:
        # Try to buy upgrade if there is no space
        print(player.shop, player.team, "gold:", player.gold, "\n")
        for team_idx in range(5):
            try:
                player.buy_combine(pet_idx, team_idx)
                return "success"
            except Exception as e:
                continue 
        return "not_enough_space"

def sell(player : Player, pet_idx : int) -> str:
    try:
        player.sell(pet_idx)
        return "success"
    except WrongObjectException as e:
        return "sold_empty_slot"
    
# TODO instead translate the pet idx not from shop indexes, but to the
# indexes of the FOOOD in the shop
def buy_food(player : Player, pet_idx : int, food_idx : int) -> str:
    try:
        player.buy_food(pet_idx, food_idx)
        return "success"
    except GoldException as e:
        return "not_enough_gold"