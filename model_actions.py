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
agent_actions_list = ["roll", "buy_pet", "sell", "buy_food", "combine", "freeze", "unfreeze", "end_turn"]

agent_actions = {
    "roll": 1,
    "buy_pet": 6,
    "sell": 5,
    "buy_food": 15,
    "combine": 5,
    "freeze": 7,
    "unfreeze": 7,
    "end_turn": 1
}

failure_codes = {"not_enough_gold" : 1, "not_enough_space": 1, "sold_empty_slot": 1, "invalid_pet_idx": 1, "no_combinable_pets": 1}

result_string_to_rewards = {
    "success": 0.0,
    "not_enough_gold": -1.0,
    "not_enough_space": -1.0,
    "sold_empty_slot": -1.0,
    "invalid_pet_idx": -1.0,
    "no_combinable_pets": -1.0,
    "invalid_idx": -1.0,
    "round_win": 1.0,
    "round_loss": -1.0,
    "game_win": 5.0,
    "game_loss": -5.0,
}

# Rolling the shop
def roll(shop : Shop) -> None:
    try:
        shop.roll()
    except GoldException:
        return "not_enough_gold"
    return "success"

# Buys a pet from the shop, adds it to the players band
# TODO instead translate the pet idx not from shop indexes, but to the
# indexes of the PETS in the shop
def buy_pet(player : Player, pet_idx : int) -> str:
    assert pet_idx >= 0 and pet_idx < 6

    number_pets = len(player.shop.pets)
    #number_food = len(player.shop.foods)

    if pet_idx >= number_pets:
        return "invalid_idx"

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
    assert pet_idx >= 0 and pet_idx < 5

    try:
        player.sell(pet_idx)
        return "success"
    except WrongObjectException as e:
        return "sold_empty_slot"
    
# TODO instead translate the pet idx not from shop indexes, but to the
# indexes of the FOOOD in the shop
def buy_food(player : Player, food_idx : int, pet_idx : int) -> str:
    assert food_idx >= 0 and food_idx < 3
    assert pet_idx >= 0 and pet_idx < 5

    number_shop_pets = len(player.shop.pets)
    number_food = len(player.shop.foods)

    # Invalid food number
    if food_idx >= number_food:
        return "invalid_idx"
    
    if pet_idx not in player.team.filled:
        return "invalid_idx"
    
    try:
        player.buy_food(food_idx + number_shop_pets, pet_idx)
        return "success"
    except GoldException as e:
        return "not_enough_gold"

# This function takes a pet in the band, and then combines it with a pet in the
# same band, only if able. If it does not combine with any pet, it fails.
def combine(player : Player, pet_idx : int) -> str:
    assert pet_idx >= 0 and pet_idx < 5

    success = False

    if pet_idx not in player.team.filled:
        return "invalid_idx"
    
    pet_species = player.team.slots[pet_idx].pet.name

    for combine_pet_idx in player.team.filled:
        if pet_idx == combine_pet_idx:
            continue
        if player.team.slots[combine_pet_idx].pet.name == pet_species:
            try:
                player.combine(pet_idx, combine_pet_idx)
                success = True
            except Exception as e:
                continue
    
    if success: return "success"
    else: return "no_combinable_pets"

def freeze(player : Player, shop_idx : int) -> str:
    assert shop_idx >= 0 and shop_idx < 7

    # If slot is not in shop
    if shop_idx >= len(player.shop.slots):
        return "invalid_idx"
    
    # If slot is not filled
    if shop_idx not in player.shop.filled:
        return "invalid_idx"

    # If slot is already frozen
    if player.shop.slots[shop_idx].frozen:
        return "invalid_idx"

    try:
        player.freeze(shop_idx)
        return "success"
    except Exception as e:
        return "invalid_idx"
    
def unfreeze(player : Player, shop_idx : int) -> str:
    assert shop_idx >= 0 and shop_idx < 7

    # If slot is not in shop
    if shop_idx >= len(player.shop.slots):
        return "invalid_idx"

    # If slot is not filled 
    if shop_idx not in player.shop.filled:
        return "invalid_idx"

    # if slot is not frozen
    if not player.shop.slots[shop_idx].frozen:
        return "invalid_idx"

    try:
        player.unfreeze(shop_idx)
        return "success"
    except Exception as e:
        return "invalid_idx"

def end_turn(player : Player) -> str:
    player.end_turn()
    return "success"

###################
# Model Functions #
###################

def call_action_from_q_index(player : Player, q_idx : int) -> str:
    # This code is gross, but it just defines the ranges the actions take
    ranges = [sum([agent_actions[agent_actions_list[j]] for j in range(i + 1)]) for i in range(len(agent_actions_list))]

    assert q_idx >= 0 and q_idx < ranges[-1]

    if q_idx == 0: # Roll
        #print("Rolling")
        return roll(player)
    elif q_idx >= ranges[0] and q_idx < ranges[1]: # Buy pet
        #print("Buying pet")
        player_index = q_idx - ranges[0]
        return buy_pet(player, player_index)
    elif q_idx >= ranges[1] and q_idx < ranges[2]: # Sell pet
        #print("Selling pet")
        player_index = q_idx - ranges[1]
        return sell(player, q_idx - 7)
    elif q_idx >= ranges[2] and q_idx < ranges[3]: # Buy food
        #print("Buying food")
        player_index = (q_idx - ranges[2]) % 5
        food_index = (q_idx - ranges[2]) // 5
        return buy_food(player, food_index, player_index)
    elif q_idx >= ranges[3] and q_idx < ranges[4]: # Combine
        #print("Combining")
        player_index = q_idx - ranges[3]
        return combine(player, player_index)
    elif q_idx >= ranges[4] and q_idx < ranges[5]: # Freeze
        #print("Freezing")
        player_index = q_idx - ranges[4]
        return freeze(player, player_index)
    elif q_idx >= ranges[5] and q_idx < ranges[6]: # Unfreeze
        #print("Unfreezing")
        player_index = q_idx - ranges[5]
        return unfreeze(player, player_index)
    elif q_idx >= ranges[6] and q_idx < ranges[7]: # End turn
        #print("Ending turn")
        return end_turn(player)
    
    return "something_went_wrong"

def auto_order_team(player : Player) -> Player:
    pass