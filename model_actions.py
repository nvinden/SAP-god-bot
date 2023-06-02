# This file contains the functions that convert the limited actions that the
# nvinden agents can preform, and does those actions in the SAP api

import sys
import os
import random

current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from sapai.pets import Pet
from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player, GoldException, WrongObjectException, FullTeamException
from config import rollout_device, training_device

import numpy as np

from copy import deepcopy
import torch

# This dictonary contains all the actions that the agent can take, and
# the total number of variations of the action
agent_actions_list = ["roll", "buy_pet", "sell", "buy_food", "combine", "freeze", "unfreeze", "end_turn"]

num_agent_actions = {
    "roll": 1,
    "buy_pet": 6,
    "sell": 5,
    "buy_food": 15,
    "combine": 5,
    "freeze": 7,
    "unfreeze": 7,
    "end_turn": 1
}

result_string_to_rewards = {
    "success": 0.0,
    "not_enough_gold": -0.2,
    "not_enough_space": -0.2,
    "sold_empty_slot": -0.2,
    "invalid_pet_idx": -0.2,
    "no_combinable_pets": -0.2,
    "invalid_idx": -0.2,
    "round_win": 1.0,
    "round_loss": -1.0,
    "game_win": 1.0,
    "game_loss": -1.0,
    "end_turn": None
}

all_items_idx = {
    "turn_number": 0,
    "player_lives_remaining": 1,
    "current_gold": 2,
    "wins": 3,
    "attack": 4,
    "health": 5,
    "cost": 6,
    "level": 7,
}


rule_breaking_actions = ["not_enough_gold", "not_enough_space", "sold_empty_slot", "invalid_pet_idx", "no_combinable_pets", "invalid_idx"]

# This contains the bias for the pet auto order
# Higher numbers means that pets are more likely to be in the front
# Lower numbers means that pets are more likely to be in the back
pet_auto_order_bias = {
    'pet-ant': 2,
    'pet-beaver': 0,
    'pet-cricket': 1,
    'pet-duck': 0,
    'pet-fish': 0,
    'pet-horse': -3,
    'pet-mosquito': 0,
    'pet-otter': 0,
    'pet-pig': 0,
    'pet-sloth': 0,
    'pet-crab': 0,
    'pet-dodo': -1,
    'pet-dog': -1,
    'pet-elephant': -3,
    'pet-flamingo': 2,
    'pet-hedgehog': -2,
    'pet-peacock': 0,
    'pet-rat': 0,
    'pet-shrimp': 0,
    'pet-spider': 1,
    'pet-swan': 0,
    'pet-badger': -5,
    'pet-blowfish': 2,
    'pet-camel': 0,
    'pet-giraffe': -1,
    'pet-kangaroo': -1,
    'pet-ox': -2,
    'pet-rabbit': 0,
    'pet-sheep': -1,
    'pet-snail': 0,
    'pet-turtle': 3,
    'pet-whale': -1,
    'pet-bison': 0,
    'pet-deer': 3,
    'pet-dolphin': 0,
    'pet-hippo': 4,
    'pet-monkey': -1,
    'pet-penguin': 0,
    'pet-rooster': 0,
    'pet-skunk': 0,
    'pet-squirrel': 0,
    'pet-worm': 0,
    'pet-cow': 0,
    'pet-crocodile': 0,
    'pet-parrot': 0,
    'pet-rhino': 0,
    'pet-scorpion': 0,
    'pet-seal': 0,
    'pet-shark': -6,
    'pet-turkey': 0,
    'pet-cat': 0,
    'pet-boar': 0,
    'pet-dragon': 0,
    'pet-fly': -5,
    'pet-gorilla': 0,
    'pet-leopard': 0,
    'pet-mammoth': 6,
    'pet-snake': -1,
    'pet-tiger': 0,
    'pet-zombie-cricket': 0,
    'pet-bus': 0,
    'pet-zombie-fly': 0,
    'pet-dirty-rat': 0,
    'pet-chick': 0,
    'pet-ram': 0,
    'pet-bee': 0
}

action_beginning_index_temp = np.array([num_agent_actions[action_name] for action_name in agent_actions_list])
action_beginning_index = []
for i in range(0, len(action_beginning_index_temp)):
    action_beginning_index.append(action_beginning_index_temp[:i].sum())
del action_beginning_index_temp

action_ranges = [sum([num_agent_actions[agent_actions_list[j]] for j in range(i + 1)]) for i in range(len(agent_actions_list))]

action2index = {agent_actions_list[i]: i for i in range(len(agent_actions_list))}
index2action = {i: agent_actions_list[i] for i in range(len(agent_actions_list))}

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
        # Try to buy upgrade if there is no spaagent_actions_listce
        #print(player.shop, player.team, "gold:", player.gold, "\n")
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
    
    # if slot is frozen
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
    return "end_turn"

###################
# Model Functions #
###################

def call_action_from_q_index(player : Player, q_idx : int) -> str:
    # This code is gross, but it just defines the ranges the actions take
    if not isinstance(q_idx, int):
        q_idx = int(q_idx)

    assert q_idx >= 0 and q_idx < action_ranges[-1]

    try:
        if q_idx == 0: # Roll
            #print("Rolling")
            return roll(player)
        elif q_idx >= action_ranges[0] and q_idx < action_ranges[1]: # Buy pet
            #print("Buying pet")
            player_index = q_idx - action_ranges[0]
            return buy_pet(player, player_index)
        elif q_idx >= action_ranges[1] and q_idx < action_ranges[2]: # Sell pet
            #print("Selling pet")
            player_index = q_idx - action_ranges[1]
            return sell(player, q_idx - 7)
        elif q_idx >= action_ranges[2] and q_idx < action_ranges[3]: # Buy food
            #print("Buying food")
            player_index = (q_idx - action_ranges[2]) % 5
            food_index = (q_idx - action_ranges[2]) // 5
            return buy_food(player, food_index, player_index)
        elif q_idx >= action_ranges[3] and q_idx < action_ranges[4]: # Combine
            #print("Combining")
            player_index = q_idx - action_ranges[3]
            return combine(player, player_index)
        elif q_idx >= action_ranges[4] and q_idx < action_ranges[5]: # Freeze
            #print("Freezing")
            player_index = q_idx - action_ranges[4]
            return freeze(player, player_index)
        elif q_idx >= action_ranges[5] and q_idx < action_ranges[6]: # Unfreeze
            #print("Unfreezing")
            player_index = q_idx - action_ranges[5]
            return unfreeze(player, player_index)
        elif q_idx >= action_ranges[6] and q_idx < action_ranges[7]: # End turn
            #print("Ending turn")
            return end_turn(player)
    except Exception as e:
        print("Something went wrong in call_action_from_q_index: ", e)
        player.gold += 1
        player.roll()
    
    return "something_went_wrong"

def auto_order_team(player : Player) -> Player:
    # Orders pets from 1. Bias, 2. Total stats (attack + health)

    value_pairs = [(pet_idx, player.team.slots[pet_idx]) for pet_idx in player.team.filled]
    value_pairs = sorted(value_pairs, key = lambda x : (pet_auto_order_bias[x[1].pet.name], x[1].attack + x[1].health), reverse = True)

    player.reorder([x[0] for x in value_pairs])

def action_index_to_action_type(action_number : int) -> str:
    for i in range(len(agent_actions_list) - 1):
        if action_number >= action_beginning_index[i] and action_number < action_beginning_index[i + 1]:
            return agent_actions_list[i]
        
    return agent_actions_list[-1]

def create_available_action_mask(player : Player) -> np.ndarray:
    action_mask = np.ones(shape = (action_beginning_index[-1] + 1), dtype = np.uint8)

    # TODO: fix buying in the case there is a food discount

    # GOLD SETTINGS
    if player.gold == 0:
        action_mask[action_beginning_index[action2index["roll"]]:action_beginning_index[action2index["roll"] + 1]] = 0

    if player.gold < 3:
        action_mask[action_beginning_index[action2index["buy_pet"]]:action_beginning_index[action2index["buy_pet"] + 1]] = 0
    else:
        # BUY PET
        pet_spaces_shop = np.array([i for i, filled in enumerate(player.shop.slots) if filled.slot_type == "pet"])

        # 1) check if there is a pet that can be bought
        no_pet_in_action_idx = np.array(list({0, 1, 2, 3, 4, 5}.difference(pet_spaces_shop))) + action_beginning_index[action2index["buy_pet"]]
        if len(no_pet_in_action_idx): action_mask[no_pet_in_action_idx] = 0

        if len(player.team.filled) == 5: # If there is no available slot
            # 2) Check if there is a pet that can be combined-bought, if team is full
            combinable_list_team = [slot.pet.name for slot in player.team.slots if slot.pet.level < 3]
            not_combinable_mask = np.array([i for i, slot in enumerate(player.shop.slots) if slot.obj.name not in combinable_list_team and slot.slot_type == "pet"]) + action_beginning_index[action2index["buy_pet"]]
            if len(not_combinable_mask): action_mask[not_combinable_mask] = 0

    # BUY FOOD
    number_of_food = len(player.shop.foods)
    indexes_of_foods = [i for i, slot in enumerate(player.shop.slots) if slot.slot_type == "food"]
    unfoodable_mask = np.array([i for i in range(num_agent_actions["buy_food"]) if not (i // 5 < number_of_food and i % 5 in player.team.filled and player.shop.slots[indexes_of_foods[i // 5]].cost <= player.gold)]) + action_beginning_index[action2index["buy_food"]]
    if len(unfoodable_mask) != 0: action_mask[unfoodable_mask] = 0


    # SELL PET
    unsellable_pet_mask = np.array([i for i in player.team.empty]) + action_beginning_index[action2index["sell"]]
    if len(unsellable_pet_mask) != 0: action_mask[unsellable_pet_mask] = 0

    # COMBINE
    pet_names_shop = [slot.pet.name for slot in player.team.slots if slot.pet.name != "pet-none"]
    uncombinable_mask = np.array([i for i, slot in enumerate(player.team.slots) if pet_names_shop.count(slot.pet.name) == 1 or slot.pet.name == "pet-none"]) + action_beginning_index[action2index["combine"]]
    if len(uncombinable_mask) != 0: action_mask[uncombinable_mask] = 0 #TODO: CHECK IF THE PETS CAN ACTUALLY BE COMBINED WITH THEIR LEVELS AND SUCH

    # FREEZE AND UNFREEZE
    unfreezeable_mask = np.array([i for i in range(num_agent_actions["freeze"]) if i >= len(player.shop.slots) or player.shop.slots[i].frozen]) + action_beginning_index[action2index["freeze"]]
    ununfreezeable_mask = np.array([i for i in range(num_agent_actions["unfreeze"]) if i >= len(player.shop.slots) or not player.shop.slots[i].frozen]) + action_beginning_index[action2index["unfreeze"]]

    if len(unfreezeable_mask) != 0: action_mask[unfreezeable_mask] = 0
    if len(ununfreezeable_mask) != 0: action_mask[ununfreezeable_mask] = 0

    return action_mask