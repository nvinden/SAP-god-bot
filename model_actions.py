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
from sapai.data import data as ALL_DATA
from config import rollout_device, training_device

import numpy as np

from copy import deepcopy
import torch
from collections import defaultdict

# This dictonary contains all the actions that the agent can take, and
# the total number of variations of the action
agent_actions_list = ["roll", "buy_pet", "sell", "buy_food", "combine", "freeze", "unfreeze", "end_turn"]

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


# Pet food index numbers:
pet_name_idx = [k for k, v in ALL_DATA['pets'].items() if k != 'pet-none' and "StandardPack" in v['packs']]
pet_name_idx = {k : i + len(list(all_items_idx)) for i, k in enumerate(pet_name_idx)}
number_of_pets = len(pet_name_idx)

pet_food_idx = [k for k, v in ALL_DATA['foods'].items() if k != 'food-none' and "StandardPack" in v['packs']]
pet_food_idx = {k : i + number_of_pets + len(list(all_items_idx)) for i, k in enumerate(pet_food_idx)}
number_of_foods = len(pet_food_idx)

pet_status_idx = [k for k, v in ALL_DATA['statuses'].items() if k != 'status-none']
pet_status_idx = {k : i + number_of_pets + number_of_foods + len(list(all_items_idx)) for i, k in enumerate(pet_status_idx)}
number_of_statuses = len(pet_status_idx)

pet2idx = [k for k, v in ALL_DATA['pets'].items() if k != 'pet-none' and "StandardPack" in v['packs']]
pet2idx = {k : i for i, k in enumerate(pet_name_idx)}
idx2pet = {i : k for i, k in enumerate(pet_name_idx)}

food2idx = [k for k, v in ALL_DATA['foods'].items() if k != 'food-none' and "StandardPack" in v['packs']]
food2idx = {k : i for i, k in enumerate(pet_food_idx)}
idx2food = {i : k for i, k in enumerate(pet_food_idx)}

# All items together
all_items_idx.update(pet_name_idx)
all_items_idx.update(pet_food_idx)
all_items_idx.update(pet_status_idx)
number_of_items = len(all_items_idx)

# status to food, and food to status
status_to_food = {
    'status-honey-bee':'food-honey',
    'status-bone-attack':'food-meat-bone',
    'status-garlic-armor':'food-garlic',
    'status-splash-attack':'food-chili',
    'status-melon-armor':'food-melon',
    'status-extra-life':'food-mushroom',
    'status-steak-attack':'food-steak',
}

food_to_status = {v : k for k, v in status_to_food.items()}

num_agent_actions = {
    "roll": 1,
    "buy_pet": number_of_pets,
    "sell": number_of_pets,
    "buy_food": number_of_foods,
    "combine": number_of_pets,
    "freeze": number_of_pets + number_of_foods,
    "unfreeze": number_of_pets + number_of_foods,
    "end_turn": 1
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
    except Exception as e:
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

    if q_idx == 0: # Roll
        #print("Rolling")
        return roll(player)
    elif q_idx >= action_ranges[0] and q_idx < action_ranges[1]: # Buy pet
        #print("Buying pet")
        pet_list_index = q_idx - action_ranges[0]
        pet_to_buy = idx2pet[pet_list_index]
        pet_shop_list = [pet.name for pet in player.shop.pets if pet.name != "pet-none"]
        
        if pet_to_buy not in pet_shop_list:
            return "invalid_idx"
        
        buy_idx = pet_shop_list.index(pet_to_buy)

        return buy_pet(player, buy_idx)
    elif q_idx >= action_ranges[1] and q_idx < action_ranges[2]: # Sell pet
        #print("Selling pet")
        pet_list_index = q_idx - action_ranges[1]
        pet_to_sell = idx2pet[pet_list_index]
        pet_team_list = [slot.pet.name for slot in player.team.slots]

        if pet_to_sell not in pet_team_list:
            return "invalid_idx"
        
        sell_idx = pet_team_list.index(pet_to_sell)

        return sell(player, sell_idx)
    elif q_idx >= action_ranges[2] and q_idx < action_ranges[3]: # Buy food
        if player.team.filled == []:
            return "invalid_idx"
        
        #print("Buying food")
        food_list_index = q_idx - action_ranges[2]
        food_to_buy = idx2food[food_list_index]
        food_shop_list = [food.name for food in player.shop.foods if food.name != "food-none"]

        if food_to_buy not in food_shop_list:
            return "invalid_idx"
        
        food_buy_idx = food_shop_list.index(food_to_buy)
        pet_index = auto_assign_food_to_pet(food_to_buy, player)

        return buy_food(player, food_buy_idx, pet_index)
    elif q_idx >= action_ranges[3] and q_idx < action_ranges[4]: # Combine
        #print("Combining")
        pet_list_index = q_idx - action_ranges[3]
        pet_to_combine = idx2pet[pet_list_index]
        pet_team_list = [slot.pet.name for slot in player.team.slots]

        if pet_team_list.count(pet_to_combine) < 2:
            return "invalid_idx"
        
        combine_idx = pet_team_list.index(pet_to_combine)

        return combine(player, combine_idx)
    elif q_idx >= action_ranges[4] and q_idx < action_ranges[5]: # Freeze
        #print("Freezing")
        freezable_index = q_idx - action_ranges[4]
        pet_food_to_freeze = idx2pet[freezable_index] if freezable_index < len(idx2pet) else idx2food[freezable_index - len(idx2pet)]
        freeze_idx = -1
        for i, slot in enumerate(player.shop.slots):
            if not slot.frozen and slot.obj.name == pet_food_to_freeze:
                freeze_idx = i
                break

        if freeze_idx == -1:
            return "invalid_idx"
        
        return freeze(player, freeze_idx)
    elif q_idx >= action_ranges[5] and q_idx < action_ranges[6]: # Unfreeze
        #print("Unfreezing")
        unfreezable_index = q_idx - action_ranges[5]
        pet_food_to_unfreeze = idx2pet[unfreezable_index] if unfreezable_index < len(idx2pet) else idx2food[unfreezable_index - len(idx2pet)]
        unfreeze_idx = -1
        for i, slot in enumerate(player.shop.slots):
            if slot.frozen and slot.obj.name == pet_food_to_unfreeze:
                unfreeze_idx = i
                break

        if unfreeze_idx == -1:
            return "invalid_idx"

        return unfreeze(player, unfreeze_idx)
    elif q_idx >= action_ranges[6] and q_idx < action_ranges[7]: # End turn
        #print("Ending turn")
        return end_turn(player)
    #except Exception as e:
    #    print("Something went wrong in call_action_from_q_index: ", e)
    #    player.gold += 1
    #    player.roll()
    
    return "success"

def auto_order_team(player : Player) -> Player:
    # Orders pets from 1. Bias, 2. Total stats (attack + health)

    value_pairs = [(pet_idx, player.team.slots[pet_idx]) for pet_idx in player.team.filled]
    value_pairs = sorted(value_pairs, key = lambda x : (pet_auto_order_bias[x[1].pet.name], x[1].attack + x[1].health), reverse = True)

    player.reorder([x[0] for x in value_pairs])

food_types = {
    "single_target_stat_up": {"food-apple", "food-cupcake", 'food-pear', 'food-milk'},
    "multi_target_stat_up": {"food-salad-bowl", "food-pizza", "food-canned-food", 'food-sushi'},
    "effect": {"food-honey", "food-meat-bone", 'food-garlic', 'food-chili', 'food-melon', 'food-mushroom', 'food-steak'},
    "death_causes": {"food-sleeping-pill"},
    "level_up": {"food-chocolate"}
}

def auto_assign_food_to_pet(food : str, player : Player) -> Player:
    if food in food_types["single_target_stat_up"]:
        pet_team_list = [slot.pet.name for slot in player.team.slots]
        if "pet-seal" in pet_team_list:
            return pet_team_list.index("pet-seal")
        elif "pet-worm" in pet_team_list:
            return pet_team_list.index("pet-worm")
        
        legal_indexes = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none"]
        
        return random.sample(legal_indexes, 1)[0]
    elif food in food_types["multi_target_stat_up"]:
        legal_indexes = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none"]
        return random.sample(legal_indexes, 1)[0]
    elif food in food_types["effect"]:
        has_no_effect_item = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none" and slot.pet.status == "none"]
        if len(has_no_effect_item) > 0:
            return_idx = random.sample(has_no_effect_item, 1)[0]
            return return_idx
        else:
            has_effect_item = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none" and slot.pet.status != "none"]
            return_idx = random.sample(has_effect_item, 1)[0]
            return return_idx
    elif food in food_types["death_causes"]:
        faint_pets = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none" and slot.pet.ability['trigger'] == "Faint"]
        if len(faint_pets) > 0:
            faint_pet_idx = random.sample(faint_pets, 1)[0]
            return faint_pet_idx
        else:
            nonfaint_pets = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none" and not slot.pet.ability['trigger'] == "Faint"]
            faint_pet_idx = random.sample(nonfaint_pets, 1)[0]
            return faint_pet_idx
    elif food in food_types["level_up"]:
        pet_experience = np.array([slot.pet.experience if slot.pet.name != "pet-none" else -1 for slot in player.team.slots])
        pet_level_3 = np.array([slot.pet.level == 3 if slot.pet.name != "pet-none" else False for slot in player.team.slots])
        pet_experience[pet_level_3] = -1
        return int(np.argmax(pet_experience))

    raise Exception("Food type not found")

def action_index_to_action_type(action_number : int) -> str:
    for i in range(len(agent_actions_list) - 1):
        if action_number >= action_beginning_index[i] and action_number < action_beginning_index[i + 1]:
            return agent_actions_list[i]
        
    return agent_actions_list[-1]

def create_available_action_mask_original(player : Player) -> np.ndarray:
    action_mask = np.ones(shape = (action_beginning_index[-1] + 1), dtype = np.uint8)

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

def min_empty_zero(list_ : list) -> int:
    if len(list_) == 0: return 3
    else: return min(list_)

def create_available_action_mask(player : Player) -> np.ndarray:
    action_mask = np.zeros(shape = (action_beginning_index[-1] + 1), dtype = np.uint8)

    pet_levels = defaultdict(list)
    for slot in player.team.slots:
        if slot.pet.name == "pet-none": continue
        pet_levels[slot.pet.name].append(slot.level)

    # 1) End turn legality
    action_mask[action_beginning_index[action2index["end_turn"]]] = 1

    # 2) Roll legality
    if player.gold >= 1:
        action_mask[action_beginning_index[action2index["roll"]]] = 1

    # 3) Buy pet legality
    purchasable_pets = [slot.obj.name for slot in player.shop.slots if slot.slot_type in ["pet", "levelup"] and slot.cost <= player.gold and (min_empty_zero(pet_levels[slot.obj.name]) < 3 or len(player.team.filled) < 5)]
    for pet in purchasable_pets:
        action_idx = action_beginning_index[action2index["buy_pet"]] + pet2idx[pet]
        action_mask[action_idx] = 1

    # 4) Buy food legality
    purchasable_foods = [slot.obj.name for slot in player.shop.slots if slot.slot_type == "food" and slot.cost <= player.gold and len(player.team.filled) > 0]
    for food in purchasable_foods:
        action_idx = action_beginning_index[action2index["buy_food"]] + food2idx[food]
        action_mask[action_idx] = 1

    # 5) Sell pet legality
    sellable_pets = [slot.obj.name for slot in player.team.slots if slot.pet.name != "pet-none"]
    for pet in sellable_pets:
        action_idx = action_beginning_index[action2index["sell"]] + pet2idx[pet]
        action_mask[action_idx] = 1

    # 6) Combine legality
    combinable_pets = [slot.obj.name for slot in player.team.slots if slot.pet.name != "pet-none" and slot.pet.level < 3 and len(pet_levels[slot.pet.name]) > 1]
    for pet in combinable_pets:
        action_idx = action_beginning_index[action2index["combine"]] + pet2idx[pet]
        action_mask[action_idx] = 1

    # 7) Freeze legality
    freezable_items = [slot.obj.name for slot in player.shop.slots if not slot.frozen]
    for item in freezable_items:
        if item in idx2food.values(): item_idx = food2idx[item] + len(pet2idx)
        elif item in idx2pet.values(): item_idx = pet2idx[item]
        else: continue
        action_idx = action_beginning_index[action2index["freeze"]] + item_idx
        action_mask[action_idx] = 1

    # 8) Unfreeze legality
    unfreezable_items = [slot.obj.name for slot in player.shop.slots if slot.frozen]
    for item in unfreezable_items:
        if item in idx2food.values(): item_idx = food2idx[item] + len(pet2idx)
        elif item in idx2pet.values(): item_idx = pet2idx[item]
        else: continue
        action_idx = action_beginning_index[action2index["unfreeze"]] + item_idx
        action_mask[action_idx] = 1

    return action_mask