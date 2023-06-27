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
from config import rollout_device, training_device, N_ACTIONS

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
def buy_pet(player : Player, pet_idx : int, best_sell_idx : int = None) -> str:
    assert pet_idx >= 0 and pet_idx < 6

    og_pet_shop = deepcopy(player.shop)
    og_pet_index = pet_idx

    sell_return_index = None

    number_pets = len(player.shop.pets)
    #number_food = len(player.shop.foods)

    if pet_idx >= number_pets:
        return "invalid_idx", sell_return_index

    try:
        player.buy_pet(pet_idx)
        return "success", sell_return_index
    except GoldException as e:
        return "not_enough_gold", sell_return_index
    except FullTeamException as e:
        # Try to buy upgrade if there is no spaagent_actions_listce
        #print(player.shop, player.team, "gold:", player.gold, "\n")
        for team_idx in range(5):
            try:
                player.buy_combine(pet_idx, team_idx)
                return "success", sell_return_index
            except Exception as e:
                continue 

        # Buy selling because there is no available space
        if best_sell_idx is not None:
            sell_instead_of_combine = True if best_sell_idx < len(pet2idx) else False
            pet_used = idx2pet[best_sell_idx % len(pet2idx)]
            if sell_instead_of_combine:
                return_pet_val = [i for i, slot in enumerate(player.team.slots) if slot.pet.name == pet_used]

                # If food selection is an effect, prioritize pets that are not already affected by an effect
                if len(return_pet_val) == 1:
                    sell_idx = return_pet_val[0]
                else:
                    pet_priority = [sell_priority_for_same_pet(player.team.slots[i].pet, pet_used) for i in return_pet_val]
                    pet_index = np.argmax(pet_priority)
                    sell_idx = return_pet_val[pet_index]

                player.shop = og_pet_shop

                player.sell(sell_idx)
                player.buy_pet(og_pet_index)

                return "success", best_sell_idx
            else:
                pet_to_combine = -1
                for i, slot in enumerate(player.team.slots):
                    if slot.pet.name == pet_used:
                        pet_to_combine = i
                        break
                combine(player, pet_to_combine)
                player.buy_pet(og_pet_index)

                return "success", best_sell_idx + len(pet2idx)

        return "not_enough_space", sell_return_index

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
        if player.team.slots[combine_pet_idx].pet.name == pet_species and player.team.slots[combine_pet_idx].pet.level < 3:
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

def call_action_from_q_index(player : Player, q_idx : int, food_best_move : int = None, sell_best_move : int = None, epsilon: float = None, config : dict = None) -> str:
    # This code is gross, but it just defines the ranges the actions take
    if not isinstance(q_idx, int):
        q_idx = int(q_idx)

    return_food_index = True if food_best_move is not None else False
    food_return_index = None
    
    return_sell_index = True if sell_best_move is not None else False
    sell_return_index = None

    pretrain = (config is not None) and ("pretrain_transformer" in config) and config["pretrain_transformer"]

    assert q_idx >= 0 and q_idx < action_ranges[-1]

    if q_idx == 0: # Roll
        #print("Rolling")
        ret_val =  roll(player)
    elif q_idx >= action_ranges[0] and q_idx < action_ranges[1]: # Buy pet
        #print("Buying pet")
        pet_list_index = q_idx - action_ranges[0]
        pet_to_buy = idx2pet[pet_list_index]
        if pretrain:
            new_shop_slots = Shop([pet_to_buy.replace("pet-", "")], turn = player.turn, shop_attack=player.shop.shop_attack, shop_health=player.shop.shop_health).slots
            old_shop_slots = player.shop.slots
            player.shop.slots = new_shop_slots
            ret_val, sell_return_index = buy_pet(player, 0, best_sell_idx=sell_best_move)
            player.shop.slots = old_shop_slots
        else:
            pet_shop_list = [pet.name for pet in player.shop.pets if pet.name != "pet-none"]
            
            if pet_to_buy not in pet_shop_list:
                ret_val = "invalid_idx"
            else:
                buy_idx = pet_shop_list.index(pet_to_buy)

                ret_val, sell_return_index = buy_pet(player, buy_idx, best_sell_idx=sell_best_move)
    elif q_idx >= action_ranges[1] and q_idx < action_ranges[2]: # Sell pet
        #print("Selling pet")
        pet_list_index = q_idx - action_ranges[1]
        pet_to_sell = idx2pet[pet_list_index]
        pet_team_list = [slot.pet.name for slot in player.team.slots]

        if pet_to_sell not in pet_team_list:
            ret_val = "invalid_idx"
        else:
            index_total_stats = {i : player.team.slots[i].pet.health + player.team.slots[i].pet.attack for i in range(len(player.team.slots)) if player.team.slots[i].pet.name == pet_to_sell}
            sell_idx = -1
            for i in range(len(player.team.slots)):
                if player.team.slots[i].pet.name == pet_to_sell:
                    if sell_idx == -1:
                        sell_idx = i
                    elif index_total_stats[i] < index_total_stats[sell_idx]:
                        sell_idx = i

            ret_val = sell(player, sell_idx)
    elif q_idx >= action_ranges[2] and q_idx < action_ranges[3]: # Buy food
        #print("Buying food")
        food_list_index = q_idx - action_ranges[2]
        food_to_buy = idx2food[food_list_index]
        if pretrain:
            if player.team.filled == []:
                ret_val = "invalid_idx"
            else:
                new_shop_slots = Shop([food_to_buy.replace("food-", "")], turn = player.turn, shop_attack=player.shop.shop_attack, shop_health=player.shop.shop_health).slots
                old_shop_slots = player.shop.slots
                player.shop.slots = new_shop_slots

                food_return_index, _ = auto_assign_food_to_pet(food_to_buy, player, food_move = food_best_move, epsilon = epsilon)

                pet_name_to_be_fed = player.team.slots[food_return_index].pet.name

                ret_val = buy_food(player, 0, food_return_index)

                player.shop.slots = old_shop_slots

                food_return_index = pet2idx[pet_name_to_be_fed]
        else:
            food_shop_list = [food.name for food in player.shop.foods if food.name != "food-none"]

            if food_to_buy not in food_shop_list or player.team.filled == []:
                ret_val = "invalid_idx"
            else:
                food_buy_idx = food_shop_list.index(food_to_buy)

                food_return_index, _ = auto_assign_food_to_pet(food_to_buy, player, food_move = food_best_move, epsilon = epsilon)

                pet_name_to_be_fed = player.team.slots[food_return_index].pet.name

                ret_val = buy_food(player, food_buy_idx, food_return_index)

                food_return_index = pet2idx[pet_name_to_be_fed]
    elif q_idx >= action_ranges[3] and q_idx < action_ranges[4]: # Combine
        #print("Combining")
        pet_list_index = q_idx - action_ranges[3]
        pet_to_combine = idx2pet[pet_list_index]
        pet_team_list = [slot.pet.name for slot in player.team.slots]

        if pet_team_list.count(pet_to_combine) < 2:
            ret_val = "invalid_idx"
        else:
            combine_idx = -1
            for i, slot in enumerate(player.team.slots):
                if slot.obj.name == pet_to_combine and slot.obj.level < 3:
                    if combine_idx == -1:
                        combine_idx = i
                    elif slot.obj.health + slot.obj.attack > player.team.slots[combine_idx].obj.health + player.team.slots[combine_idx].obj.attack:
                        combine_idx = i

            if combine_idx == -1:
                ret_val = "invalid_idx"
            else:
                ret_val = combine(player, combine_idx)
                
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
            ret_val = "invalid_idx"
        else:
            ret_val = freeze(player, freeze_idx)
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
            ret_val = "invalid_idx"
        else:
            ret_val = unfreeze(player, unfreeze_idx)
    elif q_idx >= action_ranges[6] and q_idx < action_ranges[7]: # End turn
        #print("Ending turn")
        ret_val = end_turn(player)
    #except Exception as e:
    #    print("Something went wrong in call_action_from_q_index: ", e)
    #    player.gold += 1
    #    player.roll()

    if return_food_index and return_sell_index: return ret_val, food_return_index, sell_return_index
    elif return_food_index: return ret_val, food_return_index
    elif return_sell_index: return ret_val, sell_return_index
    else: return ret_val

food_types = {
    "single_target_stat_up": {"food-apple", "food-cupcake", 'food-pear', 'food-milk'},
    "multi_target_stat_up": {"food-salad-bowl", "food-pizza", "food-canned-food", 'food-sushi'},
    "effect": {"food-honey", "food-meat-bone", 'food-garlic', 'food-chili', 'food-melon', 'food-mushroom', 'food-steak'},
    "death_causes": {"food-sleeping-pill"},
    "level_up": {"food-chocolate"}
}

def sell_priority_for_same_pet(pet, pet_name : str):
    sell_priority = 0
    sell_priority -= pet.attack
    sell_priority -= pet.health
    return sell_priority

def food_priority_for_same_pet(pet, food : str):
    food_priority = 0

    if food in food_types["single_target_stat_up"]:
        if pet.attack >= 50:
            food_priority -= 1000
        if pet.health >= 50:
            food_priority -= 1000
    elif food in food_types["multi_target_stat_up"]:
        if pet.attack >= 50:
            food_priority -= 1000
        if pet.health >= 50:
            food_priority -= 1000
    elif food in food_types["effect"]:
        if pet.status != "none":
            food_priority -= 1000
    elif food in food_types["death_causes"]:
        food_priority += (pet.level - 1) * 100
        food_priority += (50 - pet.health)
        food_priority += (50 - pet.attack)
    elif food in food_types["level_up"]:
        if pet.level >= 3:
            food_priority -= 10000
        else:
            if isinstance(pet.tier, int):
                food_priority += (pet.tier - 1) * 100
            else:
                food_priority -= 1000

    food_priority = food_priority + (min(pet.attack, 50) + min(pet.health, 50))
    return food_priority

def auto_assign_food_to_pet(food : str, player : Player, food_move = None, epsilon : float = 1.0) -> Player:
    # Use food_net to determine which pet to assign food to
    if food_move is not None:
        if len(player.team.filled) == 0:
            return None
        
        random_val = random.uniform(0, 1)

        if random_val > epsilon or epsilon is None: # Choose based on best move
            type_of_animal_chosen = idx2pet[food_move]

            return_pet_val = [i for i, slot in enumerate(player.team.slots) if slot.pet.name == type_of_animal_chosen]

            # If food selection is an effect, prioritize pets that are not already affected by an effect
            if len(return_pet_val) == 1:
                return return_pet_val[0], type_of_animal_chosen
            else:
                pet_priority = [food_priority_for_same_pet(player.team.slots[pet_idx].pet, food) for pet_idx in return_pet_val]
                pet_index = np.argmax(pet_priority)
                return_pet_val = return_pet_val[pet_index]
                return return_pet_val, type_of_animal_chosen

        else: # Choose randomly
            legal_indexes = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none"]
            random_pet = random.sample(legal_indexes, 1)[0]
            return random_pet, player.team.slots[random_pet].pet.name
        
    else: # Use procedural food:
        if food in food_types["single_target_stat_up"]:
            pet_team_list = [slot.pet.name for slot in player.team.slots]
            if "pet-seal" in pet_team_list:
                return pet_team_list.index("pet-seal"), "pet-seal"
            elif "pet-worm" in pet_team_list:
                return pet_team_list.index("pet-worm"), "pet-worm"
            
            legal_indexes = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none"]
            selected_pet = _get_highest_stat_pet(player, legal_indexes)

            return selected_pet, player.team.slots[selected_pet].pet.name
        elif food in food_types["multi_target_stat_up"]:
            legal_indexes = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none"]
            selected_pet = _get_highest_stat_pet(player, legal_indexes)

            return selected_pet, player.team.slots[selected_pet].pet.name
        elif food in food_types["effect"]:
            has_no_effect_item = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none" and slot.pet.status == "none"]
            if len(has_no_effect_item) > 0:
                return_idx = _get_highest_stat_pet(player, has_no_effect_item)
                return return_idx, player.team.slots[return_idx].pet.name
            else:
                has_effect_item = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none" and slot.pet.status != "none"]
                return_idx = _get_highest_stat_pet(player, has_effect_item)
                return return_idx, player.team.slots[return_idx].pet.name
        elif food in food_types["death_causes"]:
            faint_pets = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none" and slot.pet.ability['trigger'] == "Faint"]
            if len(faint_pets) > 0:
                faint_pet_idx = _get_highest_stat_pet(player, faint_pets)
                return faint_pet_idx, player.team.slots[faint_pet_idx].pet.name
            else:
                nonfaint_pets = [i for i, slot in enumerate(player.team.slots) if slot.pet.name != "pet-none" and not slot.pet.ability['trigger'] == "Faint"]
                faint_pet_idx = _get_highest_stat_pet(player, nonfaint_pets)
                return faint_pet_idx, player.team.slots[faint_pet_idx].pet.name
        elif food in food_types["level_up"]:
            pet_experience = np.array([slot.pet.experience if slot.pet.name != "pet-none" else -5 for slot in player.team.slots])
            pet_level_3 = np.array([slot.pet.level == 3 if slot.pet.name != "pet-none" else False for slot in player.team.slots])
            pet_experience[pet_level_3] = -1
            selected_pet = int(np.argmax(pet_experience))
        
            return selected_pet, player.team.slots[selected_pet].pet.name

        raise Exception("Food type not found")

def _get_highest_stat_pet(player : Player, list : str) -> int:
    highest_stats = -1
    highest_stats_index = -1
    for i in list:
        if player.team.slots[i].pet.name != "pet-none":
            total_stats = player.team.slots[i].pet.health + player.team.slots[i].pet.attack 
            if total_stats > highest_stats:
                highest_stats = total_stats
                highest_stats_index = i

    return highest_stats_index

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

def create_available_action_mask(player : Player, return_food_mask : bool = False, return_sell_mask : bool = False) -> np.ndarray:
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
    purchasable_pets = [slot.obj.name for slot in player.shop.slots if slot.slot_type in ["pet", "levelup"] and slot.cost <= player.gold]
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
    pet_levels_less_than_3 = {pet : [level for level in levels if level < 3] for pet, levels in pet_levels.items()}
    combinable_pets = [slot.obj.name for slot in player.team.slots if slot.pet.name != "pet-none" and slot.pet.level < 3 and len(pet_levels_less_than_3[slot.pet.name]) > 1]
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

    if return_food_mask:
        food_mask = deepcopy(action_mask[action_beginning_index[action2index["sell"]]:action_beginning_index[action2index["sell"] + 1]])

    if return_sell_mask:
        sell_animal_component = deepcopy(action_mask[action_beginning_index[action2index["sell"]]:action_beginning_index[action2index["sell"] + 1]])
        combine_pets_component = deepcopy(action_mask[action_beginning_index[action2index["combine"]]:action_beginning_index[action2index["combine"] + 1]])
        sell_mask = np.concatenate([sell_animal_component, combine_pets_component], axis = 0)


    if return_food_mask and return_sell_mask: return action_mask, deepcopy(food_mask), deepcopy(sell_mask)
    elif return_food_mask: return action_mask, food_mask
    elif return_sell_mask: return action_mask, sell_mask
    else: return action_mask


buyable_pets = [pet for pet in pet2idx.keys() if pet not in ["pet-none", "pet-zombie-cricket", "pet-bus", "pet-zombie-fly", "pet-dirty-rat", "pet-chick", "pet-ram", "pet-bee"]]
buyable_foods = [food for food in food2idx.keys() if food not in ["food-milk"]]
buyable_items = buyable_pets + buyable_foods

combo_suites = {
    "dragon": ["pet-ant", "pet-beaver", "pet-cricket", "pet-duck", "pet-fish", "pet-horse", "pet-mosquito", "pet-otter", "pet-pig", "pet-dragon", "pet-gorilla", "pet-leopard", 
                "pet-rhino", "food-chocolate", "food-melon", "food-garlic"],
    "foody": ["pet-fish", "pet-rabbit", "pet-worm", "pet-squirrel", "pet-cow", "pet-seal", "pet-cat"] + buyable_foods,
    "penguin": ["pet-penguin"] +  random.sample(buyable_pets, k = 10) + ["pet-duck", "food-chocolate"],
    "spawners": ["pet-cricket", "food-honey", "pet-spider", "pet-sheep", "pet-deer", "pet-rooster", "pet-shark", "pet-turkey", "pet-fly", "pet-snake"] + buyable_foods,
}

pets_by_level = {}
foods_by_level = {}
items_by_level = {}
for level in range(1, 7):
    current_level_mask_pets = [item for item in pet2idx.keys() if isinstance(ALL_DATA["pets"][item]["tier"], int) and (ALL_DATA["pets"][item]["tier"] == level)]
    current_level_mask_food = [item for item in food2idx.keys() if isinstance(ALL_DATA["foods"][item]["tier"], int) and (ALL_DATA["foods"][item]["tier"] == level)]
    
    pets_by_level[level] = current_level_mask_pets
    foods_by_level[level] = current_level_mask_food
    items_by_level[level] = current_level_mask_pets + current_level_mask_food

# Available pet stuff
def create_available_item_mask(players : Player, config : dict, predefined_selections : list = None, epoch = None) -> np.ndarray:
    pet_mask_types = ["full", "part", "combo"]

    if isinstance(players, Player): players = [players]

    if predefined_selections is None:
        predefined_selections = [None for _ in range(len(players))]
    else:
        assert len(predefined_selections) == len(players)
        for selection in predefined_selections:
            assert selection in pet_mask_types

    mask = np.zeros(shape = (len(players), len(idx2pet) + len(idx2food)), dtype = np.int32)
    item2idx, idx2item = get_idx_item_dicts()

    for player_num, selection_type in zip(range(len(players)), predefined_selections):
        if selection_type is None:
            selection_type = random.choices(pet_mask_types, weights = config["full_part_combo_split"])[0]

        if selection_type == "full":
            random_selection = np.array([item2idx[item] for item in buyable_items])
            mask[player_num, random_selection] = 1

        elif selection_type == "part":
            if config["train_on_increasing_partial_actions"] and epoch is not None:
                number_to_select = min(epoch, len(buyable_items))
            else:
                number_to_select = config["partial_number_of_actions"]

            all_level_selection = []
            for level in range(1, 7):
                all_level_selection = all_level_selection + random.sample(pets_by_level[level], k = 1)
                all_level_selection = all_level_selection + random.sample(foods_by_level[level], k = 1)

            random_selection = random.sample(buyable_items, k = number_to_select)
            random_selection += all_level_selection

            random_selection = np.array([item2idx[item] for item in random_selection])
            mask[player_num, random_selection] = 1
        elif selection_type == "combo":
            combo_type = random.sample(list(combo_suites.keys()), k = 1)[0]
            random_selection = np.array([item2idx[item] for item in combo_suites[combo_type]])
            mask[player_num, random_selection] = 1
        else:
            raise ValueError("Invalid selection type")

    return mask

def get_idx_item_dicts():
    item2idx = list(idx2pet.values()) + list(idx2food.values())
    item2idx = {item : idx for idx, item in enumerate(item2idx)}
    idx2item = {idx : item for item, idx in item2idx.items()}
    return item2idx, idx2item

def get_best_legal_move(player : Player, net, config : dict, epoch = 20, epoch_masking = True, mask = None, return_q_values = False, pretrain_mask = None) -> int:
    state_encodings = net.state_to_encoding(player)

    if config["pretrain_transformer"]:
        available_action_mask = create_available_item_mask(1, config, predefined_selections = ["full"])
        state_encodings = net.modify_encodings_pretrain(state_encodings, available_action_mask, config = config)

    q_value = net(state_encodings)
    q_value = q_value.squeeze(0)

    if mask is None:
        action_mask = create_available_action_mask(player)

        # if epoch makes turns illegal
        if epoch_masking:
            epoch_illegal_mask = create_epoch_illegal_mask(epoch, config)
            action_mask = action_mask * epoch_illegal_mask
        
        if config["pretrain_transformer"]:
            if pretrain_mask is None: available_action_mask = create_available_item_mask(1, config, predefined_selections = ["full"])
            else: available_action_mask = pretrain_mask

            available_action_mask = np.squeeze(available_action_mask)

            if player.gold >= 3:
                action_mask[action_beginning_index[action2index["buy_pet"]]:action_beginning_index[action2index["buy_pet"] + 1]] = available_action_mask[:len(pet2idx)]
            # buy food actions
            if len(player.team.filled) != 0 and player.gold >= 3:
                action_mask[action_beginning_index[action2index["buy_food"]]:action_beginning_index[action2index["buy_food"] + 1]] = available_action_mask[len(pet2idx):]

        max_q_action_mask = np.stack(deepcopy(action_mask))
        max_q_action_mask = torch.tensor(max_q_action_mask, dtype = torch.int32, device = rollout_device).requires_grad_(False)
    else:
        max_q_action_mask = torch.tensor(mask.astype(np.int32), dtype = torch.int32, device = rollout_device).requires_grad_(False)

    # Adding mask to the max q action vector
    where_zeros = max_q_action_mask < 0.5
    q_value[where_zeros] = -9999999

    action_taken = int(torch.argmax(q_value).cpu().numpy())

    if return_q_values:
        return action_taken, q_value
    return action_taken

def create_epoch_illegal_mask(epoch, config):
    action_mask = np.ones(N_ACTIONS, dtype = np.uint8)
    if "illegalize_rolling" in config and (epoch >= config["illegalize_rolling"][0] and epoch <= config["illegalize_rolling"][1]):
        action_mask[action_beginning_index[action2index["roll"]]:action_beginning_index[action2index["roll"] + 1]] = 0
    if "illegalize_freeze_unfreeze" in config and (epoch >= config["illegalize_freeze_unfreeze"][0] and epoch <= config["illegalize_freeze_unfreeze"][1]):
        action_mask[action_beginning_index[action2index["freeze"]]:action_beginning_index[action2index["freeze"] + 1]] = 0
        action_mask[action_beginning_index[action2index["unfreeze"]]:action_beginning_index[action2index["unfreeze"] + 1]] = 0
    if "illegalize_combine" in config and (epoch >= config["illegalize_combine"][0] and epoch <= config["illegalize_combine"][1]):
        action_mask[action_beginning_index[action2index["combine"]]:action_beginning_index[action2index["combine"] + 1]] = 0

    return action_mask


# Reordering Functions and data

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

pet_enjoys_scaling_bias = {
    'pet-ant': 0,
    'pet-beaver': 0,
    'pet-cricket': 0,
    'pet-duck': 0,
    'pet-fish': 0,
    'pet-horse': 0,
    'pet-mosquito': 0,
    'pet-otter': 0,
    'pet-pig': 0,
    'pet-sloth': 0,
    'pet-crab': 0.5,
    'pet-dodo': 1.0,
    'pet-dog': 0.0,
    'pet-elephant': 1.0,
    'pet-flamingo': 0.0,
    'pet-hedgehog': 0.0,
    'pet-peacock': 0.5,
    'pet-rat': 0.0,
    'pet-shrimp': 0.0,
    'pet-spider': 0.0,
    'pet-swan': 0,
    'pet-badger': 0.0,
    'pet-blowfish': 1.5,
    'pet-camel': 0.5,
    'pet-giraffe': 0.0,
    'pet-kangaroo': 0.0,
    'pet-ox': 1.0,
    'pet-rabbit': 0.0,
    'pet-sheep': 0.0,
    'pet-snail': 0.0,
    'pet-turtle': 0.0,
    'pet-whale': 0.0,
    'pet-bison': 0.5,
    'pet-deer': 0.0,
    'pet-dolphin': 0.5,
    'pet-hippo': 2.5,
    'pet-monkey': 0.5,
    'pet-penguin': 0.0,
    'pet-rooster': 2.5,
    'pet-skunk': 0.0,
    'pet-squirrel': 0,
    'pet-worm': 0,
    'pet-cow': 0,
    'pet-crocodile': 1.5,
    'pet-parrot': 0,
    'pet-rhino': 2.5,
    'pet-scorpion': 0.0,
    'pet-seal': 0.0,
    'pet-shark': 0.0,
    'pet-turkey': 0.0,
    'pet-cat': 0.0,
    'pet-boar': 2.5,
    'pet-dragon': 0.5,
    'pet-fly': 0.0,
    'pet-gorilla': 2.5,
    'pet-leopard': 2.5,
    'pet-mammoth': 0.5,
    'pet-snake': 0.0,
    'pet-tiger': 0.0,
    'pet-zombie-cricket': 0.0,
    'pet-bus': 0.0,
    'pet-zombie-fly': 0.0,
    'pet-dirty-rat': 0.0,
    'pet-chick': 0.0,
    'pet-ram': 0.0,
    'pet-bee': 0.0
}

spawers = {"pet-cricket", "pet-spider", "pet-sheep", "pet-deer", "pet-rooster"}

def pet_add_to_order_utility(slot_number : int, team : Team) -> float:
    pet_name = team.slots[slot_number].pet.name

    assert pet_name in pet2idx.keys()

    match pet_name:
        case 'pet-ant':
            # Likes to be in front of at least 1 pet
            number_behind = [1 for i in range(5) if i > slot_number and i in team.filled]
            return min(1.2, sum(number_behind))
        case 'pet-beaver':
            return 0.0
        case 'pet-cricket':
            return 0.0
        case 'pet-duck':
            return 0.0
        case 'pet-fish':
            return 0.0
        case 'pet-horse':
            # Likes to be behind spawners
            spawners_in_front = [1 for i in range(5) if i < slot_number and i in team.filled and team.slots[i].pet.name in spawers]
            return 0.5 * sum(spawners_in_front)
        case 'pet-mosquito':
            return 0.0
        case 'pet-otter':
            return 0.0
        case 'pet-pig':
            return 0.0
        case 'pet-sloth':
            return 0.0
        case 'pet-crab':
            return 0.0
        case 'pet-dodo':
            # Likes being in front of scaler
            pet_in_front = [i for i in range(5) if i < slot_number and i in team.filled and team.slots[i].pet.name in pet_enjoys_scaling_bias.keys()]
            
            if len(pet_in_front) == 0:
                return 0.0
            else:
                pet_in_front = team.slots[pet_in_front[0]].pet
                if pet_in_front.health >= 50 or pet_in_front.attack >= 50:
                    return 0.0
                else:
                    return pet_enjoys_scaling_bias[pet_in_front.name]
        case 'pet-dog':
            # Likes being behind spawners
            spawners_in_front = [1 for i in range(5) if i < slot_number and i in team.filled and team.slots[i].pet.name in spawers]
            return sum(spawners_in_front)
        case 'pet-elephant':
            # Complex. Likes to be in back. OR in front of pets that like taking damage
            pass
        case 'pet-flamingo':
            # Likes to be in front of 2 other pets
            number_behind = [1 for i in range(5) if i > slot_number and i in team.filled]
            return min(2.0, sum(number_behind))
        case 'pet-hedgehog':
            return 0.0
        case 'pet-peacock':
            return 0.0
        case 'pet-rat':
            # Likes to be in front (barely)
            if slot_number == min(team.filled):
                return 0.2
            else:
                return 0.0
        case 'pet-shrimp':
            return 0.0
        case 'pet-spider':
            return 0.0
        case 'pet-swan':
            return 0.0
        case 'pet-badger':
            if slot_number == max(team.filled):
                return 1.5
            else:
                return 0.0
        case 'pet-blowfish':
            # Likes to be in front
            spot_factor = [0.0, 0.0, 0.25, 0.5, 1.0]
            pets_ahead = [1 for i in range(5) if i < slot_number and i in team.filled]
            return spot_factor[4 - len(pets_ahead)]
        case 'pet-camel':
            # Likes to be behind many pets
            pets_behind = [1 for i in range(5) if i > slot_number and i in team.filled]
            return sum(pets_behind)
        case 'pet-giraffe':
            # Likes to be with scaler (level wise shows how many)
            ret_value = 0.0
            level = team.slots[slot_number].pet.level
            pets_idx_ahead = [i for i in range(5) if i < slot_number and i in team.filled]
            pets_ahead = [team.slots[i].pet for i in pets_idx_ahead]
            pets_ahead.reverse()
            for i in range(level):
                if i >= len(pets_ahead):
                    ret_value -= 1.0
                    continue
                ret_value += pet_enjoys_scaling_bias[pets_ahead[i].name]
            return ret_value
        case 'pet-kangaroo':
            # Likes to be behind big guy
            pet_ahead = [i for i in range(5) if i < slot_number and i in team.filled]
            if len(pet_ahead) == 0:
                return 0.0
            else:
                pet_ahead = team.slots[pet_ahead[0]].pet
                pet_ahead_stats = float(pet_ahead.health + pet_ahead.attack) / 60
                spawner_bonus = 0.5 if pet_ahead.name in spawers else 0.0
                defensive_bonus = 0.5 if pet_ahead.name in {"pet-gorilla", "pet-boar", "pet-hippo"} else 0.0
                food_defensive_bonus = 0.5 if pet_ahead.status in {"status-garlic-armour", "status-extra-life", "status-melon-armor"} else 0.0
                return (pet_ahead_stats + spawner_bonus + defensive_bonus + food_defensive_bonus)
        case 'pet-ox':
            # Likes to be behind at least one guy
            if slot_number == min(team.filled):
                return 0.0
            else:
                return 1.0
        case 'pet-rabbit':
            return 0.0
        case 'pet-sheep':
            # Does not like to be in the very front
            if len(team.filled) == 5 and slot_number == 0:
                return -1.0
            else:
                return 0.0
        case 'pet-snail':
            return 0.0
        case 'pet-turtle':
            # Likes to be in front of a good big target
            pet_idx_behind = [i for i in range(5) if i > slot_number and i in team.filled]
            pet_behind = [team.slots[i].pet for i in pet_idx_behind]
            pet_behind.reverse()

            level = team.slots[slot_number].pet.level
            ret_val = 0.0

            for i in range(level):
                if i >= len(pet_behind):
                    ret_val -1.0
                    continue
                
                curr_pet = pet_behind[i]
                stat_bonus = float(curr_pet.health + curr_pet.attack) / 60
                already_has_status_penalty = -1.0 if curr_pet.status in {"status-garlic-armour", "status-extra-life", "status-sneak-attack", "status-food-melon"} else 0.0
                ret_val += stat_bonus + already_has_status_penalty

            return ret_val
        case 'pet-whale':
            pet_ahead = [i for i in range(5) if i < slot_number and i in team.filled]
            if len(pet_ahead) == 0:
                return 0.0
            else:
                pet_ahead = team.slots[pet_ahead[0]].pet
                # Complex one. Look at faint abilities
                # OKAY eat
                if pet_ahead.name in {"pet-ant", "pet-cricket", "pet-hedgehog"}:
                    return 0.5
                # GOOD abilities
                elif pet_ahead.name in {"pet-flamingo", "pet-spider", "pet-sheep", "pet-scorpion"}:
                    return 1.0
                elif pet_ahead.name in {"pet-turtle", "pet-deer"}:
                    return 1.5
                elif pet_ahead.name in {"pet-mammoth"}:
                    return 2.5
                else:
                    return 0.0
        case 'pet-bison':
            return 0.0
        case 'pet-deer':
            # Likes to be in front
            if slot_number == min(team.filled):
                return 1.0
            else:
                return 0.0
        case 'pet-dolphin':
            return 0.0
        case 'pet-hippo':
            # Loves to be in the front
            spot_factor = [0.0, 0.0, 0.0, 1.0, 1.5]
            pets_ahead = [1 for i in range(5) if i < slot_number and i in team.filled]
            return spot_factor[4 - len(pets_ahead)]
        case 'pet-monkey':
            # Likes to have a good scaler at front
            if len(team.filled) == 1:
                return 0.0
            else:
                frontest_animal = team.slots[min(team.filled)].pet
                if frontest_animal.health >= 50 or frontest_animal.attack >= 50:
                    return 0.0
                else:
                    return pet_enjoys_scaling_bias[frontest_animal.name]
        case 'pet-penguin':
            return 0.0
        case 'pet-rooster':
            return 0.0
        case 'pet-skunk':
            return 0.0
        case 'pet-squirrel':
            return 0.0
        case 'pet-worm':
            return 0.0
        case 'pet-cow':
            return 0.0
        case 'pet-crocodile':
            return 0.0
        case 'pet-parrot':
            # Another complex one
            pet_ahead = [i for i in range(5) if i < slot_number and i in team.filled]
            if len(pet_ahead) == 0:
                return 0.0
            else:
                pet_ahead = team.slots[pet_ahead[0]].pet
                # Complex one. Look at faint abilities
                # OKAY eat
                if pet_ahead.name in {"pet-ant", "pet-hedgehog", "pet-sheep", "pet-spider", "pet-mosquito"}:
                    return 0.5
                # GOOD abilities
                elif pet_ahead.name in {"pet-flamingo", "pet-leopard", "pet-dolphin", "pet-kangaroo", "pet-camel", "pet-blowfish"}:
                    return 1.0
                elif pet_ahead.name in {"pet-deer", "pet-turkey", "pet-shark", "pet-rhino", "pet-skunk", "pet-crocodile", "pet-hippo"}:
                    return 1.5
                elif pet_ahead.name in {"pet-mammoth", "pet-snake", "pet-boar"}:
                    return 2.5
                else:
                    return 0.0
        case 'pet-rhino':
            # likes being in front
            spot_factor = [0.0, 0.0, 0.0, 1.0, 1.5]
            pets_ahead = [1 for i in range(5) if i < slot_number and i in team.filled]
            return spot_factor[4 - len(pets_ahead)]
        case 'pet-scorpion':
            # Likes being in front (takes out big guys)
            spot_factor = [0.0, 0.0, 0.0, 0.5, 1.0]
            pets_ahead = [1 for i in range(5) if i < slot_number and i in team.filled]
            return spot_factor[4 - len(pets_ahead)]
        case 'pet-seal':
            return 0.0
        case 'pet-shark':
            # Likes to be behind attackers (just loves the back)
            pets_ahead = [1 for i in range(5) if i < slot_number and i in team.filled]
            return float(sum(pets_ahead))
        case 'pet-turkey':
            # Likes to be behind spawners
            spawners_in_front = [1 for i in range(5) if i < slot_number and i in team.filled and team.slots[i].pet.name in spawers]
            return float(sum(spawners_in_front))
        case 'pet-cat':
            return 0.0
        case 'pet-boar':
            # likes being in front
            spot_factor = [0.0, 0.0, 0.0, 1.0, 1.5]
            pets_ahead = [1 for i in range(5) if i < slot_number and i in team.filled]
            return spot_factor[4 - len(pets_ahead)]
        case 'pet-dragon':
            return 0.0
        case 'pet-fly':
            # Likes to be behind spawner
            spawners_in_front = [1 for i in range(5) if i < slot_number and i in team.filled and team.slots[i].pet.name in spawers]
            return 1.1 * float(sum(spawners_in_front))
        case 'pet-gorilla':
            return 0.0
        case 'pet-leopard':
            return 0.0
        case 'pet-mammoth':
            # Likes to be in front a lot
            pets_behind = [1 for i in range(5) if i > slot_number and i in team.filled]
            return 1.5 * float(sum(pets_behind))
        case 'pet-snake':
            # Likes to be behind the biggest guy (preferably with a protection
            # ability)
            pet_ahead = [i for i in range(5) if i < slot_number and i in team.filled]
            if len(pet_ahead) == 0:
                return 0.0
            else:
                pet_ahead = team.slots[pet_ahead[0]].pet
                pet_ahead_stats = float(pet_ahead.health + pet_ahead.attack) / 60
                spawner_bonus = 0.5 if pet_ahead.name in spawers else 0.0
                defensive_bonus = 0.5 if pet_ahead.name in {"pet-gorilla", "pet-boar", "pet-hippo"} else 0.0
                food_defensive_bonus = 0.5 if pet_ahead.status in {"status-garlic-armour", "status-extra-life", "status-melon-armor"} else 0.0
                return (pet_ahead_stats + spawner_bonus + defensive_bonus + food_defensive_bonus) * 1.5
        case 'pet-tiger':
            # Complex go through all the abilities pretty much
            pet_ahead = [i for i in range(5) if i < slot_number and i in team.filled]
            if len(pet_ahead) == 0:
                return 0.0
            else:
                pet_ahead = team.slots[pet_ahead[0]].pet
                # Complex one. Look at faint abilities
                # OKAY eat
                if pet_ahead.name in {"pet-ant", "pet-hedgehog", "pet-sheep", "pet-spider", "pet-mosquito"}:
                    return 0.5
                # GOOD abilities
                elif pet_ahead.name in {"pet-flamingo", "pet-leopard", "pet-dolphin", "pet-kangaroo", "pet-camel", "pet-blowfish"}:
                    return 1.0
                elif pet_ahead.name in {"pet-deer", "pet-turkey", "pet-shark", "pet-rhino", "pet-skunk", "pet-crocodile", "pet-hippo"}:
                    return 1.5
                elif pet_ahead.name in {"pet-mammoth", "pet-snake", "pet-boar"}:
                    return 2.5
                else:
                    return 0.0
        case 'pet-zombie-cricket':
            return 0.0
        case 'pet-bus':
            return 0.0
        case 'pet-zombie-fly':
            return 0.0
        case 'pet-dirty-rat':
            return 0.0
        case 'pet-chick':
            return 0.0
        case 'pet-ram':
            return 0.0
        case 'pet-bee':
            return 0.0
        case _:
            raise ValueError(f'Unknown pet name: {pet_name}')

def calculate_team_score(team: Team) -> float:
    total_score = 0.0

    for i in team.filled:
        total_score += pet_add_to_order_utility(i, team)

    # Additional points for having the highest pets at the front
    # Create a list of tuples with (index, pet) pairs
    pet_list = [pet.pet for pet in team.slots if pet.pet.name != "pet-none"]
    indexed_pets = [(index, pet) for index, pet in enumerate(pet_list)]
    sorted_pets = sorted(indexed_pets, key=lambda x: (x[1].health, x[1].attack), reverse=True)
    ordered_indexes = [index for index, _ in sorted_pets]

    return total_score

def auto_order_team(player, return_team = True) -> Player:
    # Orders pets from 1. Bias, 2. Total stats (attack + health)
    if isinstance(player, Team):
        player = Player(team = player)

    if len(player.team.filled) in [0, 1]:
        if return_team: return deepcopy(player.team)
        else: return

    pet_bank = [pet.pet for pet in player.team.slots if pet.pet.name != "pet-none"]
    current_team = [pet_bank[0], ]

    for pet in pet_bank:
        team_scores = [0.0] * (len(current_team) + 1)
        for insert_idx in range(len(current_team) + 1):
            proposed_team = deepcopy(current_team)
            proposed_team.insert(insert_idx, pet)
            team = Team(proposed_team)
            greedy_score = calculate_team_score(team)

            team_scores[insert_idx] = greedy_score

        best_idx = team_scores.index(max(team_scores))
        current_team.insert(best_idx, pet)

    if return_team:
        return deepcopy(player.team)
