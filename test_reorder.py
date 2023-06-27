import os
import sys
import random

current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from model_actions import auto_order_team, pet_add_to_order_utility
from pretrain_model import generate_random_player, get_random_pet_percentages, get_random_food_percentages

pet_percentages = get_random_pet_percentages()
food_percentages = get_random_food_percentages()

for i in range(3000):
    player = generate_random_player(pet_percentages, food_percentages)
    auto_order_team(player)
    #auto_order_team(team)