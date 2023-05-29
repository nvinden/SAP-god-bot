import torch
import torch.nn as nn

import sys
import os

current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from sapai.pets import Pet
from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player, GoldException, WrongObjectException, FullTeamException

from sapai.data import data as ALL_DATA

N_ACTIONS = 47

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

class SAPAI(nn.Module):
    def __init__(self, config = None):
        super(SAPAI, self).__init__()

        # config is none, import default config, and use that
        if config is None:
            from config import DEFAULT_CONFIGURATION
            config = DEFAULT_CONFIGURATION

        self.config = config

        # Round up number_of_items so it is divisable by self.config['nhead']
        self.d_model = number_of_items + (self.config['nhead'] - number_of_items % self.config['nhead'])

        # Initializing the neural network
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.config['nhead'], batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = transformer_encoder_layer, num_layers = self.config['num_layers'])

        self.actions = nn.Linear(in_features = self.d_model * 15, out_features = N_ACTIONS)
        self.v = nn.Linear(in_features = self.d_model * 15, out_features = 1)


    def forward(self, x : torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        out_encoder = self.transformer_encoder(x)
        out_encoder = out_encoder.reshape(out_encoder.shape[0], -1)

        actions = self.actions(out_encoder)
        v = self.v(out_encoder)

        return actions, v

    def state_to_encoding(self, player) -> torch.FloatTensor:
        # If player is a player
        if isinstance(player, Player):
            player_list = [player]
        # If player is a list of players
        elif isinstance(player, list):
            player_list = player
        else:
            raise Exception("player is not a player or a list of players")

        encoding = torch.zeros(size = (len(player_list), 15, self.d_model), dtype = torch.float32)

        for batch_no, current_player in enumerate(player_list):
            # Dimension 1: Turn number, player health remaining, gold and wins
            turn_number = current_player.turn
            player_lives = current_player.lives
            gold = current_player.gold
            wins = current_player.wins

            encoding[batch_no, 0, all_items_idx['turn_number']] = turn_number / 10
            # TODO: When changes are made to the game to update to 0.26
            # fix this to normalize to 5 instead of 10
            encoding[batch_no, 0, all_items_idx["player_lives_remaining"]] = player_lives / 10
            encoding[batch_no, 0, all_items_idx["current_gold"]] = gold / 10
            encoding[batch_no, 0, all_items_idx["wins"]] = wins / 10

            # Dimension 2: Shop pets
            for i, pet in enumerate(current_player.shop.pets):
                enc_col = 1 + i
                encoding[batch_no, enc_col, all_items_idx["attack"]] = pet.attack / 25.0 - 1.0
                encoding[batch_no, enc_col, all_items_idx["health"]] = pet.health / 25.0 - 1.0
                encoding[batch_no, enc_col, all_items_idx["cost"]] = 3.0 / 3.0 #TEMP: pet.cost / 3.0
                encoding[batch_no, enc_col, all_items_idx["level"]] = pet.level / 3.0 + pet.experience / (6.0 if pet.level == 1 else 9.0)

                encoding[batch_no, enc_col, all_items_idx[pet.name]] = 1.0


            # Dimension 3: Shop foods
            for i, food in enumerate(current_player.shop.foods):
                enc_col = 7 + i
                encoding[batch_no, enc_col, all_items_idx["attack"]] = food.attack / 25.0 - 1.0
                encoding[batch_no, enc_col, all_items_idx["health"]] = food.health / 25.0 - 1.0
                encoding[batch_no, enc_col, all_items_idx["cost"]] = 3.0 / 3.0 #TEMP: pet.cost / 3.0

                encoding[batch_no, enc_col, all_items_idx[food.name]] = 1.0

            # Dimension 4: Team
            for i, pet in enumerate(current_player.team):
                if i not in current_player.team.filled:
                    continue
                
                pet = pet.pet
                enc_col = 10 + i
                encoding[batch_no, enc_col, all_items_idx["attack"]] = pet.attack / 25.0 - 1.0
                encoding[batch_no, enc_col, all_items_idx["health"]] = pet.health / 25.0 - 1.0
                encoding[batch_no, enc_col, all_items_idx["cost"]] = 1.0 / 3.0  if pet.name not in ["pet-pig"] else 2.0 / 3.0 # Allowing pig to be sold for 2 gold
                encoding[batch_no, enc_col, all_items_idx["level"]] = pet.level / 3.0 + pet.experience / (6.0 if pet.level == 1 else 9.0)

                encoding[batch_no, enc_col, all_items_idx[pet.name]] = 1.0
                if pet.status != 'none': 
                    encoding[batch_no, enc_col, all_items_idx[pet.status]] = 1.0
                if pet.status in status_to_food:
                    encoding[batch_no, enc_col, all_items_idx[status_to_food[pet.status]]] = 1.0

        return encoding

                
