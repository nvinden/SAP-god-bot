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

import numpy as np

from sapai.data import data as ALL_DATA

from config import rollout_device, training_device, N_ACTIONS
from model_actions import *

class SAPAI(nn.Module):
    def __init__(self, config = None, phase : str = "rollout"):
        super(SAPAI, self).__init__()

        # config is none, import default config, and use that
        if config is None:
            from config import DEFAULT_CONFIGURATION
            config = DEFAULT_CONFIGURATION

        self.phase = phase

        self.config = config

        # Round up number_of_items so it is divisable by self.config['nhead']
        self.n_tokens = 15
        self.d_model = number_of_items + self.n_tokens + (self.config['nhead'] - (number_of_items + self.n_tokens) % self.config['nhead'])

        # Initializing the neural network
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.config['nhead'], batch_first = True).to(self.get_device())
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = transformer_encoder_layer, num_layers = self.config['num_layers']).to(self.get_device())

        self.actions = nn.Linear(in_features = self.d_model * self.n_tokens, out_features = N_ACTIONS).to(self.get_device())

    def set_device(self, phase : str):
        assert phase == "rollout" or phase == "training"

        self.phase = phase

        if phase == "rollout":
            self.to(rollout_device)
        elif self.phase == "training":
            self.to(training_device)
        else:
            raise Exception("phase is not rollout or training")

    def get_device(self):
        if self.phase == "rollout":
            return rollout_device
        elif self.phase == "training":
            return training_device
        else:
            raise Exception("phase is not rollout or training")
        

    def forward(self, x : torch.FloatTensor, headless : bool = False) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        x = x.to(self.get_device())
        out_encoder = self.transformer_encoder(x)
        out_encoder = out_encoder.reshape(out_encoder.shape[0], -1)

        if headless:
            return out_encoder

        actions = self.actions(out_encoder)

        return actions

    def state_to_encoding(self, player) -> torch.FloatTensor:
        # If player is a player
        if isinstance(player, Player):
            player_list = [player]
        # If player is a list of players
        elif isinstance(player, list):
            player_list = player
        elif isinstance(player, np.ndarray):
            player_list = player.tolist()
        else:
            raise Exception("player is not a player or a list of players")

        encoding = torch.zeros(size = (len(player_list), self.n_tokens, self.d_model - self.n_tokens), dtype = torch.float32).to(self.get_device())
        encoding -= 1.0

        for batch_no, current_player in enumerate(player_list):
            # Dimension 1: Turn number, player health remaining, gold and wins

            encoding[batch_no, 0, all_items_idx['turn_number']] = current_player.turn / 10
            # TODO: When changes are made to the game to update to 0.26
            # fix this to normalize to 5 instead of 10
            encoding[batch_no, 0, all_items_idx["player_lives_remaining"]] = current_player.lives / 10
            encoding[batch_no, 0, all_items_idx["current_gold"]] = current_player.gold / 10
            encoding[batch_no, 0, all_items_idx["wins"]] = current_player.wins / 10

            i = 0
            # Dimension 2: Shop pets
            for slot in current_player.shop.slots:
                if slot.slot_type != "pet" or slot.obj.name == "pet-none":
                    continue
                enc_col = 1 + i
                encoding[batch_no, enc_col, all_items_idx["attack"]] = slot.obj.attack / 50.0
                encoding[batch_no, enc_col, all_items_idx["health"]] = slot.obj.health / 50.0
                encoding[batch_no, enc_col, all_items_idx["cost"]] = slot.cost / 3.0 #TEMP: pet.cost / 3.0
                encoding[batch_no, enc_col, all_items_idx["level"]] = slot.obj.level / 3.0 + slot.obj.experience / (6.0 if slot.obj.level == 1 else 9.0)
                encoding[batch_no, enc_col, all_items_idx[slot.obj.name]] = 1.0
                i += 1

            i = 0
            # Dimension 3: Shop foods
            for slot in current_player.shop.slots:
                if slot.slot_type != "food":
                    continue
                enc_col = 7 + i
                encoding[batch_no, enc_col, all_items_idx["attack"]] = slot.obj.attack / 3.0
                encoding[batch_no, enc_col, all_items_idx["health"]] = slot.obj.health / 3.0
                encoding[batch_no, enc_col, all_items_idx["cost"]] = slot.cost / 3.0 #TEMP: pet.cost / 3.0
                encoding[batch_no, enc_col, all_items_idx[slot.obj.name]] = 1.0
                i += 1

            i = 0
            # Dimension 4: Team
            for i, pet in enumerate(current_player.team):
                if i not in current_player.team.filled:
                    continue
                
                pet = pet.pet
                enc_col = 10 + i
                encoding[batch_no, enc_col, all_items_idx["attack"]] = pet.attack / 50.0
                encoding[batch_no, enc_col, all_items_idx["health"]] = pet.health / 50.0
                encoding[batch_no, enc_col, all_items_idx["cost"]] = 1.0 / 3.0  if pet.name not in ["pet-pig"] else 2.0 / 3.0 # Allowing pig to be sold for 2 gold
                encoding[batch_no, enc_col, all_items_idx["level"]] = pet.level / 3.0 + pet.experience / (6.0 if pet.level == 1 else 9.0)

                encoding[batch_no, enc_col, all_items_idx[pet.name]] = 1.0
                if pet.status != 'none': 
                    encoding[batch_no, enc_col, all_items_idx[pet.status]] = 1.0
                if pet.status in status_to_food:
                    encoding[batch_no, enc_col, all_items_idx[status_to_food[pet.status]]] = 1.0

        positional_encodings = self.get_positional_encodings(len(player_list))
        encoding = torch.cat((positional_encodings, encoding), dim = 2)

        return encoding
    
    def get_positional_encodings(self, batch_size) -> torch.FloatTensor:
        positional_encodings = torch.zeros(size = (batch_size, self.n_tokens, self.n_tokens), dtype = torch.float32).to(self.get_device())
        for i in range(self.n_tokens):
            positional_encodings[:, i, i] = 1.0
        return positional_encodings
                
