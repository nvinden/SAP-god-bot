import torch
import torch.nn as nn

import sys
import os
from copy import deepcopy

current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from sapai.pets import Pet
from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player, GoldException, WrongObjectException, FullTeamException

import numpy as np

from sapai.data import data as ALL_DATA

from config import rollout_device, training_device, N_ACTIONS
from model_actions import all_items_idx, status_to_food, idx2food, pet2idx

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
        self.d_model = config["d_model"] + (self.config['nhead'] - config["d_model"] % self.config['nhead'])

        # Initializing the neural network
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.config['nhead'], batch_first = True).to(self.get_device())
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = transformer_encoder_layer, num_layers = self.config['num_layers']).to(self.get_device())

        self.actions = nn.Linear(in_features = self.d_model * self.n_tokens, out_features = N_ACTIONS).to(self.get_device())
        self.food_actions = nn.Sequential(
            nn.Linear(in_features = self.d_model * self.n_tokens, out_features = self.d_model * self.n_tokens // 2),
            nn.BatchNorm1d(self.d_model * self.n_tokens // 2),
            nn.ReLU(),
            nn.Linear(in_features = self.d_model * self.n_tokens // 2, out_features = self.d_model * self.n_tokens // 4),
            nn.BatchNorm1d(self.d_model * self.n_tokens // 4),
            nn.ReLU(),
            nn.Linear(in_features = self.d_model * self.n_tokens // 4, out_features = len(pet2idx))
        ).to(self.get_device())
        #nn.Linear(in_features = self.d_model * self.n_tokens, out_features = len(pet2idx)).to(self.get_device())

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
        

    def forward(self, x : torch.FloatTensor, headless : bool = False, return_food_actions : bool = False) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        x = x.to(self.get_device())
        out_encoder = self.transformer_encoder(x)
        out_encoder = out_encoder.reshape(out_encoder.shape[0], -1)

        if headless:
            return out_encoder

        actions = self.actions(out_encoder)

        if return_food_actions:
            food_actions = self.food_actions(out_encoder)
            return actions, food_actions

        return actions

    def state_to_encoding(self, player, food_selected_name = None) -> torch.FloatTensor:
        # If player is a player
        if isinstance(player, Player):
            player_list = [player]
            assert food_selected_name is None or isinstance(food_selected_name, str) or (isinstance(food_selected_name, list) and len(food_selected_name) == 1)
        # If player is a list of players
        elif isinstance(player, list):
            player_list = player
            assert food_selected_name is None or isinstance(food_selected_name, list)
        elif isinstance(player, np.ndarray):
            player_list = player.tolist()
            assert food_selected_name is None or isinstance(food_selected_name, list)
        else:
            raise Exception("player is not a player or a list of players")
        
        if isinstance(food_selected_name, str):
            food_name_list = [food_selected_name]
        elif isinstance(food_selected_name, list):
            food_name_list = food_selected_name

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
            if current_player.lf_winner is not None:
                encoding[batch_no, 0, all_items_idx["wins"] + 1] = 1.0 if current_player.lf_winner == True else 0.0 # just sticking it on to the end dont worry about it

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

        if food_selected_name is not None:
            encoding = self.add_food_index_to_encoding(encoding, food_name_list)

        return encoding
    
    def add_food_index_to_encoding(self, encoding, food_name_list):
        assert encoding.shape[0] == len(food_name_list)

        encoding = deepcopy(encoding)

        for batch_no, food_index in enumerate(food_name_list):
            encoding[batch_no, 0, all_items_idx[food_index]] = 1.0
        
        return encoding

    def get_positional_encodings(self, batch_size) -> torch.FloatTensor:
        positional_encodings = torch.zeros(size = (batch_size, self.n_tokens, self.n_tokens), dtype = torch.float32).to(self.get_device())
        for i in range(self.n_tokens):
            positional_encodings[:, i, i] = 1.0
        return positional_encodings
                
class SAPAIReorder(nn.Module):
    def __init__(self, config):
        super(SAPAIReorder, self).__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.num_encoder_layers = config["num_encoder_layers"]
        self.num_decoder_layers = config["num_decoder_layers"]
        self.dim_feedforward = config["dim_feedforward"]
        self.n_tokens = 5

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transformer = nn.Transformer(d_model = self.d_model, nhead = self.nhead, num_encoder_layers = self.num_encoder_layers, 
                                          num_decoder_layers = self.num_decoder_layers, dim_feedforward = self.dim_feedforward, batch_first=True).to(self.device)
        self.out = nn.Linear(self.d_model, self.n_tokens).to(self.device)

    def forward(self, encoder_encodings : torch.Tensor, decoder_encodings : torch.Tensor, softmax : bool = False):
        if len(encoder_encodings.shape) == 2:
            encoder_encodings = encoder_encodings.unsqueeze(0)

        if len(decoder_encodings.shape) == 2:
            decoder_encodings = decoder_encodings.unsqueeze(0)

        decoder_mask = nn.Transformer.generate_square_subsequent_mask(decoder_encodings.shape[1]).to(self.device)

        out = self.transformer(encoder_encodings, decoder_encodings, tgt_mask = decoder_mask)
        out = out[:, :-1]
        out = self.out(out)
        if softmax:
            out = nn.functional.softmax(out, dim = 2)

        return out
    
    def loss(self, out, target):
        if isinstance(target, list):
            target = np.array(target)
    
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype = torch.long).to(self.device)
        
        loss = nn.functional.cross_entropy(out, target)
        return loss

    def generate_input_encodings(self, team_list):
        if isinstance(team_list, Team):
            team_list = [team_list]
        # If player is a list of players
        elif isinstance(team_list, list):
            team_list = team_list
        elif isinstance(team_list, np.ndarray):
            team_list = team_list.tolist()
        else:
            raise Exception("player is not a player or a list of players")
        
        encoding = torch.zeros(size = (len(team_list), self.n_tokens, self.d_model - 5), dtype = torch.float32).to(self.device)
        encoding -= 1.0

        for batch_no, team in enumerate(team_list):
            # Dimension 4: Team
            filled_team = [i for i in range(5) if team.slots[i].pet.name != "pet-none"]
            for enc_col, pet in enumerate(team):
                if enc_col not in filled_team:
                    continue
                
                pet = pet.pet
                encoding[batch_no, enc_col, all_items_idx["attack"]] = pet.attack / 50.0
                encoding[batch_no, enc_col, all_items_idx["health"]] = pet.health / 50.0
                encoding[batch_no, enc_col, all_items_idx["cost"]] = 1.0 / 3.0  if pet.name not in ["pet-pig"] else 2.0 / 3.0 # Allowing pig to be sold for 2 gold
                encoding[batch_no, enc_col, all_items_idx["level"]] = pet.level / 3.0 + pet.experience / (6.0 if pet.level == 1 else 9.0)

                encoding[batch_no, enc_col, all_items_idx[pet.name]] = 1.0
                if pet.status != 'none': 
                    encoding[batch_no, enc_col, all_items_idx[pet.status]] = 1.0
                if pet.status in status_to_food:
                    encoding[batch_no, enc_col, all_items_idx[status_to_food[pet.status]]] = 1.0
        
        positional_encodings = self.get_positional_encodings(len(team_list), n_tokens = self.n_tokens)
        encoding = torch.cat((encoding, positional_encodings), dim = 2)

        return encoding
    
    def inference(self, input_encoding):
        if len(input_encoding.shape) == 2:
            input_encoding = input_encoding.unsqueeze(0) 

        moves_made = np.zeros(shape = (input_encoding.shape[0], 5), dtype = np.int32) - 1
        moves_already_mask = np.zeros(shape = (input_encoding.shape[0], 5), dtype = np.int32)

        for i in range(5):
            output_encodings = self.generate_output_encodings(moves_made)

            out_mask = nn.Transformer.generate_square_subsequent_mask(output_encodings.shape[1]).to(self.device)
            output = self.transformer(input_encoding, output_encodings, tgt_mask = out_mask)
            output = self.out(output)
            output_at_timestep = output[:, i, :].cpu().detach().numpy()

            # Masking out the moves that have already been made
            output_at_timestep -= moves_already_mask * 1e9

            chosen_move = np.argmax(output_at_timestep, axis = 1)

            moves_made[:, i] = chosen_move
            for batch_no in range(input_encoding.shape[0]):
                moves_already_mask[batch_no, chosen_move[batch_no]] = 1

        return moves_made, output
    
    def generate_output_encodings(self, moves_made):
        if isinstance(moves_made, list):
            moves_made = np.array(moves_made)

        if len(moves_made.shape) == 1:
            moves_made = np.expand_dims(moves_made, axis = 0)

        assert moves_made.shape[1] == 5

        encoding = torch.zeros(size = (moves_made.shape[0], self.n_tokens + 1, self.d_model - 6), dtype = torch.float32).to(self.device)
        encoding[:, 0, 0] = 1.0

        for order_number in range(5):
            current_moves = moves_made[:, order_number]
            # if all of the moves are -1, then we stop
            if np.all(current_moves[:] == -1):
                break

            if np.any(current_moves[:] == -1):
                raise Exception("Order number is -1")
            
            for batch_no, move in enumerate(current_moves):
                encoding[batch_no, order_number + 1, move + 1] = 1.0

        positional_encodings = self.get_positional_encodings(moves_made.shape[0], n_tokens = self.n_tokens + 1)
        encoding = torch.cat((encoding, positional_encodings), dim = 2)

        return encoding

    
    def get_positional_encodings(self, batch_size, n_tokens) -> torch.FloatTensor:
        positional_encodings = torch.zeros(size = (batch_size, n_tokens, n_tokens), dtype = torch.float32).to(self.device)
        for i in range(self.n_tokens):
            positional_encodings[:, i, i] = 1.0
        return positional_encodings

        
