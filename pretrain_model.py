from config import PRETRAIN_DEFAULT_CONFIGURATION
from model import SAPAI
from model_actions import pet2idx, idx2pet, food2idx, idx2food

from copy import deepcopy
import random
import time
from itertools import chain
from collections import defaultdict

from sapai.teams import Team
from sapai.player import Player
from sapai.battle import Battle
from sapai.foods import Food

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_FEATURES = 1800

# 1) NoP: Number of Pets
def NoP_net(config):
    net = nn.Linear(in_features = IN_FEATURES, out_features = 1, device="cuda")
    return net

def NoP_get_gt(player : Player):
    return [len(player.team.filled)]

# 2) TSofP: Total Stats of Pets
def TSofP_net(config):
    net = nn.Linear(in_features = IN_FEATURES, out_features = 1, device="cuda")
    return net

def TSofP_get_gt(player : Player):
    total_stats = 0
    for pet_no in player.team.filled:
        total_stats += player.team.slots[pet_no].pet.health + player.team.slots[pet_no].pet.attack
    return [total_stats]

# 3) TN: Turn Number
def TN_net(config):
    net = nn.Linear(in_features = IN_FEATURES, out_features = 1, device="cuda")
    return net

def TN_get_gt(player : Player):
    return [player.turn]

# 4) TG: Total Gold
def TG_net(config):
    net = nn.Linear(in_features = IN_FEATURES, out_features = 1, device="cuda")
    return net

def TG_get_gt(player : Player):
    return [player.gold]

# 5) TL: Team Level
def TL_net(config):
    net = nn.Linear(in_features = IN_FEATURES, out_features = 5, device="cuda")
    return net

def TL_get_gt(player : Player):
    team_levels = [0] * 5
    for pet_no in player.team.filled:
        team_levels[pet_no] = player.team.slots[pet_no].pet.level
    return team_levels

# 6) TE: Team Experience
def TE_net(config):
    net = nn.Linear(in_features = IN_FEATURES, out_features = 5, device="cuda")
    return net

def TE_get_gt(player : Player):
    team_exp = [0] * 5
    for pet_no in player.team.filled:
        team_exp[pet_no] = player.team.slots[pet_no].pet.experience
    return team_exp

# 7) NL: Number of Lives
def NL_net(config):
    net = nn.Linear(in_features = IN_FEATURES, out_features = 1, device="cuda")
    return net

def NL_get_gt(player : Player):
    return [player.lives]

# 8) NW: Number of Wins
def NW_net(config):
    net = nn.Linear(in_features = IN_FEATURES, out_features = 1, device="cuda")
    return net

def NW_get_gt(player : Player):
    return [player.wins]

# Utilities
def get_random_pet_percentages():
    # Generating percent chances pet shows up in shop or in team
    none_pet_share = 30.0
    random_pet_percentages = {"pet-none" : none_pet_share / (none_pet_share + len(pet2idx))}
    for pet in pet2idx.keys():
        random_pet_percentages[pet] = 1.0 / (none_pet_share + len(pet2idx))

    return random_pet_percentages

def get_random_food_percentages():
    # Generating percent chances pet shows up in shop or in team
    none_pet_share = 10.0
    random_pet_percentages = {"food-none" : none_pet_share / (none_pet_share + len(food2idx))}
    for pet in food2idx.keys():
        random_pet_percentages[pet] = 1.0 / (none_pet_share + len(food2idx))

    return random_pet_percentages

def generate_random_player(random_pet_percentages, random_food_percentages) -> Player:
    player = Player()

    # 1) Random pets in the team
    n_team_pets = random.randint(0, 5)
    pet_percentage_list = [random_pet_percentages[pet] for pet in random_pet_percentages.keys()]
    pets_list = [pet.replace("pet-", "") for pet in random_pet_percentages.keys()]
    selected_team_pets = random.choices(pets_list, pet_percentage_list, k = n_team_pets)

    team = Team(selected_team_pets)

    player = Player(team = team)

    # 2) Randomly eat foods
    food_percentage_list = [random_food_percentages[food] for food in random_food_percentages.keys()]
    foods_list = [food for food in random_food_percentages.keys()]
    selected_foods = random.choices(foods_list, food_percentage_list, k = len(player.team.filled))
    for pet_no, food in zip(player.team.filled, selected_foods):
        if food != "food-none" and food != "food-sleeping-pill":
            try:
                player.team.slots[pet_no].pet.eat(Food(food))
            except:
                pass

    # 3) Random gold
    player.gold = random.randint(0, 20)

    # 4) Random level for each pet
    for pet_no in player.team.filled:
        for _ in range(random.randint(0, 5)):
            player.team.slots[pet_no].pet.gain_experience()

    # 5) Randomly give pets more stats
    for pet_no in player.team.filled:
        if random.uniform(0, 1) < 0.3:
            player.team.slots[pet_no].pet.set_health(random.randint(player.team.slots[pet_no].pet.health, 50))
            player.team.slots[pet_no].pet.set_attack(random.randint(player.team.slots[pet_no].pet.attack, 50))

    # 6) Randomly choose shop level
    shop_level = random.randint(0, 15)
    for _ in range(shop_level):
        player.end_turn()
    
    player.gold += 1
    player.roll()

    # Randomly choose wins and lives
    player.wins = random.randint(0, 10)
    player.lives = random.randint(0, 5)

    # Check if any pet has negative health or attack
    for pet_no in player.team.filled:
        if player.team.slots[pet_no].pet.health < 0 or player.team.slots[pet_no].pet.attack < 0:
            raise Exception("Negative health or attack")

    return player

def get_random_player_list(number_of_players : int = 32):
    random_pet_percentages = get_random_pet_percentages()
    random_food_percentages = get_random_food_percentages()
    player_list = []
    while(len(player_list) < number_of_players):
        try:
            player = generate_random_player(random_pet_percentages, random_food_percentages)
            player_list.append(player)
        except:
            continue

    return player_list


# Evaluation

def evaluate_pretrained_model(model, out_nets, pretraining_actions):
    model.eval()
    count = 0
    for i in range(100):
        player = get_random_player_list(1)
        player_encodings = model.state_to_encoding(player)
        predictions = model(player_encodings, headless = True)
        print(player)
        for out_net, (name, net, get_gt, normalize_value) in zip(out_nets, pretraining_actions):
            gt = get_gt(player[0])

            out = F.sigmoid(out_net(predictions)).squeeze()
            out = np.array(out.detach().cpu().numpy())
            #out = torch.mean(out).item()

            print(f"{name}: {out * normalize_value / 5.0} vs {gt}")

        print()

        

def pretrain_model():
    config = PRETRAIN_DEFAULT_CONFIGURATION

    model = SAPAI(config = config, phase = "training")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Name, net, get_gt, normalize_value
    pretraining_actions = (
        ("NoP", NoP_net, NoP_get_gt, 5.0),
        ("TSofP", TSofP_net, TSofP_get_gt, 250.0),
        ("TN", TN_net, TN_get_gt, 5.0),
        ("TG", TG_net, TG_get_gt, 12.0),
        ("TL", TL_net, TL_get_gt, 3.0),
        ("TE", TE_net, TE_get_gt, 3.0),
        ("NL", NL_net, NL_get_gt, 5.0),
        ("NW", NW_net, NW_get_gt, 10.0)
    )

    out_nets = [action_net(config) for _, action_net, _, _ in pretraining_actions]

    # Load model
    model_file = None#"pretrained_model_2000.pth"
    if model_file is not None:
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint["model"])
        out_nets[0].load_state_dict(checkpoint["NoP_net"])
        out_nets[1].load_state_dict(checkpoint["TSofP_net"])
        out_nets[2].load_state_dict(checkpoint["TN_net"])
        out_nets[3].load_state_dict(checkpoint["TG_net"])
        out_nets[4].load_state_dict(checkpoint["TL_net"])
        out_nets[5].load_state_dict(checkpoint["TE_net"])
        out_nets[6].load_state_dict(checkpoint["NL_net"])
        out_nets[7].load_state_dict(checkpoint["NW_net"])
        evaluate_pretrained_model(model, out_nets, pretraining_actions)

    params = [model.parameters()] + [net.parameters() for net in out_nets]
    params = chain(*params)
    optim = torch.optim.Adam(params, lr = config["learning_rate"])

    loss_history = defaultdict(list)
    #player_list = get_random_player_list(number_of_players = config["batch_size"])

    for epoch in range(1, config["epochs"] + 1):
        player_list = get_random_player_list(number_of_players = config["batch_size"])
        loss_categories = {action_name : [] for action_name, _, _, _ in pretraining_actions}

        player_encodings = model.state_to_encoding(player_list)
        player_encoding_out = model.forward(player_encodings, headless = True)

        loss_total = 0.0

        for (action_name, _, get_ground_truth_function, normalize_value), action_net in zip(pretraining_actions, out_nets):
            optim.zero_grad()

            prediction = action_net(player_encoding_out)
            gt = torch.tensor([get_ground_truth_function(player) for player in player_list], device = "cuda" if torch.cuda.is_available() else "cpu")
            gt = gt / normalize_value * 5.0
            
            if len(gt.shape) == 1:
                gt = gt.unsqueeze(1)

            loss = F.mse_loss(prediction, gt)
            loss.backward(retain_graph=True)
            optim.step()

            loss_categories[action_name].append(loss.item())
            loss_total += loss.item()

        loss_history["total_loss"].append(loss_total)

        for action_name, _, _, _ in pretraining_actions:
            loss_history[action_name].append(np.mean(loss_categories[action_name]))

        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {np.mean(loss_history['total_loss'])}")
            for action_name, _, _, _ in pretraining_actions:
                print(f"    {action_name} - Loss: {np.mean(loss_history[action_name])}")
            print()

            loss_history = defaultdict(list)
            
        if epoch % 1000 == 0:
            # Save model
            torch.save({
                "model": model.state_dict(),
                "NoP_net": out_nets[0].state_dict(),
                "TSofP_net": out_nets[1].state_dict(),
                "TN_net": out_nets[2].state_dict(),
                "TG_net": out_nets[3].state_dict(),
                "TL_net": out_nets[4].state_dict(),
                "TE_net": out_nets[5].state_dict(),
                "NL_net": out_nets[6].state_dict(),
                "NW_net": out_nets[7].state_dict()
                }, f"pretrained_model_{epoch}.pth")
            print(f"Saved model to pretrained_model_{epoch}.pth")
            #evaluate_pretrained_model(model, out_nets, pretraining_actions)


if __name__ == "__main__":
    pretrain_model()