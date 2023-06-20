from model import SAPAI
from config import DEFAULT_CONFIGURATION, rollout_device, training_device, VERBOSE, N_ACTIONS

from collections import defaultdict, deque
import random
from copy import deepcopy

from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player
from sapai.battle import Battle

from model_actions import auto_order_team

import numpy as np


class PastTeamsOrganizer():
    def __init__(self, config, max_past_teams = 30):
        self.config = config
        self.active_past_teams = {i : deque([], maxlen=max_past_teams) for i in range(max(config["number_of_evaluation_turns"], config["number_of_battles_per_simulation"]))}
        self.on_deck_past_teams = {i : deque([], maxlen=max_past_teams) for i in range(max(config["number_of_evaluation_turns"], config["number_of_battles_per_simulation"]))}

        self.max_past_teams = max_past_teams

    def sample(self, n_samples : int, turn_number : int):
        if len(self.active_past_teams[turn_number]) == 0: return []
        return random.sample(self.active_past_teams[turn_number], min(n_samples, len(self.active_past_teams)))
    
    def add_on_deck(self, past_team, turn_number : int):
        self.on_deck_past_teams[turn_number].append(past_team)

    def battle_past_teams(self, player_team : Team, turn_number : int, num_battles : int) -> float:
        if len(self.active_past_teams[turn_number]) <= 0: 
            if len(player_team.filled) > 0: return 1.0
            else: return 0.0

        # Reordering the player team
        temp_player = Player(team = player_team)
        auto_order_team(temp_player)
        player_team = temp_player.team

        num_battles = min(min(num_battles, len(self.active_past_teams[turn_number])), 6)

        wins = 0
        draws = 0

        battle_list = deepcopy(self.active_past_teams[turn_number])
        random.shuffle(battle_list)
        battle_list = [battle_list[i] for i in range(num_battles)]

        for opponent_team in battle_list:
            winner = self.battle(player_team, opponent_team)
            if winner == 0: wins += 1
            elif winner == 2: draws += 1

        if (num_battles - draws) <= 0: return 0.0
        return wins / (num_battles - draws)

    def battle(self, player_team : Team, opponent_team : Team):
        # Battle the players
        pl_temp = deepcopy(player_team)
        pl_temp.move_forward()
        op_temp = deepcopy(opponent_team)
        op_temp.move_forward()

        try:
            battle = Battle(pl_temp, op_temp)
            winner = battle.battle()
        except Exception as e:
            #print(e, flush = True)
            #print(pl_temp, flush = True)
            #print(op_temp, flush = True)
            winner = 0

        return winner
    
    def update_teams(self):
        self.active_past_teams = self.on_deck_past_teams
        self.on_deck_past_teams = {i : deque([], maxlen=self.max_past_teams) for i in range(max(self.config["number_of_evaluation_turns"], self.config["number_of_battles_per_simulation"]))}

    def create_random_samples_clone(self, max_past_teams : int):
        pt_ortanizer_out = PastTeamsOrganizer(self.config, max_past_teams = max_past_teams)

        for idx, team_list in self.on_deck_past_teams.items():
            new_team_list = random.sample(team_list, min(max_past_teams, len(team_list)))
            for team in new_team_list:
                pt_ortanizer_out.add_on_deck(team, idx)

        for idx, team_list in self.active_past_teams.items():
            new_team_list = random.sample(team_list, min(max_past_teams, len(team_list)))
            for team in new_team_list:
                pt_ortanizer_out.active_past_teams[idx].append(team)
        
        return pt_ortanizer_out
    
    def is_active_teams(self):
        for i in range(max(self.config["number_of_evaluation_turns"], self.config["number_of_battles_per_simulation"])):
            if len(self.active_past_teams[i]) > 0: return True
        return False

    def set_new_maxlen(self, new_maxlen : int):
        self.max_past_teams = new_maxlen

        new_active_past_teams = {i : deque([], maxlen=new_maxlen) for i in range(max(self.config["number_of_evaluation_turns"], self.config["number_of_battles_per_simulation"]))}
        new_on_deck_past_teams = {i : deque([], maxlen=new_maxlen) for i in range(max(self.config["number_of_evaluation_turns"], self.config["number_of_battles_per_simulation"]))}

        for i in range(max(self.config["number_of_evaluation_turns"], self.config["number_of_battles_per_simulation"])):
            new_active_past_teams[i].extend(self.active_past_teams[i])
            new_on_deck_past_teams[i].extend(self.on_deck_past_teams[i])

        self.active_past_teams = new_active_past_teams
        self.on_deck_past_teams = new_on_deck_past_teams