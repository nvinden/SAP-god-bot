from model_actions import *
from model import SAPAI
from config import DEFAULT_CONFIGURATION, rollout_device, training_device, VERBOSE, N_ACTIONS, USE_WANDB
from eval import evaluate_model, visualize_rollout, test_legal_move_masking, action_idx_to_string
from past_teams import PastTeamsOrganizer

from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player
from sapai.battle import Battle
from sapai.foods import Food

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from Xlib import display, X

import json
import subprocess
import re
import time

from Xlib import X, display

from pynput import keyboard

LOG_PATH = "/home/nick/.config/unity3d/Team Wood/Super Auto Pets/Player.log"
TARGET_WINDOW_TITLE = "Super Auto Pets"
MODEL_PATH = "Final Stage Attempt 1_6emsiuqy/model_test_144.pt"


# Internal SAP id to string

str_to_pet_id = { 
    'ant': 0,
    'beaver': 3,
    'cricket': 17,
    'duck': 26,
    'fish': 32,
    'horse': 39,
    'mosquito': 47,
    'otter': 51,
    'pig': 59,
    'sloth': 71,
    'crab': 16,
    'dodo': 21,
    'dog': 22,
    'elephant': 28,
    'flamingo': 29,
    'hedgehog': 37,
    'peacock': 54,
    'rat': 57,
    'shrimp': 67,
    'spider': 74,
    'swan': 76,
    'badger': 2,
    'blowfish': 7,
    'camel': 10,
    'giraffe': 33,
    'kangaroo': 40,
    'ox': 52,
    'rabbit': 60,
    'sheep': 68,
    'snail': 72,
    'turtle': 80,
    'whale': 81,
    'bison': 5,
    'deer': 20,
    'dolphin': 23,
    'hippo': 38,
    'monkey': 46,
    'penguin': 56,
    'rooster': 63,
    'skunk': 70,
    'squirrel': 75,
    'worm': 82,
    'cow': 15,
    'crocodile': 19,
    'parrot': 53,
    'rhino': 55,
    'scorpion': 65,
    'seal': 66,
    'shark': 69,
    'turkey': 79,
    'cat': 11,
    'boar': 103,
    'dragon': 25,
    'fly': 30,
    'gorilla': 36,
    'leopard': 41,
    'mammoth': 45,
    'snake': 73,
    'tiger': 77,
    'zombie-cricket': -1,
    'bus': -1,
    'zombie-fly': -1,
    'dirty-rat': -1,
    'chick': 13,
    'ram': -1,
    'bee': 4
}

str_to_food_id = {
    "apple": 0,
    "honey": 40,
    "cupcake": 50,
    "meat-bone": 9,
    "sleeping-pill": 92,
    "garlic": 38,
    "salad-bowl": 73,
    "canned-food": 16,
    "pear": 58,
    "chili": 22,
    "chocolate": 23,
    "sushi": 82,
    "melon": 13,
    "mushroom": 51,
    "pizza": 63,
    "steak": 79,
    "milk": 49
}

str_to_effect_id = {
    "honey": 8,
    "meat-bone": 6,
    "garlic": 9,
    "chili": 5,
    "melon": 13,
    "mushroom": 1,
    "steak": 79,
    "coconut": -1
}

untrained_pet_ids = {
    "mouse": -1,
    "wolverine": -1,
    "armadillo": 166

}

pet_id_to_str = {v: k for k, v in str_to_pet_id.items()}
food_id_to_str = {v: k for k, v in str_to_food_id.items()}
effect_id_to_str = {v: k for k, v in str_to_effect_id.items()}

def get_last_log(log_path):
    # Opening Log
    with open(log_path, "r") as f:
        lines = f.readlines()
        # Read lines backwards
        lines.reverse()
        for line in lines:
            if "{\"Id\":" in line:
                log_dict = json.loads(line)
                return log_dict
    return None

def create_player_from_log(log_dict):
    if log_dict is None:
        return None

    # Team members
    team_pets = []
    for i, pet in enumerate(log_dict["Mins"]["Items"]):
        if pet is None:
            continue

        pet_id = pet["Enu"]
        effect_id = pet["Perk"]

        if pet_id is not None and pet_id not in str_to_pet_id.values():
            print(f"Unknown pet id: {pet_id} at team spot {i}")
            continue

        if effect_id is not None and effect_id not in str_to_effect_id.values():
            print(f"Unknown effect id: {effect_id} at team spot {i}")
            effect_id = None

        pet_name = pet_id_to_str[pet["Enu"]]

        pet_perm_attack = pet["At"]["Perm"]
        pet_perm_health = pet["Hp"]["Perm"]
        pet_temp_attack = pet["At"]["Temp"]
        pet_temp_health = pet["Hp"]["Temp"]

        pet_xp = pet["Exp"]
        pet_level = pet["Lvl"]

        pet_effect = None if effect_id is None else effect_id_to_str[pet["Perk"]]

        pet = Pet("pet-" + pet_name)
        pet.set_attack(pet_perm_attack + pet_temp_attack)
        pet.set_health(pet_perm_health + pet_temp_health)

        pet.experience = pet_xp
        pet.level = pet_level

        if pet_effect is not None:
            food_to_eat = Food("food-" + pet_effect)
            pet.eat(food_to_eat)

        team_pets.append(pet)

    # Shop pets
    shop_pets = []
    frozen_items = []
    item_prices = []
    for i, pet in enumerate(log_dict["MiSh"]):
        if pet is None:
            continue

        pet_id = pet["Enu"]

        if pet_id is not None and pet_id not in str_to_pet_id.values():
            print(f"Unknown pet id: {pet_id} at shop spot {i}")
            continue

        pet_name = pet_id_to_str[pet["Enu"]]
        pet_price = pet["Pri"]
        pet_frozen = pet["Fro"]

        pet_perm_attack = pet["At"]["Perm"]
        pet_perm_health = pet["Hp"]["Perm"]
        pet_temp_attack = pet["At"]["Temp"]
        pet_temp_health = pet["Hp"]["Temp"]

        pet_xp = pet["Exp"]
        pet_level = pet["Lvl"]

        pet = Pet("pet-" + pet_name)
        pet.set_attack(pet_perm_attack + pet_temp_attack)
        pet.set_health(pet_perm_health + pet_temp_health)

        pet.experience = pet_xp
        pet.level = pet_level

        item_prices.append(pet_price)

        if pet_frozen:
            frozen_items.append(i)
        
        shop_pets.append(pet)

    foods = []
    for i, food in enumerate(log_dict["SpSh"]):
        
        food_id = food["Enu"]

        if food_id is not None and food_id not in str_to_food_id.values():
            print(f"Unknown food id: {food_id} at shop spot {i}")
            continue
        
        food_name = food_id_to_str[food["Enu"]]
        food_frozen = food["Fro"]
        food_price = food["Pri"]

        if food_frozen:
            frozen_items.append(i + len(shop_pets))

        food = Food("food-" + food_name)
        foods.append(food)

        item_prices.append(food_price)

    shop = Shop(shop_pets + foods)
    team = Team(team_pets)

    # Freezing and setting prices
    for i, (slot, price) in enumerate(zip(shop.slots, item_prices)):
        if i in frozen_items:
            slot.frozen = True

        slot.cost = price

    player = Player(shop = shop, team = team)

    # Player attributes
    player.gold = log_dict["Go"]
    player.turn = log_dict["Tur"]
    player.wins = log_dict["Vic"]
    player.lives = log_dict["LiMa"] - log_dict["Los"] + 0 if log_dict["Rec"] is None else log_dict["Rec"]

    return player

def get_window_info(window_title):
    command = ["xdotool", "search", "--name", window_title]
    try:
        window_id = subprocess.check_output(command).decode().strip()
    except subprocess.CalledProcessError:
        return None

    if not window_id:
        return None

    command = ["xdotool", "getwindowgeometry", window_id]
    geometry_output = subprocess.check_output(command).decode()

    pattern = r"Position: (\d+),(\d+) \(.*\)\n\s+Geometry: (\d+)x(\d+)"
    match = re.search(pattern, geometry_output)
    
    if match:
        width = int(match.group(3))
        height = int(match.group(4))
        x = int(match.group(1))
        y = int(match.group(2))
        return {'x': x, 'y': y, 'width': width, 'height': height}

    return None

def get_dimensions_for_window():
    sap_coordinates = get_window_info(TARGET_WINDOW_TITLE)

    if sap_coordinates is None:
       return None

    X_START_SCALING_FACTOR = 0.45
    X_END_SCALING_FACTOR = 0.85

    Y_START_SCALING_FACTOR = 0.00
    Y_END_SCALING_FACTOR = 0.075
    
    if sap_coordinates is None:
        return None
    
    x_min = sap_coordinates["x"]
    x_max = sap_coordinates["x"] + sap_coordinates["width"]

    y_min = sap_coordinates["y"] - 65
    y_max = sap_coordinates["y"] + sap_coordinates["height"] - 65
    
    x = int((x_max - x_min) * X_START_SCALING_FACTOR) + x_min
    width = int((x_max - x_min) * X_END_SCALING_FACTOR) - x + x_min

    y = int((y_max - y_min) * Y_START_SCALING_FACTOR) + y_min
    height = int((y_max - y_min) * Y_END_SCALING_FACTOR) - y + y_min

    return {"x": x, "y": y_min, "width": width, "height": height}


## Window Stuff
class Overlay(QWidget):
    def __init__(self, model_path : str, show_window : bool):
        super().__init__()

        # Set the window flags to make the overlay stay on top
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.WindowStaysOnTopHint)

        # Set the initial position of the overlay (top right corner)
        SAP_helper_coordinates = get_dimensions_for_window()
        if SAP_helper_coordinates is None:
            self.setGeometry(0, 0, 200, 100)
        else:
            self.setGeometry(SAP_helper_coordinates["x"], SAP_helper_coordinates["y"], SAP_helper_coordinates["width"], SAP_helper_coordinates["height"])
        self.past_sap_coordinates = SAP_helper_coordinates

        log_dict = get_last_log(LOG_PATH)
        self.player = create_player_from_log(log_dict)
        
        self.model = SAPAI(config = DEFAULT_CONFIGURATION)
        loaded_values = torch.load(model_path)
        self.model.load_state_dict(loaded_values["policy_model"])

        self.show_window = show_window

        if self.show_window:
            self.show()
        else:
            self.hide()

        print(self.player)

    def update_window(self):
        if not self.show_window:
            return

        SAP_helper_coordinates = get_dimensions_for_window()
        if SAP_helper_coordinates is None:
            if self.past_sap_coordinates is None:
                SAP_helper_coordinates = {"x": 0, "y": 0, "width": 200, "height": 100}
            else:
                SAP_helper_coordinates = self.past_sap_coordinates

        x, y, width, height = SAP_helper_coordinates["x"], SAP_helper_coordinates["y"], SAP_helper_coordinates["width"], SAP_helper_coordinates["height"]

        self.setGeometry(x, y, width, height)
        self.check_application_focus()

        self.past_sap_coordinates = SAP_helper_coordinates

    def mousePressEvent(self, event):
        event.ignore()

    def check_application_focus(self):
        # Get the active window using Xlib
        display_obj = display.Display()
        active_window = display_obj.get_input_focus().focus

        # Get the window title of the active window
        window_title = active_window.get_wm_name() if active_window else ""

        # Check if the target application is in focus
        if TARGET_WINDOW_TITLE.lower() in window_title.lower():
            self.show()
        else:
            self.hide()

    def update_team(self):
        last_team_log = get_last_log(LOG_PATH)
        self.player = create_player_from_log(last_team_log)

    def get_move_type(self, move_index : int):
        if move_index == 0:
            return "roll"

        for i, (beg_index, action_type) in enumerate(zip(action_beginning_index, action2index.keys())):
            if move_index < beg_index:
                return list(action2index.keys())[i - 1]

        return list(action2index.keys())[-1]

    def make_best_move(self):
        q_values, best_move = self.model.get_masked_q_values(self.player, epoch = 1, return_best_move = True)

        top_values = torch.topk(q_values['all'], 5)
        top_q = top_values[0][0].detach().cpu().numpy()
        top_action_idx = top_values[1][0].detach().cpu().numpy()
        top_values = [f"{float(top_q[i]):.2f} {action_idx_to_string(top_action_idx[i])}" for i in range(5)]

        if self.get_move_type(int(best_move["all"])) == "buy_food":
            print(f"Put food on {idx2pet[int(best_move['food'])]}")

        if self.get_move_type(int(best_move["all"])) == "buy_pet":
            print(f"Sell pet {idx2pet[int(best_move['sell'])]} if necessary")

        print(f"Top 5 Q-Values:\n {top_values}")



class WorkerThread(QThread):
    finished = pyqtSignal()

    def __init__(self, sap_helper : Overlay):
        super().__init__()
        self.sap_helper = sap_helper

    def run(self):
        # Create a listener
        def on_press(key):
            if key == keyboard.Key.ctrl_l:
                # Your function logic here
                print("Left Control Pressed")
                time.sleep(0.05)
                self.sap_helper.update_team()
                print(self.sap_helper.player)
                move_decision = self.sap_helper.make_best_move()

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        #listener.join()


        # Your code that needs to run while the application is running
        while True:
            #print("Thread is running")
            self.sap_helper.update_window()
            time.sleep(0.05)

def deploy():
    SHOW_WINDOW = False

    app = QApplication([])
    overlay = Overlay(model_path = MODEL_PATH, show_window = SHOW_WINDOW)

    thread = WorkerThread(overlay)
    thread.finished.connect(app.quit)
    thread.start()

    sys.exit(app.exec_())
        
if __name__ == "__main__":
    # Load the model
    deploy()