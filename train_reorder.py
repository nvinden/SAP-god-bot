import sys
import os
from copy import deepcopy
import time
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict
from itertools import chain
import datetime
import pickle

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed


current_directory = os.getcwd()
sys.path.append(current_directory + '/sapai_gym')

from model_actions import *
from model import SAPAI
from config import DEFAULT_CONFIGURATION, rollout_device, training_device, VERBOSE, N_ACTIONS, USE_WANDB
from eval import evaluate_model, visualize_rollout, test_legal_move_masking, get_best_legal_move, create_epoch_illegal_mask
from past_teams import PastTeamsOrganizer

from sapai.shop import Shop
from sapai.teams import Team
from sapai.player import Player
from sapai.battle import Battle
import math

import wandb


def train():
    pass


if __name__ == "__main__":
    train()