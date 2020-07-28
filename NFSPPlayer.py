from LearningPlayer import LearningPlayer


class NFSPPlayer(LearningPlayer):
    def __init__(self, hand_size, name, env):
        super().__init__(hand_size, name)
        self.__env = env
from LearningPlayer import LearningPlayer
import torch
import torch.optim as optim
import torch.nn.functional as F

import time
import os
import random
import numpy as np
from collections import deque

#from common.utils import update_target, print_log, load_model, save_model
from NFSPModel import DQN, Policy
#from storage import ReplayBuffer, ReservoirBuffer
import math as m


class NFSPPlayer(LearningPlayer):
    def __init__(self, hand_size, name):
        super().__init__(hand_size, name)
        self.current_model = DQN(False).to(args.device)
        self.target_model = DQN(False).to(args.device)
        self.policy = Policy().to(args.device)
        self.reservoir_buffer = ReservoirBuffer(args.buffer_size)
        self.state_deque = deque(maxlen=args.multi_step)
        self.reward_deque = deque(maxlen=args.multi_step)
        self.action_deque = deque(maxlen=args.multi_step)
        self.rl_optimizer = optim.Adam(
            self.current_model.parameters(), lr=args.lr)
        self.sl_optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.length_list = []
        self.reward_list = []
        self.rl_loss_list = []
        self.sl_loss_list = []
        self.eta = 0.5  # todo : pick eta
        self.eps_start = 0.9
        self.eps_final = 0.2
        self.eps_decay = 0.01  # todo : pick parameters that make sense
        self.round = 1

    def attack(self, table, legal_cards_to_play):
        legal_cards = [0]*37
        for card in legal_cards:
            if card[0] == -1:
                legal_cards[36] = 1
            else:
                legal_cards[card[0]-6 + card[1] * 9]
        is_best_response = False
        if random.random() > self.eta:
            # todo - check type of legal_cards
            action = self.policy.act(torch.tensor(legal_cards))
        else:
            is_best_response = True
            action = self.current_model.act(torch.FloatTensor(legal_cards),self.epsilon_by_round())

    def epsilon_by_round(self):
        return self.eps_final + (self.eps_start - self.eps_final) * m.exp(-1. * self.round / self.eps_decay)
