from LearningPlayer import *
from NFSPStorage import *

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

    @staticmethod
    def action_to_card(network_output: int):
        if network_output == 36:
            return -1, -1
        return network_output % 9 + 6, int(network_output / 9)

    def __init__(self, hand_size, name):
        # todo pick storage size that is large enough
        self.capacity = 100000
        self.multi_step = 100
        self.learning_rate = 1e-4
        super().__init__(hand_size, name)
        self.current_model = DQN(False)
        self.target_model = DQN(False)
        self.policy = Policy()
        self.reservoir_buffer = ReservoirBuffer(self.capacity)
        self.state_deque = deque(maxlen=self.multi_step)
        self.reward_deque = deque(maxlen=self.multi_step)
        self.action_deque = deque(maxlen=self.multi_step)
        self.rl_optimizer = optim.Adam(
            self.current_model.parameters(), lr=self.learning_rate)
        self.sl_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.length_list = []
        self.reward_list = []
        self.rl_loss_list = []
        self.sl_loss_list = []
        self.eta = 0.5  # todo : pick eta
        self.eps_start = 0.9
        self.eps_final = 0.2
        self.eps_decay = 0.01  # todo : pick parameters that make sense
        self.round = 1

    def act(self, table, legal_cards_to_play):
        legal_cards = [0] * 37
        for card in legal_cards_to_play:
            # -1 means no card
            if card[0] == -1:
                legal_cards[36] = 1
            else:
                legal_cards[card[0] - 6 + card[1] * 9] = 1
        self.is_best_response = False
        if random.random() > self.eta:
            # todo - check type of legal_cards
            action = self.policy.act(torch.FloatTensor(legal_cards), legal_cards)
        else:
            self.is_best_response = True
            action = self.current_model.act(torch.FloatTensor(legal_cards), self.epsilon_by_round(), legal_cards)
        return NFSPPlayer.action_to_card(action)

    def attack(self, table, legal_cards_to_play):
        return self.act(table, legal_cards_to_play)

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return self.act(table, legal_cards_to_play)

    def epsilon_by_round(self):
        return self.eps_final + (self.eps_start - self.eps_final) * m.exp(-1. * self.round / self.eps_decay)

    def learn_step(self, old_state, new_state, reward, info):
        pass
