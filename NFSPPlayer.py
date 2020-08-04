import pathlib
from LearningPlayer import LearningPlayer
import math as m
from NFSPModel import DQN, Policy
from collections import deque
import numpy as np
import random
import os
import time
import torch.nn.functional as F
import torch.optim as optim
import torch
from LearningPlayer import *
from NFSPStorage import *


class NFSPPlayer(LearningPlayer):
    def __init__(self, hand_size, name, env):
        super().__init__(hand_size, name)
        self.__env = env


# from common.utils import update_target, print_log, load_model, save_model
#from storage import ReplayBuffer, ReservoirBuffer


class NFSPPlayer(LearningPlayer):

    @staticmethod
    def action_to_card(network_output: int):
        if network_output == 36:
            return -1, -1
        return network_output % 9 + 6, int(network_output / 9)

    def __init__(self, hand_size, name):
        # todo pick storage size that is large enough
        # in a game of 3 random players on average there are 120 steps per game. so for storing
        # data for 25 games we should store 20 * 120 = 2400 steps
        self.capacity = 10000  # 2400
        self.rl_learning_rate = 0.1  # 0.1
        self.sl_learning_rate = 0.005 # 0.005
        super().__init__(hand_size, name)
        self.current_model = DQN(False)
        self.target_model = DQN(False)
        self.policy = Policy()
        self.update_target = lambda current, target: target.load_state_dict(
            current.state_dict())
        self.update_target(self.current_model, self.target_model)
        self.reservoir_buffer = ReservoirBuffer(self.capacity)
        self.replay_buffer = ReplayBuffer(self.capacity)
        self.rl_optimizer = optim.Adam(
            self.current_model.parameters(), lr=self.rl_learning_rate)
        self.sl_optimizer = optim.Adam(
            self.policy.parameters(), lr=self.sl_learning_rate)
        self.length_list = []
        self.reward_list = []
        self.rl_loss_list = []
        self.sl_loss_list = []
        self.eta = 0.1  # todo : pick eta 0.1
        self.eps_start = 0.9  # 0.9 paper 0.06
        self.eps_final = 0 # 0
        self.eps_decay = 10000  # todo : pick parameters that make sense
        self.round = 1
        self.is_best_response = False
        self.batch_size = 128  # todo check for the best batch size 128
        self.discard_pile = [0]*36
        self.T = 5
        self.t = 1
        self.update_time = 150  # paper 300, 500 experince

    def act(self, table, legal_cards_to_play):
        legal_cards_vec, state = self.get_network_input(legal_cards_to_play, table, self.discard_pile, self._hand)
        self.is_best_response = False
        if random.random() > self.eta:
            action = self.policy.act(torch.FloatTensor(state), legal_cards_vec)
        else:
            self.is_best_response = True
            action = self.current_model.act(torch.FloatTensor(
                state), self.epsilon_by_round(), legal_cards_vec)
        if self.is_best_response:
            self.reservoir_buffer.push(state, action)

        return NFSPPlayer.action_to_card(action)

    @staticmethod
    def get_network_input(legal_cards_to_play, table, discard_pile, hand):
        legal_cards_vec = NFSPPlayer.get_legal_cards_as_vector(legal_cards_to_play)
        attacking_cards_vec = NFSPPlayer.get_cards_as_vector(table[0])
        defending_cards_vec = NFSPPlayer.get_cards_as_vector(table[1])
        hand_vec = NFSPPlayer.get_cards_as_vector(hand)
        not_possible_card_vec = discard_pile
        state = legal_cards_vec + attacking_cards_vec + \
                defending_cards_vec + hand_vec + not_possible_card_vec
        return legal_cards_vec, state

    @staticmethod
    def card_numeric_rep(card: Deck.CardType) -> int:
        # if card == (-1, -1) ret 36
        return card[0] - 6 + card[1] * 9 if card[0] != -1 else 36

    @staticmethod
    def get_legal_cards_as_vector(legal_cards_to_play):
        legal_cards = [0] * 37
        for card in legal_cards_to_play:
            legal_cards[NFSPPlayer.card_numeric_rep(card)] = 1
        return legal_cards

    @staticmethod
    def get_cards_as_vector(cards):
        card_vec = [0] * 36
        for card in cards:
            card_vec[NFSPPlayer.card_numeric_rep(card)] = 1
        return card_vec

    def attack(self, table, legal_cards_to_play):
        card = self.act(table, legal_cards_to_play)
        if card[0] != -1:
            self._hand.remove(card)
        return card

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        card = self.act(table, legal_cards_to_play)
        if card[0] != -1:
            self._hand.remove(card)
        return card

    def epsilon_by_round(self):
        return self.eps_final + (self.eps_start - self.eps_final) * m.exp(-1. * self.round / self.eps_decay)
        # return self.eps_start * (1 / (self.round ** (1/2)))

    def learn_step(self, old_state, new_state, action, reward, info):
        is_attacking = False
        if len(old_state[0]) < len(new_state[0]):
            is_attacking = True
        if is_attacking:
            legal_old_cards = [card for card in old_state[2]
                               if card in self._hand or card == Deck.NO_CARD or card == action]
            legal_new_cards = [card for card in new_state[2]
                               if card in self._hand or card == Deck.NO_CARD]
        else:
            legal_old_cards = [card for card in old_state[3]
                               if card in self._hand or card == Deck.NO_CARD or card == action]
            legal_new_cards = [card for card in new_state[3]
                               if card in self._hand or card == Deck.NO_CARD]
        _, old_input = self.get_network_input(legal_old_cards, old_state, self.discard_pile, self._hand)
        _, new_input = self.get_network_input(legal_new_cards, new_state, self.discard_pile, self._hand)
        self.replay_buffer.push(old_input, NFSPPlayer.card_numeric_rep(
            action), reward, new_input, 0)
        if self.is_best_response:
            self.compute_sl_loss()
            return
        # at the end of the episode logging record must be deleted
        self.compute_rl_loss()
        self.round += 1
        self.t += 1
        if self.update_time % self.round == 0:
            self.update_target(self.current_model, self.target_model)

    def compute_sl_loss(self):
        batch_size = min(self.batch_size, len(self.reservoir_buffer))
        state_array, action = self.reservoir_buffer.sample(batch_size)
        state = torch.FloatTensor(state_array)
        action = torch.LongTensor(action)

        probs = self.policy.forward(state)
        probs_with_actions = probs.gather(1, action.unsqueeze(1))
        log_probs = probs_with_actions.log()

        loss = -1 * log_probs.mean()

        self.sl_optimizer.zero_grad()
        loss.backward()
        self.sl_optimizer.step()
        return loss

    def compute_rl_loss(self):
        batch_size = min(self.batch_size, len(self.replay_buffer))
        state, action, reward, next_state, done = self.replay_buffer.sample(
            batch_size)
        weights = torch.ones(batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        weights = torch.FloatTensor(weights)

        # Q-Learning with target network
        q_values = self.current_model.forward(state)
        target_next_q_values = self.target_model.forward(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = target_next_q_values.max(1)[0]
        # todo fix expected q-value
        expected_q_value = reward + next_q_value

        # Huber Loss
        loss = F.smooth_l1_loss(
            q_value, expected_q_value.detach(), reduction='none')
        loss = (loss * weights).mean()

        self.rl_optimizer.zero_grad()
        loss.backward()
        self.rl_optimizer.step()
        return loss

    def update_end_round(self, defending_player_name: str, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
                         successfully_defended: bool) -> None:
        """
        Updates the agent about the result of the round - weather the defending player defended successfully or not.
        :param defending_player_name: Defending player's name
        :param table: Cards on the table at the end of the round (before clearing)
        :param successfully_defended: Weather the defence was successful (which means all cards are discarded), or not (which means the defending player took all cards on the table).
        """
        # if successfully_defended:
        #     cards_discarded_this_round = table[0]+table[1]
        #     cards_discarded_this_round_vec = self.get_cards_as_vector(cards_discarded_this_round)
        #     for i,card in enumerate(cards_discarded_this_round_vec):
        #         self.discard_pile[i] += card
        # pass
        
    def initialize_for_game(self) -> None:
        self.discard_pile = [0]*36

    def save_network(self, name):
        fname = os.path.join("NFSP-models", name)
        torch.save({
            'model': self.current_model.state_dict(),
            'policy': self.policy.state_dict(),
        }, fname)
