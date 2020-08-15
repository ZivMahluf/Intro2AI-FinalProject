from Types import CardType, CardListType, FieldType, TableType, StateType, NumberType
from NFSPStorage import ReplayBuffer, ReservoirBuffer
from DurakPlayer import DurakPlayer
from NFSPModel import DQNBase, Policy

import torch.nn.functional as F
import torch.optim as optim
import math as m
import random
import torch
import os
import numpy as np

from Deck import Deck
from typing import Tuple, List


class NFSPPlayer(DurakPlayer):

    def __init__(self, hand_size, name, device='cpu'):
        """
        Constructor.
        Defines the parameters for the player.
        :param hand_size: Initial hand size.
        :param name: Name of the player.
        :param device: Which device to run on ('cpu' for CPU, or 'cuda' for a NVIDIA GPU with the required drivers installed).
        """
        super().__init__(hand_size, name)
        self.capacity = 15000
        self.rl_learning_rate = 0.01
        self.sl_learning_rate = 0.0005
        self.current_model = DQNBase()
        self.target_model = DQNBase()
        self.policy = Policy()
        self.update_target = lambda current, target: target.load_state_dict(current.state_dict())
        self.update_target(self.current_model, self.target_model)
        self.reservoir_buffer = ReservoirBuffer(self.capacity)
        self.replay_buffer = ReplayBuffer(self.capacity)
        self.rl_optimizer = optim.Adam(self.current_model.parameters(), lr=self.rl_learning_rate)
        self.sl_optimizer = optim.Adam(self.policy.parameters(), lr=self.sl_learning_rate)
        self.length_list = []
        self.reward_list = []
        self.rl_loss_list = []
        self.sl_loss_list = []
        self.eta = 0.1
        self.eps_start = 0.9
        self.eps_final = 0.01
        self.eps_decay = 300000
        self.round = 1
        self.is_best_response = True
        self.batch_size = 128
        self.discard_pile = [0]*36
        self.update_time = 1500
        self.device = device
        self.prev_reward = None
        self.prev_state = None
        self.prev_action = None

    def act(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        """
        Chooses an action to do based on the given table and the legal cards to play.
        :param table: (attacking cards, defending cards, number of cards in deck, number of cards in each player's hand)
        :param legal_cards_to_play: list of legal cards to play out of the player's hand, and might include Deck.NO_CARD.
        :return: The chosen card to play.
        """
        legal_cards_vec, state = self.get_network_input(legal_cards_to_play, table, self.discard_pile, self._hand)
        if self.prev_state:
            self.replay_buffer.push(self.prev_state, self.prev_action, self.prev_reward, state, legal_cards_vec, 0)

        if not self.is_best_response:
            action = self.policy.act(torch.FloatTensor(state).to(self.device), 0, legal_cards_vec)
        else:
            action = self.current_model.act(torch.FloatTensor(state).to(self.device), self.epsilon_by_round(), legal_cards_vec)

        return Deck.get_card_from_index(action)

    @staticmethod
    def get_network_input(legal_cards_to_play: CardListType, table: TableType, discard_pile: List[int], hand: CardListType) -> Tuple[List[int], List[int]]:
        """
        :return: Two indicator vectors:
        1. An indicator vector of the legal actions that can be taken.
        2. A feature representation of the table as a vector of indicators.
        """
        legal_actions_vec = NFSPPlayer.get_legal_actions_as_vector(legal_cards_to_play)
        attacking_cards_vec = NFSPPlayer.get_cards_as_vector(table[0])
        defending_cards_vec = NFSPPlayer.get_cards_as_vector(table[1])
        hand_vec = NFSPPlayer.get_cards_as_vector(hand)
        not_possible_card_vec = discard_pile
        state = legal_actions_vec + attacking_cards_vec + defending_cards_vec + hand_vec + not_possible_card_vec
        return legal_actions_vec, state

    @staticmethod
    def get_legal_actions_as_vector(legal_cards_to_play: CardListType) -> List[int]:
        """
        :return: An indicator vector of the legal actions that can be taken.
        """
        legal_cards = [0] * 37
        for card in legal_cards_to_play:
            legal_cards[Deck.get_index_from_card(card)] = 1
        return legal_cards

    @staticmethod
    def get_cards_as_vector(cards: CardListType) -> List[int]:
        """
        :param cards: A list of cards to transform.
        :return: A representation of the given list of cards as a vector of indicators with a fixed legnth.
        """
        card_vec = [0] * 36
        for card in cards:
            card_vec[Deck.get_index_from_card(card)] = 1
        return card_vec

    def attack(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        card = self.act(table, legal_cards_to_play)
        if card != Deck.NO_CARD:
            self._hand.remove(card)
        return card

    def defend(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        card = self.act(table, legal_cards_to_play)
        if card != Deck.NO_CARD:
            self._hand.remove(card)
        return card

    def epsilon_by_round(self) -> float:
        """
        :return: epsilon for epsilon-greedy q-learning, as a function of the round.
        """
        return self.eps_final + (self.eps_start - self.eps_final) * m.exp(-1. * self.round / self.eps_decay)

    def learn_step(self, old_state: StateType, new_state: StateType, action: CardType, reward: NumberType, old_hand) -> None:
        """
        Preforms a learning step using the previous and next states, the action taken, and the reward.
        :param old_state: previous state
        :param new_state: new state resulting from taking the action in the previous state
        :param action: action taken in the previous state
        :param reward: reward for the action
        :param old_hand: cards in hand in the previous state
        """
        if action != Deck.NO_CARD:
            is_attacking = len(old_state[0]) < len(new_state[0])
        else:
            card_to_check = old_state[0][0]
            is_attacking = card_to_check not in self._hand
        if is_attacking:
            legal_old_cards = [card for card in old_state[2]
                               if card in self._hand or card == Deck.NO_CARD or card == action]
        else:
            legal_old_cards = [card for card in old_state[3]
                               if card in self._hand or card == Deck.NO_CARD or card == action]
        _, old_input = self.get_network_input(legal_old_cards, (old_state[0], old_state[1], old_state[4], old_state[5]), self.discard_pile, old_hand)
        self.prev_state = old_input
        self.prev_action = Deck.get_index_from_card(action)
        self.prev_reward = reward

        if self.is_best_response:
            self.reservoir_buffer.push(old_input, Deck.get_index_from_card(action))
        self.compute_sl_loss()
        # at the end of the episode logging record must be deleted
        self.compute_rl_loss()
        self.round += 1
        if self.round % self.update_time == 0:
            self.update_target(self.current_model, self.target_model)

    def end_game(self) -> None:
        """
        Appends to replay buffer at the end of the game.
        """
        state, action, reward, next_state, legal_cards, done = self.replay_buffer.buffer.pop()
        if reward < 20:
            reward = -10
        done = 1
        self.replay_buffer.buffer.append((state, action, reward, next_state, legal_cards, done))

    def compute_sl_loss(self) -> None:
        """
        Updates policy neural network
        """
        batch_size = min(self.batch_size, len(self.reservoir_buffer))
        if batch_size == 0:
            return
        state_array, action = self.reservoir_buffer.sample(batch_size)
        state = torch.FloatTensor(state_array).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        probs = self.policy.forward(state)
        probs_with_actions = probs.gather(1, action.unsqueeze(1))
        log_probs = probs_with_actions.log()

        loss = -1 * log_probs.mean()

        self.sl_optimizer.zero_grad()
        loss.backward()
        self.sl_optimizer.step()

    def compute_rl_loss(self) -> None:
        """
        Updates current_model neural network
        """
        batch_size = min(self.batch_size, len(self.replay_buffer))
        if batch_size == 0:
            return
        state, action, reward, next_state, legal_cards, done = self.replay_buffer.sample(batch_size)
        weights = torch.ones(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        legal_cards = np.array(legal_cards).flatten()

        # Q-Learning with target network
        q_values = self.current_model.forward(state)
        target_next_q_values = self.target_model.forward(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = target_next_q_values.cpu().detach().numpy().flatten()
        next_q_value[np.array(legal_cards) == 0] = -1 * float('inf')
        next_q_value = torch.FloatTensor([max(next_q_value)]).to(self.device)
        expected_q_value = reward + (next_q_value * (1-done)) * 0.95

        # Huber Loss
        loss = F.smooth_l1_loss(
            q_value, expected_q_value.detach(), reduction='none')
        loss = (loss * weights).mean()

        self.rl_optimizer.zero_grad()
        loss.backward()
        self.rl_optimizer.step()

    def update_end_round(self, defending_player_name: str, table: FieldType, successfully_defended: bool) -> None:
        """
        Updates the agent about the result of the round - weather the defending player defended successfully or not.
        :param defending_player_name: Defending player's name
        :param table: Cards on the table at the end of the round (before clearing)
        :param successfully_defended: Weather the defence was successful (which means all cards are discarded), or not (which means the defending player took all cards on the table).
        """
        self.is_best_response = random.random() > self.eta
        if successfully_defended:
            cards_discarded_this_round = table[0]+table[1]
            cards_discarded_this_round_vec = self.get_cards_as_vector(cards_discarded_this_round)
            for i, card in enumerate(cards_discarded_this_round_vec):
                self.discard_pile[i] += card

    def initialize_for_game(self) -> None:
        """
        Init discard pile
        """
        super().initialize_for_game()
        self.discard_pile = [0]*36
        self.prev_reward = None
        self.prev_action = None
        self.prev_state = None
        self.is_best_response = random.random() > self.eta

    def save_network(self, name: str) -> None:
        """
        saves the neural network for future use
        Parameters
        ----------
        name: name of the file that will store the neural network
        """
        fname = os.path.join("NFSP-models", name)
        torch.save({
            'model': self.current_model.state_dict(),
            'policy': self.policy.state_dict(),
        }, fname)

    def load_model(self, path: str) -> None:
        """
        load neural network from file
        Parameters
        ----------
        path:  name of the file that will store the neural network
        """
        fname = os.path.join("NFSP-models", path)
        """
            load_model(models={"p1": p1_current_model, "p2": p2_current_model},
               policies={"p1": p1_policy, "p2": p2_policy}, args=args)
        """
        if not os.path.exists(fname):
            raise ValueError("No model saved with name {}".format(fname))
        checkpoint = torch.load(fname, None)
        self.current_model.load_state_dict(checkpoint['model'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.update_target(self.current_model, self.target_model)

    def load_network_from_other_by_reference(self, other: "NFSPPlayer") -> None:
        """
        Loads network from another NFSP player.
        :param other: player to lead the networks from.
        """
        self.policy = other.policy
        self.current_model = other.current_model
        self.target_model = other.target_model
