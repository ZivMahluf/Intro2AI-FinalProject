from NFSPModel import DQN, Policy
import os
import torch
from NFSPPlayer import NFSPPlayer
from typing import Tuple, List, Optional
from Deck import Deck
import copy
import random

class TrainedNFSPPlayer(NFSPPlayer):

    def __init__(self, hand_size, name):
        super().__init__(hand_size, name)
        self.policy = Policy()
        self.current_model = DQN(False)

    def load_from_other_player(self, other: NFSPPlayer):
        self.policy = copy.deepcopy(other.policy)
        self.current_model = copy.deepcopy(other.current_model)

    def learn_step(self, old_state, new_state, action, reward, info):
        pass

    def act(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]):
        legal_cards_vec, state = NFSPPlayer.get_network_input(legal_cards_to_play, table, self.discard_pile, self._hand)
        card = NFSPPlayer.action_to_card(self.policy.act(torch.FloatTensor(state), legal_cards_vec))
        if card[0] != -1:
            self._hand.remove(card)
        return card

        # legal_cards_vec, state = self.get_network_input(legal_cards_to_play, table, self.discard_pile, self._hand)
        # if random.random() > self.eta:
        #     action = self.policy.act(torch.FloatTensor(state).to(self.device), legal_cards_vec)
        # else:
        #     action = self.current_model.act(torch.FloatTensor(state).to(self.device), self.epsilon_by_round(), legal_cards_vec)
        # card = NFSPPlayer.action_to_card(action)
        # if card[0] != -1:
        #     self._hand.remove(card)
        # return card

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
               legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return self.act(table, legal_cards_to_play)

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
               legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return self.act(table, legal_cards_to_play)

    def initialize_for_game(self):
        super().initialize_for_game()
