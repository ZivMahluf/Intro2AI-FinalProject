from NFSPModel import DQN, Policy
import os
import torch
from NFSPPlayer import NFSPPlayer
from typing import Tuple, List, Optional
from Deck import Deck
import copy

class TrainedNFSPPlayer(NFSPPlayer):

    def __init__(self, hand_size, name):
        super().__init__(hand_size, name)
        self.policy = Policy()
        self.current_model = DQN(False)

    def load_from_other_player(self, other: NFSPPlayer):
        self.policy = copy.deepcopy(other.policy)

    def learn_step(self, old_state, new_state, action, reward):
        pass

    def act(self, table: Tuple[List[Deck.CardType], List[Deck.CardType], int, List[int]], legal_cards_to_play: List[Deck.CardType]):
        legal_cards_vec, state = NFSPPlayer.get_network_input(legal_cards_to_play, table, self.discard_pile, self._hand)
        card = NFSPPlayer.action_to_card(self.policy.act(torch.FloatTensor(state), legal_cards_vec))
        if card[0] != -1:
            self._hand.remove(card)
        return card

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType], int, List[int]],
               legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return self.act(table, legal_cards_to_play)

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType], int, List[int]],
               legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return self.act(table, legal_cards_to_play)

    def initialize_for_game(self):
        super().initialize_for_game()
