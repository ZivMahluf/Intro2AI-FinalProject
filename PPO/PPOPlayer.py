import LearningPlayer
from DurakPlayer import Deck, Tuple, List, Optional, DurakPlayer
from typing import Union
import random
import tensorflow as tf
from PPO.PPONetwork import PPONetwork, PPOModel
import joblib
import numpy as np


class PPOPlayer(DurakPlayer):

    def __init__(self, hand_size: int, name: str, training_network):
        super().__init__(hand_size, name)
        self.training_network = training_network
        self.memory = np.zeros(shape=(len(Deck.get_full_list_of_cards()) + 1,))

    def learn(self, prev_table: Tuple[List[Deck.CardType], List[Deck.CardType]], prev_action: Deck.CardType,
              reward: float, next_table: Tuple[List[Deck.CardType], List[Deck.CardType]]) -> None:
        raise NotImplementedError()

    def batch_learn(self, batch: List[
        Tuple[Tuple[List[Deck.CardType], List[Deck.CardType]], Deck.CardType, Union[int, float], Tuple[List[Deck.CardType], List[Deck.CardType]]]]):
        pass

    def update_end_round(self, defending_player_name: str, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
                         successfully_defended: bool) -> None:
        if successfully_defended:
            # update memory to include cards
            for card in table:
                self.memory[Deck.get_index_from_card(card)] = 1
        else:
            # TODO: implement later on. keep memory for each player separately
            pass

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
               legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        # attacking_card = random.choice(legal_cards_to_play)
        # if attacking_card != Deck.NO_CARD:
        #     self._hand.remove(attacking_card)
        # return attacking_card

        converted_state = self._convert_state(table)
        converted_available_cards = self._convert_available_cards(legal_cards_to_play)
        action, value, neglogpac = self.training_network.step(converted_state, converted_available_cards)
        action = Deck.get_card_from_index(action[0])
        if action != Deck.NO_CARD:
            self._hand.remove(action)
        else:
            action = Deck.NO_CARD
        return action, value, neglogpac

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
               legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        # defending_card = random.choice(legal_cards_to_play)
        # if defending_card != Deck.NO_CARD:
        #     self._hand.remove(defending_card)
        # return defending_card

        converted_state = self._convert_state(table)
        converted_available_cards = self._convert_available_cards(legal_cards_to_play)
        action, value, neglogpac = self.training_network.step(converted_state, converted_available_cards)
        action = Deck.get_card_from_index(action[0])
        if action != Deck.NO_CARD:
            self._hand.remove(action)
        return action, value, neglogpac

    def _convert_state(self, state):
        deck_length = len(Deck.get_full_list_of_cards()) + 1  # +1 for NO CARD
        full_deck = Deck.get_full_list_of_cards() + [Deck.NO_CARD]
        converted_state = np.zeros(shape=(1, deck_length * 3))
        for i in range(len(state)):
            for card in state[i]:
                card_idx = Deck.get_index_from_card(card)
                converted_state[0][deck_length * i + card_idx] = 1

        return converted_state

    def _convert_available_cards(self, legal_cards_to_play):
        deck_length = len(Deck.get_full_list_of_cards()) + 1  # +1 for NO CARD
        converted_available_cards = np.full(shape=(1, deck_length), fill_value=-np.inf)
        for card in legal_cards_to_play:
            card_idx = Deck.get_index_from_card(card)
            converted_available_cards[0][card_idx] = 0
        return converted_available_cards
