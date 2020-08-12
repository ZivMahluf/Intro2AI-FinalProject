from Types import Tuple, CardType, CardListType, FieldType, TableType
from DurakPlayer import DurakPlayer
from PPONetwork import PPONetwork
from Deck import Deck

import tensorflow as tf
import numpy as np
import itertools
import joblib
import os


class PPOPlayer(DurakPlayer):

    output_dim = len(Deck.get_full_list_of_cards()) + 1
    input_dim = 185  # hand, attacking cards, defending cards, memory and legal cards to play

    def __init__(self, hand_size: int, name: str, training_network: PPONetwork = None, sess: tf.compat.v1.Session = None):
        super().__init__(hand_size, name)
        self.memory = []
        self.last_converted_state = None
        self.last_converted_available_cards = None

        self.value = 0
        self.neglogpac = 0

        if training_network:
            self.test_phase = False
            self.training_network = training_network  # this will be done in the training phase
        else:
            self.test_phase = True
            # load the network from memory, and use it for interpretation (not training)
            self.training_network = PPONetwork(sess, self.input_dim, self.output_dim, "testNet")

            # find latest model parameters file
            files = [(int(f[5:]), f) for f in os.listdir(os.curdir + '/PPOParams') if f.find('model') != -1]
            files.sort(key=lambda x: x[0])
            latest_file = files[-1][1]
            params = joblib.load(os.curdir + '/PPOParams/' + latest_file)
            self.training_network.loadParams(params)
            print("loaded " + latest_file)

    def update_end_round(self, defending_player_name: str, table: FieldType, successfully_defended: bool) -> None:
        if successfully_defended:
            # update memory to include cards
            for card in itertools.chain(table[0], table[1]):
                self.memory.append(card)
        else:
            # You can save the memory for each player separately here. We chose not to.
            pass

    def attack(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        converted_state = self.convert_input((self.hand, table[0], table[1], self.memory, legal_cards_to_play))
        self.last_converted_state = converted_state
        converted_available_cards = self.convert_available_cards(legal_cards_to_play)
        self.last_converted_available_cards = converted_available_cards
        action, self.value, self.neglogpac = self.training_network.step(converted_state, converted_available_cards)
        action = Deck.get_card_from_index(action[0])
        if action != Deck.NO_CARD:
            self._hand.remove(action)
        return action

    def defend(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        converted_state = self.convert_input((self.hand, table[0], table[1], self.memory, legal_cards_to_play))
        self.last_converted_state = converted_state
        converted_available_cards = self.convert_available_cards(legal_cards_to_play)
        self.last_converted_available_cards = converted_available_cards
        action, self.value, self.neglogpac = self.training_network.step(converted_state, converted_available_cards)
        action = Deck.get_card_from_index(action[0])
        if action != Deck.NO_CARD:
            self._hand.remove(action)
        return action

    def get_val_neglogpac(self) -> Tuple[int, int]:
        return self.value, self.neglogpac

    @staticmethod
    def convert_input(network_input: Tuple[CardListType, CardListType, CardListType, CardListType, CardListType]) -> np.ndarray:
        # the input contains: hand, attacking_cards, defending_cards, memory, legal_cards_to_play
        deck_length = len(Deck.get_full_list_of_cards()) + 1  # +1 for NO CARD
        converted_state = np.zeros(shape=(1, deck_length * len(network_input)))
        for i in range(len(network_input)):
            for card in network_input[i]:
                card_idx = Deck.get_index_from_card(card)
                converted_state[0][deck_length * i + card_idx] = 1
        return converted_state

    @staticmethod
    def convert_available_cards(legal_cards_to_play: CardListType) -> np.ndarray:
        deck_length = len(Deck.get_full_list_of_cards()) + 1  # +1 for NO CARD
        converted_available_cards = np.full(shape=(1, deck_length), fill_value=-np.inf)
        for card in legal_cards_to_play:
            card_idx = Deck.get_index_from_card(card)
            converted_available_cards[0][card_idx] = 0
        return converted_available_cards

    def initialize_for_game(self) -> None:
        super().initialize_for_game()
        self.memory = []
