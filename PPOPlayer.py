from DurakPlayer import Deck, Tuple, List, Optional, DurakPlayer
from PPONetwork import PPONetwork
import joblib
import numpy as np
import os
import itertools


class PPOPlayer(DurakPlayer):

    def __init__(self, hand_size: int, name: str, training_network=None, sess=None):
        super().__init__(hand_size, name)
        self.memory = []
        self.last_converted_state = None
        self.last_converted_available_cards = None

        if training_network:
            self.test_phase = False
            self.training_network = training_network  # this will be done in the training phase
        else:
            self.test_phase = True
            # load the network from memory, and use it for interpretation (not training)
            output_dim = len(Deck.get_full_list_of_cards()) + 1
            input_dim = 185  # hand, attacking cards, defending cards, memory and legal cards to play
            self.training_network = PPONetwork(sess, input_dim, output_dim, "testNet")

            # find latest model parameters file
            files = [(int(f[5:]), f) for f in os.listdir(os.curdir + '/PPOParams') if f.find('model') != -1]
            files.sort(key=lambda x: x[0])
            latest_file = files[-1][1]
            params = joblib.load(os.curdir + '/PPOParams/' + latest_file)
            self.training_network.loadParams(params)

    def update_end_round(self, defending_player_name: str, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
                         successfully_defended: bool) -> None:
        if successfully_defended:
            # update memory to include cards
            for card in itertools.chain(table[0], table[1]):
                self.memory.append(card)
        else:
            # TODO: implement later on. keep memory for each player separately
            pass

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
               legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:

        converted_state = self.convert_input((self.hand, table[0], table[1], self.memory, legal_cards_to_play))
        self.last_converted_state = converted_state
        converted_available_cards = self.convert_available_cards(legal_cards_to_play)
        self.last_converted_available_cards = converted_available_cards
        action, value, neglogpac = self.training_network.step(converted_state, converted_available_cards)
        action = Deck.get_card_from_index(action[0])
        if action != Deck.NO_CARD:
            self._hand.remove(action)
        else:
            action = Deck.NO_CARD

        if self.test_phase:
            return action
        return action, value, neglogpac

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
               legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:

        converted_state = self.convert_input((self.hand, table[0], table[1], self.memory, legal_cards_to_play))
        self.last_converted_state = converted_state
        converted_available_cards = self.convert_available_cards(legal_cards_to_play)
        self.last_converted_available_cards = converted_available_cards
        action, value, neglogpac = self.training_network.step(converted_state, converted_available_cards)
        action = Deck.get_card_from_index(action[0])
        if action != Deck.NO_CARD:
            self._hand.remove(action)

        if self.test_phase:
            return action
        return action, value, neglogpac

    def convert_input(self, input):
        # the input contains: hand, attacking_cards, defending_cards, memory, legal_cards_to_play
        deck_length = len(Deck.get_full_list_of_cards()) + 1  # +1 for NO CARD
        converted_state = np.zeros(shape=(1, deck_length * len(input)))
        for i in range(len(input)):
            for card in input[i]:
                card_idx = Deck.get_index_from_card(card)
                converted_state[0][deck_length * i + card_idx] = 1

        return converted_state

    def convert_available_cards(self, legal_cards_to_play):
        deck_length = len(Deck.get_full_list_of_cards()) + 1  # +1 for NO CARD
        converted_available_cards = np.full(shape=(1, deck_length), fill_value=-np.inf)
        for card in legal_cards_to_play:
            card_idx = Deck.get_index_from_card(card)
            converted_available_cards[0][card_idx] = 0
        return converted_available_cards
