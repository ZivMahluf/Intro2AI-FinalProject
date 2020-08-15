from Types import CardListType, CardType, TableType, Tuple
from DurakPlayer import DurakPlayer
from PPONetwork import PPONetwork
from Deck import Deck
import joblib
import numpy as np
import os
import itertools


class PPOPlayer(DurakPlayer):

    """
    Parameters regarding the input and output of the network the player uses.
    """
    output_dim = len(Deck.get_full_list_of_cards()) + 1
    input_dim = 185  # hand, attacking cards, defending cards, memory and legal cards to play

    def __init__(self, hand_size: int, name: str, training_network=None, sess=None):
        """
        Constructor. Sets up the network with which the player will play.
        :param hand_size: initial hand size.
        :param name: name of the player.
        :param training_network: neural network to train with.
        :param sess: tensorflow session to train in.
        """
        super().__init__(hand_size, name)
        self.memory = []
        self.last_converted_state = None
        self.last_converted_available_cards = None

        self.value = 0
        self.neglogpac = 0

        if training_network:
            self.training_network = training_network  # this will be done in the training phase
        else:
            # load the network from memory, and use it for interpretation (not training)
            self.training_network = PPONetwork(sess, self.input_dim, self.output_dim, "testNet")

            # find latest model parameters file
            files = [(int(f[5:]), f) for f in os.listdir(os.curdir + '/PPOParams') if f.find('model') != -1]
            files.sort(key=lambda x: x[0])
            latest_file = files[-1][1]
            params = joblib.load(os.curdir + '/PPOParams/' + latest_file)
            self.training_network.loadParams(params)
            print("loaded " + latest_file)

    def update_end_round(self, defending_player_name: str, table: TableType, successfully_defended: bool) -> None:
        """
        Updates the memory of the player at the end of the round.
        :param defending_player_name: Name of the defending player
        :param table: Cards on the table (attack, defence), number of cards in the deck, number of cards in each player's hand.
        :param successfully_defended: weather the defender defended successfully.
        """
        if successfully_defended:
            # update memory to include cards
            for card in itertools.chain(table[0], table[1]):
                self.memory.append(card)

    def attack(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        """
        Chooses a card to attack with using the player's network.
        :param table: (attacking cards, defending cards, number of cards in deck, number of cards in each player's hand)
        :param legal_cards_to_play: legal cards from the player's hand to play (might include Deck.NO_CARD)
        :return: The chosen card to attack with (or Deck.NO_CARD)
        """
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
        """
        Chooses a card to defend with using the player's network.
        :param table: (attacking cards, defending cards, number of cards in deck, number of cards in each player's hand)
        :param legal_cards_to_play: legal cards from the player's hand to play (might include Deck.NO_CARD)
        :return: The chosen card to defend with (or Deck.NO_CARD)
        """
        converted_state = self.convert_input((self.hand, table[0], table[1], self.memory, legal_cards_to_play))
        self.last_converted_state = converted_state
        converted_available_cards = self.convert_available_cards(legal_cards_to_play)
        self.last_converted_available_cards = converted_available_cards
        action, self.value, self.neglogpac = self.training_network.step(converted_state, converted_available_cards)
        action = Deck.get_card_from_index(action[0])
        if action != Deck.NO_CARD:
            self._hand.remove(action)
        return action

    def get_val_neglogpac(self):
        """
        :return: Values used for training the network.
        """
        return self.value, self.neglogpac

    @staticmethod
    def convert_input(input_tuple: Tuple[CardListType, CardListType, CardListType, CardListType, CardListType]) -> np.ndarray:
        """
        Converts the given tuple of lists into a vector of features.
        :param input_tuple: (cards in hand, attacking_cards, defending_cards, memory, legal_cards_to_play)
        :return: A feature vector representing the cards in the lists.
        """
        deck_length = len(Deck.get_full_list_of_cards()) + 1  # +1 for NO CARD
        converted_state = np.zeros(shape=(1, deck_length * len(input_tuple)))
        for i in range(len(input_tuple)):
            for card in input_tuple[i]:
                card_idx = Deck.get_index_from_card(card)
                converted_state[0][deck_length * i + card_idx] = 1

        return converted_state

    @staticmethod
    def convert_available_cards(legal_cards_to_play: CardListType) -> np.ndarray:
        """
        Converts the given list of legal cards into a feature vector.
        :param legal_cards_to_play: list of legal cards  to play.
        :return: A representing feature vector.
        """
        deck_length = len(Deck.get_full_list_of_cards()) + 1  # +1 for NO CARD
        converted_available_cards = np.full(shape=(1, deck_length), fill_value=-np.inf)
        for card in legal_cards_to_play:
            card_idx = Deck.get_index_from_card(card)
            converted_available_cards[0][card_idx] = 0
        return converted_available_cards

    def initialize_for_game(self) -> None:
        """
        Initializes the player for the game.
        """
        super().initialize_for_game()
        self.memory = []

    def load_params(self, file_name) -> None:
        """
        Loads parameters from the file ./PPOParams/file_name.
        :param file_name: name of file holding the parameters.
        """
        params = joblib.load(os.path.join(os.path.join(os.curdir, 'PPOParams'), file_name))
        self.training_network.loadParams(params)
        print("loaded " + file_name)


def from_path(sess, hand_size, name, path) -> PPOPlayer:
    """
    Loads a PPOPlayer using the parameters from the file whose path is given as an input.
    If the loading fails, a player with an untrained network will be returned.
    :param sess: tensorflow session
    :param hand_size: size of starting hand for the player
    :param name: name of the player
    :param path: path of the parameters file
    :return: a PPOPlayer with parameters from the given file. If loading fails, a non-trained PPOPlayer
    """
    network = PPONetwork(sess, PPOPlayer.input_dim, PPOPlayer.output_dim, name + "_Network")
    try:
        parameters = joblib.load(path)
        network.loadParams(parameters)
        player = PPOPlayer(hand_size, name, network, sess)
        print("loaded " + name)
        return player
    except:
        player = PPOPlayer(hand_size, name, network, sess)
        return player
