from Deck import Deck
from typing import List, Tuple, Optional
import numpy as np


class DurakPlayer:

    def __init__(self, hand_size: int, name: str):
        """
        Constructor.
        :param hand_size: Initial hand size.
        :param name: Name of the player.
        """
        self._hand = []
        self._trump_suit = None
        self._other_suits = None
        self.__initial_hand_size = hand_size
        self.__name = name
        self.last_hand = []

    def take_cards(self, cards: List[Deck.CardType]) -> None:
        """
        Adds the cards to the hand.
        :param cards: A list of cards to add.
        """
        self._hand.extend(cards)

    def set_trump_suit(self, suit: int) -> None:
        """
        Sets the trump suit as the given suit.
        :param suit: suit to set as trump suit.
        """
        self._trump_suit = suit
        suits = Deck.SUITS.copy()
        suits.remove(suit)
        self._other_suits = suits

    def get_action(self, state, to_attack):
        """
        Returns an action based on the given state and attack indicator.
        :param state: (attacking cards, defending cards, legal attack cards, legal defence cards, number of cards in deck, number of cards in each player's hand)
        :param to_attack: boolean flag indicating if the action is an attack or a defence.
        :return: chosen action.
        """
        self.last_hand = self.hand.copy()
        if to_attack:
            return self.attack((state[0], state[1], state[4], state[5]), [card for card in state[2] if card in self._hand or card == Deck.NO_CARD])
        return self.defend((state[0], state[1], state[4], state[5]), [card for card in state[3] if card in self._hand or card == Deck.NO_CARD])

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType], int, List[int]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        """
        Attacks with a legal card.
        :param table: A tuple (attacking_cards, defending_cards) of cards on the table.
        :param legal_cards_to_play: A list of legal cards to play, consisting of cards in the player's hand and might also include Deck.NO_CARD.
        :return: The card with which the player attacks (the card is removed from the hand), or Deck.NO_CARD if the player does not attack with any card.
        """
        raise NotImplementedError()

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType], int, List[int]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        """
        Defends with a legal card.
        :param table: A tuple (attacking_cards, defending_cards) of cards on the table.
        :param legal_cards_to_play: A list of legal cards to play, consisting of cards in the player's hand, and Deck.NO_CARD.
        :return: The card with which the player defends (the card is removed from the hand), or Deck.NO_CARD if the player does not defend with any card.
        """
        raise NotImplementedError()

    def is_starting_hand_legal(self) -> bool:
        """
        A legal starting hand is one in which there are at most (initial hand size - 2) cards of each suit.
        :return: Weather the cards in the hand form a legal starting hand.
        """
        num_hearts = 0
        num_diamonds = 0
        num_spades = 0
        num_clubs = 0
        for _, suit in self._hand:
            if suit == Deck.HEARTS:
                num_hearts += 1
            elif suit == Deck.DIAMONDS:
                num_diamonds += 1
            elif suit == Deck.SPADES:
                num_spades += 1
            else:
                num_clubs += 1
        if (self.__initial_hand_size - 1) in [num_hearts, num_diamonds, num_spades, num_clubs]:
            return False
        return True

    def empty_hand(self) -> None:
        """
        Empties the current hand.
        """
        self._hand = list()

    def get_lowest_trump(self) -> int:
        """
        :return: The value of the lowest card with a trump suit, or Deck.NO_CARD is no card has a trump suit in the hand.
        """
        min_trump = np.inf
        for value, suit in self._hand:
            if suit == self._trump_suit:
                if min_trump == Deck.NO_CARD or value < min_trump:
                    min_trump = value
        return min_trump

    def get_weakest_card(self, cards: List[Deck.CardType]) -> Deck.CardType:
        """
        Chooses the weakest card from the given list of cards.
        The weakest card is the card with the lowest value.
        Deck.NO_CARD has a value of (infinity), non-trump cards' value is (card_value), trump cards' value is (Deck.ACE + card_value).
        :param cards: A list of cards to choose from (might include Deck.NO_CARD).
        :return: The weakest card in the list.
        """

        def get_card_value(card: Deck.CardType) -> int:
            if card == Deck.NO_CARD:
                return np.inf
            else:
                value = card[0]
                if card[1] == self._trump_suit:
                    value += Deck.ACE
                return value

        weakest = cards[0]
        min_val = get_card_value(weakest)
        for c in cards[1:]:
            val = get_card_value(c)
            if val < min_val:
                min_val = val
                weakest = c

        return weakest

    def update_end_round(self, defending_player_name: str, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
                         successfully_defended: bool) -> None:
        """
        Updates the agent about the result of the round - weather the defending player defended successfully or not.
        :param defending_player_name: Defending player's name
        :param table: Cards on the table at the end of the round (before clearing)
        :param successfully_defended: Weather the defence was successful (which means all cards are discarded), or not (which means the defending player took all cards on the table).
        """
        pass

    def set_gui(self, gui):
        pass

    def initialize_for_game(self) -> None:
        self._trump_suit = None
        self._other_suits = None
        self.last_hand = []
        self._hand = []

    @property
    def get_others_suit(self) -> List[int]:
        return self._other_suits

    @property
    def hand_size(self) -> int:
        """
        :return: Number of cards in the current hand.
        """
        return len(self._hand)

    @property
    def hand(self) -> List[Deck.CardType]:
        """
        :return: A list of all cards currently in the hand of the player.
        """
        return self._hand

    @property
    def name(self) -> str:
        """
        :return: The name of the player.
        """
        return self.__name

    def __str__(self) -> str:
        """
        :return: String representation of the player (as a string representation of the hand)
        """
        return "(" + self.__name + ", " + str(self._hand) + ")"
