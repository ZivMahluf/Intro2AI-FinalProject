from itertools import product
import numpy as np
from typing import NewType, Tuple, List


class Deck:

    """
    Parameters regarding cards.
    """
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    HEARTS = 0
    CLUBS = 1
    DIAMONDS = 2
    SPADES = 3
    STRING_RANKS = {HEARTS: "H", CLUBS: "C", DIAMONDS: "D", SPADES: "S"}
    RANKS = [HEARTS, CLUBS, DIAMONDS, SPADES]
    VALUES = [6, 7, 8, 9, 10, JACK, QUEEN, KING, ACE]
    NO_CARD = (-1, -1)
    """
    Custom card type.
    """
    CardType = Tuple[int, int]

    def __init__(self):
        """
        Constructor.
        """
        self.__deck = self.get_full_list_of_cards()
        self.__total_size = len(self.__deck)

    def draw(self, cards: int = 1) -> List[CardType]:
        """
        Draws the specified number of cards.
        :param cards: Number of cards to draw.
        :return: A list of all drawn cards (up to the number of remaining cards in the deck).
        """
        size = len(self.__deck)
        dealt_cards = [self.__deck.pop() for _ in range(min(cards, size))]
        return dealt_cards

    def to_bottom(self, card: CardType) -> None:
        """
        Puts the given card to the bottom of the deck.
        :param card: Card to put to the bottom.
        """
        self.__deck.insert(0, card)

    def shuffle(self):
        np.random.shuffle(self.__deck)

    @property
    def current_num_cards(self) -> int:
        """
        :return: Number of cards remaining in the deck.
        """
        return len(self.__deck)

    @property
    def total_num_cards(self) -> int:
        """
        :return: Size of a full deck.
        """
        return self.__total_size

    @property
    def cards(self) -> List[Tuple[int, int]]:
        """
        :return: A list of all cards in the deck.
        """
        return self.__deck

    @staticmethod
    def get_full_list_of_cards():
        """
        :return: A list of all cards in a deck.
        """
        return list(product(Deck.VALUES, Deck.RANKS))

    def __str__(self) -> str:
        """
        :return: String representation of the deck.
        """
        return str(self.__deck)
