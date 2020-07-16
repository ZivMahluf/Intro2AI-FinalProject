from itertools import product
import numpy as np
from typing import NewType, Tuple, List, Dict


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
    NO_CARD = 0
    """
    Custom card type.
    """
    CardType = NewType('CardType', Tuple[int, int])

    def __init__(self):
        """
        Constructor.
        """
        self.__deck = list(product(self.VALUES, self.RANKS))
        self.__indices = {self.__deck[i]: i for i in range(len(self.__deck))}
        self.__indices[self.NO_CARD] = len(self.__deck)
        self.__current_size = len(self.__deck)
        self.__total_size = len(self.__deck)
        np.random.shuffle(self.__deck)

    def draw(self, cards: int = 1) -> List[Tuple[int, int]]:
        """
        Draws the specified number of cards.
        :param cards: Number of cards to draw.
        :return: A list of all drawn cards (up to the number of remaining cards in the deck).
        """
        dealt_cards = [self.__deck.pop() for _ in range(min(cards, self.__current_size))]
        self.__current_size = len(self.__deck)
        return dealt_cards

    def put_back(self, card: CardType) -> None:
        """
        Puts a card back into the deck and shuffles.
        :param card: card to put back.
        """
        if card not in self.__deck:
            self.__deck.append(card)
            np.random.shuffle(self.__deck)

    @property
    def current_num_cards(self) -> int:
        """
        :return: Number of cards remaining in the deck.
        """
        return self.__current_size

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

    @property
    def card_indices(self) -> Dict[int, CardType]:
        """
        :return: A mapping of cards to indices.
        """
        return self.__indices

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
