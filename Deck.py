from itertools import product
import numpy as np
from typing import NewType, Tuple


class Deck:

    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    HEARTS = "H"
    CLUBS = "C"
    DIAMONDS = "D"
    SPADES = "S"
    NO_CARD = 0
    CardType = NewType('CardType', Tuple[int, str])

    def __init__(self):
        self.__deck = list(product([6, 7, 8, 9, 10, self.JACK, self.QUEEN, self.KING, self.ACE],
                                   [self.HEARTS, self.CLUBS, self.DIAMONDS, self.SPADES]))
        self.__current_size = len(self.__deck)
        self.__total_size = len(self.__deck)
        np.random.shuffle(self.__deck)

    def draw(self, cards=1):
        dealt_cards = [self.__deck.pop() for _ in range(min(cards, self.__current_size))]
        self.__current_size = len(self.__deck)
        return dealt_cards

    def put_back(self, card):
        if card not in self.__deck:
            self.__deck.append(card)
            np.random.shuffle(self.__deck)

    @property
    def current_num_cards(self):
        return self.__current_size

    @property
    def total_num_cards(self):
        return self.__total_size

    @property
    def cards(self):
        return self.__deck

    def __str__(self):
        return str(self.__deck)

