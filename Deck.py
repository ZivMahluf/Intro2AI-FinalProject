from Types import CardType, CardListType
from itertools import product
import numpy as np


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
    STRING_SUITS = {HEARTS: "H", CLUBS: "C", DIAMONDS: "D", SPADES: "S"}
    SUITS = [HEARTS, CLUBS, DIAMONDS, SPADES]
    VALUES = [6, 7, 8, 9, 10, JACK, QUEEN, KING, ACE]
    NO_CARD = (-1, -1)

    def __init__(self):
        """
        Constructor.
        """
        self.__deck = self.get_full_list_of_cards()
        self.__total_size = len(self.__deck)

    def draw(self, cards: int = 1) -> CardListType:
        """
        Draws the specified number of cards.
        :param cards: Number of cards to draw.
        :return: A list of all drawn cards (up to the number of remaining cards in the deck).
        """
        size = len(self.__deck)
        dealt_cards = [self.__deck.pop() for _ in range(min(cards, size))]
        return dealt_cards

    def shuffle(self) -> None:
        """
        Shuffles the deck.
        """
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
    def cards(self) -> CardListType:
        """
        :return: A list of all cards in the deck.
        """
        return self.__deck

    @staticmethod
    def get_full_list_of_cards() -> CardListType:
        """
        :return: A list of all cards in a full deck.
        """
        return list(product(Deck.VALUES, Deck.SUITS))

    @staticmethod
    def get_index_from_card(card: CardType) -> int:
        """
        :param card: Legal card that appears in the deck.
        :return: Index of the card in a full non-shuffled deck.
        """
        if card == Deck.NO_CARD:
            return Deck().total_num_cards
        return Deck.get_full_list_of_cards().index(card)

    @staticmethod
    def get_card_from_index(index: int) -> CardType:
        """
        :param index: index in range [0, len(full_deck_of_cards)].
        :return: Card at the given index in a full non-shuffled deck.
        """
        if index == Deck().total_num_cards:
            return Deck.NO_CARD
        return Deck.get_full_list_of_cards()[index]

    def __str__(self) -> str:
        """
        :return: String representation of the deck.
        """
        return str(self.__deck)
