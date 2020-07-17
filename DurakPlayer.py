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
        self._trump_rank = None
        self.__initial_hand_size = hand_size
        self.__name = name

    def take_cards(self, cards: List[Deck.CardType]) -> None:
        """
        Adds the cards to the hand.
        :param cards: A list of cards to add.
        """
        self._hand = self._hand + cards

    def set_trump_rank(self, rank: int) -> None:
        """
        Sets the trump rank as the given rank.
        :param rank: Rank to set as trump rank.
        """
        self._trump_rank = rank

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        raise NotImplementedError()

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        raise NotImplementedError()

    def is_starting_hand_legal(self) -> bool:
        """
        A legal starting hand is one in which there are at most (initial hand size - 2) cards of each rank, and there is at least one 'red'
        card (hearts or diamonds), and one 'black' card (spades or clubs).
        :return: Weather the cards in the hand form a legal starting hand.
        """
        num_hearts = 0
        num_diamonds = 0
        num_spades = 0
        num_clubs = 0
        for _, rank in self._hand:
            if rank == Deck.HEARTS:
                num_hearts += 1
            elif rank == Deck.DIAMONDS:
                num_diamonds += 1
            elif rank == Deck.SPADES:
                num_spades += 1
            else:
                num_clubs += 1
        if ((self.__initial_hand_size - 1) in [num_hearts, num_diamonds, num_spades, num_clubs]) or \
                ((num_hearts + num_diamonds) == self.__initial_hand_size) or \
                ((num_spades + num_clubs) == self.__initial_hand_size):
            return False
        return True

    def empty_hand(self) -> None:
        """
        Empties the current hand.
        """
        self._hand = []

    def get_lowest_trump(self) -> int:
        """
        :return: The value of the lowest card with a trump rank, or Deck.NO_CARD is no card has a trump rank in the hand.
        """
        min_trump = Deck.NO_CARD
        for value, series in self._hand:
            if series == self._trump_rank:
                if min_trump == Deck.NO_CARD or value < min_trump:
                    min_trump = value
        return min_trump

    def get_weakest_card(self, legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        """
        Chooses the weakest card from the given list of cards.
        The weakest card is the card with the lowest value. Any trump card has a higher value than any non-trump card.
        :param legal_cards_to_play: A list of available cards to choose from (might include Deck.NO_CARD).
        :return: The card with the lowest value from the legal cards.
        """

        def __sort_legal_cards(x):
            """
            Give all cards priority over Deck.NO_CARD
            :param x: a card
            :return: the number of the card, or inf if Deck.NO_CARD
            """
            return x[0] if x != Deck.NO_CARD else np.inf

        if len(legal_cards_to_play) == 0:
            return Deck.NO_CARD

        # sort cards by number
        legal_cards_to_play.sort(key=__sort_legal_cards)

        # if the lowest card is Deck.NO_CARD, it means that all legal moves are Deck.NO_CARD, since all other cards should appear before
        if legal_cards_to_play[0] == Deck.NO_CARD:
            return Deck.NO_CARD

        # if all cards are trumps or no cards, pick first
        # otherwise, pick first non-trump
        lowest_card = Deck.NO_CARD
        for card in legal_cards_to_play:
            if card != Deck.NO_CARD:
                value, rank = card
                if rank != self._trump_rank:
                    lowest_card = card
                    break

        if lowest_card == Deck.NO_CARD:
            # no non-trump card found, so return first trump card
            self._hand.remove(legal_cards_to_play[0])
            return legal_cards_to_play[0]
        else:
            self._hand.remove(lowest_card)
            return lowest_card

    def get_strongest_card(self, legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        """
        Chooses the strongest non-trump card from the given list of cards.
        The strongest card is the card with the highest value. Any trump card has a higher value than any non-trump card.
        :param legal_cards_to_play: A list of available cards to choose from (might include Deck.NO_CARD).
        :return: The card with the highest value from the legal cards.
        """

        def __sort_legal_cards(x):
            """
            Give all cards priority over Deck.NO_CARD
            :param x: a card
            :return: the number of the card, or 0 if Deck.NO_CARD
            """
            if x == Deck.NO_CARD:
                return 0
            return x[0]

        if len(legal_cards_to_play) == 0:
            return Deck.NO_CARD

        # sort cards by number (highest number first)
        legal_cards_to_play.sort(key=__sort_legal_cards, reverse=True)

        # if the lowest card is Deck.NO_CARD, it means that all legal moves are Deck.NO_CARD, since all other cards should appear before
        if legal_cards_to_play[0] == Deck.NO_CARD:
            return Deck.NO_CARD

        # if all cards are trumps or no cards, pick first
        # otherwise, pick first non-trump
        highest_card = Deck.NO_CARD
        for card in legal_cards_to_play:
            if card != Deck.NO_CARD:
                value, rank = card
                if rank != self._trump_rank:
                    highest_card = card
                    break

        if highest_card == Deck.NO_CARD:
            # no non-trump card found, so return first trump card
            self._hand.remove(legal_cards_to_play[0])
            return legal_cards_to_play[0]
        else:
            self._hand.remove(highest_card)
            return highest_card

    def update_round_progress(self, player_name: str, played_card: Deck.CardType) -> None:
        """
        Updates the agent about a card that was played by a player.
        :param player_name: Name of the player that played.
        :param played_card: The card played by that player.
        """
        pass

    def update_end_round(self, defending_player_name: str, table: Tuple[List[Deck.CardType], List[Deck.CardType]],
                         successfully_defended: bool) -> None:
        """
        Updates the agent about the result of the round - weather the defending player defended successfully or not.
        :param defending_player_name: Defending player's name
        :param table: Cards on the table at the end of the round (before clearing)
        :param successfully_defended: Weather the defence was successful (which means all cards are discarded), or not (which means the defending player took all cards on the table).
        """
        pass

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
        return str(self._hand)
