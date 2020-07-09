from Deck import Deck
from typing import List, Tuple, Set, Optional, Dict, NewType
import pygame
# Optional[ObjectType] means it can be None or ObjectType

# todo: maybe define a new type for table instead of using "table: Tuple[List[Deck.CardType], List[Deck.CardType]]"
# TableType = NewType('TableType', Tuple[List[Deck.CardType], List[Deck.CardType]])


class DurakPlayer:

    def __init__(self, hand_size: int, name: str):
        self._hand = []
        self._trump_rank = None
        self.__hand_size = hand_size
        self.__name = name

    def take_cards(self, cards: List[Deck.CardType]) -> None:
        self._hand = self._hand + cards

    def set_trump_rank(self, rank: int) -> None:
        self._trump_rank = rank

    def attack(self, state, legal_cards_to_play) -> None:
        raise NotImplementedError()

    def defend(self, state, legal_cards_to_play) -> None:
        raise NotImplementedError()

    def is_starting_hand_legal(self) -> bool:
        num_hearts = 0
        num_diamonds = 0
        num_spades = 0
        num_clubs = 0
        for _, series in self._hand:
            if series == Deck.HEARTS:
                num_hearts += 1
            elif series == Deck.DIAMONDS:
                num_diamonds += 1
            elif series == Deck.SPADES:
                num_spades += 1
            else:
                num_clubs += 1
        if ((self.__hand_size - 1) in [num_hearts, num_diamonds, num_spades, num_clubs]) or \
                ((num_hearts + num_diamonds) == self.__hand_size) or \
                ((num_spades + num_clubs) == self.__hand_size):
            return False
        return True

    def empty_hand(self) -> None:
        self._hand = []

    def get_lowest_trump(self) -> int:
        min_trump = Deck.NO_CARD
        for value, series in self._hand:
            if series == self._trump_rank:
                if min_trump == Deck.NO_CARD or value < min_trump:
                    min_trump = value
        return min_trump

    @property
    def hand_size(self) -> int:
        return len(self._hand)

    @property
    def hand(self) -> List[Deck.CardType]:
        return self._hand

    @property
    def name(self) -> str:
        return self.__name

    def __str__(self) -> str:
        return str(self._hand)


class BasePlayer(DurakPlayer):

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        attacking_card = self.__get_lowest_card(legal_cards_to_play)
        if attacking_card != Deck.NO_CARD:
            self._hand.remove(attacking_card)
        return attacking_card

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        defending_card = self.__get_lowest_card(legal_cards_to_play)
        if defending_card != Deck.NO_CARD:
            self._hand.remove(defending_card)
        return defending_card

    def __get_lowest_card(self, cards: List[Deck.CardType]) -> Deck.CardType:
        if len(cards) == 0:
            return Deck.NO_CARD
        lowest_card = cards[0]
        for card in cards[1:]:
            value, rank = card
            if (self._trump_rank not in [rank, lowest_card[1]]) or ((rank == self._trump_rank) and (lowest_card[1] == self._trump_rank)):
                if value < lowest_card[0]:
                    lowest_card = card
            elif lowest_card[1] == self._trump_rank:
                lowest_card = card
        return lowest_card


class DurakPlayerWithMemory(DurakPlayer):
    def __init__(self, hand_size: int, name: str, other_players_names: List[str]):
        """
        constructor
        hand_size: number of cards you start with
        name: player's name
        num_players: a list that contains the names of the other players
        """
        DurakPlayer.__init__(self, hand_size, name)
        self.other_players_hand = dict()
        for player in other_players_names:
            self.other_players_hand[player] = set()
        self.discard_pile = set()

    def add_cards_to(self, name: str, cards: Set[Deck.CardType]) -> None:
        """
        Adds [cards] to [name]'s hand.
        :param name: The player that gets the set of cards
        :param cards: A set of cards to remember
        """
        if name not in self.other_players_hand:
            raise Exception("This player doesn't exist")
        # The cards in [name]'s hand are the union of his previous hand with [cards]
        self.other_players_hand[name] = self.other_players_hand[name] | cards

    def remove_cards_from(self, name: str, cards: Set[Deck.CardType]) -> None:
        """
        Removes [cards] from [name]'s hand.
        :param name: The player that gets rid of the set of cards
        :param cards: A set of cards that goes to the discard pile
        """
        if name not in self.other_players_hand:
            raise Exception("This player doesn't exist")
        # The cards in [name]'s hand are the cards that were previously in his hand minus the new ones
        self.other_players_hand[name] = self.other_players_hand[name] - cards
        self.discard_pile = self.discard_pile | cards


class HumanPlayer(DurakPlayer):

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        attacking_card = self.__get_lowest_card(legal_cards_to_play)
        if attacking_card != Deck.NO_CARD:
            self._hand.remove(attacking_card)
        return attacking_card

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Deck.CardType:
        defending_card = self.__get_lowest_card(legal_cards_to_play)
        if defending_card != Deck.NO_CARD:
            self._hand.remove(defending_card)
        return defending_card

    def __get_lowest_card(self, cards: List[Deck.CardType]) -> Deck.CardType:
        if len(cards) == 0:
            return Deck.NO_CARD
        lowest_card = cards[0]
        for card in cards[1:]:
            value, rank = card
            if (self._trump_rank not in [rank, lowest_card[1]]) or ((rank == self._trump_rank) and (lowest_card[1] == self._trump_rank)):
                if value < lowest_card[0]:
                    lowest_card = card
            elif lowest_card[1] == self._trump_rank:
                lowest_card = card
        return lowest_card
