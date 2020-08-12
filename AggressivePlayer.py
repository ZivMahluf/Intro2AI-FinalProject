from Types import CardType, CardListType, TableType
from DurakPlayer import DurakPlayer
from Deck import Deck

import numpy as np


class AggressivePlayer(DurakPlayer):
    """
    The aggressive player attacks with the strongest non trump card possible, or lowest trump card if no non-trump cards are available.
    The aggressive player defends with the weakest card possible.
    The aggressive player prefers to play any card over not playing at all.
    """

    def attack(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        chosen_card = self.get_attack_card(legal_cards_to_play)
        if chosen_card != Deck.NO_CARD:
            self._hand.remove(chosen_card)
        return chosen_card

    def defend(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        chosen_card = self.get_weakest_card(legal_cards_to_play)
        if chosen_card != Deck.NO_CARD:
            self._hand.remove(chosen_card)
        return chosen_card

    def get_attack_card(self, cards: CardListType) -> CardType:
        """
        Chooses the attacking card from the given list of cards.
        The attacking card is the card with the highest value.
        Deck.NO_CARD has a value of (-infinity), non-trump cards' value is (card_value), trump cards' value is (-card_value).
        :param cards: A list of cards to choose from (might include Deck.NO_CARD).
        :return: The card with the highest value in the list.
        """

        def get_card_value(card: CardType) -> int:
            if card == Deck.NO_CARD:
                return -np.inf
            else:
                value = card[0]
                if card[1] == self._trump_suit:
                    value *= -1
                return value

        strongest = cards[0]
        max_val = get_card_value(strongest)
        for c in cards[1:]:
            val = get_card_value(c)
            if max_val < val:
                max_val = val
                strongest = c

        return strongest
