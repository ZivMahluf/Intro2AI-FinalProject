from Types import CardType, CardListType, TableType
from DurakPlayer import DurakPlayer
from Deck import Deck

import random


class RandomPlayer(DurakPlayer):
    """
    Random agent.
    Chooses a random action from the given legal actions to play.
    """

    def attack(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        attacking_card = random.choice(legal_cards_to_play)
        if attacking_card != Deck.NO_CARD:
            self._hand.remove(attacking_card)
        return attacking_card

    def defend(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        defending_card = random.choice(legal_cards_to_play)
        if defending_card != Deck.NO_CARD:
            self._hand.remove(defending_card)
        return defending_card
