from Types import CardType, CardListType, TableType
from DurakPlayer import DurakPlayer
from Deck import Deck


class DefensivePlayer(DurakPlayer):
    """
    The defensive player always plays the weakest card available, preferring to play any card over not playing at all.
    """

    def attack(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        chosen_card = self.get_weakest_card(legal_cards_to_play)
        if chosen_card != Deck.NO_CARD:
            self._hand.remove(chosen_card)
        return chosen_card

    def defend(self, table: TableType, legal_cards_to_play: CardListType) -> CardType:
        chosen_card = self.get_weakest_card(legal_cards_to_play)
        if chosen_card != Deck.NO_CARD:
            self._hand.remove(chosen_card)
        return chosen_card
