from DurakPlayer import DurakPlayer, Deck, Tuple, List, Optional


class DefensivePlayer(DurakPlayer):
    """
    The defensive player always plays the weakest card available, preferring to play any card over not playing at all.
    """

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType], int, List[int]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        chosen_card = self.get_weakest_card(legal_cards_to_play)
        if chosen_card != Deck.NO_CARD:
            self._hand.remove(chosen_card)
        return chosen_card

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType], int, List[int]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        chosen_card = self.get_weakest_card(legal_cards_to_play)
        if chosen_card != Deck.NO_CARD:
            self._hand.remove(chosen_card)
        return chosen_card
