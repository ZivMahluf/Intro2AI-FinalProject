from DurakPlayer import DurakPlayer, Deck, Tuple, List, Optional


class DefensivePlayer(DurakPlayer):
    """
    The defensive player attacks with the *lowest* non trump card possible, or lowest trump card otherwise.
    The defensive player defends with the lowest non trump card possible, or lowest trump card otherwise.
    """

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return self.get_weakest_card(legal_cards_to_play)

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        return self.get_weakest_card(legal_cards_to_play)
