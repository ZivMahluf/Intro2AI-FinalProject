from DurakPlayer import DurakPlayer, Deck, Tuple, List, Optional
import random


class RandomPlayer(DurakPlayer):
    """
    Random agent
    """

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        """
        chooses a random card from legal cards to play
        """
        attacking_card = random.choice(legal_cards_to_play)
        if attacking_card != Deck.NO_CARD:
            self._hand.remove(attacking_card)
        return attacking_card

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        """
        chooses a random card from legal cards to play
        """
        defending_card = random.choice(legal_cards_to_play)
        if defending_card != Deck.NO_CARD:
            self._hand.remove(defending_card)
        return defending_card
