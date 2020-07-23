from DurakPlayer import DurakPlayer, Deck, Tuple, List, Optional
import random


class LearningPlayer(DurakPlayer):
    def __init__(self, hand_size: int, name: str):
        DurakPlayer.__init__(self, hand_size, name)
        self.initialized = False

    def first_initialize(self) -> None:
        self.initialized = True

    def initialize_for_game(self) -> None:
        raise NotImplementedError()

    def learn(self, prev_table: Tuple[List[Deck.CardType], List[Deck.CardType]], prev_action: Deck.CardType, reward: float, next_table: Tuple[List[Deck.CardType], List[Deck.CardType]]) -> None:
        raise NotImplementedError()

    def attack(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        attacking_card = random.choice(legal_cards_to_play)
        if attacking_card != Deck.NO_CARD:
            self._hand.remove(attacking_card)
        return attacking_card

    def defend(self, table: Tuple[List[Deck.CardType], List[Deck.CardType]], legal_cards_to_play: List[Deck.CardType]) -> Optional[Deck.CardType]:
        defending_card = random.choice(legal_cards_to_play)
        if defending_card != Deck.NO_CARD:
            self._hand.remove(defending_card)
        return defending_card
