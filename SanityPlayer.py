from typing import Tuple, List
from Deck import Deck
from LearningPlayer import LearningPlayer


class SanityCheckPlayer(LearningPlayer):

    def first_initialize(self) -> None:
        pass

    def initialize_for_game(self) -> None:
        pass

    def learn(self, prev_table: Tuple[List[Deck.CardType], List[Deck.CardType]], prev_action: Deck.CardType, reward: float, next_table: Tuple[List[Deck.CardType], List[Deck.CardType]]) -> None:
        pass

    def batch_learn(self, prev_states: List[Tuple[List[Deck.CardType], List[Deck.CardType]]], prev_actions: List[Deck.CardType], rewards: List[float],
                    next_states: List[Tuple[List[Deck.CardType], List[Deck.CardType]]]):
        pass
